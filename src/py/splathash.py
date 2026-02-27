# SplatHash — Python implementation
#
# Copyright (c) 2025 dragonrider
# MIT License — see ../LICENSE
#
# Algorithmically identical to the Go reference implementation.
# Requires: Pillow (pip install Pillow)
#
# API:
#   encode(source) -> bytes          source: file path, file object, or PIL.Image
#   encode_raw(rgba, width, height) -> bytes
#   decode(hash_bytes) -> bytes      returns 32*32*4 RGBA bytes

from __future__ import annotations

import math
import struct
from typing import NamedTuple, Union

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore

__all__ = ["encode", "encode_raw", "decode"]

# --- Constants ---

TARGET_SIZE  = 32
RIDGE_LAMBDA = 0.001
SIGMA_TABLE  = [0.025, 0.1, 0.2, 0.35]


class _Splat(NamedTuple):
    x:        float
    y:        float
    sigma:    float
    l:        float
    a:        float
    b:        float
    is_lepton: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(source) -> bytes:
    """Encode an image to a 16-byte SplatHash.

    source may be:
      - a file path (str or Path)
      - a file-like object
      - a PIL.Image.Image instance
    """
    if _PILImage is None:
        raise ImportError("Pillow is required: pip install Pillow")
    if isinstance(source, _PILImage.Image):
        img = source
    else:
        img = _PILImage.open(source)
    img = img.convert("RGBA")
    width, height = img.size
    rgba = img.tobytes()
    return encode_raw(rgba, width, height)


def encode_raw(rgba: bytes, width: int, height: int) -> bytes:
    """Encode an image from raw RGBA bytes to a 16-byte SplatHash."""
    # 1. Downsample and convert to Oklab
    grid = _image_to_oklab_grid(rgba, width, height, TARGET_SIZE, TARGET_SIZE)

    # 2. Compute mean
    n = TARGET_SIZE * TARGET_SIZE
    mean_l = sum(grid[i * 3]     for i in range(n)) / n
    mean_a = sum(grid[i * 3 + 1] for i in range(n)) / n
    mean_b = sum(grid[i * 3 + 2] for i in range(n)) / n

    # Quantize mean immediately so the solver works against the stored value
    packed_mean = _pack_mean(mean_l, mean_a, mean_b)
    mean_l, mean_a, mean_b = _unpack_mean(packed_mean)

    # Residuals
    target_l = [grid[i * 3]     - mean_l for i in range(n)]
    target_a = [grid[i * 3 + 1] - mean_a for i in range(n)]
    target_b = [grid[i * 3 + 2] - mean_b for i in range(n)]

    # 3. Greedy basis search
    basis: list[_Splat] = []
    current_recon = [0.0] * (TARGET_SIZE * TARGET_SIZE * 3)

    for i in range(6):
        candidate, gain = _find_best_splat(
            grid, current_recon, mean_l, mean_a, mean_b,
            TARGET_SIZE, TARGET_SIZE
        )
        if gain < 0.00001:
            break
        is_lepton = (i >= 3)
        splat = _Splat(
            x=candidate.x, y=candidate.y, sigma=candidate.sigma,
            l=candidate.l, a=candidate.a, b=candidate.b,
            is_lepton=is_lepton
        )
        basis.append(splat)
        _add_splat_to_grid(current_recon, splat, TARGET_SIZE, TARGET_SIZE)

    # 4. Global Ridge Regression
    if basis:
        basis = _solve_v4_weights(basis, target_l, target_a, target_b,
                                  TARGET_SIZE, TARGET_SIZE)

    # 5. Pack
    return _pack_v4(packed_mean, basis)


def decode(hash_bytes: bytes) -> bytes:
    """Decode a 16-byte SplatHash into a 32x32 RGBA image (bytes, length 4096)."""
    if len(hash_bytes) != 16:
        raise ValueError(f"Invalid SplatHash: expected 16 bytes, got {len(hash_bytes)}")

    mean_l, mean_a, mean_b, splats = _unpack_v4(hash_bytes)
    w, h = 32, 32
    grid = [0.0] * (w * h * 3)

    # Fill background
    for i in range(w * h):
        grid[i * 3]     = mean_l
        grid[i * 3 + 1] = mean_a
        grid[i * 3 + 2] = mean_b

    # Composite splats
    for s in splats:
        _add_splat_to_grid(grid, s, w, h)

    # Convert to RGBA bytes
    out = bytearray(w * h * 4)
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 3
            r, g, b = _oklab_to_srgb(grid[idx], grid[idx + 1], grid[idx + 2])
            p = (y * w + x) * 4
            out[p]     = _clampi(int(r * 255 + 0.5), 0, 255)
            out[p + 1] = _clampi(int(g * 255 + 0.5), 0, 255)
            out[p + 2] = _clampi(int(b * 255 + 0.5), 0, 255)
            out[p + 3] = 255
    return bytes(out)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _solve_v4_weights(basis, target_l, target_a, target_b, w, h):
    n_total   = len(basis)
    n_baryons = sum(1 for s in basis if not s.is_lepton)

    # Precompute Gaussian activation maps
    activations = [_compute_basis_map(s, w, h) for s in basis]

    # Solve each channel
    x_l = _solve_channel(activations,            target_l, n_total,   RIDGE_LAMBDA)
    x_a = _solve_channel(activations[:n_baryons], target_a, n_baryons, RIDGE_LAMBDA)
    x_b = _solve_channel(activations[:n_baryons], target_b, n_baryons, RIDGE_LAMBDA)

    out = []
    for i, s in enumerate(basis):
        out.append(_Splat(
            x=s.x, y=s.y, sigma=s.sigma,
            l=x_l[i],
            a=(x_a[i] if i < n_baryons else 0.0),
            b=(x_b[i] if i < n_baryons else 0.0),
            is_lepton=s.is_lepton
        ))
    return out


def _solve_channel(activations, target, n, lam):
    if n == 0:
        return []
    m = len(target)

    # Build ATA (n x n) and ATb (n)
    ata = [0.0] * (n * n)
    atb = [0.0] * n

    for i in range(n):
        for j in range(i, n):
            s = 0.0
            ai = activations[i]
            aj = activations[j]
            for p in range(m):
                s += ai[p] * aj[p]
            ata[i * n + j] = s
            ata[j * n + i] = s
        s = 0.0
        ai = activations[i]
        for p in range(m):
            s += ai[p] * target[p]
        atb[i] = s

    # Ridge regularization
    for i in range(n):
        ata[i * n + i] += lam

    return _solve_linear_system(ata, atb, n)


def _solve_linear_system(mat, vec, n):
    a = list(mat)
    b = list(vec)
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = a[i * n + k] / a[k * n + k]
            for j in range(k, n):
                a[i * n + j] -= factor * a[k * n + j]
            b[i] -= factor * b[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += a[i * n + j] * x[j]
        x[i] = (b[i] - s) / a[i * n + i]
    return x


# ---------------------------------------------------------------------------
# Greedy search
# ---------------------------------------------------------------------------

def _find_best_splat(grid, recon, m_l, m_a, m_b, w, h):
    n = w * h
    res_l = [grid[i * 3]     - m_l - recon[i * 3]     for i in range(n)]
    res_a = [grid[i * 3 + 1] - m_a - recon[i * 3 + 1] for i in range(n)]
    res_b = [grid[i * 3 + 2] - m_b - recon[i * 3 + 2] for i in range(n)]

    best_splat = _Splat(0.0, 0.0, 0.1, 0.0, 0.0, 0.0, False)
    max_score  = -1.0

    step = 2
    for y in range(0, h, step):
        for x in range(0, w, step):
            xf = x / w
            yf = y / h
            for sigma in SIGMA_TABLE:
                rad  = int(sigma * w * 3.5)
                y0   = _clampi(y - rad, 0, h - 1)
                y1   = _clampi(y + rad, 0, h - 1)
                x0   = _clampi(x - rad, 0, w - 1)
                x1   = _clampi(x + rad, 0, w - 1)
                dot_l = dot_a = dot_b = dot_g = 0.0

                for sy in range(y0, y1 + 1):
                    dy      = sy / h - yf
                    row_base = sy * w
                    for sx in range(x0, x1 + 1):
                        dx      = sx / w - xf
                        dist_sq = dx * dx + dy * dy
                        weight  = math.exp(-dist_sq / (2 * sigma * sigma))
                        idx     = row_base + sx
                        dot_l  += weight * res_l[idx]
                        dot_a  += weight * res_a[idx]
                        dot_b  += weight * res_b[idx]
                        dot_g  += weight * weight

                if dot_g < 1e-9:
                    continue
                score = (dot_l * dot_l + dot_a * dot_a + dot_b * dot_b) / dot_g
                if score > max_score:
                    max_score  = score
                    best_splat = _Splat(
                        x=xf, y=yf, sigma=sigma,
                        l=dot_l / dot_g, a=dot_a / dot_g, b=dot_b / dot_g,
                        is_lepton=False
                    )

    return best_splat, max_score


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _compute_basis_map(s: _Splat, w: int, h: int) -> list[float]:
    out    = [0.0] * (w * h)
    rad    = int(s.sigma * w * 3.5)
    cx     = int(s.x * w)
    cy     = int(s.y * h)
    y0     = _clampi(cy - rad, 0, h - 1)
    y1     = _clampi(cy + rad, 0, h - 1)
    x0     = _clampi(cx - rad, 0, w - 1)
    x1     = _clampi(cx + rad, 0, w - 1)
    two_s2 = 2 * s.sigma * s.sigma
    for y in range(y0, y1 + 1):
        dy = y / h - s.y
        for x in range(x0, x1 + 1):
            dx           = x / w - s.x
            out[y * w + x] = math.exp(-(dx * dx + dy * dy) / two_s2)
    return out


def _add_splat_to_grid(grid: list[float], s: _Splat, w: int, h: int) -> None:
    rad    = int(s.sigma * w * 3.5)
    cx     = int(s.x * w)
    cy     = int(s.y * h)
    y0     = _clampi(cy - rad, 0, h - 1)
    y1     = _clampi(cy + rad, 0, h - 1)
    x0     = _clampi(cx - rad, 0, w - 1)
    x1     = _clampi(cx + rad, 0, w - 1)
    two_s2 = 2 * s.sigma * s.sigma
    for y in range(y0, y1 + 1):
        dy        = y / h - s.y
        row_base  = y * w * 3
        for x in range(x0, x1 + 1):
            dx    = x / w - s.x
            w_val = math.exp(-(dx * dx + dy * dy) / two_s2)
            idx   = row_base + x * 3
            grid[idx] += s.l * w_val
            if not s.is_lepton:
                grid[idx + 1] += s.a * w_val
                grid[idx + 2] += s.b * w_val


def _image_to_oklab_grid(rgba: bytes, src_w: int, src_h: int,
                          w: int, h: int) -> list[float]:
    out = [0.0] * (w * h * 3)
    for y in range(h):
        y0 = int(y * src_h / h)
        y1 = math.ceil((y + 1) * src_h / h)
        for x in range(w):
            x0 = int(x * src_w / w)
            x1 = math.ceil((x + 1) * src_w / w)
            r_sum = g_sum = b_sum = count = 0
            for iy in range(y0, y1):
                if iy >= src_h:
                    break
                for ix in range(x0, x1):
                    if ix >= src_w:
                        break
                    p     = (iy * src_w + ix) * 4
                    r_sum += rgba[p]
                    g_sum += rgba[p + 1]
                    b_sum += rgba[p + 2]
                    count += 1
            if count == 0:
                continue
            r = r_sum / count / 255.0
            g = g_sum / count / 255.0
            b = b_sum / count / 255.0
            l, a, bb = _srgb_to_oklab(r, g, b)
            idx          = (y * w + x) * 3
            out[idx]     = l
            out[idx + 1] = a
            out[idx + 2] = bb
    return out


# ---------------------------------------------------------------------------
# Color space
# ---------------------------------------------------------------------------

def _srgb_to_oklab(r: float, g: float, b: float):
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = lin(r), lin(g), lin(b)
    l1 = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m1 = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s1 = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = l1 ** (1/3)
    m_ = m1 ** (1/3)
    s_ = s1 ** (1/3)
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _oklab_to_srgb(l: float, a: float, b: float):
    l_ = l + 0.3963377774 * a + 0.2158037573 * b
    m_ = l - 0.1055613458 * a - 0.0638541728 * b
    s_ = l - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    r  = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g  = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
    def srt(c):
        if c <= 0.0031308:
            return 12.92 * c
        if c < 0:
            return 0.0
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055
    return srt(r), srt(g), srt(bl)


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

class _BitWriter:
    __slots__ = ("_buf", "_acc", "_n")

    def __init__(self):
        self._buf = bytearray()
        self._acc = 0
        self._n   = 0

    def write(self, val: int, bits: int) -> None:
        self._acc = ((self._acc << bits) | (val & ((1 << bits) - 1)))
        self._n  += bits
        while self._n >= 8:
            shift = self._n - 8
            self._buf.append((self._acc >> shift) & 0xFF)
            self._n -= 8

    def getbytes(self) -> bytes:
        if self._n > 0:
            self._buf.append((self._acc << (8 - self._n)) & 0xFF)
        return bytes(self._buf)


class _BitReader:
    __slots__ = ("_data", "_pos", "_rem", "_curr")

    def __init__(self, data: bytes):
        self._data = data
        self._pos  = 0
        self._rem  = 0
        self._curr = 0

    def read(self, bits: int) -> int:
        val = 0
        while bits > 0:
            if self._rem == 0:
                if self._pos >= len(self._data):
                    return val << bits
                self._curr = self._data[self._pos]
                self._pos += 1
                self._rem  = 8
            take   = min(self._rem, bits)
            shift  = self._rem - take
            mask   = (1 << take) - 1
            chunk  = (self._curr >> shift) & mask
            val    = (val << take) | chunk
            self._rem -= take
            bits      -= take
        return val


def _pack_v4(mean: int, splats: list[_Splat]) -> bytes:
    bw = _BitWriter()
    bw.write(mean, 16)

    # Baryons (3 × 22 bits)
    count = 0
    for s in splats:
        if s.is_lepton:
            continue
        if count >= 3:
            break
        xi   = _clampi(int(s.x * 15.0 + 0.5), 0, 15)
        yi   = _clampi(int(s.y * 15.0 + 0.5), 0, 15)
        sig_i = _sigma_idx(s.sigma)
        l_q  = _quant(s.l, -0.8, 0.8, 4)
        a_q  = _quant(s.a, -0.4, 0.4, 4)
        b_q  = _quant(s.b, -0.4, 0.4, 4)
        bw.write(xi, 4); bw.write(yi, 4); bw.write(sig_i, 2)
        bw.write(l_q, 4); bw.write(a_q, 4); bw.write(b_q, 4)
        count += 1
    while count < 3:
        bw.write(0, 22); count += 1

    # Leptons (3 × 15 bits)
    count = 0
    for s in splats:
        if not s.is_lepton:
            continue
        if count >= 3:
            break
        xi    = _clampi(int(s.x * 15.0 + 0.5), 0, 15)
        yi    = _clampi(int(s.y * 15.0 + 0.5), 0, 15)
        sig_i = _sigma_idx(s.sigma)
        l_q   = _quant(s.l, -0.8, 0.8, 5)
        bw.write(xi, 4); bw.write(yi, 4); bw.write(sig_i, 2); bw.write(l_q, 5)
        count += 1
    while count < 3:
        bw.write(0, 15); count += 1

    bw.write(0, 1)  # padding
    return bw.getbytes()


def _unpack_v4(data: bytes):
    br     = _BitReader(data)
    packed = br.read(16)
    mean_l, mean_a, mean_b = _unpack_mean(packed)
    splats = []

    # Baryons
    for _ in range(3):
        xi   = br.read(4); yi  = br.read(4); sig_i = br.read(2)
        l_q  = br.read(4); a_q = br.read(4); b_q   = br.read(4)
        if xi == 0 and yi == 0 and l_q == 0 and a_q == 0 and b_q == 0:
            continue
        splats.append(_Splat(
            x=xi / 15.0, y=yi / 15.0, sigma=SIGMA_TABLE[sig_i],
            l=_unquant(l_q, -0.8, 0.8, 4),
            a=_unquant(a_q, -0.4, 0.4, 4),
            b=_unquant(b_q, -0.4, 0.4, 4),
            is_lepton=False
        ))

    # Leptons
    for _ in range(3):
        xi    = br.read(4); yi   = br.read(4); sig_i = br.read(2); l_q = br.read(5)
        if xi == 0 and yi == 0 and l_q == 0:
            continue
        splats.append(_Splat(
            x=xi / 15.0, y=yi / 15.0, sigma=SIGMA_TABLE[sig_i],
            l=_unquant(l_q, -0.8, 0.8, 5),
            a=0.0, b=0.0,
            is_lepton=True
        ))

    return mean_l, mean_a, mean_b, splats


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _pack_mean(l: float, a: float, b: float) -> int:
    li = _clampi(int(l * 63.5), 0, 63)
    ai = _clampi(int(((a + 0.2) / 0.4) * 31.5), 0, 31)
    bi = _clampi(int(((b + 0.2) / 0.4) * 31.5), 0, 31)
    return (li << 10) | (ai << 5) | bi


def _unpack_mean(p: int):
    li = (p >> 10) & 0x3F
    ai = (p >> 5)  & 0x1F
    bi =  p        & 0x1F
    return li / 63.0, (ai / 31.0) * 0.4 - 0.2, (bi / 31.0) * 0.4 - 0.2


def _quant(v: float, lo: float, hi: float, bits: int) -> int:
    steps = (1 << bits) - 1
    norm  = (v - lo) / (hi - lo)
    return _clampi(int(norm * steps + 0.5), 0, steps)


def _unquant(v: int, lo: float, hi: float, bits: int) -> float:
    steps = (1 << bits) - 1
    return (v / steps) * (hi - lo) + lo


def _sigma_idx(sigma: float) -> int:
    best_i = 0
    best_d = abs(SIGMA_TABLE[0] - sigma)
    for i in range(1, len(SIGMA_TABLE)):
        d = abs(SIGMA_TABLE[i] - sigma)
        if d < best_d:
            best_d = d; best_i = i
    return best_i


def _clampi(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v
