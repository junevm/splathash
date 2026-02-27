# SplatHash — Python implementation
#
# Encodes any image into 16 bytes and reconstructs a 32x32 preview.
# An image is decomposed into a background color (Mean) and six Gaussian blobs (Splats):
#   - 3 Baryons: full-color Splats for dominant features
#   - 3 Leptons: luma-only Splats for texture and detail
#
# Splat positions are found by separable 2-D Gaussian correlation (matching pursuit).
# Ridge Regression then refines all weights together. All computation is in Oklab.
# The hash fits into exactly 128 bits.
#
# Requires: Pillow (pip install Pillow)
#
# API:
#   encode(source) -> bytes          source: file path, file object, or PIL.Image
#   encode_raw(rgba, width, height) -> bytes
#   decode(hash_bytes) -> bytes      returns 32*32*4 RGBA bytes

from __future__ import annotations

import math
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
GAUSS_TABLE_MAX = 1923  # max dsq = 31² + 31² = 1922 for a 32×32 grid


class _Splat(NamedTuple):
    x:        float
    y:        float
    sigma:    float
    l:        float
    a:        float
    b:        float
    is_lepton: bool


# ---------------------------------------------------------------------------
# Package-level precomputed look-up tables (same as Go/TS for cross-language parity)
# ---------------------------------------------------------------------------

# gaussLUT[si][dsq] = exp(-dsq / (2·σᵢ²·W²)), zeroed below 1e-7
_gauss_lut: list[list[float]] = [[] for _ in range(4)]

# gaussKernel1D[si] = [k(0), k(1), ..., k(hw)] (half-kernel, symmetric)
_gauss_kernel_1d: list[list[float]] = [[] for _ in range(4)]

# half-width of the non-zero region for each sigma
_kernel_hw: list[int] = [0, 0, 0, 0]

# gaussPow[si] = (Σ_{d=-hw}^{hw} k[d]²)²
_gauss_pow: list[float] = [0.0, 0.0, 0.0, 0.0]

# linToSrgbLUT[i] = sRGB-gamma(i / 1023) for i = 0 .. 1023.
_lin_to_srgb_lut: list[float] = []

# srgbLinLUT[v] = linear-light(v / 255) for v = 0 .. 255.
_srgb_lin_lut: list[float] = []

# cbrtLUT[i] = cbrt(i / 1024) for i = 0 .. 1024.
_cbrt_lut: list[float] = []


def _init_luts() -> None:
    W  = TARGET_SIZE
    W2 = W * W

    for si, sigma in enumerate(SIGMA_TABLE):
        scale2 = 2.0 * sigma * sigma * W2
        lut: list[float] = []
        for dsq in range(GAUSS_TABLE_MAX):
            v = math.exp(-dsq / scale2)
            if v < 1e-7:
                v = 0.0
            lut.append(v)
        _gauss_lut[si] = lut

        # Build 1-D half-kernel.
        hw = 0
        for d in range(W):
            if lut[d * d] < 1e-7:
                break
            hw = d
        _kernel_hw[si] = hw
        kern = [lut[d * d] for d in range(hw + 1)]
        _gauss_kernel_1d[si] = kern

        # Normalization factor gg = (Σ_d k[d]²)²
        sum1d = 0.0
        for d in range(-hw, hw + 1):
            v = kern[abs(d)]
            sum1d += v * v
        _gauss_pow[si] = sum1d * sum1d

    # sRGB → linear LUT (8-bit input).
    for v in range(256):
        c = v / 255.0
        _srgb_lin_lut.append(c / 12.92 if c <= 0.04045
                             else ((c + 0.055) / 1.055) ** 2.4)

    # linear → sRGB gamma LUT (1024 steps over [0, 1]).
    def _lin_to_srgb_scalar(c: float) -> float:
        if c <= 0.0031308:
            return 12.92 * c
        if c < 0:
            return 0.0
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

    for i in range(1024):
        _lin_to_srgb_lut.append(_lin_to_srgb_scalar(i / 1023.0))

    # Cube-root LUT for [0, 1].
    for i in range(1025):
        _cbrt_lut.append((i / 1024.0) ** (1.0 / 3.0))


_init_luts()


def _cbrt_fast(x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return _cbrt_lut[1024]
    return _cbrt_lut[int(x * 1024.0 + 0.5)]


def _lin_to_srgb_fast(c: float) -> float:
    if c <= 0:
        return 0.0
    if c >= 1:
        return 1.0
    return _lin_to_srgb_lut[int(c * 1023.0 + 0.5)]


def _sigma_idx(sigma: float) -> int:
    best_i = 0
    best_d = abs(SIGMA_TABLE[0] - sigma)
    for i in range(1, 4):
        d = abs(SIGMA_TABLE[i] - sigma)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


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
    W, H = TARGET_SIZE, TARGET_SIZE

    # 1. Downsample and convert to Oklab.
    grid = _image_to_oklab_grid(rgba, width, height, W, H)

    # 2. Compute mean and quantise immediately.
    n = W * H
    mean_l = sum(grid[i * 3]     for i in range(n)) / n
    mean_a = sum(grid[i * 3 + 1] for i in range(n)) / n
    mean_b = sum(grid[i * 3 + 2] for i in range(n)) / n
    packed_mean = _pack_mean(mean_l, mean_a, mean_b)
    mean_l, mean_a, mean_b = _unpack_mean(packed_mean)

    # 3. Initial residuals.
    res_l = [grid[i * 3]     - mean_l for i in range(n)]
    res_a = [grid[i * 3 + 1] - mean_a for i in range(n)]
    res_b = [grid[i * 3 + 2] - mean_b for i in range(n)]

    # 4. Greedy matching pursuit using separable Gaussian correlation.
    basis = _find_all_splats(res_l, res_a, res_b, W, H, 6)

    # 5. Global Ridge Regression over all splat weights simultaneously.
    if basis:
        basis = _solve_v4_weights(basis, grid, mean_l, mean_a, mean_b, W, H)

    # 6. Pack.
    return _pack_v4(packed_mean, basis)


def decode(hash_bytes: bytes) -> bytes:
    """Decode a 16-byte SplatHash into a 32x32 RGBA image (bytes, length 4096)."""
    if len(hash_bytes) != 16:
        raise ValueError(f"Invalid SplatHash: expected 16 bytes, got {len(hash_bytes)}")

    mean_l, mean_a, mean_b, splats = _unpack_v4(hash_bytes)
    w, h = 32, 32
    grid = [0.0] * (w * h * 3)

    # Fill background.
    for i in range(w * h):
        grid[i * 3]     = mean_l
        grid[i * 3 + 1] = mean_a
        grid[i * 3 + 2] = mean_b

    # Composite splats.
    for s in splats:
        _add_splat_to_grid(grid, s, w, h)

    # Convert to RGBA bytes.
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

def _solve_v4_weights(basis, grid, mean_l, mean_a, mean_b, w, h):
    n_total = len(basis)
    M = w * h

    # Reconstruct full target vectors (grid − mean).
    t_l = [grid[i * 3]     - mean_l for i in range(M)]
    t_a = [grid[i * 3 + 1] - mean_a for i in range(M)]
    t_b = [grid[i * 3 + 2] - mean_b for i in range(M)]

    # Precompute activation maps.
    activations = [_compute_basis_map(s, w, h) for s in basis]

    # Solve L for all splats. A and B only for Baryons (first up-to-3).
    n_baryons = min(n_total, 3)
    x_l = _solve_channel(activations,            t_l, n_total,   RIDGE_LAMBDA)
    x_a = _solve_channel(activations[:n_baryons], t_a, n_baryons, RIDGE_LAMBDA)
    x_b = _solve_channel(activations[:n_baryons], t_b, n_baryons, RIDGE_LAMBDA)

    out = []
    for i, s in enumerate(basis):
        out.append(_Splat(
            x=s.x, y=s.y, sigma=s.sigma,
            l=x_l[i],
            a=x_a[i] if i < 3 else 0.0,
            b=x_b[i] if i < 3 else 0.0,
            is_lepton=s.is_lepton,
        ))
    return out


def _solve_channel(activations, target, n, lam):
    if n == 0:
        return []
    m = len(target)
    ata = [0.0] * (n * n)
    atb = [0.0] * n

    for i in range(n):
        ai = activations[i]
        for j in range(i, n):
            s = 0.0
            aj = activations[j]
            for p in range(m):
                s += ai[p] * aj[p]
            ata[i * n + j] = s
            ata[j * n + i] = s
        s = 0.0
        for p in range(m):
            s += ai[p] * target[p]
        atb[i] = s

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


def _compute_basis_map(s: _Splat, w: int, h: int) -> list[float]:
    out = [0.0] * (w * h)
    si  = _sigma_idx(s.sigma)
    hw  = _kernel_hw[si]
    lut = _gauss_lut[si]
    cx  = int(s.x * w)
    cy  = int(s.y * h)
    y0  = _clampi(cy - hw, 0, h - 1)
    y1  = _clampi(cy + hw, 0, h - 1)
    x0  = _clampi(cx - hw, 0, w - 1)
    x1  = _clampi(cx + hw, 0, w - 1)
    for y in range(y0, y1 + 1):
        dy = y - cy
        row_base = y * w
        for x in range(x0, x1 + 1):
            dx  = x - cx
            dsq = dx * dx + dy * dy
            if dsq < GAUSS_TABLE_MAX:
                out[row_base + x] = lut[dsq]
    return out


# ---------------------------------------------------------------------------
# Greedy search: sequential matching pursuit
# ---------------------------------------------------------------------------

def _find_all_splats(res_l, res_a, res_b, w, h, n_splats):
    splats: list[_Splat] = []

    # Pre-allocate scratch buffers.
    N = w * h
    tmp_l    = [0.0] * N
    tmp_a    = [0.0] * N
    tmp_b    = [0.0] * N
    score_map = [-1.0] * N
    sigma_map = [-1]   * N

    while len(splats) < n_splats:
        is_baryon = len(splats) < 3

        # ── 1. Build per-pixel score map over all 4 sigmas ──────────────────
        for i in range(N):
            score_map[i] = -1.0
            sigma_map[i] = -1

        for si in range(4):
            kern   = _gauss_kernel_1d[si]
            hw     = _kernel_hw[si]
            inv_gg = 1.0 / _gauss_pow[si]

            # ── Horizontal pass (zero-padding) ─────────────────────────────
            for y in range(h):
                row_off = y * w
                for x in range(w):
                    sL = kern[0] * res_l[row_off + x]
                    sA = kern[0] * res_a[row_off + x]
                    sB = kern[0] * res_b[row_off + x]
                    for d in range(1, hw + 1):
                        k  = kern[d]
                        xl = x - d
                        if xl >= 0:
                            sL += k * res_l[row_off + xl]
                            if is_baryon:
                                sA += k * res_a[row_off + xl]
                                sB += k * res_b[row_off + xl]
                        xr = x + d
                        if xr < w:
                            sL += k * res_l[row_off + xr]
                            if is_baryon:
                                sA += k * res_a[row_off + xr]
                                sB += k * res_b[row_off + xr]
                    tmp_l[row_off + x] = sL
                    if is_baryon:
                        tmp_a[row_off + x] = sA
                        tmp_b[row_off + x] = sB

            # ── Vertical pass (zero-padding) + score update ─────────────────
            for x in range(w):
                for y in range(h):
                    sL = kern[0] * tmp_l[y * w + x]
                    sA = kern[0] * tmp_a[y * w + x]
                    sB = kern[0] * tmp_b[y * w + x]
                    for d in range(1, hw + 1):
                        k  = kern[d]
                        yu = y - d
                        if yu >= 0:
                            sL += k * tmp_l[yu * w + x]
                            if is_baryon:
                                sA += k * tmp_a[yu * w + x]
                                sB += k * tmp_b[yu * w + x]
                        yd = y + d
                        if yd < h:
                            sL += k * tmp_l[yd * w + x]
                            if is_baryon:
                                sA += k * tmp_a[yd * w + x]
                                sB += k * tmp_b[yd * w + x]
                    i = y * w + x
                    score = ((sL * sL + sA * sA + sB * sB) * inv_gg
                             if is_baryon else sL * sL * inv_gg)
                    if score > score_map[i]:
                        score_map[i] = score
                        sigma_map[i] = si

        # ── 2. Find single best pixel ────────────────────────────────────────
        best_score = -1.0
        best_idx   = -1
        for i in range(N):
            if score_map[i] > best_score:
                best_score = score_map[i]
                best_idx   = i
        if best_idx < 0 or best_score < 1e-9:
            break

        bx   = best_idx % w
        by   = best_idx // w
        si   = sigma_map[best_idx]
        kern = _gauss_kernel_1d[si]
        hw   = _kernel_hw[si]
        gg   = _gauss_pow[si]

        # ── 3. Compute L, A, B dot-products at winner (zero-padding) ─────────
        dot_l = dot_a = dot_b = 0.0
        for dy in range(-hw, hw + 1):
            yy = by + dy
            if yy < 0 or yy >= h:
                continue
            ky = kern[abs(dy)]
            for dx in range(-hw, hw + 1):
                xx = bx + dx
                if xx < 0 or xx >= w:
                    continue
                kv  = ky * kern[abs(dx)]
                off = yy * w + xx
                dot_l += kv * res_l[off]
                dot_a += kv * res_a[off]
                dot_b += kv * res_b[off]

        inv_gg = 1.0 / gg
        splat  = _Splat(
            x=bx / w, y=by / h, sigma=SIGMA_TABLE[si],
            l=dot_l * inv_gg, a=dot_a * inv_gg, b=dot_b * inv_gg,
            is_lepton=not is_baryon,
        )
        splats.append(splat)

        # ── 4. Subtract splat footprint from residuals ───────────────────────
        lut = _gauss_lut[si]
        y0  = _clampi(by - hw, 0, h - 1)
        y1  = _clampi(by + hw, 0, h - 1)
        x0  = _clampi(bx - hw, 0, w - 1)
        x1  = _clampi(bx + hw, 0, w - 1)
        for y in range(y0, y1 + 1):
            dy       = y - by
            row_base = y * w
            for x in range(x0, x1 + 1):
                dx  = x - bx
                dsq = dx * dx + dy * dy
                if dsq >= GAUSS_TABLE_MAX:
                    continue
                w_val = lut[dsq]
                if w_val == 0:
                    continue
                off         = row_base + x
                res_l[off] -= splat.l * w_val
                res_a[off] -= splat.a * w_val
                res_b[off] -= splat.b * w_val

    return splats


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _add_splat_to_grid(grid: list[float], s: _Splat, w: int, h: int) -> None:
    si  = _sigma_idx(s.sigma)
    hw  = _kernel_hw[si]
    lut = _gauss_lut[si]
    cx  = int(s.x * w)
    cy  = int(s.y * h)
    y0  = _clampi(cy - hw, 0, h - 1)
    y1  = _clampi(cy + hw, 0, h - 1)
    x0  = _clampi(cx - hw, 0, w - 1)
    x1  = _clampi(cx + hw, 0, w - 1)
    for y in range(y0, y1 + 1):
        dy       = y - cy
        row_base = y * w * 3
        for x in range(x0, x1 + 1):
            dx    = x - cx
            dsq   = dx * dx + dy * dy
            if dsq >= GAUSS_TABLE_MAX:
                continue
            w_val = lut[dsq]
            if w_val == 0:
                continue
            idx          = row_base + x * 3
            grid[idx]     += s.l * w_val
            grid[idx + 1] += s.a * w_val
            grid[idx + 2] += s.b * w_val


def _image_to_oklab_grid(rgba: bytes, src_w: int, src_h: int,
                          w: int, h: int) -> list[float]:
    out = [0.0] * (w * h * 3)
    for y in range(h):
        sy = (y * src_h + src_h // 2) // h
        for x in range(w):
            sx = (x * src_w + src_w // 2) // w
            p  = (sy * src_w + sx) * 4
            r  = _srgb_lin_lut[rgba[p]]
            g  = _srgb_lin_lut[rgba[p + 1]]
            b  = _srgb_lin_lut[rgba[p + 2]]
            l, a, bb     = _srgb_lin_to_oklab(r, g, b)
            idx          = (y * w + x) * 3
            out[idx]     = l
            out[idx + 1] = a
            out[idx + 2] = bb
    return out


# ---------------------------------------------------------------------------
# Color space
# ---------------------------------------------------------------------------

def _srgb_lin_to_oklab(r: float, g: float, b: float):
    l1 = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m1 = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s1 = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_ = _cbrt_fast(l1)
    m_ = _cbrt_fast(m1)
    s_ = _cbrt_fast(s1)
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
    return _lin_to_srgb_fast(r), _lin_to_srgb_fast(g), _lin_to_srgb_fast(bl)


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


# packV4 encodes the hash into 128 bits:
#   Mean      : 16 bits  (6L + 5A + 5B)
#   3 Baryons : 22 bits each = 66 bits  (x4 y4 σ2 L4 A4 B4 — full colour)
#   3 Leptons : 15 bits each = 45 bits  (x4 y4 σ2 L5          — luma-only)
#   Reserved  :  1 bit
#   Total     : 128 bits
def _pack_v4(mean: int, splats: list[_Splat]) -> bytes:
    bw = _BitWriter()
    bw.write(mean, 16)

    # 3 Baryon splats (full-colour) — 22 bits each: x4 y4 σ2 L4 A4 B4.
    count = 0
    for s in splats:
        if s.is_lepton:
            continue
        if count >= 3:
            break
        xi    = _clampi(int(s.x * 15.0 + 0.5), 0, 15)
        yi    = _clampi(int(s.y * 15.0 + 0.5), 0, 15)
        sig_i = _sigma_idx(s.sigma)
        l_q   = _quant(s.l, -0.8, 0.8, 4)
        a_q   = _quant(s.a, -0.4, 0.4, 4)
        b_q   = _quant(s.b, -0.4, 0.4, 4)
        bw.write(xi, 4); bw.write(yi, 4); bw.write(sig_i, 2)
        bw.write(l_q, 4); bw.write(a_q, 4); bw.write(b_q, 4)
        count += 1
    while count < 3:
        bw.write(0, 22)
        count += 1

    # 3 Lepton splats (luma-only) — 15 bits each: x4 y4 σ2 L5.
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
        bw.write(xi, 4); bw.write(yi, 4); bw.write(sig_i, 2)
        bw.write(l_q, 5)
        count += 1
    while count < 3:
        bw.write(0, 15)
        count += 1

    bw.write(0, 1)  # reserved
    return bw.getbytes()


def _unpack_v4(data: bytes):
    br     = _BitReader(data)
    packed = br.read(16)
    mean_l, mean_a, mean_b = _unpack_mean(packed)
    splats = []

    # 3 Baryon splats — 22 bits each (x4 y4 σ2 L4 A4 B4).
    for _ in range(3):
        xi    = br.read(4); yi    = br.read(4); sig_i = br.read(2)
        l_q   = br.read(4); a_q   = br.read(4); b_q   = br.read(4)
        if xi == 0 and yi == 0 and l_q == 0 and a_q == 0 and b_q == 0:
            continue
        splats.append(_Splat(
            x=xi / 15.0, y=yi / 15.0, sigma=SIGMA_TABLE[sig_i],
            l=_unquant(l_q, -0.8, 0.8, 4),
            a=_unquant(a_q, -0.4, 0.4, 4),
            b=_unquant(b_q, -0.4, 0.4, 4),
            is_lepton=False,
        ))

    # 3 Lepton splats — 15 bits each (x4 y4 σ2 L5), luma-only.
    for _ in range(3):
        xi    = br.read(4); yi    = br.read(4); sig_i = br.read(2)
        l_q   = br.read(5)
        if xi == 0 and yi == 0 and l_q == 0:
            continue
        splats.append(_Splat(
            x=xi / 15.0, y=yi / 15.0, sigma=SIGMA_TABLE[sig_i],
            l=_unquant(l_q, -0.8, 0.8, 5),
            a=0.0, b=0.0,
            is_lepton=True,
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


def _clampi(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v
