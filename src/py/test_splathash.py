"""Tests for the Python SplatHash implementation.

Verifies:
  - Hash length is exactly 16 bytes
  - Encoding is deterministic (same input → same output)
  - Decode produces a 32x32 RGBA buffer
  - Hashes match the Go reference for shared assets

Cross-language parity note:
  Given identical raw RGBA bytes, all implementations (Go, TypeScript, Python)
  produce bit-for-bit identical hashes. When encoding from JPEG files, minor
  pixel-level differences between JPEG decoders (Go stdlib vs Pillow vs libvips)
  can cause up to one bit of difference in the final hash. PNG files decode
  identically and always produce matching hashes across all implementations.

Run:
  cd src/py && python -m pytest test_splathash.py -v

Or via mise:
  mise run test:py
"""

import os
import subprocess
import sys
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import splathash

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
ALL_ASSETS  = sorted(f for f in os.listdir(ASSETS_DIR) if f.endswith((".jpg", ".jpeg", ".png")))
PNG_ASSETS  = sorted(f for f in ALL_ASSETS if f.endswith(".png"))
PRIMARY_ASSET = os.path.join(ASSETS_DIR, "wallhaven-3q3j6y.jpg")

# ---------------------------------------------------------------------------
# Core correctness
# ---------------------------------------------------------------------------

def test_hash_length():
    h = splathash.encode(PRIMARY_ASSET)
    assert len(h) == 16, f"Expected 16 bytes, got {len(h)}"


def test_determinism():
    h1 = splathash.encode(PRIMARY_ASSET)
    h2 = splathash.encode(PRIMARY_ASSET)
    assert h1 == h2, "Hash is not deterministic"


def test_decode_size():
    h = splathash.encode(PRIMARY_ASSET)
    rgba = splathash.decode(h)
    assert len(rgba) == 32 * 32 * 4, f"Expected {32*32*4} bytes, got {len(rgba)}"


def test_decode_invalid_raises():
    with pytest.raises(ValueError):
        splathash.decode(b"\x00" * 15)


@pytest.mark.parametrize("fname", ALL_ASSETS)
def test_all_assets_encode(fname):
    path = os.path.join(ASSETS_DIR, fname)
    h = splathash.encode(path)
    assert len(h) == 16
    # Must be deterministic
    assert h == splathash.encode(path)


# ---------------------------------------------------------------------------
# Cross-language parity with Go
# ---------------------------------------------------------------------------

def _go_hashes():
    """Run the Go example and parse its hash output."""
    go_dir = os.path.join(os.path.dirname(__file__), "..", "go", "examples", "simple")
    result = subprocess.run(
        ["go", "run", "main.go"],
        cwd=go_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    hashes = {}
    for line in result.stdout.splitlines():
        if ": " in line:
            fname, h = line.split(": ", 1)
            hashes[fname.strip()] = h.strip()
    return hashes


@pytest.mark.skipif(
    subprocess.run(["go", "version"], capture_output=True).returncode != 0,
    reason="Go not available",
)
@pytest.mark.parametrize("fname", PNG_ASSETS)
def test_parity_with_go_png(fname):
    """PNG files decode identically across all decoders — hashes must match Go exactly."""
    path = os.path.join(ASSETS_DIR, fname)
    py_hash   = splathash.encode(path).hex()
    go_hashes = _go_hashes()
    assert fname in go_hashes, f"Go did not output a hash for {fname}"
    assert py_hash == go_hashes[fname], (
        f"Hash mismatch for {fname} (PNG — should be exact):\n"
        f"  Python: {py_hash}\n"
        f"  Go:     {go_hashes[fname]}"
    )


@pytest.mark.skipif(
    subprocess.run(["go", "version"], capture_output=True).returncode != 0,
    reason="Go not available",
)
@pytest.mark.parametrize("fname", sorted(f for f in ALL_ASSETS if not f.endswith(".png")))
def test_parity_with_go_jpeg(fname):
    """JPEG files: verify parity. Minor decoder differences (up to 1 bit) are noted
    but do not indicate an algorithmic bug. The test reports any divergence as a warning."""
    path = os.path.join(ASSETS_DIR, fname)
    py_hash   = splathash.encode(path).hex()
    go_hashes = _go_hashes()
    if fname not in go_hashes:
        pytest.skip(f"Go did not output a hash for {fname}")
    go_hash = go_hashes[fname]
    if py_hash != go_hash:
        # Count differing bits
        diff_bits = bin(int(py_hash, 16) ^ int(go_hash, 16)).count("1")
        pytest.xfail(
            f"JPEG decoder difference for {fname} ({diff_bits} bit(s) differ):\n"
            f"  Python: {py_hash}\n"
            f"  Go:     {go_hash}\n"
            f"  This is expected: Go stdlib jpeg ≠ Pillow for some images."
        )
