# SplatHash — Python

Python implementation of [SplatHash](../../README.md): compress any image to 16 bytes and reconstruct it.

## Requirements

- Python 3.8+
- [Pillow](https://python-pillow.org/) (for loading image files)

## Installation

```bash
pip install splathash
```

For local development from the repo:

```bash
pip install -e src/py
```

## Usage

```python
from splathash import encode, encode_raw, decode

# Encode from a file path
with open("photo.jpg", "rb") as f:
    hash_bytes = encode(f)          # bytes, always 16 bytes

# Or from a file path string directly
hash_bytes = encode("photo.jpg")

# Or from a PIL Image
from PIL import Image
img = Image.open("photo.jpg")
hash_bytes = encode(img)

# Encode from raw RGBA bytes
hash_bytes = encode_raw(rgba_bytes, width, height)

# Decode to 32×32 RGBA
pixels = decode(hash_bytes)         # bytes, 32 * 32 * 4 = 4096 bytes
```

See [`example.py`](example.py) for a full working example that encodes a directory of images.

## API

### `encode(source) -> bytes`

Encodes an image to a 16-byte SplatHash.

`source` can be:
- A file path (`str` or `Path`)
- A file-like object (binary mode)
- A `PIL.Image.Image`

Returns `bytes` of exactly 16 bytes.

### `encode_raw(rgba: bytes, width: int, height: int) -> bytes`

Encodes raw RGBA pixel data. `rgba` must be `width * height * 4` bytes.

Returns `bytes` of exactly 16 bytes.

### `decode(hash_bytes: bytes) -> bytes`

Decodes a 16-byte SplatHash to a 32×32 RGBA image.

Returns `bytes` of exactly `32 * 32 * 4 = 4096` bytes (raw RGBA, row-major).

Raises `ValueError` if `hash_bytes` is not exactly 16 bytes.

## Converting the decoded bytes to an image

```python
from PIL import Image
from splathash import encode, decode

hash_bytes = encode("photo.jpg")
pixels = decode(hash_bytes)

img = Image.frombytes("RGBA", (32, 32), pixels)
img.save("preview.png")
```

## Testing

```bash
pip install pytest Pillow
pytest test_splathash.py -v
```

Or via mise from the repo root:

```bash
mise run test:py
```

## How It Works

See [ALGORITHM.md](../../ALGORITHM.md) for the full technical specification.
