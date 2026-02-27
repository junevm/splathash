# SplatHash — Go

Go implementation of [SplatHash](../../README.md): compress any image to 16 bytes and reconstruct it.

This is the **reference implementation**. All other language implementations are tested for bit-for-bit parity against this one.

## Installation

```bash
go get github.com/junevm/splathash/src/go
```

Requires Go 1.21 or later.

## Usage

```go
package main

import (
    "fmt"
    "image/jpeg"
    "os"

    splathash "github.com/junevm/splathash/src/go"
)

func main() {
    f, _ := os.Open("photo.jpg")
    img, _ := jpeg.Decode(f)
    f.Close()

    // Encode: any image.Image → 16-byte hash
    hash := splathash.EncodeImage(img)
    fmt.Printf("%x\n", hash) // e.g. a3f1bc...

    // Decode: 16-byte hash → 32×32 RGBA image.Image
    preview, err := splathash.DecodeImage(hash)
    if err != nil {
        panic(err)
    }
    _ = preview // 32×32 image.Image ready to use
}
```

See [`examples/simple/main.go`](examples/simple/main.go) for a complete working example that encodes a directory of images.

## API

### `EncodeImage(img image.Image) []byte`

Encodes any `image.Image` to a 16-byte SplatHash. The output is always exactly 16 bytes.

### `DecodeImage(hash []byte) (image.Image, error)`

Decodes a 16-byte SplatHash to a 32×32 RGBA `image.Image`. Returns an error if the input is not exactly 16 bytes.

## How It Works

SplatHash fits an image into 16 bytes by:

1. Downscaling to a 32×32 Oklab grid
2. Quantizing the mean color (16 bits)
3. Greedy search for 6 Gaussian splat positions and sizes
4. Ridge Regression to find optimal colors for all 6 splats simultaneously
5. Bit-packing everything into 128 bits

See [ALGORITHM.md](../../ALGORITHM.md) for the full technical specification.

## Testing

```bash
go test ./...
go test -bench=. -benchmem ./...
```

Or via mise from the repo root:

```bash
mise run test:go
mise run bench
```
