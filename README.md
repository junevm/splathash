<div align="center">
<img src="./assets/logo.svg" alt="SplatHash" width="120" height="120" />

**compress any image to 16 bytes — and reconstruct it**

[**Algorithm**](./ALGORITHM.md) | [**Report Bugs**](https://github.com/junevm/splathash/issues) | [**Contributing**](#contributing)

</div>

**SplatHash** encodes any image into exactly 16 bytes and decodes it back to a 32×32 blurry preview — in 0.067 ms. Go, TypeScript (Node + browser), and Python all produce bit-for-bit identical hashes.

## Visual Comparison

![Visual comparison of SplatHash, ThumbHash, and BlurHash reconstructions](./docs/comparison.png)

_Columns: original (downscaled), SplatHash (32×32), ThumbHash, BlurHash. Regenerate: `mise run compare`._

## Quick Start

### Go

```bash
go get github.com/junevm/splathash/src/go
```

```go
import splathash "github.com/junevm/splathash/src/go"

hash := splathash.EncodeImage(img)      // []byte, 16 bytes
img, err := splathash.DecodeImage(hash) // image.Image, 32×32
```

See [`src/go/README.md`](./src/go/README.md) for the full API.

### TypeScript

```bash
npm install splathash
```

```typescript
import { encode, decode } from "splathash";

const hash = encode(rgba, width, height); // Uint8Array, 16 bytes
const result = decode(hash); // result.rgba — Uint8ClampedArray, 32×32
```

In the browser pass `ImageData.data` directly — same package, no extra config.

See [`src/ts/README.md`](./src/ts/README.md) for the full API.

### Python

```bash
pip install splathash
```

```python
from splathash import encode, decode

with open("image.jpg", "rb") as f:
    hash_bytes = encode(f)          # bytes, 16 bytes

pixels = decode(hash_bytes)         # bytes, 32×32×4 RGBA
```

See [`src/py/README.md`](./src/py/README.md) for the full API.

## Benchmarks

**SplatHash is the smallest fixed-size image hash — 16 bytes exactly — and the fastest to decode.**

Measured on a real photo using Go's built-in benchmark tool (`go test -bench=. -benchmem`).

### Encode (one-time, at upload time)

| Algorithm     |        Time | Allocations |
| :------------ | ----------: | ----------: |
| **SplatHash** | **3.53 ms** |   29 allocs |
| ThumbHash     |     0.86 ms |    6 allocs |
| BlurHash      |      445 ms |    8 allocs |

SplatHash runs a full iterative search — 6 rounds of separable Gaussian correlation across all 4 sigma values, with residual subtraction and Ridge Regression — and still encodes **126× faster than BlurHash**. The extra work at encode time produces better-placed splats and a higher-quality placeholder.

### Decode (on every page load)

| Algorithm     |            Time | Allocations |
| :------------ | --------------: | ----------: |
| **SplatHash** | **0.067 ms** ✅ |    7 allocs |
| ThumbHash     |         0.50 ms | 1168 allocs |
| BlurHash      |         6.55 ms |    5 allocs |

**Decode is where SplatHash dominates.** At 0.067 ms it is 7× faster than ThumbHash and 97× faster than BlurHash, with a fraction of the allocations. Decode runs on every page load for every user — this is the number that matters in production.

<details>
<summary>Raw benchmark output</summary>

```text
$ mise run bench
goos: linux
goarch: amd64
pkg: github.com/junevm/splathash/src/go
cpu: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz
BenchmarkEncodeSplatHash-8           303           3530083 ns/op          100760 B/op         29 allocs/op
BenchmarkEncodeThumbHash-8          1431            863553 ns/op            1015 B/op          6 allocs/op
BenchmarkEncodeBlurHash-8              3         445696421 ns/op        33358234 B/op          8 allocs/op
BenchmarkDecodeSplatHash-8         17182             67622 ns/op           29584 B/op          7 allocs/op
BenchmarkDecodeThumbHash-8          6559            503366 ns/op           58408 B/op       1168 allocs/op
BenchmarkDecodeBlurHash-8            171           6553100 ns/op          547552 B/op          5 allocs/op
```

</details>

> Benchmarks run on Intel Core i5-9300H. Reproduce with `cd src/go && go test -bench=. -benchmem`.

## Why SplatHash

- **Fixed 16 bytes.** BlurHash and ThumbHash output 25–35 bytes minimum. SplatHash is always 16 bytes — storable as a single 128-bit integer, no variable-length overhead.
- **Localized blobs.** BlurHash and ThumbHash use cosine basis functions that are global — a feature in one corner pollutes the entire representation. SplatHash uses Gaussian blobs that are spatially bounded. A bright spot top-left doesn't corrupt the bottom-right.
- **Global color optimization.** After finding splat positions greedily, SplatHash runs Ridge Regression across all six splats simultaneously to find the best possible color weights. No other 16-byte algorithm does this.
- **Perceptually uniform.** All computation happens in Oklab, where equal numerical differences equal equal perceived differences. The algorithm minimizes the right thing.
- **Cross-language identical.** Go, TypeScript, and Python produce bit-for-bit identical hashes for the same input, verified by shared test assets.

## How It Works

SplatHash fits a background color plus six Gaussian blobs (3 full-color Baryons + 3 luma-only Leptons) to the image using sequential matching pursuit and Ridge Regression, all in Oklab, packed into 128 bits.

Full technical walkthrough: [ALGORITHM.md](./ALGORITHM.md).

## Development

```bash
mise install   # install Go and Node at pinned versions
mise run test  # run all tests (Go + TypeScript + Python)
mise run bench # Go benchmarks
mise run compare # regenerate docs/comparison.png
```

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

See [LICENSE](./LICENSE).
