<div align="center">
<img src="./assets/logo.svg" alt="SplatHash" width="120" height="120" />

**compress any image to 16 bytes — and reconstruct it**

[Algorithm](./ALGORITHM.md) · [Demo](https://junevm.github.io/splathash/) · [Bugs](https://github.com/junevm/splathash/issues) · [Contributing](./CONTRIBUTING.md)

</div>

**SplatHash** encodes any image into exactly 16 bytes (a 22-char base64url string) and decodes it back to a 32×32 blurry preview in 0.067 ms. Go, TypeScript, and Python all produce bit-for-bit identical hashes.

![Visual comparison of SplatHash, ThumbHash, and BlurHash reconstructions](./docs/comparison.png)

_Columns: original · SplatHash 32×32 · ThumbHash · BlurHash. Regenerate: `mise run compare`._

## Install

| Language        | Package                                     | Docs                                   |
| :-------------- | :------------------------------------------ | :------------------------------------- |
| Go              | `go get github.com/junevm/splathash/src/go` | [src/go/README.md](./src/go/README.md) |
| TypeScript / JS | `npm install splathash-ts`                  | [src/ts/README.md](./src/ts/README.md) |
| Python          | `pip install splathash-py`                  | [src/py/README.md](./src/py/README.md) |

Go is the reference implementation. All others are verified bit-for-bit against it.

## Benchmarks

Go benchmarks on Intel Core i5-9300H (`go test -bench=. -benchmem`).

|               | **SplatHash** |   ThumbHash |    BlurHash |
| :------------ | ------------: | ----------: | ----------: |
| Decode        |  **0.067 ms** |     0.50 ms |     6.55 ms |
| Encode        |       3.53 ms | **0.86 ms** |      445 ms |
| Decode allocs |         **7** |       1,168 |           5 |
| Bytes         |  **16 fixed** |       25–37 |       20–25 |
| String        |  **22 chars** | 34–50 chars | 27–30 chars |

Decode runs on every page load for every user. Encode runs once at upload. Optimize for decode.

<details>
<summary>Raw output</summary>

```
BenchmarkEncodeSplatHash-8      303      3530083 ns/op   100760 B/op   29 allocs/op
BenchmarkEncodeThumbHash-8     1431       863553 ns/op     1015 B/op    6 allocs/op
BenchmarkEncodeBlurHash-8         3    445696421 ns/op 33358234 B/op    8 allocs/op
BenchmarkDecodeSplatHash-8    17182        67622 ns/op    29584 B/op    7 allocs/op
BenchmarkDecodeThumbHash-8     6559       503366 ns/op    58408 B/op 1168 allocs/op
BenchmarkDecodeBlurHash-8       171      6553100 ns/op   547552 B/op    5 allocs/op
```

</details>

## Comparison

|                                 |      SplatHash      |  ThumbHash  |  BlurHash   |
| :------------------------------ | :-----------------: | :---------: | :---------: |
| Fixed size                      |     ✅ 16 bytes     | ❌ variable | ❌ variable |
| Storable as 128-bit int         |         ✅          |     ❌      |     ❌      |
| Perceptual color space (Oklab)  |         ✅          |     ❌      |     ❌      |
| Spatially localized basis       |    ✅ Gaussians     |  ❌ global  |  ❌ global  |
| Global weight optimization      | ✅ Ridge Regression |     ❌      |     ❌      |
| Alpha channel                   |         ✅          |     ✅      |     ❌      |
| Bit-exact cross-language parity |         ✅          |     ❌      |     ❌      |
| Configurable quality vs size    |         ❌          |     ❌      |     ✅      |

## How It Works

Background color + six Gaussian blobs placed by matching pursuit, color-optimized by Ridge Regression, all in Oklab, packed into 128 bits. Full spec: [ALGORITHM.md](./ALGORITHM.md).

## Development

```bash
mise install     # install Go and Node at pinned versions
mise run test    # run all tests (Go + TypeScript + Python)
mise run bench   # Go benchmarks
mise run compare # regenerate docs/comparison.png
```

## License

See [LICENSE](./LICENSE).
