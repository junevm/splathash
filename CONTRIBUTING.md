# Contributing to SplatHash

## Architecture

SplatHash is polyglot by design. One algorithm, multiple implementations, zero divergence.

- **Go (`go/`)** — First-class reference implementation. Algorithm changes start here. When in doubt about correct behavior, the Go output wins.
- **TypeScript (`ts/`)** — Second-class. Isomorphic: same code runs in Node.js and browsers. Must produce bit-identical output to Go.
- **Python (`py/`)** — Second-class. Requires Pillow for image loading. Must produce bit-identical output to Go.

## Adding a New Language

1. Create a directory named after the language (e.g., `rust/`, `swift/`, `java/`).
2. Implement the algorithm following [ALGORITHM.md](./ALGORITHM.md) precisely. Pay close attention to the **Implementation Notes** section at the bottom.
3. Write a test that encodes every file in `../assets/` and compares the hex hash against the Go reference (run `go run examples/simple/main.go` from `go/` to get the reference hashes).
4. Add a `test:<lang>` task to `mise.toml` that installs dependencies and runs the tests.
5. Add the language to the `depends` list of the `test` task in `mise.toml`.
6. Add a row to the Implementations table in `README.md`.

## Removing a Language

1. Delete the implementation directory.
2. Remove its `test:<lang>` task from `mise.toml` and remove it from the `test` task's `depends`.
3. Remove its row from the Implementations table in `README.md`.

## Verifying Cross-Language Parity

All implementations must produce identical 16-byte hashes for the same input image.

```bash
# Generate Go reference hashes
cd go && go run examples/simple/main.go

# Compare with TypeScript
cd ts && npm run example

# Compare with Python
cd py && python example.py
```

All three should print the same hex hashes for each filename.

Run `mise run test` to execute all test suites at once.

## Modifying the Algorithm

Any change to constants (`SIGMA_TABLE`, `RIDGE_LAMBDA`), quantization ranges, or bit-packing layout is a **breaking change** — all implementations must be updated simultaneously.

When changing the algorithm:

1. Update `go/splathash.go` first (reference).
2. Port the same change to `ts/src/splathash.ts`.
3. Port the same change to `py/splathash.py`.
4. Update `ALGORITHM.md` to describe the new behavior.
5. Verify all tests pass: `mise run test`.

## Pull Requests

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Make your changes.
4. Run `mise run test` — all tests must pass.
5. Open a pull request with a clear description of the change.
