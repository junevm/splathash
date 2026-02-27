# SplatHash Algorithm

This document explains the SplatHash V4 algorithm from first principles. No math background is assumed. The goal is to explain _why_ every step exists and _what it actually does_, not just _how_ to implement it.

---

## The Core Problem

You have an image. Maybe it's a 4K photo. You want to show a blurry placeholder while the real image loads — something that captures the color and general look of the image, but fits in 16 bytes. That's the problem SplatHash solves.

16 bytes is tiny. A single pixel in PNG is 3-4 bytes. A 32x32 thumbnail is around 3KB. So squeezing useful visual information into 16 bytes means making hard choices about what information matters most.

---

## The Starting Point: Representing an Image as Blobs

Instead of storing pixels, SplatHash describes an image as a **background color** plus a handful of **soft blobs** (called Splats) laid on top. A Splat is just a 2D Gaussian — it's brightest at its center and fades to nothing at the edges, like a soft brushstroke.

This is inspired by how humans perceive images at a glance. You don't memorize individual pixels. You notice "there's a warm blob in the upper left, a dark region in the center, the overall tone is cool blue." That's what SplatHash stores.

Each Splat has:

- A **position** (where the center is)
- A **size** (how wide the blob is)
- A **color** (in Oklab, see below)

Six splats total: 3 full-color ones (Baryons) for the major features, and 3 luma-only ones (Leptons) for texture and detail.

---

## Why Oklab Instead of RGB?

RGB is convenient for screens but bad for math. Equal RGB differences don't feel equal to humans. A shift from (0,0,0) to (30,0,0) looks dramatic, but a shift from (200,0,0) to (230,0,0) looks subtle — even though both are 30 RGB units.

Oklab is a color space designed so that equal numerical distances feel equal visually. When the algorithm tries to minimize the visual error between the original and its approximation, working in Oklab means minimizing the right thing.

The conversion:

1. Strip sRGB gamma (make it linear)
2. Multiply by a matrix to get into an intermediate LMS space (three cone types in the human eye)
3. Cube-root each component (this is the perceptual compression)
4. Multiply by another matrix to get L (lightness), a (green-red axis), b (blue-yellow axis)

In Oklab, L=0 is black, L=1 is white. a and b are typically small numbers near zero for neutral colors, larger when the color is saturated.

---

## Step 1: Downscale to 32x32

The algorithm works on a 32x32 grid, not the full image. This is a deliberate trade-off:

The full image might be 4000x3000. Working at full resolution would be slow and unnecessary — the output is a blurry placeholder anyway. 32x32 is enough to capture the structural layout of colors.

Each of the 32x32 cells gets the **area-averaged color** of the source pixels that map to it. This is a box filter — better than nearest-neighbor for capturing overall color correctly.

---

## Step 2: Compute and Quantize the Mean Color

Take the average Oklab value across all 1024 pixels in the 32x32 grid. This is the background color — the zeroth-order approximation of the image.

The mean is immediately **quantized** (rounded to the nearest representable value given the bits available). This matters: the rest of the algorithm works against the _quantized_ mean, not the original mean. If you don't do this, you'll solve for weights that try to compensate for a mean that won't actually be stored, and the reconstructed image will look wrong.

The mean is stored as 16 bits: 6 bits for L (lightness), 5 bits for a, 5 bits for b.

---

## Step 3: Greedy Basis Search (Finding the Splats)

With the mean subtracted, the algorithm has a **residual** — the difference between the original grid and the current approximation (which is just the mean color everywhere).

Now it needs to find 6 Splats that best explain this residual. It does this greedily, one at a time.

For each candidate Splat position and size:

1. Compute the Gaussian activation at every grid pixel (how much this Splat affects each pixel)
2. Project the residual onto this Gaussian (a dot product)
3. Score how much variance this Splat would explain

The Splat with the highest score is chosen, added to the reconstruction, and the residual is updated. Then find the next best Splat. Six times total.

**Why greedy?** Finding all 6 Splats simultaneously would require searching an enormous space. Greedy search is fast and finds good (not globally optimal) positions.

**Search details:**

- Positions are sampled at stride 2 (every other pixel) for speed
- Four Sigma (blob size) levels are tried: 0.025, 0.1, 0.2, 0.35 (as fractions of image width)
- The first 3 Splats found become Baryons (full-color), the next 3 become Leptons (luma-only)

The greedy coloring at this stage is temporary. The real colors are computed in step 4.

---

## Step 4: Global Weight Optimization (Ridge Regression)

This is the key step that separates SplatHash from simpler approaches.

After greedy search, you have 6 Splat positions and sizes. Their colors (weights) were estimated greedily, but those estimates are rough. The greedy algorithm found each Splat in isolation without considering how they interact.

Ridge Regression solves for the _best possible_ colors for all 6 Splats simultaneously, given their fixed positions and sizes.

### What is Ridge Regression?

You have 6 Splats, each with a Gaussian activation map (1024 numbers). Stack them into a matrix `A` where each row is one Gaussian. You want to find weights `x` (the colors) such that `A * x` is as close as possible to the residual `b`.

The plain least-squares solution is `x = (A^T A)^{-1} A^T b`. Ridge Regression adds a small regularization term λ to the diagonal of `A^T A`:

```
x = (A^T A + λI)^{-1} A^T b
```

This does two things:

1. Makes the matrix easier to invert (avoids blowup when Splats overlap heavily)
2. Penalizes extreme weight values, keeping the reconstruction stable

λ = 0.001 in SplatHash. Small enough to not distort results much, large enough to prevent numerical issues.

This step is run separately for each color channel:

- L channel: all 6 Splats participate
- a and b channels: only the 3 Baryons participate (Leptons are luma-only)

The linear system `(A^T A + λI) x = A^T b` is solved with Gaussian elimination.

### Why does this matter?

Because Splats overlap. A large Splat that covers half the image will interfere with a smaller Splat inside it. If you just keep the greedy weight estimates, they fight each other. After Ridge Regression, the weights are globally coherent — each Splat contributes exactly what's needed given all the others.

---

## Step 5: Bit Packing

The 6 Splats and the mean are packed into exactly 16 bytes (128 bits):

| Field      | Bits    | Encoding                                   |
| ---------- | ------- | ------------------------------------------ |
| Mean L     | 6       | L × 63.5, clamped to [0, 63]               |
| Mean a     | 5       | (a + 0.2) / 0.4 × 31.5, clamped to [0, 31] |
| Mean b     | 5       | (b + 0.2) / 0.4 × 31.5, clamped to [0, 31] |
| Baryon × 3 | 22 each | x(4b) y(4b) sigma(2b) L(4b) a(4b) b(4b)    |
| Lepton × 3 | 15 each | x(4b) y(4b) sigma(2b) L(5b)                |
| Padding    | 1       | zero                                       |

Total: 16 + 66 + 45 + 1 = 128 bits = 16 bytes.

**Position encoding:** x and y are stored as 4-bit integers (0–15). Divide by 15. to get back to 0..1 range.

**Sigma encoding:** 2 bits = 4 levels. These are the 4 sigma values in the SIGMA_TABLE. The nearest table entry to the actual sigma is used.

**L/a/b encoding for Baryons:** 4 bits each. L range is [-0.8, 0.8], a and b range is [-0.4, 0.4]. These ranges were chosen to cover the typical residual values after mean subtraction.

**L encoding for Leptons:** 5 bits (higher precision than Baryons since Leptons capture fine detail and have no color to help).

**Bit stream order:** Big-endian. The first field written occupies the most significant bits of the first byte. Unpacking must read in the same order.

---

## Decoding

Decoding is straightforward:

1. Unpack the 16 bytes to get mean color and 6 Splats
2. Fill a 32x32 grid with the mean color
3. For each Splat, compute its Gaussian activation at every grid pixel and add `weight × activation` to each pixel's color
4. Convert each pixel from Oklab to sRGB
5. Clamp to [0, 255] and return as RGBA

The result is a 32x32 image suitable for use as a placeholder.

---

## Why SplatHash vs. Alternatives

**BlurHash** stores cosine basis functions (like a tiny DCT). Cosines are global — a single cosine covers the entire image. This produces smooth gradients well but struggles with localized features like a bright spot in one corner. More importantly, BlurHash's smallest reasonable output is around 25 bytes (with 4 components), and it lacks any global weight optimization.

**ThumbHash** improves on BlurHash by using a similar frequency decomposition but with better color handling. Output is variable-length and typically 25–35 bytes. Its reconstruction is perceptually better than BlurHash but still global in nature.

**SplatHash** uses spatially localized basis functions (Gaussians). A Splat affects only its local neighborhood. This means a bright spot in one corner doesn't corrupt the representation of the opposite corner. The global Ridge Regression step then ensures the Splats work together optimally. The result is 16 bytes fixed — smaller than both alternatives — while retaining competitive or superior visual quality for images with distinct local features.

The fixed 16-byte output is also a practical advantage: it can be stored in a database column with no variable-length overhead, passed as a single 128-bit integer, and compared cheaply.

---

## Implementation Notes for New Languages

When porting to a new language, the following must be exact:

1. **srgbToOklab and oklabToSrgb**: Use the exact coefficients listed in the source. Floating-point differences in these transforms will produce different hashes.

2. **Area averaging in imageToOklabGrid**: The grid mapping must match exactly. The floor-based integer bounds `x0 = floor(x * srcW / w)`, `x1 = floor((x+1) * srcW / w)` must be used consistently.

3. **Quantization and dequantization**: The `quant` function rounds to nearest (`round(norm * steps)`), not truncates. The `unquant` function divides by `steps` (not `steps - 1` or `steps + 1`).

4. **packMean**: L uses `floor(l * 63.5)`, a and b use `floor(((v + 0.2) / 0.4) * 31.5)`.

5. **Bit stream**: Write and read MSB-first within each byte. The Go/TS reference implementations include `BitWriter`/`BitReader` structs that must be replicated precisely.

6. **Sigma lookup**: When encoding, find the nearest sigma table entry by absolute distance and use its index.

7. **Greedy search stride**: The position search uses stride 2, scanning at `x = 0, 2, 4, ...` and `y = 0, 2, 4, ...`. This must match or hashes will differ.

Test your implementation against the shared `assets/` test images by comparing hex-encoded hashes with the Go reference implementation.
