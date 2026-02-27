// SplatHash — TypeScript implementation
//
// Encodes any image into 16 bytes and reconstructs a 32x32 preview.
// An image is decomposed into a background color (Mean) and six Gaussian blobs (Splats):
//   - 3 Baryons: full-color Splats for dominant features
//   - 3 Leptons: luma-only Splats for texture and detail
//
// Splat positions are found by separable 2-D Gaussian correlation (matching pursuit).
// Ridge Regression then refines all weights together. All computation is in Oklab.
// The hash fits into exactly 128 bits.

export const TARGET_SIZE = 32;
export const RIDGE_LAMBDA = 0.001;
export const SIGMA_TABLE = [0.025, 0.1, 0.2, 0.35];

const GAUSS_TABLE_MAX = 1923; // max dsq = 31² + 31² = 1922 for a 32×32 grid

export interface Splat {
  x: number;
  y: number;
  sigma: number;
  l: number;
  a: number;
  b: number;
  isLepton: boolean;
}

export interface DecodedImage {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

// ── Package-level precomputed look-up tables ──────────────────────────────────

// gaussLUT[si][dsq] = exp(-dsq / (2·σᵢ²·W²))
const gaussLUT: Float64Array[] = [
  new Float64Array(GAUSS_TABLE_MAX),
  new Float64Array(GAUSS_TABLE_MAX),
  new Float64Array(GAUSS_TABLE_MAX),
  new Float64Array(GAUSS_TABLE_MAX),
];

const gaussKernel1D: Float64Array[] = new Array(4);
const kernelHW: number[] = [0, 0, 0, 0];
// gaussPow[si] = (Σ_{d=-hw}^{hw} k[d]²)² ≈ Σ_{dx,dy} G(dx,dy)² (interior)
const gaussPow: number[] = [0, 0, 0, 0];

// linToSrgbLUT[i] = sRGB-gamma(i / 1023) for i = 0 .. 1023.
const linToSrgbLUT = new Float64Array(1024);

// srgbLinLUT[v] = linear-light(v / 255) for v = 0 .. 255.
const srgbLinLUT = new Float64Array(256);

// cbrtLUT[i] = cbrt(i / 1024) for i = 0 .. 1024.
const cbrtLUT = new Float64Array(1025);

(function initLUTs() {
  const W = TARGET_SIZE;
  const W2 = W * W;

  for (let si = 0; si < 4; si++) {
    const sigma = SIGMA_TABLE[si];
    const scale2 = 2.0 * sigma * sigma * W2;
    for (let dsq = 0; dsq < GAUSS_TABLE_MAX; dsq++) {
      let v = Math.exp(-dsq / scale2);
      if (v < 1e-7) v = 0;
      gaussLUT[si][dsq] = v;
    }
    // Build 1-D half-kernel.
    let hw = 0;
    for (let d = 0; d < W; d++) {
      if (gaussLUT[si][d * d] < 1e-7) break;
      hw = d;
    }
    kernelHW[si] = hw;
    const kern = new Float64Array(hw + 1);
    for (let d = 0; d <= hw; d++) kern[d] = gaussLUT[si][d * d];
    gaussKernel1D[si] = kern;
    // Normalization factor gg = (Σ_d k[d]²)²
    let sum1D = 0.0;
    for (let d = -hw; d <= hw; d++) {
      const v = kern[Math.abs(d)];
      sum1D += v * v;
    }
    gaussPow[si] = sum1D * sum1D;
  }

  // sRGB → linear LUT (8-bit input, full-range).
  for (let v = 0; v < 256; v++) {
    const c = v / 255.0;
    srgbLinLUT[v] =
      c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }

  // linear → sRGB gamma LUT (1024 steps over [0, 1]).
  for (let i = 0; i < 1024; i++) {
    const c = i / 1023.0;
    linToSrgbLUT[i] = linToSrgbScalar(c);
  }

  // Cube-root LUT for [0, 1].
  for (let i = 0; i <= 1024; i++) {
    cbrtLUT[i] = Math.cbrt(i / 1024.0);
  }
})();

function linToSrgbScalar(c: number): number {
  if (c <= 0.0031308) return 12.92 * c;
  if (c < 0) return 0;
  return 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
}

function cbrtFast(x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return cbrtLUT[1024];
  return cbrtLUT[Math.round(x * 1024.0)];
}

function linToSrgbFast(c: number): number {
  if (c <= 0) return 0;
  if (c >= 1) return 1;
  return linToSrgbLUT[Math.round(c * 1023.0)];
}

function sigmaIndex(sigma: number): number {
  let si = 0;
  let minD = Math.abs(SIGMA_TABLE[0] - sigma);
  for (let i = 1; i < 4; i++) {
    const d = Math.abs(SIGMA_TABLE[i] - sigma);
    if (d < minD) {
      minD = d;
      si = i;
    }
  }
  return si;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Encodes an RGBA image buffer into a 16-byte SplatHash.
 * @param rgba Raw RGBA data (Uint8ClampedArray or Uint8Array)
 * @param width Image width
 * @param height Image height
 * @returns 16-byte Uint8Array hash
 */
export function encode(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  // 1. Preprocess → Oklab 32×32 grid (point-sampled, LUT-accelerated).
  const grid = imageToOklabGrid(rgba, width, height, TARGET_SIZE, TARGET_SIZE);

  // 2. Compute mean and quantise immediately.
  let meanL = 0,
    meanA = 0,
    meanB = 0;
  const n = TARGET_SIZE * TARGET_SIZE;
  for (let i = 0; i < n; i++) {
    meanL += grid[i * 3];
    meanA += grid[i * 3 + 1];
    meanB += grid[i * 3 + 2];
  }
  meanL /= n;
  meanA /= n;
  meanB /= n;
  const pMean = packMean(meanL, meanA, meanB);
  const uMean = unpackMean(pMean);
  meanL = uMean.l;
  meanA = uMean.a;
  meanB = uMean.b;

  // 3. Initial residuals.
  const resL = new Float64Array(n);
  const resA = new Float64Array(n);
  const resB = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    resL[i] = grid[i * 3] - meanL;
    resA[i] = grid[i * 3 + 1] - meanA;
    resB[i] = grid[i * 3 + 2] - meanB;
  }

  // 4. Greedy matching pursuit using separable Gaussian correlation.
  const scratch = new SearchScratch();
  let basis = findAllSplats(
    resL,
    resA,
    resB,
    TARGET_SIZE,
    TARGET_SIZE,
    scratch,
    6,
  );

  // 5. Global Ridge Regression over all splat weights simultaneously.
  if (basis.length > 0) {
    basis = solveV4Weights(
      basis,
      grid,
      meanL,
      meanA,
      meanB,
      TARGET_SIZE,
      TARGET_SIZE,
    );
  }

  // 6. Bit-pack into 16 bytes.
  return packV4(pMean, basis);
}

/**
 * Decodes a 16-byte SplatHash back into RGBA pixel data.
 * @param hash 16-byte Uint8Array
 * @returns DecodedImage object containing splats and reconstructed buffer (32x32)
 */
export function decode(hash: Uint8Array): DecodedImage {
  if (hash.length !== 16)
    throw new Error("Invalid SplatHash: Must be 16 bytes.");

  const { meanL, meanA, meanB, splats } = unpackV4(hash);
  const w = 32;
  const h = 32;
  const grid = new Float64Array(w * h * 3);

  // Fill background
  for (let i = 0; i < grid.length; i += 3) {
    grid[i] = meanL;
    grid[i + 1] = meanA;
    grid[i + 2] = meanB;
  }

  // Add splats
  for (const s of splats) {
    addSplatToGrid(grid, s, w, h);
  }

  // Convert to RGBA
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 3;
      const {
        r,
        g,
        b: bl,
      } = oklabToSrgb(grid[idx], grid[idx + 1], grid[idx + 2]);
      const pIdx = (y * w + x) * 4;
      rgba[pIdx] = clampi(Math.round(r * 255 + 0.5), 0, 255);
      rgba[pIdx + 1] = clampi(Math.round(g * 255 + 0.5), 0, 255);
      rgba[pIdx + 2] = clampi(Math.round(bl * 255 + 0.5), 0, 255);
      rgba[pIdx + 3] = 255;
    }
  }

  return { width: w, height: h, rgba };
}

// ── Solver math ───────────────────────────────────────────────────────────────

function solveV4Weights(
  basis: Splat[],
  grid: Float64Array,
  meanL: number,
  meanA: number,
  meanB: number,
  w: number,
  h: number,
): Splat[] {
  const nTotal = basis.length;
  const M = w * h;

  // Reconstruct full target vectors (grid − mean).
  const tL = new Float64Array(M);
  const tA = new Float64Array(M);
  const tB = new Float64Array(M);
  for (let i = 0; i < M; i++) {
    tL[i] = grid[i * 3] - meanL;
    tA[i] = grid[i * 3 + 1] - meanA;
    tB[i] = grid[i * 3 + 2] - meanB;
  }

  // Precompute activation maps.
  const activations = basis.map((s) => computeBasisMap(s, w, h));

  // Solve L for all splats. A and B only for Baryons (first up-to-3).
  const nBaryons = Math.min(nTotal, 3);
  const xL = solveChannel(activations, tL, nTotal, RIDGE_LAMBDA);
  const xA = solveChannel(
    activations.slice(0, nBaryons),
    tA,
    nBaryons,
    RIDGE_LAMBDA,
  );
  const xB = solveChannel(
    activations.slice(0, nBaryons),
    tB,
    nBaryons,
    RIDGE_LAMBDA,
  );

  const out: Splat[] = [];
  for (let i = 0; i < nTotal; i++) {
    out.push({
      ...basis[i],
      l: xL[i],
      a: i < 3 ? xA[i] : 0,
      b: i < 3 ? xB[i] : 0,
    });
  }
  return out;
}

function solveChannel(
  activations: Float64Array[],
  target: Float64Array,
  n: number,
  lambda: number,
): Float64Array {
  if (n === 0) return new Float64Array(0);
  const m = target.length;
  const ata = new Float64Array(n * n);
  const atb = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const rowI = activations[i];
    for (let j = i; j < n; j++) {
      let sum = 0.0;
      const rowJ = activations[j];
      for (let p = 0; p < m; p++) sum += rowI[p] * rowJ[p];
      ata[i * n + j] = sum;
      ata[j * n + i] = sum;
    }
    let sumB = 0.0;
    for (let p = 0; p < m; p++) sumB += rowI[p] * target[p];
    atb[i] = sumB;
  }

  for (let i = 0; i < n; i++) ata[i * n + i] += lambda;

  return solveLinearSystem(ata, atb, n);
}

function solveLinearSystem(
  mat: Float64Array,
  vec: Float64Array,
  n: number,
): Float64Array {
  const a = new Float64Array(mat);
  const b = new Float64Array(vec);

  for (let k = 0; k < n - 1; k++) {
    for (let i = k + 1; i < n; i++) {
      const factor = a[i * n + k] / a[k * n + k];
      for (let j = k; j < n; j++) a[i * n + j] -= factor * a[k * n + j];
      b[i] -= factor * b[k];
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0.0;
    for (let j = i + 1; j < n; j++) sum += a[i * n + j] * x[j];
    x[i] = (b[i] - sum) / a[i * n + i];
  }
  return x;
}

function computeBasisMap(s: Splat, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  const si = sigmaIndex(s.sigma);
  const hw = kernelHW[si];
  const cx = Math.floor(s.x * w);
  const cy = Math.floor(s.y * h);
  const y0 = clampi(cy - hw, 0, h - 1);
  const y1 = clampi(cy + hw, 0, h - 1);
  const x0 = clampi(cx - hw, 0, w - 1);
  const x1 = clampi(cx + hw, 0, w - 1);
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    const rowBase = y * w;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const dsq = dx * dx + dy * dy;
      if (dsq < GAUSS_TABLE_MAX) {
        out[rowBase + x] = gaussLUT[si][dsq];
      }
    }
  }
  return out;
}

// ── Greedy search: sequential matching pursuit ────────────────────────────────

class SearchScratch {
  readonly tmpL = new Float64Array(TARGET_SIZE * TARGET_SIZE);
  readonly tmpA = new Float64Array(TARGET_SIZE * TARGET_SIZE);
  readonly tmpB = new Float64Array(TARGET_SIZE * TARGET_SIZE);
  readonly scoreMap = new Float64Array(TARGET_SIZE * TARGET_SIZE);
  readonly sigmaMap = new Int8Array(TARGET_SIZE * TARGET_SIZE);
}

function findAllSplats(
  resL: Float64Array,
  resA: Float64Array,
  resB: Float64Array,
  w: number,
  h: number,
  sc: SearchScratch,
  nSplats: number,
): Splat[] {
  const splats: Splat[] = [];

  while (splats.length < nSplats) {
    const isBaryon = splats.length < 3;

    // ── 1. Build per-pixel score map over all 4 sigmas ──────────────────────
    sc.scoreMap.fill(-1);
    sc.sigmaMap.fill(-1);

    for (let si = 0; si < 4; si++) {
      const kern = gaussKernel1D[si];
      const hw = kernelHW[si];
      const invGG = 1.0 / gaussPow[si];

      // ── Horizontal pass (zero-padding) ───────────────────────────────────
      for (let y = 0; y < h; y++) {
        const rowOff = y * w;
        for (let x = 0; x < w; x++) {
          let sL = kern[0] * resL[rowOff + x];
          let sA = kern[0] * resA[rowOff + x];
          let sB = kern[0] * resB[rowOff + x];
          for (let d = 1; d <= hw; d++) {
            const k = kern[d];
            const xl = x - d;
            if (xl >= 0) {
              sL += k * resL[rowOff + xl];
              if (isBaryon) {
                sA += k * resA[rowOff + xl];
                sB += k * resB[rowOff + xl];
              }
            }
            const xr = x + d;
            if (xr < w) {
              sL += k * resL[rowOff + xr];
              if (isBaryon) {
                sA += k * resA[rowOff + xr];
                sB += k * resB[rowOff + xr];
              }
            }
          }
          sc.tmpL[rowOff + x] = sL;
          if (isBaryon) {
            sc.tmpA[rowOff + x] = sA;
            sc.tmpB[rowOff + x] = sB;
          }
        }
      }

      // ── Vertical pass (zero-padding) + score update ──────────────────────
      for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
          let sL = kern[0] * sc.tmpL[y * w + x];
          let sA = kern[0] * sc.tmpA[y * w + x];
          let sB = kern[0] * sc.tmpB[y * w + x];
          for (let d = 1; d <= hw; d++) {
            const k = kern[d];
            const yu = y - d;
            if (yu >= 0) {
              sL += k * sc.tmpL[yu * w + x];
              if (isBaryon) {
                sA += k * sc.tmpA[yu * w + x];
                sB += k * sc.tmpB[yu * w + x];
              }
            }
            const yd = y + d;
            if (yd < h) {
              sL += k * sc.tmpL[yd * w + x];
              if (isBaryon) {
                sA += k * sc.tmpA[yd * w + x];
                sB += k * sc.tmpB[yd * w + x];
              }
            }
          }
          const i = y * w + x;
          const score = isBaryon
            ? (sL * sL + sA * sA + sB * sB) * invGG
            : sL * sL * invGG;
          if (score > sc.scoreMap[i]) {
            sc.scoreMap[i] = score;
            sc.sigmaMap[i] = si;
          }
        }
      }
    }

    // ── 2. Find single best pixel ────────────────────────────────────────────
    let bestScore = -1.0;
    let bestIdx = -1;
    for (let i = 0; i < w * h; i++) {
      if (sc.scoreMap[i] > bestScore) {
        bestScore = sc.scoreMap[i];
        bestIdx = i;
      }
    }
    if (bestIdx < 0 || bestScore < 1e-9) break;

    const bx = bestIdx % w;
    const by = Math.floor(bestIdx / w);
    const si = sc.sigmaMap[bestIdx];
    const kern = gaussKernel1D[si];
    const hw = kernelHW[si];
    const gg = gaussPow[si];

    // ── 3. Compute L, A, B dot-products at winner (zero-padding) ────────────
    let dotL = 0,
      dotA = 0,
      dotB = 0;
    for (let dy = -hw; dy <= hw; dy++) {
      const yy = by + dy;
      if (yy < 0 || yy >= h) continue;
      const ky = kern[Math.abs(dy)];
      for (let dx = -hw; dx <= hw; dx++) {
        const xx = bx + dx;
        if (xx < 0 || xx >= w) continue;
        const kv = ky * kern[Math.abs(dx)];
        const off = yy * w + xx;
        dotL += kv * resL[off];
        dotA += kv * resA[off];
        dotB += kv * resB[off];
      }
    }
    const invGG = 1.0 / gg;
    const splat: Splat = {
      x: bx / w,
      y: by / h,
      sigma: SIGMA_TABLE[si],
      l: dotL * invGG,
      a: dotA * invGG,
      b: dotB * invGG,
      isLepton: !isBaryon,
    };
    splats.push(splat);

    // ── 4. Subtract splat footprint from residuals ───────────────────────────
    const y0 = clampi(by - hw, 0, h - 1);
    const y1 = clampi(by + hw, 0, h - 1);
    const x0 = clampi(bx - hw, 0, w - 1);
    const x1 = clampi(bx + hw, 0, w - 1);
    for (let y = y0; y <= y1; y++) {
      const dy = y - by;
      const rowBase = y * w;
      for (let x = x0; x <= x1; x++) {
        const dx = x - bx;
        const dsq = dx * dx + dy * dy;
        if (dsq >= GAUSS_TABLE_MAX) continue;
        const wVal = gaussLUT[si][dsq];
        if (wVal === 0) continue;
        const off = rowBase + x;
        resL[off] -= splat.l * wVal;
        resA[off] -= splat.a * wVal;
        resB[off] -= splat.b * wVal;
      }
    }
  }
  return splats;
}

// ── Bit packing ───────────────────────────────────────────────────────────────

// packV4 encodes the hash into 128 bits:
//   Mean      : 16 bits  (6L + 5A + 5B)
//   3 Baryons : 22 bits each = 66 bits  (x4 y4 σ2 L4 A4 B4 — full colour)
//   3 Leptons : 15 bits each = 45 bits  (x4 y4 σ2 L5          — luma-only)
//   Reserved  :  1 bit
//   Total     : 128 bits
function packV4(mean: number, splats: Splat[]): Uint8Array {
  const bw = new BitStream();
  bw.write(mean, 16);

  // 3 Baryon splats (full-colour) — 22 bits each: x4 y4 σ2 L4 A4 B4.
  let count = 0;
  for (const s of splats) {
    if (s.isLepton) continue;
    if (count >= 3) break;
    const xi = clampi(Math.floor(s.x * 15.0 + 0.5), 0, 15);
    const yi = clampi(Math.floor(s.y * 15.0 + 0.5), 0, 15);
    const sigI = sigmaIndex(s.sigma);
    const lQ = quant(s.l, -0.8, 0.8, 4);
    const aQ = quant(s.a, -0.4, 0.4, 4);
    const bQ = quant(s.b, -0.4, 0.4, 4);
    bw.write(xi, 4);
    bw.write(yi, 4);
    bw.write(sigI, 2);
    bw.write(lQ, 4);
    bw.write(aQ, 4);
    bw.write(bQ, 4);
    count++;
  }
  while (count < 3) {
    bw.write(0, 22);
    count++;
  }

  // 3 Lepton splats (luma-only) — 15 bits each: x4 y4 σ2 L5.
  count = 0;
  for (const s of splats) {
    if (!s.isLepton) continue;
    if (count >= 3) break;
    const xi = clampi(Math.floor(s.x * 15.0 + 0.5), 0, 15);
    const yi = clampi(Math.floor(s.y * 15.0 + 0.5), 0, 15);
    const sigI = sigmaIndex(s.sigma);
    const lQ = quant(s.l, -0.8, 0.8, 5);
    bw.write(xi, 4);
    bw.write(yi, 4);
    bw.write(sigI, 2);
    bw.write(lQ, 5);
    count++;
  }
  while (count < 3) {
    bw.write(0, 15);
    count++;
  }

  bw.write(0, 1); // reserved
  return bw.getBytes();
}

function unpackV4(hash: Uint8Array) {
  const br = new BitReader(hash);
  const meanMap = br.read(16);
  const { l, a, b } = unpackMean(meanMap);

  const splats: Splat[] = [];

  // 3 Baryon splats — 22 bits each (x4 y4 σ2 L4 A4 B4).
  for (let i = 0; i < 3; i++) {
    const xi = br.read(4);
    const yi = br.read(4);
    const sigI = br.read(2);
    const lQ = br.read(4);
    const aQ = br.read(4);
    const bQ = br.read(4);
    if (xi === 0 && yi === 0 && lQ === 0 && aQ === 0 && bQ === 0) continue;
    splats.push({
      x: xi / 15.0,
      y: yi / 15.0,
      sigma: SIGMA_TABLE[sigI],
      l: unquant(lQ, -0.8, 0.8, 4),
      a: unquant(aQ, -0.4, 0.4, 4),
      b: unquant(bQ, -0.4, 0.4, 4),
      isLepton: false,
    });
  }

  // 3 Lepton splats — 15 bits each (x4 y4 σ2 L5), luma-only.
  for (let i = 0; i < 3; i++) {
    const xi = br.read(4);
    const yi = br.read(4);
    const sigI = br.read(2);
    const lQ = br.read(5);
    if (xi === 0 && yi === 0 && lQ === 0) continue;
    splats.push({
      x: xi / 15.0,
      y: yi / 15.0,
      sigma: SIGMA_TABLE[sigI],
      l: unquant(lQ, -0.8, 0.8, 5),
      a: 0,
      b: 0,
      isLepton: true,
    });
  }

  return { meanL: l, meanA: a, meanB: b, splats };
}

// ── Bit-stream helpers ────────────────────────────────────────────────────────

class BitStream {
  private buf: number[] = [];
  private acc: bigint = 0n;
  private n = 0;

  write(val: number, bits: number) {
    this.acc = (this.acc << BigInt(bits)) | BigInt(val);
    this.n += bits;
    while (this.n >= 8) {
      const shift = BigInt(this.n - 8);
      const byteVal = Number((this.acc >> shift) & 0xffn);
      this.buf.push(byteVal);
      this.n -= 8;
    }
  }

  getBytes() {
    if (this.n > 0) {
      this.buf.push(Number((this.acc << BigInt(8 - this.n)) & 0xffn));
    }
    return new Uint8Array(this.buf);
  }
}

class BitReader {
  private data: Uint8Array;
  private pos = 0;
  private rem = 0;
  private curr = 0;

  constructor(data: Uint8Array) {
    this.data = data;
  }

  read(bits: number): number {
    let val = 0;
    let bitsRemaining = bits;
    while (bitsRemaining > 0) {
      if (this.rem === 0) {
        if (this.pos >= this.data.length) return val << bitsRemaining;
        this.curr = this.data[this.pos++];
        this.rem = 8;
      }
      const take = Math.min(this.rem, bitsRemaining);
      const shift = this.rem - take;
      const mask = (1 << take) - 1;
      const chunk = (this.curr >> shift) & mask;
      val = (val << take) | chunk;
      this.rem -= take;
      bitsRemaining -= take;
    }
    return val;
  }
}

// ── Quantization ──────────────────────────────────────────────────────────────

function packMean(l: number, a: number, b: number): number {
  const li = clampi(Math.floor(l * 63.5), 0, 63);
  const ai = clampi(Math.floor(((a + 0.2) / 0.4) * 31.5), 0, 31);
  const bi = clampi(Math.floor(((b + 0.2) / 0.4) * 31.5), 0, 31);
  return (li << 10) | (ai << 5) | bi;
}

function unpackMean(p: number) {
  const li = (p >> 10) & 0x3f;
  const ai = (p >> 5) & 0x1f;
  const bi = p & 0x1f;
  return {
    l: li / 63.0,
    a: (ai / 31.0) * 0.4 - 0.2,
    b: (bi / 31.0) * 0.4 - 0.2,
  };
}

function quant(v: number, min: number, max: number, bits: number): number {
  const steps = (1 << bits) - 1;
  const norm = (v - min) / (max - min);
  return clampi(Math.floor(norm * steps + 0.5), 0, steps);
}

function unquant(v: number, min: number, max: number, bits: number): number {
  const steps = (1 << bits) - 1;
  return (v / steps) * (max - min) + min;
}

// ── Splat rendering ───────────────────────────────────────────────────────────

function addSplatToGrid(
  grid: Float64Array,
  s: Splat,
  w: number,
  h: number,
): void {
  const si = sigmaIndex(s.sigma);
  const hw = kernelHW[si];
  const cx = Math.floor(s.x * w);
  const cy = Math.floor(s.y * h);
  const y0 = clampi(cy - hw, 0, h - 1);
  const y1 = clampi(cy + hw, 0, h - 1);
  const x0 = clampi(cx - hw, 0, w - 1);
  const x1 = clampi(cx + hw, 0, w - 1);
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    const rowBase = y * w * 3;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const dsq = dx * dx + dy * dy;
      if (dsq >= GAUSS_TABLE_MAX) continue;
      const wVal = gaussLUT[si][dsq];
      if (wVal === 0) continue;
      const idx = rowBase + x * 3;
      grid[idx] += s.l * wVal;
      grid[idx + 1] += s.a * wVal;
      grid[idx + 2] += s.b * wVal;
    }
  }
}

// ── Image preprocessing ───────────────────────────────────────────────────────

function imageToOklabGrid(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  w: number,
  h: number,
): Float64Array {
  const out = new Float64Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const sy = ((y * srcH + Math.floor(srcH / 2)) / h) | 0;
    for (let x = 0; x < w; x++) {
      const sx = ((x * srcW + Math.floor(srcW / 2)) / w) | 0;
      const off = (sy * srcW + sx) * 4;
      const r = srgbLinLUT[rgba[off]];
      const g = srgbLinLUT[rgba[off + 1]];
      const b = srgbLinLUT[rgba[off + 2]];
      const { l, a, b: bb } = srgbLinToOklab(r, g, b);
      const idx = (y * w + x) * 3;
      out[idx] = l;
      out[idx + 1] = a;
      out[idx + 2] = bb;
    }
  }
  return out;
}

// ── Colour space conversions ──────────────────────────────────────────────────

function srgbLinToOklab(r: number, g: number, b: number) {
  const l1 = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m1 = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s1 = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
  const l_ = cbrtFast(l1);
  const m_ = cbrtFast(m1);
  const s_ = cbrtFast(s1);
  return {
    l: 0.2104542553 * l_ + 0.793617785 * m_ - 0.0040720468 * s_,
    a: 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_,
    b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_,
  };
}

function oklabToSrgb(l: number, a: number, b: number) {
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = l - 0.0894841775 * a - 1.291485548 * b;
  const l3 = l_ * l_ * l_;
  const m3 = m_ * m_ * m_;
  const s3 = s_ * s_ * s_;
  const r = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
  const g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
  const bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3;
  return { r: linToSrgbFast(r), g: linToSrgbFast(g), b: linToSrgbFast(bl) };
}

// ── Utility ───────────────────────────────────────────────────────────────────

function clampi(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}
