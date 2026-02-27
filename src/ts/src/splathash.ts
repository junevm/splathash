// SplatHash — TypeScript implementation
//
// Encodes any image into 16 bytes and reconstructs a 32x32 preview.
// An image is decomposed into a background color (Mean) and six Gaussian blobs (Splats):
//   - 3 Baryons: full-color Splats for dominant features
//   - 3 Leptons: luma-only Splats for texture and detail
//
// Splat positions are found greedily; Ridge Regression then refines all weights together.
// All computation is done in Oklab. The hash fits into exactly 128 bits.

export const TARGET_SIZE = 32;
export const RIDGE_LAMBDA = 0.001;
export const SIGMA_TABLE = [0.025, 0.1, 0.2, 0.35];

export interface Splat {
  x: number; // 0..1
  y: number; // 0..1
  sigma: number; // 0..1 (One of 4 discrete values)
  l: number; // Oklab L
  a: number; // Oklab a
  b: number; // Oklab b
  isLepton: boolean; // True if Luma-only
}

export interface DecodedImage {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
}

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
  // 1. Preprocess to Oklab Grid
  const grid = imageToOklabGrid(rgba, width, height, TARGET_SIZE, TARGET_SIZE);

  // 2. Compute Mean
  let meanL = 0,
    meanA = 0,
    meanB = 0;
  const n = Math.floor(grid.length / 3);
  for (let i = 0; i < n; i++) {
    meanL += grid[i * 3];
    meanA += grid[i * 3 + 1];
    meanB += grid[i * 3 + 2];
  }
  meanL /= n;
  meanA /= n;
  meanB /= n;

  // 3. Quantize Mean immediately so the solver optimizes against the reconstructed mean.
  const pMean = packMean(meanL, meanA, meanB);
  const uMean = unpackMean(pMean);
  meanL = uMean.l;
  meanA = uMean.a;
  meanB = uMean.b;

  // Residuals
  const targetL = new Float64Array(n);
  const targetA = new Float64Array(n);
  const targetB = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    targetL[i] = grid[i * 3] - meanL;
    targetA[i] = grid[i * 3 + 1] - meanA;
    targetB[i] = grid[i * 3 + 2] - meanB;
  }

  // 3. Basis Search (Greedy)
  let basis: Splat[] = [];
  const currentRecon = new Float64Array(TARGET_SIZE * TARGET_SIZE * 3);

  for (let i = 0; i < 6; i++) {
    const candidate = findBestSplat(
      grid,
      currentRecon,
      meanL,
      meanA,
      meanB,
      TARGET_SIZE,
      TARGET_SIZE,
    );
    if (candidate.score < 0.00001) break;

    const s = candidate.splat;
    s.isLepton = i >= 3; // First 3 are Baryons (Full Color), Next 3 are Leptons (Luma Only)
    basis.push(s);

    addSplatToGrid(currentRecon, s, TARGET_SIZE, TARGET_SIZE);
  }

  // 4. Global Linear Projection
  if (basis.length > 0) {
    basis = solveV4Weights(
      basis,
      targetL,
      targetA,
      targetB,
      TARGET_SIZE,
      TARGET_SIZE,
    );
  }

  // 5. Pack
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
      const l = grid[idx],
        a = grid[idx + 1],
        b = grid[idx + 2];
      const { r, g, b: bl } = oklabToSrgb(l, a, b);
      const pIdx = (y * w + x) * 4;
      rgba[pIdx] = clampi(Math.round(r * 255), 0, 255);
      rgba[pIdx + 1] = clampi(Math.round(g * 255), 0, 255);
      rgba[pIdx + 2] = clampi(Math.round(bl * 255), 0, 255);
      rgba[pIdx + 3] = 255;
    }
  }

  return { width: w, height: h, rgba };
}

// --- Implementation Details ---

function solveV4Weights(
  basis: Splat[],
  tL: Float64Array,
  tA: Float64Array,
  tB: Float64Array,
  w: number,
  h: number,
): Splat[] {
  const nTotal = basis.length;
  let nBaryons = 0;
  for (const s of basis) if (!s.isLepton) nBaryons++;

  // Precompute Activations
  const activations = basis.map((s) => computeBasisMap(s, w, h));

  // Linear Solve
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

  // Update Basis
  for (let i = 0; i < nTotal; i++) {
    basis[i].l = xL[i];
    if (i < nBaryons) {
      basis[i].a = xA[i];
      basis[i].b = xB[i];
    } else {
      basis[i].a = 0;
      basis[i].b = 0;
    }
  }
  return basis;
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

  // Build Normal Equations (ATA * x = ATb)
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let sum = 0.0;
      const actI = activations[i];
      const actJ = activations[j];
      for (let p = 0; p < m; p++) sum += actI[p] * actJ[p];
      ata[i * n + j] = sum;
      ata[j * n + i] = sum;
    }
    let sumB = 0.0;
    const actI = activations[i];
    for (let p = 0; p < m; p++) sumB += actI[p] * target[p];
    atb[i] = sumB;
  }

  // Ridge Regularization
  for (let i = 0; i < n; i++) ata[i * n + i] += lambda;

  return solveLinearSystem(ata, atb, n);
}

function solveLinearSystem(
  mat: Float64Array,
  vec: Float64Array,
  n: number,
): Float64Array {
  // Gaussian Elimination
  const a = new Float64Array(mat); // Copy
  const b = new Float64Array(vec); // Copy

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

function findBestSplat(
  grid: Float64Array,
  recon: Float64Array,
  mL: number,
  mA: number,
  mB: number,
  w: number,
  h: number,
) {
  let maxScore = -1.0;
  let bestSplat: Splat = {
    x: 0,
    y: 0,
    sigma: 0.1,
    l: 0,
    a: 0,
    b: 0,
    isLepton: false,
  };

  // Compute Residuals
  const n = Math.floor(grid.length / 3);
  const resL = new Float64Array(n);
  const resA = new Float64Array(n);
  const resB = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    resL[i] = grid[i * 3] - mL - recon[i * 3];
    resA[i] = grid[i * 3 + 1] - mA - recon[i * 3 + 1];
    resB[i] = grid[i * 3 + 2] - mB - recon[i * 3 + 2];
  }

  const step = 2; // Optimization: stride
  const sigmas = SIGMA_TABLE;

  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      const xf = x / w;
      const yf = y / h;

      for (const sigma of sigmas) {
        let dotL = 0,
          dotA = 0,
          dotB = 0,
          dotG = 0;
        const rad = Math.floor(sigma * w * 3.5);
        const y0 = clampi(y - rad, 0, h - 1);
        const y1 = clampi(y + rad, 0, h - 1);
        const x0 = clampi(x - rad, 0, w - 1);
        const x1 = clampi(x + rad, 0, w - 1);

        for (let sy = y0; sy <= y1; sy++) {
          const dy = sy / h - yf;
          const rowBase = sy * w;
          for (let sx = x0; sx <= x1; sx++) {
            const dx = sx / w - xf;
            const distSq = dx * dx + dy * dy;
            const weight = Math.exp(-distSq / (2 * sigma * sigma));

            const idx = rowBase + sx;
            dotL += weight * resL[idx];
            dotA += weight * resA[idx];
            dotB += weight * resB[idx];
            dotG += weight * weight;
          }
        }

        if (dotG < 1e-9) continue;

        // Score based on energy
        const score = (dotL * dotL + dotA * dotA + dotB * dotB) / dotG;
        if (score > maxScore) {
          maxScore = score;
          bestSplat = {
            x: xf,
            y: yf,
            sigma,
            l: dotL / dotG,
            a: dotA / dotG,
            b: dotB / dotG,
            isLepton: false,
          };
        }
      }
    }
  }
  return { splat: bestSplat, score: maxScore };
}

function computeBasisMap(s: Splat, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  const rad = Math.floor(s.sigma * w * 3.5);
  const cx = Math.floor(s.x * w);
  const cy = Math.floor(s.y * h);
  const y0 = clampi(cy - rad, 0, h - 1);
  const y1 = clampi(cy + rad, 0, h - 1);
  const x0 = clampi(cx - rad, 0, w - 1);
  const x1 = clampi(cx + rad, 0, w - 1);

  for (let y = y0; y <= y1; y++) {
    const dy = y / h - s.y;
    for (let x = x0; x <= x1; x++) {
      const dx = x / w - s.x;
      out[y * w + x] = Math.exp(-(dx * dx + dy * dy) / (2 * s.sigma * s.sigma));
    }
  }
  return out;
}

function addSplatToGrid(grid: Float64Array, s: Splat, w: number, h: number) {
  const rad = Math.floor(s.sigma * w * 3.5);
  const cx = Math.floor(s.x * w);
  const cy = Math.floor(s.y * h);
  const y0 = clampi(cy - rad, 0, h - 1);
  const y1 = clampi(cy + rad, 0, h - 1);
  const x0 = clampi(cx - rad, 0, w - 1);
  const x1 = clampi(cx + rad, 0, w - 1);

  for (let y = y0; y <= y1; y++) {
    const dy = y / h - s.y;
    for (let x = x0; x <= x1; x++) {
      const dx = x / w - s.x;
      const weight = Math.exp(-(dx * dx + dy * dy) / (2 * s.sigma * s.sigma));
      const idx = (y * w + x) * 3;
      grid[idx] += s.l * weight;
      if (!s.isLepton) {
        grid[idx + 1] += s.a * weight;
        grid[idx + 2] += s.b * weight;
      }
    }
  }
}

// --- Image Helpers ---

function imageToOklabGrid(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  w: number,
  h: number,
): Float64Array {
  const out = new Float64Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    const y0 = Math.floor((y * srcH) / h);
    const y1 = Math.ceil(((y + 1) * srcH) / h);
    for (let x = 0; x < w; x++) {
      const x0 = Math.floor((x * srcW) / w);
      const x1 = Math.ceil(((x + 1) * srcW) / w);
      let rSum = 0,
        gSum = 0,
        bSum = 0,
        count = 0;

      for (let iy = y0; iy < y1; iy++) {
        if (iy >= srcH) break;
        for (let ix = x0; ix < x1; ix++) {
          if (ix >= srcW) break;
          const idx = (iy * srcW + ix) * 4;
          rSum += rgba[idx];
          gSum += rgba[idx + 1];
          bSum += rgba[idx + 2];
          count++;
        }
      }
      if (count === 0) continue;

      const r = rSum / count / 255.0;
      const g = gSum / count / 255.0;
      const b = bSum / count / 255.0;
      const lab = srgbToOklab(r, g, b);

      const idx = (y * w + x) * 3;
      out[idx] = lab.l;
      out[idx + 1] = lab.a;
      out[idx + 2] = lab.b;
    }
  }
  return out;
}

function srgbToOklab(r: number, g: number, b: number) {
  const lin = (c: number) =>
    c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  r = lin(r);
  g = lin(g);
  b = lin(b);
  const l1 = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
  const m1 = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
  const s1 = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
  const l_ = Math.cbrt(l1),
    m_ = Math.cbrt(m1),
    s_ = Math.cbrt(s1);
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
  let r = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3;
  let g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3;
  let bl = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3;

  const srt = (c: number) =>
    c <= 0.0031308
      ? 12.92 * c
      : c < 0
        ? 0
        : 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
  return { r: srt(r), g: srt(g), b: srt(bl) };
}

// --- Bit Packing ---

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
        if (this.pos >= this.data.length) return val << bitsRemaining; // EOF
        this.curr = this.data[this.pos++];
        this.rem = 8;
      }
      // Optimization for partial read
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

function packV4(mean: number, splats: Splat[]): Uint8Array {
  const bw = new BitStream();

  // Header (16 bits)
  bw.write(mean, 16);

  // Baryons (3x)
  let count = 0;
  for (const s of splats) {
    if (s.isLepton) continue;
    if (count >= 3) break;

    const xi = clampi(Math.round(s.x * 15.0), 0, 15);
    const yi = clampi(Math.round(s.y * 15.0), 0, 15);
    const sigI = getSigmaIdx(s.sigma);
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

  // Leptons (3x)
  count = 0;
  for (const s of splats) {
    if (!s.isLepton) continue;
    if (count >= 3) break;

    const xi = clampi(Math.round(s.x * 15.0), 0, 15);
    const yi = clampi(Math.round(s.y * 15.0), 0, 15);
    const sigI = getSigmaIdx(s.sigma);
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

  bw.write(0, 1); // Pad

  return bw.getBytes();
}

function unpackV4(hash: Uint8Array) {
  const br = new BitReader(hash);
  const meanMap = br.read(16);
  const { l, a, b } = unpackMean(meanMap);

  const splats: Splat[] = [];

  // Baryons
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

  // Leptons
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

function getSigmaIdx(s: number): number {
  let minD = 100.0;
  let idx = 0;
  for (let i = 0; i < SIGMA_TABLE.length; i++) {
    const d = Math.abs(SIGMA_TABLE[i] - s);
    if (d < minD) {
      minD = d;
      idx = i;
    }
  }
  return idx;
}

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
  return clampi(Math.round(norm * steps), 0, steps);
}

function unquant(v: number, min: number, max: number, bits: number): number {
  const steps = (1 << bits) - 1;
  const norm = v / steps;
  return norm * (max - min) + min;
}

function clampi(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}
