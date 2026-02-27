// Package splathash implements SplatHash, a 16-byte perceptual image hashing algorithm.
//
// An image is decomposed into a background color (Mean) and six Gaussian blobs (Splats):
//   - 3 Baryons: full-color Splats for dominant color features
//   - 3 Leptons: luma-only Splats for texture and detail
//
// Splat positions are found by separable 2-D Gaussian correlation (matching pursuit).
// Ridge Regression then refines the color weights of all Splats simultaneously to
// minimise reconstruction error.  All computation is done in Oklab for perceptual
// uniformity.  The final hash fits into exactly 128 bits.
package splathash

import (
	"errors"
	"image"
	"image/color"
	"math"
)

// TargetSize is the internal grid size for analysis (32×32).
const (
	TargetSize    = 32
	RidgeLambda   = 0.001
	gaussTableMax = 1923 // max dsq = 31² + 31² = 1922 for a 32×32 grid
)

var ErrInvalidHash = errors.New("invalid splathash: length must be 16 bytes")

// sigmaValues are the four available Gaussian spread values (2-bit encoded in hash).
var sigmaValues = [4]float64{0.025, 0.1, 0.2, 0.35}

// ── Package-level precomputed look-up tables ────────────────────────────────

// gaussLUT[si][dsq] = exp(-dsq / (2·σᵢ²·W²))
// dsq is the squared integer-pixel distance (0 .. 1922) for a 32×32 grid.
var gaussLUT [4][gaussTableMax]float64

// Precomputed separable 1-D Gaussian kernels and their half-widths.
var (
	gaussKernel1D [4][]float64 // gaussKernel1D[si][d] = exp(-d²/(2σᵢ²W²))
	kernelHW      [4]int       // half-width of the non-zero region for each sigma
	// gaussPow[si] = (Σ_{d=-hw}^{hw} k[d]²)²  ≈  Σ_{dx,dy} G(dx,dy)² (interior)
	gaussPow [4]float64
)

// linToSrgbLUT[i] = sRGB-gamma(i / 1023) for i = 0 .. 1023.
var linToSrgbLUT [1024]float64

// srgbLinLUT[v] = linear-light(v / 255) for v = 0 .. 255.
var srgbLinLUT [256]float64

// cbrtLUT[i] = cbrt(i / 1024) for i = 0 .. 1024.
var cbrtLUT [1025]float64

func init() {
	const W = TargetSize
	const W2 = W * W

	for si, sigma := range sigmaValues {
		scale2 := 2.0 * sigma * sigma * float64(W2)
		for dsq := 0; dsq < gaussTableMax; dsq++ {
			v := math.Exp(-float64(dsq) / scale2)
			if v < 1e-7 {
				v = 0
			}
			gaussLUT[si][dsq] = v
		}
		// Build 1-D half-kernel.
		hw := 0
		for d := 0; d < W; d++ {
			if gaussLUT[si][d*d] < 1e-7 {
				break
			}
			hw = d
		}
		kernelHW[si] = hw
		kern := make([]float64, hw+1)
		for d := 0; d <= hw; d++ {
			kern[d] = gaussLUT[si][d*d]
		}
		gaussKernel1D[si] = kern
		// Normalization factor gg = (Σ_d k[d]²)²
		sum1D := 0.0
		for d := -hw; d <= hw; d++ {
			v := kern[iabs(d)]
			sum1D += v * v
		}
		gaussPow[si] = sum1D * sum1D
	}

	// sRGB → linear LUT (8-bit input, full-range).
	for v := 0; v < 256; v++ {
		c := float64(v) / 255.0
		if c <= 0.04045 {
			srgbLinLUT[v] = c / 12.92
		} else {
			srgbLinLUT[v] = math.Pow((c+0.055)/1.055, 2.4)
		}
	}

	// linear → sRGB gamma LUT (1 024 steps over [0, 1]).
	for i := 0; i < 1024; i++ {
		c := float64(i) / 1023.0
		linToSrgbLUT[i] = linToSrgbScalar(c)
	}

	// Cube-root LUT for [0, 1].
	for i := 0; i <= 1024; i++ {
		cbrtLUT[i] = math.Cbrt(float64(i) / 1024.0)
	}
}

// ── Small math helpers ───────────────────────────────────────────────────────

func iabs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func linToSrgbScalar(c float64) float64 {
	if c <= 0.0031308 {
		return 12.92 * c
	}
	if c < 0 {
		return 0
	}
	return 1.055*math.Pow(c, 1.0/2.4) - 0.055
}

// cbrtFast approximates cube-root for x ∈ [0, 1] using the precomputed LUT.
// Error ≲ 5e-4 – well below the per-channel quantisation noise of the hash.
func cbrtFast(x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return cbrtLUT[1024]
	}
	return cbrtLUT[int(x*1024.0+0.5)]
}

// linToSrgbFast applies sRGB gamma using the LUT.
func linToSrgbFast(c float64) float64 {
	if c <= 0 {
		return 0
	}
	if c >= 1 {
		return 1
	}
	return linToSrgbLUT[int(c*1023.0+0.5)]
}

func sigmaIndex(sigma float64) int {
	si, minD := 0, math.Abs(sigmaValues[0]-sigma)
	for i := 1; i < 4; i++ {
		if d := math.Abs(sigmaValues[i] - sigma); d < minD {
			minD = d
			si = i
		}
	}
	return si
}

// ── Public API ───────────────────────────────────────────────────────────────

// Splat represents a Gaussian blob positioned in the image.
// Baryons (IsLepton=false) carry full colour; Leptons carry only luminance.
type Splat struct {
	X, Y    float64
	Sigma   float64
	L, A, B float64
	IsLepton bool
}

// EncodeImage generates a 16-byte SplatHash from the provided image.
// It is thread-safe and deterministic.
func EncodeImage(img image.Image) []byte {
	if img == nil {
		return nil
	}

	// 1. Preprocess → Oklab 32×32 grid (point-sampled, LUT-accelerated).
	grid := imageToOklabGrid(img, TargetSize, TargetSize)

	// 2. Compute mean and quantise immediately so the solver optimises against
	// the reconstructed mean rather than the perfect floating-point mean.
	mean := computeMean(grid)
	packedMean := packMean(mean[0], mean[1], mean[2])
	meanL, meanA, meanB := unpackMean(packedMean)

	// 3. Initial residuals (flat per-pixel slices, not interleaved).
	N := TargetSize * TargetSize
	resL := make([]float64, N)
	resA := make([]float64, N)
	resB := make([]float64, N)
	for i := 0; i < N; i++ {
		idx := i * 3
		resL[i] = grid[idx] - meanL
		resA[i] = grid[idx+1] - meanA
		resB[i] = grid[idx+2] - meanB
	}

	// 4. Greedy matching pursuit using separable Gaussian correlation.
	// Scratch buffers preallocated once; reused across all 6 splat searches.
	scratch := newSearchScratch()
	basis := findAllSplats(resL, resA, resB, TargetSize, TargetSize, scratch, 6)

	// 5. Global Ridge Regression over all splat weights simultaneously.
	if len(basis) > 0 {
		basis = solveV4Weights(basis, resL, resA, resB, grid, meanL, meanA, meanB, TargetSize, TargetSize)
	}

	// 6. Bit-pack into 16 bytes.
	return packV4(packedMean, basis)
}

// DecodeImage reconstructs a 32×32 RGBA image from a 16-byte SplatHash.
func DecodeImage(hash []byte) (image.Image, error) {
	if len(hash) != 16 {
		return nil, ErrInvalidHash
	}

	meanL, meanA, meanB, splats := unpackV4(hash)
	w, h := 32, 32
	grid := make([]float64, w*h*3)

	// Fill with background colour.
	for i := 0; i < len(grid); i += 3 {
		grid[i] = meanL
		grid[i+1] = meanA
		grid[i+2] = meanB
	}

	// Composite splats.
	for _, s := range splats {
		addSplatToGrid(grid, s, w, h)
	}

	out := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := (y*w + x) * 3
			R, G, B := oklabToSrgb(grid[idx], grid[idx+1], grid[idx+2])
			out.SetRGBA(x, y, color.RGBA{
				R: uint8(clampi(int(R*255+0.5), 0, 255)),
				G: uint8(clampi(int(G*255+0.5), 0, 255)),
				B: uint8(clampi(int(B*255+0.5), 0, 255)),
				A: 255,
			})
		}
	}
	return out, nil
}

// ── Solver math ─────────────────────────────────────────────────────────────

// solveV4Weights re-solves for optimal L, A, B weights using ridge regression.
// Note: resL/A/B at this point reflect the residual AFTER greedy subtraction;
// we reconstruct the full targets from the grid to get correct ATb vectors.
func solveV4Weights(basis []Splat, resL, resA, resB, grid []float64,
	meanL, meanA, meanB float64, w, h int) []Splat {

	N_Total := len(basis)

	// Reconstruct full target vectors (grid − mean) from the original grid.
	M := w * h
	tL := make([]float64, M)
	tA := make([]float64, M)
	tB := make([]float64, M)
	for i := 0; i < M; i++ {
		idx := i * 3
		tL[i] = grid[idx] - meanL
		tA[i] = grid[idx+1] - meanA
		tB[i] = grid[idx+2] - meanB
	}

	// Precompute activation maps.
	activations := make([][]float64, N_Total)
	for i := 0; i < N_Total; i++ {
		activations[i] = computeBasisMap(basis[i], w, h)
	}

	// Solve L for all 6 splats (Baryons + Leptons both contribute luminance).
	// Solve A and B for Baryons only (first up-to-3); Leptons are luma-only.
	nBaryons := min(N_Total, 3)
	xL := solveChannel(activations, tL, N_Total, RidgeLambda)
	xA := solveChannel(activations[:nBaryons], tA, nBaryons, RidgeLambda)
	xB := solveChannel(activations[:nBaryons], tB, nBaryons, RidgeLambda)

	out := make([]Splat, N_Total)
	for i := 0; i < N_Total; i++ {
		out[i] = basis[i]
		out[i].L = xL[i]
		if i < 3 {
			out[i].A = xA[i]
			out[i].B = xB[i]
		} else {
			out[i].A = 0
			out[i].B = 0
		}
	}
	return out
}

func solveChannel(activations [][]float64, target []float64, N int, lambda float64) []float64 {
	if N == 0 {
		return nil
	}
	M := len(target)
	ATA := make([]float64, N*N)
	ATb := make([]float64, N)
	for i := 0; i < N; i++ {
		rowI := activations[i]
		for j := i; j < N; j++ {
			sum := 0.0
			rowJ := activations[j]
			for p := 0; p < M; p++ {
				sum += rowI[p] * rowJ[p]
			}
			ATA[i*N+j] = sum
			ATA[j*N+i] = sum
		}
		sumB := 0.0
		for p := 0; p < M; p++ {
			sumB += rowI[p] * target[p]
		}
		ATb[i] = sumB
	}
	for i := 0; i < N; i++ {
		ATA[i*N+i] += lambda
	}
	return solveLinearSystem(ATA, ATb, N)
}

// computeBasisMap computes the Gaussian activation field for a splat using the LUT.
func computeBasisMap(s Splat, w, h int) []float64 {
	out := make([]float64, w*h)
	si := sigmaIndex(s.Sigma)
	hw := kernelHW[si]
	cx := int(s.X * float64(w))
	cy := int(s.Y * float64(h))
	y0 := clampi(cy-hw, 0, h-1)
	y1 := clampi(cy+hw, 0, h-1)
	x0 := clampi(cx-hw, 0, w-1)
	x1 := clampi(cx+hw, 0, w-1)
	for y := y0; y <= y1; y++ {
		dy := y - cy
		rowBase := y * w
		for x := x0; x <= x1; x++ {
			dx := x - cx
			dsq := dx*dx + dy*dy
			if dsq < gaussTableMax {
				out[rowBase+x] = gaussLUT[si][dsq]
			}
		}
	}
	return out
}

func solveLinearSystem(A, b []float64, N int) []float64 {
	mat := make([]float64, len(A))
	copy(mat, A)
	vec := make([]float64, len(b))
	copy(vec, b)
	for k := 0; k < N-1; k++ {
		for i := k + 1; i < N; i++ {
			factor := mat[i*N+k] / mat[k*N+k]
			for j := k; j < N; j++ {
				mat[i*N+j] -= factor * mat[k*N+j]
			}
			vec[i] -= factor * vec[k]
		}
	}
	x := make([]float64, N)
	for i := N - 1; i >= 0; i-- {
		sum := 0.0
		for j := i + 1; j < N; j++ {
			sum += mat[i*N+j] * x[j]
		}
		x[i] = (vec[i] - sum) / mat[i*N+i]
	}
	return x
}

// ── Greedy search: sequential matching pursuit ──────────────────────────────

// searchScratch holds pre-allocated buffers for findAllSplats.
// Baryon iterations (first 3) compute correlation maps for all three Oklab
// channels (L, A, B) so that positions are chosen based on full perceptual
// residual. Lepton iterations (4–6) use L-only maps. ~60 KB total; fits in L2.
type searchScratch struct {
	tmpL     [TargetSize * TargetSize]float64
	tmpA     [TargetSize * TargetSize]float64
	tmpB     [TargetSize * TargetSize]float64
	corrL    [TargetSize * TargetSize]float64 // scratch for current sigma L correlation
	corrA    [TargetSize * TargetSize]float64 // scratch for current sigma A correlation (Baryon only)
	corrB    [TargetSize * TargetSize]float64 // scratch for current sigma B correlation (Baryon only)
	scoreMap [TargetSize * TargetSize]float64
	sigmaMap [TargetSize * TargetSize]int8
}

func newSearchScratch() *searchScratch { return &searchScratch{} }

// findAllSplats implements sequential matching pursuit with full perceptual
// position search for the three Baryon splats and L-only search for the three
// luma-only Leptons.
//
// For each of the 6 iterations, all four sigmas are scored via full separable
// 1-D Gaussian correlation with zero-padding (no boundary clamping). Baryons
// use the full perceptual score (corrL²+corrA²+corrB²)/‖G‖²; Leptons use
// L-only corrL²/‖G‖². The sigma and position with the highest score wins.
// The splat's Gaussian footprint is then subtracted from all residual channels
// so subsequent searches explain orthogonal variance (matching pursuit).
func findAllSplats(resL, resA, resB []float64, w, h int, sc *searchScratch, nSplats int) []Splat {
	var splats []Splat

	for len(splats) < nSplats {
		isBaryon := len(splats) < 3

		// ── 1. Build per-pixel score map over all 4 sigmas ────────────────
		for i := 0; i < w*h; i++ {
			sc.scoreMap[i] = -1
			sc.sigmaMap[i] = -1
		}

		for si := 0; si < 4; si++ {
			kern := gaussKernel1D[si]
			hw := kernelHW[si]
			invGG := 1.0 / gaussPow[si]

			// ── Horizontal pass (zero-padding) ──────────────────────────
			for y := 0; y < h; y++ {
				rowOff := y * w
				for x := 0; x < w; x++ {
					sL := kern[0] * resL[rowOff+x]
					sA := kern[0] * resA[rowOff+x]
					sB := kern[0] * resB[rowOff+x]
					for d := 1; d <= hw; d++ {
						k := kern[d]
						xl := x - d
						if xl >= 0 {
							sL += k * resL[rowOff+xl]
							if isBaryon {
								sA += k * resA[rowOff+xl]
								sB += k * resB[rowOff+xl]
							}
						}
						xr := x + d
						if xr < w {
							sL += k * resL[rowOff+xr]
							if isBaryon {
								sA += k * resA[rowOff+xr]
								sB += k * resB[rowOff+xr]
							}
						}
					}
					sc.tmpL[rowOff+x] = sL
					if isBaryon {
						sc.tmpA[rowOff+x] = sA
						sc.tmpB[rowOff+x] = sB
					}
				}
			}

			// ── Vertical pass (zero-padding) + score update ──────────────
			for x := 0; x < w; x++ {
				for y := 0; y < h; y++ {
					sL := kern[0] * sc.tmpL[y*w+x]
					sA := kern[0] * sc.tmpA[y*w+x]
					sB := kern[0] * sc.tmpB[y*w+x]
					for d := 1; d <= hw; d++ {
						k := kern[d]
						yu := y - d
						if yu >= 0 {
							sL += k * sc.tmpL[yu*w+x]
							if isBaryon {
								sA += k * sc.tmpA[yu*w+x]
								sB += k * sc.tmpB[yu*w+x]
							}
						}
						yd := y + d
						if yd < h {
							sL += k * sc.tmpL[yd*w+x]
							if isBaryon {
								sA += k * sc.tmpA[yd*w+x]
								sB += k * sc.tmpB[yd*w+x]
							}
						}
					}
					i := y*w + x
					var score float64
					if isBaryon {
						score = (sL*sL + sA*sA + sB*sB) * invGG
					} else {
						score = sL * sL * invGG
					}
					if score > sc.scoreMap[i] {
						sc.scoreMap[i] = score
						sc.sigmaMap[i] = int8(si)
					}
				}
			}
		}

		// ── 2. Find single best pixel ──────────────────────────────────────
		bestScore, bestIdx := -1.0, -1
		for i, s := range sc.scoreMap {
			if s > bestScore {
				bestScore = s
				bestIdx = i
			}
		}
		if bestIdx < 0 || bestScore < 1e-9 {
			break
		}

		bx := bestIdx % w
		by := bestIdx / w
		si := int(sc.sigmaMap[bestIdx])
		kern := gaussKernel1D[si]
		hw := kernelHW[si]
		gg := gaussPow[si]

		// ── 3. Compute L, A, B dot-products at winner (zero-padding) ──────
		dotL, dotA, dotB := 0.0, 0.0, 0.0
		for dy := -hw; dy <= hw; dy++ {
			yy := by + dy
			if yy < 0 || yy >= h {
				continue
			}
			ky := kern[iabs(dy)]
			for dx := -hw; dx <= hw; dx++ {
				xx := bx + dx
				if xx < 0 || xx >= w {
					continue
				}
				kv := ky * kern[iabs(dx)]
				off := yy*w + xx
				dotL += kv * resL[off]
				dotA += kv * resA[off]
				dotB += kv * resB[off]
			}
		}
		invGG := 1.0 / gg
		splat := Splat{
			X:        float64(bx) / float64(w),
			Y:        float64(by) / float64(h),
			Sigma:    sigmaValues[si],
			L:        dotL * invGG,
			A:        dotA * invGG,
			B:        dotB * invGG,
			IsLepton: !isBaryon,
		}
		splats = append(splats, splat)

		// ── 4. Subtract splat footprint from residuals ─────────────────────
		y0 := clampi(by-hw, 0, h-1)
		y1 := clampi(by+hw, 0, h-1)
		x0 := clampi(bx-hw, 0, w-1)
		x1 := clampi(bx+hw, 0, w-1)
		for y := y0; y <= y1; y++ {
			dy := y - by
			rowBase := y * w
			for x := x0; x <= x1; x++ {
				dx := x - bx
				dsq := dx*dx + dy*dy
				if dsq >= gaussTableMax {
					continue
				}
				wVal := gaussLUT[si][dsq]
				if wVal == 0 {
					continue
				}
				off := rowBase + x
				resL[off] -= splat.L * wVal
				resA[off] -= splat.A * wVal
				resB[off] -= splat.B * wVal
			}
		}
	}
	return splats
}

// ── Bit packing ──────────────────────────────────────────────────────────────

// packV4 encodes the hash into 128 bits:
//
//	Mean      : 16 bits  (6L + 5A + 5B)
//	3 Baryons : 22 bits each = 66 bits  (x4 y4 σ2 L4 A4 B4 — full colour)
//	3 Leptons : 15 bits each = 45 bits  (x4 y4 σ2 L5      — luma-only)
//	Reserved  :  1 bit
//	Total     : 128 bits
func packV4(mean uint16, splats []Splat) []byte {
	bw := &bitWriter{}
	bw.Write(uint64(mean), 16)

	// 3 Baryon splats (full-colour) — 22 bits each: x4 y4 σ2 L4 A4 B4.
	count := 0
	for _, s := range splats {
		if s.IsLepton {
			continue
		}
		if count >= 3 {
			break
		}
		xi := clampi(int(s.X*15.0+0.5), 0, 15)
		yi := clampi(int(s.Y*15.0+0.5), 0, 15)
		sigI := sigmaIndex(s.Sigma)
		lQ := quant(s.L, -0.8, 0.8, 4)
		aQ := quant(s.A, -0.4, 0.4, 4)
		bQ := quant(s.B, -0.4, 0.4, 4)
		bw.Write(uint64(xi), 4)
		bw.Write(uint64(yi), 4)
		bw.Write(uint64(sigI), 2)
		bw.Write(uint64(lQ), 4)
		bw.Write(uint64(aQ), 4)
		bw.Write(uint64(bQ), 4)
		count++
	}
	for count < 3 {
		bw.Write(0, 22)
		count++
	}

	// 3 Lepton splats (luma-only) — 15 bits each: x4 y4 σ2 L5.
	count = 0
	for _, s := range splats {
		if !s.IsLepton {
			continue
		}
		if count >= 3 {
			break
		}
		xi := clampi(int(s.X*15.0+0.5), 0, 15)
		yi := clampi(int(s.Y*15.0+0.5), 0, 15)
		sigI := sigmaIndex(s.Sigma)
		lQ := quant(s.L, -0.8, 0.8, 5)
		bw.Write(uint64(xi), 4)
		bw.Write(uint64(yi), 4)
		bw.Write(uint64(sigI), 2)
		bw.Write(uint64(lQ), 5)
		count++
	}
	for count < 3 {
		bw.Write(0, 15)
		count++
	}

	bw.Write(0, 1) // reserved
	return bw.Bytes()
}

func unpackV4(hash []byte) (mL, mA, mB float64, splats []Splat) {
	br := &bitReader{data: hash}
	meanMap := uint16(br.Read(16))
	mL, mA, mB = unpackMean(meanMap)

	// 3 Baryon splats — 22 bits each (x4 y4 σ2 L4 A4 B4).
	for i := 0; i < 3; i++ {
		xi := br.Read(4)
		yi := br.Read(4)
		sigI := br.Read(2)
		lQ := br.Read(4)
		aQ := br.Read(4)
		bQ := br.Read(4)
		if lQ == 0 && aQ == 0 && bQ == 0 && xi == 0 && yi == 0 {
			continue
		}
		splats = append(splats, Splat{
			X:        float64(xi) / 15.0,
			Y:        float64(yi) / 15.0,
			Sigma:    sigmaValues[sigI],
			L:        unquant(lQ, -0.8, 0.8, 4),
			A:        unquant(aQ, -0.4, 0.4, 4),
			B:        unquant(bQ, -0.4, 0.4, 4),
			IsLepton: false,
		})
	}
	// 3 Lepton splats — 15 bits each (x4 y4 σ2 L5), luma-only.
	for i := 0; i < 3; i++ {
		xi := br.Read(4)
		yi := br.Read(4)
		sigI := br.Read(2)
		lQ := br.Read(5)
		if lQ == 0 && xi == 0 && yi == 0 {
			continue
		}
		splats = append(splats, Splat{
			X:        float64(xi) / 15.0,
			Y:        float64(yi) / 15.0,
			Sigma:    sigmaValues[sigI],
			L:        unquant(lQ, -0.8, 0.8, 5),
			A:        0,
			B:        0,
			IsLepton: true,
		})
	}
	return
}

// ── Bit-stream helpers ───────────────────────────────────────────────────────

type bitWriter struct {
	buf []byte
	acc uint64
	n   uint
}

func (b *bitWriter) Write(u uint64, bits uint) {
	b.acc = (b.acc << bits) | (u & ((1 << bits) - 1))
	b.n += bits
	for b.n >= 8 {
		shift := b.n - 8
		b.buf = append(b.buf, byte(b.acc>>shift))
		b.n -= 8
	}
}

func (b *bitWriter) Bytes() []byte {
	if b.n > 0 {
		b.buf = append(b.buf, byte(b.acc<<(8-b.n)))
	}
	return b.buf
}

type bitReader struct {
	data []byte
	pos  int
	rem  uint
	curr byte
}

func (b *bitReader) Read(bits uint) uint64 {
	var val uint64
	for bits > 0 {
		if b.rem == 0 {
			if b.pos >= len(b.data) {
				return val << bits
			}
			b.curr = b.data[b.pos]
			b.pos++
			b.rem = 8
		}
		take := bits
		if b.rem < take {
			take = b.rem
		}
		shift := b.rem - take
		mask := byte((1 << take) - 1)
		chunk := (b.curr >> shift) & mask
		val = (val << take) | uint64(chunk)
		b.rem -= take
		bits -= take
	}
	return val
}

// ── Quantization ─────────────────────────────────────────────────────────────

func packMean(l, a, b float64) uint16 {
	li := clampi(int(l*63.5), 0, 63)
	ai := clampi(int(((a+0.2)/0.4)*31.5), 0, 31)
	bi := clampi(int(((b+0.2)/0.4)*31.5), 0, 31)
	return uint16((li << 10) | (ai << 5) | bi)
}

func unpackMean(p uint16) (l, a, b float64) {
	li := (p >> 10) & 0x3F
	ai := (p >> 5) & 0x1F
	bi := p & 0x1F
	l = float64(li) / 63.0
	a = (float64(ai)/31.0*0.4) - 0.2
	b = (float64(bi)/31.0*0.4) - 0.2
	return
}

func quant(v, min, max float64, bits uint) uint64 {
	steps := float64((uint64(1) << bits) - 1)
	norm := (v - min) / (max - min)
	i := int(norm*steps + 0.5)
	return uint64(clampi(i, 0, int(steps)))
}

func unquant(v uint64, min, max float64, bits uint) float64 {
	steps := float64((uint64(1) << bits) - 1)
	return (float64(v)/steps)*(max-min) + min
}

// ── Splat rendering ──────────────────────────────────────────────────────────

// addSplatToGrid adds a splat's colour contribution using the Gaussian LUT.
func addSplatToGrid(grid []float64, s Splat, w, h int) {
	si := sigmaIndex(s.Sigma)
	hw := kernelHW[si]
	cx := int(s.X * float64(w))
	cy := int(s.Y * float64(h))
	y0 := clampi(cy-hw, 0, h-1)
	y1 := clampi(cy+hw, 0, h-1)
	x0 := clampi(cx-hw, 0, w-1)
	x1 := clampi(cx+hw, 0, w-1)
	for y := y0; y <= y1; y++ {
		dy := y - cy
		rowBase := y * w * 3
		for x := x0; x <= x1; x++ {
			dx := x - cx
			dsq := dx*dx + dy*dy
			if dsq >= gaussTableMax {
				continue
			}
			wVal := gaussLUT[si][dsq]
			if wVal == 0 {
				continue
			}
			idx := rowBase + x*3
			grid[idx] += s.L * wVal
			grid[idx+1] += s.A * wVal
			grid[idx+2] += s.B * wVal
		}
	}
}

// ── Image preprocessing ──────────────────────────────────────────────────────

// imageToOklabGrid converts an image to a W×H Oklab grid using point sampling.
// Point sampling at cell centres reduces pixel reads from O(srcW·srcH) to
// O(W·H), giving a 2000× speedup for typical source images.  Quality at
// 32×32 output resolution is visually equivalent to area averaging.
func imageToOklabGrid(img image.Image, w, h int) []float64 {
	bounds := img.Bounds()
	srcW := bounds.Dx()
	srcH := bounds.Dy()
	minX := bounds.Min.X
	minY := bounds.Min.Y
	out := make([]float64, w*h*3)

	// ── *image.RGBA fast path ────────────────────────────────────────────────
	if rgba, ok := img.(*image.RGBA); ok {
		for y := 0; y < h; y++ {
			sy := minY + (y*srcH+srcH/2)/h
			for x := 0; x < w; x++ {
				sx := minX + (x*srcW+srcW/2)/w
				off := rgba.PixOffset(sx, sy)
				r := srgbLinLUT[rgba.Pix[off]]
				g := srgbLinLUT[rgba.Pix[off+1]]
				b := srgbLinLUT[rgba.Pix[off+2]]
				l, a, bb := srgbLinToOklab(r, g, b)
				idx := (y*w + x) * 3
				out[idx] = l
				out[idx+1] = a
				out[idx+2] = bb
			}
		}
		return out
	}

	// ── *image.NRGBA fast path ───────────────────────────────────────────────
	if nrgba, ok := img.(*image.NRGBA); ok {
		for y := 0; y < h; y++ {
			sy := minY + (y*srcH+srcH/2)/h
			for x := 0; x < w; x++ {
				sx := minX + (x*srcW+srcW/2)/w
				off := nrgba.PixOffset(sx, sy)
				r := srgbLinLUT[nrgba.Pix[off]]
				g := srgbLinLUT[nrgba.Pix[off+1]]
				b := srgbLinLUT[nrgba.Pix[off+2]]
				l, a, bb := srgbLinToOklab(r, g, b)
				idx := (y*w + x) * 3
				out[idx] = l
				out[idx+1] = a
				out[idx+2] = bb
			}
		}
		return out
	}

	// ── *image.YCbCr fast path (JPEG decoded by standard library) ────────────
	if ycbcr, ok := img.(*image.YCbCr); ok {
		minXy := ycbcr.Rect.Min.X
		minYy := ycbcr.Rect.Min.Y
		for y := 0; y < h; y++ {
			sy := minY + (y*srcH+srcH/2)/h
			syi := sy - minYy
			yiBase := syi * ycbcr.YStride
			for x := 0; x < w; x++ {
				sx := minX + (x*srcW+srcW/2)/w
				sxi := sx - minXy
				ci := ycbcr.COffset(sx, sy)
				yy := int(ycbcr.Y[yiBase+sxi])
				cb := int(ycbcr.Cb[ci]) - 128
				cr := int(ycbcr.Cr[ci]) - 128
				// BT.601 full-range (JPEG) integer approximation:
				r8 := clampByte((yy*256 + cr*359 + 128) >> 8)
				g8 := clampByte((yy*256 - cb*88 - cr*183 + 128) >> 8)
				b8 := clampByte((yy*256 + cb*454 + 128) >> 8)
				r := srgbLinLUT[r8]
				g := srgbLinLUT[g8]
				b := srgbLinLUT[b8]
				l, a, bb := srgbLinToOklab(r, g, b)
				idx := (y*w + x) * 3
				out[idx] = l
				out[idx+1] = a
				out[idx+2] = bb
			}
		}
		return out
	}

	// ── *image.Gray fast path ────────────────────────────────────────────────
	if gray, ok := img.(*image.Gray); ok {
		for y := 0; y < h; y++ {
			sy := minY + (y*srcH+srcH/2)/h
			for x := 0; x < w; x++ {
				sx := minX + (x*srcW+srcW/2)/w
				off := gray.PixOffset(sx, sy)
				lLin := srgbLinLUT[gray.Pix[off]]
				// Achromatic: a = b = 0 in Oklab.
				l, a, bb := srgbLinToOklab(lLin, lLin, lLin)
				idx := (y*w + x) * 3
				out[idx] = l
				out[idx+1] = a
				out[idx+2] = bb
			}
		}
		return out
	}

	// ── Generic fallback (point-sampled, no area average) ────────────────────
	for y := 0; y < h; y++ {
		sy := minY + (y*srcH+srcH/2)/h
		for x := 0; x < w; x++ {
			sx := minX + (x*srcW+srcW/2)/w
			rc, gc, bc, _ := img.At(sx, sy).RGBA() // 16-bit
			r := srgbLinLUT[rc>>8]
			g := srgbLinLUT[gc>>8]
			b := srgbLinLUT[bc>>8]
			l, a, bb := srgbLinToOklab(r, g, b)
			idx := (y*w + x) * 3
			out[idx] = l
			out[idx+1] = a
			out[idx+2] = bb
		}
	}
	return out
}

func computeMean(grid []float64) [3]float64 {
	var l, a, b float64
	n := float64(len(grid) / 3)
	for i := 0; i < len(grid); i += 3 {
		l += grid[i]
		a += grid[i+1]
		b += grid[i+2]
	}
	return [3]float64{l / n, a / n, b / n}
}

// ── Colour space conversions ─────────────────────────────────────────────────

// srgbLinToOklab converts linear-light (NOT gamma-encoded) sRGB → Oklab.
// Uses the cbrtFast LUT to avoid three math.Cbrt() calls per pixel.
func srgbLinToOklab(r, g, b float64) (l, a, bb float64) {
	l1 := 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
	m1 := 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
	s1 := 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
	l_ := cbrtFast(l1)
	m_ := cbrtFast(m1)
	s_ := cbrtFast(s1)
	l = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
	a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
	bb = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
	return
}

// srgbToOklab converts gamma-encoded 8-bit-normalised sRGB values to Oklab.
// Used by the generic fallback path inside imageToOklabGrid.
func srgbToOklab(r, g, b float64) (l, a, bb float64) {
	lin := func(c float64) float64 {
		if c <= 0.04045 {
			return c / 12.92
		}
		return math.Pow((c+0.055)/1.055, 2.4)
	}
	return srgbLinToOklab(lin(r), lin(g), lin(b))
}

// oklabToSrgb converts Oklab → gamma-encoded sRGB using the LUT for the
// power-law gamma function, eliminating three math.Pow() calls per pixel.
func oklabToSrgb(l, a, b float64) (r, g, bl float64) {
	l_ := l + 0.3963377774*a + 0.2158037573*b
	m_ := l - 0.1055613458*a - 0.0638541728*b
	s_ := l - 0.0894841775*a - 1.2914855480*b
	l_ = l_ * l_ * l_
	m_ = m_ * m_ * m_
	s_ = s_ * s_ * s_
	r = +4.0767416621*l_ - 3.3077115913*m_ + 0.2309699292*s_
	g = -1.2684380046*l_ + 2.6097574011*m_ - 0.3413193965*s_
	bl = -0.0041960863*l_ - 0.7034186147*m_ + 1.7076147010*s_
	return linToSrgbFast(r), linToSrgbFast(g), linToSrgbFast(bl)
}

// ── Utility ───────────────────────────────────────────────────────────────────

func clampi(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func clampByte(v int) byte {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return byte(v)
}
