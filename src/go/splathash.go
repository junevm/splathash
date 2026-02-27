// Package splathash implements SplatHash, a 16-byte perceptual image hashing algorithm.
//
// An image is decomposed into a background color (Mean) and six Gaussian blobs (Splats):
//   - 3 Baryons: full-color Splats for dominant color features
//   - 3 Leptons: luma-only Splats for texture and detail
//
// Splat positions are found by a greedy search. Ridge Regression then refines the color
// weights of all Splats simultaneously to minimize reconstruction error. All computation
// is done in Oklab for perceptual uniformity. The final hash fits into exactly 128 bits.
package splathash

import (
	"errors"
	"image"
	"image/color"
	"math"
)

// TargetSize is the internal grid size for analysis (32x32).
const (
	TargetSize  = 32
	RidgeLambda = 0.001
)

var ErrInvalidHash = errors.New("invalid splathash: length must be 16 bytes")

// Splat represents a Gaussian blob positioned in the image.
// Baryons (IsLepton=false) carry full color; Leptons (IsLepton=true) carry only luminance.
type Splat struct {
	X, Y     float64 // Normalized Position (0..1)
	Sigma    float64 // Spread/Radius (0..1)
	L, A, B  float64 // Oklab Color Space Weights
	IsLepton bool    // If true, A and B are ignored (effectively 0)
}

// EncodeImage generates a 16-byte SplatHash from the provided image.
// It is thread-safe and deterministic.
func EncodeImage(img image.Image) []byte {
	if img == nil {
		return nil
	}

	// 1. Preprocess to Oklab Grid
	grid := imageToOklabGrid(img, TargetSize, TargetSize)
	
	// 2. Compute Mean
	mean := computeMean(grid)
	
	// 3. Quantize the Mean immediately so the solver optimizes against the
	// reconstructed mean rather than the perfect floating-point mean.
	packedMean := packMean(mean[0], mean[1], mean[2])
	meanL, meanA, meanB := unpackMean(packedMean)
	
	// Subtract reconstructed mean
	targetL := make([]float64, len(grid)/3)
	targetA := make([]float64, len(grid)/3)
	targetB := make([]float64, len(grid)/3)
	for i := 0; i < len(grid)/3; i++ {
		targetL[i] = grid[i*3] - meanL
		targetA[i] = grid[i*3+1] - meanA
		targetB[i] = grid[i*3+2] - meanB
	}
	
	// 4. Basis Search (Greedy Matching Pursuit)
	// Find 6 Splat locations: the first 3 become Baryons (full color),
	// the next 3 become Leptons (luma only).
	
	var basis []Splat
	currentRecon := make([]float64, len(grid)) 
	
	// Reduced sigma set (2 bits = 4 levels)
	sigmas := []float64{0.025, 0.1, 0.2, 0.35} 
	
	for i := 0; i < 6; i++ {
		candidate, gain := findBestSplat(grid, currentRecon, meanL, meanA, meanB, TargetSize, TargetSize, sigmas)
		if gain < 0.00001 { break }
		
		// Assign Type
		if i < 3 {
			candidate.IsLepton = false // Baryon
		} else {
			candidate.IsLepton = true  // Lepton
		}
		
		basis = append(basis, candidate)
		addSplatToGrid(currentRecon, candidate, TargetSize, TargetSize)
	}
	
	// 5. Global Linear Projection (Ridge Regression)
	// Solve for optimal L weights across all 6 splats,
	// and optimal A/B weights across the 3 Baryons.
	
	if len(basis) > 0 {
		basis = solveV4Weights(basis, targetL, targetA, targetB, TargetSize, TargetSize)
	}
	
	// 5. Pack
	return packV4(packedMean, basis)
}

func DecodeImage(hash []byte) (image.Image, error) {
	if len(hash) != 16 {
		return nil, ErrInvalidHash
	}
	
	meanL, meanA, meanB, splats := unpackV4(hash)
	
	w, h := 32, 32
	grid := make([]float64, w*h*3)
	
	// Fill background
	for i := 0; i < len(grid); i += 3 {
		grid[i] = meanL
		grid[i+1] = meanA
		grid[i+2] = meanB
	}
	
	// Add splats
	for _, s := range splats {
		addSplatToGrid(grid, s, w, h)
	}
	
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			idx := (y*w + x) * 3
			l, a, b := grid[idx], grid[idx+1], grid[idx+2]
			R, G, B := oklabToSrgb(l, a, b)
			out.SetRGBA(x, y, color.RGBA{
				R: uint8(clampi(int(R*255), 0, 255)),
				G: uint8(clampi(int(G*255), 0, 255)),
				B: uint8(clampi(int(B*255), 0, 255)),
				A: 255,
			})
		}
	}
	return out, nil
}

// --- Solver Math ---

func solveV4Weights(basis []Splat, tL, tA, tB []float64, w, h int) []Splat {
	N_Total := len(basis)
	N_Baryons := 0
	for _, s := range basis {
		if !s.IsLepton { N_Baryons++ }
	}
	
	// Precompute Activations
	activations := make([][]float64, N_Total)
	for i := 0; i < N_Total; i++ {
		activations[i] = computeBasisMap(basis[i], w, h)
	}
	
	// 1. Solve L channel (Uses all N splats)
	xL := solveChannel(activations, tL, N_Total, RidgeLambda)
	
	// 2. Solve A & B channels (Uses only N_Baryons splats)
	// We need a sub-slice of activations
	subActivations := activations[:N_Baryons]
	xA := solveChannel(subActivations, tA, N_Baryons, RidgeLambda)
	xB := solveChannel(subActivations, tB, N_Baryons, RidgeLambda)
	
	// Update Basis
	out := make([]Splat, N_Total)
	for i := 0; i < N_Total; i++ {
		out[i] = basis[i]
		out[i].L = xL[i]
		if i < N_Baryons {
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
	if N == 0 { return nil }
	M := len(target)
	
	ATA := make([]float64, N*N)
	ATb := make([]float64, N)
	
	for i := 0; i < N; i++ {
		for j := i; j < N; j++ {
			sum := 0.0
			rowI := activations[i]
			rowJ := activations[j]
			for p := 0; p < M; p++ {
				sum += rowI[p] * rowJ[p]
			}
			ATA[i*N+j] = sum
			ATA[j*N+i] = sum
		}
		
		sumB := 0.0
		rowI := activations[i]
		for p := 0; p < M; p++ {
			sumB += rowI[p] * target[p]
		}
		ATb[i] = sumB
	}
	
	// Regularization
	for i := 0; i < N; i++ {
		ATA[i*N+i] += lambda
	}
	
	return solveLinearSystem(ATA, ATb, N)
}

func computeBasisMap(s Splat, w, h int) []float64 {
	out := make([]float64, w*h)
	radius := int(s.Sigma * float64(w) * 3.5)
	cx := int(s.X * float64(w))
	cy := int(s.Y * float64(h))
	y0 := clampi(cy-radius, 0, h-1)
	y1 := clampi(cy+radius, 0, h-1)
	x0 := clampi(cx-radius, 0, w-1)
	x1 := clampi(cx+radius, 0, w-1)

	for y := y0; y <= y1; y++ {
		yf := float64(y) / float64(h)
		dy := yf - s.Y
		rowBase := y * w
		for x := x0; x <= x1; x++ {
			xf := float64(x) / float64(w)
			dx := xf - s.X
			distSq := dx*dx + dy*dy
			weight := math.Exp(-distSq / (2 * s.Sigma * s.Sigma))
			out[rowBase + x] = weight
		}
	}
	return out
}

func solveLinearSystem(A []float64, b []float64, N int) []float64 {
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

// --- Greedy Search ---
func findBestSplat(grid, currentRecon []float64, mL, mA, mB float64, w, h int, sigmas []float64) (Splat, float64) {
	var bestSplat Splat
	maxScore := -1.0
	resL := make([]float64, len(currentRecon)/3)
	resA := make([]float64, len(currentRecon)/3)
	resB := make([]float64, len(currentRecon)/3)
	for i := 0; i < len(resL); i++ {
		idx := i*3
		resL[i] = grid[idx] - mL - currentRecon[idx]
		resA[i] = grid[idx+1] - mA - currentRecon[idx+1]
		resB[i] = grid[idx+2] - mB - currentRecon[idx+2]
	}
	
	step := 2 
	for y := 0; y < h; y += step {
		for x := 0; x < w; x += step {
			xf := float64(x) / float64(w)
			yf := float64(y) / float64(h)
			for _, sigma := range sigmas {
				dotG_R_L, dotG_R_A, dotG_R_B, dotG_G := 0.0, 0.0, 0.0, 0.0
				radius := int(sigma * float64(w) * 3.5)
				y0 := clampi(y-radius, 0, h-1)
				y1 := clampi(y+radius, 0, h-1)
				x0 := clampi(x-radius, 0, w-1)
				x1 := clampi(x+radius, 0, w-1)
				
				for sy := y0; sy <= y1; sy++ {
					syf := float64(sy) / float64(h)
					dy := syf - yf
					rowBase := sy * w
					for sx := x0; sx <= x1; sx++ {
						sxf := float64(sx) / float64(w)
						dx := sxf - xf
						distSq := dx*dx + dy*dy
						weight := math.Exp(-distSq / (2 * sigma * sigma))
						idx := rowBase + sx
						dotG_R_L += weight * resL[idx]
						dotG_R_A += weight * resA[idx]
						dotG_R_B += weight * resB[idx]
						dotG_G += weight * weight
					}
				}
				if dotG_G < 1e-9 { continue }
				
				score := (dotG_R_L*dotG_R_L + dotG_R_A*dotG_R_A + dotG_R_B*dotG_R_B) / dotG_G
				if score > maxScore {
					maxScore = score
					bestSplat = Splat{
						X: xf, Y: yf,
						Sigma: sigma,
						L: dotG_R_L / dotG_G,
						A: dotG_R_A / dotG_G,
						B: dotG_R_B / dotG_G,
					}
				}
			}
		}
	}
	return bestSplat, maxScore
}

// --- Bit Packing ---

func packV4(mean uint16, splats []Splat) []byte {
	bw := &bitWriter{}
	
	// Header (16 bits)
	bw.Write(uint64(mean), 16)
	
	// Baryons (3x 22 bits)
	sigTbl := []float64{0.025, 0.1, 0.2, 0.35}
	
	count := 0
	for _, s := range splats {
		if s.IsLepton { continue }
		if count >= 3 { break }
		
		// Encode
		xi := clampi(int(s.X * 15.0 + 0.5), 0, 15) // 4b
		yi := clampi(int(s.Y * 15.0 + 0.5), 0, 15) // 4b
		
		sigI := 0
		minD := 100.0
		for i, v := range sigTbl {
			if d := math.Abs(v - s.Sigma); d < minD { minD = d; sigI = i }
		} // 2b
		
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
	// Pad if fewer than 3 Baryons found
	for count < 3 {
		bw.Write(0, 22); count++
	}
	
	// Leptons (3x 15 bits)
	count = 0
	for _, s := range splats {
		if !s.IsLepton { continue }
		if count >= 3 { break }
		
		xi := clampi(int(s.X * 15.0 + 0.5), 0, 15)
		yi := clampi(int(s.Y * 15.0 + 0.5), 0, 15)
		
		sigI := 0
		minD := 100.0
		for i, v := range sigTbl {
			if d := math.Abs(v - s.Sigma); d < minD { minD = d; sigI = i }
		}
		
		// Luma has 5 bits! Higher precision for details.
		lQ := quant(s.L, -0.8, 0.8, 5)
		
		bw.Write(uint64(xi), 4)
		bw.Write(uint64(yi), 4)
		bw.Write(uint64(sigI), 2)
		bw.Write(uint64(lQ), 5)
		
		count++
	}
	for count < 3 {
		bw.Write(0, 15); count++
	}
	
	// Pad last bit
	bw.Write(0, 1)
	
	return bw.Bytes()
}

func unpackV4(hash []byte) (mL, mA, mB float64, splats []Splat) {
	br := &bitReader{data: hash}
	
	// Header
	meanMap := uint16(br.Read(16))
	mL, mA, mB = unpackMean(meanMap)
	
	sigTbl := []float64{0.025, 0.1, 0.2, 0.35}
	
	// 3 Baryons
	for i := 0; i < 3; i++ {
		xi := br.Read(4)
		yi := br.Read(4)
		sigI := br.Read(2)
		lQ := br.Read(4)
		aQ := br.Read(4)
		bQ := br.Read(4)
		
		if lQ == 0 && aQ == 0 && bQ == 0 && xi == 0 && yi == 0 { continue } // Skip padding
		
		s := Splat{
			X: float64(xi) / 15.0,
			Y: float64(yi) / 15.0,
			Sigma: sigTbl[sigI],
			L: unquant(lQ, -0.8, 0.8, 4),
			A: unquant(aQ, -0.4, 0.4, 4),
			B: unquant(bQ, -0.4, 0.4, 4),
			IsLepton: false,
		}
		splats = append(splats, s)
	}
	
	// 3 Leptons
	for i := 0; i < 3; i++ {
		xi := br.Read(4)
		yi := br.Read(4)
		sigI := br.Read(2)
		lQ := br.Read(5)
		
		if lQ == 0 && xi == 0 && yi == 0 { continue }
		
		s := Splat{
			X: float64(xi) / 15.0,
			Y: float64(yi) / 15.0,
			Sigma: sigTbl[sigI],
			L: unquant(lQ, -0.8, 0.8, 5),
			A: 0,
			B: 0,
			IsLepton: true,
		}
		splats = append(splats, s)
	}
	
	return
}

// --- Internal Bit Helpers ---

type bitWriter struct {
	buf []byte
	acc uint64
	n   uint
}

func (b *bitWriter) Write(u uint64, bits uint) {
	b.acc = (b.acc << bits) | (u & ((1 << int(bits)) - 1))
	b.n += bits
	for b.n >= 8 {
		shift := b.n - 8
		b.buf = append(b.buf, byte(b.acc >> shift))
		b.n -= 8
	}
}

func (b *bitWriter) Bytes() []byte {
	// Flush remaining
	if b.n > 0 {
		b.buf = append(b.buf, byte(b.acc << (8-b.n)))
	}
	return b.buf
}

type bitReader struct {
	data []byte
	pos  int // byte pos
	rem  uint // remaining bits in current byte
	curr byte
}

func (b *bitReader) Read(bits uint) uint64 {
	var val uint64 = 0
	for bits > 0 {
		if b.rem == 0 {
			if b.pos >= len(b.data) {
				return val << bits // Out of data
			}
			b.curr = b.data[b.pos]
			b.pos++
			b.rem = 8
		}
		
		take := bits
		if b.rem < take { take = b.rem }
		
		shift := b.rem - take
		mask := byte((1 << take) - 1)
		chunk := (b.curr >> shift) & mask
		
		val = (val << take) | uint64(chunk)
		b.rem -= take
		bits -= take
	}
	return val
}

// --- Quantization Helpers ---

// Pack Mean: L(6b), A(5b), B(5b)
// L: 0..1 -> 0..63
// A: -0.2..0.2 -> 0..31 (Typical range for bg average)
// B: -0.2..0.2 -> 0..31
func packMean(l, a, b float64) uint16 {
	li := clampi(int(l * 63.5), 0, 63)
	ai := clampi(int(((a + 0.2)/0.4) * 31.5), 0, 31)
	bi := clampi(int(((b + 0.2)/0.4) * 31.5), 0, 31)
	return uint16((li << 10) | (ai << 5) | bi)
}

func unpackMean(p uint16) (l, a, b float64) {
	li := (p >> 10) & 0x3F
	ai := (p >> 5) & 0x1F
	bi := p & 0x1F
	l = float64(li) / 63.0
	a = (float64(ai)/31.0 * 0.4) - 0.2
	b = (float64(bi)/31.0 * 0.4) - 0.2
	return
}

func quant(v, min, max float64, bits uint) uint64 {
	steps := float64((uint64(1) << bits) - 1)
	norm := (v - min) / (max - min)
	i := int(norm * steps + 0.5)
	return uint64(clampi(i, 0, int(steps)))
}

func unquant(v uint64, min, max float64, bits uint) float64 {
	steps := float64((uint64(1) << bits) - 1)
	norm := float64(v) / steps
	return norm*(max-min) + min
}

// --- Shared Helpers ---
func addSplatToGrid(grid []float64, s Splat, w, h int) {
	radius := int(s.Sigma * float64(w) * 3.5)
	cx := int(s.X * float64(w))
	cy := int(s.Y * float64(h))
	y0 := clampi(cy-radius, 0, h-1)
	y1 := clampi(cy+radius, 0, h-1)
	x0 := clampi(cx-radius, 0, w-1)
	x1 := clampi(cx+radius, 0, w-1)

	for y := y0; y <= y1; y++ {
		yf := float64(y) / float64(h)
		dy := yf - s.Y
		rowBase := y * w * 3
		for x := x0; x <= x1; x++ {
			xf := float64(x) / float64(w)
			dx := xf - s.X
			distSq := dx*dx + dy*dy
			wVal := math.Exp(-distSq / (2 * s.Sigma * s.Sigma))
			idx := rowBase + x*3
			grid[idx] += s.L * wVal
			if !s.IsLepton {
				grid[idx+1] += s.A * wVal
				grid[idx+2] += s.B * wVal
			}
		}
	}
}

func imageToOklabGrid(img image.Image, w, h int) []float64 {
	bounds := img.Bounds()
	out := make([]float64, w*h*3)
	srcW := bounds.Dx(); srcH := bounds.Dy()
	for y := 0; y < h; y++ {
		y0 := float64(y) * float64(srcH) / float64(h)
		y1 := float64(y+1) * float64(srcH) / float64(h)
		for x := 0; x < w; x++ {
			x0 := float64(x) * float64(srcW) / float64(w)
			x1 := float64(x+1) * float64(srcW) / float64(w)
			var rSum, gSum, bSum, count float64
			iy0 := int(y0); iy1 := int(math.Ceil(y1))
			ix0 := int(x0); ix1 := int(math.Ceil(x1))
			if iy0 < 0 { iy0 = 0 }; if iy1 > srcH { iy1 = srcH }
			if ix0 < 0 { ix0 = 0 }; if ix1 > srcW { ix1 = srcW }
			for iy := iy0; iy < iy1; iy++ {
				for ix := ix0; ix < ix1; ix++ {
					r, g, b, _ := img.At(bounds.Min.X+ix, bounds.Min.Y+iy).RGBA()
					rSum += float64(r); gSum += float64(g); bSum += float64(b); count++
				}
			}
			if count == 0 { continue }
			r := (rSum / count) / 65535.0
			g := (gSum / count) / 65535.0
			b := (bSum / count) / 65535.0
			l, a, bb := srgbToOklab(r, g, b)
			idx := (y*w + x) * 3
			out[idx] = l; out[idx+1] = a; out[idx+2] = bb
		}
	}
	return out
}

func computeMean(grid []float64) [3]float64 {
	var l, a, b float64
	n := float64(len(grid) / 3)
	for i := 0; i < len(grid); i += 3 {
		l += grid[i]; a += grid[i+1]; b += grid[i+2]
	}
	return [3]float64{l / n, a / n, b / n}
}

func clampi(v, min, max int) int {
	if v < min { return min }
	if v > max { return max }
	return v
}

func srgbToOklab(r, g, b float64) (l, a, bb float64) {
	lin := func(c float64) float64 {
		if c <= 0.04045 { return c / 12.92 }
		return math.Pow((c+0.055)/1.055, 2.4)
	}
	r = lin(r); g = lin(g); b = lin(b)
	l1 := 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
	m1 := 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
	s1 := 0.0883024619*r + 0.2817188376*g + 0.6299787005*b
	l_ := math.Cbrt(l1); m_ := math.Cbrt(m1); s_ := math.Cbrt(s1)
	l = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
	a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
	bb = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
	return
}

func oklabToSrgb(l, a, b float64) (r, g, bl float64) {
	l_ := l + 0.3963377774*a + 0.2158037573*b
	m_ := l - 0.1055613458*a - 0.0638541728*b
	s_ := l - 0.0894841775*a - 1.2914855480*b
	l_ = l_ * l_ * l_; m_ = m_ * m_ * m_; s_ = s_ * s_ * s_
	r = +4.0767416621*l_ - 3.3077115913*m_ + 0.2309699292*s_
	g = -1.2684380046*l_ + 2.6097574011*m_ - 0.3413193965*s_
	bl = -0.0041960863*l_ - 0.7034186147*m_ + 1.7076147010*s_
	srt := func(c float64) float64 {
		if c <= 0.0031308 { return 12.92 * c }
		if c < 0 { return 0 }
		return 1.055*math.Pow(c, 1.0/2.4) - 0.055
	}
	return srt(r), srt(g), srt(bl)
}
