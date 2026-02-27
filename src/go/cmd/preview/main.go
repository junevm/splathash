// preview generates a side-by-side visual comparison PNG of SplatHash, ThumbHash,
// and BlurHash reconstructions for a sample of images from the assets/ directory.
// Output is written to docs/comparison.png at the repository root.
package main

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	blurhash "github.com/bbrks/go-blurhash"
	splathash "github.com/junevm/splathash/src/go"
	thumbhash "go.n16f.net/thumbhash"
	xdraw "golang.org/x/image/draw"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

const (
	cellW   = 128 // width of each image cell
	cellH   = 96  // height of each image cell (4:3)
	headerH = 22  // column header row height
	labelH  = 18  // row label height below each image row
	gap     = 2   // gap between cells
	maxRows = 4   // max images to compare
)

var (
	colLabels = []string{"Original", "SplatHash (16B)", "ThumbHash (~30B)", "BlurHash (~25B)"}
	colColors = []color.RGBA{
		{R: 60, G: 60, B: 60, A: 255},
		{R: 0, G: 80, B: 160, A: 255},
		{R: 0, G: 120, B: 70, A: 255},
		{R: 150, G: 70, B: 0, A: 255},
	}
	bg = color.RGBA{R: 25, G: 25, B: 25, A: 255}
)

func main() {
	// Paths are relative to the working directory (src/go when run via mise).
	assetsDir := filepath.Join("..", "..", "assets")
	outPath := filepath.Join("..", "..", "docs", "comparison.png")

	images := loadImages(assetsDir, maxRows)
	if len(images) == 0 {
		fmt.Fprintln(os.Stderr, "no images found in assets/")
		os.Exit(1)
	}

	cols := len(colLabels)
	rows := len(images)
	totalW := cols*(cellW+gap) - gap
	totalH := headerH + rows*(cellH+labelH+gap)

	out := image.NewRGBA(image.Rect(0, 0, totalW, totalH))
	fill(out, out.Bounds(), bg)

	// Column headers
	for c, label := range colLabels {
		x := c * (cellW + gap)
		fill(out, image.Rect(x, 0, x+cellW, headerH), colColors[c])
		drawText(out, x+3, 15, label, color.White)
	}

	// Image rows
	for row, ni := range images {
		img := ni.img
		y0 := headerH + row*(cellH+labelH+gap)

		// Encode + decode with each algorithm
		shHash := splathash.EncodeImage(img)
		shDec, _ := splathash.DecodeImage(shHash)

		thHash := thumbhash.EncodeImage(img)
		thDec, _ := thumbhash.DecodeImage(thHash)

		bhStr, _ := blurhash.Encode(4, 3, img)
		// Decode at 1/8 scale — blurhash is designed for low-res previews
		bhW := max(img.Bounds().Dx()/8, 8)
		bhH := max(img.Bounds().Dy()/8, 8)
		bhDec, _ := blurhash.Decode(bhStr, bhW, bhH, 1)

		srcs := []image.Image{img, shDec, thDec, bhDec}
		for c, src := range srcs {
			if src == nil {
				continue
			}
			x := c * (cellW + gap)
			dst := image.Rect(x, y0, x+cellW, y0+cellH)
			xdraw.BiLinear.Scale(out, dst, src, src.Bounds(), xdraw.Over, nil)
		}

		// Row label
		name := ni.name
		if len(name) > 22 {
			name = name[:22] + "…"
		}
		drawText(out, 3, y0+cellH+13, name, color.RGBA{R: 160, G: 160, B: 160, A: 255})
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir: %v\n", err)
		os.Exit(1)
	}
	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "create: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	if err := png.Encode(f, out); err != nil {
		fmt.Fprintf(os.Stderr, "encode: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Wrote %s (%dx%d)\n", outPath, totalW, totalH)
}

type namedImage struct {
	name string
	img  image.Image
}

func loadImages(dir string, limit int) []namedImage {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var result []namedImage
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		ext := strings.ToLower(filepath.Ext(name))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}
		img, err := loadImage(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		result = append(result, namedImage{name, img})
		if len(result) >= limit {
			break
		}
	}
	return result
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	switch strings.ToLower(filepath.Ext(path)) {
	case ".jpg", ".jpeg":
		return jpeg.Decode(f)
	case ".png":
		return png.Decode(f)
	default:
		return nil, fmt.Errorf("unsupported format: %s", path)
	}
}

func fill(dst *image.RGBA, r image.Rectangle, c color.Color) {
	xdraw.Draw(dst, r, image.NewUniform(c), image.Point{}, xdraw.Src)
}

func drawText(dst *image.RGBA, x, y int, text string, clr color.Color) {
	d := &font.Drawer{
		Dst:  dst,
		Src:  image.NewUniform(clr),
		Face: basicfont.Face7x13,
		Dot:  fixed.Point26_6{X: fixed.Int26_6(x << 6), Y: fixed.Int26_6(y << 6)},
	}
	d.DrawString(text)
}
