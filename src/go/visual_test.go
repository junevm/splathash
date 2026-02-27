package splathash

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bbrks/go-blurhash"
	thumbhash "go.n16f.net/thumbhash"
)

type namedImage struct {
	name string
	img  image.Image
}

func loadAssetImagesNamed(t *testing.T) []namedImage {
	t.Helper()
	searchPaths := []string{"../assets", "../../assets", "../../../assets"}
	var assetsDir string
	for _, p := range searchPaths {
		if info, err := os.Stat(p); err == nil && info.IsDir() {
			assetsDir = p
			break
		}
	}
	if assetsDir == "" {
		return nil
	}
	entries, err := os.ReadDir(assetsDir)
	if err != nil {
		return nil
	}
	var result []namedImage
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext := strings.ToLower(filepath.Ext(name))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}
		path := filepath.Join(assetsDir, name)
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		var img image.Image
		switch ext {
		case ".jpg", ".jpeg":
			img, err = jpeg.Decode(f)
		case ".png":
			img, err = png.Decode(f)
		}
		f.Close()
		if err != nil {
			continue
		}
		result = append(result, namedImage{name: name, img: img})
	}
	return result
}

func TestGenerateVisualComparison(t *testing.T) {
	namedImages := loadAssetImagesNamed(t)
	if len(namedImages) == 0 {
		t.Skip("no asset images found")
	}

	var html bytes.Buffer
	html.WriteString(`<!DOCTYPE html>
<html>
<head>
<style>
body { font-family: sans-serif; background: #333; color: #fff; }
table { border-collapse: collapse; width: 100%; }
td, th { border: 1px solid #555; padding: 5px; text-align: center; vertical-align: top; }
img, .css-preview { width: 200px; height: 150px; object-fit: cover; display: block; margin: 0 auto; }
.css-preview { width: 200px; height: 150px; } /* Fixed size for pure CSS */
.stats { font-size: 0.8em; color: #aaa; margin-top: 4px; }
</style>
</head>
<body>
<h1>SplatHash vs ThumbHash vs BlurHash</h1>
<table>
<thead>
<tr>
	<th>Original</th>
	<th>SplatHash</th>
	<th>ThumbHash</th>
	<th>BlurHash</th>
</tr>
</thead>
<tbody>
`)

	for _, ni := range namedImages {
		name := ni.name
		img := ni.img

		// SplatHash
		sh := EncodeImage(img)
		shDecoded, _ := DecodeImage(sh)

		// ThumbHash
		th := thumbhash.EncodeImage(img)
		thDecoded, _ := thumbhash.DecodeImage(th)

		// BlurHash (4x3 components is standardish)
		bhStr, _ := blurhash.Encode(4, 3, img)
		bhDecoded, _ := blurhash.Decode(bhStr, img.Bounds().Dx()/8, img.Bounds().Dy()/8, 1) // reduced res for preview

		// Encodings to Base64 for HTML
		origB64 := imgToBase64(img)
		shDecodedB64 := imgToBase64(shDecoded)
		thDecodedB64 := imgToBase64(thDecoded)
		bhDecodedB64 := imgToBase64(bhDecoded)

		html.WriteString(fmt.Sprintf(`<tr>
			<td>
				<img src="%s">
				<div class="stats">%s<br>%dx%d</div>
			</td>
			<td>
				<img src="%s">
				<div class="stats">Size: %d bytes (bin)</div>
			</td>
			<td>
				<img src="%s">
				<div class="stats">Size: %d bytes (bin)</div>
			</td>
			<td>
				<img src="%s">
				<div class="stats">Size: %d chars</div>
			</td>
		</tr>`,
			origB64, name, img.Bounds().Dx(), img.Bounds().Dy(),
			shDecodedB64, len(sh),
			thDecodedB64, len(th),
			bhDecodedB64, len(bhStr),
		))
	}

	html.WriteString(`</tbody></table></body></html>`)

	err := os.WriteFile("comparison.html", html.Bytes(), 0644)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Generated comparison.html")
}

func imgToBase64(img image.Image) string {
	if img == nil {
		return ""
	}
	var buf bytes.Buffer
	jpeg.Encode(&buf, img, nil)
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}
