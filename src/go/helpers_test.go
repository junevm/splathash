package splathash

import (
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// loadAssetImages loads all JPEG and PNG images from the shared assets directory.
// It searches parent directories so the test works regardless of working directory.
func loadAssetImages(t *testing.T) []image.Image {
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
		t.Log("assets directory not found; skipping asset-based tests")
		return nil
	}

	entries, err := os.ReadDir(assetsDir)
	if err != nil {
		t.Logf("failed to read assets dir: %v", err)
		return nil
	}

	var images []image.Image
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		ext  := strings.ToLower(filepath.Ext(name))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}
		path := filepath.Join(assetsDir, name)
		f, err := os.Open(path)
		if err != nil {
			t.Logf("skipping %s: %v", name, err)
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
			t.Logf("skipping %s: decode error: %v", name, err)
			continue
		}
		images = append(images, img)
	}
	return images
}
