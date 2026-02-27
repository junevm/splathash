package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"sort"
	"strings"

	splathash "github.com/junevm/splathash/src/go"
)

func main() {
	searchPaths := []string{"../assets", "../../assets", "../../../assets", "../../../../assets"}
	var assetsDir string
	for _, p := range searchPaths {
		if info, err := os.Stat(p); err == nil && info.IsDir() {
			assetsDir = p
			break
		}
	}
	if assetsDir == "" {
		fmt.Fprintln(os.Stderr, "assets directory not found")
		os.Exit(1)
	}

	entries, err := os.ReadDir(assetsDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "failed to read assets dir:", err)
		os.Exit(1)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })

	for _, entry := range entries {
		name := entry.Name()
		ext  := strings.ToLower(filepath.Ext(name))
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
		hash := splathash.EncodeImage(img)
		fmt.Printf("%s: %x\n", name, hash)
	}
}
