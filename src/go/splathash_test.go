package splathash_test

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"

	splathash "github.com/junevm/splathash/src/go"
)

func TestEncodeDecodeRealImage(t *testing.T) {
	var assetPath string
	for _, candidate := range []string{"../../assets", "../../../assets"} {
		p := filepath.Join(candidate, "wallhaven-3q3j6y.jpg")
		if _, err := os.Stat(p); err == nil {
			assetPath = p
			break
		}
	}
	if assetPath == "" {
		t.Skip("Skipping real image test: asset not found")
		return
	}

	f, err := os.Open(assetPath)
	if err != nil {
		t.Skipf("Skipping real image test: Asset not found at %s (error: %v)", assetPath, err)
		return
	}
	defer f.Close()
	
	img, err := jpeg.Decode(f)
	if err != nil {
		t.Fatalf("Failed to decode jpeg: %v", err)
	}
	
	// Encode
	hash := splathash.EncodeImage(img)
	if len(hash) != 16 {
		t.Fatalf("Expected 16 bytes, got %d", len(hash))
	}
	
	hexStr := hex.EncodeToString(hash)
	t.Logf("Hash for %s: %s", assetPath, hexStr)
	
	// Determinism Check
	hash2 := splathash.EncodeImage(img)
	if !bytes.Equal(hash, hash2) {
		t.Fatal("Hash is not deterministic!")
	}
	
	// Decode (Visualize)
	recon, err := splathash.DecodeImage(hash)
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if recon.Bounds().Dx() != 32 || recon.Bounds().Dy() != 32 {
		t.Errorf("Reconstructed image should be 32x32, got %dx%d", recon.Bounds().Dx(), recon.Bounds().Dy())
	}
}

func ExampleEncodeImage() {
	// Not runnable in godoc playground without fs access, but shows usage.
	f, _ := os.Open("image.jpg")
	img, _ := jpeg.Decode(f)
	hash := splathash.EncodeImage(img)
	fmt.Printf("%x\n", hash)
}
