package splathash

import (
	"testing"

	"github.com/bbrks/go-blurhash"
	thumbhash "go.n16f.net/thumbhash"
)

func BenchmarkEncodeSplatHash(b *testing.B) {
	images := loadAssetImages(&testing.T{})
	if len(images) == 0 {
		b.Fatal("no assets")
	}
	img := images[0] // Use first image
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		EncodeImage(img)
	}
}

func BenchmarkEncodeThumbHash(b *testing.B) {
	images := loadAssetImages(&testing.T{})
	if len(images) == 0 {
		b.Fatal("no assets")
	}
	img := images[0]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thumbhash.EncodeImage(img)
	}
}

func BenchmarkEncodeBlurHash(b *testing.B) {
	images := loadAssetImages(&testing.T{})
	if len(images) == 0 {
		b.Fatal("no assets")
	}
	img := images[0]
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blurhash.Encode(4, 3, img)
	}
}

func BenchmarkDecodeSplatHash(b *testing.B) {
	images := loadAssetImages(&testing.T{})
	img := images[0]
	hash := EncodeImage(img)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = DecodeImage(hash)
	}
}

func BenchmarkDecodeThumbHash(b *testing.B) {
	images := loadAssetImages(&testing.T{})
	img := images[0]
	hash := thumbhash.EncodeImage(img)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		thumbhash.DecodeImage(hash)
	}
}

func BenchmarkDecodeBlurHash(b *testing.B) {
	images := loadAssetImages(&testing.T{})
	img := images[0]
	hash, _ := blurhash.Encode(4, 3, img)
	width := img.Bounds().Dx() / 8
	height := img.Bounds().Dy() / 8
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		blurhash.Decode(hash, width, height, 1)
	}
}
