import './style.css';
import { rgbaToThumbHash, thumbHashToRGBA } from 'thumbhash';
import { encode as blurhashEncode, decode as blurhashDecode } from 'blurhash';
// Import local SplatHash version for the demo via internal ts src
import { encode as splathashEncode, decode as splathashDecode } from '../../src/ts/src/splathash';

// --- Utility Functions ---

function toHex(array: Uint8Array): string {
    return Array.from(array).map(b => b.toString(16).padStart(2, '0')).join('');
}

function putImageDataToCanvas(id: string, imageData: ImageData) {
    const canvas = document.getElementById(id) as HTMLCanvasElement;
    if (!canvas) return;
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d');
    if (ctx) {
        ctx.putImageData(imageData, 0, 0);
    }
}

// --- Image Processing Variables ---

const MAX_DIM = 100; // Cap encoding dimension for speed

// --- Main Execution Logic ---

function setupFileListener() {
    const input = document.getElementById('imageInput') as HTMLInputElement;

    input.addEventListener('change', (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => processImage(img);
            img.src = event.target?.result as string;
        };
        reader.readAsDataURL(file);
    });
}

function processImage(img: HTMLImageElement) {
    // Show sections and hide placeholder
    document.getElementById('splathash-demo')!.style.display = 'block';
    document.getElementById('tbl-placeholder')!.style.display = 'none';
    document.getElementById('tbl-data')!.style.display = 'table-row-group';

    // 1. Draw Original correctly bounding size while retaining ratio
    let w = img.width;
    let h = img.height;
    if (w > MAX_DIM || h > MAX_DIM) {
        if (w > h) {
            h = Math.round((h * MAX_DIM) / w);
            w = MAX_DIM;
        } else {
            w = Math.round((w * MAX_DIM) / h);
            h = MAX_DIM;
        }
    }

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(img, 0, 0, w, h);
    const imageData = ctx.getImageData(0, 0, w, h);

    // Update Interactive Dashboard Original
    putImageDataToCanvas('demo-original', imageData);
    putImageDataToCanvas('tbl-orig-img', imageData);
    document.getElementById('tbl-orig-size')!.innerText = `${(img.width * img.height * 4).toLocaleString()} bytes (raw)`;

    const pixels = imageData.data;

    // --- SplatHash ---
    const t0SplatEnc = performance.now();
    const splatArray = splathashEncode(pixels, w, h);
    const t1SplatEnc = performance.now();

    const t0SplatDec = performance.now();
    const splatDecodedPixels = splathashDecode(splatArray);
    const t1SplatDec = performance.now();

    const splatHex = toHex(splatArray);
    const splatEncTime = (t1SplatEnc - t0SplatEnc).toFixed(2);
    const splatDecTime = (t1SplatDec - t0SplatDec).toFixed(2);

    putImageDataToCanvas('demo-splat', new ImageData(new Uint8ClampedArray(splatDecodedPixels.rgba), 32, 32));
    putImageDataToCanvas('tbl-splat-img', new ImageData(new Uint8ClampedArray(splatDecodedPixels.rgba), 32, 32));

    // Update Main Demo Box
    document.getElementById('demo-hash')!.innerText = splatHex;
    document.getElementById('demo-enc')!.innerText = `${splatEncTime} ms`;
    document.getElementById('demo-dec')!.innerText = `${splatDecTime} ms`;

    // Update Table
    document.getElementById('tbl-splat-hash')!.innerText = splatHex;
    document.getElementById('tbl-splat-enc')!.innerText = `${splatEncTime} ms`;
    document.getElementById('tbl-splat-dec')!.innerText = `${splatDecTime} ms`;

    // --- ThumbHash ---
    const t0ThumbEnc = performance.now();
    const thumbArray = rgbaToThumbHash(w, h, pixels);
    const t1ThumbEnc = performance.now();

    const t0ThumbDec = performance.now();
    const thumbDecoded = thumbHashToRGBA(thumbArray);
    const t1ThumbDec = performance.now();

    // resize to original scale approximation
    const thumbImgData = new ImageData(new Uint8ClampedArray(thumbDecoded.rgba), thumbDecoded.w, thumbDecoded.h);
    
    putImageDataToCanvas('tbl-thumb-img', thumbImgData);
    document.getElementById('tbl-thumb-hash')!.innerText = toHex(thumbArray);
    document.getElementById('tbl-thumb-size')!.innerText = `${thumbArray.length} bytes`;
    document.getElementById('tbl-thumb-enc')!.innerText = `${(t1ThumbEnc - t0ThumbEnc).toFixed(2)} ms`;
    document.getElementById('tbl-thumb-dec')!.innerText = `${(t1ThumbDec - t0ThumbDec).toFixed(2)} ms`;

    // --- BlurHash ---
    // BlurHash components (commonly 4x3)
    const t0BlurEnc = performance.now();
    const blurHashStr = blurhashEncode(pixels, w, h, 4, 3);
    const t1BlurEnc = performance.now();

    const t0BlurDec = performance.now();
    const blurDecodedPixels = blurhashDecode(blurHashStr, w, h);
    const t1BlurDec = performance.now();

    putImageDataToCanvas('tbl-blur-img', new ImageData(new Uint8ClampedArray(blurDecodedPixels), w, h));
    document.getElementById('tbl-blur-hash')!.innerText = blurHashStr;
    document.getElementById('tbl-blur-size')!.innerText = `${blurHashStr.length} chars (variable)`;
    document.getElementById('tbl-blur-enc')!.innerText = `${(t1BlurEnc - t0BlurEnc).toFixed(2)} ms`;
    document.getElementById('tbl-blur-dec')!.innerText = `${(t1BlurDec - t0BlurDec).toFixed(2)} ms`;
}

// Init
setupFileListener();
