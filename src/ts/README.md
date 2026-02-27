# SplatHash — TypeScript / JavaScript

TypeScript implementation of [SplatHash](https://github.com/junevm/splathash): compress any image to 16 bytes and reconstruct it.

Works in **Node.js** and **browsers** — the library itself has zero runtime dependencies.

## Installation

```bash
npm install splathash
```

## Usage

### Node.js

The library works on raw RGBA bytes. Use any image-loading library (e.g. `sharp`) to get them:

```typescript
import sharp from "sharp";
import { encode, decode } from "splathash";

const { data, info } = await sharp("photo.jpg")
  .ensureAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });

// Encode: raw RGBA → 16-byte hash
const hash = encode(new Uint8ClampedArray(data), info.width, info.height);
console.log(Buffer.from(hash).toString("hex")); // e.g. a3f1bc...

// Decode: 16-byte hash → 32×32 RGBA
const result = decode(hash);
// result.width  = 32
// result.height = 32
// result.rgba   = Uint8ClampedArray (32 * 32 * 4 bytes)
```

### Browser

Same package, same API. Get raw RGBA from a `<canvas>`:

```typescript
import { encode, decode } from "splathash";

const canvas = document.getElementById("myCanvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

const hash = encode(data, canvas.width, canvas.height);
```

Or from an `<img>` element:

```typescript
function hashImage(img: HTMLImageElement): Uint8Array {
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return encode(data, canvas.width, canvas.height);
}
```

See [`examples/simple.ts`](examples/simple.ts) for a full Node.js example.

## API

### `encode(rgba, width, height): Uint8Array`

| Parameter | Type | Description |
| :-------- | :--- | :---------- |
| `rgba` | `Uint8ClampedArray \| Uint8Array` | Raw RGBA pixel data (4 bytes per pixel) |
| `width` | `number` | Image width in pixels |
| `height` | `number` | Image height in pixels |

Returns a `Uint8Array` of exactly 16 bytes.

### `decode(hash): DecodedImage`

| Parameter | Type | Description |
| :-------- | :--- | :---------- |
| `hash` | `Uint8Array` | 16-byte SplatHash |

Returns a `DecodedImage`:

```typescript
interface DecodedImage {
  width: number;          // always 32
  height: number;         // always 32
  rgba: Uint8ClampedArray; // 32 * 32 * 4 bytes
}
```

Throws if the hash is not exactly 16 bytes.

## Building from Source

```bash
npm install
npm run build   # emits to dist/
npm test        # build + run tests
```

Or via mise from the repo root:

```bash
mise run test:ts
```

## How It Works

See [ALGORITHM.md](https://github.com/junevm/splathash/blob/main/ALGORITHM.md) for the full technical specification.
