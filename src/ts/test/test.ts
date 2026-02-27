import * as assert from "assert";
import * as fs from "fs";
import * as path from "path";
import sharp from "sharp";
import { decode, encode } from "../src";

console.log("Running SplatHash Tests (Real Assets)...");

// Use asset from ../../../../assets/
const assetPath = path.resolve(
  __dirname,
  "../../../../assets/wallhaven-3q3j6y.jpg",
);

async function runTests() {
  if (!fs.existsSync(assetPath)) {
    console.warn(`Skipping real asset test: File not found at ${assetPath}`);
    return;
  }

  console.log(`Loading ${assetPath}...`);
  const { data, info } = await sharp(assetPath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const width = info.width;
  const height = info.height;

  // Convert Buffer to Uint8ClampedArray
  const rgba = new Uint8ClampedArray(data);

  // 2. Encode
  console.log("Encoding...");
  const hash = encode(rgba, width, height);
  const hex = Buffer.from(hash).toString("hex");
  console.log(`Hash: ${hex}`);

  assert.strictEqual(hash.length, 16, "Hash length must be 16 bytes");

  // 3. Determinism
  console.log("Checking Determinism...");
  const hash2 = encode(rgba, width, height);
  assert.strictEqual(
    Buffer.from(hash2).toString("hex"),
    hex,
    "Hash must be deterministic",
  );

  // 4. Decode
  console.log("Decoding...");
  const decoded = decode(hash);

  assert.strictEqual(decoded.width, 32);
  assert.strictEqual(decoded.height, 32);
  assert.strictEqual(decoded.rgba.length, 32 * 32 * 4);

  console.log("✅ All tests passed!");
}

runTests().catch((err) => {
  console.error("Test Failed:", err);
  process.exit(1);
});
