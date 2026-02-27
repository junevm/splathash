import fs from "fs";
import path from "path";
import sharp from "sharp";
import { encode } from "../src";

async function main() {
  // Assume we run from 'src/ts/' directory via 'npm run example'
  const assetDir = path.resolve(process.cwd(), "../../assets");

  if (!fs.existsSync(assetDir)) {
    console.error(`Error: Assets not found at ${assetDir}`);
    console.error("Please run this script from the 'src/ts/' directory.");
    return;
  }

  const files = fs
    .readdirSync(assetDir)
    .filter((f) => f.endsWith(".jpg") || f.endsWith(".png"));

  console.log(`Found ${files.length} images in assets/`);

  for (const f of files) {
    const p = path.join(assetDir, f);
    try {
      const { data, info } = await sharp(p)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const rgba = new Uint8ClampedArray(data);

      const hash = encode(rgba, info.width, info.height);
      console.log(
        `File: ${f.padEnd(20)} -> Hash: ${Buffer.from(hash).toString("hex")}`,
      );
    } catch (e) {
      console.error(`Error processing ${f}:`, e);
    }
  }
}

main();
