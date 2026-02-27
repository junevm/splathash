"""SplatHash Python example — encode all images in ../../assets/ and print their hashes."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import splathash

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")

for fname in sorted(os.listdir(ASSETS_DIR)):
    if not fname.endswith((".jpg", ".jpeg", ".png")):
        continue
    path    = os.path.join(ASSETS_DIR, fname)
    h       = splathash.encode(path)
    print(f"{fname}: {h.hex()}")
