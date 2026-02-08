#!/usr/bin/env python3
"""Extract CS2 radar map images from awpy's cache to our static assets.

Usage:
    python -m opensight.scripts.extract_maps

If awpy map images aren't available, downloads them from GitHub.
Falls back to generating placeholder images with map name labels.
"""

from __future__ import annotations

import shutil
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

# Target directory for map images
STATIC_MAPS_DIR = Path(__file__).parent.parent / "static" / "maps"
MAPS = [
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_vertigo",
    "de_train",
]

_RADAR_BASE = "https://raw.githubusercontent.com/2mlml/cs2-radar-images/master"


def extract_from_awpy() -> bool:
    """Try to copy map images from awpy's cache."""
    awpy_maps = Path.home() / ".awpy" / "maps"
    if not awpy_maps.exists():
        print(f"  awpy maps directory not found at {awpy_maps}")
        return False

    STATIC_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0

    for map_name in MAPS:
        dst = STATIC_MAPS_DIR / f"{map_name}.png"
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"  {map_name}.png already exists, skipping")
            copied += 1
            continue

        src = awpy_maps / f"{map_name}.png"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {map_name}.png from awpy cache")
            copied += 1
        else:
            print(f"  {map_name}.png not found in awpy cache")

    return copied > 0


def download_from_github() -> bool:
    """Download radar images from the cs2-radar-images GitHub repo."""
    STATIC_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for map_name in MAPS:
        dst = STATIC_MAPS_DIR / f"{map_name}.png"
        if dst.exists() and dst.stat().st_size > 1000:
            print(f"  {map_name}.png already exists, skipping")
            downloaded += 1
            continue

        url = f"{_RADAR_BASE}/{map_name}.png"
        print(f"  Downloading {map_name}.png ...", end=" ", flush=True)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OpenSight/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 1000:
                print(f"too small ({len(data)} bytes), skipping")
                continue
            dst.write_bytes(data)
            print(f"OK ({len(data):,} bytes)")
            downloaded += 1
        except Exception as e:
            print(f"failed: {e}")

    return downloaded > 0


def _write_png(path: Path, width: int, height: int, rgba_rows: list[bytes]) -> None:
    """Write a minimal PNG file without any external dependencies."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    raw = b""
    for row in rgba_rows:
        raw += b"\x00" + row  # filter byte 0 (None) per row
    idat = zlib.compress(raw, 9)
    png = sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    path.write_bytes(png)


def generate_placeholders() -> bool:
    """Generate placeholder PNGs using pure Python (no Pillow needed)."""
    STATIC_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0

    for map_name in MAPS:
        dst = STATIC_MAPS_DIR / f"{map_name}.png"
        if dst.exists() and dst.stat().st_size > 1000:
            continue

        # 256x256 dark placeholder (upscaled to 1024x1024 by CSS/canvas)
        size = 256
        bg = (20, 20, 30, 255)
        grid_color = (40, 40, 50, 128)
        rows: list[bytes] = []
        for y in range(size):
            row = bytearray()
            for x in range(size):
                if x % 32 == 0 or y % 32 == 0:
                    row.extend(grid_color)
                else:
                    row.extend(bg)
            rows.append(bytes(row))

        _write_png(dst, size, size, rows)
        print(f"  Generated placeholder for {map_name} ({dst.stat().st_size:,} bytes)")
        generated += 1

    return generated > 0


def main() -> None:
    print("Extracting CS2 radar map images...")
    print(f"Target: {STATIC_MAPS_DIR}\n")

    # Strategy 1: Copy from awpy cache
    print("Strategy 1: awpy cache")
    if extract_from_awpy():
        missing = [m for m in MAPS if not (STATIC_MAPS_DIR / f"{m}.png").exists()]
        if not missing:
            print("\nAll maps found in awpy cache.")
            _report()
            return

    # Strategy 2: Download from GitHub
    print("\nStrategy 2: Download from GitHub")
    if download_from_github():
        missing = [m for m in MAPS if not (STATIC_MAPS_DIR / f"{m}.png").exists()]
        if not missing:
            print("\nAll maps downloaded successfully.")
            _report()
            return

    # Strategy 3: Generate placeholders for any remaining
    missing = [m for m in MAPS if not (STATIC_MAPS_DIR / f"{m}.png").exists()]
    if missing:
        print(f"\nStrategy 3: Generating placeholders for {len(missing)} missing maps")
        generate_placeholders()

    _report()


def _report() -> None:
    existing = sorted(STATIC_MAPS_DIR.glob("de_*.png"))
    real = [f for f in existing if f.stat().st_size > 5000]
    placeholder = [f for f in existing if f.stat().st_size <= 5000]
    print(f"\n{len(existing)} map images in {STATIC_MAPS_DIR}")
    if real:
        print(f"  {len(real)} real radar images")
    if placeholder:
        print(f"  {len(placeholder)} placeholders (replace with real images for best results)")


if __name__ == "__main__":
    main()
    sys.exit(0)
