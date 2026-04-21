#!/usr/bin/env python3
"""Concatenate two or more PNG images left-to-right; last CLI argument is the output directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


def concat_horizontal(images: list[Image.Image]) -> Image.Image:
    rgb_images = [_to_rgb(im) for im in images]
    heights = [im.height for im in rgb_images]
    widths = [im.width for im in rgb_images]
    total_w = sum(widths)
    max_h = max(heights)
    out = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in rgb_images:
        y = (max_h - im.height) // 2
        out.paste(im, (x, y))
        x += im.width
    return out


def _default_out_name(png_paths: list[Path]) -> str:
    stems = [p.stem for p in png_paths]
    return "_".join(stems) + ".png"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate PNGs left to right in order. Last argument is the output directory."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Two or more PNG paths, then the output directory",
    )
    args = parser.parse_args()
    raw = [p.expanduser() for p in args.paths]
    if len(raw) < 3:
        parser.error("Need at least 2 PNG files plus output directory (3 arguments minimum).")

    png_paths = [p.resolve() for p in raw[:-1]]
    out_dir = raw[-1].expanduser().resolve()

    for p in png_paths:
        if not p.is_file():
            raise SystemExit(f"Not a file: {p}")
        if p.suffix.lower() != ".png":
            raise SystemExit(f"Expected .png input: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)

    images = [Image.open(p) for p in png_paths]
    try:
        combined = concat_horizontal(images)
    finally:
        for im in images:
            im.close()

    out_path = out_dir / _default_out_name(png_paths)
    combined.save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
