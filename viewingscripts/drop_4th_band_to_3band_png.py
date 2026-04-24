#!/usr/bin/env python3
"""Read 4-band GeoTIFFs from an input directory and write 3-band RGB PNGs (band 4 dropped)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio


def iter_tifs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff"))


def three_band_chw_to_rgb_uint8(data: np.ndarray) -> np.ndarray:
    """Convert (3, H, W) to (H, W, 3) uint8 for PNG.

    uint8 inputs are passed through unchanged. Other dtypes are scaled with one
    linear map (global min/max across all three bands) so colors stay balanced.
    """
    if data.shape[0] != 3:
        raise ValueError(f"expected 3 bands on first axis, got shape {data.shape}")

    if data.dtype == np.uint8:
        return np.transpose(data, (1, 2, 0))

    d = data.astype(np.float64, copy=False)
    dmin = float(np.nanmin(d))
    dmax = float(np.nanmax(d))
    if dmax <= dmin:
        out = np.zeros((3, data.shape[1], data.shape[2]), dtype=np.uint8)
        return np.transpose(out, (1, 2, 0))

    if np.issubdtype(data.dtype, np.floating) and dmax <= 1.0 and dmin >= 0.0:
        scaled = np.clip(d, 0.0, 1.0) * 255.0
    else:
        scaled = (d - dmin) / (dmax - dmin) * 255.0

    rgb = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    return np.transpose(rgb, (1, 2, 0))


def drop_fourth_band_to_png(src_path: Path, dst_path: Path) -> None:
    with rasterio.open(src_path) as src:
        if src.count != 4:
            raise ValueError(
                f"expected exactly 4 bands, found {src.count} (bands 1-3 are kept, band 4 is dropped)"
            )
        data = src.read(indexes=[1, 2, 3])

    rgb_hwc = three_band_chw_to_rgb_uint8(data)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_hwc, mode="RGB").save(dst_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read each 4-band GeoTIFF in the input directory and write a 3-band RGB PNG "
            "to the output directory by dropping the 4th band (keeps bands 1-3)."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing source .tif / .tiff files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where 3-band PNGs are written (.png next to each input stem)",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Not a directory: {input_dir}", file=sys.stderr)
        return 1

    tifs = iter_tifs(input_dir)
    if not tifs:
        print(f"No .tif or .tiff files in {input_dir}", file=sys.stderr)
        return 1

    errors = 0
    for src_path in tifs:
        dst_path = output_dir / f"{src_path.stem}.png"
        try:
            drop_fourth_band_to_png(src_path, dst_path)
            print(dst_path)
        except ValueError as e:
            print(f"{src_path.name}: {e}", file=sys.stderr)
            errors += 1

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
