#!/usr/bin/env python3
"""Convert one class-ID TIFF mask in a directory to a single full-resolution color PNG (no tiling)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (REPO_ROOT, SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.constants import CLASS_COLORS  # noqa: E402
from convert_masks_to_color_png import colorize_mask, hex_to_rgb, normalize_mask_array  # noqa: E402


def find_single_tif(directory: Path) -> Path:
    """Return the path to the only ``.tif`` / ``.tiff`` in ``directory``."""
    tifs = sorted(directory.glob("*.tif")) + sorted(directory.glob("*.tiff"))
    if len(tifs) == 0:
        raise SystemExit(f"No .tif or .tiff files found in {directory}")
    if len(tifs) > 1:
        names = ", ".join(p.name for p in tifs)
        raise SystemExit(f"Expected exactly one TIFF in {directory}, found {len(tifs)}: {names}")
    return tifs[0]


def resolve_output_path(output: Path, stem: str) -> Path:
    """If ``output`` looks like a ``.png`` file, use it; otherwise treat as a directory."""
    output = output.expanduser()
    if output.suffix.lower() == ".png":
        return output.resolve()
    return (output.resolve() / f"{stem}.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Colorize the single class-ID mask TIFF in a directory and write one PNG (full raster, no tiling)."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory that contains exactly one mask .tif or .tiff",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path: a .png file, or a directory (writes <mask_stem>.png there)",
    )
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    tif_path = find_single_tif(input_dir)
    mask = np.array(Image.open(tif_path))
    mask = normalize_mask_array(mask)

    color_lut = np.array([hex_to_rgb(c) for c in CLASS_COLORS], dtype=np.uint8)
    rgb, unknown = colorize_mask(mask, color_lut)
    if unknown.size > 0:
        print(
            "WARNING: unknown class values (shown as black): "
            + ", ".join(str(v) for v in unknown.tolist()),
            file=sys.stderr,
        )

    out_path = resolve_output_path(args.output, tif_path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
