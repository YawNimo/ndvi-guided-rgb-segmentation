#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

CLASS_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (0, 255, 0),      # GREEN
    1: (255, 165, 0),    # ORANGE
    2: (0, 128, 0),      # DArk green
    3: (0, 0, 255),        # BLUE
    4: (128, 255, 128),        # LIGHT GREEN
}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert class-id TIFF masks (0/1/2) to color PNGs."#todo fix comment
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "datasets" / "landcover_dataset" / "masks",
        help="Directory containing input TIFF masks (default: landcover_dataset/masks)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "colored_pngs",
        help="Directory where color PNGs will be written (default: landcover_verification/unmapped_colored_pngs)",
    )
    parser.add_argument(
        "--allow-unknown-labels",
        action="store_true",
        help="Allow labels outside {0,1,2}; unknown labels are rendered as black.",
    )
    return parser.parse_args()


def _collect_tifs(directory: Path) -> list[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])


def _load_mask(tif_path: Path) -> np.ndarray:
    arr = tifffile.imread(tif_path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape} for {tif_path}")
    return arr.astype(np.uint8)


def _colorize(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLOR_MAP.items():
        rgb[mask == cls_id] = color
    known = np.isin(mask, list(CLASS_COLOR_MAP.keys()))
    unknown_labels = np.unique(mask[~known]) if (~known).any() else np.array([], dtype=mask.dtype)
    return rgb, unknown_labels


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory: {input_dir}", file=sys.stderr)
        return 1

    tif_paths = _collect_tifs(input_dir)
    if not tif_paths:
        print(f"ERROR: No TIFF masks found in {input_dir}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for tif_path in tif_paths:
        if written >= 5:
            break
        mask = _load_mask(tif_path)
        rgb, unknown_labels = _colorize(mask)
        if unknown_labels.size > 0 and not args.allow_unknown_labels:
            labels = ", ".join(str(int(v)) for v in unknown_labels.tolist())
            print(
                f"ERROR: {tif_path.name} contains unsupported labels: {labels}. "
                "Use --allow-unknown-labels to continue (unknown labels become black).",
                file=sys.stderr,
            )
            return 1

        out_path = output_dir / f"{tif_path.stem}.png"
        Image.fromarray(rgb, mode="RGB").save(out_path)
        written += 1

    print(f"Wrote {written} color PNGs to {output_dir}")
    print("Color mapping: 0->blue, 1->orange, 2->green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
