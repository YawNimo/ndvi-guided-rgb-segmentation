#!/usr/bin/env python3
"""Tile landcover RGB rasters and write remapped GT mask tiles.

This prepares verification inputs aligned with the main pipeline tiling style:
- tile naming: ``{stem}_tile_{i}.tif`` (row-major)
- tile size: defaults to preprocessing MAX_TILE_SIZE (1000)
- mask labels: remapped to model class order before scoring
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from landcover_remap import EXPECTED_INPUT_LABELS, validate_labels
from remap_constants import LANDCOVER_CLASS_MAPPING
from preprocessing.env_vars import MAX_TILE_SIZE
from preprocessing.utils.create_tiles import split_into_tiles


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent / "datasets"
    parser = argparse.ArgumentParser(
        description="Create tiled RGB images and remapped tiled GT masks for landcover verification."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=base / "landcover_dataset" / "images",
        help="Directory with source RGB GeoTIFF files.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=base / "landcover_dataset" / "masks",
        help="Directory with source GT mask GeoTIFF files.",
    )
    parser.add_argument(
        "--out-images-dir",
        type=Path,
        default=base / "tiled_images",
        help="Output directory for tiled RGB TIFF files.",
    )
    parser.add_argument(
        "--out-remapped-masks-dir",
        type=Path,
        default=base / "remapped_landcover_masks",
        help="Output directory for tiled remapped GT TIFF files.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=MAX_TILE_SIZE,
        help=f"Tile edge length in pixels (default: {MAX_TILE_SIZE}).",
    )
    parser.add_argument(
        "--allow-extra-labels",
        action="store_true",
        help="Allow unexpected raw mask labels.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate pairing/labels/shapes only; do not write tiles.",
    )
    args = parser.parse_args()
    if args.tile_size <= 0:
        parser.error("--tile-size must be > 0")
    return args


def _read_tif(path: Path) -> np.ndarray:
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        if "requires the 'imagecodecs' package" not in str(exc):
            raise
        if rasterio is None:
            raise ValueError(
                f"{path} is JPEG-compressed TIFF and imagecodecs is unavailable. "
                "Install imagecodecs or rasterio."
            ) from exc
        with rasterio.open(path) as src:
            data = src.read()
        if data.ndim == 3:
            return np.moveaxis(data, 0, -1)
        return data


def _load_rgb(path: Path) -> np.ndarray:
    arr = _read_tif(path)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got {arr.shape} for {path}")
    if arr.shape[0] in {3, 4} and arr.shape[-1] not in {3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels, got {arr.shape} for {path}")
    rgb = arr[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def _load_mask(path: Path) -> np.ndarray:
    arr = _read_tif(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {arr.shape} for {path}")
    return np.asarray(arr, dtype=np.uint8)


def _collect_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}])
    mask_lookup = {p.stem: p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}}
    pairs: list[tuple[Path, Path]] = []
    missing_masks: list[str] = []
    for image_path in images:
        mask_path = mask_lookup.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.stem)
            continue
        pairs.append((image_path, mask_path))
    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        raise ValueError(
            f"Missing matching masks for {len(missing_masks)} image(s): {preview}"
            + (" ..." if len(missing_masks) > 5 else "")
        )
    return pairs


def main() -> int:
    args = parse_args()
    if not args.images_dir.is_dir():
        print(f"ERROR: Missing images directory: {args.images_dir}", file=sys.stderr)
        return 1
    if not args.masks_dir.is_dir():
        print(f"ERROR: Missing masks directory: {args.masks_dir}", file=sys.stderr)
        return 1

    try:
        pairs = _collect_pairs(args.images_dir, args.masks_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not pairs:
        print("ERROR: No paired image/mask TIFF files found.", file=sys.stderr)
        return 1

    if not args.preflight_only:
        args.out_images_dir.mkdir(parents=True, exist_ok=True)
        args.out_remapped_masks_dir.mkdir(parents=True, exist_ok=True)

    lut = np.arange(256, dtype=np.uint8)
    for src, dst in LANDCOVER_CLASS_MAPPING.items():
        lut[src] = dst

    written_images = 0
    written_masks = 0
    total_tiles = 0
    observed_labels: set[int] = set()

    for image_path, mask_path in pairs:
        rgb = _load_rgb(image_path)
        raw_mask = _load_mask(mask_path)

        if rgb.shape[:2] != raw_mask.shape:
            print(
                f"ERROR: Shape mismatch for {image_path.stem}: image={rgb.shape[:2]} mask={raw_mask.shape}",
                file=sys.stderr,
            )
            return 1

        labels = set(np.unique(raw_mask).tolist())
        observed_labels.update(int(v) for v in labels)
        try:
            validate_labels(labels, EXPECTED_INPUT_LABELS, allow_extra_labels=args.allow_extra_labels)
        except ValueError as exc:
            print(f"ERROR: {mask_path.name}: {exc}", file=sys.stderr)
            return 1
        unmapped = labels - set(LANDCOVER_CLASS_MAPPING.keys())
        if unmapped:
            print(
                f"ERROR: {mask_path.name} has unmapped labels {sorted(unmapped)}; "
                f"mapping keys are {sorted(LANDCOVER_CLASS_MAPPING.keys())}",
                file=sys.stderr,
            )
            return 1

        rgb_tiles = split_into_tiles(rgb, args.tile_size)
        mask_tiles = split_into_tiles(raw_mask, args.tile_size)
        if len(rgb_tiles) != len(mask_tiles):
            print(
                f"ERROR: Tile count mismatch for {image_path.stem}: "
                f"rgb={len(rgb_tiles)} mask={len(mask_tiles)}",
                file=sys.stderr,
            )
            return 1

        total_tiles += len(rgb_tiles)
        if args.preflight_only:
            continue

        for idx, (rgb_tile, mask_tile) in enumerate(zip(rgb_tiles, mask_tiles)):
            tile_name = f"{image_path.stem}_tile_{idx}.tif"
            tifffile.imwrite(args.out_images_dir / tile_name, rgb_tile)
            tifffile.imwrite(args.out_remapped_masks_dir / tile_name, lut[mask_tile])
            written_images += 1
            written_masks += 1

    print(f"Paired source scenes: {len(pairs)}")
    print(f"Observed raw labels: {sorted(observed_labels)}")
    print(f"Total generated tiles: {total_tiles}")
    if args.preflight_only:
        print("Preflight complete. No files written (--preflight-only).")
    else:
        print(f"Wrote RGB tiles: {written_images} -> {args.out_images_dir}")
        print(f"Wrote remapped mask tiles: {written_masks} -> {args.out_remapped_masks_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
