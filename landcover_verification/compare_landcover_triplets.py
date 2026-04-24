#!/usr/bin/env python3
"""Create visual triplet comparisons for landcover verification.

For each matched tile stem, this script renders:
1) Original RGB image
2) Colored remapped GT mask
3) Colored prediction mask

Outputs are saved as side-by-side PNG montages to support quick visual QA.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

try:
    import rasterio
except ImportError:  # pragma: no cover - optional fallback dependency
    rasterio = None

CLASS_COLOR_MAP: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),      # blue
    1: (255, 165, 0),    # orange
    2: (128, 255, 128),  # light green
    3: (0, 128, 0),      # dark green (kept for compatibility)
}


@dataclass(frozen=True)
class Triplet:
    tile_id: str
    rgb_path: Path
    gt_path: Path
    pred_path: Path


def _datasets_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets"


def parse_args() -> argparse.Namespace:
    datasets = _datasets_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Compare landcover triplets by generating per-tile montages of "
            "RGB | colored remapped GT | colored prediction."
        )
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=datasets / "tiled_images",
        help="Directory containing tiled RGB TIFF images.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=datasets / "remapped_landcover_masks",
        help="Directory containing remapped GT TIFF masks.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=datasets / "pred_masks",
        help="Directory containing prediction TIFF masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=datasets / "triplet_comparisons",
        help="Directory for output montage PNG files.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=2048,
        help=(
            "Resize each panel so max(height,width) <= this value. "
            "Set 0 to disable resizing."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on processed triplets; 0 means process all.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip stems that are missing one or more counterparts.",
    )
    args = parser.parse_args()
    if args.max_side < 0:
        parser.error("--max-side must be >= 0")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    return args


def _tifs(directory: Path) -> list[Path]:
    return sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    )


def _lookup(directory: Path) -> dict[str, Path]:
    return {fp.stem: fp for fp in _tifs(directory)}


def _load_rgb(path: Path) -> np.ndarray:
    arr = _read_tif_with_fallback(path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D RGB image, got shape {arr.shape} for {path}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape {arr.shape} for {path}")
    rgb = arr[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def _load_mask(path: Path) -> np.ndarray:
    arr = _read_tif_with_fallback(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {arr.shape} for {path}")
    return arr.astype(np.uint8)


def _read_tif_with_fallback(path: Path) -> np.ndarray:
    try:
        return tifffile.imread(path)
    except ValueError as exc:
        # JPEG-compressed TIFFs can require imagecodecs for tifffile.
        if "requires the 'imagecodecs' package" not in str(exc):
            raise
        if rasterio is None:
            raise ValueError(
                f"{path} is JPEG-compressed TIFF and imagecodecs is unavailable. "
                "Install imagecodecs or rasterio."
            ) from exc
        try:
            with rasterio.open(path) as src:
                data = src.read()
        except Exception as raster_exc:  # pragma: no cover - IO failure path
            raise ValueError(f"Failed to read {path} via tifffile and rasterio: {raster_exc}") from exc
        if data.ndim == 3:
            return np.moveaxis(data, 0, -1)
        return data


def _colorize_mask(mask: np.ndarray, source_path: Path) -> np.ndarray:
    known = np.isin(mask, list(CLASS_COLOR_MAP.keys()))
    if (~known).any():
        unknown_labels = np.unique(mask[~known]).tolist()
        labels = ", ".join(str(int(v)) for v in unknown_labels)
        raise ValueError(
            f"{source_path.name} contains unmapped classes: {labels}. "
            f"Expected subset of {sorted(CLASS_COLOR_MAP.keys())}"
        )
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLOR_MAP.items():
        rgb[mask == cls_id] = color
    return rgb


def _resize_image(arr: np.ndarray, max_side: int, is_mask_rgb: bool) -> np.ndarray:
    if max_side <= 0:
        return arr
    h, w = arr.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return arr
    scale = max_side / float(largest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resample = Image.Resampling.NEAREST if is_mask_rgb else Image.Resampling.BILINEAR
    pil_img = Image.fromarray(arr, mode="RGB")
    resized = pil_img.resize((new_w, new_h), resample=resample)
    return np.array(resized)


def _as_panel(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr, mode="RGB")


def _compose_horizontal(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    rgb_img = _as_panel(rgb)
    gt_img = _as_panel(gt)
    pred_img = _as_panel(pred)

    width = rgb_img.width + gt_img.width + pred_img.width
    height = max(rgb_img.height, gt_img.height, pred_img.height)
    canvas = Image.new("RGB", (width, height), color=(15, 15, 15))
    y_offset = 0
    x = 0
    for panel in (rgb_img, gt_img, pred_img):
        canvas.paste(panel, (x, y_offset))
        x += panel.width
    return canvas


def _collect_triplets(images_dir: Path, gt_dir: Path, pred_dir: Path, skip_missing: bool) -> tuple[list[Triplet], list[str]]:
    rgb_lookup = _lookup(images_dir)
    gt_lookup = _lookup(gt_dir)
    pred_lookup = _lookup(pred_dir)

    tile_ids = sorted(set(rgb_lookup) | set(gt_lookup) | set(pred_lookup))
    triplets: list[Triplet] = []
    skipped: list[str] = []

    for tile_id in tile_ids:
        rgb = rgb_lookup.get(tile_id)
        gt = gt_lookup.get(tile_id)
        pred = pred_lookup.get(tile_id)
        if rgb is None or gt is None or pred is None:
            missing = []
            if rgb is None:
                missing.append("rgb")
            if gt is None:
                missing.append("gt")
            if pred is None:
                missing.append("pred")
            msg = f"{tile_id}:missing_{'_'.join(missing)}"
            if skip_missing:
                skipped.append(msg)
                continue
            raise ValueError(f"Missing counterpart files for {tile_id}: {', '.join(missing)}")
        triplets.append(Triplet(tile_id=tile_id, rgb_path=rgb, gt_path=gt, pred_path=pred))

    return triplets, skipped


def _validate_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise ValueError(f"{name} directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{name} path is not a directory: {path}")


def main() -> int:
    args = parse_args()
    try:
        _validate_dir(args.images_dir, "images")
        _validate_dir(args.gt_dir, "gt")
        _validate_dir(args.pred_dir, "pred")
        triplets, skipped = _collect_triplets(
            images_dir=args.images_dir,
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            skip_missing=args.skip_missing,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not triplets:
        print("ERROR: No valid RGB/GT/PRED triplets found.", file=sys.stderr)
        return 1

    if args.limit > 0:
        triplets = triplets[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for triplet in triplets:
        try:
            rgb = _load_rgb(triplet.rgb_path)
            gt_mask = _load_mask(triplet.gt_path)
            pred_mask = _load_mask(triplet.pred_path)
            if rgb.shape[:2] != gt_mask.shape:
                raise ValueError(
                    f"{triplet.tile_id}: RGB shape {rgb.shape[:2]} does not match GT shape {gt_mask.shape}"
                )
            if rgb.shape[:2] != pred_mask.shape:
                raise ValueError(
                    f"{triplet.tile_id}: RGB shape {rgb.shape[:2]} does not match PRED shape {pred_mask.shape}"
                )

            gt_rgb = _colorize_mask(gt_mask, triplet.gt_path)
            pred_rgb = _colorize_mask(pred_mask, triplet.pred_path)

            rgb = _resize_image(rgb, args.max_side, is_mask_rgb=False)
            gt_rgb = _resize_image(gt_rgb, args.max_side, is_mask_rgb=True)
            pred_rgb = _resize_image(pred_rgb, args.max_side, is_mask_rgb=True)

            if not (rgb.shape == gt_rgb.shape == pred_rgb.shape):
                raise ValueError(
                    f"{triplet.tile_id}: panel shapes diverged after resize: "
                    f"rgb={rgb.shape}, gt={gt_rgb.shape}, pred={pred_rgb.shape}"
                )

            montage = _compose_horizontal(rgb, gt_rgb, pred_rgb)
            out_path = args.output_dir / f"{triplet.tile_id}.png"
            montage.save(out_path)
            written += 1
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    print(f"Wrote {written} triplet comparison PNGs to {args.output_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} stems due to missing files (--skip-missing enabled).")
    print("Panel order: RGB | GT (colored remapped mask) | PRED (colored prediction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
