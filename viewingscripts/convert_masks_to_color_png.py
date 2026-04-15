#!/usr/bin/env python3
"""Convert class-ID TIFF masks to color PNGs for visual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.constants import CLASS_COLORS  # noqa: E402


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color like #1f77b4 into an RGB tuple."""
    cleaned = hex_color.lstrip("#")
    if len(cleaned) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4)) # type: ignore


def normalize_mask_array(mask: np.ndarray) -> np.ndarray:
    """Normalize a loaded TIFF mask into a single channel class-ID array."""
    if mask.ndim == 2:
        return mask
    if mask.ndim == 3:
        return mask[..., 0]
    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def colorize_mask(mask: np.ndarray, color_lut: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map class IDs to RGB colors; unknown class values are colored black."""
    h, w = mask.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)

    class_count = color_lut.shape[0]
    valid = (mask >= 0) & (mask < class_count)
    if np.any(valid):
        colorized[valid] = color_lut[mask[valid]]

    unknown_values = np.unique(mask[~valid]) if np.any(~valid) else np.array([], dtype=mask.dtype)
    return colorized, unknown_values


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize an RGB image to a target height while preserving aspect ratio."""
    if image.shape[0] == target_height:
        return image

    image_pil = Image.fromarray(image)
    target_width = int(round(image.shape[1] * target_height / image.shape[0]))
    resized = image_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return np.array(resized)


def create_triplet(rgb_img: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Combine RGB input, colored ground truth, and colored prediction side by side."""
    target_height = rgb_img.shape[0]
    gt_mask = resize_to_height(gt_mask, target_height)
    pred_mask = resize_to_height(pred_mask, target_height)

    total_width = rgb_img.shape[1] + gt_mask.shape[1] + pred_mask.shape[1]
    combined = np.zeros((target_height, total_width, 3), dtype=np.uint8)
    cursor = 0

    combined[:, cursor : cursor + rgb_img.shape[1], :] = rgb_img
    cursor += rgb_img.shape[1]
    combined[:, cursor : cursor + gt_mask.shape[1], :] = gt_mask
    cursor += gt_mask.shape[1]
    combined[:, cursor : cursor + pred_mask.shape[1], :] = pred_mask

    return combined


def convert_masks(
    input_dir: Path,
    output_dir: Path,
    images_dir: Path | None = None,
    pred_dir: Path | None = None,
) -> int:
    """Convert all TIFF masks in input_dir to RGB PNGs in output_dir, combining RGB, GT, and prediction triplets."""
    if images_dir is None:
        images_dir = REPO_ROOT / "input" / "images"
    if pred_dir is None:
        pred_dir = REPO_ROOT / "results" / "unet" / "pred_masks"
    
    tiff_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    print(f"Found {len(tiff_files)} TIFF mask files in {input_dir}")

    if not tiff_files:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    color_lut = np.array([hex_to_rgb(color) for color in CLASS_COLORS], dtype=np.uint8)

    warning_count = 0
    skipped_count = 0
    for tif_path in tiff_files:
        mask = np.array(Image.open(tif_path))
        mask = normalize_mask_array(mask)

        gt_colorized, gt_unknown_values = colorize_mask(mask, color_lut)
        if gt_unknown_values.size > 0:
            warning_count += 1
            unknown_list = ", ".join(str(v) for v in gt_unknown_values.tolist())
            print(f"WARNING: {tif_path.name} contains unknown class values: {unknown_list}")
        
        # Try to load the corresponding original image
        original_img_path = images_dir / f"{tif_path.stem}.tif"
        if not original_img_path.exists():
            original_img_path = images_dir / f"{tif_path.stem}.tiff"
        
        if not original_img_path.exists():
            print(f"SKIP: No matching original image found for {tif_path.name}")
            skipped_count += 1
            continue
        
        original_img = np.array(Image.open(original_img_path))
        
        # Ensure original image is RGB
        if original_img.ndim == 2:
            original_img = np.stack([original_img] * 3, axis=-1)
        elif original_img.ndim == 3 and original_img.shape[2] == 4:
            # Drop alpha channel if present
            original_img = original_img[:, :, :3]
        elif original_img.ndim == 3 and original_img.shape[2] > 3:
            # Take first 3 channels
            original_img = original_img[:, :, :3]

        pred_mask_path = pred_dir / tif_path.name
        if not pred_mask_path.exists():
            pred_mask_path = pred_dir / f"{tif_path.stem}.tiff"

        if not pred_mask_path.exists():
            print(f"SKIP: No matching prediction mask found for {tif_path.name}")
            skipped_count += 1
            continue

        pred_mask = np.array(Image.open(pred_mask_path))
        pred_mask = normalize_mask_array(pred_mask)
        pred_colorized, pred_unknown_values = colorize_mask(pred_mask, color_lut)
        if pred_unknown_values.size > 0:
            warning_count += 1
            unknown_list = ", ".join(str(v) for v in pred_unknown_values.tolist())
            print(f"WARNING: {pred_mask_path.name} contains unknown class values: {unknown_list}")
        
        # Create triplet image
        combined = create_triplet(original_img.astype(np.uint8), gt_colorized, pred_colorized)

        out_path = output_dir / f"{tif_path.stem}.png"
        Image.fromarray(combined, mode="RGB").save(out_path)
        print(f"Wrote: {out_path}")

    print("Conversion complete")
    print(f"Converted files: {len(tiff_files) - skipped_count}")
    print(f"Skipped files (no matching original image): {skipped_count}")
    print(f"Files with unknown class warnings: {warning_count}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for mask colorization and side-by-side rendering.

    Returns:
        argparse.Namespace: Parsed paths for input masks, output directory,
        and optional source image directory.
    """
    parser = argparse.ArgumentParser(description="Convert class-ID TIFF masks to color PNGs with RGB, ground truth, and prediction triplets")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "input" / "masks",
        help="Directory containing class-ID TIFF mask files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "viewingscripts" / "out",
        help="Directory where color PNGs will be written (default: viewingscripts/out)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing original images (default: input/images)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help="Model name used to resolve the default prediction directory under results/<model>/pred_masks",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Directory containing predicted masks (default: results/<model>/pred_masks)",
    )
    return parser.parse_args()


def main() -> int:
    """Run CLI conversion flow and return process exit code.

    Returns:
        int: ``0`` when conversion completes.
    """
    args = parse_args()
    pred_dir = args.pred_dir if args.pred_dir is not None else REPO_ROOT / "results" / args.model / "pred_masks"
    return convert_masks(args.input_dir, args.output_dir, args.images_dir, pred_dir)


if __name__ == "__main__":
    raise SystemExit(main())
