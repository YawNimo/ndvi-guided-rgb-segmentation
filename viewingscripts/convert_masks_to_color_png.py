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
    return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))


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


def create_side_by_side(original_img: np.ndarray, colorized_mask: np.ndarray) -> np.ndarray:
    """Combine original image and colorized mask side by side."""
    # Ensure both images have the same height
    h_orig = original_img.shape[0]
    h_mask = colorized_mask.shape[0]
    
    if h_orig != h_mask:
        # Resize colorized mask to match original height if needed
        mask_pil = Image.fromarray(colorized_mask)
        w_mask = int(colorized_mask.shape[1] * h_orig / h_mask)
        mask_pil = mask_pil.resize((w_mask, h_orig), Image.Resampling.LANCZOS)
        colorized_mask = np.array(mask_pil)
    
    # Create a canvas wide enough for both images side by side
    h = original_img.shape[0]
    w_orig = original_img.shape[1]
    w_mask = colorized_mask.shape[1]
    total_width = w_orig + w_mask
    
    combined = np.zeros((h, total_width, 3), dtype=np.uint8)
    combined[:, :w_orig, :] = original_img
    combined[:, w_orig:, :] = colorized_mask
    
    return combined


def convert_masks(input_dir: Path, output_dir: Path, images_dir: Path | None = None) -> int:
    """Convert all TIFF masks in input_dir to RGB PNGs in output_dir, combining with original images side by side."""
    if images_dir is None:
        images_dir = REPO_ROOT / "input" / "images"
    
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

        colorized, unknown_values = colorize_mask(mask, color_lut)
        if unknown_values.size > 0:
            warning_count += 1
            unknown_list = ", ".join(str(v) for v in unknown_values.tolist())
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
        
        # Create side-by-side image
        combined = create_side_by_side(original_img.astype(np.uint8), colorized)

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
    parser = argparse.ArgumentParser(description="Convert class-ID TIFF masks to color PNGs with original images side by side")
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
    return parser.parse_args()


def main() -> int:
    """Run CLI conversion flow and return process exit code.

    Returns:
        int: ``0`` when conversion completes.
    """
    args = parse_args()
    return convert_masks(args.input_dir, args.output_dir, args.images_dir)


if __name__ == "__main__":
    raise SystemExit(main())
