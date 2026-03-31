#!/usr/bin/env python3

"""Utilities for Gaussian blurring tiled raster images."""

from pathlib import Path

import cv2
import rasterio

try:
	from ..env_vars import BLURRED_IMAGES_DIR, GAUSSIAN_KERNEL_VALUE, UNTILED_IMAGES_DIR
except ImportError:
	from env_vars import BLURRED_IMAGES_DIR, GAUSSIAN_KERNEL_VALUE, UNTILED_IMAGES_DIR


def _to_valid_kernel_size(kernel_value: int) -> int:
	"""Convert configured kernel value into OpenCV-compatible odd kernel size.

	Args:
		kernel_value (int): Positive kernel value from preprocessing settings.

	Returns:
		int: Odd kernel size passed to ``cv2.GaussianBlur``.
	"""
	if kernel_value < 1:
		raise ValueError("GAUSSIAN_KERNEL_VALUE must be >= 1")
	# OpenCV requires an odd kernel size, so map value=2 to size=5.
	return (2 * kernel_value) + 1


def blur_images(src_dir: Path, dst_dir: Path, kernel_value: int) -> list[Path]:
	"""Blur all TIFF files in ``src_dir`` and write results to ``dst_dir``.

	Args:
		src_dir (Path): Directory containing source ``.tif`` tiles.
		dst_dir (Path): Directory where blurred tiles are written.
		kernel_value (int): Blur kernel control value converted to odd size.

	Returns:
		list[Path]: Output paths of blurred TIFF files.
	"""
	dst_dir.mkdir(parents=True, exist_ok=True)
	blurred_paths: list[Path] = []
	kernel_size = _to_valid_kernel_size(kernel_value)

	for src_path in sorted(src_dir.glob("*.tif")):
		dst_path = dst_dir / src_path.name

		with rasterio.open(src_path) as src:
			image = src.read()
			profile = src.profile.copy()

			blurred = image.copy()
			for band_idx in range(image.shape[0]):
				band = image[band_idx]
				blurred[band_idx] = cv2.GaussianBlur(band, (kernel_size, kernel_size), 0)

			with rasterio.open(dst_path, "w", **profile) as dst:
				dst.write(blurred)

		blurred_paths.append(dst_path)
		print(f"Created blurred image: {dst_path.name}")

	return blurred_paths


if __name__ == "__main__":
	blurred_paths = blur_images(UNTILED_IMAGES_DIR, BLURRED_IMAGES_DIR, GAUSSIAN_KERNEL_VALUE)
	print(f"Created {len(blurred_paths)} blurred images")