#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rasterio

try:
	from ..env_vars import (
		MASK_OUTPUT_DIR,
		NDVI_THRESHOLDS,
		NDWI_WATER_THRESHOLD,
		TILE_OUTPUT_DIR,
	)
except ImportError:
	from env_vars import MASK_OUTPUT_DIR, NDVI_THRESHOLDS, NDWI_WATER_THRESHOLD, TILE_OUTPUT_DIR


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
	denominator = green + nir
	with np.errstate(divide="ignore", invalid="ignore"):
		ndwi = np.where(denominator == 0, 0.0, (green - nir) / denominator)
	return ndwi.astype(np.float32)


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
	denominator = red + nir
	with np.errstate(divide="ignore", invalid="ignore"):
		ndvi = np.where(denominator == 0, 0.0, (nir - red) / denominator)
	return ndvi.astype(np.float32)


def classify_hybrid_ndvi_ndwi(ndvi: np.ndarray, ndwi: np.ndarray) -> np.ndarray:
	class_mask = np.full(ndvi.shape, 1, dtype=np.uint8)

	# NDWI disambiguates non-vegetation pixels: water (0) vs impervious (1).
	water_pixels = ndwi >= NDWI_WATER_THRESHOLD
	class_mask[water_pixels] = 0

	threshold_items = list(NDVI_THRESHOLDS.items())
	for index, (class_label, (low, high)) in enumerate(threshold_items):
		if index == len(threshold_items) - 1:
			pixels_in_range = (ndvi >= low) & (ndvi <= high)
		else:
			pixels_in_range = (ndvi >= low) & (ndvi < high)
		class_mask[pixels_in_range] = class_label

	return class_mask


def create_mask(src_dir: Path, dst_dir: Path) -> list[Path]:
	dst_dir.mkdir(parents=True, exist_ok=True)
	mask_paths = []

	for src_path in sorted(src_dir.glob("*.tif")):
		dst_path = dst_dir / src_path.name

		with rasterio.open(src_path) as src:
			if src.count < 4:
				raise ValueError(f"Expected at least 4 bands in {src_path}, found {src.count}")

			red = src.read(3).astype(np.float32)
			green = src.read(2).astype(np.float32)
			nir = src.read(4).astype(np.float32)

			ndvi = compute_ndvi(red, nir)
			ndwi = compute_ndwi(green, nir)
			class_mask = classify_hybrid_ndvi_ndwi(ndvi, ndwi)

			profile = src.profile.copy()
			profile.update(count=1, dtype=rasterio.uint8)

			with rasterio.open(dst_path, "w", **profile) as dst:
				dst.write(class_mask, 1)

		mask_paths.append(dst_path)
		print(f"Wrote mask: {dst_path.name}")

	return mask_paths


if __name__ == "__main__":
	mask_paths = create_mask(TILE_OUTPUT_DIR, MASK_OUTPUT_DIR)
	print(f"Created {len(mask_paths)} masks")
