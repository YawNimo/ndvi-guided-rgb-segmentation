from pathlib import Path

import numpy as np
import rasterio


INPUT_DIR = Path("./input/images")
OUTPUT_DIR = Path("./input/masks")

NDVI_THRESHOLDS = {
	0: (-1.0, 0.0),     # water
	1: (0.0, 0.15),     # impervious
	2: (0.15, 0.25),    # sparse vegetation
	3: (0.25, 1.0),     # dense vegetation
}


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
	denominator = red + nir
	with np.errstate(divide="ignore", invalid="ignore"):
		ndvi = np.where(denominator == 0, 0.0, (nir - red) / denominator)
	return ndvi.astype(np.float32)


def classify_ndvi(ndvi: np.ndarray) -> np.ndarray:
	class_mask = np.zeros(ndvi.shape, dtype=np.uint8)
	threshold_items = list(NDVI_THRESHOLDS.items())

	for index, (class_label, (low, high)) in enumerate(threshold_items):
		if index == len(threshold_items) - 1:
			pixels_in_range = (ndvi >= low) & (ndvi <= high)
		else:
			pixels_in_range = (ndvi >= low) & (ndvi < high)
		class_mask[pixels_in_range] = class_label

	return class_mask


def process_file(src_path: Path, dst_dir: Path) -> Path:
	dst_path = dst_dir / src_path.name

	with rasterio.open(src_path) as src:
		if src.count < 4:
			raise ValueError(f"Expected at least 4 bands in {src_path}, found {src.count}")

		red = src.read(3).astype(np.float32)
		nir = src.read(4).astype(np.float32)

		ndvi = compute_ndvi(red, nir)
		class_mask = classify_ndvi(ndvi)

		profile = src.profile.copy()
		profile.update(count=1, dtype=rasterio.uint8)

		with rasterio.open(dst_path, "w", **profile) as dst:
			dst.write(class_mask, 1)

	return dst_path


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	tif_paths = sorted(INPUT_DIR.glob("*.tif"))

	for tif_path in tif_paths:
		dst_path = process_file(tif_path, OUTPUT_DIR)
		print(f"Wrote {dst_path}")


if __name__ == "__main__":
	main()
