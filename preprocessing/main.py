from pathlib import Path

import numpy as np
import rasterio


INPUT_DIR = Path("./input/untiled_images")
TILE_OUTPUT_DIR = Path("./input/images")
MASK_OUTPUT_DIR = Path("./input/masks")

NDVI_THRESHOLDS = {
	0: (-1.0, 0.0),     # water
	1: (0.0, 0.15),     # impervious
	2: (0.15, 0.25),    # sparse vegetation
	3: (0.25, 1.0),     # dense vegetation
}

MAX_IMAGE_SIZE = 1000  # needed to avoid memory issues when processing large images


def split_into_tiles(image: np.ndarray, tile_size: int) -> list[np.ndarray]:
	tiles = []
	for i in range(0, image.shape[0], tile_size):
		for j in range(0, image.shape[1], tile_size):
			tile = image[i:i + tile_size, j:j + tile_size]
			tiles.append(tile)
	return tiles


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


def create_tiles(src_path: Path, dst_dir: Path) -> list[Path]:
	import tifffile

	dst_dir.mkdir(parents=True, exist_ok=True)
	tile_paths = []

	image = tifffile.imread(src_path)
	tiles = split_into_tiles(image, MAX_IMAGE_SIZE)

	for i, tile in enumerate(tiles):
		tile_name = f"{src_path.stem}_tile_{i}.tif"
		tile_path = dst_dir / tile_name
		tifffile.imwrite(tile_path, tile)
		tile_paths.append(tile_path)

	return tile_paths


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
	MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	TILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	for tif_path in sorted(INPUT_DIR.glob("*.tif")):
		dst_paths = create_tiles(tif_path, TILE_OUTPUT_DIR)
		print(f"Processed {tif_path} into {len(dst_paths)} tiles")
	
	for tile_path in TILE_OUTPUT_DIR.glob("*.tif"):
		mask_path = process_file(tile_path, MASK_OUTPUT_DIR)
		print(f"Wrote {mask_path}")


if __name__ == "__main__":
	main()
