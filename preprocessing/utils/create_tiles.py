#!/usr/bin/env python3

from pathlib import Path

import numpy as np

try:
	from ..env_vars import UNTILED_IMAGES_DIR, TILE_OUTPUT_DIR, MAX_TILE_SIZE
except ImportError:
	from env_vars import UNTILED_IMAGES_DIR, TILE_OUTPUT_DIR, MAX_TILE_SIZE


def split_into_tiles(image: np.ndarray, tile_size: int) -> list[np.ndarray]:
	tiles = []
	for i in range(0, image.shape[0], tile_size):
		for j in range(0, image.shape[1], tile_size):
			tile = image[i:i + tile_size, j:j + tile_size]
			tiles.append(tile)
	return tiles


def create_tiles(src_dir: Path, dst_dir: Path) -> list[Path]:
	import tifffile

	dst_dir.mkdir(parents=True, exist_ok=True)
	tile_paths = []

	for src_path in sorted(src_dir.glob("*.tif")):
		image = tifffile.imread(src_path)
		tiles = split_into_tiles(image, MAX_TILE_SIZE)

		for i, tile in enumerate(tiles):
			tile_name = f"{src_path.stem}_tile_{i}.tif"
			tile_path = dst_dir / tile_name
			tifffile.imwrite(tile_path, tile)
			tile_paths.append(tile_path)
			print(f"Created tile: {tile_path.name}")

	return tile_paths


if __name__ == "__main__":
	tile_paths = create_tiles(UNTILED_IMAGES_DIR, TILE_OUTPUT_DIR)
	print(f"Created {len(tile_paths)} tiles")
