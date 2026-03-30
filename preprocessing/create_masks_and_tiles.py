from pathlib import Path

import numpy as np
import rasterio

from env_vars import (
	UNTILED_IMAGES_DIR,
	TILE_OUTPUT_DIR,
	MASK_OUTPUT_DIR,
)
from utils.create_tiles import create_tiles
from utils.create_masks import create_mask


def main() -> None:
	MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	TILE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	for tif_path in sorted(UNTILED_IMAGES_DIR.glob("*.tif")):
		dst_paths = create_tiles(tif_path, TILE_OUTPUT_DIR)
		print(f"Processed {tif_path} into {len(dst_paths)} tiles")
	
	for tile_path in TILE_OUTPUT_DIR.glob("*.tif"):
		mask_path = create_mask(tile_path, MASK_OUTPUT_DIR)
		print(f"Wrote {mask_path}")


if __name__ == "__main__":
	main()
