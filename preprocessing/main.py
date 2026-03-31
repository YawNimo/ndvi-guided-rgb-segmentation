import argparse

from env_vars import (
	UNTILED_IMAGES_DIR,
	BLURRED_IMAGES_DIR,
	TILE_OUTPUT_DIR,
	MASK_OUTPUT_DIR,
	GAUSSIAN_KERNEL_VALUE,
	MAX_DOWNLOADS,
)
from utils.download import download_zips
from utils.unzip import extract_all_zips
from utils.clean_folders import clean_non_tif, remove_zip_and_untiled_dirs
from utils.blur_images import blur_images
from utils.create_tiles import create_tiles
from utils.create_masks import create_mask


def run_pipeline() -> None:
	# Download
	print("Downloading ZIP files...")
	download_zips(max_downloads=MAX_DOWNLOADS)
	
	# Extract
	print("Extracting ZIP files...")
	extract_all_zips()
	
	# Create tiles
	print("Creating tiles...")
	tile_paths = create_tiles(UNTILED_IMAGES_DIR, TILE_OUTPUT_DIR)
	print(f"Created {len(tile_paths)} tiles")

	# Blur tiles before mask creation
	print("Blurring images...")
	blurred_paths = blur_images(TILE_OUTPUT_DIR, BLURRED_IMAGES_DIR, GAUSSIAN_KERNEL_VALUE)
	print(f"Created {len(blurred_paths)} blurred images")

	# Create masks from blurred tiles
	print("Creating masks...")
	mask_paths = create_mask(BLURRED_IMAGES_DIR, MASK_OUTPUT_DIR)
	print(f"Created {len(mask_paths)} masks")


def run_cleanup() -> None:
	print("Running cleanup...")
	clean_non_tif()
	remove_zip_and_untiled_dirs()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run preprocessing pipeline (tile, blur, and mask) and/or cleanup."
	)
	parser.add_argument(
		"--run",
		action="store_true",
		help="Run non-cleaning steps: tile images, blur tiles, and create masks from blurred tiles.",
	)
	parser.add_argument(
		"--cleanup",
		action="store_true",
		help="Run cleaning steps.",
	)
	args = parser.parse_args()

	if not args.run and not args.cleanup:
		parser.error("Pass --run, --cleanup, or both.")

	return args


def main() -> None:
	args = parse_args()

	if args.run:
		run_pipeline()

	if args.cleanup:
		run_cleanup()


if __name__ == "__main__":
	main()
