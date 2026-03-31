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


def download() -> None:
	print("Downloading ZIP files...")
	download_zips(max_downloads=MAX_DOWNLOADS)
	print("Extracting ZIP files...")
	extract_all_zips()


def tile() -> None:
	print("Creating tiles...")
	tile_paths = create_tiles(UNTILED_IMAGES_DIR, TILE_OUTPUT_DIR)
	print(f"Created {len(tile_paths)} tiles")


def blur() -> None:
	print("Blurring images...")
	blurred_paths = blur_images(TILE_OUTPUT_DIR, BLURRED_IMAGES_DIR, GAUSSIAN_KERNEL_VALUE)
	print(f"Created {len(blurred_paths)} blurred images")


def mask() -> None:
	print("Creating masks...")
	mask_paths = create_mask(BLURRED_IMAGES_DIR, MASK_OUTPUT_DIR)
	print(f"Created {len(mask_paths)} masks")


def run_cleanup() -> None:
	print("Running cleanup...")
	clean_non_tif()
	remove_zip_and_untiled_dirs()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run preprocessing pipeline steps individually or in sequence."
	)
	parser.add_argument(
		"--download",
		action="store_true",
		help="Download and extract ZIP files.",
	)
	parser.add_argument(
		"--tile",
		action="store_true",
		help="Create tiles from images.",
	)
	parser.add_argument(
		"--blur",
		action="store_true",
		help="Blur tiles.",
	)
	parser.add_argument(
		"--mask",
		action="store_true",
		help="Create masks from blurred tiles.",
	)
	parser.add_argument(
		"--cleanup",
		action="store_true",
		help="Run cleaning steps.",
	)
	args = parser.parse_args()

	if not any([args.download, args.tile, args.blur, args.mask, args.cleanup]):
		parser.error("Pass at least one of: --download, --tile, --blur, --mask, --cleanup")

	return args


def main() -> None:
	args = parse_args()

	# Execute steps in fixed order regardless of argument order
	if args.download:
		download()
	
	if args.tile:
		tile()
	
	if args.blur:
		blur()
	
	if args.mask:
		mask()

	if args.cleanup:
		run_cleanup()


if __name__ == "__main__":
	main()
