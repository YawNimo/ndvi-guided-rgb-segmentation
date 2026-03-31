from pathlib import Path


# Base paths
INPUT_PATH = Path.cwd() / "input"

# Extraction and processing directories
UNTILED_IMAGES_DIR = INPUT_PATH / "untiled_images"
BLURRED_IMAGES_DIR = INPUT_PATH / "blurred_images"

# Processing output directories
TILE_OUTPUT_DIR = INPUT_PATH / "images"
MASK_OUTPUT_DIR = INPUT_PATH / "masks"

CSV_COLUMN_NAME = "Aerial 2023 GeoTIFF"

# Processing settings
MAX_TILE_SIZE = 1000  # to avoid memory issues when processing large images

GAUSSIAN_KERNEL_VALUE = 9

MAX_DOWNLOADS=1

NDVI_THRESHOLDS = {
	0: (-1.0, -0.15),     # water
	90: (-0.15, 0.2),     # impervious
	150: (0.2, 0.65),    # sparse vegetation
	200: (0.65, 1.0),     # dense vegetation
}
