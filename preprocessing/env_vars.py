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

MAX_DOWNLOADS=100

# NDVI is used to separate vegetation levels.
NDVI_THRESHOLDS = {
	2: (0.20, 0.65),      # sparse vegetation
	3: (0.65, 1.0),       # dense vegetation
}

# NDWI is used only to split low-NDVI pixels into water vs impervious.
NDWI_WATER_THRESHOLD = 0.10
