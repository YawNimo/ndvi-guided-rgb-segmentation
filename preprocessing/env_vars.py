"""Preprocessing paths and NDVI/NDWI threshold configuration."""

from pathlib import Path


INPUT_PATH = Path.cwd() / "input"

UNTILED_IMAGES_DIR = INPUT_PATH / "untiled_images"
BLURRED_IMAGES_DIR = INPUT_PATH / "blurred_images"

TILE_OUTPUT_DIR = INPUT_PATH / "images"
MASK_OUTPUT_DIR = INPUT_PATH / "masks"

CSV_COLUMN_NAME = "Aerial 2019 GeoTIFF"

MAX_TILE_SIZE = 1000  # to avoid memory issues when processing large images

GAUSSIAN_KERNEL_VALUE = 9

MAX_DOWNLOADS=100

NDVI_THRESHOLDS = {
	2: (0.2, 0.5),      # sparse vegetation
	3: (0.5, 1.0),       # dense vegetation
}

NDWI_WATER_THRESHOLD = 0.20
