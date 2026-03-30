from pathlib import Path


# Base paths
INPUT_PATH = Path.cwd() / "input"

# Extraction and processing directories
UNTILED_IMAGES_DIR = INPUT_PATH / "untiled_images"

# Processing output directories
TILE_OUTPUT_DIR = INPUT_PATH / "images"
MASK_OUTPUT_DIR = INPUT_PATH / "masks"

# Processing settings
MAX_TILE_SIZE = 1000  # to avoid memory issues when processing large images

NDVI_THRESHOLDS = {
	0: (-1.0, 0.0),     # water
	1: (0.0, 0.15),     # impervious
	2: (0.15, 0.25),    # sparse vegetation
	3: (0.25, 1.0),     # dense vegetation
}
