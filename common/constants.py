"""
Shared environment and class constants.
"""

DEFAULT_INPUT_DIR = "input"
INPUT_IMAGES_SUBDIR = "images"
INPUT_MASKS_SUBDIR = "masks"

DEFAULT_RESULTS_DIR = "results"
OUTPUT_PRED_MASKS_SUBDIR = "pred_masks"

DEFAULT_CHECKPOINTS_DIR = "checkpoints"

CLASSES = {
    0: "water",
    1: "impervious",
    2: "sparse veg",
    3: "dense veg",
}

CLASS_NAMES = [CLASSES[i] for i in range(len(CLASSES))]

CLASS_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#98df8a",
    "#2ca02c",
]
