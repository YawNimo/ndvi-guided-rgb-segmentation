"""Command-line argument parsing for inference runs."""

from common.constants import (
    DEFAULT_INPUT_DIR,
    DEFAULT_RESULTS_DIR,
    INPUT_IMAGES_SUBDIR,
    INPUT_MASKS_SUBDIR,
    OUTPUT_PRED_MASKS_SUBDIR,
)
import argparse
import torch


def parse_args():
    """Parse CLI arguments for model inference execution.

    Returns:
        argparse.Namespace: Parsed inference configuration.
    """
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, choices=["unet", "deeplab", "spanetfull"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to *_best.pt checkpoint")
    p.add_argument(
        "--tiles_base",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Path to dataset root containing images/ and masks/",
    )
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    p.add_argument(
        "--images_subdir",
        type=str,
        default=INPUT_IMAGES_SUBDIR,
        help="Image subdirectory under --tiles_base when --images_dir is not set",
    )
    p.add_argument(
        "--masks_subdir",
        type=str,
        default=INPUT_MASKS_SUBDIR,
        help="Mask subdirectory under --tiles_base when --masks_dir is not set",
    )
    p.add_argument(
        "--pred_subdir",
        type=str,
        default=OUTPUT_PRED_MASKS_SUBDIR,
        help="Prediction subdirectory under results/<model> when --pred_output_dir is not set",
    )
    p.add_argument(
        "--images_dir",
        type=str,
        default="",
        help="Optional explicit image directory, overrides --tiles_base/--images_subdir",
    )
    p.add_argument(
        "--masks_dir",
        type=str,
        default="",
        help="Optional explicit mask directory, overrides --tiles_base/--masks_subdir",
    )
    p.add_argument(
        "--pred_output_dir",
        type=str,
        default="",
        help="Optional explicit output directory for predictions, overrides results/<model>/--pred_subdir",
    )

    p.add_argument("--make_viz", type=str, default=False, help="Output visualization of predictions (only works on a max of 3 tiles)")

    p.add_argument("--limit", type=int, default=0, help="Limit number of images if not fixed (0 = all)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # U-Net options are only consumed when ``--model unet`` is selected.
    p.add_argument("--unet_encoder", type=str, default="resnet34")
    p.add_argument("--unet_weights", type=str, default="imagenet")

    return p.parse_args()

