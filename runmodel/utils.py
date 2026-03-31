from common.constants import DEFAULT_INPUT_DIR, DEFAULT_RESULTS_DIR
import argparse
import torch

def parse_args():
    
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, choices=["unet", "deeplab", "spanetfull"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to *_best.pt checkpoint")
    p.add_argument("--tiles_base", type=str, default=DEFAULT_INPUT_DIR,
               help="Path to dataset root containing images/ and masks/")
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)

    p.add_argument("--make_viz", type=str, default=False, help="Output visualization of predictions (only works on a max of 3 tiles)")

    p.add_argument("--limit", type=int, default=0, help="Limit number of images if not fixed (0 = all)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # UNet options (only used if model=unet)
    p.add_argument("--unet_encoder", type=str, default="resnet34")
    p.add_argument("--unet_weights", type=str, default="imagenet")

    return p.parse_args()

