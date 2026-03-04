def parse_args():
    import argparse
    
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, choices=["unet", "deeplab", "spanetfull"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to *_best.pt checkpoint")
    p.add_argument("--tiles_base", type=str, required=True,
               help="Path to dataset root containing images/ and masks/")
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)

    p.add_argument("--fixed", action="store_true", help="Run only fixed tiles visualization + preds")
    p.add_argument("--fixed_tiles", type=str, default="", help="Optional JSON list or comma-separated names")

    p.add_argument("--limit", type=int, default=0, help="Limit number of images if not fixed (0 = all)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # UNet options (only used if model=unet)
    p.add_argument("--unet_encoder", type=str, default="resnet34")
    p.add_argument("--unet_weights", type=str, default="imagenet")

    return p.parse_args()

def load_checkpoint(model, ckpt_path, device):
    print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model