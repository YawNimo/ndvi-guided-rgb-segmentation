"""Inference loop helpers for tile-by-tile prediction."""

import torch
from common.io_utils import load_rgb, save_pred_mask
import numpy as np
from pathlib import Path

@torch.no_grad()
def __predict(model, x, device):
    """Run one forward pass and return argmax class predictions.

    Args:
        model (torch.nn.Module): Segmentation model in eval mode.
        x (torch.Tensor): Input tensor with shape ``(1, 3, H, W)``.
        device (torch.device): Inference device.

    Returns:
        np.ndarray: Predicted class mask as ``uint8`` array with shape ``(H, W)``.
    """
    x = x.to(device, non_blocking=True)
    out = model(x)
    if isinstance(out, dict):
        out = out["out"]
    pred = torch.argmax(out, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred

def inference_loop(tile_names, img_dir, pred_dir, model, device):
    """Run inference for all requested tile names and save predicted masks.

    Args:
        tile_names (list[str]): Tile filenames to process.
        img_dir (Path): Directory containing RGB tiles.
        pred_dir (Path): Output directory for prediction TIFF files.
        model (torch.nn.Module): Segmentation model.
        device (torch.device): Inference device.

    Returns:
        None: Writes predicted masks and prints missing-input warnings.
    """
    missing = []
    for name in tile_names:
        img_fp = img_dir / name
        if not img_fp.exists():
            missing.append(str(img_fp))
            continue

        rgb_u8, x = load_rgb(img_fp)
        pred = __predict(model, x, device=device)

        out_fp = pred_dir / f"{Path(name).stem}.tif"
        save_pred_mask(pred, out_fp)

    if missing:
        print("Missing images:")
        for m in missing[:10]:
            print(" -", m)
        if len(missing) > 10:
            print(f" ... and {len(missing)-10} more")

    print(f"Saved preds to: {pred_dir}")