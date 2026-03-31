from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_rgb(img_fp: Path):
    rgb_u8 = np.array(Image.open(img_fp).convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0
    return rgb_u8, x.unsqueeze(0)


def load_gt_mask(msk_fp: Path):
    m = np.array(Image.open(msk_fp), dtype=np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def save_pred_mask(pred: np.ndarray, out_fp: Path):
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(out_fp)


def load_checkpoint(model, ckpt_path, device):
    print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
