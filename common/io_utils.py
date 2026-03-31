"""I/O helpers used by inference and validation scripts."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def load_rgb(img_fp: Path) -> Tuple[np.ndarray, torch.Tensor]:
    """Load an RGB image as both numpy and model-ready torch tensor.

    Args:
        img_fp (Path): Path to an RGB image file.

    Returns:
        Tuple[np.ndarray, torch.Tensor]:
            - RGB image as ``uint8`` numpy array with shape ``(H, W, 3)``.
            - Float tensor with shape ``(1, 3, H, W)`` normalized to ``[0, 1]``.

    Usage:
        Use the tensor output as model input and the numpy output for visualization.
    """
    rgb_u8 = np.array(Image.open(img_fp).convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0
    return rgb_u8, x.unsqueeze(0)


def load_gt_mask(msk_fp: Path) -> np.ndarray:
    """Load a ground-truth mask as a 2D ``uint8`` class-index array.

    Args:
        msk_fp (Path): Path to a mask image file.

    Returns:
        np.ndarray: 2D ``uint8`` mask array with shape ``(H, W)``.

    Usage:
        This is typically used before metric computation or side-by-side plotting.
    """
    m = np.array(Image.open(msk_fp), dtype=np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def save_pred_mask(pred: np.ndarray, out_fp: Path) -> None:
    """Save a predicted mask image to disk.

    Args:
        pred (np.ndarray): Predicted mask array to write.
        out_fp (Path): Output file path.

    Returns:
        None: Writes the file as a side effect.
    """
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(out_fp)


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """Load model weights from a checkpoint and set evaluation mode.

    Args:
        model (torch.nn.Module): Model instance to populate.
        ckpt_path (Path): Checkpoint path produced by training.
        device (torch.device): Device used to deserialize checkpoint tensors.

    Returns:
        torch.nn.Module: The same model instance in ``eval`` mode.

    Usage:
        Call this once before inference loops to ensure deterministic eval behavior.
    """
    print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
