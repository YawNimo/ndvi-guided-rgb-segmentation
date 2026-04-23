"""Inference loop helpers for tile-by-tile prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from common.io_utils import load_rgb, save_pred_mask


def _window_starts(length: int, patch: int, stride: int) -> list[int]:
    """Row/column start indices so every pixel is covered; last window flush to the far edge."""
    if length <= patch:
        return [0]
    starts: list[int] = []
    pos = 0
    while pos + patch < length:
        starts.append(pos)
        pos += stride
    last = length - patch
    if not starts or starts[-1] != last:
        starts.append(last)
    # Deduplicate (can happen if length-patch was already included)
    out: list[int] = []
    for s in starts:
        if not out or s != out[-1]:
            out.append(s)
    return out


def _extract_padded_patch(rgb: np.ndarray, sy: int, sx: int, patch: int) -> np.ndarray:
    """Crop ``rgb`` at (sy, sx) and pad bottom/right to ``(patch, patch, 3)`` with reflect."""
    h, w = rgb.shape[0], rgb.shape[1]
    eh = min(patch, h - sy)
    ew = min(patch, w - sx)
    region = rgb[sy : sy + eh, sx : sx + ew, :]
    pad_h = patch - region.shape[0]
    pad_w = patch - region.shape[1]
    if pad_h or pad_w:
        region = np.pad(region, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return region


@torch.no_grad()
def __forward_logits(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return logits ``(1, C, h, w)`` on the inference device."""
    x = x.to(device, non_blocking=True)
    use_amp = device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        out = model(x)
    if isinstance(out, dict):
        out = out["out"]
    return out.float()


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
    logits = __forward_logits(model, x, device)
    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


def _predict_windowed(
    model: torch.nn.Module,
    rgb_u8: np.ndarray,
    device: torch.device,
    patch_size: int,
    overlap: int,
    num_classes: int,
) -> np.ndarray:
    """Sliding-window inference with mean logits over overlaps."""
    h, w = rgb_u8.shape[0], rgb_u8.shape[1]
    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("stride (patch_size - overlap) must be positive")

    starts_y = _window_starts(h, patch_size, stride)
    starts_x = _window_starts(w, patch_size, stride)

    acc = np.zeros((num_classes, h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)

    for sy in starts_y:
        for sx in starts_x:
            patch = _extract_padded_patch(rgb_u8, sy, sx, patch_size)
            x = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            logits = __forward_logits(model, x, device)
            logits_np = logits.squeeze(0).detach().cpu().numpy()

            eh = min(patch_size, h - sy)
            ew = min(patch_size, w - sx)
            sl = logits_np[:, :eh, :ew]
            acc[:, sy : sy + eh, sx : sx + ew] += sl
            cnt[sy : sy + eh, sx : sx + ew] += 1.0
            del logits, logits_np, x

    cnt_safe = np.maximum(cnt, 1e-6)
    fused = acc / cnt_safe[np.newaxis, :, :]
    return np.argmax(fused, axis=0).astype(np.uint8)


def inference_loop(
    tile_names,
    img_dir,
    pred_dir,
    model,
    device,
    inference_patch_size: int = 0,
    inference_overlap: int = 0,
    num_classes: int = 4,
):
    """Run inference for all requested tile names and save predicted masks.

    Args:
        tile_names (list[str]): Tile filenames to process.
        img_dir (Path): Directory containing RGB tiles.
        pred_dir (Path): Output directory for prediction TIFF files.
        model (torch.nn.Module): Segmentation model.
        device (torch.device): Inference device.
        inference_patch_size: If > 0, run sliding-window inference with this window size.
        inference_overlap: Overlap in pixels between windows (used when patch size > 0).
        num_classes: Number of segmentation classes (logit channels).

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
        if inference_patch_size > 0:
            pred = _predict_windowed(
                model,
                rgb_u8,
                device,
                patch_size=inference_patch_size,
                overlap=inference_overlap,
                num_classes=num_classes,
            )
        else:
            pred = __predict(model, x, device=device)

        out_fp = pred_dir / f"{Path(name).stem}.tif"
        save_pred_mask(pred, out_fp)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if missing:
        print("Missing images:")
        for m in missing[:10]:
            print(" -", m)
        if len(missing) > 10:
            print(f" ... and {len(missing)-10} more")

    print(f"Saved preds to: {pred_dir}")