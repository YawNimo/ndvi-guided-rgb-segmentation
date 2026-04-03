"""
Visualization and plotting utilities for segmentation training.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from PIL import Image


# Class visualization colors
CLASS_NAMES = ["water", "impervious", "sparse_veg", "dense_veg"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#98df8a", "#2ca02c"]


def plot_loss(history, output_path=None, title="Loss vs Epoch"):
    """
    Plot train and validation loss curves.

    Parameters
    ----------
    history : list of dict
        Training history from train_model().
    output_path : Path or str, optional
        If provided, save plot to this path.
    title : str
        Plot title.
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", label="train loss", linewidth=2)
    ax.plot(epochs, val_loss, marker="s", label="val loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Loss plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_metrics(history, output_path=None, metric_key="val_f1_macro"):
    """
    Plot a metric over epochs (e.g., F1, IoU).

    Parameters
    ----------
    history : list of dict
        Training history from train_model().
    output_path : Path or str, optional
        If provided, save plot to this path.
    metric_key : str
        Key from history dict to plot (e.g., 'val_f1_macro', 'val_iou_macro').
    """
    if not history or metric_key not in history[0]:
        print(f"Warning: metric '{metric_key}' not found in history")
        return

    epochs = [h["epoch"] for h in history]
    values = [h[metric_key] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, values, marker="o", linewidth=2, markersize=8, color="green")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric_key.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"{metric_key.replace('_', ' ').title()} vs Epoch", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Metric plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


@torch.no_grad()
def predict_tile(model, rgb_path, device):
    """
    Run inference on a single RGB tile.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model.
    rgb_path : Path or str
        Path to RGB tile image.
    device : str
        Device to run inference on.

    Returns
    -------
    np.ndarray
        Predicted mask (H, W) as uint8.
    """
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device, non_blocking=True)

    model.eval()
    logits = model(x)
    if isinstance(logits, dict):
        logits = logits["out"]

    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def visualize_predictions(
    model,
    model_name,
    img_dir,
    msk_dir,
    num_samples=4,
    device="cuda",
    output_path=None,
):
    """
    Visualize predictions on random tiles: RGB | Ground Truth | Prediction.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model.
    model_name : str
        Model name for title.
    img_dir : Path or str
        Directory containing input RGB tiles.
    msk_dir : Path or str
        Directory containing ground truth masks.
    num_samples : int
        Number of random tiles to visualize.
    device : str
        Device to run inference on.
    output_path : Path or str, optional
        If provided, save plot to this path.
    """
    img_dir = Path(img_dir)
    msk_dir = Path(msk_dir)

    image_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == ".tif"])
    if not image_files:
        print(f"No .tif files found in {img_dir}")
        return

    if num_samples > len(image_files):
        num_samples = len(image_files)

    selected_indices = np.random.choice(len(image_files), num_samples, replace=False)
    selected_files = [image_files[i] for i in selected_indices]

    fig, axs = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, img_path in enumerate(selected_files):
        # Try common mask naming conventions
        msk_path = None
        for candidate in [msk_dir / f"{img_path.stem}_mask.tif", msk_dir / f"{img_path.stem}.tif"]:
            if candidate.exists():
                msk_path = candidate
                break

        if msk_path is None:
            print(f"Skipping {img_path.name}: no matching mask found")
            continue

        # Load RGB tile
        rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # Load ground truth mask
        gt = np.array(Image.open(msk_path), dtype=np.uint8)
        if gt.ndim == 3:
            gt = gt[..., 0]

        # Predict
        pred = predict_tile(model, img_path, device)

        # Plot
        axs[i, 0].imshow(rgb)
        axs[i, 0].set_title(f"RGB: {img_path.name}")
        axs[i, 0].axis("off")

        classes_cmap = ListedColormap(CLASS_COLORS)
        axs[i, 1].imshow(gt, cmap=classes_cmap, vmin=0, vmax=3)
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred, cmap=classes_cmap, vmin=0, vmax=3)
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")

    fig.suptitle(f"Model: {model_name}", fontsize=14, y=0.995)
    fig.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"Predictions plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def print_metrics_summary(metrics_json_path):
    """
    Print a formatted summary of best metrics from JSON.

    Parameters
    ----------
    metrics_json_path : Path or str
        Path to best_metrics.json from training.
    """
    metrics_json_path = Path(metrics_json_path)
    if not metrics_json_path.exists():
        print(f"Metrics file not found: {metrics_json_path}")
        return

    with open(metrics_json_path, "r") as f:
        data = json.load(f)

    best = data.get("best", {})
    epoch = best.get("epoch", "?")
    val_f1_macro = best.get("val_f1_macro", "?")
    val_iou_macro = best.get("val_iou_macro", "?")
    val_pixel_acc = best.get("val_pixel_acc", "?")

    f1_per_class = best.get("val_f1_per_class", [])
    iou_per_class = best.get("val_iou_per_class", [])

    print("\n" + "=" * 60)
    print("BEST METRICS SUMMARY")
    print("=" * 60)
    print(f"Best Epoch: {epoch}")
    print(f"  Macro F1: {val_f1_macro:.4f}")
    print(f"  Macro IoU: {val_iou_macro:.4f}")
    print(f"  Pixel Accuracy: {val_pixel_acc:.4f}")

    if f1_per_class:
        print("\nPer-Class F1 Scores:")
        for i, (name, f1) in enumerate(zip(CLASS_NAMES, f1_per_class)):
            print(f"  {name:>12s}: {f1:.4f}")

    if iou_per_class:
        print("\nPer-Class IoU Scores:")
        for i, (name, iou) in enumerate(zip(CLASS_NAMES, iou_per_class)):
            print(f"  {name:>12s}: {iou:.4f}")

    print("=" * 60 + "\n")
