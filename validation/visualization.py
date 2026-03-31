from __future__ import annotations

"""Plotting utilities for validation metrics and qualitative samples."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from PIL import Image


def plot_triplets(
    out_png: Path,
    image_paths: list[Path],
    gt_masks: list[np.ndarray],
    pred_masks: list[np.ndarray],
    class_names: list[str],
    class_colors: list[str],
    title: str,
) -> None:
    """Save RGB/GT/prediction triplet panels for selected tiles."""
    if not image_paths:
        return

    cmap_mask = ListedColormap(class_colors)
    boundaries = np.arange(-0.5, len(class_names) + 0.5, 1.0)
    norm_mask = BoundaryNorm(boundaries, cmap_mask.N)

    rows = len(image_paths)
    fig, axs = plt.subplots(rows, 3, figsize=(16, max(4, 4 * rows)), squeeze=False)

    for i, img_path in enumerate(image_paths):
        rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        axs[i, 0].imshow(rgb)
        axs[i, 0].set_title(img_path.name)
        axs[i, 0].axis("off")

        axs[i, 1].imshow(gt_masks[i], cmap=cmap_mask, norm=norm_mask)
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred_masks[i], cmap=cmap_mask, norm=norm_mask)
        axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")

    fig.subplots_adjust(right=0.90, wspace=0.05, hspace=0.25)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = plt.cm.ScalarMappable(cmap=cmap_mask, norm=norm_mask)
    cbar = fig.colorbar(mappable, cax=cax, ticks=list(range(len(class_names))))
    cbar.set_ticklabels(class_names)

    fig.suptitle(title, fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dice_histogram(out_png: Path, macro_dice_scores: list[float]) -> None:
    """Save histogram plot for per-tile macro Dice scores."""
    if not macro_dice_scores:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(macro_dice_scores, bins=20, color="#4C78A8", edgecolor="black", alpha=0.85)
    ax.set_title("Per-tile Macro Dice Distribution")
    ax.set_xlabel("Macro Dice")
    ax.set_ylabel("Tile Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    out_png: Path,
    conf_mat: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
) -> None:
    """Save confusion matrix heatmap, optionally row-normalized."""
    data = conf_mat.astype(np.float64)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        data = data / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="Blues")
    ax.set_title("Confusion Matrix" + (" (Row-Normalized)" if normalize else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    threshold = data.max() * 0.6 if data.size > 0 else 0.0
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            text_val = f"{data[r, c]:.2f}" if normalize else str(int(conf_mat[r, c]))
            ax.text(c, r, text_val, ha="center", va="center", color=("white" if data[r, c] > threshold else "black"))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_dice(
    out_png: Path,
    class_names: list[str],
    class_dice_scores: list[float],
) -> None:
    """Save bar chart of mean Dice scores per class."""
    if not class_dice_scores:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(class_names))
    bars = ax.bar(x, class_dice_scores, color=["#1f77b4", "#ff7f0e", "#98df8a", "#2ca02c"][: len(class_names)])

    ax.set_ylim(0.0, 1.0)
    ax.set_title("Mean Dice by Class")
    ax.set_ylabel("Dice")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    for bar, score in zip(bars, class_dice_scores):
        ax.text(bar.get_x() + bar.get_width() / 2.0, min(score + 0.02, 0.99), f"{score:.3f}", ha="center", va="bottom", fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
