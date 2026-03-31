"""Visualization helpers for side-by-side RGB, ground truth, and prediction panels."""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from common.constants import CLASS_COLORS, CLASS_NAMES
from pathlib import Path
import numpy as np
from PIL import Image
from common.io_utils import load_gt_mask

def plot_triplets(img_dir, msk_dir, pred_dir, tile_names, out_png=None, title=None, show=False):
    """Plot RGB, ground-truth, and prediction triplets for a list of tiles.

    Args:
        img_dir (Path): Directory with RGB tiles.
        msk_dir (Path): Directory with ground-truth masks.
        pred_dir (Path): Directory with predicted masks.
        tile_names (list[str]): Tile filenames to visualize.
        out_png (Path | None): Optional output image path.
        title (str | None): Optional plot title.
        show (bool): Whether to display interactively.

    Returns:
        None: Renders and optionally saves the figure.
    """
    cmap_mask = ListedColormap(CLASS_COLORS)
    norm_mask = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_mask.N)

    rows = len(tile_names)
    fig, axs = plt.subplots(rows, 3, figsize=(16, 4 * rows), squeeze=False)

    for i, name in enumerate(tile_names):
        img_fp = img_dir / name
        gt_fp = msk_dir / f"{Path(name).stem}.tif"
        pred_fp = pred_dir / f"{Path(name).stem}.tif"

        rgb = np.array(Image.open(img_fp).convert("RGB"), dtype=np.uint8)
        gt = load_gt_mask(gt_fp) if gt_fp.exists() else None
        pr = np.array(Image.open(pred_fp), dtype=np.uint8) if pred_fp.exists() else None

        axs[i, 0].imshow(rgb)
        axs[i, 0].set_title(name)
        axs[i, 0].axis("off")

        if gt is not None:
            axs[i, 1].imshow(gt, cmap=cmap_mask, norm=norm_mask)
            axs[i, 1].set_title("Ground Truth")
        else:
            axs[i, 1].text(0.5, 0.5, "GT not found", ha="center", va="center")
            axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        if pr is not None:
            im = axs[i, 2].imshow(pr, cmap=cmap_mask, norm=norm_mask)
            axs[i, 2].set_title("Prediction")
        else:
            axs[i, 2].text(0.5, 0.5, "Pred not found", ha="center", va="center")
            axs[i, 2].set_title("Prediction")
        axs[i, 2].axis("off")

    fig.subplots_adjust(right=0.90, wspace=0.05, hspace=0.25)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = plt.cm.ScalarMappable(cmap=cmap_mask, norm=norm_mask)
    cbar = fig.colorbar(mappable, cax=cax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(CLASS_NAMES)

    if title:
        fig.suptitle(title, fontsize=14)

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print("Saved visualization:", out_png)

    if show:
        plt.show()
