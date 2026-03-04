def load_rgb(img_fp: Path):
    rgb_u8 = np.array(Image.open(img_fp).convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0
    return rgb_u8, x.unsqueeze(0)  # (1,3,H,W)

def load_gt_mask(msk_fp: Path):
    m = np.array(Image.open(msk_fp), dtype=np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def save_pred_mask(pred: np.ndarray, out_fp: Path):
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(out_fp)


def plot_triplets(img_dir, msk_dir, pred_dir, tile_names, out_png=None, title=None):
    cmap_mask = ListedColormap(CLASS_COLORS)
    norm_mask = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_mask.N)

    rows = len(tile_names)
    fig, axs = plt.subplots(rows, 3, figsize=(16, 4 * rows))

    for i, name in enumerate(tile_names):
        img_fp = img_dir / name
        gt_fp = msk_dir / f"{Path(name).stem}_mask.png"
        pred_fp = pred_dir / f"{Path(name).stem}_pred.png"

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
    cbar = fig.colorbar(im, cax=cax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(CLASS_NAMES)

    if title:
        fig.suptitle(title, fontsize=14)
        plt.subplots_adjust(top=0.95)

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print("Saved visualization:", out_png)

    plt.show()
