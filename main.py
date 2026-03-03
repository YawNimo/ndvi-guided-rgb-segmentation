# main.py
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# Only needed if you want DeepLab from torchvision
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models import ResNet50_Weights

# Only needed if you want UNet from segmentation_models_pytorch
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# -----------------------------
# Defaults (match your notebook)
# -----------------------------
DEFAULT_TILES_BASE = "/content/drive/MyDrive/ResearchProject (1)/dataset_tiles_512"
DEFAULT_RESULTS_DIR = "/content/drive/MyDrive/ResearchProject (1)/model_results"

# Your 4 locked visuals (current notebook)
DEFAULT_FIXED_TILES = [
    "935650_se_t0058.png",
    "930660_se_t0056.png",
    "935650_se_t0006.png",
    "930650_ne_t0019.png",
]

CLASS_NAMES = ["water (0)", "impervious (1)", "sparse veg (2)", "dense veg (3)"]
CLASS_COLORS = ["#1f77b4", "#ff7f0e", "#98df8a", "#2ca02c"]


# -----------------------------
# Model definitions
# -----------------------------
def build_unet(num_classes=4, encoder_name="resnet34", encoder_weights="imagenet"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    return model


def build_deeplab(num_classes=4):
    model = deeplabv3_resnet50(
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    )
    # Replace classifier head
    in_ch = model.classifier[-1].in_channels
    model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    # Aux head safety
    if model.aux_classifier is not None:
        in_ch_aux = model.aux_classifier[-1].in_channels
        model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, num_classes, kernel_size=1)

    return model


# ---- SPANetFull (paste your exact notebook classes here) ----
# IMPORTANT: keep this identical to your notebook implementation
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
        )
        self.layer1 = base.layer1
        self.layer2 = base.layer2  # low
        self.layer3 = base.layer3
        self.layer4 = base.layer4  # high

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        low = self.layer2(x)
        x = self.layer3(low)
        high = self.layer4(x)
        return low, high


class SPAMBlock(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 4), reduction=4):
        super().__init__()
        self.pool_sizes = pool_sizes
        mid_channels = max(in_channels // reduction, 1)

        self.conv_reduce = nn.Conv2d(in_channels * len(pool_sizes), mid_channels, kernel_size=1, bias=False)
        self.bn_reduce = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv_attn = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        pooled_list = []
        for ps in self.pool_sizes:
            p = F.adaptive_avg_pool2d(x, output_size=(ps, ps))
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            pooled_list.append(p)

        multi_scale = torch.cat(pooled_list, dim=1)
        h = self.relu(self.bn_reduce(self.conv_reduce(multi_scale)))
        attn = self.sigmoid(self.conv_attn(h))
        return x * attn


class FeatureFusionModule(nn.Module):
    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(low_channels // 4, 1)

        self.conv1 = nn.Conv2d(high_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, low_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, low, high):
        B, C_l, H_l, W_l = low.shape
        h = self.pool(high)
        h = self.relu(self.bn1(self.conv1(h)))
        h = self.sigmoid(self.conv2(h))
        attn = h.expand(-1, -1, H_l, W_l)
        fused = low * attn
        return fused


class SPANetFull(nn.Module):
    def __init__(self, num_classes=4, pretrained_backbone=True):
        super().__init__()
        self.encoder = ResNet50Encoder(pretrained=pretrained_backbone)

        self.low_ch = 512
        self.high_ch = 2048

        self.spam_low = SPAMBlock(self.low_ch)
        self.spam_high = SPAMBlock(self.high_ch)
        self.ffm = FeatureFusionModule(self.low_ch, self.high_ch)

        self.dec_conv1 = nn.Conv2d(self.low_ch, 256, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        low, high = self.encoder(x)

        low_enh = self.spam_low(low)
        high_enh = self.spam_high(high)

        fused_low = self.ffm(low_enh, high_enh)

        y = self.dec_bn1(F.relu(self.dec_conv1(fused_low)))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)

        y = self.dec_bn2(F.relu(self.dec_conv2(y)))
        y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        logits = self.classifier(y)
        return logits


def build_model(name: str, num_classes=4, unet_encoder="resnet34", unet_weights="imagenet"):
    name = name.lower()
    if name == "unet":
        return build_unet(num_classes=num_classes, encoder_name=unet_encoder, encoder_weights=unet_weights)
    if name == "deeplab":
        return build_deeplab(num_classes=num_classes)
    if name == "spanetfull":
        return SPANetFull(num_classes=num_classes, pretrained_backbone=True)
    raise ValueError(f"Unknown model '{name}'. Choose from: unet, deeplab, spanetfull")


# -----------------------------
# IO helpers
# -----------------------------
def load_rgb(img_fp: Path):
    rgb_u8 = np.array(Image.open(img_fp).convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(rgb_u8).permute(2, 0, 1).float() / 255.0
    return rgb_u8, x.unsqueeze(0)  # (1,3,H,W)

def load_gt_mask(msk_fp: Path):
    m = np.array(Image.open(msk_fp), dtype=np.uint8)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)

@torch.no_grad()
def predict(model, x, device):
    x = x.to(device, non_blocking=True)
    out = model(x)
    if isinstance(out, dict):
        out = out["out"]
    pred = torch.argmax(out, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred


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


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, required=True, choices=["unet", "deeplab", "spanetfull"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to *_best.pt checkpoint")
    p.add_argument("--tiles_base", type=str, default=DEFAULT_TILES_BASE)
    p.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)

    p.add_argument("--fixed", action="store_true", help="Run only fixed tiles visualization + preds")
    p.add_argument("--fixed_tiles", type=str, default="", help="Optional JSON list or comma-separated names")

    p.add_argument("--limit", type=int, default=0, help="Limit number of images if not fixed (0 = all)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # UNet options (only used if model=unet)
    p.add_argument("--unet_encoder", type=str, default="resnet34")
    p.add_argument("--unet_weights", type=str, default="imagenet")

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tiles_base = Path(args.tiles_base)
    img_dir = tiles_base / "images"
    msk_dir = tiles_base / "masks"

    results_dir = Path(args.results_dir)
    pred_dir = results_dir / args.model / "pred_masks"

    # Select tile list
    if args.fixed:
        if args.fixed_tiles.strip():
            s = args.fixed_tiles.strip()
            if s.startswith("["):
                tile_names = json.loads(s)
            else:
                tile_names = [x.strip() for x in s.split(",") if x.strip()]
        else:
            tile_names = DEFAULT_FIXED_TILES
    else:
        tile_names = sorted([p.name for p in img_dir.glob("*.png")])
        if args.limit and args.limit > 0:
            tile_names = tile_names[: args.limit]

    # Build + load model
    model = build_model(
        args.model,
        num_classes=4,
        unet_encoder=args.unet_encoder,
        unet_weights=args.unet_weights
    ).to(device)

    ckpt_path = Path(args.ckpt)
    print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Inference loop
    missing = []
    for name in tile_names:
        img_fp = img_dir / name
        if not img_fp.exists():
            missing.append(str(img_fp))
            continue

        rgb_u8, x = load_rgb(img_fp)
        pred = predict(model, x, device=device)

        out_fp = pred_dir / f"{Path(name).stem}_pred.png"
        save_pred_mask(pred, out_fp)

    if missing:
        print("⚠️ Missing images:")
        for m in missing[:10]:
            print(" -", m)
        if len(missing) > 10:
            print(f" ... and {len(missing)-10} more")

    print(f"Saved preds to: {pred_dir}")

    # Visualization (only meaningful for fixed tiles)
    if args.fixed:
        viz_fp = results_dir / args.model / "fixed_viz.png"
        plot_triplets(
            img_dir=img_dir,
            msk_dir=msk_dir,
            pred_dir=pred_dir,
            tile_names=tile_names,
            out_png=viz_fp,
            title=f"{args.model.upper()} | RGB vs GT vs Pred"
        )


if __name__ == "__main__":
    main()
