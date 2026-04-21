"""Inference entrypoint for generating predicted masks from trained checkpoints."""

import sys
import torch
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.models import build_model
from utils import parse_args
from common.io_utils import load_checkpoint
from inference import inference_loop
from visualization_utils import plot_triplets
if __name__ == "__main__":    
    # Parse CLI arguments, run inference, and optionally render triplet plots.
    args = parse_args()
    device = torch.device(args.device)

    tiles_base = Path(args.tiles_base)
    img_dir = Path(args.images_dir) if args.images_dir else tiles_base / args.images_subdir
    msk_dir = Path(args.masks_dir) if args.masks_dir else tiles_base / args.masks_subdir

    results_dir = Path(args.results_dir)

    if args.pred_output_dir:
        pred_dir = Path(args.pred_output_dir)
    else:
        pred_dir = results_dir / args.model / args.pred_subdir
    
    ckpt_path = Path(args.ckpt)

    tile_names = sorted([p.name for p in img_dir.glob("*.tif")])
    if args.limit and args.limit > 0:
        tile_names = tile_names[: args.limit]

    # Build + load model
    model = build_model(
        args.model,
        num_classes=4,
        unet_encoder=args.unet_encoder,
        unet_weights=args.unet_weights
    ).to(device)

    load_checkpoint(model, ckpt_path, device)

    # Inference loop
    inference_loop(
        tile_names=tile_names,
        img_dir=img_dir,
        pred_dir=pred_dir,
        model=model,
        device=device
    )

    # Visualization (only works up to a max of 3 tiles)
    if args.make_viz and len(tile_names) <= 3:
        # Generate a fixed side-by-side preview when a small tile set is used.
        viz_fp = results_dir / args.model / "fixed_viz.png"
        plot_triplets(
            img_dir=img_dir,
            msk_dir=msk_dir,
            pred_dir=pred_dir,
            tile_names=tile_names,
            out_png=viz_fp,
            title=f"{args.model.upper()} | RGB vs GT vs Pred"
        )
