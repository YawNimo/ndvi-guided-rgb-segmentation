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
from common.constants import INPUT_IMAGES_SUBDIR, INPUT_MASKS_SUBDIR, OUTPUT_PRED_MASKS_SUBDIR




if __name__ == "__main__":    
    args = parse_args()
    device = torch.device(args.device)

    tiles_base = Path(args.tiles_base)
    img_dir = tiles_base / INPUT_IMAGES_SUBDIR
    msk_dir = tiles_base / INPUT_MASKS_SUBDIR

    results_dir = Path(args.results_dir)
    
    
    pred_dir = results_dir / args.model / OUTPUT_PRED_MASKS_SUBDIR
    
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
        # TODO: move this to some visualization.py
        viz_fp = results_dir / args.model / "fixed_viz.png"
        plot_triplets(
            img_dir=img_dir,
            msk_dir=msk_dir,
            pred_dir=pred_dir,
            tile_names=tile_names,
            out_png=viz_fp,
            title=f"{args.model.upper()} | RGB vs GT vs Pred"
        )
