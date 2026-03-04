import torch
from pathlib import Path
from build_models import build_model
from utils import load_checkpoint, parse_args
from inference import inference_loop
from io_helpers import plot_triplets




if __name__ == "__main__":    
    args = parse_args()
    device = torch.device(args.device)

    tiles_base = Path(args.tiles_base)
    img_dir = tiles_base / "images"
    msk_dir = tiles_base / "masks"

    results_dir = Path(args.results_dir)
    pred_dir = results_dir / args.model / "pred_masks"
    
    ckpt_path = Path(args.ckpt)

    # Select tile list
    if args.fixed:
        s = args.fixed_tiles.strip()
        if s.startswith("["):
            tile_names = json.loads(s)
        else:
            tile_names = [x.strip() for x in s.split(",") if x.strip()]
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

    load_checkpoint(model, ckpt_path, device)

    # Inference loop
    inference_loop(
        tile_names=tile_names,
        img_dir=img_dir,
        pred_dir=pred_dir,
        model=model,
        device=device
    )

    # Visualization (only meaningful for fixed tiles)
    if args.fixed:
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
