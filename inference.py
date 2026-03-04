@torch.no_grad()
def __predict(model, x, device):
    x = x.to(device, non_blocking=True)
    out = model(x)
    if isinstance(out, dict):
        out = out["out"]
    pred = torch.argmax(out, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return pred

def inference_loop(tile_names, img_dir, pred_dir, model, device):
    missing = []
    for name in tile_names:
        img_fp = img_dir / name
        if not img_fp.exists():
            missing.append(str(img_fp))
            continue

        rgb_u8, x = load_rgb(img_fp)
        pred = __predict(model, x, device=device)

        out_fp = pred_dir / f"{Path(name).stem}_pred.png"
        save_pred_mask(pred, out_fp)

    if missing:
        print("Missing images:")
        for m in missing[:10]:
            print(" -", m)
        if len(missing) > 10:
            print(f" ... and {len(missing)-10} more")

    print(f"Saved preds to: {pred_dir}")