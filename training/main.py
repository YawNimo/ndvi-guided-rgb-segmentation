"""
Training-only entrypoint for NDVI-guided RGB segmentation.
"""

import json
import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from plots import plot_loss, plot_metrics, visualize_predictions, print_metrics_summary


# Allow importing from src/ when running this file directly.
# todo refactor to use a proper package structure and avoid this hack.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from build_models import build_model


# ==============================================================================
# Configuration
# ==============================================================================

SEED = 42

NUM_CLASSES = 4
CLASS_NAMES = ["water", "impervious", "sparse_veg", "dense_veg"]

TILES_BASE = ROOT_DIR / "input"
IMG_DIR = TILES_BASE / "images"
MSK_DIR = TILES_BASE / "masks"
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
EARLY_STOP_PATIENCE = 5

LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3

NUM_WORKERS = 2
PIN_MEMORY = True
USE_AMP = True
GRAD_ACCUM_STEPS = 1
FREEZE_BN = False
PLOT_LOSS = False
PLOT_METRICS = False
PLOT_PREDICTIONS = False

MODEL_NAME = "unet"  # one of: unet, deeplab, spanetfull
SUPPORTED_IMAGE_EXTS = (".tif",)  # in case pngs want to be supported later
MAX_IMAGES = None


def parse_args():
	"""
	Parse command-line arguments to override default configuration.
	
	An example is in notes.txt
	"""
	parser = argparse.ArgumentParser(
		description="Train semantic segmentation models on pre-tiled RGB/mask PNG pairs."
	)
	parser.add_argument("--model", type=str, default=MODEL_NAME, choices=["unet", "deeplab", "spanetfull"])

	parser.add_argument("--epochs", type=int, default=EPOCHS)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
	parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
	parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)

	parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
	parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)

	parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)  # number of cpu workers for data loading
	parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=PIN_MEMORY)  # whether to use pinned memory for data loading
	parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=USE_AMP)  # whether to use automatic mixed precision (needed for training on GPU with limited memory)
	parser.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)  # number of steps to accumulate gradients for before updating model weights (helps with GPU memory issues)
	parser.add_argument("--freeze-bn", action="store_true", help="Freeze BatchNorm layers during training")  # can help with small batch sizes and reduce GPU memory usage
	parser.add_argument("--plot-loss", action="store_true", help="Plot train/val loss after training")
	parser.add_argument("--plot-metrics", action="store_true", help="Plot validation metrics after training")
	parser.add_argument("--plot-predictions", action="store_true", help="Visualize predictions on random samples")

	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument("--max-images", type=int, default=MAX_IMAGES)

	parser.add_argument("--img-dir", type=Path, default=IMG_DIR)
	parser.add_argument("--msk-dir", type=Path, default=MSK_DIR)
	parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
	parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)

	args = parser.parse_args()

	if args.train_ratio <= 0 or args.val_ratio <= 0:
		raise ValueError("--train-ratio and --val-ratio must be > 0")
	if args.train_ratio + args.val_ratio >= 1.0:
		raise ValueError("--train-ratio + --val-ratio must be < 1.0")
	if args.epochs <= 0:
		raise ValueError("--epochs must be > 0")
	if args.batch_size <= 0:
		raise ValueError("--batch-size must be > 0")
	if args.learning_rate <= 0:
		raise ValueError("--learning-rate must be > 0")
	if args.grad_accum_steps <= 0:
		raise ValueError("--grad-accum-steps must be > 0")
	if args.max_images is not None and args.max_images <= 0:
		raise ValueError("--max-images must be > 0 when provided")

	return args


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	os.environ["PYTHONHASHSEED"] = str(seed)


def _simple_train_augment(image: torch.Tensor, mask: torch.Tensor):
	"""Lightweight geometric augmentation without external dependencies."""
	# todo better data augmentation pipeline with albumentations or similar library
	if random.random() < 0.5:
		image = torch.flip(image, dims=[2])
		mask = torch.flip(mask, dims=[1])

	if random.random() < 0.5:
		image = torch.flip(image, dims=[1])
		mask = torch.flip(mask, dims=[0])

	k = random.randint(0, 3)
	if k:
		image = torch.rot90(image, k, dims=[1, 2])
		mask = torch.rot90(mask, k, dims=[0, 1])

	return image, mask


class TileSegDataset(Dataset):
	"""Dataset for paired RGB image tiles and segmentation masks.

	It indexes precomputed (image_path, mask_path) pairs, converts them to
	tensors, and applies lightweight geometric augmentation when train=True.
	"""

	def __init__(self, pairs, indices, train=False):
		self.pairs = pairs
		self.indices = indices
		self.train = train

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, i):
		img_fp, msk_fp = self.pairs[self.indices[i]]

		image = np.array(Image.open(img_fp).convert("RGB"), dtype=np.uint8)
		mask = np.array(Image.open(msk_fp), dtype=np.uint8)
		if mask.ndim == 3:
			mask = mask[..., 0]

		image = TF.to_tensor(image)
		mask = torch.from_numpy(mask).long()

		if self.train:
			image, mask = _simple_train_augment(image, mask)

		return image, mask


def _list_supported_images(directory: Path):
	"""List image files in a directory with supported extensions."""
	files = [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() == ".tif"]
	return files


def _mask_stem_candidates(image_stem: str):
	# given `image_001.tif`, the mask names accepted would be `image_001_mask.tif` or `image_001.tif`
	return [f"{image_stem}_mask", image_stem]


def _build_mask_lookup(mask_files):
	lookup = {}
	for m in mask_files:
		stem = m.stem
		lookup.setdefault(stem, m)
		if stem.endswith("_mask"):
			lookup.setdefault(stem[:-5], m)
	return lookup


def create_data_split():
	assert IMG_DIR.exists(), f"Missing image dir: {IMG_DIR}"
	assert MSK_DIR.exists(), f"Missing mask dir: {MSK_DIR}"

	img_files = _list_supported_images(IMG_DIR)
	if MAX_IMAGES is not None:
		img_files = img_files[:MAX_IMAGES]
	mask_files = _list_supported_images(MSK_DIR)
	mask_lookup = _build_mask_lookup(mask_files)
	pairs = []
	missing = 0

	for img_fp in img_files:
		msk_fp = None
		for cand in _mask_stem_candidates(img_fp.stem):
			if cand in mask_lookup:
				msk_fp = mask_lookup[cand]
				break

		if msk_fp is None:
			missing += 1
			continue
		pairs.append((img_fp, msk_fp))

	if len(pairs) == 0:
		raise ValueError(
			"No image/mask pairs found. Check filename conventions and extensions in "
			f"{IMG_DIR} and {MSK_DIR}. Supported extensions: {SUPPORTED_IMAGE_EXTS}"
		)

	random.seed(SEED)
	idx = list(range(len(pairs)))
	random.shuffle(idx)

	n = len(idx)
	train_end = int(TRAIN_RATIO * n)
	val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

	train_idx = idx[:train_end]
	val_idx = idx[train_end:val_end]
	test_idx = idx[val_end:]

	if len(train_idx) == 0:
		raise ValueError(
			"Train split is empty after pairing/splitting. Increase data size or adjust "
			"--train-ratio/--val-ratio."
		)

	print(f"Image tiles found: {len(img_files)}")
	print(f"Mask files found: {len(mask_files)}")
	print(f"Max images cap: {MAX_IMAGES}")
	print(f"Paired tiles: {len(pairs)}")
	print(f"Missing masks: {missing}")
	print(f"Split -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

	return pairs, train_idx, val_idx, test_idx


def create_dataloaders(pairs, train_idx, val_idx, test_idx):
	train_ds = TileSegDataset(pairs, train_idx, train=True)
	val_ds = TileSegDataset(pairs, val_idx, train=False)
	test_ds = TileSegDataset(pairs, test_idx, train=False)

	train_loader = DataLoader(
		train_ds,
		batch_size=BATCH_SIZE,
		shuffle=True,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=PIN_MEMORY,
	)

	return train_loader, val_loader, test_loader


def compute_class_weights():
	mask_files = _list_supported_images(MSK_DIR)
	assert mask_files, f"No mask files found in {MSK_DIR}"

	counts = np.zeros(NUM_CLASSES, dtype=np.float64)
	for msk_path in mask_files:
		mask = np.array(Image.open(msk_path), dtype=np.uint8)
		if mask.ndim == 3:
			mask = mask[..., 0]
		uniq, cts = np.unique(mask, return_counts=True)
		for u, c in zip(uniq, cts):
			if 0 <= int(u) < NUM_CLASSES:
				counts[int(u)] += float(c)

	total = counts.sum()
	freq = counts / max(total, 1.0)
	inv = 1.0 / (freq + 1e-6)
	weights = np.sqrt(inv)
	weights = weights / weights.min()

	print("Class weights:")
	for i, w in enumerate(weights.tolist()):
		print(f"  {CLASS_NAMES[i]}: {w:.4f}")

	return torch.tensor(weights, dtype=torch.float32)


def dice_loss(logits, targets, num_classes=NUM_CLASSES, eps=1e-6):
	probs = F.softmax(logits, dim=1)
	targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

	dims = (0, 2, 3)
	intersection = torch.sum(probs * targets_oh, dims)
	union = torch.sum(probs + targets_oh, dims)

	dice = (2 * intersection + eps) / (union + eps)
	return 1 - dice.mean()


def make_combined_loss(class_weights_tensor, device):
	ce_loss = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

	def combined_loss(logits, targets):
		return ce_loss(logits, targets) + dice_loss(logits, targets)

	return combined_loss


def _safe_div(num, den):
	return num / den if den > 0 else 0.0


def freeze_batchnorm_layers(model):
	for m in model.modules():
		if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
			m.eval()
			for p in m.parameters():
				p.requires_grad = False


@torch.no_grad()
def compute_segmentation_metrics(conf_mat: torch.Tensor):
	tp = conf_mat.diag().float()
	pred_sum = conf_mat.sum(dim=0).float()
	true_sum = conf_mat.sum(dim=1).float()

	precision = torch.tensor([_safe_div(tp[i].item(), pred_sum[i].item()) for i in range(NUM_CLASSES)])
	recall = torch.tensor([_safe_div(tp[i].item(), true_sum[i].item()) for i in range(NUM_CLASSES)])
	f1 = torch.tensor([
		_safe_div(2.0 * precision[i].item() * recall[i].item(), precision[i].item() + recall[i].item())
		for i in range(NUM_CLASSES)
	])
	iou = torch.tensor([
		_safe_div(tp[i].item(), true_sum[i].item() + pred_sum[i].item() - tp[i].item())
		for i in range(NUM_CLASSES)
	])

	pixel_acc = _safe_div(tp.sum().item(), conf_mat.sum().item())

	return {
		"f1_macro": float(f1.mean().item()),
		"iou_macro": float(iou.mean().item()),
		"pixel_acc": float(pixel_acc),
		"f1_per_class": f1.tolist(),
		"iou_per_class": iou.tolist(),
		"prec_per_class": precision.tolist(),
		"rec_per_class": recall.tolist(),
	}


def _update_confusion_matrix(conf_mat, preds, targets, num_classes=NUM_CLASSES):
	with torch.no_grad():
		mask = (targets >= 0) & (targets < num_classes)
		t = targets[mask].view(-1)
		p = preds[mask].view(-1)
		bins = num_classes * t + p
		hist = torch.bincount(bins, minlength=num_classes ** 2)
		conf_mat += hist.reshape(num_classes, num_classes)


def run_one_epoch(
	model,
	loader,
	optimizer,
	combined_loss_fn,
	device,
	train=True,
	use_amp=True,
	grad_accum_steps=1,
	freeze_bn=False,
	epoch=1,
	total_epochs=1,
	model_name="model",
):
	if train:
		model.train()
		if freeze_bn:
			freeze_batchnorm_layers(model)
	else:
		model.eval()

	total_loss = 0.0
	n_batches = 0
	conf_mat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
	amp_enabled = bool(use_amp and device == "cuda")
	scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

	start = time.perf_counter()
	if train:
		optimizer.zero_grad(set_to_none=True)

	phase = "train" if train else "val"
	bar_desc = f"[{model_name}] Epoch {epoch:02d}/{total_epochs} {phase}"
	iterator = tqdm(loader, desc=bar_desc, leave=False)

	for x, y in iterator:
		x = x.to(device, non_blocking=True)
		y = y.to(device, non_blocking=True)

		with torch.set_grad_enabled(train):
			with torch.cuda.amp.autocast(enabled=amp_enabled):
				logits = model(x)
				if isinstance(logits, dict):
					logits = logits["out"]
				loss = combined_loss_fn(logits, y)

			if train:
				scaled_loss = loss / grad_accum_steps
				if amp_enabled:
					scaler.scale(scaled_loss).backward()
				else:
					scaled_loss.backward()

				if (n_batches + 1) % grad_accum_steps == 0:
					if amp_enabled:
						scaler.step(optimizer)
						scaler.update()
					else:
						optimizer.step()
					optimizer.zero_grad(set_to_none=True)

		total_loss += loss.item()
		n_batches += 1
		iterator.set_postfix(loss=f"{(total_loss / max(1, n_batches)):.4f}")

		preds = torch.argmax(logits, dim=1)
		_update_confusion_matrix(conf_mat, preds, y)

	if train and (n_batches % grad_accum_steps != 0):
		if amp_enabled:
			scaler.step(optimizer)
			scaler.update()
		else:
			optimizer.step()
		optimizer.zero_grad(set_to_none=True)

	elapsed = time.perf_counter() - start
	avg_loss = total_loss / max(1, n_batches)
	metrics = compute_segmentation_metrics(conf_mat)

	return avg_loss, metrics, elapsed


def train_model(
	model,
	train_loader,
	val_loader,
	combined_loss_fn,
	model_name,
	device,
	lr=LEARNING_RATE,
	weight_decay=WEIGHT_DECAY,
	epochs=EPOCHS,
	use_amp=USE_AMP,
	grad_accum_steps=GRAD_ACCUM_STEPS,
	freeze_bn=FREEZE_BN,
):
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	scheduler = ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=LR_SCHEDULER_FACTOR,
		patience=LR_SCHEDULER_PATIENCE,
	)

	history = []
	best = {
		"epoch": None,
		"val_f1_macro": -1.0,
		"path_ckpt": None,
		"path_metrics": None,
	}
	epochs_no_improve = 0
	epoch_bar = tqdm(range(1, epochs + 1), desc=f"[{model_name}] epochs", position=0)

	for epoch in epoch_bar:
		tr_loss, tr_metrics, tr_time = run_one_epoch(
			model,
			train_loader,
			optimizer,
			combined_loss_fn,
			device,
			train=True,
			use_amp=use_amp,
			grad_accum_steps=grad_accum_steps,
			freeze_bn=freeze_bn,
			epoch=epoch,
			total_epochs=epochs,
			model_name=model_name,
		)
		va_loss, va_metrics, va_time = run_one_epoch(
			model,
			val_loader,
			optimizer,
			combined_loss_fn,
			device,
			train=False,
			use_amp=use_amp,
			grad_accum_steps=grad_accum_steps,
			freeze_bn=freeze_bn,
			epoch=epoch,
			total_epochs=epochs,
			model_name=model_name,
		)

		scheduler.step(va_loss)
		current_lr = optimizer.param_groups[0]["lr"]
		val_f1 = float(va_metrics["f1_macro"])
		epoch_bar.set_postfix(val_f1=f"{val_f1:.4f}", lr=f"{current_lr:.1e}")

		row = {
			"epoch": epoch,
			"train_loss": tr_loss,
			"val_loss": va_loss,
			"train_time_sec": tr_time,
			"val_time_sec": va_time,
			"learning_rate": current_lr,
			"train_f1_macro": float(tr_metrics["f1_macro"]),
			"val_f1_macro": val_f1,
			"train_iou_macro": float(tr_metrics["iou_macro"]),
			"val_iou_macro": float(va_metrics["iou_macro"]),
			"val_pixel_acc": float(va_metrics["pixel_acc"]),
			"val_f1_per_class": va_metrics["f1_per_class"],
			"val_iou_per_class": va_metrics["iou_per_class"],
			"val_prec_per_class": va_metrics["prec_per_class"],
			"val_rec_per_class": va_metrics["rec_per_class"],
		}
		history.append(row)

		print(
			f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
			f"val_f1={val_f1:.4f} val_iou={row['val_iou_macro']:.4f} "
			f"loss={va_loss:.4f} lr={current_lr:.1e} | "
			f"time={tr_time:.1f}s/{va_time:.1f}s"
		)

		if val_f1 > best["val_f1_macro"]:
			best["val_f1_macro"] = val_f1
			best["epoch"] = epoch
			epochs_no_improve = 0

			ckpt_path = CHECKPOINT_DIR / f"{model_name}_best.pt"
			metrics_path = RESULTS_DIR / f"{model_name}_best_metrics.json"

			torch.save(model.state_dict(), ckpt_path)
			with open(metrics_path, "w", encoding="utf-8") as f:
				json.dump({"best": row, "history": history}, f, indent=2)

			best["path_ckpt"] = str(ckpt_path)
			best["path_metrics"] = str(metrics_path)
			print(f"  -> New best! Saved checkpoint to {ckpt_path.name}")
		else:
			epochs_no_improve += 1
			if epochs_no_improve >= EARLY_STOP_PATIENCE:
				print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} epochs.")
				break

	print(f"\nBest epoch: {best['epoch']} | Best val F1 (macro): {best['val_f1_macro']:.4f}")
	print(f"Checkpoint: {best['path_ckpt']}")
	return best, history


def print_model_info(model, name):
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model: {name}")
	print(f"Parameters: {total:,} total, {trainable:,} trainable")


def main():
	global SEED, MODEL_NAME
	global EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_PATIENCE
	global TRAIN_RATIO, VAL_RATIO
	global NUM_WORKERS, PIN_MEMORY
	global USE_AMP, GRAD_ACCUM_STEPS
	global FREEZE_BN
	global PLOT_LOSS, PLOT_METRICS, PLOT_PREDICTIONS
	global MAX_IMAGES
	global IMG_DIR, MSK_DIR, RESULTS_DIR, CHECKPOINT_DIR

	args = parse_args()

	SEED = args.seed
	MODEL_NAME = args.model

	EPOCHS = args.epochs
	BATCH_SIZE = args.batch_size
	LEARNING_RATE = args.learning_rate
	WEIGHT_DECAY = args.weight_decay
	EARLY_STOP_PATIENCE = args.early_stop_patience

	TRAIN_RATIO = args.train_ratio
	VAL_RATIO = args.val_ratio

	NUM_WORKERS = args.num_workers
	PIN_MEMORY = args.pin_memory
	USE_AMP = args.amp
	GRAD_ACCUM_STEPS = args.grad_accum_steps
	FREEZE_BN = args.freeze_bn
	PLOT_LOSS = args.plot_loss
	PLOT_METRICS = args.plot_metrics
	PLOT_PREDICTIONS = args.plot_predictions
	MAX_IMAGES = args.max_images

	IMG_DIR = args.img_dir
	MSK_DIR = args.msk_dir
	RESULTS_DIR = args.results_dir
	CHECKPOINT_DIR = args.checkpoint_dir

	set_seed(SEED)
	if args.device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = args.device
		if device == "cuda" and not torch.cuda.is_available():
			raise RuntimeError("CUDA requested with --device cuda, but CUDA is not available")

	if device != "cuda":
		USE_AMP = False

	if BATCH_SIZE == 1 and not FREEZE_BN:
		FREEZE_BN = True
		print("Batch size is 1; auto-enabling BatchNorm freezing to avoid BN shape errors.")

	print(f"Device: {device}")
	if device == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")
	print(f"AMP enabled: {USE_AMP}")
	print(f"Gradient accumulation steps: {GRAD_ACCUM_STEPS}")
	print(f"Freeze BatchNorm: {FREEZE_BN}")

	pairs, train_idx, val_idx, test_idx = create_data_split()
	train_loader, val_loader, _ = create_dataloaders(pairs, train_idx, val_idx, test_idx)

	class_weights = compute_class_weights()
	combined_loss_fn = make_combined_loss(class_weights, device)

	model = build_model(MODEL_NAME, num_classes=NUM_CLASSES).to(device)
	print_model_info(model, MODEL_NAME)

	try:
		train_model(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			combined_loss_fn=combined_loss_fn,
			model_name=MODEL_NAME,
			device=device,
			use_amp=USE_AMP,
			grad_accum_steps=GRAD_ACCUM_STEPS,
			freeze_bn=FREEZE_BN,
		)

		print("\n" + "=" * 80)
		print("TRAINING COMPLETE")
		print("=" * 80)
		print_metrics_summary(RESULTS_DIR / f"{MODEL_NAME}_best_metrics.json")

		if PLOT_LOSS:
			print("Generating loss plot...")
			with open(RESULTS_DIR / f"{MODEL_NAME}_best_metrics.json") as f:
				hist_data = json.load(f)
			plot_loss(
				hist_data["history"],
				output_path=RESULTS_DIR / f"{MODEL_NAME}_loss_curve.png",
				title=f"{MODEL_NAME}: Loss vs Epoch",
			)

		if PLOT_METRICS:
			print("Generating metric plots...")
			with open(RESULTS_DIR / f"{MODEL_NAME}_best_metrics.json") as f:
				hist_data = json.load(f)
			plot_metrics(
				hist_data["history"],
				output_path=RESULTS_DIR / f"{MODEL_NAME}_f1_curve.png",
				metric_key="val_f1_macro",
			)
			plot_metrics(
				hist_data["history"],
				output_path=RESULTS_DIR / f"{MODEL_NAME}_iou_curve.png",
				metric_key="val_iou_macro",
			)

		if PLOT_PREDICTIONS:
			print("Generating prediction visualizations...")
			ckpt = torch.load(CHECKPOINT_DIR / f"{MODEL_NAME}_best.pt", map_location=device)
			model.load_state_dict(ckpt)
			model.to(device)
			visualize_predictions(
				model,
				MODEL_NAME,
				IMG_DIR,
				MSK_DIR,
				num_samples=4,
				device=device,
				output_path=RESULTS_DIR / f"{MODEL_NAME}_predictions.png",
			)
	except torch.OutOfMemoryError as e:
		if device == "cuda":
			print("\nCUDA OOM detected.")
			print("Try one or more of the following:")
			print("  - lower --batch-size (e.g., 1)")
			print("  - keep --amp enabled")
			print("  - increase --grad-accum-steps (e.g., 2 or 4)")
			print("  - switch --model to unet")
			print("  - use --device cpu for debugging")
			torch.cuda.empty_cache()
		raise e


if __name__ == "__main__":
	main()
