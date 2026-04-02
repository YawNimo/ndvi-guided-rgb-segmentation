"""
Training-only entrypoint for NDVI-guided RGB segmentation.
"""

import json
import argparse
import hashlib
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


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from common.models import build_model


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
USE_WEIGHTED_DICE = True
LOSS_TYPE = "gdl"  # one of: soft_dice, gdl
DICE_WEIGHT = 1.0
SCHEDULER_TYPE = "plateau"  # one of: plateau, cosine
WARMUP_EPOCHS = 1
DETERMINISTIC = True
CUDNN_BENCHMARK = False
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 2
FUSED_ADAMW = False
VAL_BATCH_SIZE = None
VAL_FREQUENCY = 1
FULL_METRICS_FREQUENCY = 1
CACHE_CLASS_WEIGHTS = True

MODEL_NAME = "unet"  # one of: unet, deeplab, spanetfull
SUPPORTED_IMAGE_EXTS = (".tif",)  # in case pngs want to be supported later
SAMPLE_SIZE = None
SAMPLE_SEED = SEED
RUN_NAME = "default"


def parse_args():
	"""
	Parse command-line arguments to override default configuration.

	Returns:
		argparse.Namespace: Parsed and validated training configuration.
	"""
	parser = argparse.ArgumentParser(
		description="Train semantic segmentation models on pre-tiled RGB/mask PNG pairs."
	)
	parser.add_argument("--model", type=str, default=MODEL_NAME, choices=["unet", "deeplab", "spanetfull"])

	parser.add_argument("--epochs", type=int, default=EPOCHS)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--val-batch-size", type=int, default=VAL_BATCH_SIZE)
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
	parser.add_argument("--run-name", type=str, default=RUN_NAME, help="Run identity for checkpoint/result file names")
	parser.add_argument(
		"--weighted-dice",
		action=argparse.BooleanOptionalAction,
		default=USE_WEIGHTED_DICE,
		help="Enable class-weighted Dice in CE+Dice combined loss",
	)
	parser.add_argument("--loss-type", type=str, default=LOSS_TYPE, choices=["soft_dice", "gdl"])
	parser.add_argument("--dice-weight", type=float, default=DICE_WEIGHT)
	parser.add_argument("--scheduler", type=str, default=SCHEDULER_TYPE, choices=["plateau", "cosine"])
	parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
	parser.add_argument("--val-frequency", type=int, default=VAL_FREQUENCY)
	parser.add_argument("--full-metrics-frequency", type=int, default=FULL_METRICS_FREQUENCY)
	parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=DETERMINISTIC)
	parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=CUDNN_BENCHMARK)
	parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=PERSISTENT_WORKERS)
	parser.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR)
	parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=FUSED_ADAMW)
	parser.add_argument("--cache-class-weights", action=argparse.BooleanOptionalAction, default=CACHE_CLASS_WEIGHTS)

	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
	parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
	parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)

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
	if args.val_batch_size is not None and args.val_batch_size <= 0:
		raise ValueError("--val-batch-size must be > 0 when provided")
	if args.grad_accum_steps <= 0:
		raise ValueError("--grad-accum-steps must be > 0")
	if args.sample_size is not None and args.sample_size <= 0:
		raise ValueError("--sample-size must be > 0 when provided")
	if args.dice_weight <= 0:
		raise ValueError("--dice-weight must be > 0")
	if args.warmup_epochs < 0:
		raise ValueError("--warmup-epochs must be >= 0")
	if args.prefetch_factor <= 0:
		raise ValueError("--prefetch-factor must be > 0")
	if args.val_frequency <= 0:
		raise ValueError("--val-frequency must be > 0")
	if args.full_metrics_frequency <= 0:
		raise ValueError("--full-metrics-frequency must be > 0")
	if args.deterministic and args.cudnn_benchmark:
		raise ValueError("--deterministic and --cudnn-benchmark cannot both be enabled")

	return args


def set_seed(seed: int = 42, deterministic: bool = True, cudnn_benchmark: bool = False):
	"""Set deterministic random seeds for Python, NumPy, and PyTorch."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = bool(deterministic)
	torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
	os.environ["PYTHONHASHSEED"] = str(seed)


def _simple_train_augment(image: torch.Tensor, mask: torch.Tensor):
	"""Apply random flips and 90-degree rotations to an image/mask pair.

	Args:
		image (torch.Tensor): Input image tensor of shape ``(3, H, W)``.
		mask (torch.Tensor): Class-index mask tensor of shape ``(H, W)``.

	Returns:
		tuple[torch.Tensor, torch.Tensor]: Augmented image and mask tensors.
	"""
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
		"""Initialize dataset with paired paths and a split index view.

		Args:
			pairs (list[tuple[Path, Path]]): Image/mask path pairs.
			indices (list[int]): Subset indices into ``pairs``.
			train (bool): Whether to apply training augmentations.
		"""
		self.pairs = pairs
		self.indices = indices
		self.train = train

	def __len__(self):
		"""Return number of samples in this split."""
		return len(self.indices)

	def __getitem__(self, i):
		"""Load one image/mask pair and return tensorized sample."""
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
	"""Return accepted mask-stem variants for a given image stem."""
	# given `image_001.tif`, the mask names accepted would be `image_001_mask.tif` or `image_001.tif`
	return [f"{image_stem}_mask", image_stem]


def _build_mask_lookup(mask_files):
	"""Build filename-stem lookup with optional `_mask` normalization."""
	lookup = {}
	for m in mask_files:
		stem = m.stem
		lookup.setdefault(stem, m)
		if stem.endswith("_mask"):
			lookup.setdefault(stem[:-5], m)
	return lookup


def create_data_split():
	"""Create train/val/test index splits from discovered image-mask pairs.

	Returns:
		tuple: ``(pairs, train_idx, val_idx, test_idx, split_info)``.
	"""
	assert IMG_DIR.exists(), f"Missing image dir: {IMG_DIR}"
	assert MSK_DIR.exists(), f"Missing mask dir: {MSK_DIR}"

	img_files = _list_supported_images(IMG_DIR)
	total_img_files = len(img_files)
	if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(img_files):
		rng = random.Random(SAMPLE_SEED)
		img_files = sorted(rng.sample(img_files, SAMPLE_SIZE))
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

	selected_paths = [str(p) for p in img_files]
	split_signature = hashlib.sha256("\n".join(selected_paths).encode("utf-8")).hexdigest()[:16]

	print(f"Image tiles found (total): {total_img_files}")
	print(f"Image tiles selected: {len(img_files)}")
	print(f"Mask files found: {len(mask_files)}")
	print(f"Sample size: {SAMPLE_SIZE}")
	print(f"Sample seed: {SAMPLE_SEED}")
	print(f"Selection signature: {split_signature}")
	print(f"Paired tiles: {len(pairs)}")
	print(f"Missing masks: {missing}")
	print(f"Split -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

	split_info = {
		"total_images_found": total_img_files,
		"images_selected": len(img_files),
		"sample_size": SAMPLE_SIZE,
		"sample_seed": SAMPLE_SEED,
		"selection_signature": split_signature,
	}

	return pairs, train_idx, val_idx, test_idx, split_info


def create_dataloaders(pairs, train_idx, val_idx, test_idx):
	"""Build train/validation/test dataloaders from split indices."""
	train_ds = TileSegDataset(pairs, train_idx, train=True)
	val_ds = TileSegDataset(pairs, val_idx, train=False)
	test_ds = TileSegDataset(pairs, test_idx, train=False)
	train_dl_kwargs = {
		"batch_size": BATCH_SIZE,
		"num_workers": NUM_WORKERS,
		"pin_memory": PIN_MEMORY,
	}
	if NUM_WORKERS > 0:
		train_dl_kwargs["persistent_workers"] = PERSISTENT_WORKERS
		train_dl_kwargs["prefetch_factor"] = PREFETCH_FACTOR

	val_batch_size = VAL_BATCH_SIZE if VAL_BATCH_SIZE is not None else BATCH_SIZE
	val_dl_kwargs = dict(train_dl_kwargs)
	val_dl_kwargs["batch_size"] = val_batch_size

	train_loader = DataLoader(
		train_ds,
		shuffle=True,
		**train_dl_kwargs,
	)
	val_loader = DataLoader(
		val_ds,
		shuffle=False,
		**val_dl_kwargs,
	)
	test_loader = DataLoader(
		test_ds,
		shuffle=False,
		**val_dl_kwargs,
	)

	return train_loader, val_loader, test_loader


def _class_weight_cache_path(mask_files):
	"""Build a stable cache path for class-weight audits from mask metadata."""
	parts = [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in mask_files]
	signature = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]
	return RESULTS_DIR / f"class_weight_cache_{signature}.json", signature


def compute_class_weights(cache_class_weights=True):
	"""Compute inverse-frequency class weights from all mask pixels.

	Returns:
		tuple[torch.Tensor, dict]: Class weights tensor and audit payload.
	"""
	mask_files = _list_supported_images(MSK_DIR)
	assert mask_files, f"No mask files found in {MSK_DIR}"

	cache_path, cache_signature = _class_weight_cache_path(mask_files)
	if cache_class_weights and cache_path.exists():
		with open(cache_path, "r", encoding="utf-8") as f:
			cached = json.load(f)
		cached_weights = np.array(cached["weights"], dtype=np.float32)
		audit = cached["audit"]
		audit["cache_signature"] = cache_signature
		audit["cache_hit"] = True
		print(f"Loaded cached class-imbalance audit: {cache_path.name}")
		return torch.tensor(cached_weights, dtype=torch.float32), audit

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

	print("Class imbalance audit:")
	for i, name in enumerate(CLASS_NAMES):
		print(
			f"  {name}: count={int(counts[i])} "
			f"freq={freq[i]:.6f} weight={weights[i]:.4f}"
		)

	audit = {
		"num_mask_files": len(mask_files),
		"num_classes": NUM_CLASSES,
		"class_names": CLASS_NAMES,
		"pixel_count_per_class": {CLASS_NAMES[i]: int(counts[i]) for i in range(NUM_CLASSES)},
		"frequency_per_class": {CLASS_NAMES[i]: float(freq[i]) for i in range(NUM_CLASSES)},
		"ce_weight_per_class": {CLASS_NAMES[i]: float(weights[i]) for i in range(NUM_CLASSES)},
		"total_pixels": int(total),
		"frequency_sum": float(freq.sum()),
		"cache_signature": cache_signature,
		"cache_hit": False,
	}

	if cache_class_weights:
		RESULTS_DIR.mkdir(parents=True, exist_ok=True)
		with open(cache_path, "w", encoding="utf-8") as f:
			json.dump({"weights": weights.tolist(), "audit": audit}, f, indent=2)
		print(f"Saved class-weight cache: {cache_path.name}")

	return torch.tensor(weights, dtype=torch.float32), audit


def soft_dice_loss(logits, targets, class_weights=None, num_classes=NUM_CLASSES, eps=1e-6):
	"""Compute soft Dice loss over all classes.

	If class weights are provided, class Dice is aggregated with normalized
	weights. Otherwise, unweighted mean Dice is used.
	"""
	probs = F.softmax(logits, dim=1)
	targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

	dims = (0, 2, 3)
	intersection = torch.sum(probs * targets_oh, dims)
	union = torch.sum(probs + targets_oh, dims)

	dice = (2 * intersection + eps) / (union + eps)
	if class_weights is None:
		return 1 - dice.mean()

	norm_weights = class_weights / class_weights.sum().clamp_min(eps)
	return 1 - torch.sum(dice * norm_weights)


def generalized_dice_loss(logits, targets, num_classes=NUM_CLASSES, eps=1e-6):
	"""Canonical Generalized Dice Loss (Sudre et al.) using inverse squared class volume."""
	probs = F.softmax(logits, dim=1)
	targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

	dims = (0, 2, 3)
	intersection = torch.sum(probs * targets_oh, dims)
	probs_sum = torch.sum(probs, dims)
	target_sum = torch.sum(targets_oh, dims)

	valid = target_sum > 0
	if not torch.any(valid):
		return 1 - ((2 * intersection + eps) / (probs_sum + target_sum + eps)).mean()

	weights = torch.zeros_like(target_sum)
	weights[valid] = 1.0 / torch.square(target_sum[valid].clamp_min(eps))

	numerator = torch.sum(2.0 * weights * intersection)
	denominator = torch.sum(weights * (probs_sum + target_sum))
	gdice = (numerator + eps) / (denominator + eps)
	return 1.0 - gdice


def make_combined_loss(
	class_weights_tensor,
	device,
	use_weighted_dice=False,
	loss_type="gdl",
	dice_weight=1.0,
):
	"""Create combined CE + Dice loss function closure."""
	class_weights_on_device = class_weights_tensor.to(device)
	ce_loss = nn.CrossEntropyLoss(weight=class_weights_on_device)

	def combined_loss(logits, targets):
		if loss_type == "soft_dice":
			dice_weights = class_weights_on_device if use_weighted_dice else None
			dice_term = soft_dice_loss(logits, targets, class_weights=dice_weights)
		else:
			dice_term = generalized_dice_loss(logits, targets)
		return ce_loss(logits, targets) + dice_weight * dice_term

	return combined_loss


def _safe_div(num, den):
	"""Safely divide two scalars and return 0.0 when denominator is zero."""
	return num / den if den > 0 else 0.0


def freeze_batchnorm_layers(model):
	"""Freeze all BatchNorm layers in a model in eval mode."""
	for m in model.modules():
		if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
			m.eval()
			for p in m.parameters():
				p.requires_grad = False


@torch.no_grad()
def compute_segmentation_metrics(conf_mat: torch.Tensor, include_per_class=True):
	"""Compute macro and optional per-class segmentation metrics from confusion matrix."""
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
	metrics = {
		"f1_macro": float(f1.mean().item()),
		"iou_macro": float(iou.mean().item()),
		"pixel_acc": float(pixel_acc),
	}
	if include_per_class:
		metrics.update(
			{
				"f1_per_class": f1.tolist(),
				"iou_per_class": iou.tolist(),
				"prec_per_class": precision.tolist(),
				"rec_per_class": recall.tolist(),
			}
		)
	return metrics


def _update_confusion_matrix(conf_mat, preds, targets, num_classes=NUM_CLASSES):
	"""Accumulate predictions/targets into a confusion matrix tensor."""
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
	include_per_class_metrics=True,
):
	"""Run one training or validation epoch.

	Returns:
		tuple[float, dict, float]: Average loss, metrics dict, and elapsed seconds.
	"""
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
	metrics = compute_segmentation_metrics(conf_mat, include_per_class=include_per_class_metrics)

	return avg_loss, metrics, elapsed


def train_model(
	model,
	train_loader,
	val_loader,
	combined_loss_fn,
	run_metadata,
	model_name,
	device,
	lr=LEARNING_RATE,
	weight_decay=WEIGHT_DECAY,
	epochs=EPOCHS,
	use_amp=USE_AMP,
	grad_accum_steps=GRAD_ACCUM_STEPS,
	freeze_bn=FREEZE_BN,
	scheduler_type=SCHEDULER_TYPE,
	warmup_epochs=WARMUP_EPOCHS,
	fused_adamw=FUSED_ADAMW,
	val_frequency=VAL_FREQUENCY,
	full_metrics_frequency=FULL_METRICS_FREQUENCY,
):
	"""Train model with validation tracking, checkpointing, and early stopping.

	Returns:
		tuple[dict, list[dict]]: Best checkpoint metadata and full training history.
	"""
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)

	adamw_kwargs = {"lr": lr, "weight_decay": weight_decay}
	if fused_adamw and device == "cuda":
		adamw_kwargs["fused"] = True
	optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)

	if scheduler_type == "plateau":
		scheduler = ReduceLROnPlateau(
			optimizer,
			mode="min",
			factor=LR_SCHEDULER_FACTOR,
			patience=LR_SCHEDULER_PATIENCE,
		)
	else:
		warmup_iters = min(max(0, warmup_epochs), max(0, epochs - 1))
		if warmup_iters > 0:
			warmup = torch.optim.lr_scheduler.LinearLR(
				optimizer,
				start_factor=0.2,
				end_factor=1.0,
				total_iters=warmup_iters,
			)
			cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
				optimizer,
				T_max=max(1, epochs - warmup_iters),
			)
			scheduler = torch.optim.lr_scheduler.SequentialLR(
				optimizer,
				schedulers=[warmup, cosine],
				milestones=[warmup_iters],
			)
		else:
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

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

		should_validate = (epoch % val_frequency == 0) or (epoch == epochs)
		full_metrics_this_epoch = should_validate and ((epoch % full_metrics_frequency == 0) or (epoch == epochs))

		va_loss = None
		va_metrics = None
		va_time = 0.0
		val_f1 = None
		if should_validate:
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
				include_per_class_metrics=full_metrics_this_epoch,
			)
			val_f1 = float(va_metrics["f1_macro"])

		if scheduler_type == "plateau":
			if should_validate:
				scheduler.step(va_loss)
		else:
			scheduler.step()
		current_lr = optimizer.param_groups[0]["lr"]
		epoch_bar.set_postfix(val_f1=(f"{val_f1:.4f}" if val_f1 is not None else "skip"), lr=f"{current_lr:.1e}")

		row = {
			"epoch": epoch,
			"train_loss": tr_loss,
			"val_loss": va_loss,
			"train_time_sec": tr_time,
			"val_time_sec": va_time,
			"learning_rate": current_lr,
			"val_skipped": not should_validate,
			"train_f1_macro": float(tr_metrics["f1_macro"]),
			"val_f1_macro": val_f1,
			"train_iou_macro": float(tr_metrics["iou_macro"]),
			"val_iou_macro": float(va_metrics["iou_macro"]) if va_metrics is not None else None,
			"val_pixel_acc": float(va_metrics["pixel_acc"]) if va_metrics is not None else None,
			"val_f1_per_class": va_metrics.get("f1_per_class") if va_metrics is not None else None,
			"val_iou_per_class": va_metrics.get("iou_per_class") if va_metrics is not None else None,
			"val_prec_per_class": va_metrics.get("prec_per_class") if va_metrics is not None else None,
			"val_rec_per_class": va_metrics.get("rec_per_class") if va_metrics is not None else None,
		}
		history.append(row)

		if should_validate:
			print(
				f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
				f"val_f1={val_f1:.4f} val_iou={row['val_iou_macro']:.4f} "
				f"loss={va_loss:.4f} lr={current_lr:.1e} | "
				f"time={tr_time:.1f}s/{va_time:.1f}s"
			)
		else:
			print(
				f"[{model_name}] Epoch {epoch:02d}/{epochs} | "
				f"validation skipped (val_frequency={val_frequency}) "
				f"lr={current_lr:.1e} | time={tr_time:.1f}s"
			)

		if should_validate and val_f1 > best["val_f1_macro"]:
			best["val_f1_macro"] = val_f1
			best["epoch"] = epoch
			epochs_no_improve = 0

			run_name = str(run_metadata.get("run_name", "default"))
			ckpt_path = CHECKPOINT_DIR / f"{run_name}_{model_name}_best.pt"
			metrics_path = RESULTS_DIR / f"{run_name}_{model_name}_best_metrics.json"

			torch.save(model.state_dict(), ckpt_path)
			with open(metrics_path, "w", encoding="utf-8") as f:
				json.dump({"run_metadata": run_metadata, "best": row, "history": history}, f, indent=2)

			best["path_ckpt"] = str(ckpt_path)
			best["path_metrics"] = str(metrics_path)
			print(f"  -> New best! Saved checkpoint to {ckpt_path.name}")
		elif should_validate:
			epochs_no_improve += 1
			if epochs_no_improve >= EARLY_STOP_PATIENCE:
				print(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} validation checks.")
				break

	print(f"\nBest epoch: {best['epoch']} | Best val F1 (macro): {best['val_f1_macro']:.4f}")
	print(f"Checkpoint: {best['path_ckpt']}")
	return best, history


def print_model_info(model, name):
	"""Print total and trainable parameter counts for a model."""
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Model: {name}")
	print(f"Parameters: {total:,} total, {trainable:,} trainable")


def main():
	"""Parse runtime config, train selected model, and optionally generate plots."""
	global SEED, MODEL_NAME
	global EPOCHS, BATCH_SIZE, VAL_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EARLY_STOP_PATIENCE
	global TRAIN_RATIO, VAL_RATIO
	global NUM_WORKERS, PIN_MEMORY
	global USE_AMP, GRAD_ACCUM_STEPS
	global FREEZE_BN
	global PLOT_LOSS, PLOT_METRICS, PLOT_PREDICTIONS, USE_WEIGHTED_DICE
	global LOSS_TYPE, DICE_WEIGHT, SCHEDULER_TYPE, WARMUP_EPOCHS
	global VAL_FREQUENCY, FULL_METRICS_FREQUENCY
	global DETERMINISTIC, CUDNN_BENCHMARK
	global PERSISTENT_WORKERS, PREFETCH_FACTOR, FUSED_ADAMW, CACHE_CLASS_WEIGHTS
	global SAMPLE_SIZE, SAMPLE_SEED, RUN_NAME
	global IMG_DIR, MSK_DIR, RESULTS_DIR, CHECKPOINT_DIR

	args = parse_args()

	SEED = args.seed
	MODEL_NAME = args.model

	EPOCHS = args.epochs
	BATCH_SIZE = args.batch_size
	VAL_BATCH_SIZE = args.val_batch_size
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
	USE_WEIGHTED_DICE = args.weighted_dice
	LOSS_TYPE = args.loss_type
	DICE_WEIGHT = args.dice_weight
	SCHEDULER_TYPE = args.scheduler
	WARMUP_EPOCHS = args.warmup_epochs
	VAL_FREQUENCY = args.val_frequency
	FULL_METRICS_FREQUENCY = args.full_metrics_frequency
	DETERMINISTIC = args.deterministic
	CUDNN_BENCHMARK = args.cudnn_benchmark
	PERSISTENT_WORKERS = args.persistent_workers
	PREFETCH_FACTOR = args.prefetch_factor
	FUSED_ADAMW = args.fused_adamw
	CACHE_CLASS_WEIGHTS = args.cache_class_weights
	SAMPLE_SIZE = args.sample_size
	SAMPLE_SEED = args.sample_seed
	RUN_NAME = args.run_name

	IMG_DIR = args.img_dir
	MSK_DIR = args.msk_dir
	RESULTS_DIR = args.results_dir
	CHECKPOINT_DIR = args.checkpoint_dir

	set_seed(SEED, deterministic=DETERMINISTIC, cudnn_benchmark=CUDNN_BENCHMARK)
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
	print(f"Weighted Dice enabled: {USE_WEIGHTED_DICE}")
	print(f"Loss type: {LOSS_TYPE}")
	print(f"Dice weight: {DICE_WEIGHT}")
	print(f"Scheduler: {SCHEDULER_TYPE}")
	print(f"Run name: {RUN_NAME}")
	print(f"Deterministic: {DETERMINISTIC}")
	print(f"cuDNN benchmark: {CUDNN_BENCHMARK}")
	print(f"Persistent workers: {PERSISTENT_WORKERS}")
	print(f"Prefetch factor: {PREFETCH_FACTOR}")
	print(f"Validation batch size: {VAL_BATCH_SIZE if VAL_BATCH_SIZE is not None else BATCH_SIZE}")
	print(f"Validation frequency: every {VAL_FREQUENCY} epoch(s)")
	print(f"Full metrics frequency: every {FULL_METRICS_FREQUENCY} validation epoch(s)")
	print(f"Fused AdamW: {FUSED_ADAMW}")
	print(f"Cache class weights: {CACHE_CLASS_WEIGHTS}")

	pairs, train_idx, val_idx, test_idx, split_info = create_data_split()
	train_loader, val_loader, _ = create_dataloaders(pairs, train_idx, val_idx, test_idx)

	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	class_weights, imbalance_audit = compute_class_weights(cache_class_weights=CACHE_CLASS_WEIGHTS)
	imbalance_audit_path = RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_class_imbalance_audit.json"
	with open(imbalance_audit_path, "w", encoding="utf-8") as f:
		json.dump(imbalance_audit, f, indent=2)
	print(f"Saved class-imbalance audit: {imbalance_audit_path}")

	combined_loss_fn = make_combined_loss(
		class_weights,
		device,
		use_weighted_dice=USE_WEIGHTED_DICE,
		loss_type=LOSS_TYPE,
		dice_weight=DICE_WEIGHT,
	)

	run_metadata = {
		"run_name": RUN_NAME,
		"model": MODEL_NAME,
		"seed": SEED,
		"device": device,
		"batch_size": BATCH_SIZE,
		"val_batch_size": (VAL_BATCH_SIZE if VAL_BATCH_SIZE is not None else BATCH_SIZE),
		"weighted_dice_enabled": bool(USE_WEIGHTED_DICE),
		"loss_type": LOSS_TYPE,
		"dice_weight": DICE_WEIGHT,
		"scheduler": SCHEDULER_TYPE,
		"warmup_epochs": WARMUP_EPOCHS,
		"val_frequency": VAL_FREQUENCY,
		"full_metrics_frequency": FULL_METRICS_FREQUENCY,
		"deterministic": bool(DETERMINISTIC),
		"cudnn_benchmark": bool(CUDNN_BENCHMARK),
		"persistent_workers": bool(PERSISTENT_WORKERS),
		"prefetch_factor": PREFETCH_FACTOR,
		"fused_adamw": bool(FUSED_ADAMW),
		"cache_class_weights": bool(CACHE_CLASS_WEIGHTS),
		"sample_size": SAMPLE_SIZE,
		"sample_seed": SAMPLE_SEED,
		"split_info": split_info,
		"ce_weight_per_class": imbalance_audit["ce_weight_per_class"],
		"frequency_per_class": imbalance_audit["frequency_per_class"],
		"class_imbalance_audit_path": str(imbalance_audit_path),
	}

	model = build_model(MODEL_NAME, num_classes=NUM_CLASSES).to(device)
	print_model_info(model, MODEL_NAME)

	try:
		train_model(
			model=model,
			train_loader=train_loader,
			val_loader=val_loader,
			combined_loss_fn=combined_loss_fn,
			run_metadata=run_metadata,
			model_name=MODEL_NAME,
			device=device,
			lr=LEARNING_RATE,
			weight_decay=WEIGHT_DECAY,
			epochs=EPOCHS,
			use_amp=USE_AMP,
			grad_accum_steps=GRAD_ACCUM_STEPS,
			freeze_bn=FREEZE_BN,
			scheduler_type=SCHEDULER_TYPE,
			warmup_epochs=WARMUP_EPOCHS,
			fused_adamw=FUSED_ADAMW,
			val_frequency=VAL_FREQUENCY,
			full_metrics_frequency=FULL_METRICS_FREQUENCY,
		)

		print("\n" + "=" * 80)
		print("TRAINING COMPLETE")
		print("=" * 80)
		print_metrics_summary(RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_best_metrics.json")

		if PLOT_LOSS:
			print("Generating loss plot...")
			with open(RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_best_metrics.json") as f:
				hist_data = json.load(f)
			plot_loss(
				hist_data["history"],
				output_path=RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_loss_curve.png",
				title=f"{MODEL_NAME}: Loss vs Epoch",
			)

		if PLOT_METRICS:
			print("Generating metric plots...")
			with open(RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_best_metrics.json") as f:
				hist_data = json.load(f)
			plot_metrics(
				hist_data["history"],
				output_path=RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_f1_curve.png",
				metric_key="val_f1_macro",
			)
			plot_metrics(
				hist_data["history"],
				output_path=RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_iou_curve.png",
				metric_key="val_iou_macro",
			)

		if PLOT_PREDICTIONS:
			print("Generating prediction visualizations...")
			ckpt = torch.load(CHECKPOINT_DIR / f"{RUN_NAME}_{MODEL_NAME}_best.pt", map_location=device)
			model.load_state_dict(ckpt)
			model.to(device)
			visualize_predictions(
				model,
				MODEL_NAME,
				IMG_DIR,
				MSK_DIR,
				num_samples=4,
				device=device,
				output_path=RESULTS_DIR / f"{RUN_NAME}_{MODEL_NAME}_predictions.png",
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
