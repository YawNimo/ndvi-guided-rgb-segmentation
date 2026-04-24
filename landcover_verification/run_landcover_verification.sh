#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/ehurd1@cfreg.local/ndvi-guided-rgb-segmentation"
VENV_PYTHON="${ROOT_DIR}/venv/bin/python"

RAW_GT_DIR="${ROOT_DIR}/landcover_verification/datasets/landcover_dataset/masks"
RAW_RGB_DIR="${ROOT_DIR}/landcover_verification/datasets/landcover_dataset/images"
TILED_RGB_DIR="${ROOT_DIR}/landcover_verification/datasets/tiled_images"
REMAPPED_GT_DIR="${ROOT_DIR}/landcover_verification/datasets/remapped_landcover_masks"
PRED_DIR="${ROOT_DIR}/landcover_verification/datasets/pred_masks"
TRIPLET_DIR="${ROOT_DIR}/landcover_verification/datasets/triplet_comparisons"

MODEL="unet"
CKPT_PATH="${ROOT_DIR}/checkpoints/pipeline_best_unet_best.pt"

# Sliding-window inference: full-tile forward OOMs on ~9k tiles with typical 8GB GPUs.
# Override via env or flags (see usage).
INFERENCE_PATCH_SIZE="${LANDCOVER_INFERENCE_PATCH_SIZE:-1024}"
INFERENCE_OVERLAP="${LANDCOVER_INFERENCE_OVERLAP:-256}"
RUNMODEL_LIMIT="${LANDCOVER_RUNMODEL_LIMIT:-10}"
TILE_SIZE="${LANDCOVER_TILE_SIZE:-500}"
TRIPLET_LIMIT="${LANDCOVER_TRIPLET_LIMIT:-3}"

DO_TILE=1
DO_RUNMODEL=1
DO_SCORE=1
DO_TRIPLETS=1

log() {
  echo "[run_landcover_verification] $*"
}

die() {
  echo "[run_landcover_verification] ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: bash landcover_verification/run_landcover_verification.sh [flags]

Flags:
  --skip-tile                Skip slicing RGB + remapping/slicing GT masks
  --skip-runmodel            Skip runmodel inference
  --skip-score               Skip F1/IoU scoring
  --skip-triplets            Skip triplet PNG generation
  --runmodel-limit N         Limit number of input tiles for runmodel (default: 10, 0 = all)
  --triplet-limit N          Number of triplet PNGs to generate (default: 3, 0 = all)
  --tile-size N              Tile edge for RGB/GT slicing (default: 500, or \$LANDCOVER_TILE_SIZE)
  --inference-patch-size N  Sliding-window patch edge (default: 1024, or \$LANDCOVER_INFERENCE_PATCH_SIZE).
                             0 = full-image forward (may CUDA OOM on large tiles). If > 0, must be multiple of 32.
  --inference-overlap N     Window overlap in pixels (default: 256, or \$LANDCOVER_INFERENCE_OVERLAP).
                             Must be < patch size when patch size > 0; must be 0 when patch size is 0.
  -h, --help                 Show this help text

Environment (optional overrides for inference memory tuning):
  LANDCOVER_INFERENCE_PATCH_SIZE   default 1024
  LANDCOVER_INFERENCE_OVERLAP      default 256
  LANDCOVER_RUNMODEL_LIMIT         default 10 (0 = all tiles)
  LANDCOVER_TRIPLET_LIMIT          default 3 (0 = all matched tiles)
  LANDCOVER_TILE_SIZE              default 500
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tile)
      DO_TILE=0
      ;;
    --skip-remap-input-masks)
      # Backward-compatible alias for old flag name.
      DO_TILE=0
      ;;
    --skip-runmodel)
      DO_RUNMODEL=0
      ;;
    --skip-score)
      DO_SCORE=0
      ;;
    --skip-triplets)
      DO_TRIPLETS=0
      ;;
    --inference-patch-size)
      [[ $# -ge 2 ]] || die "--inference-patch-size requires a value"
      INFERENCE_PATCH_SIZE="$2"
      shift 2
      continue
      ;;
    --runmodel-limit)
      [[ $# -ge 2 ]] || die "--runmodel-limit requires a value"
      RUNMODEL_LIMIT="$2"
      shift 2
      continue
      ;;
    --triplet-limit)
      [[ $# -ge 2 ]] || die "--triplet-limit requires a value"
      TRIPLET_LIMIT="$2"
      shift 2
      continue
      ;;
    --tile-size)
      [[ $# -ge 2 ]] || die "--tile-size requires a value"
      TILE_SIZE="$2"
      shift 2
      continue
      ;;
    --inference-overlap)
      [[ $# -ge 2 ]] || die "--inference-overlap requires a value"
      INFERENCE_OVERLAP="$2"
      shift 2
      continue
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

[[ -x "${VENV_PYTHON}" ]] || die "Missing venv python: ${VENV_PYTHON}"
[[ -d "${RAW_GT_DIR}" ]] || die "Missing input mask directory: ${RAW_GT_DIR}"
[[ -d "${RAW_RGB_DIR}" ]] || die "Missing input RGB directory: ${RAW_RGB_DIR}"

if [[ "${DO_RUNMODEL}" -eq 1 ]]; then
  if [[ "${RUNMODEL_LIMIT}" -lt 0 ]]; then
    die "--runmodel-limit / LANDCOVER_RUNMODEL_LIMIT must be >= 0"
  fi
fi

if [[ "${DO_TRIPLETS}" -eq 1 ]]; then
  if [[ "${TRIPLET_LIMIT}" -lt 0 ]]; then
    die "--triplet-limit / LANDCOVER_TRIPLET_LIMIT must be >= 0"
  fi
fi

if [[ "${DO_TILE}" -eq 1 ]]; then
  if [[ "${TILE_SIZE}" -le 0 ]]; then
    die "--tile-size / LANDCOVER_TILE_SIZE must be > 0"
  fi
fi

if [[ "${DO_RUNMODEL}" -eq 1 ]]; then
  if [[ "${INFERENCE_PATCH_SIZE}" -lt 0 ]]; then
    die "--inference-patch-size / LANDCOVER_INFERENCE_PATCH_SIZE must be >= 0"
  fi
  if [[ "${INFERENCE_PATCH_SIZE}" -gt 0 ]]; then
    if (( INFERENCE_PATCH_SIZE % 32 != 0 )); then
      die "inference patch size must be a multiple of 32 when non-zero (got ${INFERENCE_PATCH_SIZE})"
    fi
    if [[ "${INFERENCE_OVERLAP}" -ge "${INFERENCE_PATCH_SIZE}" ]]; then
      die "inference overlap must be < patch size (patch=${INFERENCE_PATCH_SIZE}, overlap=${INFERENCE_OVERLAP})"
    fi
  elif [[ "${INFERENCE_OVERLAP}" -ne 0 ]]; then
    die "inference overlap must be 0 when patch size is 0"
  fi
fi

if [[ "${DO_TILE}" -eq 1 ]]; then
  log "Step 1/4: Slice RGB and remap/slice GT masks"
  "${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/tile_landcover_for_verification.py" \
    --images-dir "${RAW_RGB_DIR}" \
    --masks-dir "${RAW_GT_DIR}" \
    --out-images-dir "${TILED_RGB_DIR}" \
    --out-remapped-masks-dir "${REMAPPED_GT_DIR}" \
    --tile-size "${TILE_SIZE}"
else
  log "Step 1/4: Skip tiling/remap (--skip-tile)"
fi

if [[ "${DO_RUNMODEL}" -eq 1 ]]; then
  [[ -f "${CKPT_PATH}" ]] || die "Missing checkpoint: ${CKPT_PATH}"
  [[ -d "${TILED_RGB_DIR}" ]] || die "Missing tiled RGB directory: ${TILED_RGB_DIR}"
  [[ -d "${REMAPPED_GT_DIR}" ]] || die "Missing remapped input mask directory: ${REMAPPED_GT_DIR}"
  log "Step 2/4: Run runmodel on tiled RGB -> ${PRED_DIR}"
  log "Runmodel limit: ${RUNMODEL_LIMIT} tile(s) (0 means all)"
  log "Inference: --inference-patch-size ${INFERENCE_PATCH_SIZE} --inference-overlap ${INFERENCE_OVERLAP}"
  "${VENV_PYTHON}" "${ROOT_DIR}/runmodel/main.py" \
    --model "${MODEL}" \
    --ckpt "${CKPT_PATH}" \
    --images_dir "${TILED_RGB_DIR}" \
    --masks_dir "${REMAPPED_GT_DIR}" \
    --pred_output_dir "${PRED_DIR}" \
    --limit "${RUNMODEL_LIMIT}" \
    --inference-patch-size "${INFERENCE_PATCH_SIZE}" \
    --inference-overlap "${INFERENCE_OVERLAP}"
else
  log "Step 2/4: Skip runmodel (--skip-runmodel)"
fi

if [[ "${DO_SCORE}" -eq 1 ]]; then
  [[ -d "${PRED_DIR}" ]] || die "Missing prediction directory for scoring: ${PRED_DIR}"
  [[ -d "${REMAPPED_GT_DIR}" ]] || die "Missing remapped input mask directory for scoring: ${REMAPPED_GT_DIR}"
  log "Step 3/4: Score F1 and IoU between pred masks and remapped input masks"
  "${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/score_landcover_metrics.py" \
    --pred-dir "${PRED_DIR}" \
    --gt-dir "${REMAPPED_GT_DIR}" \
    --out-csv "${ROOT_DIR}/landcover_verification/datasets/landcover_metrics_scores.csv"
else
  log "Step 3/4: Skip scoring (--skip-score)"
fi

if [[ "${DO_TRIPLETS}" -eq 1 ]]; then
  [[ -d "${TILED_RGB_DIR}" ]] || die "Missing tiled RGB directory for triplets: ${TILED_RGB_DIR}"
  [[ -d "${REMAPPED_GT_DIR}" ]] || die "Missing remapped GT directory for triplets: ${REMAPPED_GT_DIR}"
  [[ -d "${PRED_DIR}" ]] || die "Missing prediction directory for triplets: ${PRED_DIR}"
  log "Step 4/4: Generate triplet PNGs (limit=${TRIPLET_LIMIT}) -> ${TRIPLET_DIR}"
  "${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/compare_landcover_triplets.py" \
    --images-dir "${TILED_RGB_DIR}" \
    --gt-dir "${REMAPPED_GT_DIR}" \
    --pred-dir "${PRED_DIR}" \
    --output-dir "${TRIPLET_DIR}" \
    --limit "${TRIPLET_LIMIT}"
else
  log "Step 4/4: Skip triplet generation (--skip-triplets)"
fi

log "Landcover verification pipeline complete."
