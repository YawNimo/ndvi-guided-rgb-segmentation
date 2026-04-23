#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/ehurd1@cfreg.local/ndvi-guided-rgb-segmentation"
VENV_PYTHON="${ROOT_DIR}/venv/bin/python"

RGB_INPUT_DIR="${ROOT_DIR}/landcover_verification/datasets/landcover_dataset/images"
UNMAPPED_PRED_DIR="${ROOT_DIR}/landcover_verification/datasets/unmapped_pred_masks"
REMAPPED_PRED_DIR="${ROOT_DIR}/landcover_verification/datasets/remapped_pred_masks"
REMAPPED_GT_DIR="${ROOT_DIR}/landcover_verification/datasets/remapped_landcover_masks"

MODEL="unet"
CKPT_PATH="${ROOT_DIR}/checkpoints/pipeline_best_unet_best.pt"

log() {
  echo "[run_landcover_verification] $*"
}

die() {
  echo "[run_landcover_verification] ERROR: $*" >&2
  exit 1
}

[[ -x "${VENV_PYTHON}" ]] || die "Missing venv python: ${VENV_PYTHON}"
[[ -d "${RGB_INPUT_DIR}" ]] || die "Missing RGB input directory: ${RGB_INPUT_DIR}"
[[ -f "${CKPT_PATH}" ]] || die "Missing checkpoint: ${CKPT_PATH}"

# Large RGB GeoTIFFs can OOM on a single full-GPU forward; sliding-window inference avoids that.
# Fallback if VRAM is still tight: add --device cpu (slow but reliable).
log "Step 1/4: Run runmodel inference to unmapped prediction masks"
"${VENV_PYTHON}" "${ROOT_DIR}/runmodel/main.py" \
  --model "${MODEL}" \
  --ckpt "${CKPT_PATH}" \
  --images_dir "${RGB_INPUT_DIR}" \
  --pred_output_dir "${UNMAPPED_PRED_DIR}" \
  --inference-patch-size 960 \
  --inference-overlap 128

log "Step 2/4: Remap predicted masks"
"${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/convert_pred_masks.py" \
  --input-dir "${UNMAPPED_PRED_DIR}" \
  --output-dir "${REMAPPED_PRED_DIR}"

if [[ ! -d "${REMAPPED_GT_DIR}" ]] || [[ -z "$(ls -A "${REMAPPED_GT_DIR}" 2>/dev/null)" ]]; then
  log "Step 3/4: Remap GT masks (not found yet)"
  "${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/convert_landcover_dataset.py" \
    --input-dir "${ROOT_DIR}/landcover_verification/datasets/landcover_dataset/masks" \
    --output-dir "${REMAPPED_GT_DIR}"
else
  log "Step 3/4: Skip GT remap (already present at ${REMAPPED_GT_DIR})"
fi

log "Step 4/4: Score F1 and IoU between remapped predictions and GT"
"${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/score_landcover_metrics.py" \
  --pred-dir "${REMAPPED_PRED_DIR}" \
  --gt-dir "${REMAPPED_GT_DIR}" \
  --out-csv "${ROOT_DIR}/landcover_verification/datasets/landcover_metrics_scores.csv"

log "Landcover verification pipeline complete."
