#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/ehurd1@cfreg.local/ndvi-guided-rgb-segmentation"
VENV_PYTHON="${ROOT_DIR}/venv/bin/python"

RAW_GT_DIR="${ROOT_DIR}/landcover_verification/datasets/landcover_dataset/masks"
REMAPPED_GT_DIR="${ROOT_DIR}/landcover_verification/datasets/remapped_landcover_masks"
PRED_DIR="${ROOT_DIR}/landcover_verification/datasets/pred_masks"

MODEL="unet"
CKPT_PATH="${ROOT_DIR}/checkpoints/pipeline_best_unet_best.pt"

DO_REMAP_INPUT_MASKS=1
DO_RUNMODEL=1
DO_SCORE=1

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
  --skip-remap-input-masks   Skip remapping input masks
  --skip-runmodel            Skip runmodel inference
  --skip-score               Skip F1/IoU scoring
  -h, --help                 Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-remap-input-masks)
      DO_REMAP_INPUT_MASKS=0
      ;;
    --skip-runmodel)
      DO_RUNMODEL=0
      ;;
    --skip-score)
      DO_SCORE=0
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

if [[ "${DO_REMAP_INPUT_MASKS}" -eq 1 ]]; then
  log "Step 1/3: Remap input masks"
  "${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/convert_landcover_dataset.py" \
    --input-dir "${RAW_GT_DIR}" \
    --output-dir "${REMAPPED_GT_DIR}"
else
  log "Step 1/3: Skip remapping input masks (--skip-remap-input-masks)"
fi

if [[ "${DO_RUNMODEL}" -eq 1 ]]; then
  [[ -f "${CKPT_PATH}" ]] || die "Missing checkpoint: ${CKPT_PATH}"
  [[ -d "${REMAPPED_GT_DIR}" ]] || die "Missing remapped input mask directory: ${REMAPPED_GT_DIR}"
  log "Step 2/3: Run runmodel on remapped masks -> ${PRED_DIR}"
  "${VENV_PYTHON}" "${ROOT_DIR}/runmodel/main.py" \
    --model "${MODEL}" \
    --ckpt "${CKPT_PATH}" \
    --images_dir "${REMAPPED_GT_DIR}" \
    --pred_output_dir "${PRED_DIR}"
else
  log "Step 2/3: Skip runmodel (--skip-runmodel)"
fi

if [[ "${DO_SCORE}" -eq 1 ]]; then
  [[ -d "${PRED_DIR}" ]] || die "Missing prediction directory for scoring: ${PRED_DIR}"
  [[ -d "${REMAPPED_GT_DIR}" ]] || die "Missing remapped input mask directory for scoring: ${REMAPPED_GT_DIR}"
  log "Step 3/3: Score F1 and IoU between pred masks and remapped input masks"
  "${VENV_PYTHON}" "${ROOT_DIR}/landcover_verification/score_landcover_metrics.py" \
    --pred-dir "${PRED_DIR}" \
    --gt-dir "${REMAPPED_GT_DIR}" \
    --out-csv "${ROOT_DIR}/landcover_verification/datasets/landcover_metrics_scores.csv"
else
  log "Step 3/3: Skip scoring (--skip-score)"
fi

log "Landcover verification pipeline complete."
