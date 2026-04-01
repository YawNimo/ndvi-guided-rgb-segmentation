#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/ehurd1@cfreg.local/ndvi-guided-rgb-segmentation"
VENV_ACTIVATE="${ROOT_DIR}/venv/bin/activate"
RUN_NAME="pipeline_best"


log() {
	echo "[run_pipeline] $*"
}

die() {
	echo "[run_pipeline] ERROR: $*" >&2
	exit 1
}

format_duration() {
	local total_seconds=$1
	local hours=$((total_seconds / 3600))
	local minutes=$(((total_seconds % 3600) / 60))
	local seconds=$((total_seconds % 60))
	printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
}

activate_venv() {
	[[ -f "$VENV_ACTIVATE" ]] || die "Missing venv activation script: $VENV_ACTIVATE"
	# shellcheck disable=SC1090
	source "$VENV_ACTIVATE"
	log "Activated venv: $VENV_ACTIVATE"
}

run_preprocessing() {
	log "Running preprocessing pipeline..."
	(
		python preprocessing/main.py --download --tile --blur --mask
	)
}

run_training() {
	log "Running training pipeline..."
	(
		python training/main.py \
			--run-name "$RUN_NAME" \
			--model unet \
			--epochs 12 \
			--early-stop-patience 6 \
			--batch-size 4 \
			--val-batch-size 8 \
			--amp \
			--loss-type gdl \
			--dice-weight 0.7 \
			--scheduler plateau \
			--val-frequency 1 \
			--full-metrics-frequency 1 \
			--deterministic \
			--no-cudnn-benchmark \
			--cache-class-weights \
			--num-workers 4 \
			--persistent-workers \
			--plot-loss \
			--prefetch-factor 4 \
			--plot-metrics \
			--plot-predictions
	)
}

run_model() {
	log "Running model inference pipeline..."
	(
		python runmodel/main.py --model=unet --ckpt="checkpoints/${RUN_NAME}_unet_best.pt"
	)
}

run_create_validation_visuals() {
	log "Running validation visuals pipeline..."
	(
		python validation/main.py --model=unet
	)
}

main() {
	local start_epoch
	local end_epoch
	local elapsed_seconds
	local start_time
	local end_time

	start_epoch=$(date +%s)
	start_time=$(date '+%Y-%m-%d %H:%M:%S %Z')
	log "Start time: ${start_time}"
	activate_venv

	run_preprocessing
	run_training
	run_model
	run_create_validation_visuals

	end_epoch=$(date +%s)
	end_time=$(date '+%Y-%m-%d %H:%M:%S %Z')
	elapsed_seconds=$((end_epoch - start_epoch))

	log "End time:   ${end_time}"
	log "Elapsed:    $(format_duration "$elapsed_seconds")"
	log "Pipeline complete."
}

main "$@"
