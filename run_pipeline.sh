#!/usr/bin/env bash

set -euo pipefail


log() {
	echo "[download_and_train] $*"
}

die() {
	echo "[download_and_train] ERROR: $*" >&2
	exit 1
}

format_duration() {
	local total_seconds=$1
	local hours=$((total_seconds / 3600))
	local minutes=$(((total_seconds % 3600) / 60))
	local seconds=$((total_seconds % 60))
	printf "%02dh:%02dm:%02ds" "$hours" "$minutes" "$seconds"
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
			--model unet \
			--epochs 50 \
			--early-stop-patience 8 \
			--batch-size 1 \
			--amp \
			--freeze-bn \
			--plot-loss \
			--plot-metrics \
			--plot-predictions \
			--num-workers 0
	)
}

run_model() {
	log "Running model..."
	(
		python runmodel/main.py --model=unet --ckpt=checkpoints/unet_best.pt
	)
}

run_create_validation_visuals() {
	log "Running validation visuals..."
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
