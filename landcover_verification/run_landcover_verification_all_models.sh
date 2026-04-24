#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_landcover_verification.sh"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "[run_landcover_verification_all_models] ERROR: Missing script: ${RUN_SCRIPT}" >&2
  exit 1
fi

MODELS=(
  "unet"
  "spanetfull"
  "deeplab"
)

is_first_run=1
for model in "${MODELS[@]}"; do
  echo "[run_landcover_verification_all_models] Running model: ${model}"
  if [[ "${is_first_run}" -eq 1 ]]; then
    bash "${RUN_SCRIPT}" --model "${model}" "$@"
    is_first_run=0
  else
    bash "${RUN_SCRIPT}" --model "${model}" --skip-tile "$@"
  fi
done

echo "[run_landcover_verification_all_models] Completed all model runs."
