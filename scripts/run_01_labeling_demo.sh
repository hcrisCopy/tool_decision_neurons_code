#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

MODEL_ALIAS="${1:-qwen3-1.7b}"
DATA_ROOT="${DATA_ROOT:-../tool_decision_neurons_data}"
BACKEND="${BACKEND:-hf}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"

python code/01_labeling/build_when2tool_labels.py \
  --model-alias "${MODEL_ALIAS}" \
  --data-root "${DATA_ROOT}" \
  --backend "${BACKEND}" \
  --subsets single_hop \
  --splits train \
  --max-samples "${MAX_SAMPLES}" \
  --overwrite
