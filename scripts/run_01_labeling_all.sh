#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

DATA_ROOT="${DATA_ROOT:-../tool_decision_neurons_data}"
BACKEND="${BACKEND:-vllm}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

MODELS=(
  qwen3-1.7b
  qwen3-4b-instruct
  qwen3-14b
  qwen3-32b
  llama3.1-8b
  llama3.3-70b
)

for MODEL_ALIAS in "${MODELS[@]}"; do
  python code/01_labeling/build_when2tool_labels.py \
    --model-alias "${MODEL_ALIAS}" \
    --data-root "${DATA_ROOT}" \
    --backend "${BACKEND}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --overwrite
done
