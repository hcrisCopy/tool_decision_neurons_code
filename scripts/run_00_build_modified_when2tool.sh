#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}"

DATA_ROOT="${DATA_ROOT:-../tool_decision_neurons_data}"

python code/00_dataset_preparation/build_modified_when2tool.py \
  --data-root "${DATA_ROOT}" \
  --overwrite
