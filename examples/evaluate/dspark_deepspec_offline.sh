#!/bin/bash
# Offline DSpark evaluation on DeepSpec-style JSONL datasets.
#
# Usage:
#   VERIFIER_MODEL=/path/to/verifier \
#   DRAFT_MODEL=/path/to/dspark/checkpoint \
#   DEEPSPEC_EVAL_DATASETS=/path/to/DeepSpec/eval_datasets \
#   ASCEND_DEVICES=8,9,10,11,12,13,14,15 \
#   bash examples/evaluate/dspark_deepspec_offline.sh

set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

VERIFIER_MODEL="${VERIFIER_MODEL:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
DEEPSPEC_EVAL_DATASETS="${DEEPSPEC_EVAL_DATASETS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dspark_deepspec_offline}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DEVICE="${DEVICE:-npu}"
DTYPE="${DTYPE:-bfloat16}"
DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-auto}"
ASCEND_DEVICES="${ASCEND_DEVICES:-8,9,10,11,12,13,14,15}"
D2T_PATH="${D2T_PATH:-}"
T2D_PATH="${T2D_PATH:-}"
DATASETS="${DATASETS:-}"
NO_PROGRESS="${NO_PROGRESS:-0}"
SKIP_ARTIFACTS="${SKIP_ARTIFACTS:-0}"
PYTHON="${PYTHON:-python3}"

if [[ -z "$VERIFIER_MODEL" ]]; then
    echo "ERROR: set VERIFIER_MODEL to the target/verifier model path or HF id."
    exit 1
fi

if [[ -z "$DRAFT_MODEL" ]]; then
    echo "ERROR: set DRAFT_MODEL to a DSpark/speculators checkpoint path."
    exit 1
fi

if [[ -z "$DEEPSPEC_EVAL_DATASETS" ]]; then
    echo "ERROR: set DEEPSPEC_EVAL_DATASETS to DeepSpec/eval_datasets."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../scripts/evaluate" && pwd)"

cmd=(
    "$PYTHON" "$SCRIPT_DIR/dspark_offline_eval.py"
    --verifier-model "$VERIFIER_MODEL"
    --draft-model "$DRAFT_MODEL"
    --datasets-root "$DEEPSPEC_EVAL_DATASETS"
    --output-dir "$OUTPUT_DIR"
    --max-samples "$MAX_SAMPLES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --draft-attn-impl "$DRAFT_ATTN_IMPL"
    --trust-remote-code
)

if [[ -n "$ASCEND_DEVICES" ]]; then
    cmd+=(--ascend-devices "$ASCEND_DEVICES")
fi

if [[ -n "$D2T_PATH" || -n "$T2D_PATH" ]]; then
    cmd+=(--d2t-path "$D2T_PATH" --t2d-path "$T2D_PATH")
fi

if [[ -n "$DATASETS" ]]; then
    cmd+=(--datasets "$DATASETS")
fi

if [[ "$NO_PROGRESS" == "1" ]]; then
    cmd+=(--no-progress)
fi

if [[ "$SKIP_ARTIFACTS" == "1" ]]; then
    cmd+=(--skip-artifacts)
fi

"${cmd[@]}"
