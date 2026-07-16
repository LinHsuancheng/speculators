#!/bin/bash
# Offline DSpark/speculators evaluation on DeepSpec eval_datasets.
#
# Usage:
#   VERIFIER_MODEL=/path/to/qwen3 \
#   DRAFT_MODEL=/path/to/dspark-speculator \
#   DEEPSPEC_EVAL_DATASETS=/path/to/DeepSpec/eval_datasets \
#   bash examples/evaluate/dspark_deepspec_offline.sh

set -euo pipefail

VERIFIER_MODEL="${VERIFIER_MODEL:-}"
DRAFT_MODEL="${DRAFT_MODEL:-}"
DEEPSPEC_EVAL_DATASETS="${DEEPSPEC_EVAL_DATASETS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/dspark_deepspec_offline}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-auto}"
DATASETS="${DATASETS:-}"
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
    --device "$DEVICE"
    --dtype "$DTYPE"
    --draft-attn-impl "$DRAFT_ATTN_IMPL"
    --trust-remote-code
)

if [[ -n "$DATASETS" ]]; then
    cmd+=(--datasets "$DATASETS")
fi

"${cmd[@]}"
