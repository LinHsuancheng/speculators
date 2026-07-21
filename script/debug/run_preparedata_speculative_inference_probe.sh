#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/script/debug:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:=/models/Qwen3-4B}"
: "${DRAFT_MODEL:=/outputs/dspark_qwen3_4b_real_baseline/checkpoints/checkpoint_best}"
: "${DATA_PATH:=/data/open_perfectblend_qwen3_4b_100k}"
: "${SAMPLE_START:=0}"
: "${NUM_SAMPLES:=16}"
: "${RANDOM_SAMPLES:=1}"
: "${MAX_NEW_TOKENS:=256}"
: "${MIN_PROMPT_TOKENS:=1}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"

cmd=(
    python3 script/debug/preparedata_speculative_inference_probe.py
    --verifier-model "$VERIFIER_MODEL"
    --draft-model "$DRAFT_MODEL"
    --data-path "$DATA_PATH"
    --sample-start "$SAMPLE_START"
    --num-samples "$NUM_SAMPLES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --min-prompt-tokens "$MIN_PROMPT_TOKENS"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --draft-attn-impl "$DRAFT_ATTN_IMPL"
)

if [[ "$RANDOM_SAMPLES" == "1" ]]; then
    cmd+=(--random-samples)
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi

exec "${cmd[@]}"
