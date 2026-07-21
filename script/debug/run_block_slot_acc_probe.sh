#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/script/debug:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:=/models/Qwen3-4B}"
: "${DRAFT_MODEL:=/outputs/dspark_qwen3_4b_real_baseline/checkpoints/checkpoint_best}"
: "${DATA_PATH:=/data/open_perfectblend_qwen3_4b_100k}"
: "${PROMPT_DATASET:=}"
: "${PROMPT:=}"
: "${SAMPLE_START:=0}"
: "${NUM_SAMPLES:=4}"
: "${ANCHORS_PER_SAMPLE:=32}"
: "${MAX_NEW_TOKENS:=128}"
: "${ENABLE_THINKING:=false}"
: "${PRINT_PROMPT:=1}"
: "${COMPARE_TARGET_ONLY:=1}"
: "${PROMPT_TAIL_TOKENS:=32}"
: "${PROMPT_TAIL_CHARS:=400}"
: "${DEBUG_CONTEXT_TOKENS:=32}"
: "${PREPAREDATA_SAMPLE_START:=0}"
: "${PREPAREDATA_NUM_SAMPLES:=0}"
: "${PREPAREDATA_ANCHORS_PER_SAMPLE:=32}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"

cmd=(
    python3 script/debug/block_slot_acc_probe.py
    --verifier-model "$VERIFIER_MODEL"
    --draft-model "$DRAFT_MODEL"
    --data-path "$DATA_PATH"
    --sample-start "$SAMPLE_START"
    --num-samples "$NUM_SAMPLES"
    --anchors-per-sample "$ANCHORS_PER_SAMPLE"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --enable-thinking "$ENABLE_THINKING"
    --prompt-tail-tokens "$PROMPT_TAIL_TOKENS"
    --prompt-tail-chars "$PROMPT_TAIL_CHARS"
    --debug-context-tokens "$DEBUG_CONTEXT_TOKENS"
    --preparedata-sample-start "$PREPAREDATA_SAMPLE_START"
    --preparedata-num-samples "$PREPAREDATA_NUM_SAMPLES"
    --preparedata-anchors-per-sample "$PREPAREDATA_ANCHORS_PER_SAMPLE"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --draft-attn-impl "$DRAFT_ATTN_IMPL"
)

if [[ "$PRINT_PROMPT" == "0" ]]; then
    cmd+=(--no-print-prompt)
fi

if [[ "$COMPARE_TARGET_ONLY" == "0" ]]; then
    cmd+=(--no-compare-target-only)
fi

if [[ -n "$PROMPT_DATASET" ]]; then
    cmd+=(--prompt-dataset "$PROMPT_DATASET")
fi

if [[ -n "$PROMPT" ]]; then
    cmd+=(--prompt "$PROMPT")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi

exec "${cmd[@]}"
