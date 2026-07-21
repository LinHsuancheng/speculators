#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:=/models/Qwen3-4B}"
: "${DRAFT_MODEL:=/outputs/dspark_qwen3_4b_real_baseline/checkpoints/checkpoint_best}"
: "${DATA_PATH:=/data/open_perfectblend_qwen3_4b_100k}"
: "${HIDDEN_STATES_PATH:=}"
: "${SAMPLE_INDEX:=0}"
: "${ANCHOR_POSITION:=}"
: "${TOTAL_SEQ_LEN:=3072}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${HIDDEN_STATES_DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"
: "${SAMPLE_FROM_ANCHOR:=0}"

cmd=(
    python3 script/debug/replay_dspark_anchor_step.py
    --verifier-model "$VERIFIER_MODEL"
    --draft-model "$DRAFT_MODEL"
    --data-path "$DATA_PATH"
    --sample-index "$SAMPLE_INDEX"
    --total-seq-len "$TOTAL_SEQ_LEN"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --hidden-states-dtype "$HIDDEN_STATES_DTYPE"
    --draft-attn-impl "$DRAFT_ATTN_IMPL"
)

if [[ -n "$HIDDEN_STATES_PATH" ]]; then
    cmd+=(--hidden-states-path "$HIDDEN_STATES_PATH")
fi

if [[ -n "$ANCHOR_POSITION" ]]; then
    cmd+=(--anchor-position "$ANCHOR_POSITION")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi

if [[ "$SAMPLE_FROM_ANCHOR" == "1" ]]; then
    cmd+=(--sample-from-anchor)
fi

exec "${cmd[@]}"
