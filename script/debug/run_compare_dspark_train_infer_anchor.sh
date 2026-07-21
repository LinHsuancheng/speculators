#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/script/debug:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:=/models/Qwen3-4B}"
: "${DRAFT_MODEL:=/outputs/dspark_qwen3_4b_real_baseline/checkpoints/checkpoint_best}"
: "${DATA_PATH:=/data/open_perfectblend_qwen3_4b_100k}"
: "${VLLM_ENDPOINT:=http://localhost:8000/v1}"
: "${VLLM_MODEL:=}"
: "${REQUEST_TIMEOUT:=120}"
: "${MAX_RETRIES:=3}"
: "${DELETE_VLLM_HIDDEN_STATE:=0}"
: "${SAMPLE_INDEX:=0}"
: "${ANCHOR_POSITION:?Set ANCHOR_POSITION to the fixed anchor index}"
: "${TOTAL_SEQ_LEN:=3072}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${HIDDEN_STATES_DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"

cmd=(
    python3 script/debug/compare_dspark_train_infer_anchor.py
    --verifier-model "$VERIFIER_MODEL"
    --draft-model "$DRAFT_MODEL"
    --data-path "$DATA_PATH"
    --vllm-endpoint "$VLLM_ENDPOINT"
    --request-timeout "$REQUEST_TIMEOUT"
    --max-retries "$MAX_RETRIES"
    --sample-index "$SAMPLE_INDEX"
    --anchor-position "$ANCHOR_POSITION"
    --total-seq-len "$TOTAL_SEQ_LEN"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --hidden-states-dtype "$HIDDEN_STATES_DTYPE"
    --draft-attn-impl "$DRAFT_ATTN_IMPL"
)

if [[ -n "$VLLM_MODEL" ]]; then
    cmd+=(--vllm-model "$VLLM_MODEL")
fi

if [[ "$DELETE_VLLM_HIDDEN_STATE" == "1" ]]; then
    cmd+=(--delete-vllm-hidden-state)
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi

exec "${cmd[@]}"
