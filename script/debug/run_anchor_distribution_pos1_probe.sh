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
: "${TRAIN_SAMPLE_START:=0}"
: "${TRAIN_SAMPLES:=8}"
: "${TRAIN_ANCHORS_PER_SAMPLE:=8}"
: "${TRAIN_RANDOM_ANCHORS_PER_SAMPLE:=8}"
: "${PROMPT_DATASET:=}"
: "${PROMPT:=}"
: "${PROMPT_START:=0}"
: "${NUM_PROMPTS:=4}"
: "${MAX_NEW_TOKENS:=128}"
: "${GENERATED_TOKEN_ANCHORS:=64}"
: "${POSITION_BUCKETS:=8}"
: "${TOTAL_SEQ_LEN:=3072}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${HIDDEN_STATES_DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"

cmd=(
    python3 script/debug/anchor_distribution_pos1_probe.py
    --verifier-model "$VERIFIER_MODEL"
    --draft-model "$DRAFT_MODEL"
    --data-path "$DATA_PATH"
    --vllm-endpoint "$VLLM_ENDPOINT"
    --request-timeout "$REQUEST_TIMEOUT"
    --max-retries "$MAX_RETRIES"
    --train-sample-start "$TRAIN_SAMPLE_START"
    --train-samples "$TRAIN_SAMPLES"
    --train-anchors-per-sample "$TRAIN_ANCHORS_PER_SAMPLE"
    --train-random-anchors-per-sample "$TRAIN_RANDOM_ANCHORS_PER_SAMPLE"
    --prompt-start "$PROMPT_START"
    --num-prompts "$NUM_PROMPTS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --generated-token-anchors "$GENERATED_TOKEN_ANCHORS"
    --position-buckets "$POSITION_BUCKETS"
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
