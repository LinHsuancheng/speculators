#!/usr/bin/env bash
set -euo pipefail

VERIFIER_MODEL="${VERIFIER_MODEL:-/models/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-/outputs/dspark_qwen3_4b_real_baseline/checkpoints/checkpoint_best}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-bfloat16}"
DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-sdpa}"
NUM_ANCHORS="${NUM_ANCHORS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
THINK_PREFIX_SOURCE="${THINK_PREFIX_SOURCE:-auto}"
STRICT_THINK_PREFIX="${STRICT_THINK_PREFIX:-0}"
SHOW_ANCHORS="${SHOW_ANCHORS:-5}"
SEED="${SEED:-0}"

ARGS=(
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --data-path "$DATA_PATH"
  --sample-index "$SAMPLE_INDEX"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --draft-attn-impl "$DRAFT_ATTN_IMPL"
  --num-anchors "$NUM_ANCHORS"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --enable-thinking "$ENABLE_THINKING"
  --think-prefix-source "$THINK_PREFIX_SOURCE"
  --show-anchors "$SHOW_ANCHORS"
  --seed "$SEED"
)

if [[ "${THINK_PREFIX_TEXT:-}" != "" ]]; then
  ARGS+=(--think-prefix-text "$THINK_PREFIX_TEXT")
fi

if [[ "${RAW_JSONL:-}" != "" ]]; then
  ARGS+=(--raw-jsonl "$RAW_JSONL")
fi

if [[ "${RAW_SAMPLE_INDEX:-}" != "" ]]; then
  ARGS+=(--raw-sample-index "$RAW_SAMPLE_INDEX")
fi

if [[ "${ASSISTANT_TURN:-}" != "" ]]; then
  ARGS+=(--assistant-turn "$ASSISTANT_TURN")
fi

if [[ "${STRICT_PROMPT_PREFIX:-1}" == "0" ]]; then
  ARGS+=(--no-strict-prompt-prefix)
fi

if [[ "$STRICT_THINK_PREFIX" == "1" ]]; then
  ARGS+=(--strict-think-prefix)
fi

if [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ "${D2T_PATH:-}" != "" ]]; then
  ARGS+=(--d2t-path "$D2T_PATH")
fi

if [[ "${T2D_PATH:-}" != "" ]]; then
  ARGS+=(--t2d-path "$T2D_PATH")
fi

python script/debug/stored_vs_greedy_continuation_slot_probe.py "${ARGS[@]}"
