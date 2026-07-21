#!/usr/bin/env bash
set -euo pipefail

VERIFIER_MODEL="${VERIFIER_MODEL:-/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-bfloat16}"
SAMPLE_START="${SAMPLE_START:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
TOKENS_PER_SAMPLE="${TOKENS_PER_SAMPLE:-64}"
SEED="${SEED:-0}"
MODE="${MODE:-full-sequence}"
SHOW_EXAMPLES="${SHOW_EXAMPLES:-8}"

ARGS=(
  --verifier-model "$VERIFIER_MODEL"
  --data-path "$DATA_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --sample-start "$SAMPLE_START"
  --num-samples "$NUM_SAMPLES"
  --tokens-per-sample "$TOKENS_PER_SAMPLE"
  --mode "$MODE"
  --show-examples "$SHOW_EXAMPLES"
  --seed "$SEED"
)

if [[ "${TRUST_REMOTE_CODE:-1}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

if [[ "${RANDOM_SAMPLES:-1}" == "1" ]]; then
  ARGS+=(--random-samples)
fi

if [[ "${RANDOM_TOKENS:-1}" == "1" ]]; then
  ARGS+=(--random-tokens)
fi

if [[ "${INCLUDE_UNMASKED:-0}" == "1" ]]; then
  ARGS+=(--include-unmasked)
fi

python script/debug/preparedata_resample_next_token_probe.py "${ARGS[@]}"
