#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

# Server defaults. Override any of these with environment variables.
VERIFIER_NAME_OR_PATH="${VERIFIER_NAME_OR_PATH:-/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
DEVICE="${DEVICE:-npu:0}"

TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-3072}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
NUM_LAYERS="${NUM_LAYERS:-5}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 9 17 25 33}"
DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-sdpa}"
MARKOV_RANK="${MARKOV_RANK:-256}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
LOSS_FN="${LOSS_FN:-{\"ce\": 0.1, \"tv\": 0.9}}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
SAMPLED_ACCEPTANCE_LOSS_ALPHA="${SAMPLED_ACCEPTANCE_LOSS_ALPHA:-1.0}"

HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-bfloat16}"
ON_MISSING="${ON_MISSING:-generate}"
ON_GENERATE="${ON_GENERATE:-delete}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
MAX_RETRIES="${MAX_RETRIES:-3}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
NOISE_STD="${NOISE_STD:-0.05}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BATCH_INDEX="${BATCH_INDEX:-0}"
TOPK="${TOPK:-5}"
PRINT_ANCHOR_LIMIT="${PRINT_ANCHOR_LIMIT:-24}"
LR="${LR:-6e-4}"

RUN_BACKWARD="${RUN_BACKWARD:-1}"
LEGACY_DATA="${LEGACY_DATA:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/trace_dspark_sampled_step_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TARGET_LAYER_ID_ARGS <<< "${TARGET_LAYER_IDS}"

args=(
  "${REPO_ROOT}/script/debug/trace_dspark_sampled_step.py"
  --verifier-name-or-path "${VERIFIER_NAME_OR_PATH}"
  --data-path "${DATA_PATH}"
  --vllm-endpoint "${VLLM_ENDPOINT}"
  --device "${DEVICE}"
  --total-seq-len "${TOTAL_SEQ_LEN}"
  --block-size "${BLOCK_SIZE}"
  --max-anchors "${MAX_ANCHORS}"
  --num-layers "${NUM_LAYERS}"
  --draft-vocab-size "${DRAFT_VOCAB_SIZE}"
  --target-layer-ids "${TARGET_LAYER_ID_ARGS[@]}"
  --draft-attn-impl "${DRAFT_ATTN_IMPL}"
  --markov-rank "${MARKOV_RANK}"
  --markov-head-type "${MARKOV_HEAD_TYPE}"
  --loss-fn "${LOSS_FN}"
  --confidence-head-alpha "${CONFIDENCE_HEAD_ALPHA}"
  --sampled-acceptance-loss-alpha "${SAMPLED_ACCEPTANCE_LOSS_ALPHA}"
  --hidden-states-dtype "${HIDDEN_STATES_DTYPE}"
  --on-missing "${ON_MISSING}"
  --on-generate "${ON_GENERATE}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --max-retries "${MAX_RETRIES}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --noise-std "${NOISE_STD}"
  --seed "${SEED}"
  --temperature "${TEMPERATURE}"
  --batch-index "${BATCH_INDEX}"
  --topk "${TOPK}"
  --print-anchor-limit "${PRINT_ANCHOR_LIMIT}"
  --lr "${LR}"
)

if [[ -n "${HIDDEN_STATES_PATH}" ]]; then
  args+=(--hidden-states-path "${HIDDEN_STATES_PATH}")
fi

if [[ "${RUN_BACKWARD}" != "1" ]]; then
  args+=(--skip-backward)
fi

if [[ "${LEGACY_DATA}" == "1" ]]; then
  args+=(--legacy-data)
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

echo "Writing trace to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
