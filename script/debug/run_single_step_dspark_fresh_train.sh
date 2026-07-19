#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

VERIFIER_NAME_OR_PATH="${VERIFIER_NAME_OR_PATH:-/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
DEVICE="${DEVICE:-npu:15}"
DATASET_INDEX="${DATASET_INDEX:-45760}"
BATCH_INDEX="${BATCH_INDEX:-}"

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
HIDDEN_FILE_TIMEOUT="${HIDDEN_FILE_TIMEOUT:-30}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
NOISE_STD="${NOISE_STD:-0.05}"
SEED="${SEED:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LR="${LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LOCAL_START="${LOCAL_START:-67}"
GT_LEN="${GT_LEN:-7}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
HIDDEN_TOL="${HIDDEN_TOL:-0.01}"
LOGPROB_TOL="${LOGPROB_TOL:-0.5}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
PRESERVE_HIDDEN_STATE_SOURCE="${PRESERVE_HIDDEN_STATE_SOURCE:-0}"
SKIP_BACKWARD="${SKIP_BACKWARD:-0}"
KEEP_DIRECT_HIDDEN_FILE="${KEEP_DIRECT_HIDDEN_FILE:-0}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/single_step_dspark_fresh_train_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TARGET_LAYER_ID_ARGS <<< "${TARGET_LAYER_IDS}"

args=(
  "${REPO_ROOT}/script/debug/single_step_dspark_fresh_train.py"
  --verifier-name-or-path "${VERIFIER_NAME_OR_PATH}"
  --data-path "${DATA_PATH}"
  --vllm-endpoint "${VLLM_ENDPOINT}"
  --device "${DEVICE}"
  --dataset-index "${DATASET_INDEX}"
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
  --hidden-file-timeout "${HIDDEN_FILE_TIMEOUT}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --noise-std "${NOISE_STD}"
  --seed "${SEED}"
  --temperature "${TEMPERATURE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --local-start "${LOCAL_START}"
  --gt-len "${GT_LEN}"
  --prompt-logprobs "${PROMPT_LOGPROBS}"
  --hidden-tol "${HIDDEN_TOL}"
  --logprob-tol "${LOGPROB_TOL}"
)

if [[ -n "${HIDDEN_STATES_PATH}" ]]; then
  args+=(--hidden-states-path "${HIDDEN_STATES_PATH}")
fi

if [[ -n "${BATCH_INDEX}" ]]; then
  args+=(--batch-index "${BATCH_INDEX}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "${PRESERVE_HIDDEN_STATE_SOURCE}" == "1" ]]; then
  args+=(--preserve-hidden-state-source)
fi

if [[ "${SKIP_BACKWARD}" == "1" ]]; then
  args+=(--skip-backward)
fi

if [[ "${KEEP_DIRECT_HIDDEN_FILE}" == "1" ]]; then
  args+=(--keep-direct-hidden-file)
fi

echo "Writing fresh single-step DSpark train debug to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
