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
DATASET_INDEX="${DATASET_INDEX:-45760}"
BATCH_INDEX="${BATCH_INDEX:-}"

TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-3072}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2560}"
NUM_TARGET_LAYERS="${NUM_TARGET_LAYERS:-5}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 9 17 25 33}"
HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-bfloat16}"
NOISE_STD="${NOISE_STD:-0.05}"
ON_MISSING="${ON_MISSING:-generate}"
ON_GENERATE="${ON_GENERATE:-delete}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
MAX_RETRIES="${MAX_RETRIES:-3}"
HIDDEN_FILE_TIMEOUT="${HIDDEN_FILE_TIMEOUT:-30}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SEED="${SEED:-0}"
LOCAL_START="${LOCAL_START:-67}"
GT_LEN="${GT_LEN:-7}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
SKIP_DIRECT_VLLM="${SKIP_DIRECT_VLLM:-0}"
KEEP_DIRECT_HIDDEN_FILE="${KEEP_DIRECT_HIDDEN_FILE:-0}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/check_dspark_dataset_tensor_lifecycle_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TARGET_LAYER_ID_ARGS <<< "${TARGET_LAYER_IDS}"

args=(
  "${REPO_ROOT}/script/debug/check_dspark_dataset_tensor_lifecycle.py"
  --verifier-name-or-path "${VERIFIER_NAME_OR_PATH}"
  --data-path "${DATA_PATH}"
  --vllm-endpoint "${VLLM_ENDPOINT}"
  --dataset-index "${DATASET_INDEX}"
  --total-seq-len "${TOTAL_SEQ_LEN}"
  --hidden-size "${HIDDEN_SIZE}"
  --num-target-layers "${NUM_TARGET_LAYERS}"
  --target-layer-ids "${TARGET_LAYER_ID_ARGS[@]}"
  --hidden-states-dtype "${HIDDEN_STATES_DTYPE}"
  --noise-std "${NOISE_STD}"
  --on-missing "${ON_MISSING}"
  --on-generate "${ON_GENERATE}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --max-retries "${MAX_RETRIES}"
  --hidden-file-timeout "${HIDDEN_FILE_TIMEOUT}"
  --num-workers "${NUM_WORKERS}"
  --prefetch-factor "${PREFETCH_FACTOR}"
  --seed "${SEED}"
  --local-start "${LOCAL_START}"
  --gt-len "${GT_LEN}"
  --prompt-logprobs "${PROMPT_LOGPROBS}"
)

if [[ -n "${HIDDEN_STATES_PATH}" ]]; then
  args+=(--hidden-states-path "${HIDDEN_STATES_PATH}")
fi

if [[ -n "${BATCH_INDEX}" ]]; then
  args+=(--batch-index "${BATCH_INDEX}")
fi

if [[ "${SKIP_DIRECT_VLLM}" == "1" ]]; then
  args+=(--skip-direct-vllm)
fi

if [[ "${KEEP_DIRECT_HIDDEN_FILE}" == "1" ]]; then
  args+=(--keep-direct-hidden-file)
fi

echo "Writing DSpark dataset tensor lifecycle trace to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
