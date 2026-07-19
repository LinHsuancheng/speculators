#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
HIDDEN_STATES_PATH="${HIDDEN_STATES_PATH:-}"
DATASET_INDEX="${DATASET_INDEX:-45760}"
GENERATED_PATH="${GENERATED_PATH:-}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
DEVICE="${DEVICE:-npu:15}"
DTYPE="${DTYPE:-bfloat16}"
HIDDEN_STATES_DTYPE="${HIDDEN_STATES_DTYPE:-bfloat16}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 9 17 25 33}"
FINAL_LAYER_ID="${FINAL_LAYER_ID:-}"
TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-3072}"
LOCAL_START="${LOCAL_START:-67}"
GT_LEN="${GT_LEN:-7}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
MAX_RETRIES="${MAX_RETRIES:-3}"
HIDDEN_FILE_TIMEOUT="${HIDDEN_FILE_TIMEOUT:-30}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
KEEP_HIDDEN_FILES="${KEEP_HIDDEN_FILES:-0}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/check_dspark_helper_generation_binding_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TARGET_LAYER_ID_ARGS <<< "${TARGET_LAYER_IDS}"

args=(
  "${REPO_ROOT}/script/debug/check_dspark_helper_generation_binding.py"
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --dataset-index "${DATASET_INDEX}"
  --vllm-endpoint "${VLLM_ENDPOINT}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --hidden-states-dtype "${HIDDEN_STATES_DTYPE}"
  --target-layer-ids "${TARGET_LAYER_ID_ARGS[@]}"
  --total-seq-len "${TOTAL_SEQ_LEN}"
  --local-start "${LOCAL_START}"
  --gt-len "${GT_LEN}"
  --prompt-logprobs "${PROMPT_LOGPROBS}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --max-retries "${MAX_RETRIES}"
  --hidden-file-timeout "${HIDDEN_FILE_TIMEOUT}"
)

if [[ -n "${HIDDEN_STATES_PATH}" ]]; then
  args+=(--hidden-states-path "${HIDDEN_STATES_PATH}")
fi

if [[ -n "${GENERATED_PATH}" ]]; then
  args+=(--generated-path "${GENERATED_PATH}")
fi

if [[ -n "${FINAL_LAYER_ID}" ]]; then
  args+=(--final-layer-id "${FINAL_LAYER_ID}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "${KEEP_HIDDEN_FILES}" == "1" ]]; then
  args+=(--keep-hidden-files)
fi

echo "Writing DSpark helper generation binding check to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
