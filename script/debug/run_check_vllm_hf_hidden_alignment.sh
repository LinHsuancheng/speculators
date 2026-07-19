#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
DATASET_INDEX="${DATASET_INDEX:-45760}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
DEVICE="${DEVICE:-npu:15}"
DTYPE="${DTYPE:-bfloat16}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 9 17 25 33}"
FINAL_LAYER_ID="${FINAL_LAYER_ID:-}"
PREFIX_LEN="${PREFIX_LEN:-68}"
GT_LEN="${GT_LEN:-7}"
RAW_MAX_LEN="${RAW_MAX_LEN:-0}"
REPEAT="${REPEAT:-2}"
WARMUP_GENERATE="${WARMUP_GENERATE:-1}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
HIDDEN_FILE_TIMEOUT="${HIDDEN_FILE_TIMEOUT:-30}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
KEEP_HIDDEN_FILES="${KEEP_HIDDEN_FILES:-0}"
PRINT_TOKEN_WINDOW="${PRINT_TOKEN_WINDOW:-4}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/check_vllm_hf_hidden_alignment_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TARGET_LAYER_ID_ARGS <<< "${TARGET_LAYER_IDS}"

args=(
  "${REPO_ROOT}/script/debug/check_vllm_hf_hidden_alignment.py"
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --dataset-index "${DATASET_INDEX}"
  --vllm-endpoint "${VLLM_ENDPOINT}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --target-layer-ids "${TARGET_LAYER_ID_ARGS[@]}"
  --prefix-len "${PREFIX_LEN}"
  --gt-len "${GT_LEN}"
  --raw-max-len "${RAW_MAX_LEN}"
  --repeat "${REPEAT}"
  --prompt-logprobs "${PROMPT_LOGPROBS}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --hidden-file-timeout "${HIDDEN_FILE_TIMEOUT}"
  --print-token-window "${PRINT_TOKEN_WINDOW}"
)

if [[ -n "${FINAL_LAYER_ID}" ]]; then
  args+=(--final-layer-id "${FINAL_LAYER_ID}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "${KEEP_HIDDEN_FILES}" == "1" ]]; then
  args+=(--keep-hidden-files)
fi

if [[ "${WARMUP_GENERATE}" != "1" ]]; then
  args+=(--no-warmup-generate)
fi

echo "Writing alignment check to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
