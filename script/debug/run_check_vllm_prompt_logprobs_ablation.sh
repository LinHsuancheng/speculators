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
PROMPT_LEN="${PROMPT_LEN:-0}"
PREFIX_LENS="${PREFIX_LENS:-32 68 128}"
GT_LEN="${GT_LEN:-7}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
HIDDEN_FILE_TIMEOUT="${HIDDEN_FILE_TIMEOUT:-30}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
KEEP_HIDDEN_FILES="${KEEP_HIDDEN_FILES:-0}"
FULL_SEQUENCE="${FULL_SEQUENCE:-1}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/check_vllm_prompt_logprobs_ablation_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TARGET_LAYER_ID_ARGS <<< "${TARGET_LAYER_IDS}"
read -r -a PREFIX_LEN_ARGS <<< "${PREFIX_LENS}"

args=(
  "${REPO_ROOT}/script/debug/check_vllm_prompt_logprobs_ablation.py"
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --dataset-index "${DATASET_INDEX}"
  --vllm-endpoint "${VLLM_ENDPOINT}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --target-layer-ids "${TARGET_LAYER_ID_ARGS[@]}"
  --prompt-len "${PROMPT_LEN}"
  --prefix-lens "${PREFIX_LEN_ARGS[@]}"
  --gt-len "${GT_LEN}"
  --prompt-logprobs "${PROMPT_LOGPROBS}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --hidden-file-timeout "${HIDDEN_FILE_TIMEOUT}"
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

if [[ "${FULL_SEQUENCE}" != "1" ]]; then
  args+=(--no-full-sequence)
fi

echo "Writing prompt-logprobs ablation to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
