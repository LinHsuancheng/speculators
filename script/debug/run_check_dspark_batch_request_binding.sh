#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
DATASET_INDEX="${DATASET_INDEX:-45760}"
BATCH_INDICES="${BATCH_INDICES:-}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
TOTAL_SEQ_LEN="${TOTAL_SEQ_LEN:-3072}"
LOCAL_START="${LOCAL_START:-67}"
GT_LEN="${GT_LEN:-7}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
HIDDEN_FILE_TIMEOUT="${HIDDEN_FILE_TIMEOUT:-30}"
MODE="${MODE:-both}"
MAX_WORKERS="${MAX_WORKERS:-0}"
LOGPROB_TOL="${LOGPROB_TOL:-0.5}"
HIDDEN_TOL="${HIDDEN_TOL:-0.01}"
COMPARE_FULL_HIDDEN="${COMPARE_FULL_HIDDEN:-0}"
KEEP_HIDDEN_FILES="${KEEP_HIDDEN_FILES:-0}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/check_dspark_batch_request_binding_$(date +%Y%m%d_%H%M%S).log}"

args=(
  "${REPO_ROOT}/script/debug/check_dspark_batch_request_binding.py"
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --dataset-index "${DATASET_INDEX}"
  --vllm-endpoint "${VLLM_ENDPOINT}"
  --total-seq-len "${TOTAL_SEQ_LEN}"
  --local-start "${LOCAL_START}"
  --gt-len "${GT_LEN}"
  --prompt-logprobs "${PROMPT_LOGPROBS}"
  --request-timeout "${REQUEST_TIMEOUT}"
  --hidden-file-timeout "${HIDDEN_FILE_TIMEOUT}"
  --mode "${MODE}"
  --max-workers "${MAX_WORKERS}"
  --logprob-tol "${LOGPROB_TOL}"
  --hidden-tol "${HIDDEN_TOL}"
)

if [[ -n "${BATCH_INDICES}" ]]; then
  read -r -a BATCH_INDEX_ARGS <<< "${BATCH_INDICES}"
  args+=(--batch-indices "${BATCH_INDEX_ARGS[@]}")
fi

if [[ "${COMPARE_FULL_HIDDEN}" == "1" ]]; then
  args+=(--compare-full-hidden)
fi

if [[ "${KEEP_HIDDEN_FILES}" == "1" ]]; then
  args+=(--keep-hidden-files)
fi

echo "Writing DSpark batch request binding check to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
