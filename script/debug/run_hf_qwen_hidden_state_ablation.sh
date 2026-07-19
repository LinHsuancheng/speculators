#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-bfloat16}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 9 17 25 33 36}"
REPEAT="${REPEAT:-2}"
POSITIONS="${POSITIONS:-}"
TEXT="${TEXT:-}"
TOKEN_IDS="${TOKEN_IDS:-}"
PRINT_TOKEN_LIMIT="${PRINT_TOKEN_LIMIT:-80}"
PRINT_HIDDEN_LIMIT="${PRINT_HIDDEN_LIMIT:-8}"
TOPK="${TOPK:-5}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/hf_qwen_hidden_state_ablation_$(date +%Y%m%d_%H%M%S).log}"

read -r -a TARGET_LAYER_ID_ARGS <<< "${TARGET_LAYER_IDS}"

args=(
  "${REPO_ROOT}/script/debug/hf_qwen_hidden_state_ablation.py"
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --sample-index "${SAMPLE_INDEX}"
  --max-seq-len "${MAX_SEQ_LEN}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --target-layer-ids "${TARGET_LAYER_ID_ARGS[@]}"
  --repeat "${REPEAT}"
  --print-token-limit "${PRINT_TOKEN_LIMIT}"
  --print-hidden-limit "${PRINT_HIDDEN_LIMIT}"
  --topk "${TOPK}"
)

if [[ -n "${TEXT}" ]]; then
  args+=(--text "${TEXT}")
fi

if [[ -n "${TOKEN_IDS}" ]]; then
  args+=(--token-ids "${TOKEN_IDS}")
fi

if [[ -n "${POSITIONS}" ]]; then
  read -r -a POSITION_ARGS <<< "${POSITIONS}"
  args+=(--positions "${POSITION_ARGS[@]}")
fi

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  args+=(--trust-remote-code)
fi

echo "Writing HF ablation trace to ${LOG_FILE}"
echo "Command:"
printf ' %q' "${PYTHON_BIN}" "${args[@]}"
echo

"${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${LOG_FILE}"
