#!/usr/bin/env bash
set -euo pipefail

# Edit these values on the server if needed.
DATA_PATH="${DATA_PATH:-/data/open_perfectblend_qwen3_4b_100k}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
INDICES="${INDICES:-24509,32147}"

# Optional knobs.
SCORE_PREFIX_TOKENS="${SCORE_PREFIX_TOKENS:-128}"
SCORE_TOKENS="${SCORE_TOKENS:-8}"
PROMPT_LOGPROBS="${PROMPT_LOGPROBS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/smoke_training_pipeline.py" \
  --data-path "${DATA_PATH}" \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --indices "${INDICES}" \
  --score-prefix-tokens "${SCORE_PREFIX_TOKENS}" \
  --score-tokens "${SCORE_TOKENS}" \
  --prompt-logprobs "${PROMPT_LOGPROBS}"
