#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/dspark_trace_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/quiet_trace_dspark_sampled_step_$(date +%Y%m%d_%H%M%S).log}"

export LOG_FILE
"${SCRIPT_DIR}/run_trace_dspark_sampled_step.sh" >"${LOG_FILE}.runner" 2>&1

if ! python3 "${SCRIPT_DIR}/check_trace_dspark_sampled_step.py" "${LOG_FILE}"; then
  echo "runner_log=${LOG_FILE}.runner"
  echo "--- key trace ---"
  grep -E "TRACE (packed_raw_alignment|gt_verifier_compare.summary|sampled_verifier_prefix_compare.summary)|hidden_max_abs_diff|verifier_last_max_abs_diff|diff_doc|local_pruned" "${LOG_FILE}" || true
  exit 1
fi
