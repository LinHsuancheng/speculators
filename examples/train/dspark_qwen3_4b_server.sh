#!/bin/bash
# DSpark vLLM Server Script for Qwen3-4B on Ascend NPU
# Launches vLLM server only (no training)

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-4B}"
VLLM_PORT=8000
TARGET_LAYER_IDS="2 18 33"

# NPU assignment
VLLM_NPUS="0,1,2,3"

# vLLM configuration - fix memory + enable parallelism
# --enable-prefix-caching: the on-policy exact-acceptance loss scores
# gold_prefix + sampled_Y per anchor; the gold prefix is fixed across steps, so
# prefix caching reuses its KV and each step only pays for the sampled tokens.
VLLM_EXTRA_ARGS=(
    --data-parallel-size 4
    --max-model-len 4096
    --max-num-seqs 768
    --enable-prefix-caching
)

# ============ Launch vLLM Server ============
echo "=== Launching vLLM server on NPUs: $VLLM_NPUS ==="
echo "Model: $MODEL"
echo "Port: $VLLM_PORT"
echo "Target layers: $TARGET_LAYER_IDS"
echo "Extra args: ${VLLM_EXTRA_ARGS[@]}"

env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --no-include-last-layer \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}"
