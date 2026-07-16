#!/bin/bash
# DSpark vLLM Server Script for Qwen3-4B on Ascend NPU
# Launches vLLM server only (no training)

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=34655
export GLOO_SOCKET_IFNAME=lo

export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=3600
export HCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NPU_ASYNC_ERROR_HANDLING=1
export ASCEND_LAUNCH_BLOCKING=0
# ============ Configuration ============
MODEL="/models/Qwen3-4B"
DATASET="sharegpt"
OUTPUT_DIR="./outputs/dspark_qwen3_4b_real_baseline"
ARROW_DIR="/data/open_perfectblend_qwen3_4b_100k"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=3072
EPOCHS=5
LR=6e-4

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
MAX_ANCHORS=512
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="1 9 17 25 33"
DRAFT_ATTN_IMPL="sdpa"

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# Ascend NPU assignments
VLLM_NPUS="8,9,10,11"
TRAIN_NPUS="12,13,14,15"
NUM_TRAIN_NPUS=4

# vLLM configuration - fix memory + enable parallelism
VLLM_EXTRA_ARGS=( --data-parallel-size 4 )

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server on Ascend NPU(s): $VLLM_NPUS ==="
env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}

trap cleanup INT TERM

echo "Waiting for vLLM server to be ready..."
until curl --noproxy '*' -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    echo "vLLM not ready yet..."
    sleep 5
done
echo "vLLM server ready."
echo "vLLM is running. Press Ctrl+C to stop it."

wait "$VLLM_PID"