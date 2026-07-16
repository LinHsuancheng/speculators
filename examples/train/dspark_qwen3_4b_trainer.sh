#!/bin/bash
# Online DSpark Training Script for Qwen3-4B on Ascend NPU
# Sampled-loss smoke training configuration (4 vLLM + 1 training)
set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.48.17.185 no_proxy=localhost,127.0.0.1,80.48.17.185
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
OUTPUT_DIR="/outputs/dspark_qwen3_4b_test_only"
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
LOSS_FN="ce"
CONFIDENCE_HEAD_ALPHA=1.0
SAMPLED_ACCEPTANCE_LOSS_ALPHA=1.0

# Ascend NPU assignments
VLLM_NPUS="8,9,10,11"
TRAIN_NPU="12"

# vLLM configuration - fix memory + enable parallelism
VLLM_EXTRA_ARGS=(
    --data-parallel-size 4
)

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
# python scripts/prepare_data.py \
#     --model "$MODEL" \
#     --data "$DATASET" \
#     --output "$OUTPUT_DIR" \
#     --max-samples "$MAX_SAMPLES" \
#     --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server
echo "=== Step 2: Launching vLLM on NPUs: $VLLM_NPUS ==="
# env TASK_QUEUE_ENABLE=1 ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
#     --target-layer-ids $TARGET_LAYER_IDS \
#     --no-include-last-layer \
#     -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" &
# VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
until curl --noproxy localhost,127.0.0.1 -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    sleep 5
done
echo "vLLM ready."

# Step 3: Train DSpark
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"


# tensorboard --logdir ./logs --host 0.0.0.0 --port 6007 & 

echo "=== Step 3: Training on NPU: $TRAIN_NPU ==="
echo "Log file: $LOG_FILE"
env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPU" python scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$ARROW_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --log-freq 5 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --sampled-acceptance-loss-alpha "$SAMPLED_ACCEPTANCE_LOSS_ALPHA" \
    --enable-sampled-acceptance-loss \
    --logger tensorboard \
    --on-missing generate \
    --on-generate delete > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill \$(cat $PID_FILE)"

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/" 
