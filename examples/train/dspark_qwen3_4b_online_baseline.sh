#!/bin/bash
# DSpark Qwen3-4B Online Training - Baseline
# Clean baseline config for 4v4 setup (4 vLLM + 4 training)

set -euo pipefail

# Ascend environment variables
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# ============ Basic Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATASET="sharegpt"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_qwen3_4b_baseline}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-3072}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-6e-4}"

# NPU assignment
VLLM_NPUS="${VLLM_NPUS:-0,1,2,3}"
TRAIN_NPUS="${TRAIN_NPUS:-4,5,6,7}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-4}"

# vLLM configuration - fix memory issue + enable parallelism
VLLM_EXTRA_ARGS=(
    --data-parallel-size 4
    --max-model-len 4096
    --max-num-seqs 768
)

# DSpark model configuration
SPECULATOR_TYPE="dspark"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DRAFT_ATTN_IMPL="sdpa"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 18 33}"

# Training configuration
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# ============ Cleanup Handler ============
cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ============ Step 1: Prepare Data ============
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# ============ Step 2: Launch vLLM ============
echo "=== Step 2: Launching vLLM on NPUs: $VLLM_NPUS ==="
env TASK_QUEUE_ENABLE=1 ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --no-include-last-layer \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
MAX_WAIT=300
ELAPSED=0
until curl --noproxy localhost,127.0.0.1 -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    if [[ $ELAPSED -ge $MAX_WAIT ]]; then
        echo "vLLM failed to start after ${MAX_WAIT}s"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done
echo "vLLM ready."

# ============ Step 3: Training ============
echo "=== Step 3: Training DSpark on NPUs: $TRAIN_NPUS ==="
env TASK_QUEUE_ENABLE=2 ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --num-workers 48 \
    --prefetch-factor 8 \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
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
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
