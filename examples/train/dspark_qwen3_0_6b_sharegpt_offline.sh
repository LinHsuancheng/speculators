#!/bin/bash
# Offline DSpark Training Script for Qwen3-0.6B on Ascend NPU
#
# Runs the offline DSpark pipeline on one Ascend device: data preparation,
# vLLM hidden-state generation, vLLM shutdown, then training on the same device.
# This is the recommended mode when only one device, such as device 7, is available.
#
# Usage:
#   bash examples/train/dspark_qwen3_0_6b_sharegpt_offline_ascend.sh

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54 no_proxy=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DATASET="${DATASET:-sharegpt}"      # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_qwen3_0_6b_sharegpt_ascend_offline}"
HIDDEN_STATES_DIR="${HIDDEN_STATES_DIR:-$OUTPUT_DIR/hidden_states}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-6e-4}"
CONCURRENCY="${CONCURRENCY:-16}"

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 25}"
DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-sdpa}"

# Markov + confidence head settings
MARKOV_RANK="${MARKOV_RANK:-256}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"   # vanilla | gated | rnn
LOSS_FN="${LOSS_FN:-{\"ce\": 0.1, \"tv\": 0.9}}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"

# Offline reuses the same Ascend device sequentially. Device 7 is the default. In docker, 14 and 15 will be mapped to 0 and 1.
ASCEND_NPUS="${ASCEND_NPUS:-0,1}"
NUM_NPUS="${NUM_NPUS:-2}"

# Extra vLLM arguments for Ascend. Matches the runnable TYS 8B server pattern.
VLLM_EXTRA_ARGS=(--data-parallel-size 2)
# =======================================

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Stopping vLLM server..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

echo "=== Step 2: Launching vLLM server on Ascend NPU(s): $ASCEND_NPUS ==="
env ASCEND_RT_VISIBLE_DEVICES="$ASCEND_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
until curl --noproxy localhost,127.0.0.1 -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    echo "vLLM not ready yet..."
    sleep 5
done
echo "vLLM server ready."

echo "=== Step 3: Generating hidden states ==="
python scripts/data_generation_offline.py \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HIDDEN_STATES_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency "$CONCURRENCY" \
    --validate-outputs

echo "=== Step 4: Stopping vLLM server ==="
cleanup
unset VLLM_PID
trap - EXIT
echo "vLLM server stopped. Ascend NPU(s) freed for training."

echo "=== Step 5: Training DSpark on Ascend NPU(s): $ASCEND_NPUS ==="
env TASK_QUEUE_ENABLE=2 ASCEND_RT_VISIBLE_DEVICES="$ASCEND_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-path "$HIDDEN_STATES_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
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
    --on-missing raise

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
