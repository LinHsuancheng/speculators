#!/bin/bash
# Online DSpark Training Script for Qwen3-0.6B on Ascend NPU
#
# Runs vLLM and training at the same time. This requires separate Ascend devices
# for vLLM and training. If only device 7 is available, use the offline script.
#
# Usage:
#   VLLM_NPUS=6 TRAIN_NPUS=7 bash examples/train/dspark_qwen3_0_6b_sharegpt_online_ascend.sh

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# ============ Configuration ============
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
DATASET="${DATASET:-sharegpt}"      # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_qwen3_0_6b_sharegpt_ascend}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-4}"

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

# Online mode needs separate device sets. Defaults assume training on device 7.
VLLM_NPUS="${VLLM_NPUS:-6}"
TRAIN_NPUS="${TRAIN_NPUS:-7}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-1}"

# Extra vLLM arguments. Tune these if KV cache memory is too large or too small.
VLLM_EXTRA_ARGS=(
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
    --max-model-len "$SEQ_LENGTH"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS:-8}"
)
# =======================================

if [[ "$VLLM_NPUS" == "$TRAIN_NPUS" ]]; then
    echo "Online mode requires distinct VLLM_NPUS and TRAIN_NPUS. Use offline mode when only one device is available." >&2
    exit 1
fi

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

echo "=== Step 2: Launching vLLM server on Ascend NPU(s): $VLLM_NPUS ==="
env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for vLLM server to be ready..."
until curl --noproxy localhost,127.0.0.1 -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    echo "vLLM not ready yet..."
    sleep 5
done
echo "vLLM server ready."

echo "=== Step 3: Training DSpark on Ascend NPU(s): $TRAIN_NPUS ==="
env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
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
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
