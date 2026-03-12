#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./scripts/pipeline_one.sh <model_name>"
    echo ""
    echo "Available models:"
    echo "  baseline            - tds_conv_ctc (default)"
    echo "  small_learnable     - Transformer small + learnable pos"
    echo "  small_sinusoidal    - Transformer small + sinusoidal pos"
    echo "  large_learnable     - Transformer large + learnable pos"
    echo "  large_sinusoidal    - Transformer large + sinusoidal pos"
    exit 1
fi

MODEL=$1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create logs directory if it doesn't exist
mkdir -p logs

case $MODEL in
    baseline)
        MODEL_FLAG=""
        MODEL_NAME="tds_conv_ctc"
        ;;
    small_learnable|small_sinusoidal|large_learnable|large_sinusoidal)
        MODEL_FLAG="model=transformer_$MODEL"
        MODEL_NAME="transformer_$MODEL"
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

timestamp=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="logs/train_${MODEL_NAME}_${timestamp}.log"
TEST_LOG="logs/test_${MODEL_NAME}_${timestamp}.log"

echo "========================================"
echo "Pipeline: $MODEL_NAME"
echo "========================================"

# Training
echo ""
echo "[1/2] Training..."
echo "Log file: ${TRAIN_LOG}"
{
    echo "Starting training: ${MODEL_NAME} at $(date)"
    python -m emg2qwerty.train \
        $MODEL_FLAG \
        user=single_user \
        trainer.accelerator=mps \
        trainer.devices=1
    echo "Completed training at $(date)"
} 2>&1 | tee "$TRAIN_LOG"

# Find latest checkpoint
CHECKPOINT=$(find logs -name "last.ckpt" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found in logs/"
    exit 1
fi

echo ""
echo "Checkpoint: $CHECKPOINT"

# Testing
echo ""
echo "[2/2] Testing..."
echo "Log file: ${TEST_LOG}"
{
    echo "Starting test at $(date)"
    python -m emg2qwerty.train \
        user=single_user \
        checkpoint="$CHECKPOINT" \
        train=False \
        trainer.accelerator=mps \
        decoder=ctc_greedy
    echo "Completed test at $(date)"
} 2>&1 | tee "$TEST_LOG"

echo ""
echo "========================================"
echo "Pipeline complete: $MODEL_NAME"
echo "Checkpoint: $CHECKPOINT"
echo "Train log: ${TRAIN_LOG}"
echo "Test log: ${TEST_LOG}"
echo "========================================"
