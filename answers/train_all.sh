#!/bin/bash

# Train all models for emg2qwerty project
# Uses MPS (Apple Silicon) acceleration

set -e

# Required for MPS fallback on operations not natively supported
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "EMG2QWERTY Training Pipeline"
echo "=========================================="
echo ""

# Function to run training with logging
train_model() {
    local model_name=$1
    local user_type=$2
    local model_config=$3
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/train_${model_name}_${user_type}_${timestamp}.log"
    
    echo "----------------------------------------"
    echo "Training: $model_name ($user_type)"
    echo "Log file: ${log_file}"
    echo "----------------------------------------"
    
    {
        echo "Starting training: ${model_name} at $(date)"
        python -m emg2qwerty.train \
            user="$user_type" \
            model="$model_config" \
            trainer.accelerator=mps \
            trainer.devices=1
        echo "Completed training: ${model_name} at $(date)"
    } 2>&1 | tee "$log_file"
    
    echo ""
    echo "Completed: $model_name ($user_type)"
    echo ""
}

# Available models
MODELS=(
    "tds_conv_ctc"
    "transformer_small_sinusoidal"
    "transformer_small_learnable"
    "transformer_large_sinusoidal"
    "transformer_large_learnable"
)

# Train single-user models
echo "=== SINGLE-USER MODELS ==="
for model in "${MODELS[@]}"; do
    train_model "$model" "single_user" "$model"
done

echo "=========================================="
echo "All training complete!"
echo "Check logs/ directory for checkpoints and log files"
echo "=========================================="
