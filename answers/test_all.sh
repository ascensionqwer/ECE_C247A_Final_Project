#!/bin/bash

# Test all trained models for emg2qwerty project
# Uses MPS (Apple Silicon) acceleration

set -e

# Required for MPS fallback on operations not natively supported
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "EMG2QWERTY Testing Pipeline"
echo "=========================================="
echo ""

# Checkpoint directory - modify this to point to your trained checkpoints
# Format: logs/YYYY-MM-DD/HH-MM-SS/checkpoints/
CHECKPOINT_DIR=${1:-"logs"}

# Find all checkpoint directories
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory '$CHECKPOINT_DIR' not found"
    echo "Usage: ./test_all.sh [checkpoint_directory]"
    exit 1
fi

# Function to test a checkpoint
test_checkpoint() {
    local checkpoint_path=$1
    local decoder=$2
    local user_type=$3
    local checkpoint_name=$(basename "$checkpoint_path" .ckpt)
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="logs/test_${checkpoint_name}_${decoder}_${timestamp}.log"
    
    echo "----------------------------------------"
    echo "Testing: $checkpoint_path"
    echo "Decoder: $decoder, User: $user_type"
    echo "Log file: ${log_file}"
    echo "----------------------------------------"
    
    {
        echo "Starting test at $(date)"
        python -m emg2qwerty.train \
            user="$user_type" \
            checkpoint="'$checkpoint_path'" \
            train=False \
            trainer.accelerator=mps \
            decoder="$decoder"
        echo "Completed test at $(date)"
    } 2>&1 | tee "$log_file"
    
    echo ""
}

# Function to find best checkpoint in a directory
find_best_checkpoint() {
    local dir=$1
    # Look for last.ckpt or best checkpoint
    if [ -f "$dir/last.ckpt" ]; then
        echo "$dir/last.ckpt"
    elif ls "$dir"/*.ckpt 1> /dev/null 2>&1; then
        # Return first checkpoint found
        ls "$dir"/*.ckpt | head -1
    fi
}

echo "Searching for checkpoints in: $CHECKPOINT_DIR"
echo ""

# Find all checkpoint files
CHECKPOINTS=()
while IFS= read -r -d '' ckpt; do
    CHECKPOINTS+=("$ckpt")
done < <(find "$CHECKPOINT_DIR" -name "*.ckpt" -print0 2>/dev/null | sort -z)

if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    echo "Please train models first using ./train_all.sh"
    exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoint(s):"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "  - $ckpt"
done
echo ""

# Test each checkpoint with both decoders
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "=========================================="
    echo "Checkpoint: $checkpoint"
    echo "=========================================="
    
    # Determine user type from checkpoint path (heuristic)
    USER_TYPE="single_user"
    
    # Test with greedy decoding (fast)
    test_checkpoint "$checkpoint" "ctc_greedy" "$USER_TYPE"
    
    # Test with beam search (better accuracy)
    test_checkpoint "$checkpoint" "ctc_beam" "$USER_TYPE"
done

echo "=========================================="
echo "All testing complete!"
echo "Check logs/ directory for test log files"
echo "=========================================="
