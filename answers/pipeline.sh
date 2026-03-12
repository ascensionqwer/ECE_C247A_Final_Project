#!/bin/bash

# Complete EMG2QWERTY Pipeline
# Trains and tests all models

set -e

echo "=========================================="
echo "EMG2QWERTY Complete Pipeline"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Train all model variants"
echo "  2. Test all trained models"
echo ""
echo "Accelerator: MPS (Apple Silicon)"
echo ""

# Required for MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Parse arguments
SKIP_TRAIN=${SKIP_TRAIN:-false}
SKIP_TEST=${SKIP_TEST:-false}

for arg in "$@"; do
    case $arg in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --help|-h)
            echo "Usage: ./pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-train    Skip training phase"
            echo "  --skip-test     Skip testing phase"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  SKIP_TRAIN=true   Same as --skip-train"
            echo "  SKIP_TEST=true    Same as --skip-test"
            exit 0
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Phase 1: Training
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 1: TRAINING"
    echo "=========================================="
    echo ""
    
    "$SCRIPT_DIR/train_all.sh"
else
    echo "Skipping training phase (--skip-train)"
fi

# Phase 2: Testing
if [ "$SKIP_TEST" = false ]; then
    echo ""
    echo "=========================================="
    echo "PHASE 2: TESTING"
    echo "=========================================="
    echo ""
    
    "$SCRIPT_DIR/test_all.sh" logs
else
    echo "Skipping testing phase (--skip-test)"
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results are saved in the logs/ directory."
echo "Check logs/YYYY-MM-DD/HH-MM-SS/ for:"
echo "  - checkpoints/  : Model weights"
echo "  - hydra_configs/: Configuration used"
echo ""
