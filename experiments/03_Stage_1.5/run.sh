#!/bin/bash
# Stage 1.5 Training Launcher
# Usage: ./run.sh [--resume checkpoint.pt]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
CONFIG="$PROJECT_ROOT/configs/stage1_5_config.json"
RESUME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--resume checkpoint.pt] [--config config.json]"
            echo ""
            echo "Options:"
            echo "  --resume    Resume from checkpoint"
            echo "  --config    Path to config JSON (default: configs/stage1_5_config.json)"
            echo "  -h, --help  Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

echo "==================================="
echo "  Stage 1.5 Training"
echo "==================================="
echo "Config: $CONFIG"
echo "Resume: ${RESUME:-None}"
echo ""

if [ -n "$RESUME" ]; then
    python experiments/01_denoising_poc/train_stage1_5.py \
        --config "$CONFIG" \
        --resume "$RESUME"
else
    python experiments/01_denoising_poc/train_stage1_5.py \
        --config "$CONFIG"
fi

echo ""
echo "==================================="
echo "  Training finished"
echo "==================================="
