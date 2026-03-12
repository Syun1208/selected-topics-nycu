#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root (hw1/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Default arguments
MODEL="resnet_nn:seresnextaa101d_32x8d.sw_in12k_ft_in1k_288"  # resnet18 | resnet34 | resnet50 | resnet101 | resnet152
CONFIG="${CONFIG:-configs/train/${MODEL}.yaml}"
SEED=42
NPROC=1   # number of GPUs; set NPROC=4 for 4-GPU training
DEVICES=7  # comma-separated GPU IDs, e.g. DEVICES=0,1,2,3
LOCAL_RANK=7

export CUDA_VISIBLE_DEVICES="$DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


if [ "$NPROC" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC" train.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        "$@"
else
    python train.py \
        --config "$CONFIG" \
        --seed "$SEED" \
        "$@"
fi
