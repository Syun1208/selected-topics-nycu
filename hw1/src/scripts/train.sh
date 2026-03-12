#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

MODEL="seresnextaa101d_32x8d.sw_in12k_ft_in1k_288_imbalance"
CONFIG="${CONFIG:-configs/train/${MODEL}.yaml}"
SEED=42
NPROC=1
DEVICES=3

echo "Using GPUs: $DEVICES"
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
