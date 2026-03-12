#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root (hw1/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Default arguments
CONFIG="${CONFIG:-configs/test.yaml}"
SEED=42
DEVICES=7  

export CUDA_VISIBLE_DEVICES="$DEVICES"

python test.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    "$@"
