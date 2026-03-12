#!/usr/bin/env bash

set -euo pipefail

CONFIG="configs/test/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0.yaml"
GPU_IDS="0,2,4,6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "  HW2 Test"
echo "  Config  : ${CONFIG}"
echo "  GPU IDs : ${GPU_IDS}"
echo "============================================================"

python test.py --config "${CONFIG}" --gpu-ids "${GPU_IDS}"
