#!/usr/bin/env bash

set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=configs/train/aligndetr/resnet50.a1_in1k/0.yaml
GPU_IDS="6,7"                 
NUM_GPUS=""         
NUM_MACHINES=1
MACHINE_RANK=0
DIST_URL="auto"
RESUME=""
EXTRA_OPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG="$2"; shift 2 ;;
        --gpu-ids)
            GPU_IDS="$2"; shift 2 ;;
        -g|--gpus)
            NUM_GPUS="$2"; shift 2 ;;
        -r|--resume)
            RESUME="--resume"; shift ;;
        --num-machines)
            NUM_MACHINES="$2"; shift 2 ;;
        --machine-rank)
            MACHINE_RANK="$2"; shift 2 ;;
        --dist-url)
            DIST_URL="$2"; shift 2 ;;
        --)
            shift; EXTRA_OPTS+=("$@"); break ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config/-c is required." >&2
    echo "Usage: bash src/scripts/train.sh -c <config.yaml> [-g <num_gpus>] [--resume] ..." >&2
    exit 1
fi

if [[ -n "$GPU_IDS" ]]; then
    # Derive count from the comma-separated list: "0,2,3" → 3
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')
    echo "GPU IDs: ${GPU_IDS}  (${NUM_GPUS} GPU(s))"
elif [[ -z "$NUM_GPUS" ]]; then
    # Auto-detect from nvidia-smi
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
        echo "Auto-detected ${NUM_GPUS} GPU(s)."
    else
        NUM_GPUS=1
        echo "nvidia-smi not found; defaulting to 1 GPU."
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "  HW2 Training"
echo "  Config       : ${CONFIG}"
echo "  GPU IDs      : ${GPU_IDS:-<from YAML or auto>}"
echo "  GPUs/machine : ${NUM_GPUS}"
echo "  Machines     : ${NUM_MACHINES}"
echo "  Machine rank : ${MACHINE_RANK}"
echo "  dist-url     : ${DIST_URL}"
echo "  Resume       : ${RESUME:-no}"
if [[ ${#EXTRA_OPTS[@]} -gt 0 ]]; then
    echo "  Extra opts   : ${EXTRA_OPTS[*]}"
fi
echo "============================================================"

GPU_IDS_ARG=""
[[ -n "$GPU_IDS" ]] && GPU_IDS_ARG="--gpu-ids ${GPU_IDS}"

python train.py \
    --config        "${CONFIG}" \
    --num-gpus      "${NUM_GPUS}" \
    --num-machines  "${NUM_MACHINES}" \
    --machine-rank  "${MACHINE_RANK}" \
    --dist-url      "${DIST_URL}" \
    ${RESUME} \
    ${GPU_IDS_ARG} \
    "${EXTRA_OPTS[@]+"${EXTRA_OPTS[@]}"}"
