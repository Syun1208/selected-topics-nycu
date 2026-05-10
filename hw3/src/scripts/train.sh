#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="/home/longpm/miniconda3/envs/hw3/bin/python3.10"
TORCHRUN="/home/longpm/miniconda3/envs/hw3/bin/torchrun"

CONFIG="configs/train/cascade_mask_rcnn/x-101-64x4d-fpn/v1.yaml"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

cd "${PROJECT_ROOT}"

_read_cfg() {
    "${PYTHON}" -c \
        "import yaml; cfg = yaml.safe_load(open('${CONFIG}')); print($1)" 2>/dev/null
}

MODEL=$(_read_cfg 'cfg["model"]["name"]')
BACKBONE=$(_read_cfg 'cfg["model"]["backbone"]')
GPU_IDS=$(_read_cfg 'str(cfg.get("training",{}).get("gpu_ids","") or "").strip()')

N_GPUS=1
if [[ -n "${GPU_IDS}" ]]; then
    N_GPUS=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/${MODEL}/${BACKBONE}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Training config : ${CONFIG}"
echo "Model           : ${MODEL}"
echo "Backbone        : ${BACKBONE}"
echo "GPUs            : ${GPU_IDS:-<default>} (${N_GPUS}x)"
echo "Log file        : ${LOG_FILE}"
echo "========================================"

PYWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"

if [[ "${N_GPUS}" -gt 1 ]]; then
    echo "Launching DDP with torchrun (${N_GPUS} GPUs: ${GPU_IDS})"
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTHONWARNINGS="${PYWARNINGS}" \
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=0 \
    "${TORCHRUN}" --nproc_per_node="${N_GPUS}" --master_port=29500 \
        train.py --config "${CONFIG}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "${LOG_FILE}"
else
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PYTHONWARNINGS="${PYWARNINGS}" \
    "${PYTHON}" train.py --config "${CONFIG}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "${LOG_FILE}"
fi

echo ""
echo "========================================"
echo "Training finished."
echo "  Log -> ${LOG_FILE}"
echo "========================================"
