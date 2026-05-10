#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="/home/longpm/miniconda3/envs/hw3/bin/python3.10"

CONFIG="configs/test/cascade_mask_rcnn/convnext_l/v1.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)   CONFIG="$2";   shift 2 ;;
        --gpu-ids)  GPU_IDS="$2";  shift 2 ;;
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
GPU_IDS="${GPU_IDS:-$(_read_cfg 'str(cfg.get("inference",{}).get("gpu_ids","0") or "0").strip()')}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/${MODEL}/${BACKBONE}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Test config  : ${CONFIG}"
echo "Model        : ${MODEL}"
echo "Backbone     : ${BACKBONE}"
echo "GPU IDs      : ${GPU_IDS}"
echo "Log file     : ${LOG_FILE}"
echo "========================================"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
"${PYTHON}" test.py \
    --config  "${CONFIG}" \
    --gpu_ids "${GPU_IDS}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================"
echo "Inference finished."
echo "  Log → ${LOG_FILE}"
echo "========================================"
