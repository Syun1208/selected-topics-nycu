#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-/home/longpm/miniconda3/envs/hw3/bin/python3.10}"

CONFIG="configs/train/promptir/v2.yaml"
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
GPU_IDS=$(_read_cfg 'cfg.get("training",{}).get("num_gpus","")')

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/logs/${MODEL}/${BACKBONE}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "Training config : ${CONFIG}"
echo "Model           : ${MODEL}"
echo "Backbone        : ${BACKBONE}"
echo "GPUs            : ${GPU_IDS:-<default>}"
echo "Log file        : ${LOG_FILE}"
echo "========================================"

PYWARNINGS="ignore::DeprecationWarning,ignore::FutureWarning,ignore::UserWarning"
PYTHONWARNINGS="${PYWARNINGS}" \
    "${PYTHON}" train.py --config "${CONFIG}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================"
echo "Training finished."
echo "  Log -> ${LOG_FILE}"
echo "========================================"
