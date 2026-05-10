#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATA_DIR="${PROJECT_ROOT}/data"
EXPORT_TRAIN_PATH="${PROJECT_ROOT}/data/processed/train"
EXPORT_VAL_PATH="${PROJECT_ROOT}/data/processed/val"
EXPORT_TEST_PATH="${PROJECT_ROOT}/data/processed/test_release"
EXPORT_TEST_JSON_PATH="${PROJECT_ROOT}/data/processed/test_image_name_to_ids.json"
REPORT_DIR="${PROJECT_ROOT}/data/processed/reports"
GPUS="6"
N_AUGMENTS="2"
IMG_SIZE="1024"
VAL_RATIO="0.2"
VAL_SEED="42"
SKIP_CLEANING=""
SKIP_AUGMENT=""
ALLOW_WARNINGS="--allow-warnings"
LOG_LEVEL="INFO"

LOG_DIR="${PROJECT_ROOT}/logs/cleaner"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)             DATA_DIR="$2";             shift 2 ;;
        --export-train-path)    EXPORT_TRAIN_PATH="$2";    shift 2 ;;
        --export-val-path)      EXPORT_VAL_PATH="$2";      shift 2 ;;
        --export-test-path)     EXPORT_TEST_PATH="$2";     shift 2 ;;
        --export-test-json-path) EXPORT_TEST_JSON_PATH="$2"; shift 2 ;;
        --report-dir)           REPORT_DIR="$2";           shift 2 ;;
        --gpus)                 GPUS="$2";                 shift 2 ;;
        --n-augments)           N_AUGMENTS="$2";           shift 2 ;;
        --img-size)             IMG_SIZE="$2";             shift 2 ;;
        --val-ratio)            VAL_RATIO="$2";            shift 2 ;;
        --val-seed)             VAL_SEED="$2";             shift 2 ;;
        --skip-cleaning)        SKIP_CLEANING="--skip-cleaning"; shift ;;
        --skip-augment)         SKIP_AUGMENT="--skip-augment";   shift ;;
        --no-allow-warnings)    ALLOW_WARNINGS="";         shift ;;
        --log-level)            LOG_LEVEL="$2";            shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -20 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if command -v conda &>/dev/null && conda env list | grep -q "selected-topics"; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate selected-topics
fi

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "  Data Cleaning + Augmentation Pipeline"
echo "========================================================"
echo "  data-dir            : ${DATA_DIR}"
echo "  export-train-path   : ${EXPORT_TRAIN_PATH}"
echo "  export-val-path     : ${EXPORT_VAL_PATH}"
echo "  export-test-path    : ${EXPORT_TEST_PATH}"
echo "  export-test-json    : ${EXPORT_TEST_JSON_PATH}"
echo "  report-dir          : ${REPORT_DIR}"
echo "  val-ratio           : ${VAL_RATIO}"
echo "  gpus                : ${GPUS}"
echo "  n-augments          : ${N_AUGMENTS}"
echo "  img-size            : ${IMG_SIZE}"
echo "  skip-cleaning       : ${SKIP_CLEANING:-no}"
echo "  skip-augment        : ${SKIP_AUGMENT:-no}"
echo "  log-file            : ${LOG_FILE}"
echo "========================================================"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" python src/data/processors/compose.py \
    --data-dir              "${DATA_DIR}"               \
    --export-train-path     "${EXPORT_TRAIN_PATH}"      \
    --export-val-path       "${EXPORT_VAL_PATH}"        \
    --export-test-path      "${EXPORT_TEST_PATH}"       \
    --export-test-json-path "${EXPORT_TEST_JSON_PATH}"  \
    --report-dir            "${REPORT_DIR}"             \
    --val-ratio             "${VAL_RATIO}"              \
    --val-seed              "${VAL_SEED}"               \
    --gpus                  "${GPUS}"                   \
    --n-augments            "${N_AUGMENTS}"             \
    --img-size              "${IMG_SIZE}"               \
    --log-level             "${LOG_LEVEL}"              \
    --log-file              "${LOG_FILE}"               \
    ${SKIP_CLEANING}                                    \
    ${SKIP_AUGMENT}                                     \
    ${ALLOW_WARNINGS} 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================================"
echo "Pipeline finished."
echo "  Train  → ${EXPORT_TRAIN_PATH}"
echo "  Val    → ${EXPORT_VAL_PATH}"
echo "  Test   → ${EXPORT_TEST_PATH}"
echo "  JSON   → ${EXPORT_TEST_JSON_PATH}"
echo "  Log    → ${LOG_FILE}"
echo "========================================================"
