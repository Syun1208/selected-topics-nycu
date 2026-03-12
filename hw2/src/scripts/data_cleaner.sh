#!/usr/bin/env bash

# clean-only, augment-only: không bao giờ set = 1
# deblurry, denoise, deblurry-denoise: set = 1 kèm _require_* check
# full: set = 1 chỉ khi checkpoint thực sự tồn tại (graceful fallback)

set -euo pipefail
cd "$(dirname "$0")/../.."

PRESET="augment-only" # clean-only
TRAIN_JSON="data/train.json"
IMAGES_DIR="data/train"
CLEANVISION_CSV="notebooks/outputs/cleanvision_issues.csv"
OUTPUT_JSON="data/clean_train.json"
OUTPUT_IMAGES_DIR="data/clean_train"
AUG_TARGET=30000
SEED=42
NUM_WORKERS=4
NAFNET_DEVICE="cuda"
NAFNET_GPU_IDS="3,5,7"
DEBLURRY_CHECKPOINT_PATH=""
DEBLURRY_CONFIG_PATH=""
DENOISE_CHECKPOINT_PATH=""
DENOISE_CONFIG_PATH=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)                   PRESET="$2";                    shift 2 ;;
        --train-json)               TRAIN_JSON="$2";                shift 2 ;;
        --images-dir)               IMAGES_DIR="$2";                shift 2 ;;
        --cleanvision-csv)          CLEANVISION_CSV="$2";           shift 2 ;;
        --output-json)              OUTPUT_JSON="$2";               shift 2 ;;
        --output-images-dir)        OUTPUT_IMAGES_DIR="$2";         shift 2 ;;
        --aug-target)               AUG_TARGET="$2";                shift 2 ;;
        --seed)                     SEED="$2";                      shift 2 ;;
        --num-workers)              NUM_WORKERS="$2";               shift 2 ;;
        --nafnet-device)            NAFNET_DEVICE="$2";             shift 2 ;;
        --nafnet-gpu-ids)           NAFNET_GPU_IDS="$2";            shift 2 ;;
        --deblurry-checkpoint-path) DEBLURRY_CHECKPOINT_PATH="$2"; shift 2 ;;
        --deblurry-config-path)     DEBLURRY_CONFIG_PATH="$2";     shift 2 ;;
        --denoise-checkpoint-path)  DENOISE_CHECKPOINT_PATH="$2";  shift 2 ;;
        --denoise-config-path)      DENOISE_CONFIG_PATH="$2";      shift 2 ;;
        --remove-dark|--remove-blurry|--remove-near-duplicates|\
        --repair-dark|--repair-blurry|\
        --deblurry-blurry|--denoise-noisy)
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
        --help|-h)
            python -m src.data.processors.data_cleaner --help
            exit 0 ;;
        *) echo "[data_cleaner.sh] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

_require_deblurry() {
    if [[ -z "${DEBLURRY_CHECKPOINT_PATH}" && -z "${DEBLURRY_CONFIG_PATH}" ]]; then
        echo "[data_cleaner.sh] ERROR: preset '${PRESET}' requires --deblurry-checkpoint-path or --deblurry-config-path" >&2
        echo "  Download NAFNet-GoPro-width64.pth from:" >&2
        echo "    https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC" >&2
        exit 1
    fi
}

_require_denoise() {
    if [[ -z "${DENOISE_CHECKPOINT_PATH}" && -z "${DENOISE_CONFIG_PATH}" ]]; then
        echo "[data_cleaner.sh] ERROR: preset '${PRESET}' requires --denoise-checkpoint-path or --denoise-config-path" >&2
        echo "  Download NAFNet-SIDD-width64.pth from:" >&2
        echo "    https://drive.google.com/file/d/14Fht4x2Ft4HEDMoBT4SRyiqgQ73YRa6I" >&2
        exit 1
    fi
}

_deblurry_flags() {
    local flags="--deblurry-blurry"
    [[ -n "${DEBLURRY_CHECKPOINT_PATH}" ]] && flags="${flags} --deblurry-checkpoint-path ${DEBLURRY_CHECKPOINT_PATH}"
    [[ -n "${DEBLURRY_CONFIG_PATH}" ]]     && flags="${flags} --deblurry-config-path ${DEBLURRY_CONFIG_PATH}"
    echo "${flags}"
}

_denoise_flags() {
    local flags="--denoise-noisy"
    [[ -n "${DENOISE_CHECKPOINT_PATH}" ]] && flags="${flags} --denoise-checkpoint-path ${DENOISE_CHECKPOINT_PATH}"
    [[ -n "${DENOISE_CONFIG_PATH}" ]]     && flags="${flags} --denoise-config-path ${DENOISE_CONFIG_PATH}"
    echo "${flags}"
}

USE_DEBLURRY=0
USE_DENOISE=0

case "$PRESET" in
    clean-only)
        FLAGS="--remove-dark --remove-near-duplicates --repair-blurry"
        ;;
    augment-only)
        FLAGS="--aug-target ${AUG_TARGET}"
        ;;
    deblurry)
        _require_deblurry
        USE_DEBLURRY=1
        FLAGS="--remove-dark --remove-near-duplicates \
               --aug-target ${AUG_TARGET}"
        ;;
    denoise)
        _require_denoise
        USE_DENOISE=1
        FLAGS="--remove-dark --remove-near-duplicates \
               --aug-target ${AUG_TARGET}"
        ;;
    deblurry-denoise)
        _require_deblurry
        _require_denoise
        USE_DEBLURRY=1
        USE_DENOISE=1
        FLAGS="--remove-dark --remove-near-duplicates \
               --aug-target ${AUG_TARGET}"
        ;;
    full)
        # full preset: chạy NAFNet chỉ khi checkpoint thực sự được cung cấp
        [[ -n "${DEBLURRY_CHECKPOINT_PATH}" || -n "${DEBLURRY_CONFIG_PATH}" ]] && USE_DEBLURRY=1
        [[ -n "${DENOISE_CHECKPOINT_PATH}"  || -n "${DENOISE_CONFIG_PATH}"  ]] && USE_DENOISE=1
        FLAGS="--remove-dark --remove-near-duplicates --repair-blurry --repair-dark \
               --aug-target ${AUG_TARGET}"
        ;;
    custom)
        FLAGS="${EXTRA_ARGS}"
        EXTRA_ARGS=""
        ;;
    *)
        echo "[data_cleaner.sh] Unknown preset: ${PRESET}" >&2
        echo "  Choices: clean-only | augment-only | deblurry | denoise | deblurry-denoise | full | custom" >&2
        exit 1
        ;;
esac

# Thêm NAFNet flags chỉ khi preset yêu cầu (USE_DEBLURRY/USE_DENOISE=1)
# Tránh inject --deblurry-blurry/--denoise-noisy vào mọi preset do default paths khác rỗng
if [[ "${USE_DEBLURRY}" -eq 1 ]]; then
    FLAGS="${FLAGS} $(_deblurry_flags)"
fi
if [[ "${USE_DENOISE}" -eq 1 ]]; then
    FLAGS="${FLAGS} $(_denoise_flags)"
fi

FLAGS="${FLAGS} ${EXTRA_ARGS}"

echo "============================================================"
echo "  HW2 Data Cleaner"
echo "  Preset              : ${PRESET}"
echo "  Train JSON          : ${TRAIN_JSON}"
echo "  Images dir          : ${IMAGES_DIR}"
echo "  CleanVision CSV     : ${CLEANVISION_CSV}"
echo "  Output JSON         : ${OUTPUT_JSON}"
echo "  Output images       : ${OUTPUT_IMAGES_DIR}"
echo "  Aug target          : ${AUG_TARGET} instances/class"
echo "  Num workers         : ${NUM_WORKERS} (0 = all cores)"
echo "  NAFNet device       : ${NAFNET_DEVICE}"
if [[ -n "${NAFNET_GPU_IDS}" ]]; then
echo "  NAFNet GPU IDs      : ${NAFNET_GPU_IDS}"
fi
if [[ -n "${DEBLURRY_CHECKPOINT_PATH}" ]]; then
echo "  Deblurry checkpoint : ${DEBLURRY_CHECKPOINT_PATH}"
fi
if [[ -n "${DEBLURRY_CONFIG_PATH}" ]]; then
echo "  Deblurry config     : ${DEBLURRY_CONFIG_PATH}"
fi
if [[ -n "${DENOISE_CHECKPOINT_PATH}" ]]; then
echo "  Denoise checkpoint  : ${DENOISE_CHECKPOINT_PATH}"
fi
if [[ -n "${DENOISE_CONFIG_PATH}" ]]; then
echo "  Denoise config      : ${DENOISE_CONFIG_PATH}"
fi
echo "  Extra flags         : ${FLAGS}"
echo "============================================================"

# shellcheck disable=SC2086
python -m src.data.processors.data_cleaner \
    --train-json          "${TRAIN_JSON}"        \
    --images-dir          "${IMAGES_DIR}"        \
    --cleanvision-csv     "${CLEANVISION_CSV}"   \
    --output-json         "${OUTPUT_JSON}"       \
    --output-images-dir   "${OUTPUT_IMAGES_DIR}" \
    --seed                "${SEED}"              \
    --num-workers         "${NUM_WORKERS}"       \
    --nafnet-device       "${NAFNET_DEVICE}"     \
    ${NAFNET_GPU_IDS:+--nafnet-gpu-ids "${NAFNET_GPU_IDS}"} \
    ${FLAGS}

echo ""
echo "Done. Update your YAML config to use the cleaned dataset:"
echo "  dataset:"
echo "    train_json:       ${OUTPUT_JSON}"
echo "    train_images_dir: ${OUTPUT_IMAGES_DIR}"
