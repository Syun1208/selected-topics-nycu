set -euo pipefail

CONFIG="configs/test/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0.yaml"
N_IMAGES=6
TARGET_LAYER="backbone.model.layer4"
SCORE_THRESHOLD=0.3
GPU_IDS="2"
SEED="42"
SHOW=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG="$2"; shift 2 ;;
        -n|--n-images)
            N_IMAGES="$2"; shift 2 ;;
        --target-layer)
            TARGET_LAYER="$2"; shift 2 ;;
        --score-threshold)
            SCORE_THRESHOLD="$2"; shift 2 ;;
        --gpu-ids)
            GPU_IDS="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --show)
            SHOW="--show"; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: -c / --config is required." >&2
    echo "Usage: bash src/scripts/visualize.sh -c <config.yaml> [OPTIONS]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Resolve config relative to PROJECT_ROOT if it's a relative path
[[ "$CONFIG" != /* ]] && CONFIG="${PROJECT_ROOT}/${CONFIG}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG" >&2
    exit 1
fi

EXTRA_ARGS=()
[[ -n "$SEED" ]] && EXTRA_ARGS+=(--seed "$SEED")
[[ -n "$SHOW" ]] && EXTRA_ARGS+=("$SHOW")

echo "============================================================"
echo "  HW2 Visualization  (GradCAM + t-SNE)"
echo "  Config          : ${CONFIG}"
echo "  Images dir      : <from YAML>"
echo "  N images        : ${N_IMAGES}"
echo "  Target layer    : ${TARGET_LAYER}"
echo "  Score threshold : ${SCORE_THRESHOLD}"
echo "  GPU IDs         : ${GPU_IDS}"
echo "  Seed            : ${SEED:-<random>}"
echo "  Show            : ${SHOW:-no}"
echo "============================================================"

python visualize.py \
    --config            "${CONFIG}" \
    --n-images          "${N_IMAGES}" \
    --target-layer      "${TARGET_LAYER}" \
    --score-threshold   "${SCORE_THRESHOLD}" \
    --gpu-ids           "${GPU_IDS}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
