#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-/home/longpm/miniconda3/envs/hw3/bin/python3.10}"

cd "${PROJECT_ROOT}"

"${PYTHON}" -m src.utils.visualization \
    --wandb-dir "${1:-wandb}" \
    --output-dir "${2:-charts/wandb_plots}" \
    --combined --smooth 5
