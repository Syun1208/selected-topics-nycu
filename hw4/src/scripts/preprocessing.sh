#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-/home/longpm/miniconda3/envs/hw3/bin/python3.10}"

DATA_DIR="data/hw4_realse_dataset"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        *) shift ;;
    esac
done

cd "${PROJECT_ROOT}"

"${PYTHON}" preprocessing.py --data-dir "${DATA_DIR}"
