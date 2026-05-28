#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source .venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="5"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

VENV_SITE="$PWD/.venv/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$VENV_SITE/nvidia/nvjitlink/lib:$VENV_SITE/nvidia/cusparse/lib:$VENV_SITE/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

# Baseline
# CONFIG=conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-a.yaml

# CONFIG=conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-024-d-adaptive.yaml
# CONFIG=conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-024-d-adaptive-light.yaml
# CONFIG=conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-024-d-recip-union.yaml
# CONFIG=conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-024-d-coarse-verify.yaml
# CONFIG=conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-024-d-match-prune.yaml
# CONFIG=conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-024-d-match-prune-strong.yaml
# CONFIG="conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-024-d-grid-filter.yaml"
# CONFIG="conf/pipeline/imc2025/gluestick/gluestick-001.yaml"
# CONFIG="conf/pipeline/imc2025/cascade/cascade-xfeat-mast3r-001.yaml"
# CONFIG="conf/pipeline/imc2025/ccgraph/ccgraph-001.yaml"
# CONFIG="conf/pipeline/imc2025/ccgraph/colmap-tune-001.yaml"
# CONFIG="conf/pipeline/imc2025/multiscale/multiscale-2048.yaml"
# CONFIG="conf/pipeline/imc2025/cascade/cascade-xfeat-mast3r-001.yaml"
# CONFIG="conf/pipeline/imc2025/mast3rmpsfmsparse/mast3rmpsfmsparse-baseline.yaml"
# Best performance
CONFIG="/home/longpm/works/selected-topics-nycu/final-project/conf/pipeline/imc2025/combination/baseline+moretopk.yaml"
DATASETS=ETs

IFS=',' read -ra _gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
NGPU=0
for id in "${_gpu_ids[@]}"; do
  [[ -n "${id// /}" ]] && NGPU=$((NGPU + 1))
done

echo "[run.sh] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (nproc=$NGPU)"
echo "[run.sh] CONFIG=$CONFIG  DATASETS=$DATASETS"

if (( NGPU > 1 )); then
  echo "[run.sh] Multi-GPU -> torchrun kernel.py --dist"
  torchrun --nnodes 1 --nproc_per_node "$NGPU" --standalone \
    -m kernel -p "$CONFIG" -d imc2025train --datasets "$DATASETS" --dist "$@"

  echo "[run.sh] Computing score on submission.csv ..."
  python -c "
import utils.imc25.metric
from data import DEFAULT_DATASET_DIR
final, per_ds = utils.imc25.metric.score(
    gt_csv=DEFAULT_DATASET_DIR/'train_labels.csv',
    user_csv='submission.csv',
    thresholds_csv=DEFAULT_DATASET_DIR/'train_thresholds.csv',
    mask_csv=None, inl_cf=0, strict_cf=-1, verbose=True,
)
print('final_score:', final)
print('dataset_scores:', per_ds)
"
else
  echo "[run.sh] Single-GPU -> evaluate_imc2025.py"
  python evaluate_imc2025.py -c "$CONFIG" --datasets "$DATASETS" "$@"
fi

echo "[run.sh] Done. CONFIG=$CONFIG"
