#!/bin/bash
# =============================================================================
# HuBMAP - Hacking the Human Vasculature
# Training pipeline following the 1st-place solution by tascj
# https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429060
#
# Model ensemble (best → weakest single model):
#   1. RTMDet-x          (configs/r0.py)            — best single model
#   2. YOLOX-x           (configs/y0.py)
#   3. MaskRCNN + SwinS  (configs/m0.py)
#   4. SwinB + RTMDet    (configs/coco/sb.py → configs/sb0.py)  [2-stage]
#   5. SwinS + RTMDet    (configs/coco/s.py  → configs/s0.py)   [2-stage]
#
# Usage:
#   bash train.sh           — run all stages sequentially
#   bash train.sh r0        — run only the RTMDet-x stage
#   bash train.sh r0 y0     — run selected stages
#
# Stages: r0 | y0 | m0 | sb | s
# =============================================================================
set -e

# ─── Paths ────────────────────────────────────────────────────────────────────
CONDA_ENV=/home/longpm/miniconda3/envs/hw3
TORCHRUN=$CONDA_ENV/bin/torchrun
CODE_DIR=/home/longpm/works/kaggle-hubmap-hacking-the-human-vasculature
DATA_DIR=$CODE_DIR/data_augmented
COCO_DATA=$DATA_DIR/custom_coco
TRAIN_IMGS=$COCO_DATA/images/train
VAL_IMGS=$COCO_DATA/images/val
VAL_ANN=$COCO_DATA/annotations/val.json

GPU_IDS=3,7
export CUDA_VISIBLE_DEVICES=$GPU_IDS

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400  # 4 hours
export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_TIMEOUT=1800                       # 30 min per collective op
export MMENGINE_SKIP_DATALOADER_ADVANCE=1      # skip slow dataloader replay on resume
export OPENCV_LOG_LEVEL=ERROR                  # silence libtiff ExtraSamples warnings

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
MASTER_PORT=29500


NUM_CLASSES=4
CLASSES="[class1,class2,class3,class4]"


build_coco_if_missing() {
    # augment.py writes per-instance dirs to $DATA_DIR/train/<uuid>/{image.tif,classN.tif}.
    # mmdet wants a COCO-format dataset under $COCO_DATA — build it once if absent.
    if [ -f "$COCO_DATA/annotations/train.json" ]; then
        echo "[Setup] $COCO_DATA already built — skipping conversion."
        return
    fi
    if [ ! -d "$DATA_DIR/train" ]; then
        echo "[Setup] ERROR: $DATA_DIR/train not found. Run augment.sh first." >&2
        exit 1
    fi
    echo "[Setup] Building COCO-format dataset at $COCO_DATA ..."
    "$CONDA_ENV/bin/python" "$CODE_DIR/tools/build_augmented_coco.py" \
        --augmented-train "$DATA_DIR/train" \
        --reference-coco  "$CODE_DIR/data/custom_coco" \
        --out             "$COCO_DATA"
}

setup_symlinks() {
    echo "[Setup] Symlinking annotation files in $DATA_DIR ..."
    ln -sf "$COCO_DATA/annotations/train.json" "$DATA_DIR/dtrain0i.json"
    ln -sf "$COCO_DATA/annotations/train.json" "$DATA_DIR/dtrain1i.json"
    ln -sf "$COCO_DATA/annotations/train.json" "$DATA_DIR/dtrain_dataset2.json"
    ln -sf "$COCO_DATA/annotations/train.json" "$DATA_DIR/dtrain_dataset2_dropdup.json"
    ln -sf "$COCO_DATA/annotations/train.json" "$DATA_DIR/dtrainval.json"
    ln -sf "$COCO_DATA/annotations/val.json"   "$DATA_DIR/dval0i.json"
    ln -sf "$COCO_DATA/annotations/val.json"   "$DATA_DIR/dval1i.json"
    echo "[Setup] Done."
}


run_train() {
    local config=$1; shift
    echo ""
    echo "  Config      : $config"
    echo "  GPUs        : $GPU_IDS  (count=$NUM_GPUS, port=$MASTER_PORT)"
    echo "  Extra args  : $*"
    echo ""
    "$TORCHRUN" \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        "$CODE_DIR/train.py" "$config" \
        --amp \
        --launcher pytorch \
        --data-root "$DATA_DIR" \
        --resume auto \
        "$@"
    MASTER_PORT=$((MASTER_PORT + 1))
}


CONCAT_OPTS=(
    "train_dataloader.dataset.datasets.0.data_prefix.img=$TRAIN_IMGS"
    "train_dataloader.dataset.datasets.1.data_prefix.img=$TRAIN_IMGS"
    "train_dataloader.dataset.datasets.0.metainfo.classes=$CLASSES"
    "train_dataloader.dataset.datasets.1.metainfo.classes=$CLASSES"
    "val_dataloader.dataset.data_prefix.img=$VAL_IMGS"
    "val_dataloader.dataset.metainfo.classes=$CLASSES"
    "val_evaluator.0.ann_file=$VAL_ANN"
)

# Determine which stages to run (default: all)
STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=(r0 y0 m0 sb s)
fi

run_stage() { [[ " ${STAGES[*]} " =~ " $1 " ]]; }

cd "$CODE_DIR"
build_coco_if_missing
setup_symlinks

# =============================================================================
# Stage 1 — RTMDet-x  (BEST SINGLE MODEL)
#   Backbone : CSPNeXt-x pretrained on COCO
#   Neck     : CSPNeXtPAFPN
#   Head     : RTMDetSepBNHead + FCNMaskHead (binary per-class masks)
#   img_scale: 768×768
#   batch    : 8 images / GPU, 8 GPUs → 64 total
# =============================================================================
if run_stage r0; then
    echo ""
    echo "======================================================"
    echo " Stage 1 | RTMDet-x  (configs/r0.py)"
    echo "======================================================"
    run_train configs/r0.py \
        --work-dir work_dirs/r0 \
        --cfg-options \
            "${CONCAT_OPTS[@]}" \
            "model.detector.bbox_head.num_classes=$NUM_CLASSES"
fi


if run_stage y0; then
    echo ""
    echo "======================================================"
    echo " Stage 2 | YOLOX-x  (configs/y0.py)"
    echo "======================================================"
    run_train configs/y0.py \
        --work-dir work_dirs/y0 \
        --cfg-options \
            "${CONCAT_OPTS[@]}" \
            "model.detector.bbox_head.num_classes=$NUM_CLASSES"
fi



if run_stage m0; then
    echo ""
    echo "======================================================"
    echo " Stage 3 | MaskRCNN + SwinS  (configs/m0.py)"
    echo "======================================================"
    run_train configs/m0.py \
        --work-dir work_dirs/m0 \
        --cfg-options \
            "${CONCAT_OPTS[@]}" \
            "model.detector.roi_head.bbox_head.num_classes=$NUM_CLASSES" \
            "model.detector.roi_head.mask_head.num_classes=$NUM_CLASSES"
fi


if run_stage sb; then
    if [ -d "$DATA_DIR/coco" ]; then
        echo ""
        echo "======================================================"
        echo " Stage 4a | COCO pretraining — SwinB  (configs/coco/sb.py)"
        echo "======================================================"
        run_train configs/coco/sb.py \
            --work-dir work_dirs/coco_sb
    else
        echo ""
        echo "[Stage 4a] SKIP — COCO dataset not found at $DATA_DIR/coco"
        echo "           sb0.py will use the default ImageNet pretrained checkpoint."
    fi

    echo ""
    echo "======================================================"
    echo " Stage 4b | SwinB + RTMDet fine-tune  (configs/sb0.py)"
    echo "======================================================"
    run_train configs/sb0.py \
        --work-dir work_dirs/sb0 \
        --cfg-options \
            "${CONCAT_OPTS[@]}" \
            "model.detector.bbox_head.num_classes=$NUM_CLASSES"
fi


if run_stage s; then
    if [ -d "$DATA_DIR/coco" ]; then
        echo ""
        echo "======================================================"
        echo " Stage 5a | COCO pretraining — SwinS  (configs/coco/s.py)"
        echo "======================================================"
        run_train configs/coco/s.py \
            --work-dir work_dirs/coco_s
    else
        echo ""
        echo "[Stage 5a] SKIP — COCO dataset not found at $DATA_DIR/coco"
    fi

    echo ""
    echo "======================================================"
    echo " Stage 5b | SwinS + RTMDet fine-tune  (configs/s0.py)"
    echo "======================================================"
    run_train configs/s0.py \
        --work-dir work_dirs/s0 \
        --cfg-options \
            "${CONCAT_OPTS[@]}" \
            "model.detector.bbox_head.num_classes=$NUM_CLASSES"
fi

echo ""
echo "======================================================"
echo " All requested stages complete."
echo " Checkpoints: $CODE_DIR/work_dirs/"
echo ""
echo " Next steps:"
echo "   1. Find best iteration from each model's training log"
echo "   2. Extract checkpoint:  python tools/dump_ckpt.py <work_dir> <iter>"
echo "   3. Run inference + ensemble (see test.py)"
echo "======================================================"
