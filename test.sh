#!/bin/bash
# =============================================================================
# HuBMAP - Inference / submission pipeline
#
# Usage:
#   bash test.sh                                   — latest r0 checkpoint
#   bash test.sh work_dirs/r0/iter_9072.pth        — specific checkpoint
#   bash test.sh work_dirs/r0/iter_9072.pth 5      — specific checkpoint, GPU 5
#
# Output: work_dirs/r0_test/test-results.json
# =============================================================================
set -e

CONDA_ENV=/home/longpm/miniconda3/envs/hw3
CODE_DIR=/home/longpm/works/kaggle-hubmap-hacking-the-human-vasculature
DATA_DIR=$CODE_DIR/data
TEST_IMGS=$DATA_DIR/test_release
TEST_META=$DATA_DIR/test_image_name_to_ids.json

CKPT=${1:-$(ls -t "$CODE_DIR/work_dirs/r0/iter_"*.pth 2>/dev/null | head -1)}
GPU_ID=4
OUT_DIR=$CODE_DIR/work_dirs/r0_test

export CUDA_VISIBLE_DEVICES=$GPU_ID

if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in work_dirs/r0/. Pass one explicitly."
    exit 1
fi

mkdir -p "$OUT_DIR"

echo ""
echo "======================================================"
echo " HuBMAP Inference"
echo "  Checkpoint : $CKPT"
echo "  GPU        : $GPU_ID"
echo "  Output     : $OUT_DIR"
echo "======================================================"


# ── Step 1: Create COCO-format annotation for test images ──────────────────
echo ""
echo "[1/3] Building test_coco.json ..."
$CONDA_ENV/bin/python3 - <<PYEOF
import json

with open('$TEST_META') as f:
    images = json.load(f)

coco = {
    "images": images,
    "annotations": [],
    "categories": [
        {"id": 1, "name": "class1", "supercategory": "object"},
        {"id": 2, "name": "class2", "supercategory": "object"},
        {"id": 3, "name": "class3", "supercategory": "object"},
        {"id": 4, "name": "class4", "supercategory": "object"},
    ]
}

out = '$DATA_DIR/test_coco.json'
with open(out, 'w') as f:
    json.dump(coco, f)
print(f"  {len(images)} images → {out}")
PYEOF


# ── Step 2: Write temporary test config ────────────────────────────────────
CFG_TMP=/tmp/r0_test_hubmap.py
cat > "$CFG_TMP" <<CFGEOF
_base_ = ['$CODE_DIR/configs/r0.py']

# Match exactly what was used during training (train.sh: NUM_CLASSES=4, CLASSES=[class1..4])
_num_classes = 4
_classes = ('class1', 'class2', 'class3', 'class4')

model = dict(detector=dict(
    bbox_head=dict(num_classes=_num_classes),
    test_cfg=dict(
        hflip_tta=True,
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.00001,
        # NMS IoU bumped 0.5 → 0.6 — less aggressive culling for crowded vessels.
        # (Soft NMS was tried, no improvement.)
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=1100,
    )
))

# Remove LoadAnnotations — test_release has no GT annotations
_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(768, 768), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='$DATA_DIR/',
        ann_file='test_coco.json',
        data_prefix=dict(img='$TEST_IMGS/'),
        metainfo=dict(classes=_classes),
        test_mode=True,
        pipeline=_test_pipeline,
        backend_args=None))

default_hooks = dict(visualization=dict(type='DetVisualizationHook', draw=False))

test_evaluator = [dict(
    type='FastCocoMetric',
    ann_file='$DATA_DIR/test_coco.json',
    metric=['bbox', 'segm'],
    format_only=True,
    outfile_prefix='$OUT_DIR/r0',
    backend_args=None,
    classwise=True)]
CFGEOF


# ── Step 3: Run inference ──────────────────────────────────────────────────
echo ""
echo "[2/3] Running inference ..."
cd "$CODE_DIR"
$CONDA_ENV/bin/python3 test.py \
    "$CFG_TMP" \
    "$CKPT" \
    --work-dir "$OUT_DIR"


# ── Step 4: Build test-results.json ─────────────────────────────────────────
# Codabench eval scores mAP across all 4 trained categories (class1..class4),
# not just blood_vessel — so we keep ALL categories in the submission.
# (Empirical: dropping/merging categories took the score from 0.6187 to 0.000.)
echo ""
echo "[3/3] Building test-results.json ..."
$CONDA_ENV/bin/python3 - <<PYEOF
import json, os

segm_f = '$OUT_DIR/r0.segm.json'
if not os.path.exists(segm_f):
    raise FileNotFoundError(f"segm output not found: {segm_f}")

with open(segm_f) as f:
    segm = json.load(f)

# Ensure RLE counts is a plain string, not bytes (some pycocotools versions emit bytes)
for s in segm:
    seg = s.get('segmentation', {})
    if isinstance(seg.get('counts'), bytes):
        seg['counts'] = seg['counts'].decode('utf-8')

out = '$OUT_DIR/test-results.json'
with open(out, 'w') as f:
    json.dump(segm, f)

from collections import Counter
dist = dict(Counter(s['category_id'] for s in segm))
print(f"  {len(segm)} predictions ({dist}) → {out}")
PYEOF


echo ""
echo "======================================================"
echo " Done."
echo " Submission : $OUT_DIR/test-results.json"
echo "======================================================"
