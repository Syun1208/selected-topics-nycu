#!/usr/bin/env bash
# =============================================================================
# test.sh -- Run the XFeat + COLMAP pipeline on the IMC-2025 test set.
#
# Extracts XFeat features, matches image pairs, runs COLMAP Structure-from-Motion
# and writes submission.csv (image_id, dataset, scene, rotation_matrix,
# translation_vector) in the same format as sample_submission.csv.
#
# Edit the settings below, then run:   bash test.sh
# =============================================================================
set -e
cd "$(dirname "$0")"

# ----------------------------- Hardware -------------------------------------
GPU_IDS="0"                 # comma-separated GPU ids, e.g. "0". "" = CPU

# --------------------------- Pipeline settings ------------------------------
TOP_K=4096                  # XFeat keypoints kept per image
MIN_COSSIM=0.82             # XFeat match cosine-similarity threshold
MIN_MATCHES=15              # minimum matches to keep an image pair
MIN_MODEL_SIZE=3            # minimum images for a COLMAP sub-reconstruction

# ------------------------------- Paths --------------------------------------
TEST_DIR="data/image-matching/test"
SAMPLE_SUBMISSION="data/image-matching/sample_submission.csv"
OUTPUT="submission.csv"
WORK_DIR="imc_xfeat/work"

# Use the fine-tuned checkpoint from train.sh if present, else the pretrained one.
WEIGHTS="imc_xfeat/checkpoints/xfeat_imc_latest.pt"
if [ ! -f "$WEIGHTS" ]; then
  WEIGHTS="accelerated_features/weights/xfeat.pt"
fi
# =============================================================================

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
echo "Using XFeat weights: $WEIGHTS"

python -m imc_xfeat.run_submission \
  --test-dir          "$TEST_DIR" \
  --sample-submission "$SAMPLE_SUBMISSION" \
  --output            "$OUTPUT" \
  --weights           "$WEIGHTS" \
  --work-dir          "$WORK_DIR" \
  --gpu-ids           "$GPU_IDS" \
  --top-k             "$TOP_K" \
  --min-cossim        "$MIN_COSSIM" \
  --min-matches       "$MIN_MATCHES" \
  --min-model-size    "$MIN_MODEL_SIZE"

echo "Done. Submission written to: $OUTPUT"
