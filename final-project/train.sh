set -e
cd "$(dirname "$0")"

# ----------------------------- Hardware -------------------------------------
GPU_IDS="6,7"                 # comma-separated GPU ids, e.g. "0" or "0,1". "" = CPU
NUM_WORKERS=4               # DataLoader worker processes

# --------------------------- Hyper-parameters -------------------------------
BATCH_SIZE=8                # images per training step
LEARNING_RATE=1e-4          # Adam learning rate
WEIGHT_DECAY=1e-5           # Adam weight decay
STEPS=2000                  # total optimisation steps
GRAD_CLIP=1.0               # gradient-norm clipping
TRAIN_RES="800,608"         # training resolution W,H (both divisible by 32)
DIFFICULTY=0.3              # homography warp strength (0.1 easy ... 0.5 hard)
MAX_CORR=2048               # max correspondences per image used in the loss
SEED=0

# ------------------------------ Logging -------------------------------------
SAVE_EVERY=500              # checkpoint interval (steps)
LOG_EVERY=20                # console log interval (steps)

# ------------------------------- Paths --------------------------------------
TRAIN_DIR="data/image-matching/train"
INIT_WEIGHTS="accelerated_features/weights/xfeat.pt"   # XFeat pretrained init
CKPT_DIR="imc_xfeat/checkpoints"                       # fine-tuned checkpoints
# =============================================================================

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

python -m imc_xfeat.finetune \
  --train-dir     "$TRAIN_DIR" \
  --weights       "$INIT_WEIGHTS" \
  --ckpt-dir      "$CKPT_DIR" \
  --gpu-ids       "$GPU_IDS" \
  --num-workers   "$NUM_WORKERS" \
  --batch-size    "$BATCH_SIZE" \
  --lr            "$LEARNING_RATE" \
  --weight-decay  "$WEIGHT_DECAY" \
  --steps         "$STEPS" \
  --grad-clip     "$GRAD_CLIP" \
  --train-res     "$TRAIN_RES" \
  --difficulty    "$DIFFICULTY" \
  --max-corr      "$MAX_CORR" \
  --save-every    "$SAVE_EVERY" \
  --log-every     "$LOG_EVERY" \
  --seed          "$SEED"

echo "Done. Fine-tuned weights: $CKPT_DIR/xfeat_imc_latest.pt"
