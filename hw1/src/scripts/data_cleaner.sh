# Usage (dry-run — reports only, no files written):
# Runs the data cleaner script in dry-run mode, which only reports issues without modifying any files.
# python src/data/data_cleaner.py \
#     --data_dir data/train \
#     --checkpoint checkpoints/seresnextaa101d_32x8d_sw_in12k_ft_in1k_288/best_model.pth \
#     --model_name seresnextaa101d_32x8d.sw_in12k_ft_in1k_288 \
#     --dry_run

# Usage (clone + clean):
# Runs the data cleaner script to clone and clean the dataset, applying all checks and modifications.
python src/data/data_cleaner.py \
    --data_dir data/train \
    --checkpoint checkpoints/seresnextaa101d_32x8d_sw_in12k_ft_in1k_288/best_model.pth \
    --model_name seresnextaa101d_32x8d.sw_in12k_ft_in1k_288

# Flags to selectively enable/disable checks:
# Use these flags to skip specific checks during cleaning:
#   --no_integrity    : Skip integrity check
#   --no_cleanvision  : Skip CleanVision check
#   --no_label_noise  : Skip label noise check
#   --no_outliers     : Skip outlier detection
#   --no_duplicates   : Skip duplicate detection