#!/usr/bin/env bash

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python3 -u augment.py \
    --data-dir ./data/train \
    --path-save ./data_augmented/train/ \
    --target-instances 30000 \
    --gpu-id 0 \
    --no-mosaic --no-mixup
