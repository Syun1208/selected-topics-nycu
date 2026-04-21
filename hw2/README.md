# NYCU CV2026 HW2 — Digit Detection

## **Course:** NYCU Selected Topics in Visual Recognition (CV2026) — Homework 2  
### ***Student:*** Pham Minh Long
### ***Student ID***: 314540080  

---

## 1. Introduction
**Task:** 10-class digit detection (0–9) using DETR-family architectures trained from scratch (no pretrained DETR weights); only the ResNet backbone is initialised from a pretrained checkpoint.

This repository implements and benchmarks three DETR-family object detectors for digit detection:

| Method | Key idea |
|---|---|
| **RT-DETRv2** | Real-time anchor-free detector with HybridEncoder (AIFI + CNN) and NMS-free inference via RTDETRPostProcessor |
| **DINO** | DN-DETR variant with contrastive de-noising training, multi-scale deformable attention, and optional EMA weights |
| **AlignDETR** | DETR with progressive one-to-many assignment via MixedMatcher (`match_num`, tau) to align training and inference |

### Data Processing & Augmentation

A multi-stage data cleaning pipeline (`src/data/processors/data_cleaner.py`) is applied before training:

- **Quality filtering** — CleanVision detects and removes dark, blurry, or near-duplicate images.
- **Image restoration** — Optional NAFNet-based deblurring (`--deblurry-blurry`) and denoising (`--denoise-noisy`) using pre-trained NAFNet-GoPro and NAFNet-SIDD checkpoints.
- **Class-balanced augmentation** — Random flip, multi-scale resize (`480–800 px`, max 1333–1344), large-scale jitter with `RandomCrop (absolute_range)`, targeting ≥ 30 000 instances per class.

### Models & Backbones

| Architecture | Backbone | Pre-training |
|---|---|---|
| RT-DETRv2 | PResNet-101 (`resnet101`, variant-d) | ImageNet-1k |
| DINO | TimmBackbone `resnet50.a1_in1k` | ImageNet-1k |
| DINO | TimmBackbone `seresnextaa101d_32x8d.sw_in12k_ft_in1k_288` | ImageNet-12k → ImageNet-1k (288 px) |
| AlignDETR | TimmBackbone `resnet101` | ImageNet-1k |
| AlignDETR | TimmBackbone `resnet50.a1_in1k` | ImageNet-1k |
| AlignDETR | TimmBackbone `seresnextaa101d_32x8d.sw_in12k_ft_in1k_288` | ImageNet-12k → ImageNet-1k (288 px) |

All backbones use **FrozenBatchNorm2d** and a 4-level **ChannelMapper** neck (256 output channels).

### Loss Functions & Optimisation

- **Variable Focal Loss (VFL)** — used in RT-DETRv2 for classification (weight 1, α=0.75, γ=2).
- **Sigmoid Focal Loss** — used in DINO and AlignDETR (α=0.25, γ=2).
- **GIoU Loss** — for bounding-box regression (weight 2 for RT-DETR, 2.0 for DINO/AlignDETR).
- **L1 Loss** — box coordinate regression (weight 5).
- **Hungarian Matcher** — bipartite matching with cost weights (class 2, bbox 5, giou 2).
- **MixedMatcher (AlignDETR)** — progressive one-to-many assignment; `match_num=[2,2,2,2,2,2,1]`, `tau=1.5`.
- **Contrastive de-noising (CDN)** — `dn_number=100`, `label_noise_ratio=0.5`, `box_noise_scale=1.0`.
- **Optimizer** — AdamW, `lr=1e-4`, `weight_decay=1e-4`, backbone LR × 0.1; gradient clipping `max_norm=0.1`.
- **LR schedule** — linear warmup (1 000 steps) + MultiStep decay at 75 % and 90 % of total iterations.
- **Model EMA** — enabled for DINO/seresnextaa101d (decay 0.9998, used at eval only).

### Post-processing

| Method | Post-processing |
|---|---|
| RT-DETRv2 | **NMS-free** — `RTDETRPostProcessor`, top-300 queries, confidence score thresholding |
| DINO | **NMS** — `batched_nms`, IoU threshold 0.7, top-100 per image, pre-topk 10 000 |
| AlignDETR | **NMS** — `batched_nms`, score threshold tuning per experiment |

---

## 2. Environment Setup

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate selected-topics
```

### Option B — pip / uv

```bash
# pip
pip install -r requirements.txt

# uv
uv pip install -r requirements.txt
```

### Option C — Docker

```bash
docker build -t hw2 .
docker run --gpus all -it --rm \
  -v $(pwd):/workspace hw2 bash
```

> **Note:** detectron2 and detrex must be installed from source.  
> The bundled detrex source is located at `configs/detrex/`.

---

## 3. Usage

### 3.1 Data Cleaning & Augmentation

Cleans the raw training set and augments it to the target class balance before training.

```bash
# Augmentation only (default preset)
bash src/scripts/data_cleaner.sh --preset augment-only

# Full pipeline: quality filtering + augmentation
bash src/scripts/data_cleaner.sh --preset full

# With NAFNet deblurring + denoising
bash src/scripts/data_cleaner.sh \
  --preset deblurry-denoise \
  --deblurry-checkpoint-path checkpoints/NAFNet-GoPro-width64.pth \
  --denoise-checkpoint-path  checkpoints/NAFNet-SIDD-width64.pth

# Available presets: clean-only | augment-only | deblurry | denoise | deblurry-denoise | full | custom
```

Outputs `data/clean_train.json` and `data/clean_train/`. Update the YAML config to point to these paths before training.

---

### 3.2 Training

Launches distributed training using the Detectron2 `launch` utility.

```bash
# Single-GPU
bash src/scripts/train.sh \
  --config configs/train/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0.yaml \
  --gpu-ids 0

# Multi-GPU
bash src/scripts/train.sh \
  --config configs/train/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0.yaml \
  --gpu-ids 0,1

# Resume from checkpoint
bash src/scripts/train.sh \
  --config configs/train/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0.yaml \
  --gpu-ids 0 --resume
```

Config path convention: `configs/train/<method>/<backbone>/<run_id>.yaml`.  
Checkpoints and logs are written to the `output_dir` field in the YAML.

---

### 3.3 Inference / Test

Runs inference on the test set and produces a COCO-format submission JSON.

```bash
bash src/scripts/test.sh
# or directly:
python test.py \
  --config configs/test/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0.yaml \
  --gpu-ids 0
```

The submission file is written to the `output_dir` specified in the test YAML.

---

### 3.4 Visualisation (GradCAM + t-SNE)

Generates GradCAM saliency maps and t-SNE feature embeddings for qualitative analysis.

```bash
bash src/scripts/visualize.sh \
  --config configs/test/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0.yaml \
  --n-images 6 \
  --target-layer backbone.model.layer4 \
  --score-threshold 0.3 \
  --gpu-ids 0
```

| Argument | Description |
|---|---|
| `--config` | Path to test YAML config |
| `--n-images` | Number of images to visualise (default 6) |
| `--target-layer` | Layer name for GradCAM (e.g. `backbone.model.layer4`) |
| `--score-threshold` | Minimum confidence score to display a box |
| `--gpu-ids` | GPU device IDs |
| `--show` | Display results interactively (optional flag) |

---

## 4. Performance Snapshot

### Leaderboard

![Performance Leaderboard](images/performance_leaderboard.png)

> Public leaderboard as of 2026-04-21. **Rank #1** with mAP@[0.5:0.95] = **0.44**.

---

### Experiment Comparison

Val mAP is COCO AP@[0.5:0.95] evaluated on the local validation split.

| Method | Backbone | Queries | Embed dim | Enc / Dec layers | Batch size | Max iter | LR | AMP | Val mAP |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| RT-DETRv2 | `resnet101` | 300 | 384 | 1 / 6 | 1 | 75 000 | 1e-4 | ✓ | 43.10 |
| RT-DETRv2 | `resnet50.a1_in1k` | 300 | 384 | 1 / 6 | 2 | 75 000 | 1e-4 | ✓ | 41.75 |
| DINO | `resnet50.a1_in1k` | 900 | 256 | 6 / 6 | 2 | 100 000 | 1e-4 | ✓ | 45.59 |
| DINO | `seresnextaa101d_32x8d` | 900 | 256 | 6 / 6 | 2 | 75 000 | 1e-4 | ✗ | **51.23** |
| AlignDETR | `resnet50.a1_in1k` | 900 | 256 | 6 / 6 | 2 | 100 000 | 1e-4 | ✓ | 46.69 |
| AlignDETR | `resnet101` | 900 | 256 | 6 / 6 | 1 | 750 000 | 1e-4 | ✗ | 46.99 |
| AlignDETR | `seresnextaa101d_32x8d` | 900 | 256 | 6 / 6 | 1 | 100 000 | 1e-4 | ✗ | 43.08 |

**Notes:**
- All runs use AdamW (`weight_decay=1e-4`), backbone LR × 0.1, gradient clipping (`max_norm=0.1`), and a MultiStep schedule with 1 000-step linear warmup; milestones at 75 % and 90 % of total iterations.
- DINO/`seresnextaa101d` uses Model EMA (decay 0.9998, eval-only) and adds an auxiliary AnchorHead branch (3 anchors, 3 conv layers, focal + GIoU losses) for multi-task training.
- AlignDETR uses `MixedMatcher` with progressive K-assignment (`match_num=[2,2,2,2,2,2,1]`, `tau=1.5`).
- RT-DETRv2 employs **NMS-free** inference; DINO and AlignDETR apply `batched_nms` (IoU threshold 0.7).
- Best single-model result: **DINO + seresnextaa101d_32x8d** → **51.23 val mAP**.

---

### Val Predictions

Sample detection results on the validation set for each method and backbone.

**RT-DETRv2**

![RT-DETRv2 Val Predictions](charts/rtdetr/val_predictions_subplot.png)

**DINO**

![DINO Val Predictions](charts/dino/val_predictions_subplot.png)

**AlignDETR**

![AlignDETR Val Predictions](charts/aligndetr/val_predictions_subplot.png)

---

### GradCAM Visualisations

GradCAM saliency maps (multi-layer subplot) show which image regions each model attends to when predicting digit bounding boxes.

#### RT-DETRv2 — GradCAM Layer Comparison

| `resnet101` | `resnet50.a1_in1k` |
|:---:|:---:|
| ![](charts/rtdetr/resnet101/0/gradcam/0_layer_comparison.png) | ![](charts/rtdetr/resnet50.a1_in1k/0/gradcam/0_layer_comparison.png) |

#### DINO — GradCAM Layer Comparison

| `resnet50.a1_in1k` | `seresnextaa101d_32x8d` |
|:---:|:---:|
| ![](charts/dino/resnet50.a1_in1k/0/gradcam/0_layer_comparison.png) | ![](charts/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0/gradcam/0_layer_comparison.png) |

#### AlignDETR — GradCAM Layer Comparison

| `resnet50.a1_in1k` | `resnet101` | `seresnextaa101d_32x8d` |
|:---:|:---:|:---:|
| ![](charts/aligndetr/resnet50.a1_in1k/0/gradcam/0_layer_comparison.png) | ![](charts/aligndetr/resnet101/0/gradcam/0_layer_comparison.png) | ![](charts/aligndetr/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0/gradcam/0_layer_comparison.png) |

---

### Confusion Matrices

Confusion matrices on the local validation set (rows = ground-truth class, columns = predicted class).

#### RT-DETRv2 — Confusion Matrix

| `resnet101` | `resnet50.a1_in1k` |
|:---:|:---:|
| ![](charts/rtdetr/resnet101/0/confusion_matrix.png) | ![](charts/rtdetr/resnet50.a1_in1k/0/confusion_matrix.png) |

#### DINO — Confusion Matrix

| `resnet50.a1_in1k` | `seresnextaa101d_32x8d` |
|:---:|:---:|
| ![](charts/dino/resnet50.a1_in1k/0/confusion_matrix.png) | ![](charts/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0/confusion_matrix.png) |

#### AlignDETR — Confusion Matrix

| `resnet50.a1_in1k` | `resnet101` | `seresnextaa101d_32x8d` |
|:---:|:---:|:---:|
| ![](charts/aligndetr/resnet50.a1_in1k/0/confusion_matrix.png) | ![](charts/aligndetr/resnet101/0/confusion_matrix.png) | ![](charts/aligndetr/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0/confusion_matrix.png) |

---

## 5. References

- **Detectron2** — <https://github.com/facebookresearch/detectron2>
- **detrex** — <https://github.com/IDEA-Research/detrex>
- **RT-DETR** — <https://github.com/lyuwenyu/RT-DETR>