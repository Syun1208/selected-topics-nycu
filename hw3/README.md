# NYCU CV2026 HW3 — Instance Segmentation

## **Course:** NYCU Selected Topics in Visual Recognition (CV2026) — Homework 3
### ***Student:*** Pham Minh Long
### ***Student ID***: 314540080

---

## 1. Introduction
**Task:** 4-class cell instance segmentation on fluorescence-microscopy images (209 train / 101 test). Submissions are graded on COCO-style **mask AP@[0.50:0.95]** (private) and **AP50** (public CodaBench leaderboard).

> **No pretrained instance-segmentation model is used.** Every detector head (RPN, 3-stage cascade bbox heads, mask head(s), and HTC's semantic head) is **trained from scratch** on the cell dataset. Only the **backbone** (ResNeXt-101 / Swin-B) is initialised from an ImageNet-pretrained checkpoint — no COCO-pretrained Mask R-CNN / HTC / Cascade Mask R-CNN weights are loaded (`pretrain_weights.path: null` in every YAML config).

This repository implements and benchmarks two cascade-style instance segmentation detectors built on **MMDetection 3.x**, with a strict **≤ 200 M-parameter budget** for the submitted model:

| Method | Key idea |
|---|---|
| **Cascade Mask R-CNN** | 3-stage cascade with progressively higher IoU thresholds (0.5/0.6/0.7) for both bbox & mask heads, refining proposals stage-by-stage |
| **HTC (Hybrid Task Cascade)** | Cascade Mask R-CNN + interleaved bbox/mask flow + a Fused Semantic Head that injects semantic-segmentation context into mask features |

### Data Processing & Augmentation

A configurable Albumentations pipeline (`src/data/processors/augmentation.py`) is applied online during training:

- **Train / val split** — deterministic split with `val_ratio=0.2`, `seed=42` → ≈ 167 train / 42 val out of 209 samples.
- **Class-imbalance handling** — `WeightedRandomSampler` driven by `OversamplingScheduler`, plus **CopyPasteAugmentation** (paste 1–4 instances per image with edge feathering, `rare_boost=3.0×` on rare classes).
- **Geometric** — `RandomScale (0.5–2.0×) → PadIfNeeded → RandomCrop (1024) → HorizontalFlip`.
- **Photometric** — `RandomBrightnessContrast`, `RandomGamma`, `CLAHE`, `RandomToneCurve`, `OneOf{GaussianBlur, MotionBlur, MedianBlur}`, `OneOf{GaussNoise, ISONoise}`, `ImageCompression`.
- **Validation** — only `LongestMaxSize → PadIfNeeded` (deterministic).
- **Normalization** — ImageNet mean/std (`123.675/116.28/103.53`, `58.395/57.12/57.375`).

A standalone cleaner (`src/scripts/data_cleaner.sh`) writes a `cleaning_report.json` with mask-integrity, bbox-consistency, class-imbalance and per-class size statistics under `data/processed/reports/`.

### Models & Backbones

All detectors share a **5-level FPN** (`out_channels=256`) and a **3-stage cascade RoI head** (stage loss weights `[1, 0.5, 0.25]`). Only configurations within the ≤ 200 M-parameter budget are compared.

| Architecture | Backbone | Model size | Pre-training |
|---|---|---:|---|
| Cascade Mask R-CNN | ResNeXt-101 (64×4d) | **135.1 M** | open-mmlab `resnext101_64x4d` (ImageNet-1k) |
| Cascade Mask R-CNN | Swin-B (window-7-224) | **139.8 M** | `swin_base_patch4_window7_224_22k` (ImageNet-22K) |
| HTC | ResNeXt-101 (64×4d) | **138.0 M** | open-mmlab `resnext101_64x4d` (ImageNet-1k) |
| HTC | Swin-B (window-7-224) | **142.7 M** | `swin_base_patch4_window7_224_22k` (ImageNet-22K) |

Model size = total `numel()` over the saved `state_dict` of `best.pth`. ResNeXt backbones use `frozen_stages=1` + `BN`; Swin backbones use `with_cp=True` (gradient checkpointing) for memory.

### Loss Functions & Optimisation

- **RPN classification** — `CrossEntropyLoss(use_sigmoid=True)`, weight 1.
- **RPN bbox regression** — `SmoothL1Loss(beta=1/9)`, weight 1.
- **Cascade bbox cls / reg (× 3 stages)** — `CrossEntropyLoss` + `SmoothL1Loss(beta=1.0)`, weight 1 each.
- **Mask head** — `CrossEntropyLoss(use_mask=True)`, weight 1 (Cascade uses `FCNMaskHead`; HTC uses 3 stacked `HTCMaskHead`s with mask-info-flow).
- **HTC semantic head** — `CrossEntropyLoss(ignore_index=255)`, weight 0.2.
- **Optimizer** — AdamW for Swin (`lr=5e-5 ~ 1e-4`, `wd=0.05`, `grad_clip=1.0`); SGD for ResNeXt (`lr=1.25e-3 ~ 2e-3`, `wd=1e-4`, `grad_clip=35.0`).
- **LR schedule** — 1-epoch linear warmup + step decay at epochs `[24, 32]` (×0.1), 36 epochs total.
- **Mixed precision** — `fp16=true` for Swin runs; `with_cp=True` on every backbone keeps peak memory < 24 GB at `img_size=1024`.

### Post-processing

| Stage | Setting |
|---|---|
| RPN | `nms_pre=1000`, `max_per_img=1000`, NMS IoU 0.7 |
| RoI head | `score_thr=0.05`, NMS IoU 0.5, `max_per_img=100`, `mask_thr_binary=0.5` |
| Submission | RLE-encode masks (`pycocotools.mask.encode`), rescale bboxes by letterbox factor, resize masks (nearest) to original `H×W`, drop preds with `score<0.05` |

---

## 2. Environment Setup

### Option A — Conda (recommended)

```bash
conda create -n hw3 python=3.10 -y
conda activate hw3

# 1) PyTorch (pick the cu118 / cu121 wheel that matches your driver)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# 2) Project dependencies
pip install -r requirements.txt

# 3) Local editable mmdet (vendored)
pip install -e configs/mmdetection
```

### Option B — pip / uv

```bash
# pip
pip install -r requirements.txt
pip install -e configs/mmdetection

# uv
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e configs/mmdetection
```

### Option C — Docker

```bash
# Build the mmdet base image (CUDA 11.8 + PyTorch 2.2 + mmdet 3.x)
docker build -t hw3:base -f configs/mmdetection/docker/Dockerfile configs/mmdetection

docker run --gpus all -it --rm \
    -v "$PWD":/workspace -w /workspace hw3:base bash -lc "
        pip install -r requirements.txt &&
        pip install -e configs/mmdetection &&
        bash src/scripts/train.sh --config configs/train/cascade_mask_rcnn/resnext101/v1.yaml"
```

> **Note:** `mmdet 3.x` is installed editable from the vendored source at `configs/mmdetection/`.

---

## 3. Usage

### 3.1 Data Cleaning & EDA

Cleans the raw training set, performs the train/val split, and writes a JSON cleaning report.

```bash
bash src/scripts/data_cleaner.sh \
    --data-dir data \
    --val-ratio 0.2 \
    --val-seed 42 \
    --img-size 1024
```

EDA notebook (figures land in `notebooks/outputs/`):

```bash
jupyter lab notebooks/eda.ipynb
```

---

### 3.2 Training

The launcher reads `training.gpu_ids` from the YAML and automatically switches to `torchrun` DDP if more than one GPU is requested.

```bash
# Single-GPU
bash src/scripts/train.sh \
    --config configs/train/cascade_mask_rcnn/resnext101/v1.yaml

# Multi-GPU (override gpu_ids)
bash src/scripts/train.sh \
    --config configs/train/cascade_mask_rcnn/resnext101/v1.yaml \
    --gpu-ids 1,2,4

# Resume
bash src/scripts/train.sh \
    --config configs/train/cascade_mask_rcnn/resnext101/v1.yaml \
    --resume checkpoints/cascade_mask_rcnn/resnext101/v1/last.pth
```

Config path convention: `configs/train/<model>/<backbone>/v1.yaml`.
Outputs: `checkpoints/<model>/<backbone>/v1/{best,last}.pth`, `charts/<model>/<backbone>/v1/*.png`, `logs/<model>/<backbone>/<run_id>.log`.

---

### 3.3 Inference / Test

Runs inference on `data/test_release/` and writes a COCO-format submission JSON + ZIP.

```bash
bash src/scripts/test.sh \
    --config  configs/test/cascade_mask_rcnn/resnext101/v1.yaml \
    --gpu-ids 0

# or directly
python test.py \
    --config     configs/test/cascade_mask_rcnn/resnext101/v1.yaml \
    --checkpoint checkpoints/cascade_mask_rcnn/resnext101/v1/best.pth \
    --gpu_ids    0 \
    --score_thr  0.05
```

Submission output: `submissions/<model>/<backbone>/v1/test-results.json` and `v1.zip`.

---

### 3.4 Visualisation

`test.py` automatically saves overlay PNGs for every test image to
`submissions/<model>/<backbone>/v1/visualize/` (boxes + class-coloured masks + scores).

For training-time diagnostics, the trainer regenerates a 7-chart bundle in `charts/<model>/<backbone>/v1/` after every epoch:

| Chart | Description |
|---|---|
| `loss_curves.png` | Total train/val loss vs. epoch |
| `loss_components.png` | Per-head loss (RPN cls/reg, cascade cls/reg/mask, HTC semantic) |
| `ap_per_class.png` | Validation mAP per class |
| `pr_curve.png` | Precision-Recall curve (mask IoU) |
| `f1_recall_curve.png` | F1 vs. confidence threshold |
| `confusion_matrix.png` | Per-class confusion matrix |
| `predictions_vis.png` | Sample predictions with GT overlay |

---

## 4. Performance Snapshot

### Leaderboard

![CodaBench leaderboard](notebooks/outputs/performance_snapshot.png)

> Public CodaBench leaderboard — submission ID `leonaienginer`. **Rank #5** with **AP50 = 0.6163**.

---

### EDA — dataset summary

<table>
<tr>
  <td><img src="notebooks/outputs/01_class_distribution.png" alt="Class distribution"/></td>
  <td><img src="notebooks/outputs/02_image_size_distribution.png" alt="Image size distribution"/></td>
</tr>
<tr>
  <td><img src="notebooks/outputs/03_instance_count_per_class.png" alt="Instance count per class"/></td>
  <td><img src="notebooks/outputs/04_instance_count_summary.png" alt="Instance count summary"/></td>
</tr>
<tr>
  <td><img src="notebooks/outputs/05_instance_area_distribution.png" alt="Instance area distribution"/></td>
  <td><img src="notebooks/outputs/06_instance_area_violin.png" alt="Instance area violin"/></td>
</tr>
<tr>
  <td><img src="notebooks/outputs/07_channel_intensity_distribution.png" alt="Channel intensity distribution"/></td>
  <td><img src="notebooks/outputs/08_correlation_heatmap.png" alt="Correlation heatmap"/></td>
</tr>
<tr>
  <td><img src="notebooks/outputs/09_train_samples.png" alt="Train samples"/></td>
  <td><img src="notebooks/outputs/10_test_samples.png" alt="Test samples"/></td>
</tr>
<tr>
  <td><img src="notebooks/outputs/11_psi_distribution.png" alt="PSI distribution"/></td>
  <td><img src="notebooks/outputs/12_distribution_shift_samples.png" alt="Distribution shift samples"/></td>
</tr>
<!-- <tr>
  <td colspan="2"><img src="notebooks/outputs/13_psi_heatmap.png" alt="PSI heatmap"/></td>
</tr> -->
</table>

Key observations driving the design:

- **Severe class imbalance** — class3 / class4 each have <650 instances vs. >14k for class1/2. Motivates `WeightedRandomSampler` + rare-class copy-paste boost.
- **Most instances are small** — 99.99 % of class2, 98 % of class3 and 82 % of class1 are below the COCO "small" threshold; class4 has a long tail (max 54,554 px²). FPN with all 5 levels and `RandomScale(0.5–2.0×)` is therefore essential.
- **Train↔test channel-intensity shift** — visible in the PSI heatmap — partially mitigated by `CLAHE`, `RandomGamma`, `RandomBrightnessContrast`, `RandomToneCurve`.

---

### Experiment Comparison

Val mAP is COCO **mask AP@[0.50:0.95]** evaluated on the local validation split (20 % of 209 samples). Only configurations within the ≤ 200 M-parameter budget are reported.

| Method | Backbone | Img size | Batch | Epochs | LR | AMP | Params | Best epoch | Val mAP |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cascade Mask R-CNN | `resnext101_64x4d` | 1024 | 2 | 36 | 2e-3 (SGD) | ✗ | 135.1 M | 26 | **0.7200** |
| Cascade Mask R-CNN | `swin_base_patch4_window7_224_22k` | 1024 | 2 | 36 | 1e-4 (AdamW) | ✓ | 139.8 M | 35 | 0.3037 |
| HTC | `resnext101_64x4d` | 800 | 1 | 36 | 1.25e-3 (SGD) | ✓ | 138.0 M | 24 | 0.4364 |
| HTC | `swin_base_patch4_window7_224_22k` | 800 | 1 | 36 | 5e-5 (AdamW) | ✓ | 142.7 M | 34 | 0.2372 |

**Notes:**
- All runs use a 1-epoch linear warmup + step-decay at epochs `[24, 32]` (×0.1), 3-stage cascade RoI head (`stage_loss_weights=[1, 0.5, 0.25]`), and a 5-level FPN.
- Copy-paste augmentation with `rare_boost=3.0×` and `WeightedRandomSampler` are enabled for every run.
- Inference: `score_thr=0.05`, `NMS IoU=0.5`, `max_per_img=100`, `mask_thr_binary=0.5`.
- Best single-model result (≤ 200 M): **Cascade Mask R-CNN + ResNeXt-101-64×4d** → **0.7200** val mAP, **0.6163** public AP50 (submitted).

---

### Training & validation curves (per model)

#### Cascade Mask R-CNN + ResNeXt-101 (submitted)

<table>
<tr>
  <td><img src="charts/cascade_mask_rcnn/resnext101/v1/loss_curves.png" alt="loss"/></td>
  <td><img src="charts/cascade_mask_rcnn/resnext101/v1/loss_components.png" alt="loss components"/></td>
</tr>
<tr>
  <td><img src="charts/cascade_mask_rcnn/resnext101/v1/ap_per_class.png" alt="AP per class"/></td>
  <td><img src="charts/cascade_mask_rcnn/resnext101/v1/pr_curve.png" alt="PR curve"/></td>
</tr>
<tr>
  <td><img src="charts/cascade_mask_rcnn/resnext101/v1/f1_recall_curve.png" alt="F1 vs recall"/></td>
  <td><img src="charts/cascade_mask_rcnn/resnext101/v1/confusion_matrix.png" alt="confusion matrix"/></td>
</tr>
<tr>
  <td colspan="2"><img src="charts/cascade_mask_rcnn/resnext101/v1/predictions_vis.png" alt="predictions"/></td>
</tr>
</table>

#### Cascade Mask R-CNN + Swin-B

<table>
<tr>
  <td><img src="charts/cascade_mask_rcnn/swin_b/v1/loss_curves.png" alt="loss"/></td>
  <td><img src="charts/cascade_mask_rcnn/swin_b/v1/loss_components.png" alt="loss components"/></td>
</tr>
<tr>
  <td><img src="charts/cascade_mask_rcnn/swin_b/v1/ap_per_class.png" alt="AP per class"/></td>
  <td><img src="charts/cascade_mask_rcnn/swin_b/v1/pr_curve.png" alt="PR curve"/></td>
</tr>
<tr>
  <td><img src="charts/cascade_mask_rcnn/swin_b/v1/f1_recall_curve.png" alt="F1 vs recall"/></td>
  <td><img src="charts/cascade_mask_rcnn/swin_b/v1/confusion_matrix.png" alt="confusion matrix"/></td>
</tr>
<tr>
  <td colspan="2"><img src="charts/cascade_mask_rcnn/swin_b/v1/predictions_vis.png" alt="predictions"/></td>
</tr>
</table>

#### HTC + ResNeXt-101

<table>
<tr>
  <td><img src="charts/htc/resnext101/v1/loss_curves.png" alt="loss"/></td>
  <td><img src="charts/htc/resnext101/v1/loss_components.png" alt="loss components"/></td>
</tr>
<tr>
  <td><img src="charts/htc/resnext101/v1/ap_per_class.png" alt="AP per class"/></td>
  <td><img src="charts/htc/resnext101/v1/pr_curve.png" alt="PR curve"/></td>
</tr>
<tr>
  <td><img src="charts/htc/resnext101/v1/f1_recall_curve.png" alt="F1 vs recall"/></td>
  <td><img src="charts/htc/resnext101/v1/confusion_matrix.png" alt="confusion matrix"/></td>
</tr>
<tr>
  <td colspan="2"><img src="charts/htc/resnext101/v1/predictions_vis.png" alt="predictions"/></td>
</tr>
</table>

#### HTC + Swin-B

<table>
<tr>
  <td><img src="charts/htc/swin_b/v1/loss_curves.png" alt="loss"/></td>
  <td><img src="charts/htc/swin_b/v1/loss_components.png" alt="loss components"/></td>
</tr>
<tr>
  <td><img src="charts/htc/swin_b/v1/ap_per_class.png" alt="AP per class"/></td>
  <td><img src="charts/htc/swin_b/v1/pr_curve.png" alt="PR curve"/></td>
</tr>
<tr>
  <td><img src="charts/htc/swin_b/v1/f1_recall_curve.png" alt="F1 vs recall"/></td>
  <td><img src="charts/htc/swin_b/v1/confusion_matrix.png" alt="confusion matrix"/></td>
</tr>
<tr>
  <td colspan="2"><img src="charts/htc/swin_b/v1/predictions_vis.png" alt="predictions"/></td>
</tr>
</table>

---

## 5. References

- **MMDetection** — <https://github.com/open-mmlab/mmdetection>
- **HTC (Hybrid Task Cascade for Instance Segmentation)** — <https://arxiv.org/pdf/1901.07518>
- **Cascade Mask R-CNN / Cascade R-CNN** — <https://arxiv.org/pdf/1906.09756>
- **Swin Transformer** — <https://github.com/microsoft/Swin-Transformer>
