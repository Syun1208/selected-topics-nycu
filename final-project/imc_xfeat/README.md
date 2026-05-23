# IMC-2025 pipeline (XFeat + COLMAP)

Final-project pipeline for the **Image Matching Challenge 2025** task: given
folders of images, recover the 6-DoF camera pose (rotation matrix + translation
vector) of every image and write a `submission.csv`.

It uses **XFeat** (`accelerated_features/`) for local feature extraction and
matching, and **COLMAP** (via `pycolmap`) for Structure-from-Motion.

```
imc_xfeat/
  xfeat_utils.py    load the XFeat model from the bundled accelerated_features repo
  imc_io.py         list images, read/write the submission CSV
  finetune.py       TRAIN: self-supervised homography fine-tuning of XFeat
  reconstruct.py    XFeat extract + match + COLMAP Structure-from-Motion
  run_submission.py TEST: run the pipeline and write submission.csv
```

## Requirements

```bash
pip install torch torchvision opencv-python numpy tqdm pycolmap
```

(`torch`, `opencv-python`, `numpy`, `tqdm` and `pycolmap` are required;
`kornia` is **not** needed.)

## Train — `train.sh`

Fine-tunes XFeat on the IMC training images. The training images have
ground-truth poses but **no depth maps / intrinsics**, so the standard XFeat
correspondence loss does not apply. Instead each image is warped by a random
homography, which yields exact dense correspondences for self-supervision.

```bash
bash train.sh
```

Edit the hyper-parameters at the top of `train.sh` — `GPU_IDS`, `BATCH_SIZE`,
`LEARNING_RATE`, `STEPS`, `TRAIN_RES`, `DIFFICULTY`, ... Checkpoints are written
to `imc_xfeat/checkpoints/` (`xfeat_imc_latest.pt` is the most recent).

## Test — `test.sh`

Runs the XFeat + COLMAP pipeline on `data/image-matching/test` and writes
`submission.csv` in the same format as `sample_submission.csv`.

```bash
bash test.sh
```

`test.sh` automatically uses the fine-tuned checkpoint
(`imc_xfeat/checkpoints/xfeat_imc_latest.pt`) if it exists, otherwise the
bundled pretrained `xfeat.pt`. COLMAP returns one reconstruction per connected
group of images, so each group becomes one `cluster{i}` scene; images that
cannot be registered are written with an identity pose and `scene=outliers`.
