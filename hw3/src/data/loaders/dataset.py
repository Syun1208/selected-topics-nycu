import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from src.data.models.sample import InstanceAnnotation, TrainSample, TestSample
from src.data.processors.augmentation import AugmentationPipeline

NUM_CLASSES = 4
CLASS_NAMES = ["class1", "class2", "class3", "class4"]


def _tif_to_rgb(path: str) -> np.ndarray:
    img = tifffile.imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img.astype(np.uint8)


def _mask_to_instances(mask: np.ndarray, cls_id: int) -> List[InstanceAnnotation]:
    instances = []
    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]
    for iid in instance_ids:
        binary = (mask == iid).astype(np.uint8)
        rows, cols = np.where(binary)
        if len(rows) == 0:
            continue
        x, y = int(cols.min()), int(rows.min())
        w, h = int(cols.max()) - x + 1, int(rows.max()) - y + 1
        area = float(binary.sum())
        instances.append(InstanceAnnotation(
            category_id=cls_id,
            instance_id=int(iid),
            mask=binary,
            bbox=(float(x), float(y), float(w), float(h)),
            area=area,
        ))
    return instances


def load_train_sample(sample_dir: str, sample_id: str) -> TrainSample:
    image = _tif_to_rgb(os.path.join(sample_dir, "image.tif"))
    H, W = image.shape[:2]
    all_instances = []
    for cls_id in range(1, NUM_CLASSES + 1):
        mask_path = os.path.join(sample_dir, f"class{cls_id}.tif")
        if os.path.exists(mask_path):
            mask = tifffile.imread(mask_path).astype(np.int32)
            all_instances.extend(_mask_to_instances(mask, cls_id))
    return TrainSample(
        image_id=sample_id,
        image=image,
        height=H,
        width=W,
        instances=all_instances,
    )


class CellInstanceDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        sample_ids: Optional[List[str]] = None,
        augmentation_cfg: Optional[dict] = None,
        mode: str = "train",
        source_pool_size: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.sample_ids = sample_ids or sorted(
            d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
        )
        aug_cfg = augmentation_cfg or {}
        self.aug_pipeline = AugmentationPipeline(aug_cfg, mode=mode)

        self._pool_ids = self.sample_ids[:source_pool_size]
        # Pre-load pool once so __getitem__ doesn't re-read disk on every call.
        self._pool_cache: List[Dict] = []
        if mode == "train" and self.aug_pipeline.copy_paste is not None:
            for pid in self._pool_ids:
                ps = self._load_raw(pid)
                self._pool_cache.append({
                    "id": pid,
                    "image": ps.image,
                    "masks": [inst.mask for inst in ps.instances],
                    "category_ids": [inst.category_id for inst in ps.instances],
                })

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_raw(self, sample_id: str) -> TrainSample:
        return load_train_sample(str(self.data_dir / sample_id), sample_id)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        sample = self._load_raw(sid)

        image = sample.image
        masks = [inst.mask for inst in sample.instances]
        category_ids = [inst.category_id for inst in sample.instances]

        if self.mode == "train" and self.aug_pipeline.copy_paste is not None:
            pool = [p for p in self._pool_cache if p["id"] != sid]
            image, masks, category_ids = self.aug_pipeline.copy_paste(
                image, masks, category_ids, pool
            )

        if masks:
            aug_result = self.aug_pipeline.apply(image, masks, category_ids)
            image = aug_result["image"]
            masks = aug_result["masks"]
            category_ids = aug_result["category_ids"]


        gt_bboxes = []
        gt_labels = []
        gt_masks_bin = []
        for m, cat in zip(masks, category_ids):
            m = np.asarray(m, dtype=np.uint8)
            rows, cols = np.where(m > 0)
            if len(rows) == 0:
                continue
            x1, y1 = int(cols.min()), int(rows.min())
            x2, y2 = int(cols.max() + 1), int(rows.max() + 1)
            gt_bboxes.append([x1, y1, x2, y2])
            gt_labels.append(cat - 1)
            gt_masks_bin.append(m)

        H, W = image.shape[:2]
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        image_norm = (image.astype(np.float32) - mean) / std
        return {
            "image_id": sid,
            "img": torch.from_numpy(image_norm).permute(2, 0, 1),
            "gt_bboxes": torch.tensor(gt_bboxes, dtype=torch.float32)
                          if gt_bboxes else torch.zeros((0, 4), dtype=torch.float32),
            "gt_labels": torch.tensor(gt_labels, dtype=torch.long)
                          if gt_labels else torch.zeros((0,), dtype=torch.long),
            "gt_masks": gt_masks_bin,
            "img_metas": {
                "img_shape": (H, W, 3),
                "pad_shape": (H, W, 3),
                "ori_shape": (H, W, 3),
                # scale_factor=[1,1,1,1]: GT bboxes are in augmented (H×W) space so
                # we must keep predictions in the same space — no rescaling.
                "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "flip": False,
                "filename": str(self.data_dir / sid / "image.tif"),
                "batch_input_shape": (H, W),
            },
        }

    def get_oversampling_weights(self) -> np.ndarray:
        from src.data.processors.augmentation import OversamplingScheduler
        counts = []
        for sid in self.sample_ids:
            sample = self._load_raw(sid)
            per_class = [0] * NUM_CLASSES
            for inst in sample.instances:
                per_class[inst.category_id - 1] += 1
            counts.append(per_class)
        scheduler = OversamplingScheduler(counts, NUM_CLASSES)
        return scheduler.get_weights()


class CellTestDataset(Dataset):

    def __init__(self, data_dir: str, mapping_json: str, img_size: int = 1024):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        with open(mapping_json) as f:
            meta_list = json.load(f)
        self.meta = meta_list

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        meta = self.meta[idx]
        img_path = self.data_dir / meta["file_name"]
        image = _tif_to_rgb(str(img_path))
        ori_H, ori_W = image.shape[:2]


        import cv2
        scale = self.img_size / max(ori_H, ori_W)
        new_H, new_W = int(ori_H * scale), int(ori_W * scale)
        image_resized = cv2.resize(image, (new_W, new_H))


        pad_h = self.img_size - new_H
        pad_w = self.img_size - new_W
        image_padded = np.pad(
            image_resized,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
        )

        return {
            "image_id": meta["id"],
            "file_name": meta["file_name"],
            "img": torch.from_numpy(image_padded).permute(2, 0, 1).float(),
            "img_metas": {
                "img_shape": (self.img_size, self.img_size, 3),
                "ori_shape": (ori_H, ori_W, 3),
                "scale_factor": np.array([scale, scale, scale, scale]),
                "flip": False,
                "filename": str(img_path),
            },
        }
