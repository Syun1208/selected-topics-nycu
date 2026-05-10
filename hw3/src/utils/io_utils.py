import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tifffile


def load_tif(path: str) -> np.ndarray:
    return tifffile.imread(path)


def load_image_rgb(path: str) -> np.ndarray:
    img = tifffile.imread(path) if path.endswith(".tif") else cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img.astype(np.uint8)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def encode_mask_rle(binary_mask: np.ndarray) -> dict:
    from pycocotools import mask as coco_mask
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = coco_mask.encode(mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_mask_rle(rle: dict) -> np.ndarray:
    from pycocotools import mask as coco_mask
    if isinstance(rle["counts"], str):
        rle = dict(rle)
        rle["counts"] = rle["counts"].encode("utf-8")
    return coco_mask.decode(rle).astype(np.uint8)


def mask_to_polygon(binary_mask: np.ndarray) -> List[List[float]]:
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for c in contours:
        c = c.flatten().tolist()
        if len(c) >= 6:
            polygons.append(c)
    return polygons


def save_checkpoint(state: dict, path: str) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def scan_dir(data_dir: str) -> List[str]:
    return sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )


def train_val_split(
    sample_ids: List[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    import random
    rng = random.Random(seed)
    ids = list(sample_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def stratified_val_split(
    sample_ids: List[str],
    train_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_classes: int = 4,
) -> tuple:
    """Split train_ids into train/val ensuring each class has ~val_ratio in val."""
    import random
    from pathlib import Path
    rng = random.Random(seed)

    # Collect sample IDs that contain each class
    class_to_ids: dict = {c: [] for c in range(1, num_classes + 1)}
    for sid in sample_ids:
        for c in range(1, num_classes + 1):
            if (Path(train_dir) / sid / f"class{c}.tif").exists():
                class_to_ids[c].append(sid)

    val_set: set = set()
    for ids in class_to_ids.values():
        if not ids:
            continue
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n = max(1, round(len(shuffled) * val_ratio))
        val_set.update(shuffled[:n])

    val_ids   = [s for s in sample_ids if s in val_set]
    train_ids = [s for s in sample_ids if s not in val_set]
    return train_ids, val_ids
