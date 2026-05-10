import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def _build_train_transform(cfg: dict) -> "A.Compose":
    assert HAS_ALBUMENTATIONS, "albumentations is required"
    img_size = cfg.get("img_size", 1024)
    min_scale = cfg.get("min_scale", 0.5)
    max_scale = cfg.get("max_scale", 2.0)

    transforms = [
        A.RandomScale(scale_limit=(min_scale - 1.0, max_scale - 1.0), p=1.0),
        A.PadIfNeeded(
            min_height=img_size, min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT, p=1.0,
        ),
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),

        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),

        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.2),
        A.RandomToneCurve(scale=0.1, p=0.2),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.3),
        A.ImageCompression(quality_range=(60, 100), p=0.2),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.3,
        ),
        additional_targets={"masks": "masks"},
    )


def _build_val_transform(cfg: dict) -> "A.Compose":
    assert HAS_ALBUMENTATIONS, "albumentations is required"
    img_size = cfg.get("img_size", 1024)
    return A.Compose(
        [A.LongestMaxSize(max_size=img_size), A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        additional_targets={"masks": "masks"},
    )


class CopyPasteAugmentation:

    def __init__(
        self,
        paste_prob: float = 0.5,
        max_paste: int = 4,
        rare_classes: Optional[List[int]] = None,
        rare_boost: float = 3.0,
    ):
        self.paste_prob = paste_prob
        self.max_paste = max_paste
        self.rare_classes = set(rare_classes or [])
        self.rare_boost = rare_boost

    def __call__(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        category_ids: List[int],
        source_pool: List[Dict],
    ) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
        if random.random() > self.paste_prob or not source_pool:
            return image, masks, category_ids

        src = random.choice(source_pool)
        src_image = src["image"]
        src_masks = src["masks"]
        src_cats = src["category_ids"]
        H, W = image.shape[:2]


        if src_image.shape[:2] != (H, W):
            scale = min(H / src_image.shape[0], W / src_image.shape[1])
            nh, nw = int(src_image.shape[0] * scale), int(src_image.shape[1] * scale)
            src_image = cv2.resize(src_image, (nw, nh))
            src_masks = [cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST) for m in src_masks]

            pad_h, pad_w = H - nh, W - nw
            src_image = cv2.copyMakeBorder(src_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            src_masks = [
                cv2.copyMakeBorder(m, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                for m in src_masks
            ]

        n_paste = random.randint(1, self.max_paste)
        indices = list(range(len(src_masks)))

        weights = []
        for cat in src_cats:
            weights.append(self.rare_boost if cat in self.rare_classes else 1.0)
        if not weights:
            return image, masks, category_ids

        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = np.random.choice(len(indices), size=min(n_paste, len(indices)), replace=False, p=probs)

        image = image.copy()
        new_masks = list(masks)
        new_cats = list(category_ids)

        for idx in chosen:
            paste_mask = src_masks[idx].astype(bool)
            if not paste_mask.any():
                continue

            kernel = np.ones((5, 5), np.uint8)
            border = cv2.dilate(paste_mask.astype(np.uint8), kernel) - paste_mask.astype(np.uint8)
            alpha = paste_mask.astype(np.float32)
            alpha[border.astype(bool)] = 0.7
            for c in range(image.shape[2]):
                image[:, :, c] = (
                    image[:, :, c] * (1 - alpha) + src_image[:, :, c] * alpha
                ).astype(np.uint8)
            new_masks.append(paste_mask.astype(np.uint8))
            new_cats.append(src_cats[idx])

        return image, new_masks, new_cats


class OversamplingScheduler:

    def __init__(self, sample_class_counts: List[List[int]], num_classes: int = 4):
        self.sample_class_counts = np.array(sample_class_counts)
        total_per_class = self.sample_class_counts.sum(axis=0).astype(float)
        total_per_class = np.maximum(total_per_class, 1)

        class_weights = 1.0 / total_per_class
        class_weights /= class_weights.sum()

        weights = []
        for counts in sample_class_counts:
            present = [class_weights[i] for i, c in enumerate(counts) if c > 0]
            weights.append(max(present) if present else 0.0)
        self.weights = np.array(weights, dtype=np.float32)

    def get_weights(self) -> np.ndarray:
        return self.weights


class AugmentationPipeline:

    def __init__(self, cfg: dict, mode: str = "train"):
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.cfg = cfg
        if HAS_ALBUMENTATIONS:
            self.transform = (
                _build_train_transform(cfg) if mode == "train"
                else _build_val_transform(cfg)
            )
        else:
            self.transform = None

        copy_paste_cfg = cfg.get("copy_paste", {})
        self.copy_paste = (
            CopyPasteAugmentation(
                paste_prob=copy_paste_cfg.get("prob", 0.5),
                max_paste=copy_paste_cfg.get("max_paste", 4),
                rare_classes=copy_paste_cfg.get("rare_classes", [4]),
                rare_boost=copy_paste_cfg.get("rare_boost", 3.0),
            )
            if (mode == "train" and copy_paste_cfg.get("enabled", True))
            else None
        )

    def apply(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
        category_ids: List[int],
        bboxes_xyxy: Optional[List[List[float]]] = None,
    ) -> dict:
        if self.transform is None:
            return {"image": image, "masks": masks, "category_ids": category_ids}


        if bboxes_xyxy is None:
            bboxes_xyxy = []
            for m in masks:
                rows, cols = np.where(m > 0)
                if len(rows) == 0:
                    bboxes_xyxy.append([0, 0, 1, 1])
                else:
                    bboxes_xyxy.append([
                        float(cols.min()), float(rows.min()),
                        float(cols.max() + 1), float(rows.max() + 1)
                    ])


        H, W = image.shape[:2]
        clipped = []
        valid_idx = []
        for i, (box, m) in enumerate(zip(bboxes_xyxy, masks)):
            x1, y1, x2, y2 = box
            x1, x2 = np.clip([x1, x2], 0, W)
            y1, y2 = np.clip([y1, y2], 0, H)
            if x2 > x1 and y2 > y1:
                clipped.append([x1, y1, x2, y2])
                valid_idx.append(i)
        masks = [masks[i] for i in valid_idx]
        category_ids = [category_ids[i] for i in valid_idx]

        try:
            result = self.transform(
                image=image,
                masks=masks,
                bboxes=clipped,
                category_ids=category_ids,
            )
        except Exception:
            result = {"image": image, "masks": masks, "category_ids": category_ids}

        return {
            "image": result["image"],
            "masks": result.get("masks", masks),
            # Keep input category_ids (same length/order as masks). Albumentations
            # filters category_ids by bbox min_visibility but does NOT filter the
            # masks list, causing a length mismatch. __getitem__ recomputes bboxes
            # from masks anyway and skips all-zero masks with `if len(rows) == 0`.
            "category_ids": category_ids,
            "bboxes": result.get("bboxes", []),
        }
