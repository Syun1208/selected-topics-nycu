import warnings

import albumentations as A
import cv2
import numpy as np


def _coco_to_pascal(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _pascal_to_coco(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def _clip_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    return [
        max(0.0, min(x1, img_w)),
        max(0.0, min(y1, img_h)),
        max(0.0, min(x2, img_w)),
        max(0.0, min(y2, img_h)),
    ]


def apply_aug_to_image(image: np.ndarray, annotations: list, transform: A.Compose):
    bboxes_pascal, cat_ids = [], []
    for ann in annotations:
        x1, y1, x2, y2 = _coco_to_pascal(ann["bbox"])
        h, w = image.shape[:2]
        x1, y1, x2, y2 = _clip_bbox([x1, y1, x2, y2], w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        bboxes_pascal.append([x1, y1, x2, y2])
        cat_ids.append(ann["category_id"])

    if not bboxes_pascal:
        return image, []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        result = transform(image=image, bboxes=bboxes_pascal, category_ids=cat_ids)
    aug_image = result["image"]
    new_anns = []
    for bbox, cat_id, orig_ann in zip(result["bboxes"], result["category_ids"], annotations):
        new_ann = dict(orig_ann)
        new_ann["bbox"] = _pascal_to_coco(list(bbox))
        new_ann["category_id"] = cat_id
        new_ann["area"] = float(new_ann["bbox"][2] * new_ann["bbox"][3])
        new_anns.append(new_ann)
    return aug_image, new_anns


def make_strong_aug(seed: int = 0) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.4,
            ),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.2), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.12), p=0.4),
            A.ImageCompression(quality_range=(60, 95), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-5, 5),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7,
            ),
            A.SafeRotate(
                limit=(-10, 10),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
            A.Rotate(
                limit=(-8, 8),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.ElasticTransform(
                alpha=1.0,
                sigma=10,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.25,
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.15,
                p=0.25,
            ),
            A.OpticalDistortion(
                distort_limit=0.1,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.25,
            ),
            A.ThinPlateSpline(
                scale_range=(0.01, 0.03),
                num_control_points=4,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2,
            ),
            A.GridElasticDeform(
                num_grid_xy=(6, 6),
                magnitude=5,
                p=0.2,
            ),
            A.RandomScale(scale_limit=0.1, p=0.2),
            A.CropAndPad(
                percent=(-0.05, 0.05),
                keep_size=True,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.2,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(4, 8),
                hole_width_range=(4, 8),
                fill=0,
                p=0.2,
            ),
            A.GridDropout(ratio=0.3, p=0.2),
            A.Erasing(scale=(0.01, 0.05), p=0.2),
            A.PixelDropout(dropout_prob=0.02, p=0.2),
            A.RandomGridShuffle(grid=(3, 3), p=0.1),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_visibility=0.3, label_fields=["category_ids"]
        ),
        seed=seed,
    )


def make_repair_aug() -> A.Compose:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return A.Compose(
            [
                A.Sharpen(alpha=(0.3, 0.7), lightness=(0.9, 1.2), p=0.8),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.1, 0.3), contrast_limit=(0.1, 0.3), p=0.9
                ),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=0.7),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc", min_visibility=0.4, label_fields=["category_ids"]
            ),
        )
