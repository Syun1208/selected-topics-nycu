import os
import warnings
from pathlib import Path
from typing import List, Optional

import cv2

from .augmentations import apply_aug_to_image, make_repair_aug, make_strong_aug
from .deblur import deblur_nafnet
from .denoise import denoise_nafnet


def _copy_worker(task: tuple):
    (
        img_info,
        src_path,
        dst_path,
        issues,
        repair_dark,
        repair_blurry,
        deblurry_blurry,
        nafnet_device,
        deblurry_checkpoint_path,
        deblurry_config_path,
        denoise_noisy,
        denoise_checkpoint_path,
        denoise_config_path,
        nafnet_gpu_ids,
        anns,
    ) = task

    image = cv2.imread(src_path)
    if image is None:
        return None

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if deblurry_blurry and issues.get("is_blurry_issue"):
        try:
            image = deblur_nafnet(
                image,
                deblurry_checkpoint_path,
                nafnet_device,
                config_path=deblurry_config_path,
                gpu_ids=nafnet_gpu_ids,
            )
        except Exception as exc:
            warnings.warn(
                f"[NAFNet deblur] failed for {src_path}: {exc}. Skipping.",
                stacklevel=2,
            )

    if denoise_noisy and issues.get("is_noisy_issue"):
        try:
            image = denoise_nafnet(
                image,
                denoise_checkpoint_path,
                nafnet_device,
                config_path=denoise_config_path,
                gpu_ids=nafnet_gpu_ids,
            )
        except Exception as exc:
            warnings.warn(
                f"[NAFNet denoise] failed for {src_path}: {exc}. Skipping.",
                stacklevel=2,
            )

    should_repair = (repair_dark and issues.get("is_dark_issue")) or (
        repair_blurry and issues.get("is_blurry_issue")
    )
    if should_repair and anns:
        image, anns = apply_aug_to_image(image, anns, make_repair_aug())

    cv2.imwrite(dst_path, image)
    return (img_info, anns)


def _aug_worker(task: tuple) -> list:
    img_info, src_path, output_dir, cid, aug_per_image, seed, anns = task

    image = cv2.imread(src_path)
    if image is None:
        return []

    strong_aug = make_strong_aug(seed=seed)
    results = []
    stem = Path(src_path).stem

    for i in range(aug_per_image):
        aug_image, aug_anns = apply_aug_to_image(image, anns, strong_aug)
        if not aug_anns:
            continue
        aug_fname = f"aug_{stem}_{cid}_{i}.png"
        dst = os.path.join(output_dir, aug_fname)
        cv2.imwrite(dst, aug_image)
        aug_h, aug_w = aug_image.shape[:2]
        results.append((img_info, aug_fname, aug_anns, aug_h, aug_w))

    return results
