"""Main entry-point for dataset cleaning, augmentation, deblurring, and denoising.

Usage:
    python -m src.data.processors.data_cleaner [OPTIONS]
    # or via shell wrapper:
    bash src/data/processors/data_cleaner.sh [OPTIONS]

Sub-modules:
    augmentations  — albumentations pipelines + bbox helpers
    deblur         — NAFNet-based deblurring  (shared load_nafnet)
    denoise        — NAFNet-based denoising   (reuses load_nafnet from deblur)
    coco_utils     — COCO / CleanVision I/O
    workers        — multiprocessing worker functions
"""

import argparse
import json
import math
import os
import random
import warnings
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .coco_utils import build_image_annotations, load_cleanvision, load_coco
from .deblur import parse_gpu_ids
from .workers import _aug_worker, _copy_worker



def run(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    nafnet_gpu_ids = parse_gpu_ids(getattr(args, "nafnet_gpu_ids", None))

    if args.deblurry_blurry and not args.deblurry_checkpoint_path and not args.deblurry_config_path:
        raise ValueError(
            "--deblurry-blurry requires --deblurry-checkpoint-path or --deblurry-config-path.\n"
            "Download NAFNet-GoPro-width64.pth from:\n"
            "  https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC"
        )
    if args.denoise_noisy and not args.denoise_checkpoint_path and not args.denoise_config_path:
        raise ValueError(
            "--denoise-noisy requires --denoise-checkpoint-path or --denoise-config-path.\n"
            "Download NAFNet-SIDD-width64.pth from:\n"
            "  https://drive.google.com/file/d/14Fht4x2Ft4HEDMoBT4SRyiqgQ73YRa6I"
        )

    nafnet_active = args.deblurry_blurry or args.denoise_noisy
    if nafnet_active and num_workers > 2:
        warnings.warn(
            f"[NAFNet] deblurry mode with {num_workers} workers loads one model "
            "per process. Consider --num-workers 1 or 2 to avoid OOM.",
            stacklevel=2,
        )
    print(f"Using {num_workers} CPU worker(s)")

    os.makedirs(args.output_images_dir, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    print("Loading COCO annotations …")
    coco = load_coco(args.train_json)
    img_anns = build_image_annotations(coco)

    print("Loading CleanVision issues …")
    cv_issues = load_cleanvision(args.cleanvision_csv) if args.cleanvision_csv else {}

    images_dir = Path(args.images_dir).resolve()

    print("Filtering images …")
    kept_images, removed = [], Counter()
    for img_info in coco["images"]:
        abs_path = str((images_dir / img_info["file_name"]).resolve())
        issues = cv_issues.get(abs_path, {})
        if args.remove_near_duplicates and issues.get("is_near_duplicates_issue"):
            removed["near_duplicates"] += 1
            continue
        if args.remove_dark and issues.get("is_dark_issue"):
            removed["dark"] += 1
            continue
        if args.remove_blurry and issues.get("is_blurry_issue"):
            removed["blurry"] += 1
            continue
        kept_images.append(img_info)

    print(f"  Kept {len(kept_images)} / {len(coco['images'])} images")
    for reason, count in removed.items():
        print(f"  Removed {count} ({reason})")

    class_images: dict = defaultdict(list)
    instance_counts: Counter = Counter()
    for img_info in kept_images:
        anns = img_anns.get(img_info["id"], [])
        for ann in anns:
            instance_counts[ann["category_id"]] += 1
        for cid in set(a["category_id"] for a in anns):
            class_images[cid].append(img_info)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    print("\nInstance distribution after filtering:")
    for cid in sorted(instance_counts):
        print(f"  class {cats[cid]:>3}: {instance_counts[cid]:>6}")

    aug_target = args.aug_target or max(instance_counts.values())

    print(f"\nCopying {len(kept_images)} images with {num_workers} workers …")
    copy_tasks = []
    for img_info in kept_images:
        src = str(images_dir / img_info["file_name"])
        dst = str(Path(args.output_images_dir) / img_info["file_name"])
        abs_src = str((images_dir / img_info["file_name"]).resolve())
        issues = cv_issues.get(abs_src, {})
        anns = img_anns.get(img_info["id"], [])
        copy_tasks.append(
            (
                img_info,
                src,
                dst,
                issues,
                args.repair_dark,
                args.repair_blurry,
                args.deblurry_blurry,
                args.nafnet_device,
                args.deblurry_checkpoint_path,
                args.deblurry_config_path,
                args.denoise_noisy,
                args.denoise_checkpoint_path,
                args.denoise_config_path,
                nafnet_gpu_ids,
                anns,
            )
        )

    new_images, new_annotations = [], []
    next_img_id = max(img["id"] for img in coco["images"]) + 1
    next_ann_id = max(a["id"] for a in coco["annotations"]) + 1

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(_copy_worker, copy_tasks), total=len(copy_tasks), desc="Copy"):
            if result is None:
                continue
            img_info, anns = result
            new_images.append(img_info)
            new_annotations.extend(anns)

    if args.aug_target:
        print(
            f"\nAugmenting minority classes (target={aug_target}) " f"with {num_workers} workers …"
        )
        for cid in sorted(instance_counts):
            current = instance_counts[cid]
            if current >= aug_target:
                continue
            candidates = class_images[cid]
            if not candidates:
                continue
            # Fix: tính dựa trên avg instances/image để không undergenerate.
            # Công thức cũ: (aug_target - current) // len(candidates) bỏ qua
            # việc mỗi ảnh trung bình có nhiều hơn 1 instance của class này.
            avg_instances_per_image = current / len(candidates)
            needed_new_images = (aug_target - current) / max(1.0, avg_instances_per_image)
            aug_per_image = max(1, math.ceil(needed_new_images / len(candidates)))
            print(
                f"  class {cats[cid]}: {current} → {aug_target} "
                f" (×{aug_per_image} on {len(candidates)} images)"
            )

            aug_tasks = [
                (
                    img_info,
                    str(images_dir / img_info["file_name"]),
                    args.output_images_dir,
                    cid,
                    aug_per_image,
                    args.seed + cid + i,
                    img_anns.get(img_info["id"], []),
                )
                for i, img_info in enumerate(candidates)
            ]

            generated = 0
            with Pool(num_workers) as pool:
                for batch in tqdm(
                    pool.imap(_aug_worker, aug_tasks),
                    total=len(aug_tasks),
                    desc=f"  class {cats[cid]}",
                    leave=False,
                ):
                    for img_info, aug_fname, aug_anns, aug_h, aug_w in batch:
                        new_img = dict(img_info)
                        new_img["id"] = next_img_id
                        new_img["file_name"] = aug_fname
                        new_img["height"] = aug_h
                        new_img["width"] = aug_w
                        new_images.append(new_img)
                        for ann in aug_anns:
                            new_ann = dict(ann)
                            new_ann["id"] = next_ann_id
                            new_ann["image_id"] = next_img_id
                            new_annotations.append(new_ann)
                            next_ann_id += 1
                        next_img_id += 1
                        generated += 1

            print(f"    → {generated} augmented images generated")

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_annotations,
    }
    with open(args.output_json, "w") as f:
        json.dump(new_coco, f)

    print(f"\nSaved {len(new_images)} images, {len(new_annotations)} annotations")
    print(f"  JSON   → {args.output_json}")
    print(f"  Images → {args.output_images_dir}")

    final_counts = Counter(a["category_id"] for a in new_annotations)
    print("\nFinal instance distribution:")
    for cid in sorted(final_counts):
        print(f"  class {cats[cid]:>3}: {final_counts[cid]:>6}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and augment a COCO-format detection dataset."
    )

    parser.add_argument("--train-json", default="data/train.json")
    parser.add_argument("--images-dir", default="data/train")
    parser.add_argument("--cleanvision-csv", default="notebooks/outputs/cleanvision_issues.csv")
    parser.add_argument("--output-json", default="data/clean_train.json")
    parser.add_argument("--output-images-dir", default="data/clean_train")

    g = parser.add_argument_group("parallelism")
    g.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU workers for parallel processing. 0 = all cores (default: 0)",
    )

    g = parser.add_argument_group("filtering")
    g.add_argument("--remove-dark", action="store_true")
    g.add_argument(
        "--remove-blurry",
        action="store_true",
        help="Remove blurry images (caution: 76%% of dataset flagged)",
    )
    g.add_argument("--remove-near-duplicates", action="store_true")

    g = parser.add_argument_group("repair")
    g.add_argument(
        "--repair-dark", action="store_true", help="Apply CLAHE + brightness repair to dark images"
    )
    g.add_argument(
        "--repair-blurry",
        action="store_true",
        help="Apply albumentations sharpening repair to blurry images",
    )
    g.add_argument(
        "--deblurry-blurry",
        action="store_true",
        help="Deblur blurry images with NAFNet (deep learning). Requires --deblurry-checkpoint-path.",
    )
    g.add_argument(
        "--deblurry-checkpoint-path",
        default=None,
        help=(
            "Path to NAFNet-GoPro-width64.pth checkpoint for deblurring.\n"
            "Download: https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC\n"
            "Required when --deblurry-blurry is set (unless provided via --deblurry-config-path)."
        ),
    )
    g.add_argument(
        "--deblurry-config-path",
        default=None,
        help=(
            "Path to a NAFNet YAML config for deblurring (e.g. NAFNet-GoPro-width32.yml). "
            "Reads network_g arch params from the config. "
            "path.pretrain_network_g is used as checkpoint fallback if --deblurry-checkpoint-path is not set."
        ),
    )
    g.add_argument(
        "--denoise-noisy",
        action="store_true",
        help="Denoise noisy images with NAFNet (deep learning). Requires --denoise-checkpoint-path.",
    )
    g.add_argument(
        "--denoise-checkpoint-path",
        default=None,
        help=(
            "Path to NAFNet-SIDD-width64.pth checkpoint for denoising.\n"
            "Download: https://drive.google.com/file/d/14Fht4x2Ft4HEDMoBT4SRyiqgQ73YRa6I\n"
            "Required when --denoise-noisy is set (unless provided via --denoise-config-path)."
        ),
    )
    g.add_argument(
        "--denoise-config-path",
        default=None,
        help=(
            "Path to a NAFNet YAML config for denoising (e.g. NAFNet-SIDD-width32.yml). "
            "Reads network_g arch params from the config. "
            "path.pretrain_network_g is used as checkpoint fallback if --denoise-checkpoint-path is not set."
        ),
    )
    g.add_argument(
        "--nafnet-device",
        default="cpu",
        help="Torch device for NAFNet inference: 'cpu' or 'cuda' (default: cpu)",
    )
    g.add_argument(
        "--nafnet-gpu-ids",
        default=None,
        help=(
            "Comma-separated CUDA GPU IDs for NAFNet inference (e.g. '0' or '0,1,2'). "
            "Overrides --nafnet-device when set. Multiple IDs enable DataParallel."
        ),
    )

    g = parser.add_argument_group("augmentation")
    g.add_argument(
        "--aug-target",
        type=int,
        default=None,
        help="Target instance count per class (no aug if not set)",
    )
    g.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
