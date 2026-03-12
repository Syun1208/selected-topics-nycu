import argparse
import json
import os
import warnings
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

import cv2
from tqdm import tqdm

from .coco_utils import build_image_annotations, load_coco
from .deblur import deblur_nafnet as _nafnet_infer
from .deblur import parse_gpu_ids


def denoise_nafnet(
    image,
    checkpoint_path: str,
    device: str = "cpu",
    config_path: str = None,
    gpu_ids: Optional[List[int]] = None,
):
    return _nafnet_infer(image, checkpoint_path, device, config_path=config_path, gpu_ids=gpu_ids)


def _denoise_worker(task: tuple):
    img_info, src_path, dst_path, anns, checkpoint_path, device, config_path, gpu_ids = task

    image = cv2.imread(src_path)
    if image is None:
        return None

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    try:
        image = denoise_nafnet(
            image, checkpoint_path, device, config_path=config_path, gpu_ids=gpu_ids
        )
    except Exception as exc:
        warnings.warn(
            f"[NAFNet denoise] failed for {src_path}: {exc}. Copying original.",
            stacklevel=2,
        )

    cv2.imwrite(dst_path, image)
    return (img_info, anns)


def run(args) -> None:
    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    gpu_ids = parse_gpu_ids(getattr(args, "nafnet_gpu_ids", None))

    if num_workers > 2:
        warnings.warn(
            f"[NAFNet] denoising with {num_workers} workers loads one model "
            "per process. Consider --num-workers 1 or 2 to avoid OOM.",
            stacklevel=2,
        )
    device_info = f"cuda:{gpu_ids}" if gpu_ids else args.device
    print(f"Using {num_workers} worker(s), device={device_info}")

    os.makedirs(args.output_images_dir, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    print("Loading COCO annotations …")
    coco = load_coco(args.input_json)
    img_anns = build_image_annotations(coco)
    images_dir = Path(args.images_dir).resolve()

    print(f"Denoising {len(coco['images'])} images …")
    tasks = []
    for img_info in coco["images"]:
        src = str(images_dir / img_info["file_name"])
        dst = str(Path(args.output_images_dir) / img_info["file_name"])
        anns = img_anns.get(img_info["id"], [])
        tasks.append(
            (
                img_info,
                src,
                dst,
                anns,
                args.denoise_checkpoint_path,
                args.device,
                args.denoise_config_path,
                gpu_ids,
            )
        )

    new_images, new_annotations = [], []
    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap(_denoise_worker, tasks), total=len(tasks), desc="Denoise"):
            if result is None:
                continue
            img_info, anns = result
            new_images.append(img_info)
            new_annotations.extend(anns)

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

    counts = Counter(a["category_id"] for a in new_annotations)
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    print("\nInstance distribution:")
    for cid in sorted(counts):
        print(f"  class {cats[cid]:>3}: {counts[cid]:>6}")
