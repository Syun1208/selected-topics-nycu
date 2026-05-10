import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.data.processors.cleaner import DataCleaner
from src.data.processors.augmentation import AugmentationPipeline

logger = logging.getLogger("compose")

NUM_CLASSES = 4


def _load_raw_sample(sample_dir: Path) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    img = tifffile.imread(str(sample_dir / "image.tif"))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = img.astype(np.uint8)

    masks: Dict[int, np.ndarray] = {}
    for cls_id in range(1, NUM_CLASSES + 1):
        p = sample_dir / f"class{cls_id}.tif"
        if p.exists():
            masks[cls_id] = tifffile.imread(str(p)).astype(np.float64)
    return img, masks


def _masks_to_instances(masks: Dict[int, np.ndarray]) -> Tuple[List[np.ndarray], List[int]]:
    binary_masks, category_ids = [], []
    for cls_id, mask in masks.items():
        for iid in np.unique(mask):
            if iid == 0:
                continue
            binary_masks.append((mask == iid).astype(np.uint8))
            category_ids.append(int(cls_id))
    return binary_masks, category_ids


def _rebuild_instance_mask(
    binary_masks: List[np.ndarray],
    category_ids: List[int],
    shape: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    class_masks: Dict[int, np.ndarray] = {}
    instance_counters: Dict[int, int] = {}
    for binary, cat in zip(binary_masks, category_ids):
        if not binary.any():
            continue
        instance_counters[cat] = instance_counters.get(cat, 0) + 1
        if cat not in class_masks:
            class_masks[cat] = np.zeros(shape, dtype=np.float64)
        class_masks[cat][binary > 0] = float(instance_counters[cat])
    return class_masks


def _save_sample(
    out_dir: Path,
    sample_id: str,
    image: np.ndarray,
    class_masks: Dict[int, np.ndarray],
) -> None:
    dst = out_dir / sample_id
    dst.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(dst / "image.tif"), image.astype(np.uint8))
    for cls_id, mask in class_masks.items():
        tifffile.imwrite(str(dst / f"class{cls_id}.tif"), mask.astype(np.float64))


def run_cleaning(
    cleaner: DataCleaner,
) -> dict:
    checks = [
        ("Mask errors",         cleaner.check_mask_errors),
        ("Bbox consistency",    cleaner.check_bbox_mask_consistency),
        ("Class imbalance",     cleaner.check_class_imbalance),
        ("Duplicates",          cleaner.check_duplicates),
        ("Size distribution",   cleaner.check_object_size_distribution),
    ]

    report = {}
    bar = tqdm(checks, desc="[1/3] Cleaning checks", unit="check",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    for check_name, check_fn in bar:
        bar.set_postfix_str(check_name)
        tqdm.write(f"      → {check_name} ...")
        key = check_name.lower().replace(" ", "_")
        report[key] = check_fn()
        tqdm.write(f"        done")

    if cleaner.report_dir:
        import json as _json
        cleaner.report_dir.mkdir(parents=True, exist_ok=True)
        out = cleaner.report_dir / "cleaning_report.json"
        serializable = {}
        for k, v in report.items():
            serializable[k] = {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
        with open(out, "w") as f:
            _json.dump(serializable, f, indent=2)
        tqdm.write(f"      Report → {out}")

    return report


def _split_ids(
    valid_ids: List[str],
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    ids = list(valid_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    return ids[n_val:], ids[:n_val]


def process_val(
    train_dir: Path,
    export_path: Path,
    val_ids: List[str],
) -> int:
    export_path.mkdir(parents=True, exist_ok=True)
    bar = tqdm(
        val_ids,
        desc="[2/4] Export val  ",
        unit="sample",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    exported = 0
    for sid in bar:
        img, masks_dict = _load_raw_sample(train_dir / sid)
        binary_masks, category_ids = _masks_to_instances(masks_dict)
        rebuilt = _rebuild_instance_mask(binary_masks, category_ids, img.shape[:2])
        _save_sample(export_path, sid, img, rebuilt)
        exported += 1
    tqdm.write(f"      Exported {exported} val samples → {export_path}")
    return exported


def process_train(
    train_dir: Path,
    export_path: Path,
    valid_ids: List[str],
    aug_cfg: dict,
    n_augments: int,
    skip_augment: bool,
) -> int:
    aug_pipeline = AugmentationPipeline(aug_cfg, mode="train") if not skip_augment else None

    pool_samples = []
    if aug_pipeline and aug_pipeline.copy_paste is not None:
        pool_ids = valid_ids[:20]
        for sid in tqdm(pool_ids, desc="  Loading copy-paste pool", unit="img", leave=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            img, masks_dict = _load_raw_sample(train_dir / sid)
            binary, cats = _masks_to_instances(masks_dict)
            pool_samples.append({"image": img, "masks": binary, "category_ids": cats})

    exported = 0
    total_expected = len(valid_ids) * (1 + (n_augments if not skip_augment else 0))

    outer = tqdm(
        valid_ids,
        desc="[3/4] Export train",
        unit="sample",
        total=len(valid_ids),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for sid in outer:
        img, masks_dict = _load_raw_sample(train_dir / sid)
        binary_masks, category_ids = _masks_to_instances(masks_dict)

        rebuilt = _rebuild_instance_mask(binary_masks, category_ids, img.shape[:2])
        _save_sample(export_path, sid, img, rebuilt)
        exported += 1
        outer.set_postfix(exported=exported, aug_copies=exported - len(valid_ids[:valid_ids.index(sid) + 1]))

        if skip_augment or aug_pipeline is None or n_augments == 0:
            continue

        for aug_idx in tqdm(
            range(n_augments),
            desc=f"    ↳ augment {sid[:8]}",
            unit="aug",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        ):
            aug_img = img.copy()
            aug_masks = list(binary_masks)
            aug_cats = list(category_ids)

            if aug_pipeline.copy_paste is not None and pool_samples:
                aug_img, aug_masks, aug_cats = aug_pipeline.copy_paste(
                    aug_img, aug_masks, aug_cats, pool_samples
                )

            if aug_masks:
                result = aug_pipeline.apply(aug_img, aug_masks, aug_cats)
                aug_img = result["image"]
                aug_masks = result["masks"]
                aug_cats = result["category_ids"]

            aug_id = f"{sid}_aug{aug_idx:03d}"
            rebuilt_aug = _rebuild_instance_mask(aug_masks, aug_cats, aug_img.shape[:2])
            _save_sample(export_path, aug_id, aug_img, rebuilt_aug)
            exported += 1

    tqdm.write(f"      Exported {exported} samples (orig + aug) → {export_path}")
    return exported


def process_test(
    test_dir: Path,
    export_path: Path,
    export_json_path: Path,
    mapping_json: Path,
) -> None:
    export_path.mkdir(parents=True, exist_ok=True)
    export_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(mapping_json) as f:
        meta_list = json.load(f)

    for meta in tqdm(
        meta_list,
        desc="[4/4] Copy test images",
        unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        src = test_dir / meta["file_name"]
        dst = export_path / meta["file_name"]
        if src.exists():
            shutil.copy2(str(src), str(dst))

    shutil.copy2(str(mapping_json), str(export_json_path))
    tqdm.write(f"      Test images → {export_path}")
    tqdm.write(f"      Test JSON   → {export_json_path}")


def _setup_logging(log_level: str, log_file: Optional[str]) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=fmt,
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Data cleaning + augmentation pipeline: reads raw data and exports processed datasets."
    )
    p.add_argument("--data-dir", default="data",
                   help="Root data folder (contains train/ and test_release/)")
    p.add_argument("--export-train-path", default="data/processed/train",
                   help="Output directory for cleaned (+ augmented) training samples")
    p.add_argument("--export-val-path", default="data/processed/val",
                   help="Output directory for cleaned validation samples (no augmentation)")
    p.add_argument("--val-ratio", type=float, default=0.2,
                   help="Fraction of valid samples reserved for validation (default: 0.1)")
    p.add_argument("--val-seed", type=int, default=42,
                   help="Random seed for train/val split")
    p.add_argument("--export-test-path", default="data/processed/test_release",
                   help="Output directory for processed test images")
    p.add_argument("--export-test-json-path", default="data/processed/test_image_name_to_ids.json",
                   help="Output path for the test mapping JSON")
    p.add_argument("--gpus", default="0",
                   help="Comma-separated GPU IDs (sets CUDA_VISIBLE_DEVICES)")
    p.add_argument("--n-augments", type=int, default=2,
                   help="Offline augmented copies per training sample (0 = none)")
    p.add_argument("--img-size", type=int, default=1024,
                   help="Target image size for augmentation transforms")
    p.add_argument("--skip-cleaning", action="store_true",
                   help="Skip data cleaning checks (use all samples)")
    p.add_argument("--skip-augment", action="store_true",
                   help="Export clean data only, without augmentation")
    p.add_argument("--allow-warnings", action="store_true", default=True,
                   help="Include samples that have only warnings (not hard errors)")
    p.add_argument("--report-dir", default=None,
                   help="Directory for the cleaning report JSON")
    p.add_argument("--log-file", default=None,
                   help="Write structured log output to this file (in addition to stdout)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    _setup_logging(args.log_level, args.log_file)

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test_release"
    mapping_json = data_dir / "test_image_name_to_ids.json"
    export_train = Path(args.export_train_path)
    export_val = Path(args.export_val_path)
    export_test = Path(args.export_test_path)
    export_test_json = Path(args.export_test_json_path)
    report_dir = Path(args.report_dir) if args.report_dir else export_train.parent / "reports"

    export_train.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Data Compose Pipeline")
    logger.info(f"  data-dir          : {data_dir}")
    logger.info(f"  export-train-path : {export_train}")
    logger.info(f"  export-val-path   : {export_val}")
    logger.info(f"  export-test-path  : {export_test}")
    logger.info(f"  export-test-json  : {export_test_json}")
    logger.info(f"  val-ratio         : {args.val_ratio}")
    logger.info(f"  gpus              : {args.gpus}")
    logger.info(f"  n-augments        : {args.n_augments}")
    logger.info(f"  skip-cleaning     : {args.skip_cleaning}")
    logger.info(f"  skip-augment      : {args.skip_augment}")
    if args.log_file:
        logger.info(f"  log-file          : {args.log_file}")
    logger.info("=" * 60)

    if args.skip_cleaning:
        valid_ids = sorted(
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
        tqdm.write(f"[1/4] Cleaning skipped — using all {len(valid_ids)} samples.")
        logger.info(f"Skipping cleaning. Using all {len(valid_ids)} samples.")
    else:
        cleaner = DataCleaner(str(train_dir), report_dir=str(report_dir))
        run_cleaning(cleaner)
        valid_ids = cleaner.get_valid_sample_ids(allow_warnings=args.allow_warnings)
        logger.info(f"Valid samples after cleaning: {len(valid_ids)}")

    train_ids, val_ids = _split_ids(valid_ids, val_ratio=args.val_ratio, seed=args.val_seed)
    tqdm.write(f"      Split → train: {len(train_ids)}, val: {len(val_ids)} (ratio={args.val_ratio})")
    logger.info(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    aug_cfg = {
        "img_size": args.img_size,
        "min_scale": 0.5,
        "max_scale": 2.0,
        "copy_paste": {
            "enabled": not args.skip_augment,
            "prob": 0.5,
            "max_paste": 4,
            "rare_classes": [4],
            "rare_boost": 3.0,
        },
    }

    process_val(
        train_dir=train_dir,
        export_path=export_val,
        val_ids=val_ids,
    )
    logger.info(f"Val export complete: {len(val_ids)} samples.")

    total = process_train(
        train_dir=train_dir,
        export_path=export_train,
        valid_ids=train_ids,
        aug_cfg=aug_cfg,
        n_augments=args.n_augments,
        skip_augment=args.skip_augment,
    )
    logger.info(f"Train export complete: {total} samples total.")

    if test_dir.exists() and mapping_json.exists():
        process_test(test_dir, export_test, export_test_json, mapping_json)
        logger.info("Test export complete.")
    else:
        logger.warning("Test directory or mapping JSON not found — skipping test export.")

    logger.info("Pipeline done.")


if __name__ == "__main__":
    main()
