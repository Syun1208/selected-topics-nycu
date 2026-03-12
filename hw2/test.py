#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import setup_sys_path

setup_sys_path(_PROJECT_ROOT)

from detectron2.config import LazyConfig
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import default_setup

from src.data.loaders.register import HW2_CLASS_NAMES
from src.services.tester import DetrexTester
from src.utils.checkpoint import find_latest_checkpoint
from src.utils.config import apply_yaml_overrides, load_yaml_config
from src.utils.logger import setup_logger

logger = logging.getLogger("hw2.test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW2 inference script")
    parser.add_argument(
        "--config",
        required=True,
        metavar="YAML",
        help="Path to test YAML config (e.g. configs/test/.../0.yaml)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        metavar="IDS",
        help="Override testing.gpu_ids from YAML (e.g. '0' or '0,1')",
    )
    return parser


def _register_test_split(test_images_dir: str) -> str:
    name = "hw2_test"
    if name in DatasetCatalog.list():
        return name

    image_paths = sorted(
        glob.glob(os.path.join(test_images_dir, "*.png"))
        + glob.glob(os.path.join(test_images_dir, "*.jpg"))
    )

    def _get_dicts():
        import struct

        dicts = []
        for i, img_path in enumerate(image_paths):
            with open(img_path, "rb") as f:
                header = f.read(24)
            if header[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", header[16:24])
            elif header[:2] == b"\xff\xd8":
                from PIL import Image

                with Image.open(img_path) as im:
                    w, h = im.size
            else:
                from PIL import Image

                with Image.open(img_path) as im:
                    w, h = im.size
            stem = os.path.splitext(os.path.basename(img_path))[0]
            dicts.append(
                {
                    "file_name": img_path,
                    "image_id": int(stem) if stem.isdigit() else i + 1,
                    "height": h,
                    "width": w,
                    "annotations": [],
                }
            )
        return dicts

    DatasetCatalog.register(name, _get_dicts)
    MetadataCatalog.get(name).set(thing_classes=list(HW2_CLASS_NAMES))
    logger.info("Registered test split '%s' (%d images).", name, len(image_paths))
    return name


def main():
    args = build_parser().parse_args()

    yaml_cfg = load_yaml_config(args.config)

    # GPU IDs: CLI override > YAML
    gpu_ids = args.gpu_ids or str(yaml_cfg.get("testing", {}).get("gpu_ids", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # output_file — save predictions directly to this path
    output_file = str(yaml_cfg.get("output_file"))
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)

    setup_logger()
    logger.info("Config      : %s", args.config)
    logger.info("GPU IDs     : %s", gpu_ids)
    logger.info("Output file : %s", output_file)

    # Register test images
    ds = yaml_cfg.dataset
    test_images_dir = str(ds.get("test_images_dir", "data/test"))
    test_dataset_name = _register_test_split(test_images_dir)

    # Load model config (reuses the train model.py)
    model_config_path = os.path.join(_PROJECT_ROOT, str(yaml_cfg.model.config))
    lazy_cfg = LazyConfig.load(model_config_path)
    lazy_cfg = apply_yaml_overrides(lazy_cfg, yaml_cfg)

    # Override test dataloader to use the registered test dataset (not "valid")
    lazy_cfg.dataloader.test.dataset.names = test_dataset_name
    lazy_cfg.dataloader.test.num_workers = 0

    # Checkpoint: YAML field → auto-detect from log dir → error
    checkpoint = str(yaml_cfg.get("checkpoint", "")).strip()
    if not checkpoint or not os.path.isfile(checkpoint):
        logger.error(
            "Checkpoint not found at '%s'. Attempting to auto-detect from log dir...", checkpoint
        )
    logger.info("Auto-detected checkpoint: %s", checkpoint)

    lazy_cfg.train.init_checkpoint = checkpoint
    lazy_cfg.train.output_dir = output_dir
    logger.info("Checkpoint  : %s", checkpoint)

    class _FakeArgs:
        config_file = args.config
        dist_url = "auto"
        eval_only = True
        machine_rank = 0
        num_gpus = 1
        num_machines = 1
        opts = []
        resume = False

    default_setup(lazy_cfg, _FakeArgs())

    tester = DetrexTester()
    tester.setup(lazy_cfg)

    # Run inference and save to output_file
    predictions = tester._run_inference()
    with open(output_file, "w") as f:
        json.dump(predictions, f)
    logger.info("Saved %d predictions → %s", len(predictions), output_file)


if __name__ == "__main__":
    main()
