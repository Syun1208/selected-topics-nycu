#!/usr/bin/env python3
import argparse
import logging
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import setup_sys_path

setup_sys_path(_PROJECT_ROOT)

from src.utils.logger import setup_logger

logger = logging.getLogger("hw2.visualize")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GradCAM + t-SNE visualization from a test YAML config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, metavar="YAML")
    parser.add_argument("--images-dir", default=None, metavar="DIR")
    parser.add_argument("--n-images", type=int, default=6, metavar="N")
    parser.add_argument(
        "--target-layer", default="backbone.model.layer4", metavar="LAYER"
    )
    parser.add_argument("--score-threshold", type=float, default=0.3, metavar="THR")
    parser.add_argument("--gpu-ids", default="0", metavar="IDS")
    parser.add_argument("--seed", type=int, default=None, metavar="SEED")
    parser.add_argument("--show", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    setup_logger()
    logger.info("Config         : %s", args.config)
    logger.info("Images dir     : %s", args.images_dir or "<from YAML>")
    logger.info("N images       : %d", args.n_images)
    logger.info("Target layer   : %s", args.target_layer)
    logger.info("Score threshold: %.2f", args.score_threshold)
    logger.info("GPU IDs        : %s", args.gpu_ids)
    logger.info("Seed           : %s", args.seed)

    from src.utils.visualize import visualize_from_yaml

    gradcam_fig, tsne_fig = visualize_from_yaml(
        yaml_path=args.config,
        test_images_dir=args.images_dir,
        n_images=args.n_images,
        target_layer=args.target_layer,
        score_threshold=args.score_threshold,
        seed=args.seed,
        show=args.show,
    )

    if gradcam_fig is not None:
        logger.info("GradCAM figure saved.")
    if tsne_fig is not None:
        logger.info("t-SNE figure saved.")
    if gradcam_fig is None and tsne_fig is None:
        logger.warning(
            "No figures produced — check that the model produces detections."
        )


if __name__ == "__main__":
    main()
