import argparse
import logging
import os
import sys
import warnings

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser(description="PromptIR Image Restoration Inference")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--test-path", dest="test_path", default=None)
    p.add_argument("--output-npz", dest="output_npz", default=None)
    p.add_argument("--gpu-ids", dest="gpu_ids", default=None)
    p.add_argument("--self-ensemble", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--tile", type=lambda s: s.lower() == "true", default=None)
    p.add_argument("--tile-size", dest="tile_size", type=int, default=None)
    p.add_argument("--tile-overlap", dest="tile_overlap", type=int, default=None)
    return p.parse_args()


def setup_logging() -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("test")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint:
        cfg.setdefault("paths", {})["checkpoint_path"] = args.checkpoint
    inf = cfg.setdefault("inference", {})
    if args.gpu_ids is not None:
        inf["gpu_ids"] = args.gpu_ids
    if args.self_ensemble is not None:
        inf["self_ensemble"] = args.self_ensemble
    if args.use_ema is not None:
        inf["use_ema"] = args.use_ema
    if args.tile is not None:
        inf["tile"] = args.tile
    if args.tile_size is not None:
        inf["tile_size"] = args.tile_size
    if args.tile_overlap is not None:
        inf["tile_overlap"] = args.tile_overlap
    if args.test_path:
        cfg.setdefault("data", {})["test_path"] = args.test_path

    np.random.seed(0)
    torch.manual_seed(0)

    from src.utils.config import build_output_paths
    out_paths = build_output_paths(cfg)
    cfg.setdefault("paths", {}).update(out_paths)

    from src.services.tester import PromptIRTester
    tester = PromptIRTester(cfg)
    tester.run(output_path=args.output_npz)

    logger.info("Inference complete.")


if __name__ == "__main__":
    main()
