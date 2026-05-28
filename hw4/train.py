import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    p = argparse.ArgumentParser(description="PromptIR Image Restoration Training")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--init-from", dest="init_from", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num-gpus", dest="num_gpus", default=None)
    p.add_argument("--wandb-project", dest="wandb_project", default=None)
    p.add_argument("--wandb-name", dest="wandb_name", default=None)
    return p.parse_args()


def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def _parse_gpus(value):
    if "," in value:
        return [int(v) for v in value.split(",") if v.strip() != ""]
    return int(value)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.epochs:
        cfg.setdefault("training", {})["max_epochs"] = args.epochs
    if args.num_gpus is not None:
        cfg.setdefault("training", {})["num_gpus"] = _parse_gpus(args.num_gpus)
    if args.resume:
        cfg.setdefault("training", {})["resume_from"] = args.resume
    if args.init_from:
        cfg.setdefault("training", {})["init_from"] = args.init_from
    if args.wandb_project:
        cfg.setdefault("logging", {})["wandb_project"] = args.wandb_project
    if args.wandb_name:
        cfg.setdefault("logging", {})["wandb_run_name"] = args.wandb_name

    from src.utils.config import build_output_paths
    out_paths = build_output_paths(cfg)
    cfg.setdefault("paths", {}).update(out_paths)

    model_name = cfg["model"]["name"]
    backbone = cfg["model"]["backbone"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        cfg["paths"].get("base_dir", "."), "logs", model_name, backbone, f"{run_id}.log"
    )
    setup_logging(log_file)
    logger = logging.getLogger("train")

    logger.info(f"Config   : {args.config}")
    logger.info(f"Log file : {log_file}")
    logger.info(f"Ckpt dir : {cfg['paths']['checkpoint_dir']}")
    logger.info(f"Chart dir: {cfg['paths']['chart_dir']}")

    from src.services.trainer import PromptIRTrainer
    trainer = PromptIRTrainer(cfg)
    trainer.run()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
