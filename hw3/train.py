import argparse
import logging
import os
import sys
import torch
import warnings
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

import yaml

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Instance Segmentation Training")
    p.add_argument("--config", required=True, help="Path to train YAML config")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device", default=None, help="Override device (e.g. cuda:0, cpu)")
    p.add_argument("--epochs", type=int, default=None, help="Override max_epochs from config")
    p.add_argument("--gpu-ids", default=None,
                   help="GPU IDs override, e.g. '3' or '3,6,7'. Also read from training.gpu_ids in config.")
    return p.parse_args()


def setup_logging(log_file: str) -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # CLI overrides
    if args.gpu_ids:
        cfg.setdefault("training", {})["gpu_ids"] = args.gpu_ids
    if args.device:
        cfg.setdefault("training", {})["device"] = args.device
    if args.epochs:
        cfg.setdefault("training", {})["max_epochs"] = args.epochs

    # DDP: torchrun sets LOCAL_RANK when launched with --nproc_per_node > 1.
    # CUDA_VISIBLE_DEVICES in the launch script remaps physical GPU IDs so that
    # LOCAL_RANK 0/1/2 maps correctly to the requested GPUs (e.g. 1,5,7).
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=2),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        cfg.setdefault("training", {}).update({
            "local_rank": local_rank,
            "world_size": dist.get_world_size(),
        })

    is_main = local_rank <= 0  # single GPU (-1) or DDP rank 0

    from src.utils.config import build_output_paths
    out_paths = build_output_paths(cfg)
    cfg["paths"].update(out_paths)

    model_name = cfg["model"]["name"]
    backbone = cfg["model"]["backbone"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        cfg["paths"].get("base_dir", "."), "logs", model_name, backbone, f"{run_id}.log"
    )

    if is_main:
        setup_logging(log_file)
    logger = logging.getLogger("train")

    if is_main:
        logger.info(f"Config   : {args.config}")
        logger.info(f"Log file : {log_file}")
        logger.info(f"Ckpt dir : {cfg['paths']['checkpoint_dir']}")
        logger.info(f"Chart dir: {cfg['paths']['chart_dir']}")
        logger.info(f"gpu_ids  : {cfg.get('training', {}).get('gpu_ids', 'not set')}")
        if local_rank >= 0:
            logger.info(f"DDP      : rank={local_rank}, world_size={cfg['training']['world_size']}")
        logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")

    from src.services.trainer import MMDetTrainer
    trainer = MMDetTrainer(cfg)

    # Pass resume path through config so build_model() loads it at the right time.
    if args.resume:
        cfg.setdefault("training", {})["resume_from"] = args.resume
        if is_main:
            logger.info(f"Will resume from: {args.resume}")

    trainer.run()

    if local_rank >= 0:
        import torch.distributed as dist
        dist.destroy_process_group()

    if is_main:
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
