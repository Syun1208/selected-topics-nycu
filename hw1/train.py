import argparse
import dataclasses
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.distributed as dist
import yaml

from src.utils.seed import set_seed
from src.services.implement.train import Trainer
from src.data.dataclass import (
    DataConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
    TrainingConfig,
)


def load_config(config_path: str) -> TrainConfig:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_raw = dict(raw.get("model", {}))
    lora_raw = model_raw.pop("lora", {})
    lora_cfg = LoraConfig(**lora_raw)
    model_cfg = ModelConfig(**model_raw, lora=lora_cfg)
    data_raw = raw.get("data", {})
    data_cfg = DataConfig(
        train_dir=data_raw.get("train_dir", "data/train"),
        val_dir=data_raw.get("val_dir", "data/val"),
        test_dir=data_raw.get("test_dir", "data/test"),
        image_size=data_raw.get("image_size", 224),
        crop_size=data_raw.get("crop_size", 320),
        resize_size=data_raw.get("resize_size", 334),
        batch_size=data_raw.get("batch_size", 64),
        num_workers=data_raw.get("num_workers", 4),
        use_augmentation=data_raw.get("use_augmentation", True),
    )
    train_cfg = TrainingConfig(**raw.get("training", {}))
    output_cfg = OutputConfig(**raw.get("output", {}))

    return TrainConfig(
        model=model_cfg, data=data_cfg, training=train_cfg, output=output_cfg
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def init_distributed():
    """Initialize DDP from torchrun env vars.

    Returns (rank, world_size, local_rank, is_distributed).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1, 0, False

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, local_rank, True


def setup_logging(log_dir: str, rank: int) -> None:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(
            logging.FileHandler(
                log_path /
                f"train_{timestamp}.log"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def log_config(config, args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  config file : {args.config}")
    logger.info(f"  seed        : {args.seed}")
    for section_name, section in dataclasses.asdict(config).items():
        logger.info(f"  [{section_name}]")
        for k, v in section.items():
            logger.info(f"    {k}: {v}")
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, is_distributed = init_distributed()

    set_seed(args.seed)

    config = load_config(args.config)

    setup_logging(config.output.log_dir, rank)
    if rank == 0:
        log_config(config, args)

    trainer = Trainer(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_distributed=is_distributed,
    )
    trainer.model_name = Path(args.config).stem
    trainer.setup(config)

    if config.model.checkpoint:
        trainer.load_checkpoint(config.model.checkpoint)

    trainer.train()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
