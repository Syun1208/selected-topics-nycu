import logging
from pathlib import Path
from typing import Optional

import pandas as pd

import torch.nn as nn

MAX_PARAMS = 100_000_000


def log_model_size(
    model: nn.Module,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Print and log total trainable parameters.

    Raises ValueError if > 100M."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    msg = f"Model parameters: total={total:,}  trainable={trainable:,}"

    if logger:
        logger.info(msg)
    else:
        print(msg)

    # Unwrap to inner timm model if available for a cleaner architecture view
    inner = getattr(model, "model", model)
    arch = str(inner)
    if logger:
        logger.info("Model architecture:\n%s", arch)
    else:
        print(arch)

    if total > MAX_PARAMS:
        raise ValueError(
            f"Model has {total:,} parameters, exceeding the 100M limit."
        )

    return total


def log_performance_csv(
    model_name: str,
    model_size: int,
    epochs: int,
    lr: float,
    optimizer: str,
    image_size: int,
    train_acc: float,
    val_acc: float,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    runs_dir: str = "runs",
) -> None:
    """Log training performance metrics to runs/<model_name>/performance.csv.

    Appends a row if the file already exists, so multiple runs accumulate.
    """
    out_dir = Path(runs_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "performance.csv"

    row = pd.DataFrame([{
        "model": model_name,
        "model_size": model_size,
        "epochs": epochs,
        "lr": lr,
        "optimizer": optimizer,
        "image_size": image_size,
        "train_loss": round(train_loss, 6),
        "val_loss": round(val_loss, 6),
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
    }])

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, row], ignore_index=True)
    else:
        df = row

    df.to_csv(csv_path, index=False)
