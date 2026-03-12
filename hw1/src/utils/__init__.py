import logging
from typing import Optional

import torch.nn as nn

MAX_PARAMS = 100_000_000


def log_model_size(
    model: nn.Module,
    logger: Optional[logging.Logger] = None,
) -> int:
    """Print and log total trainable parameters. Raises ValueError if > 100M."""
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
