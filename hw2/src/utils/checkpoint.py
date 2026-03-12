import glob
import logging
import os
import re
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    pattern = os.path.join(output_dir, "model_*.pth")
    checkpoints = glob.glob(pattern)

    final = os.path.join(output_dir, "model_final.pth")
    if os.path.isfile(final):
        return final

    if not checkpoints:
        return None

    def _iter_from_name(path: str) -> int:
        match = re.search(r"model_(\d+)\.pth$", path)
        return int(match.group(1)) if match else -1

    latest = max(checkpoints, key=_iter_from_name)
    logger.info("Latest checkpoint found: %s", latest)
    return latest


def load_model_weights(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    from detectron2.checkpoint import DetectionCheckpointer

    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    if strict:
        state = torch.load(checkpoint_path, map_location="cpu")
        state_dict = state.get("model", state)
        model.load_state_dict(state_dict, strict=True)
        logger.info("Loaded checkpoint (strict) from '%s'.", checkpoint_path)
    else:
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(checkpoint_path)
        logger.info("Loaded checkpoint (non-strict) from '%s'.", checkpoint_path)

    return model
