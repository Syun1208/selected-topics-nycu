import logging
import os
import time
from typing import Optional

import pandas as pd
import torch.nn as nn


def log_config(cfg, logger: logging.Logger | None = None) -> None:
    _log = logger or logging.getLogger("hw2")
    try:
        from omegaconf import OmegaConf

        _log.info("═══ Config ═══\n%s\n══════════════", OmegaConf.to_yaml(cfg))
    except Exception as e:
        _log.warning("Could not serialize config to YAML: %s", e)


def log_model(model: nn.Module, logger: logging.Logger | None = None) -> None:
    _log = logger or logging.getLogger("hw2")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info(
        "═══ Model Architecture ═══\n%s\nTotal params: %s  |  Trainable: %s\n══════════════════════════",
        model,
        f"{total:,}",
        f"{trainable:,}",
    )


def resolve_event_csv_path(output_dir: str, project_root: str | None = None) -> str:
    base = project_root or os.getcwd()
    abs_out = output_dir if os.path.isabs(output_dir) else os.path.join(base, output_dir)
    abs_out = os.path.normpath(abs_out)

    parts = abs_out.split(os.sep)
    exp_num, backbone, arch = parts[-1], parts[-2], parts[-3]

    csv_dir = os.path.join(base, "checkpoints", arch, backbone, exp_num)
    os.makedirs(csv_dir, exist_ok=True)
    return os.path.join(csv_dir, "train_event.csv")


def _load_csv(csv_path: str) -> list[dict]:
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path).to_dict("records")
        except Exception:
            pass
    return []


def _save_csv(csv_path: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    priority = ["iter", "epoch", "timestamp", "phase", "total_loss"]
    front = [c for c in priority if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(csv_path, index=False)


def log_train_metrics(csv_path: str, iters_per_epoch: int | None = None) -> None:
    from detectron2.utils.events import get_event_storage

    storage = get_event_storage()
    row: dict = {
        "iter": storage.iter,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "train",
    }
    if iters_per_epoch:
        row["epoch"] = round(storage.iter / iters_per_epoch, 3)
    for k, (v, _) in storage.latest().items():
        row[k] = float(v) if isinstance(v, (int, float)) else v

    rows = _load_csv(csv_path)
    rows.append(row)
    _save_csv(csv_path, rows)


def log_eval_metrics(
    csv_path: str,
    iteration: int,
    metrics: dict,
    iters_per_epoch: int | None = None,
) -> None:
    row: dict = {
        "iter": iteration,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "eval",
    }
    if iters_per_epoch:
        row["epoch"] = round(iteration / iters_per_epoch, 3)

    for task, val in metrics.items():
        if isinstance(val, dict):
            for k, v in val.items():
                row[f"{task}/{k}"] = float(v) if isinstance(v, (int, float)) else v
        else:
            row[task] = float(val) if isinstance(val, (int, float)) else val

    rows = _load_csv(csv_path)
    rows.append(row)
    _save_csv(csv_path, rows)


def setup_logger(
    output_dir: Optional[str] = None,
    name: str = "hw2",
    distributed_rank: int = 0,
    abbrev_name: Optional[str] = None,
) -> logging.Logger:
    from detectron2.utils.logger import setup_logger as d2_setup_logger

    logger = d2_setup_logger(
        output=output_dir,
        distributed_rank=distributed_rank,
        name=name,
        abbrev_name=abbrev_name or name,
    )

    if output_dir and distributed_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "run.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.info("Log file: %s", log_file)

    return logger
