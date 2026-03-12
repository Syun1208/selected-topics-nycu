import logging
import os
import sys
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def setup_sys_path(project_root: Optional[str] = None) -> None:
    root = project_root or _project_root()
    for p in [
        os.path.join(root, "configs", "detrex"),
        os.path.join(root, "configs", "detectron2"),
        root,
    ]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def load_yaml_config(config_path: str) -> DictConfig:
    assert os.path.isfile(config_path), f"Config file not found: {config_path }"
    return OmegaConf.load(config_path)


def apply_cli_opts(yaml_cfg: DictConfig, opts) -> DictConfig:
    for opt in opts:
        if "=" not in opt:
            continue
        key, value = opt.split("=", 1)
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        if value in ("true", "True"):
            value = True
        elif value in ("false", "False"):
            value = False
        OmegaConf.update(yaml_cfg, key, value, merge=True)
    return yaml_cfg


def build_detectron2_args(
    config_path: str,
    dist_url: str = "auto",
    eval_only: bool = False,
    num_gpus: int = 1,
    num_machines: int = 1,
    machine_rank: int = 0,
    resume: bool = False,
):
    class _Args:
        pass

    args = _Args()
    args.config_file = config_path
    args.dist_url = dist_url
    args.eval_only = eval_only
    args.num_gpus = num_gpus
    args.num_machines = num_machines
    args.machine_rank = machine_rank
    args.opts = []
    args.resume = resume
    return args


def apply_yaml_overrides(lazy_cfg: Any, yaml_cfg: DictConfig) -> Any:
    if "output_dir" in yaml_cfg:
        lazy_cfg.train.output_dir = yaml_cfg.output_dir

    backbone_name = OmegaConf.select(yaml_cfg, "model.pretrained_backbone", default="")
    
    try:
        if backbone_name:
            lazy_cfg.model.backbone.model_name = backbone_name
            lazy_cfg.model.backbone.pretrained = True
        else:
            lazy_cfg.model.backbone.pretrained = False
    except Exception:
        pass
    
    if backbone_name and hasattr(lazy_cfg.model, "backbone_name"):
        lazy_cfg.model.backbone_name = backbone_name
        lazy_cfg.model.backbone_pretrained = True

    if OmegaConf.select(yaml_cfg, "model.num_classes"):
        lazy_cfg.model.num_classes = yaml_cfg.model.num_classes
        try:
            lazy_cfg.model.criterion.num_classes = yaml_cfg.model.num_classes
        except Exception:
            pass

    t = OmegaConf.select(yaml_cfg, "training", default={})
    if t:
        _apply_training_overrides(lazy_cfg, t)

    ds = OmegaConf.select(yaml_cfg, "dataset", default={})
    if ds:
        _apply_dataset_overrides(lazy_cfg, ds)

    try:
        lazy_cfg.dataloader.evaluator.output_dir = lazy_cfg.train.output_dir
    except Exception:
        pass

    return lazy_cfg


def _apply_training_overrides(lazy_cfg: Any, t) -> None:
    _SCALAR_MAP = {
        "max_iter": "train.max_iter",
        "eval_period": "train.eval_period",
        "log_period": "train.log_period",
        "checkpoint_period": "train.checkpointer.period",
        "amp": "train.amp.enabled",
        "learning_rate": "optimizer.lr",
        "weight_decay": "optimizer.weight_decay",
        "batch_size": "dataloader.train.total_batch_size",
        "init_checkpoint": "train.init_checkpoint",
    }
    for key, path in _SCALAR_MAP.items():
        if key in t:
            try:
                _setattr_nested(lazy_cfg, path, t[key])
            except Exception:
                pass

    if "num_workers" in t:
        try:
            lazy_cfg.dataloader.train.num_workers = t.num_workers
            lazy_cfg.dataloader.test.num_workers = t.num_workers
        except Exception:
            pass

    clip = OmegaConf.select(t, "clip_grad", default=None)
    if clip is not None:
        try:
            lazy_cfg.train.clip_grad.enabled = clip.get("enabled", True)
            if "max_norm" in clip:
                lazy_cfg.train.clip_grad.params.max_norm = clip.max_norm
        except Exception:
            pass


def _apply_dataset_overrides(lazy_cfg: Any, ds) -> None:
    if "register_train" in ds:
        lazy_cfg.dataloader.train.dataset.names = ds.register_train
    if "register_valid" in ds:
        lazy_cfg.dataloader.test.dataset.names = ds.register_valid
        lazy_cfg.dataloader.evaluator.dataset_name = ds.register_valid


def _setattr_nested(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
