import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl

from src.interface.base_trainer import BaseTrainer
from src.data.dataset import PromptTrainDataset
from src.models.lightning_module import PromptIRLightningModule
from src.utils.ema import EMACallback

logger = logging.getLogger(__name__)


class PromptIRTrainer(BaseTrainer):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model: Optional[PromptIRLightningModule] = None
        self.train_loader: Optional[DataLoader] = None
        self._callbacks: List[pl.Callback] = []
        self._logger: Any = None
        self._ckpt_best: Optional[pl.callbacks.ModelCheckpoint] = None

    def build_model(self) -> PromptIRLightningModule:
        train_cfg = self.cfg.get("training", {})

        warmup_epochs = train_cfg.get("warmup_epochs")
        if warmup_epochs is None:
            warmup_epochs = min(15, max(1, self.max_epochs // 10))

        scheduler_cfg = {
            "warmup_epochs": int(warmup_epochs),
            "max_epochs": int(self.max_epochs),
            "eta_min": float(train_cfg.get("eta_min", 1e-6)),
        }

        optimizer_cfg = {
            "lr": float(train_cfg.get("lr", 2e-4)),
            "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
            "betas": tuple(train_cfg.get("betas", (0.9, 0.999))),
        }

        loss_cfg = dict(self.cfg.get("loss", {}))
        loss_cfg.setdefault("type", "l1")
        loss_cfg.setdefault("w_pixel", 1.0)
        loss_cfg.setdefault("w_edge", 0.1)
        loss_cfg.setdefault("w_fft", 0.0)

        self.model = PromptIRLightningModule(
            model_cfg=self.cfg.get("model", {}),
            loss_cfg=loss_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            log_image_every=int(train_cfg.get("log_image_every", 6000)),
        )

        self._maybe_init_weights()
        logger.info(
            f"Built PromptIR (loss={loss_cfg['type']}, "
            f"w_pixel={loss_cfg['w_pixel']}, "
            f"w_edge={loss_cfg['w_edge']}, w_fft={loss_cfg['w_fft']})"
        )
        return self.model

    def _maybe_init_weights(self) -> None:
        train_cfg = self.cfg.get("training", {})
        init_from = train_cfg.get("init_from")
        resume_from = train_cfg.get("resume_from")
        if not init_from:
            return
        if resume_from:
            raise ValueError(
                "--init-from and --resume are mutually exclusive: "
                "--resume restores full training state, "
                "--init-from only loads weights for fine-tuning."
            )
        init_path = os.path.expanduser(init_from)
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f"--init-from checkpoint not found: {init_path}")
        logger.info(f"Initializing weights from {init_path} (no optimizer/epoch restore)")

        ckpt = torch.load(init_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        net_state = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
        if not net_state:
            net_state = state
        missing, unexpected = self.model.net.load_state_dict(net_state, strict=False)
        if missing:
            logger.warning(f"Missing keys: {len(missing)} (showing up to 5) {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys: {len(unexpected)} (showing up to 5) {unexpected[:5]}")

    def build_dataloaders(self) -> dict:
        data_cfg = self.cfg.get("data", {})
        train_cfg = self.cfg.get("training", {})

        ds_args = SimpleNamespace(
            data_file_dir=data_cfg.get("data_file_dir", "data/hw4_realse_dataset/"),
            derain_dir=data_cfg.get("derain_dir", "data/hw4_realse_dataset/"),
            desnow_dir=data_cfg.get("desnow_dir", "data/hw4_realse_dataset/"),
            de_type=list(data_cfg.get("de_type", ["derain", "desnow"])),
            patch_size=int(data_cfg.get("patch_size", 128)),
            num_aug=int(data_cfg.get("num_aug", 120)),
        )

        trainset = PromptTrainDataset(ds_args)

        num_workers = int(data_cfg.get("num_workers", 4))
        self.train_loader = DataLoader(
            trainset,
            batch_size=int(train_cfg.get("batch_size", 2)),
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )
        return {"train": self.train_loader}

    def build_callbacks(self) -> list:
        train_cfg = self.cfg.get("training", {})
        ckpt_name = self.cfg.get("experiment", {}).get(
            "ckpt_filename", f"best_{self.backbone}_{self.exp_name}"
        )

        self._ckpt_best = pl.callbacks.ModelCheckpoint(
            monitor="psnr",
            mode="max",
            save_top_k=1,
            save_last=bool(train_cfg.get("save_last", True)),
            dirpath=str(self.checkpoint_dir),
            filename=ckpt_name,
        )
        self._callbacks = [
            self._ckpt_best,
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        ]
        if bool(train_cfg.get("ema", False)):
            ema_decay = float(train_cfg.get("ema_decay", 0.999))
            self._callbacks.append(EMACallback(decay=ema_decay))
            logger.info(f"EMA enabled (decay={ema_decay})")
        return self._callbacks

    def build_logger(self) -> Any:
        log_cfg = self.cfg.get("logging", {})
        project = log_cfg.get("wandb_project")
        if project:
            from lightning.pytorch.loggers import WandbLogger
            wandb_dir = log_cfg.get("wandb_dir", "wandb")
            os.makedirs(wandb_dir, exist_ok=True)
            run_name = log_cfg.get("wandb_run_name", f"{self.backbone}_{self.exp_name}")
            self._logger = WandbLogger(project=project, name=run_name, save_dir=wandb_dir)
        else:
            from lightning.pytorch.loggers import TensorBoardLogger
            tb_dir = log_cfg.get("tensorboard_dir", "logs/")
            self._logger = TensorBoardLogger(save_dir=tb_dir)
        return self._logger

    def fit(self) -> Dict[str, Any]:
        train_cfg = self.cfg.get("training", {})

        callbacks = self.build_callbacks()
        pl_logger = self.build_logger()

        gpu_ids = train_cfg.get("num_gpus", 1)
        num_devices = len(gpu_ids) if isinstance(gpu_ids, list) else int(gpu_ids)
        strategy = "ddp_find_unused_parameters_true" if num_devices > 1 else "auto"

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu",
            devices=gpu_ids,
            strategy=strategy,
            precision=str(train_cfg.get("precision", "bf16-mixed")),
            gradient_clip_val=float(train_cfg.get("grad_clip", 0.5)),
            logger=pl_logger,
            callbacks=callbacks,
            log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        )

        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            ckpt_path=train_cfg.get("resume_from"),
        )

        result = {
            "best_model_path": getattr(self._ckpt_best, "best_model_path", None),
            "last_model_path": getattr(self._ckpt_best, "last_model_path", None),
            "best_score": (
                self._ckpt_best.best_model_score.item()
                if self._ckpt_best.best_model_score is not None else None
            ),
        }
        logger.info(f"Best PSNR : {result['best_score']}")
        logger.info(f"Best ckpt : {result['best_model_path']}")
        logger.info(f"Last ckpt : {result['last_model_path']}")
        return result
