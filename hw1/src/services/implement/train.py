import math
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader, DistributedSampler, WeightedRandomSampler
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataclass import BatchResult, EpochResult, TrainConfig
from models.lora_model import count_parameters, get_lora_state_dict
from data.dataset import (
    ClassificationDataset,
    get_train_transforms,
    get_val_transforms,
)
from utils.visualize import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_curves,
)
from models.model_factory import ModelFactory
from services.interface.train import TrainerInterface
from utils import log_model_size, log_performance_csv
from utils.logger import get_logger
from utils.timer import Timer


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Trainer(TrainerInterface):
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
        is_distributed: bool = False
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_distributed = is_distributed

        self.config: Optional[TrainConfig] = None
        self.model: Optional[nn.Module] = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.train_sampler: Optional[DistributedSampler] = None
        self.writer: Optional[SummaryWriter] = None
        self.best_val_acc: float = 0.0
        self.best_train_acc: float = 0.0
        self.best_val_loss: float = float("inf")
        self.best_train_loss: float = float("inf")
        self.checkpoints: List[str] = []
        self.scaler: Optional[torch.amp.GradScaler] = None
        self.model_name: str = ""
        self.csv_path: Optional[Path] = None
        self.charts_dir: Optional[Path] = None

        if is_distributed:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.logger = get_logger("trainer") if self.is_main else None

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def _log(self, msg: str) -> None:
        if self.is_main and self.logger:
            self.logger.info(msg)

    def setup(self, config: TrainConfig) -> None:
        self.config = config
        self._build_model()
        self._build_dataloaders()
        self._build_optimizer()
        self._build_scheduler()
        if config.training.use_class_weights:
            class_weights = (
                self.train_loader.dataset.get_class_weights().to(self.device)
            )
            self._log(
                f"Using weighted CrossEntropyLoss (class_weights min={class_weights.min():.3f} max={class_weights.max():.3f})")
        else:
            class_weights = None
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.training.label_smoothing,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.device.type == "cuda"
        )

        if self.is_main:
            Path(config.output.checkpoint_dir).mkdir(
                parents=True, exist_ok=True
            )
            self.writer = SummaryWriter(log_dir=config.output.log_dir)
            self.logger = get_logger(
                "trainer", log_dir=config.output.log_dir
            )
            name = self.model_name or config.model.backbone
            charts_base = Path("charts") / name
            charts_base.mkdir(parents=True, exist_ok=True)
            exp_id = sum(1 for p in charts_base.iterdir() if p.is_dir())
            self.charts_dir = charts_base / str(exp_id)
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            self.csv_path = self.charts_dir / "training.csv"
            pd.DataFrame(columns=[
                "epoch", "train_loss", "train_acc",
                "val_loss", "val_acc", "lr",
            ]).to_csv(self.csv_path, index=False)

        self._log(f"Device: {self.device}  |  World size: {self.world_size}")
        self._log(f"Backbone: {config.model.backbone}")
        if config.model.lora.enabled:
            trainable, total = count_parameters(self._raw_model())
            self._log(
                f"LoRA enabled  rank={config.model.lora.rank}  "
                f"alpha={config.model.lora.alpha}  "
                f"target={config.model.lora.target_modules}  |  "
                f"trainable={trainable:,} / {total:,} params "
                f"({100 * trainable / total:.2f}%)"
            )
        self._log(f"Train samples: {len(self.train_loader.dataset)}")
        self._log(f"Val samples:   {len(self.val_loader.dataset)}")

        train_counts = self.train_loader.dataset.get_class_counts()
        val_counts = self.val_loader.dataset.get_class_counts()
        self._log("Per-class sample counts (class_id: train / val):")
        for cls_id, (tc, vc) in enumerate(
                zip(train_counts.tolist(), val_counts.tolist())):
            self._log(f"  class {cls_id:>3d}: {tc:>5d} train / {vc:>4d} val")
        log_model_size(self._raw_model(), self.logger)

    def _build_dataloaders(self) -> None:
        cfg = self.config
        train_tf = get_train_transforms(model=self._raw_model())
        val_tf = get_val_transforms(
            resize_size=cfg.data.resize_size, crop_size=cfg.data.crop_size
        )

        train_ds = ClassificationDataset(
            cfg.data.train_dir, transform=train_tf
        )
        val_ds = ClassificationDataset(
            cfg.data.val_dir, transform=val_tf
        )

        if self.is_distributed:
            if cfg.training.use_weighted_sampler:
                self._log(
                    "use_weighted_sampler=True is ignored in DDP mode "
                    "(DistributedSampler is used instead); "
                    "class imbalance is handled via use_class_weights "
                    "in the loss.")
            self.train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            train_shuffle = False
        elif cfg.training.use_weighted_sampler:
            sample_weights = train_ds.get_sample_weights()
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_shuffle = False
        else:
            self.train_sampler = None
            train_shuffle = True

        self.train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=train_shuffle,
            sampler=self.train_sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )

    def _build_model(self) -> None:
        base = ModelFactory.create(self.config.model).to(self.device)

        if self.is_distributed:
            self.model = DDP(
                base,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        else:
            self.model = base

    def _raw_model(self) -> nn.Module:
        """Unwrap DDP to get the underlying module."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _build_optimizer(self) -> None:
        cfg = self.config.training
        if self.config.model.lora.enabled:
            # Only optimize the unfrozen LoRA A/B matrices.
            params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            params = list(self.model.parameters())
        if cfg.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=cfg.lr,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:  # adamw (default)
            self.optimizer = torch.optim.AdamW(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )

    def _build_scheduler(self) -> None:
        cfg = self.config.training
        total_steps = cfg.epochs * len(self.train_loader)
        warmup_steps = cfg.warmup_epochs * len(self.train_loader)

        if cfg.scheduler == "cosine":
            min_lr_ratio = cfg.min_lr / cfg.lr

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return 1e-4 + (1.0 - 1e-4) * step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(
                    1, total_steps - warmup_steps
                )
                return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                    1.0 + math.cos(math.pi * progress)
                )

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
        elif cfg.scheduler == "step":
            if warmup_steps > 0:
                def lr_lambda(step: int) -> float:  # noqa: F811
                    if step < warmup_steps:
                        return 1e-4 + (1.0 - 1e-4) * step / max(
                            1, warmup_steps
                        )
                    decay_steps = (step - warmup_steps) // 10
                    return 0.1**decay_steps

                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=10, gamma=0.1
                )
        else:  # plateau
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5
            )

    def _freeze_bn(self) -> None:
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _train_one_epoch(self, epoch: int) -> BatchResult:
        self.model.train()
        if self.config.training.freeze_batch_normalize:
            self._freeze_bn()
        if self.train_sampler is not None and hasattr(
                self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

        cfg = self.config
        total_loss, correct, total = 0.0, 0, 0
        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [train]",
            leave=False,
            disable=not self.is_main,
            colour="green",
        )
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            images, labels_a, labels_b, lam = mixup_data(
                images, labels, cfg.training.mixup_alpha
            )

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda"
            ):
                outputs = self.model(images)
                loss = mixup_criterion(
                    self.criterion, outputs, labels_a, labels_b, lam
                )
                loss = loss / cfg.training.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % cfg.training.accumulation_steps == 0:
                if cfg.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        cfg.training.gradient_clip,
                    )
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if not isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ) and self.scaler.get_scale() == scale_before:
                    self.scheduler.step()

            total_loss += loss.item() * cfg.training.accumulation_steps
            preds = outputs.argmax(dim=1)
            mixed = (
                lam * (preds == labels_a).float()
                + (1 - lam) * (preds == labels_b).float()
            )
            correct += mixed.sum().item()
            total += labels.size(0)
            if self.is_main:
                pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

        # Aggregate metrics across all GPUs
        if self.is_distributed:
            metrics = torch.tensor(
                [total_loss, correct, total],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss = metrics[0].item() / self.world_size  # average loss
            correct = int(metrics[1].item())
            total = int(metrics[2].item())

        return BatchResult(
            loss=total_loss / len(self.train_loader),
            correct=correct,
            total=total,
        )

    @torch.no_grad()
    def _validate(self) -> BatchResult:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        if self.is_distributed:
            val_sampler = DistributedSampler(
                self.val_loader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            val_loader = DataLoader(
                self.val_loader.dataset,
                batch_size=self.val_loader.batch_size,
                sampler=val_sampler,
                num_workers=self.val_loader.num_workers,
                pin_memory=True,
            )
        else:
            val_loader = self.val_loader

        for images, labels in tqdm(
            val_loader,
            desc="Validation",
            leave=False,
            disable=not self.is_main,
        ):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda"
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        if self.is_distributed:
            metrics = torch.tensor(
                [total_loss, correct, total],
                dtype=torch.float64,
                device=self.device,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss = metrics[0].item() / self.world_size
            correct = int(metrics[1].item())
            total = int(metrics[2].item())

        return BatchResult(
            loss=total_loss / len(val_loader),
            correct=correct,
            total=total,
        )

    def train_epoch(self, epoch: int) -> EpochResult:
        timer = Timer().start()
        train_result = self._train_one_epoch(epoch)
        val_result = self._validate()
        elapsed = timer.stop()

        current_lr = self.optimizer.param_groups[0]["lr"]
        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            self.scheduler.step(val_result.accuracy)

        result = EpochResult(
            epoch=epoch,
            train_loss=train_result.loss,
            train_acc=train_result.accuracy,
            val_loss=val_result.loss,
            val_acc=val_result.accuracy,
            lr=current_lr,
        )
        self._log(
            f"Epoch {epoch:03d} | "
            f"train_loss={result.train_loss:.4f} "
            f"train_acc={result.train_acc:.4f} | "
            f"val_loss={result.val_loss:.4f} "
            f"val_acc={result.val_acc:.4f} | "
            f"lr={result.lr:.2e} | time={elapsed:.1f}s"
        )
        if self.is_main and self.writer:
            self.writer.add_scalars(
                "loss",
                {"train": result.train_loss, "val": result.val_loss},
                epoch,
            )
            self.writer.add_scalars(
                "acc",
                {"train": result.train_acc, "val": result.val_acc},
                epoch,
            )
            self.writer.add_scalar("lr", result.lr, epoch)

        if self.is_main and self.csv_path:
            pd.DataFrame([{
                "epoch": epoch,
                "train_loss": round(result.train_loss, 6),
                "train_acc": round(result.train_acc, 6),
                "val_loss": round(result.val_loss, 6),
                "val_acc": round(result.val_acc, 6),
                "lr": round(result.lr, 8),
            }]).to_csv(self.csv_path, mode="a", index=False, header=False)

        if self.is_main and self.charts_dir:
            plot_training_curves(
                str(self.csv_path), save_dir=str(self.charts_dir)
            )
            class_names = (
                self.val_loader.dataset.classes
                if hasattr(self.val_loader.dataset, "classes") else None
            )
            plot_confusion_matrix(
                self._raw_model(), self.val_loader, self.device,
                class_names=class_names, save_dir=str(self.charts_dir),
            )
            plot_roc_curves(
                self._raw_model(), self.val_loader, self.device,
                num_classes=self.config.model.num_classes,
                class_names=class_names, save_dir=str(self.charts_dir),
            )

        return result

    def train(self) -> None:
        cfg = self.config
        self._log(f"Starting training for {cfg.training.epochs} epochs")

        for epoch in range(1, cfg.training.epochs + 1):
            result = self.train_epoch(epoch)
            is_best = result.val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = result.val_acc
                self.best_train_acc = result.train_acc
                self.best_val_loss = result.val_loss
                self.best_train_loss = result.train_loss
            self.save_checkpoint(epoch, result.val_acc, is_best)
            if self.is_distributed:
                dist.barrier()  # keep all ranks in sync after checkpoint

        self._log(
            f"Training complete. Best val acc: {self.best_val_acc:.4f}"
        )
        if self.is_main and self.writer:
            self.writer.close()

        if self.is_main:
            cfg = self.config
            model_size = sum(p.numel() for p in self._raw_model().parameters())
            name = self.model_name or cfg.model.backbone
            log_performance_csv(
                model_name=name,
                model_size=model_size,
                epochs=cfg.training.epochs,
                lr=cfg.training.lr,
                optimizer=cfg.training.optimizer,
                image_size=cfg.data.crop_size,
                train_acc=self.best_train_acc,
                val_acc=self.best_val_acc,
                train_loss=self.best_train_loss,
                val_loss=self.best_val_loss,
            )
            self._log(
                f"Performance logged to runs/{name}/performance.csv "
                f"(val_acc={self.best_val_acc:.4f})"
            )

    def save_checkpoint(
        self, epoch: int, val_acc: float, is_best: bool
    ) -> None:
        if not self.is_main:
            return
        cfg = self.config
        ckpt_dir = Path(cfg.output.checkpoint_dir)
        lora_cfg = cfg.model.lora
        if lora_cfg.enabled and lora_cfg.save_lora_only:
            model_state = get_lora_state_dict(self._raw_model())
        else:
            model_state = self._raw_model().state_dict()
        state = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "backbone": cfg.model.backbone,
            "num_classes": cfg.model.num_classes,
            "lora": lora_cfg.enabled,
        }
        ckpt_path = str(
            ckpt_dir / f"epoch_{epoch:03d}_acc{val_acc:.4f}.pth"
        )
        torch.save(state, ckpt_path)
        self.checkpoints.append(ckpt_path)

        if is_best:
            torch.save(state, str(ckpt_dir / "best_model.pth"))
            self._log(f"New best model saved: val_acc={val_acc:.4f}")

        if len(self.checkpoints) > cfg.output.save_top_k:
            old = self.checkpoints.pop(0)
            if os.path.exists(old):
                os.remove(old)

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self._raw_model().load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.best_val_acc = state.get("val_acc", 0.0)
        epoch = state.get("epoch", 0)
        self._log(
            f"Resumed from {path} "
            f"(epoch {epoch}, val_acc={self.best_val_acc:.4f})"
        )
        return epoch
