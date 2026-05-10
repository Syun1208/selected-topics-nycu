from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseTrainer(ABC):

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name: str = cfg["model"]["name"]
        self.backbone: str = cfg["model"]["backbone"]
        self.exp_name: str = cfg.get("experiment", {}).get("name", "v1")

        self.checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.chart_dir = Path(cfg["paths"]["chart_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.chart_dir.mkdir(parents=True, exist_ok=True)

        self.device: str = cfg.get("training", {}).get("device", "cuda")
        self.max_epochs: int = cfg.get("training", {}).get("max_epochs", 12)
        self.best_metric: float = 0.0
        self.current_epoch: int = 0


        self.train_loss_history: list = []
        self.val_loss_history: list = []
        self.val_map_history: list = []


    @abstractmethod
    def build_model(self) -> Any: ...

    @abstractmethod
    def build_dataloaders(self) -> dict: ...

    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]: ...

    @abstractmethod
    def validate(self, epoch: int) -> Dict[str, float]: ...

    @abstractmethod
    def save_checkpoint(self, path: str, is_best: bool = False) -> None: ...

    @abstractmethod
    def load_checkpoint(self, path: str) -> None: ...

    def run(self) -> None:
        import logging
        from tqdm import tqdm
        logger = logging.getLogger(self.__class__.__name__)

        self.build_model()
        self.build_dataloaders()
        self.on_train_start()

        epoch_bar = tqdm(
            range(self.current_epoch, self.max_epochs),
            desc="Epochs", unit="ep",
            initial=self.current_epoch, total=self.max_epochs,
            dynamic_ncols=True, colour="green",
        )
        for epoch in epoch_bar:
            self.current_epoch = epoch

            train_metrics = self.train_epoch(epoch)
            self.train_loss_history.append(train_metrics.get("loss", 0.0))

            val_metrics = self.validate(epoch)
            mask_ap = val_metrics.get("mask_ap", 0.0)
            self.val_map_history.append(mask_ap)

            epoch_bar.set_postfix(
                loss=f"{train_metrics.get('loss', 0.0):.4f}",
                mAP=f"{mask_ap:.4f}",
                best=f"{self.best_metric:.4f}",
            )
            logger.info(
                f"Epoch [{epoch + 1}/{self.max_epochs}] "
                f"loss={train_metrics.get('loss', 0.0):.4f} "
                f"mAP={mask_ap:.4f}"
            )

            last_ckpt = str(self.checkpoint_dir / "last.pth")
            self.save_checkpoint(last_ckpt, is_best=False)

            if mask_ap > self.best_metric:
                self.best_metric = mask_ap
                best_ckpt = str(self.checkpoint_dir / "best.pth")
                self.save_checkpoint(best_ckpt, is_best=True)
                logger.info(f"  New best mAP={mask_ap:.4f} -> best.pth")

            self.on_epoch_end(epoch, train_metrics, val_metrics)

        self.on_train_end()

    def on_train_start(self) -> None: pass

    def on_epoch_end(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None: pass

    def on_train_end(self) -> None: pass
