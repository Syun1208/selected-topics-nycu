from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


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
        self.max_epochs: int = cfg.get("training", {}).get("max_epochs", 150)
        self.best_metric: float = 0.0
        self.current_epoch: int = 0

        self.train_loss_history: List[float] = []
        self.val_psnr_history: List[float] = []
        self.val_ssim_history: List[float] = []

    @abstractmethod
    def build_model(self) -> Any: ...

    @abstractmethod
    def build_dataloaders(self) -> dict: ...

    @abstractmethod
    def build_callbacks(self) -> list: ...

    @abstractmethod
    def build_logger(self) -> Any: ...

    @abstractmethod
    def fit(self) -> Dict[str, Any]: ...

    def run(self) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(self.__class__.__name__)

        self.build_model()
        self.build_dataloaders()
        self.on_train_start()

        result = self.fit()

        self.on_train_end()
        logger.info("Training finished.")
        return result

    def on_train_start(self) -> None:
        pass

    def on_train_end(self) -> None:
        pass
