import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class BaseTester(ABC):

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name: str = cfg["model"]["name"]
        self.backbone: str = cfg["model"]["backbone"]
        self.exp_name: str = cfg.get("experiment", {}).get("name", "v1")

        base_dir = Path(cfg["paths"].get("base_dir", "."))
        self.submission_dir = (
            base_dir / "submissions" / self.model_name / self.backbone / self.exp_name
        )
        self.submission_dir.mkdir(parents=True, exist_ok=True)

        self.device: str = cfg.get("inference", {}).get("device", "cuda")
        self.use_ema: bool = bool(cfg.get("inference", {}).get("use_ema", True))
        self.self_ensemble: bool = bool(cfg.get("inference", {}).get("self_ensemble", True))
        self.tile: bool = bool(cfg.get("inference", {}).get("tile", False))
        self.tile_size: int = int(cfg.get("inference", {}).get("tile_size", 128))
        self.tile_overlap: int = int(cfg.get("inference", {}).get("tile_overlap", 32))

    @abstractmethod
    def build_model(self) -> Any: ...

    @abstractmethod
    def build_dataloader(self) -> Any: ...

    @abstractmethod
    def predict_batch(self, batch: Any) -> Dict[str, Any]: ...

    @abstractmethod
    def format_output(self, predictions: List[dict]) -> Dict[str, Any]: ...

    def run(self, output_path: str = None) -> Dict[str, Any]:
        import logging
        logger = logging.getLogger(self.__class__.__name__)

        self.build_model()
        loader = self.build_dataloader()

        all_preds: List[dict] = []
        for batch in loader:
            preds = self.predict_batch(batch)
            all_preds.append(preds)

        outputs = self.format_output(all_preds)

        primary_path = self.submission_dir / "pred.npz"
        self._save_npz(outputs, str(primary_path))
        logger.info(f"Saved {len(outputs)} restored images -> {primary_path}")

        if output_path and str(output_path) != str(primary_path):
            self._save_npz(outputs, output_path)
            logger.info(f"Also saved copy -> {output_path}")

        return outputs

    @staticmethod
    def _save_npz(images_dict: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez(path, **images_dict)
