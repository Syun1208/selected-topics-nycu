from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class BaseTester(ABC):

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name: str = cfg["model"]["name"]
        self.backbone: str = cfg["model"]["backbone"]
        self.exp_name: str = cfg.get("experiment", {}).get("name", "v1")

        self.chart_dir = Path(cfg["paths"]["chart_dir"])
        self.chart_dir.mkdir(parents=True, exist_ok=True)


        base_dir = Path(cfg["paths"].get("base_dir", "."))
        self.submission_dir = (
            base_dir / "submissions" / self.model_name / self.backbone / self.exp_name
        )
        self.submission_dir.mkdir(parents=True, exist_ok=True)

        self.device: str = cfg.get("inference", {}).get("device", "cuda")
        self.score_thr: float = cfg.get("inference", {}).get("score_thr", 0.05)
        self.nms_iou_thr: float = cfg.get("inference", {}).get("nms_iou_thr", 0.5)


    @abstractmethod
    def build_model(self) -> Any: ...

    @abstractmethod
    def build_dataloader(self) -> Any: ...

    @abstractmethod
    def predict_batch(self, batch: dict) -> List[dict]: ...

    @abstractmethod
    def format_coco_output(self, predictions: List[dict]) -> List[dict]: ...


    def run(self, output_path: str = None) -> List[dict]:
        import json
        import logging
        logger = logging.getLogger(self.__class__.__name__)

        self.build_model()
        loader = self.build_dataloader()

        all_preds = []
        for batch_idx, batch in enumerate(loader):
            preds = self.predict_batch(batch)
            all_preds.extend(preds)
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(loader)} batches")

        coco_results = self.format_coco_output(all_preds)


        primary_path = self.submission_dir / "test-results.json"
        with open(primary_path, "w") as f:
            json.dump(coco_results, f, indent=2)
        logger.info(f"Saved {len(coco_results)} predictions → {primary_path}")


        if output_path and str(output_path) != str(primary_path):
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(coco_results, f, indent=2)
            logger.info(f"Also saved copy → {output_path}")

        return coco_results
