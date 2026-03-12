import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataclass import Prediction, TestConfig
from data.dataset import TestDataset, get_val_transforms
from models.model_factory import ModelFactory
from services.interface.test import TesterInterface
from utils.logger import get_logger


class Tester(TesterInterface):
    def __init__(self) -> None:
        self.config: Optional[TestConfig] = None
        self.model: Optional[torch.nn.Module] = None
        self.test_loader: Optional[DataLoader] = None
        self.device = torch.device(
            f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")
        self.logger = get_logger("tester")

    def setup(self, config: TestConfig) -> None:
        self.config = config
        self._build_dataloader()
        self._load_model()
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")

    def _build_dataloader(self) -> None:
        cfg = self.config
        transform = get_val_transforms(
            resize_size=cfg.data.resize_size, crop_size=cfg.data.crop_size
        )
        test_ds = TestDataset(cfg.data.test_dir, transform=transform)
        self.test_loader = DataLoader(
            test_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )

    def _load_model(self) -> None:
        cfg = self.config.model
        model_cls = ModelFactory.create(cfg=cfg)
        backbone_name = (
            cfg.backbone[len("resnet_nn:"):]
            if cfg.backbone.startswith("resnet_nn:")
            else cfg.backbone
        )
        self.model = model_cls.from_checkpoint(
            checkpoint_path=cfg.checkpoint,
            backbone=backbone_name,
            num_classes=cfg.num_classes,
        ).to(self.device)
        self.model.eval()
        self.logger.info(f"Loaded checkpoint: {cfg.checkpoint}")

    @torch.no_grad()
    def predict(self) -> List[Prediction]:
        predictions: List[Prediction] = []
        for images, image_names in tqdm(self.test_loader, desc="Inference"):
            images = images.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda"
            ):
                outputs = self.model(images)
            pred_labels = outputs.argmax(dim=1).cpu().tolist()
            for name, label in zip(image_names, pred_labels):
                predictions.append(Prediction(image_name=name, label=label))
        self.logger.info(f"Predicted {len(predictions)} images")
        return predictions

    def save_submission(
        self, predictions: List[Prediction], output_path: str
    ) -> None:
        df = pd.DataFrame(
            [
                {"image_name": p.image_name, "pred_label": p.label}
                for p in predictions
            ]
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Submission saved to {output_path} ({len(df)} rows)")
