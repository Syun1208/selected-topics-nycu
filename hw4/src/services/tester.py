import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.interface.base_tester import BaseTester
from src.data.dataset import TestSpecificDataset, resolve_test_path
from src.models.promptir import build_promptir
from src.models.lightning_module import PromptIRLightningModule
from src.utils.tta import predict

logger = logging.getLogger(__name__)


class PromptIRTester(BaseTester):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model: Optional[torch.nn.Module] = None
        self.test_loader: Optional[DataLoader] = None
        self.device_obj: torch.device = torch.device("cpu")

    def build_model(self) -> torch.nn.Module:
        ckpt_path = self.cfg["paths"].get("checkpoint_path", "")
        if not ckpt_path or not os.path.exists(ckpt_path):
            best = os.path.join(self.cfg["paths"].get("checkpoint_dir", ""), "best.ckpt")
            if os.path.exists(best):
                ckpt_path = best
            else:
                raise FileNotFoundError(
                    "Checkpoint not found. Set paths.checkpoint_path in config "
                    "or pass --checkpoint."
                )
        ckpt_path = os.path.expanduser(ckpt_path)

        gpu_ids = self.cfg.get("inference", {}).get("gpu_ids", "0")
        gpu_ids = self._parse_gpu_ids(gpu_ids)
        if torch.cuda.is_available() and gpu_ids:
            torch.cuda.set_device(gpu_ids[0])
            self.device_obj = torch.device(f"cuda:{gpu_ids[0]}")
            if len(gpu_ids) > 1:
                logger.info(
                    f"batch_size=1 inference runs on a single GPU; using cuda:{gpu_ids[0]}"
                )
        else:
            self.device_obj = torch.device("cpu")
        logger.info(f"Using device: {self.device_obj}")

        if not self.use_ema:
            logger.info(f"Loading model from {ckpt_path} (live weights)")
            wrapper = PromptIRLightningModule.load_from_checkpoint(ckpt_path).to(self.device_obj)
            wrapper.eval()
            self.model = wrapper
            return self.model

        logger.info(f"Loading model from {ckpt_path} (EMA weights)")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "ema_state_dict" not in ckpt:
            logger.warning(
                "'ema_state_dict' not found in checkpoint; falling back to live weights."
            )
            wrapper = PromptIRLightningModule.load_from_checkpoint(ckpt_path).to(self.device_obj)
            wrapper.eval()
            self.model = wrapper
            return self.model

        net = build_promptir(self.cfg.get("model", {}))
        missing, unexpected = net.load_state_dict(ckpt["ema_state_dict"], strict=False)
        if missing:
            logger.warning(f"Missing EMA keys: {len(missing)} (showing up to 5) {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected EMA keys: {len(unexpected)} (showing up to 5) {unexpected[:5]}")
        net.to(self.device_obj).eval()
        self.model = net
        return self.model

    def build_dataloader(self) -> DataLoader:
        data_cfg = self.cfg.get("data", {})
        test_path = resolve_test_path(data_cfg.get("test_path", "data/hw4_realse_dataset/test"))
        logger.info(f"Reading test images from {test_path}")

        ds_args = SimpleNamespace(test_path=test_path)
        dataset = TestSpecificDataset(ds_args)
        self.test_loader = DataLoader(
            dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
        )
        logger.info(f"Test set: {len(dataset)} images")
        return self.test_loader

    def predict_batch(self, batch) -> Dict[str, Any]:
        ([clean_name], degrad_patch) = batch
        degrad_patch = degrad_patch.to(self.device_obj)

        with torch.no_grad():
            restored = predict(
                self.model,
                degrad_patch,
                self_ensemble=self.self_ensemble,
                tile=self.tile,
                tile_size=self.tile_size,
                tile_overlap=self.tile_overlap,
            )

        restored = torch.clamp(restored, 0, 1)
        img = restored[0].cpu().numpy()
        img = np.round(img * 255.0).astype(np.uint8)
        return {"file_name": clean_name[0] + ".png", "image": img}

    def format_output(self, predictions: List[dict]) -> Dict[str, Any]:
        images_dict: Dict[str, np.ndarray] = {}
        for pred in predictions:
            images_dict[pred["file_name"]] = pred["image"]
        return images_dict

    def run(self, output_path: str = None) -> Dict[str, np.ndarray]:
        self.build_model()
        loader = self.build_dataloader()

        all_preds: List[dict] = []
        logger.info(
            f"Start testing... (self_ensemble={self.self_ensemble}, "
            f"tile={self.tile}, use_ema={self.use_ema})"
        )
        for batch in tqdm(loader, colour="green", desc="Inference"):
            all_preds.append(self.predict_batch(batch))

        outputs = self.format_output(all_preds)

        primary_path = self.submission_dir / "pred.npz"
        self._save_npz(outputs, str(primary_path))
        logger.info(f"Saved {len(outputs)} restored images -> {primary_path}")

        if output_path and str(output_path) != str(primary_path):
            self._save_npz(outputs, output_path)
            logger.info(f"Also saved copy -> {output_path}")

        sample_key = next(iter(outputs))
        sample = outputs[sample_key]
        logger.info(
            f"Sample: key='{sample_key}', shape={sample.shape}, "
            f"dtype={sample.dtype}, range=[{sample.min()}, {sample.max()}]"
        )
        return outputs

    @staticmethod
    def _parse_gpu_ids(value) -> List[int]:
        if isinstance(value, list):
            return [int(v) for v in value if int(v) >= 0]
        ids = [int(v) for v in str(value).split(",") if v.strip() != ""]
        return [i for i in ids if i >= 0]
