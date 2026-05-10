import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.interface.base_tester import BaseTester
from src.data.loaders.dataset import CellTestDataset
from src.utils.io_utils import encode_mask_rle, load_checkpoint
from src.services.trainer import _build_mmdet_model

logger = logging.getLogger(__name__)


class MMDetTester(BaseTester):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model: Optional[Any] = None
        self.test_loader: Optional[DataLoader] = None


    def build_model(self) -> Any:
        model_name = self.cfg["model"]["name"].lower()
        if model_name == "cascade_mask_rcnn":
            from src.models.cascade_mask_rcnn import build_cascade_mask_rcnn
            model_cfg = build_cascade_mask_rcnn(self.cfg["model"])
        elif model_name == "htc":
            from src.models.htc import build_htc
            model_cfg = build_htc(self.cfg["model"])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model = _build_mmdet_model(model_cfg, self.device)

        ckpt_path = self.cfg["paths"].get("checkpoint_path", "")
        if not ckpt_path or not os.path.exists(ckpt_path):

            ckpt_dir = self.cfg["paths"].get("checkpoint_dir", "")
            best = os.path.join(ckpt_dir, "best.pth")
            if os.path.exists(best):
                ckpt_path = best
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found. Set paths.checkpoint_path in config."
                )

        ckpt = load_checkpoint(ckpt_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        logger.info(f"Loaded weights from {ckpt_path}")
        return self.model


    def build_dataloader(self) -> DataLoader:
        data_cfg = self.cfg.get("data", {})
        test_dir = data_cfg.get("test_dir", "data/test_release")
        mapping_json = data_cfg.get("mapping_json", "data/test_image_name_to_ids.json")
        img_size = int(data_cfg.get("img_size", 1024))
        num_workers = int(data_cfg.get("num_workers", 4))

        dataset = CellTestDataset(test_dir, mapping_json, img_size=img_size)
        self.test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda x: x,
        )
        logger.info(f"Test set: {len(dataset)} images")
        return self.test_loader


    _MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    _STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

    def predict_batch(self, batch: list) -> List[dict]:
        from mmdet.structures import DetDataSample

        results = []
        for item in batch:
            image_id = item["image_id"]
            img = item["img"]

            # Normalize to match training preprocessing
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np - self._MEAN) / self._STD
            img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

            h, w = img.shape[-2:]
            raw_metas = dict(item["img_metas"])
            raw_metas.setdefault("batch_input_shape", (h, w))
            raw_metas.setdefault("img_shape", (h, w))
            raw_metas.setdefault("ori_shape", (h, w))

            sf = raw_metas.get("scale_factor")
            if sf is not None:
                sf = np.array(sf).reshape(-1)
                raw_metas["scale_factor"] = sf[:2].tolist() if len(sf) >= 2 else float(sf[0])

            data_sample = DetDataSample()
            data_sample.set_metainfo(raw_metas)

            with torch.no_grad():
                preds = self.model(img, [data_sample], mode="predict")

            pred = preds[0].pred_instances
            keep = (pred.scores >= self.score_thr).nonzero(as_tuple=False).squeeze(1).cpu().numpy()

            bboxes, scores, labels, masks = [], [], [], []
            for i in keep:
                x1, y1, x2, y2 = pred.bboxes[i].cpu().tolist()
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(pred.scores[i].cpu()))
                labels.append(int(pred.labels[i].cpu()))

            if hasattr(pred, "masks") and len(keep) > 0:
                try:
                    raw_masks = pred.masks[keep]
                    if hasattr(raw_masks, "masks"):
                        masks = [raw_masks.masks[j] for j in range(len(raw_masks))]
                    else:
                        masks = raw_masks.cpu().numpy().tolist()
                except Exception:
                    pass

            results.append({
                "image_id": image_id,
                "bboxes": bboxes,
                "scores": scores,
                "labels": labels,
                "masks": masks,
            })
        return results


    def format_coco_output(self, predictions: List[dict]) -> List[dict]:
        coco_results = []
        for pred in predictions:
            image_id = int(pred["image_id"])
            masks = pred.get("masks", [])

            for i, (bbox, score, label) in enumerate(
                zip(pred["bboxes"], pred["scores"], pred["labels"])
            ):
                mask = masks[i] if i < len(masks) else None


                segmentation = None
                if mask is not None:
                    try:
                        m = np.asarray(mask, dtype=np.uint8)
                        rle = encode_mask_rle(m)

                        segmentation = {
                            "size": list(rle["size"]),
                            "counts": rle["counts"] if isinstance(rle["counts"], str)
                                      else rle["counts"].decode("utf-8"),
                        }
                    except Exception as e:
                        logger.debug(f"RLE encode failed for image {image_id}: {e}")

                entry = {
                    "image_id": image_id,
                    "bbox": [round(float(x), 10) for x in bbox],
                    "score": round(float(score), 5),
                    "category_id": int(label) + 1,
                    "segmentation": segmentation,
                }
                coco_results.append(entry)

        return coco_results
