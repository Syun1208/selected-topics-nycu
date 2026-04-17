import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.layers.nms import batched_nms
from detectron2.utils.file_io import PathManager

from src.interfaces.base_tester import BaseTester

logger = logging.getLogger(__name__)


class DetrexTester(BaseTester):

    def __init__(self) -> None:
        self._cfg = None
        self._model = None

    def setup(self, cfg) -> None:
        self._cfg = cfg
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)

        ema_kwargs = {}
        try:
            from detrex.modeling import ema

            ema.may_build_model_ema(cfg, model)
            ema_kwargs = ema.may_get_ema_checkpointer(cfg, model)
        except Exception:
            pass

        checkpointer = DetectionCheckpointer(model, **ema_kwargs)
        checkpointer.load(cfg.train.init_checkpoint)
        logger.info("Loaded checkpoint: %s", cfg.train.init_checkpoint)

        try:
            if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
                from detrex.modeling import ema as _ema

                _ema.apply_model_ema(model)
        except Exception:
            pass

        model.eval()
        self._model = model

    def evaluate(self) -> Dict[str, Any]:
        cfg = self._cfg
        logger.info("Running evaluation on validation set …")
        ret = inference_on_dataset(
            self._model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        print_csv_format(ret)
        return ret

    def predict(self, output_dir: str) -> None:
        PathManager.mkdirs(output_dir)
        predictions = self._run_inference()
        pred_file = os.path.join(output_dir, "predictions.json")
        self._write_json(predictions, pred_file)
        logger.info("Saved %d predictions to '%s'.", len(predictions), pred_file)

    def submission(self, output_dir: str) -> None:
        PathManager.mkdirs(output_dir)
        predictions = self._run_inference()
        submission_file = os.path.join(output_dir, "submission.json")
        self._write_json(predictions, submission_file)
        logger.info("Saved %d predictions to '%s'.", len(predictions), submission_file)

    def save_results(self, results: Dict[str, Any], output_dir: str) -> None:
        PathManager.mkdirs(output_dir)
        out_file = os.path.join(output_dir, "eval_results.json")
        self._write_json(results, out_file)
        logger.info("Saved evaluation results to '%s'.", out_file)






    @staticmethod
    def _soft_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
        sigma: float = 0.5,
        min_score: float = 1e-4,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=boxes.device)


        max_coord = boxes.max() + 1.0
        offsets = labels.float() * (max_coord + 1.0)
        shifted = boxes + offsets.unsqueeze(1)

        scores = scores.clone().float()
        x1, y1, x2, y2 = shifted.unbind(1)
        areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        active = torch.ones(len(scores), dtype=torch.bool, device=boxes.device)
        keep: List[int] = []

        for _ in range(len(scores)):

            tmp = scores.clone()
            tmp[~active] = -1.0
            i = int(tmp.argmax())
            if scores[i] < min_score:
                break
            keep.append(i)
            active[i] = False

            rest = active.nonzero(as_tuple=False).squeeze(1)
            if rest.numel() == 0:
                break


            ix1 = torch.max(shifted[i, 0], shifted[rest, 0])
            iy1 = torch.max(shifted[i, 1], shifted[rest, 1])
            ix2 = torch.min(shifted[i, 2], shifted[rest, 2])
            iy2 = torch.min(shifted[i, 3], shifted[rest, 3])
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)


            scores[rest] *= torch.exp(-(iou ** 2) / sigma)
            active[rest] &= scores[rest] > min_score

        return torch.tensor(keep, dtype=torch.long, device=boxes.device)




    @staticmethod
    def _post_process_image(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        pp: Dict,
    ):
        nms_type  = pp.get("nms_type", "none")
        pp_iou    = float(pp.get("nms_iou_threshold", 0.3))
        sigma     = float(pp.get("soft_sigma", 0.5))
        min_size  = float(pp.get("min_box_size", 0))
        max_det   = int(pp.get("max_det", 300))

        if boxes.numel() == 0:
            return boxes, scores, classes


        if min_size > 0:
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            mask = torch.max(w, h) >= min_size
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

        if boxes.numel() == 0:
            return boxes, scores, classes


        if nms_type == "soft":
            keep = DetrexTester._soft_nms(boxes, scores, classes, sigma=sigma)
        elif nms_type == "hard":
            keep = batched_nms(boxes, scores, classes, pp_iou)
        else:

            keep = scores.argsort(descending=True)


        keep = keep[:max_det]
        return boxes[keep], scores[keep], classes[keep]




    def _run_inference(
        self,
        score_threshold: float = 0.05,
        post_process: Optional[Dict] = None,
    ) -> List[Dict]:
        cfg = self._cfg
        test_loader = instantiate(cfg.dataloader.test)
        self._model.eval()
        predictions: List[Dict] = []
        pp = post_process or {}

        if pp:
            logger.info(
                "Post-processing: nms_type=%s  iou=%.2f  sigma=%.2f  "
                "min_box_size=%s  max_det=%s",
                pp.get("nms_type", "none"),
                float(pp.get("nms_iou_threshold", 0.3)),
                float(pp.get("soft_sigma", 0.5)),
                pp.get("min_box_size", 0),
                pp.get("max_det", 300),
            )

        total = len(test_loader)
        raw_total = 0
        kept_total = 0

        with torch.no_grad():
            for batch in tqdm(
                test_loader, total=total, desc="Inference", unit="batch", colour="green"
            ):
                outputs = self._model(batch)
                for inp, out in zip(batch, outputs):
                    instances = out.get("instances")
                    if instances is None:
                        continue

                    image_id = inp.get("image_id", inp.get("file_name", "unknown"))
                    boxes   = instances.pred_boxes.tensor.cpu()
                    scores  = instances.scores.cpu()
                    classes = instances.pred_classes.cpu()
                    raw_total += len(scores)


                    boxes, scores, classes = self._post_process_image(
                        boxes, scores, classes, pp
                    )
                    kept_total += len(scores)

                    for box, score, cls in zip(
                        boxes.tolist(), scores.tolist(), classes.tolist()
                    ):
                        if score < score_threshold:
                            continue
                        x1, y1, x2, y2 = box
                        predictions.append(
                            {
                                "image_id": image_id,
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": score,
                                "category_id": cls + 1,
                            }
                        )

        logger.info(
            "Post-processing: %d raw → %d kept → %d above score_threshold=%.3f",
            raw_total, kept_total, len(predictions), score_threshold,
        )
        return predictions

    @staticmethod
    def _write_json(data: Any, path: str) -> None:
        with open(path, "w") as f:
            json.dump(data, f)
