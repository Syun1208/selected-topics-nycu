import json
import logging
import os
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
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

    def _run_inference(self, score_threshold: float = 0.05) -> List[Dict]:
        cfg = self._cfg
        test_loader = instantiate(cfg.dataloader.test)
        self._model.eval()
        predictions: List[Dict] = []

        total = len(test_loader)
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
                    boxes = instances.pred_boxes.tensor.cpu().tolist()
                    scores = instances.scores.cpu().tolist()
                    classes = instances.pred_classes.cpu().tolist()
                    for box, score, cls in zip(boxes, scores, classes):
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

        return predictions

    @staticmethod
    def _write_json(data: Any, path: str) -> None:
        with open(path, "w") as f:
            json.dump(data, f)
