import logging
import os
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
    category=UserWarning,
)
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine import SimpleTrainer, hooks
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventWriter,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.file_io import PathManager

from src.interfaces.base_trainer import BaseTrainer
from src.utils.logger import (
    log_config,
    log_eval_metrics,
    log_model,
    log_train_metrics,
    resolve_event_csv_path,
)
from src.utils.visualize import (
    metrics_from_csv,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_pf1_curve,
    plot_rf1_curve,
    plot_training_curves,
    visualize_batch_predictions,
)

logger = logging.getLogger(__name__)


def _derive_ckpt_dir(output_dir: str) -> str:

    parts = os.path.normpath(output_dir).split(os.sep)
    return os.path.join("checkpoints", parts[-3], parts[-2], parts[-1])


def _derive_chart_dir(output_dir: str) -> str:

    parts = os.path.normpath(output_dir).split(os.sep)

    arch = parts[-3] if len(parts) >= 3 else "model"
    backbone = parts[-2] if len(parts) >= 2 else "backbone"
    run = parts[-1] if len(parts) >= 1 else "0"
    return os.path.join("charts", arch, backbone, run)


def _save_checkpoint(
    model: torch.nn.Module,
    path: str,
    iteration: int,
    optimizer: torch.optim.Optimizer | None = None,
    metric: float | None = None,
    best_metric: float | None = None,
    grad_scaler=None,
) -> None:

    unwrapped = model.module if hasattr(model, "module") else model
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": unwrapped.state_dict(),
        "iteration": iteration,
        "optimizer_type": type(optimizer).__name__ if optimizer is not None else None,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if grad_scaler is not None:
        ckpt["grad_scaler"] = grad_scaler.state_dict()
    if metric is not None:
        ckpt["metric"] = metric
    if best_metric is not None:
        ckpt["best_metric"] = best_metric
    torch.save(ckpt, path)
    logger.info(
        "Saved checkpoint → %s  (iter %d, optimizer=%s, metric=%s, best=%s)",
        path,
        iteration,
        ckpt["optimizer_type"] or "N/A",
        f"{metric :.4f}" if metric is not None else "N/A",
        f"{best_metric :.4f}" if best_metric is not None else "N/A",
    )


class _CSVWriter(EventWriter):

    def __init__(self, csv_path: str, iters_per_epoch: int | None):
        self.csv_path = csv_path
        self.iters_per_epoch = iters_per_epoch

    def write(self):
        log_train_metrics(self.csv_path, self.iters_per_epoch)

    def close(self):
        pass


class _AMPTrainer(SimpleTrainer):

    def __init__(self, model, dataloader, optimizer, amp=False, clip_grad_params=None):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "_AMPTrainer does not support multi-device per process."
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        self.amp = amp
        self.clip_grad_params = clip_grad_params
        self.grad_scaler = None

        if amp:
            self.grad_scaler = torch.amp.GradScaler("cuda")

    def run_step(self):
        assert self.model.training, "Model must be in training mode."

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with torch.amp.autocast("cuda", enabled=self.amp):
            loss_dict = self.model(data)
            losses = (
                loss_dict
                if isinstance(loss_dict, torch.Tensor)
                else sum(loss_dict.values())
            )
            if isinstance(loss_dict, torch.Tensor):
                loss_dict = {"total_loss": loss_dict}

        self.optimizer.zero_grad()
        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self._clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self._clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def _clip_grads(self, params):
        params = [p for p in params if p.requires_grad and p.grad is not None]
        if params:
            torch.nn.utils.clip_grad_norm_(params, **self.clip_grad_params)

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


class _VisualizationHook(hooks.HookBase):

    def __init__(
        self,
        chart_dir: str,
        csv_path: str,
        vis_period: int = 500,
        val_loader_cfg=None,
        class_names: list | None = None,
        iters_per_epoch: int | None = None,
    ):
        self._chart_dir = chart_dir
        self._csv_path = csv_path
        self._vis_period = vis_period
        self._val_loader_cfg = val_loader_cfg
        self._class_names = class_names
        self._iters_per_epoch = iters_per_epoch
        self._last_metric: float | None = None

        parts = os.path.normpath(chart_dir).split(os.sep)
        run = parts[-1] if len(parts) >= 1 else "0"
        bbone = parts[-2] if len(parts) >= 2 else "backbone"
        arch = parts[-3] if len(parts) >= 3 else "model"
        self._run_tag = f"{arch }  /  {bbone }  /  run {run }"

        os.makedirs(chart_dir, exist_ok=True)

    def after_step(self):
        cur = self.trainer.iter + 1

        if cur % self._vis_period == 0 and os.path.exists(self._csv_path):
            try:
                self._save_training_curves()
            except Exception as e:
                logger.warning("VisualizationHook: training curves failed: %s", e)

    def after_train(self):
        try:
            if os.path.exists(self._csv_path):
                self._save_training_curves()
        except Exception as e:
            logger.warning("VisualizationHook: final curves failed: %s", e)

    def _save_training_curves(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metrics = metrics_from_csv(
            self._csv_path,
            x_col="epoch" if self._iters_per_epoch else "iter",
        )
        if not metrics:
            return

        cur = getattr(self, "trainer", None)
        iter_ = (self.trainer.iter + 1) if cur is not None else 0
        acc = (
            f"  |  val mAP {self ._last_metric :.4f}"
            if self._last_metric is not None
            else ""
        )
        title = f"Training Curves  |  {self ._run_tag }  |  iter {iter_ :,}{acc }"

        out = os.path.join(self._chart_dir, "training_curves.png")
        fig = plot_training_curves(metrics, suptitle=title, show=False, save_path=out)
        plt.close(fig)
        logger.info("Saved training curves → %s", out)

    @staticmethod
    def _unwrap_coco_evaluator(evaluator):

        try:
            from detectron2.evaluation import DatasetEvaluators

            if isinstance(evaluator, DatasetEvaluators):
                for ev in evaluator._evaluators:
                    if hasattr(ev, "_coco_eval"):
                        return ev
        except Exception:
            pass
        return evaluator

    def save_eval_charts(self, evaluator, metric: float | None = None) -> None:

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self._last_metric = metric

        evaluator = self._unwrap_coco_evaluator(evaluator)

        try:
            coco_eval_obj = None

            if hasattr(evaluator, "_coco_eval"):
                coco_eval_obj = evaluator._coco_eval.get("bbox", None)
        except Exception:
            pass

        if (
            coco_eval_obj is None
            or not hasattr(coco_eval_obj, "eval")
            or not coco_eval_obj.eval
        ):
            logger.warning(
                "VisualizationHook: COCO eval object not available; skipping eval charts."
            )
            return

        try:
            self._save_pr_curves(coco_eval_obj)
        except Exception as e:
            logger.warning("VisualizationHook: PR curves failed: %s", e)

        try:
            self._save_confusion_matrix(evaluator)
        except Exception as e:
            logger.warning("VisualizationHook: confusion matrix failed: %s", e)

    def _save_pr_curves(self, coco_eval_obj) -> None:
        import matplotlib.pyplot as plt

        prec_arr = coco_eval_obj.eval["precision"]
        n_classes = prec_arr.shape[2]
        recall_pts = np.linspace(0.0, 1.0, prec_arr.shape[1])

        names = (
            self._class_names[:n_classes]
            if self._class_names
            else [str(i) for i in range(n_classes)]
        )

        precisions, recalls, f1_scores, ap_scores = {}, {}, {}, {}
        for k, name in enumerate(names):
            p = prec_arr[0, :, k, 0, 2]
            valid = p >= 0
            pv = p[valid]
            rv = recall_pts[valid]
            ap = float(pv.mean()) if pv.size else 0.0
            precisions[name] = pv
            recalls[name] = rv
            # F1 at each recall threshold:
            # rv here is the actual recall threshold (the x-axis sampling point),
            # which equals the true recall by definition in the COCO interpolated
            # PR curve — each index r means "max precision at recall >= r/100",
            # so the recall AT that point IS r/100 (= rv).
            # However rv[0] = 0 so F1[0] = 0 always.  We also expose the
            # per-class best-F1 scalar via ap_scores so callers can surface it.
            f1_curve = 2 * pv * rv / (pv + rv + 1e-8)
            f1_scores[name] = f1_curve
            ap_scores[name] = ap

        cur = getattr(self, "trainer", None)
        iter_ = (self.trainer.iter + 1) if cur is not None else 0
        tag = f"{self ._run_tag }  |  iter {iter_ :,}"

        for fn, x_data, y_data, base_title, fname in [
            (
                plot_pr_curve,
                recalls,
                precisions,
                "Precision-Recall Curve",
                "pr_curve.png",
            ),
            (
                plot_pf1_curve,
                precisions,
                f1_scores,
                "Precision-F1 Curve",
                "pf1_curve.png",
            ),
            (plot_rf1_curve, recalls, f1_scores, "Recall-F1 Curve", "rf1_curve.png"),
        ]:
            out = os.path.join(self._chart_dir, fname)
            fig = fn(
                x_data,
                y_data,
                ap_scores,
                title=f"{base_title }  |  {tag }",
                show=False,
                save_path=out,
            )
            plt.close(fig)
            logger.info("Saved %s → %s", fname, out)

    def _save_confusion_matrix(self, evaluator) -> None:
        import matplotlib.pyplot as plt

        try:
            coco_api = evaluator._coco_api
            predictions = evaluator._predictions
        except AttributeError:
            return

        n_cls = len(self._class_names) if self._class_names else len(coco_api.cats)
        cm = np.zeros((n_cls, n_cls), dtype=np.int64)

        pred_by_img: dict = {}
        for entry in predictions:
            img_id = entry["image_id"]
            pred_by_img.setdefault(img_id, []).extend(entry.get("instances", []))

        def _iou_xyxy(b1, b2):
            xa = max(b1[0], b2[0])
            ya = max(b1[1], b2[1])
            xb = min(b1[2], b2[2])
            yb = min(b1[3], b2[3])
            inter = max(0, xb - xa) * max(0, yb - ya)
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / max(a1 + a2 - inter, 1e-6)

        for img_id, gt_anns in coco_api.imgToAnns.items():
            preds = pred_by_img.get(img_id, [])
            if not preds:
                continue

            preds = [p for p in preds if p.get("score", 1.0) >= 0.3]
            for gt in gt_anns:
                gx, gy, gw, gh = gt["bbox"]
                gt_box = [gx, gy, gx + gw, gy + gh]
                gt_cls = gt["category_id"] - 1
                best_iou, best_cls = 0.0, gt_cls
                for p in preds:
                    px, py, pw, ph = p["bbox"]
                    pb = [px, py, px + pw, py + ph]
                    iou = _iou_xyxy(gt_box, pb)
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = p["category_id"]  # pred_classes are 0-indexed; GT cat_id is 1-indexed so GT uses -1
                if best_iou >= 0.5 and 0 <= gt_cls < n_cls and 0 <= best_cls < n_cls:
                    cm[gt_cls, best_cls] += 1

        names = (
            self._class_names[:n_cls]
            if self._class_names
            else [str(i) for i in range(n_cls)]
        )

        cur = getattr(self, "trainer", None)
        iter_ = (self.trainer.iter + 1) if cur is not None else 0
        cm_title = f"Confusion Matrix  |  {self ._run_tag }  |  iter {iter_ :,}"

        out = os.path.join(self._chart_dir, "confusion_matrix.png")
        fig = plot_confusion_matrix(
            cm, names, normalize=True, title=cm_title, show=False, save_path=out
        )
        plt.close(fig)
        logger.info("Saved confusion_matrix.png → %s", out)

    def save_val_predictions(
        self,
        model,
        val_accuracy: float | None = None,
        score_threshold: float = 0.4,
    ) -> None:
        if self._val_loader_cfg is None:
            return
        try:
            import glob as _glob
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            batch = self._sample_val_batch(n=6)
            if not batch:
                return

            raw_by_id: dict = {}
            try:
                from detectron2.data import DatasetCatalog

                dataset_name = self._val_loader_cfg.dataset.names
                for d in DatasetCatalog.get(dataset_name):
                    raw_by_id[d["image_id"]] = d
            except Exception:
                pass

            was_training = model.training
            model.eval()
            images, gt_boxes_l, gt_labels_l = [], [], []
            pred_boxes_l, pred_labels_l, pred_scores_l, titles = [], [], [], []

            with torch.no_grad():
                for sample in batch:
                    img_t = sample["image"]
                    img_h, img_w = img_t.shape[-2], img_t.shape[-1]
                    img_np = img_t.permute(1, 2, 0).cpu().numpy()
                    img_np = (
                        img_np.astype("uint8")
                        if img_np.max() > 1.5
                        else (img_np * 255).astype("uint8")
                    )

                    orig_h = sample.get("height", img_h)
                    orig_w = sample.get("width", img_w)
                    scale_h = img_h / orig_h
                    scale_w = img_w / orig_w
                    image_id = sample.get("image_id")
                    raw_annots = (
                        raw_by_id.get(image_id, {}).get("annotations", [])
                        if image_id
                        else []
                    )
                    if raw_annots:
                        gt_boxes_l.append(
                            [
                                [
                                    a["bbox"][0] * scale_w,
                                    a["bbox"][1] * scale_h,
                                    (a["bbox"][0] + a["bbox"][2]) * scale_w,
                                    (a["bbox"][1] + a["bbox"][3]) * scale_h,
                                ]
                                for a in raw_annots
                            ]
                        )

                        gt_labels_l.append([a["category_id"] for a in raw_annots])
                    else:
                        gt_boxes_l.append(None)
                        gt_labels_l.append(None)

                    infer_sample = {**sample, "height": img_h, "width": img_w}
                    out_inst = model([infer_sample])[0]["instances"]
                    pred_boxes_l.append(
                        out_inst.pred_boxes.tensor.cpu().tolist()
                        if out_inst.has("pred_boxes")
                        else None
                    )
                    pred_labels_l.append(
                        out_inst.pred_classes.cpu().tolist()
                        if out_inst.has("pred_classes")
                        else None
                    )
                    pred_scores_l.append(
                        out_inst.scores.cpu().tolist()
                        if out_inst.has("scores")
                        else None
                    )
                    images.append(img_np)
                    titles.append(os.path.basename(sample.get("file_name", "")))

            if was_training:
                model.train()

            for old in _glob.glob(
                os.path.join(self._chart_dir, "val_predictions_*.png")
            ):
                try:
                    os.remove(old)
                except OSError:
                    pass

            acc_tag = f"_{val_accuracy :.4f}" if val_accuracy is not None else ""
            out_path = os.path.join(self._chart_dir, f"val_predictions{acc_tag }.png")

            cur = getattr(self, "trainer", None)
            iter_ = (self.trainer.iter + 1) if cur is not None else 0
            acc_s = f"  |  mAP {val_accuracy :.4f}" if val_accuracy is not None else ""
            sup = (
                f"Validation – GT (green) vs Predictions (red)  |  "
                f"{self ._run_tag }  |  iter {iter_ :,}{acc_s }"
            )

            fig = visualize_batch_predictions(
                images=images,
                gt_boxes_list=gt_boxes_l,
                gt_labels_list=gt_labels_l,
                pred_boxes_list=pred_boxes_l,
                pred_labels_list=pred_labels_l,
                pred_scores_list=pred_scores_l,
                class_names=self._class_names,
                score_threshold=score_threshold,
                titles=titles,
                ncols=3,
                figsize_per_cell=(5.5, 4.5),
                suptitle=sup,
                show=False,
                save_path=out_path,
            )
            plt.close(fig)
            logger.info("Saved val predictions → %s", out_path)

        except Exception as e:
            logger.warning("VisualizationHook: val predictions failed: %s", e)

    def _sample_val_batch(self, n: int = 6) -> list:

        if self._val_loader_cfg is None:
            return []
        try:

            from omegaconf import OmegaConf

            loader_cfg = OmegaConf.merge(self._val_loader_cfg, {"num_workers": 0})
            val_loader = instantiate(loader_cfg)
            pool: list = []
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    pool.extend(batch)
                else:
                    pool.append(batch)
                if len(pool) >= n * 4:
                    break
            if not pool:
                return []
            import random as _random

            return _random.sample(pool, min(n, len(pool)))
        except Exception as e:
            logger.warning("VisualizationHook: could not sample val batch: %s", e)
            return []


class DetrexTrainer(BaseTrainer):

    def __init__(self) -> None:
        self._trainer = None
        self._checkpointer = None
        self._cfg = None
        self._resume = False
        self._csv_path: str | None = None
        self._ckpt_dir: str | None = None
        self._chart_dir: str | None = None
        self._iters_per_epoch: int | None = None
        self._best_metric: float = -1.0
        self._vis_hook: _VisualizationHook | None = None

    def setup(self, cfg, resume: bool = False) -> None:
        self._cfg = cfg
        self._resume = resume

        if comm.is_main_process():
            log_config(cfg, logger)

        model = instantiate(cfg.model)

        if comm.is_main_process():
            log_model(model, logger)

        model.to(cfg.train.device)

        cfg.optimizer.params.model = model
        optim = instantiate(cfg.optimizer)

        train_loader = instantiate(cfg.dataloader.train)

        model = create_ddp_model(model, **cfg.train.ddp)

        try:
            from detrex.modeling import ema

            ema.may_build_model_ema(cfg, model)
            ema_checkpointer_kwargs = ema.may_get_ema_checkpointer(cfg, model)
        except Exception:
            ema_checkpointer_kwargs = {}

        clip_grad = cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None
        self._trainer = _AMPTrainer(
            model=model,
            dataloader=train_loader,
            optimizer=optim,
            amp=cfg.train.amp.enabled,
            clip_grad_params=clip_grad,
        )

        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)

        self._checkpointer = DetectionCheckpointer(
            model,
            output_dir,
            trainer=self._trainer,
            **ema_checkpointer_kwargs,
        )

        iters_per_epoch = None
        try:
            total_images = len(instantiate(cfg.dataloader.train).dataset)
            bs = cfg.dataloader.train.total_batch_size
            iters_per_epoch = max(1, total_images // bs)
        except Exception:
            pass

        if comm.is_main_process():
            self._csv_path = resolve_event_csv_path(output_dir)
            self._ckpt_dir = _derive_ckpt_dir(output_dir)
            self._chart_dir = _derive_chart_dir(output_dir)
            PathManager.mkdirs(self._ckpt_dir)
            PathManager.mkdirs(self._chart_dir)
            self._iters_per_epoch = iters_per_epoch
            writers = [
                CommonMetricPrinter(cfg.train.max_iter),
                JSONWriter(os.path.join(output_dir, "metrics.json")),
                TensorboardXWriter(output_dir),
                _CSVWriter(self._csv_path, iters_per_epoch),
            ]
        else:
            writers = []

        try:
            from detrex.modeling import ema

            ema_hook = ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None
        except Exception:
            ema_hook = None

        vis_hook = None
        if comm.is_main_process() and self._chart_dir is not None:
            class_names = None
            try:
                from detectron2.data import MetadataCatalog

                meta = MetadataCatalog.get(
                    cfg.dataloader.train.dataset.names
                    if hasattr(cfg.dataloader.train.dataset, "names")
                    else cfg.dataloader.train.dataset.get("names", "")
                )
                class_names = list(meta.thing_classes)
            except Exception:
                pass

            vis_period = getattr(cfg.train, "vis_period", cfg.train.eval_period)

            vis_hook = _VisualizationHook(
                chart_dir=self._chart_dir,
                csv_path=self._csv_path,
                vis_period=vis_period,
                val_loader_cfg=cfg.dataloader.test,
                class_names=class_names,
                iters_per_epoch=iters_per_epoch,
            )
            self._vis_hook = vis_hook

        self._trainer.register_hooks(
            [
                hooks.IterationTimer(),
                ema_hook,
                hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
                hooks.EvalHook(
                    cfg.train.eval_period,
                    lambda: self._run_eval(),
                ),
                vis_hook,
                (
                    hooks.PeriodicWriter(writers, period=cfg.train.log_period)
                    if comm.is_main_process()
                    else None
                ),
            ]
        )

        logger.info("Trainer setup complete. Output dir: %s", output_dir)

    def _ensure_initial_lr(self) -> None:
        if self._trainer is None or self._trainer.optimizer is None:
            return
        for group in self._trainer.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
    
    def _load_custom_checkpoint(self, path: str) -> int:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)  # noqa: S614 — trusted local checkpoint
            model = self._trainer.model
            unwrapped = model.module if hasattr(model, "module") else model

            if not (isinstance(ckpt, dict) and "model" in ckpt):

                unwrapped.load_state_dict(ckpt, strict=False)
                logger.info(
                    "Loaded model weights from %s (legacy format, starting at iter 0)",
                    path,
                )
                return 0

            unwrapped.load_state_dict(ckpt["model"], strict=False)

            if "optimizer" in ckpt and self._trainer.optimizer is not None:
                self._trainer.optimizer.load_state_dict(ckpt["optimizer"])

            if "grad_scaler" in ckpt:
                grad_scaler = getattr(self._trainer, "grad_scaler", None)
                if grad_scaler is not None:
                    grad_scaler.load_state_dict(ckpt["grad_scaler"])


            if "best_metric" in ckpt and ckpt["best_metric"] is not None:
                self._best_metric = float(ckpt["best_metric"])

            saved_iter = int(ckpt.get("iteration", 0))
            self._trainer.iter = saved_iter

            metric = ckpt.get("metric", None)
            opt_type = ckpt.get("optimizer_type", "unknown")
            logger.info(
                "Resumed from %s  →  iter %d, optimizer=%s, metric=%s, best_metric=%s",
                path,
                saved_iter,
                opt_type,
                f"{metric :.4f}" if metric is not None else "N/A",
                f"{self ._best_metric :.4f}" if self._best_metric > 0 else "N/A",
            )
            return saved_iter + 1

        except Exception as e:
            logger.warning("Failed to load custom checkpoint %s: %s", path, e)
            return 0

    def train(self) -> None:
        cfg = self._cfg
        start_iter = 0

        ckpt_dir = _derive_ckpt_dir(cfg.train.output_dir)
        last_pth = os.path.join(ckpt_dir, "last_model.pth")
        if os.path.isfile(last_pth):
            logger.info("Found last_model.pth → loading from %s", last_pth)
            start_iter = self._load_custom_checkpoint(last_pth)
        elif cfg.train.init_checkpoint:
            self._checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)
            logger.info("Loaded init_checkpoint as weights only. Starting from iter 0.")

        logger.info(
            "Starting training from iter %d to %d.", start_iter, cfg.train.max_iter
        )
        self._trainer.train(start_iter, cfg.train.max_iter)

    def save_checkpoint(self, output_dir: str, iteration: int) -> None:
        self._checkpointer.save(f"model_{iteration :07d}")
        logger.info("Saved checkpoint at iteration %d.", iteration)

    @staticmethod
    def _patch_coco_evaluator(evaluator) -> None:

        try:
            from detectron2.evaluation import COCOEvaluator, DatasetEvaluators

            inner = evaluator
            if isinstance(evaluator, DatasetEvaluators):
                for ev in evaluator._evaluators:
                    if isinstance(ev, COCOEvaluator):
                        inner = ev
                        break

            if not isinstance(inner, COCOEvaluator):
                return

            orig_derive = inner._derive_coco_results.__func__

            def _patched(self_ev, coco_eval, iou_type, class_names=None):
                if coco_eval is not None:
                    if not hasattr(self_ev, "_coco_eval"):
                        self_ev._coco_eval = {}
                    self_ev._coco_eval[iou_type] = coco_eval
                return orig_derive(
                    self_ev, coco_eval, iou_type, class_names=class_names
                )

            import types

            inner._derive_coco_results = types.MethodType(_patched, inner)
        except Exception as e:
            logger.warning(
                "_patch_coco_evaluator failed (PR/CM charts will be skipped): %s", e
            )

    def _run_eval(self):
        cfg = self._cfg
        model = self._trainer.model

        logger.info("Running evaluation …")
        evaluator = instantiate(cfg.dataloader.evaluator)
        self._patch_coco_evaluator(evaluator)
        ret = inference_on_dataset(model, instantiate(cfg.dataloader.test), evaluator)
        print_csv_format(ret)

        if comm.is_main_process():
            if self._csv_path is not None:
                try:
                    log_eval_metrics(
                        self._csv_path, self._trainer.iter, ret, self._iters_per_epoch
                    )
                except Exception as e:
                    logger.warning("log_eval_metrics failed: %s", e)

            metric: float | None = None
            for key in ("bbox/AP", "segm/AP"):
                try:
                    metric = float(
                        ret["bbox"]["AP"] if key == "bbox/AP" else ret["segm"]["AP"]
                    )
                    break
                except (KeyError, TypeError):
                    pass
            if metric is None:
                try:
                    task = next(iter(ret))
                    metric = float(next(iter(ret[task].values())))
                except Exception:
                    pass

            if self._vis_hook is not None:

                try:
                    self._vis_hook.save_eval_charts(evaluator, metric)
                except Exception as e:
                    logger.warning("VisualizationHook: eval charts failed: %s", e)

                self._vis_hook.save_val_predictions(
                    model=model,
                    val_accuracy=metric,
                )

                try:
                    self._vis_hook._save_training_curves()
                except Exception as e:
                    logger.warning(
                        "VisualizationHook: post-eval training curves failed: %s", e
                    )

            if self._ckpt_dir is not None and metric is not None:
                cur_iter = self._trainer.iter
                optimizer = self._trainer.optimizer

                grad_scaler = getattr(self._trainer, "grad_scaler", None)

                _save_checkpoint(
                    model,
                    os.path.join(self._ckpt_dir, "last_model.pth"),
                    iteration=cur_iter,
                    optimizer=optimizer,
                    metric=metric,
                    best_metric=self._best_metric if self._best_metric > 0 else None,
                    grad_scaler=grad_scaler,
                )
                if metric > self._best_metric:
                    self._best_metric = metric
                    _save_checkpoint(
                        model,
                        os.path.join(self._ckpt_dir, "best_model.pth"),
                        iteration=cur_iter,
                        optimizer=optimizer,
                        metric=metric,
                        best_metric=metric,
                        grad_scaler=grad_scaler,
                    )
                    logger.info("New best metric %.4f → saved best_model.pth", metric)

        return ret
