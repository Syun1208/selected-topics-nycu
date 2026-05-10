import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.interface.base_trainer import BaseTrainer
from src.data.loaders.dataset import CellInstanceDataset
from src.utils.io_utils import scan_dir, train_val_split, stratified_val_split, save_checkpoint, load_checkpoint
from src.utils.visualization import save_all_charts

logger = logging.getLogger(__name__)


def _build_mmdet_model(model_cfg: dict, device: str):
    try:
        from mmdet.utils import register_all_modules
        from mmdet.registry import MODELS
    except ImportError as _e:
        raise ImportError(
            f"Failed to import MMDetection ({_e}). "
            "Run: pip install mmdet mmengine mmcv"
        ) from _e

    register_all_modules()
    try:
        import mmpretrain.models as _
    except ImportError:
        pass
    from mmengine.config import ConfigDict
    model = MODELS.build(ConfigDict(model_cfg))
    model.init_weights()
    model = model.to(device)
    return model


def _collate_fn(batch):
    return batch


class MMDetTrainer(BaseTrainer):

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.model: Optional[Any] = None
        self.optimizer: Optional[Any] = None
        self.scheduler: Optional[Any] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self._val_samples_cache: Optional[List] = None

        # DDP fields (set when launched via torchrun)
        self._local_rank: int = int(cfg.get("training", {}).get("local_rank", -1))
        self._is_distributed: bool = self._local_rank >= 0
        self.is_main: bool = self._local_rank <= 0  # True for single GPU or DDP rank 0

        # Parse gpu_ids for single-GPU device selection: "3" → cuda:3
        gpu_ids_str = str(cfg.get("training", {}).get("gpu_ids", "")).strip()
        self._gpu_ids: List[int] = (
            [int(x.strip()) for x in gpu_ids_str.split(",") if x.strip().isdigit()]
            if gpu_ids_str else []
        )

        # Device: CUDA_VISIBLE_DEVICES remaps physical GPUs so that
        # DDP LOCAL_RANK 0/1/2 → cuda:0/1/2, and single-GPU → cuda:0.
        if self._is_distributed:
            self.device = f"cuda:{self._local_rank}"
        elif self._gpu_ids:
            self.device = "cuda:0"  # CUDA_VISIBLE_DEVICES already selected the GPU

        # AMP scaler (enabled when fp16: true in config)
        self._use_fp16: bool = bool(cfg.get("training", {}).get("fp16", False))
        self._scaler = torch.cuda.amp.GradScaler(enabled=self._use_fp16)

    # ------------------------------------------------------------------
    # Model property helpers
    # ------------------------------------------------------------------

    @property
    def _raw_model(self):
        """Underlying MMDet model, unwrapped from DDP/DataParallel."""
        if isinstance(self.model, (DDP, torch.nn.DataParallel)):
            return self.model.module
        return self.model

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_model(self) -> Any:
        model_name = self.cfg["model"]["name"].lower()
        if model_name == "cascade_mask_rcnn":
            from src.models.cascade_mask_rcnn import build_cascade_mask_rcnn
            model_cfg = build_cascade_mask_rcnn(self.cfg["model"])
        elif model_name == "htc":
            from src.models.htc import build_htc
            model_cfg = build_htc(self.cfg["model"])
        elif model_name == "mask2former":
            from src.models.mask2former import build_mask2former
            model_cfg = build_mask2former(self.cfg["model"])
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # DDP: let rank-0 download/cache pretrained weights first;
        # other ranks wait at the barrier then load from the local cache.
        if self._is_distributed:
            import torch.distributed as dist
            if self._local_rank != 0:
                dist.barrier()

        self.model = _build_mmdet_model(model_cfg, self.device)

        if self._is_distributed:
            import torch.distributed as dist
            if self._local_rank == 0:
                dist.barrier()
        if self.is_main:
            logger.info(f"Built {model_name} with backbone={self.cfg['model']['backbone']}")

        # DDP: each process wraps its own replica; gradients are all-reduced automatically
        if self._is_distributed:
            self.model = DDP(self.model, device_ids=[self._local_rank], find_unused_parameters=False)
            if self.is_main:
                world_size = self.cfg.get("training", {}).get("world_size", 1)
                logger.info(f"DDP enabled: world_size={world_size}")

        train_cfg = self.cfg.get("training", {})
        self.optimizer = self._build_optimizer(train_cfg)

        # Load pretrained backbone weights (skipped if resume_from is also set)
        pretrain_path = self.cfg.get("model", {}).get("pretrain_weights", None)
        if isinstance(pretrain_path, dict):
            pretrain_path = pretrain_path.get("path", None)
        if pretrain_path and os.path.exists(str(pretrain_path)):
            ckpt = load_checkpoint(str(pretrain_path), map_location=self.device)
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            self._raw_model.load_state_dict(state, strict=False)
            if self.is_main:
                logger.info(f"Loaded pretrained weights from {pretrain_path}")

        # Resume from checkpoint (overrides pretrained weights)
        resume_path = self.cfg.get("training", {}).get("resume_from") or None
        if resume_path and os.path.exists(str(resume_path)):
            ckpt = load_checkpoint(str(resume_path), map_location=self.device)
            self._raw_model.load_state_dict(ckpt.get("state_dict", ckpt))
            if "optimizer" in ckpt and self.optimizer:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            self.current_epoch = ckpt.get("epoch", 0) + 1
            self.best_metric = ckpt.get("best_metric", 0.0)
            if self.is_main:
                logger.info(f"Resumed from {resume_path} (next epoch: {self.current_epoch})")

        return self.model

    def _build_optimizer(self, train_cfg: dict):
        lr = float(train_cfg.get("lr", 0.02))
        weight_decay = float(train_cfg.get("weight_decay", 0.0001))
        optimizer_type = train_cfg.get("optimizer", "auto").lower()

        if optimizer_type == "auto":
            backbone = self.cfg.get("model", {}).get("backbone", "")
            optimizer_type = "adamw" if backbone in ("swin_b", "swin_l", "convnext_l") else "sgd"

        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        return torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    def _build_scheduler(self, train_cfg: dict):
        warmup_epochs = int(train_cfg.get("warmup_epochs", 1))
        milestones = train_cfg.get("lr_milestones", [8, 11])
        milestones = [m * (len(self.train_loader) if self.train_loader else 1)
                      for m in milestones]

        def lr_lambda(step):
            total_warmup = warmup_epochs * max(len(self.train_loader) if self.train_loader else 1, 1)
            if step < total_warmup:
                return step / max(total_warmup, 1)
            factor = 1.0
            for m in milestones:
                if step >= m:
                    factor *= 0.1
            return factor

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def build_dataloaders(self) -> dict:
        data_cfg = self.cfg.get("data", {})
        train_dir = data_cfg.get("train_dir", "data/train")
        aug_cfg = data_cfg.get("augmentation", {})
        batch_size = int(self.cfg.get("training", {}).get("batch_size", 2))
        num_workers = int(data_cfg.get("num_workers", 4))
        use_oversampling = data_cfg.get("use_oversampling", True)

        all_ids = scan_dir(train_dir)
        val_ratio = float(data_cfg.get("val_ratio", 0.2))
        train_ids, val_ids = stratified_val_split(all_ids, train_dir, val_ratio=val_ratio)
        val_data_dir = train_dir
        if self.is_main:
            logger.info(f"Train: {len(train_ids)} | Val: {len(val_ids)} (stratified {val_ratio:.0%} per class)")

        train_dataset = CellInstanceDataset(
            train_dir, train_ids, augmentation_cfg=aug_cfg, mode="train"
        )
        val_dataset = CellInstanceDataset(
            val_data_dir, val_ids, augmentation_cfg=aug_cfg, mode="val"
        )

        # Sampler: DDP uses DistributedSampler; single GPU uses WeightedRandomSampler
        if self._is_distributed:
            from torch.utils.data import DistributedSampler
            world_size = self.cfg.get("training", {}).get("world_size", 1)
            sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=self._local_rank, shuffle=True
            )
            if self.is_main and use_oversampling:
                logger.warning("WeightedRandomSampler is not used in DDP mode (DistributedSampler instead)")
        else:
            sampler = None
            if use_oversampling:
                weights = train_dataset.get_oversampling_weights()
                sampler = WeightedRandomSampler(
                    weights=torch.from_numpy(weights),
                    num_samples=len(train_dataset),
                    replacement=True,
                )

        pin = torch.cuda.is_available()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # sampler handles shuffle
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
        )

        self._val_ids = val_ids
        self._train_dir = train_dir
        self._val_dir = val_data_dir
        return {"train": self.train_loader, "val": self.val_loader}

    # ------------------------------------------------------------------
    # Training loop (override base to add DDP sync)
    # ------------------------------------------------------------------

    def run(self) -> None:
        from tqdm import tqdm as _tqdm

        self.build_model()
        self.build_dataloaders()
        self.on_train_start()

        epoch_bar = _tqdm(
            range(self.current_epoch, self.max_epochs),
            desc="Epochs", unit="ep",
            initial=self.current_epoch, total=self.max_epochs,
            dynamic_ncols=True, colour="green",
            disable=not self.is_main,
        )

        for epoch in epoch_bar:
            self.current_epoch = epoch

            # DistributedSampler needs epoch to shuffle differently each epoch
            if self._is_distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self.train_epoch(epoch)
            self.train_loss_history.append(train_metrics.get("loss", 0.0))

            # Validation: only rank 0 runs it; others skip and come here immediately.
            # DO NOT use dist.barrier() — ranks 1/2 would wait up to 30 min (NCCL watchdog).
            # Instead, rank 0 computes metrics and checkpoints, then broadcasts mask_ap.
            # The broadcast is the only NCCL op here and acts as the synchronisation point.
            val_metrics = self.validate(epoch)

            if self.is_main:
                mask_ap = val_metrics.get("mask_ap", 0.0)
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

            # Broadcast mask_ap and best_metric from rank 0 → synchronises all ranks
            # (non-main ranks are already idle here waiting for this broadcast)
            if self._is_distributed:
                import torch.distributed as dist
                sync = torch.tensor(
                    [val_metrics.get("mask_ap", 0.0), self.best_metric],
                    device=self.device,
                )
                dist.broadcast(sync, src=0)
                mask_ap = float(sync[0])
                self.best_metric = float(sync[1])
            else:
                mask_ap = val_metrics.get("mask_ap", 0.0)

            self.val_map_history.append(mask_ap)

            self.on_epoch_end(epoch, train_metrics, val_metrics)

        self.on_train_end()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        from tqdm import tqdm
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        running: Dict[str, float] = {}
        n_skipped = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train E{epoch + 1}", unit="batch",
            dynamic_ncols=True, leave=False, colour="green",
            disable=not self.is_main,
        )
        grad_clip = float(self.cfg.get("training", {}).get("grad_clip", 35.0))

        for batch in pbar:
            # ── forward ──────────────────────────────────────────────────
            losses_dict = None
            try:
                losses_dict = self._forward_train(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                logger.warning("Train batch OOM — skipping")
            except Exception as e:
                logger.warning(f"Train batch error ({type(e).__name__}): {e}")

            # ── DDP sync: all ranks decide together whether to skip ───────
            # dist.all_reduce runs BEFORE any backward, so both ranks are
            # guaranteed to reach this point (one from the except, one from
            # the normal path).  The subsequent backward (skip or real) then
            # keeps DDP's reducer in sync.
            if self._is_distributed:
                import torch.distributed as dist
                ok = torch.tensor([0 if losses_dict is None else 1],
                                  device=self.device, dtype=torch.int32)
                dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                all_ok = ok.item() == 1
            else:
                all_ok = losses_dict is not None

            if not all_ok:
                # At least one rank failed.  Every rank that had a DDP forward
                # MUST call backward to release the reducer — otherwise the
                # rank that succeeded hangs in NCCL waiting for all-reduce.
                try:
                    if losses_dict is not None:
                        # My forward succeeded — zero out the actual loss so
                        # the computation graph (and DDP hooks) are reused.
                        bl = sum(v * 0.0 for v in losses_dict.values())
                    else:
                        # My forward failed — noop through all parameters.
                        bl = sum(p.sum() * 0.0
                                 for p in self.model.parameters()
                                 if p.requires_grad)
                    self._scaler.scale(bl).backward()
                    self._scaler.update()
                except Exception:
                    pass
                self.optimizer.zero_grad()
                n_skipped += 1
                continue

            # ── normal backward + optimizer step ─────────────────────────
            loss = sum(losses_dict.values())
            self.optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            for k, v in losses_dict.items():
                running[k] = running.get(k, 0.0) + v.item()

            if self.is_main:
                postfix: Dict[str, str] = {}
                for k, v in losses_dict.items():
                    short = (k.replace("loss_", "")
                              .replace("s0.", "s0/").replace("s1.", "s1/").replace("s2.", "s2/"))
                    postfix[short] = f"{v.item():.3f}"
                postfix["total"] = f"{loss.item():.4f}"
                postfix["avg"]   = f"{total_loss / n_batches:.4f}"
                pbar.set_postfix(postfix)

        if n_skipped and self.is_main:
            logger.warning(f"Epoch {epoch+1}: skipped {n_skipped}/{n_batches+n_skipped} batches")
        avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}
        avg_losses["loss"] = total_loss / max(n_batches, 1)
        return avg_losses

    def _forward_train(self, batch):
        if not batch:
            return None
        try:
            from mmdet.structures import DetDataSample
            from mmdet.structures.mask import BitmapMasks
            from mmengine.structures import InstanceData

            inputs = []
            data_samples = []

            for item in batch:
                img = item["img"].to(self.device)
                h, w = img.shape[-2:]
                inputs.append(img)

                gt_instances = InstanceData()
                gt_instances.bboxes = item["gt_bboxes"].to(self.device)
                gt_instances.labels = item["gt_labels"].to(self.device)

                raw_masks = item.get("gt_masks", [])
                if raw_masks:
                    mask_arr = np.stack([np.array(m, dtype=np.uint8) for m in raw_masks], axis=0)
                    gt_instances.masks = BitmapMasks(mask_arr, height=h, width=w)
                else:
                    gt_instances.masks = BitmapMasks(np.zeros((0, h, w), dtype=np.uint8), height=h, width=w)

                data_sample = DetDataSample()
                data_sample.gt_instances = gt_instances
                data_sample.set_metainfo(self._fix_img_metas(item["img_metas"], h, w))

                # HTC FusedSemanticHead requires gt_sem_seg (class map built from instance masks)
                if self.cfg["model"]["name"].lower() == "htc":
                    from mmengine.structures import PixelData
                    sem_seg = np.zeros((h, w), dtype=np.int64)
                    labels_np = item["gt_labels"].numpy()
                    for mask, label in zip(raw_masks, labels_np):
                        sem_seg[np.array(mask, dtype=bool)] = int(label) + 1  # 0=bg, 1..N=classes
                    gt_sem = PixelData()
                    gt_sem.sem_seg = torch.from_numpy(sem_seg).unsqueeze(0).to(self.device)
                    data_sample.gt_sem_seg = gt_sem

                data_samples.append(data_sample)

            img = torch.stack(inputs, dim=0)
            with torch.cuda.amp.autocast(enabled=self._use_fp16):
                losses = self.model(img, data_samples, mode="loss")
            # MMDet returns some losses as lists of tensors (e.g. loss_rpn_cls/bbox:
            # one tensor per FPN level). Sum them so all parameters get gradients.
            result: Dict[str, torch.Tensor] = {}
            for k, v in losses.items():
                if "loss" not in k:
                    continue
                if isinstance(v, torch.Tensor):
                    result[k] = v
                elif isinstance(v, (list, tuple)):
                    valid = [t for t in v if isinstance(t, torch.Tensor)]
                    if valid:
                        result[k] = sum(valid)
            return result if result else None
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning("Train batch OOM — skipping (reduce batch_size or enable with_cp)")
            return None
        except Exception as e:
            logger.warning(f"Train forward error ({type(e).__name__}): {e}")
            return None

    def _ddp_skip(self) -> None:
        """Legacy helper — skip logic is now handled inline in train_epoch."""
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Validation (rank 0 only)
    # ------------------------------------------------------------------

    def validate(self, epoch: int) -> Dict[str, float]:
        # Non-main ranks skip — they wait at the barrier in run()
        if not self.is_main:
            num_classes = self.cfg["model"].get("num_classes", 4)
            return {
                "mask_ap": 0.0,
                "ap_per_class": {},
                "pr_data": {},
                "confusion_matrix": np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64),
            }

        from tqdm import tqdm
        self.model.eval()
        all_preds = []
        all_gt = []
        vis_raw = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val E{epoch + 1}", unit="batch",
                        dynamic_ncols=True, leave=False)
            for batch in pbar:
                for item in batch:
                    preds = self._forward_validate(item, score_thr=0.05)
                    all_preds.append(preds)
                    all_gt.append({
                        "gt_bboxes": item["gt_bboxes"].numpy(),
                        "gt_labels": item["gt_labels"].numpy(),
                        "gt_masks": item.get("gt_masks", []),
                    })
                    if len(vis_raw) < 6:
                        vis_raw.append({
                            "img_chw": item["img"].numpy(),
                            "gt_masks": item.get("gt_masks", []),
                            "gt_labels": item["gt_labels"].numpy().tolist(),
                            "pred_masks": preds["masks"],
                            "pred_labels": list(np.asarray(preds["labels"], int)),
                            "pred_scores": list(np.asarray(preds["scores"], float)),
                        })

        total_preds = sum(len(p["bboxes"]) for p in all_preds)
        logger.info(f"Val E{epoch + 1}: {total_preds} predictions across {len(all_preds)} images")
        logger.info("Computing val metrics (mask IoU)...")
        metrics = self._compute_val_metrics(all_preds, all_gt)
        logger.info(f"Val metrics done: mAP={metrics.get('mask_ap', 0):.4f}")
        # Free large mask arrays immediately to reduce peak RAM before the DDP barrier
        del all_preds, all_gt
        metrics["vis_raw"] = vis_raw
        return metrics

    def _forward_validate(self, item: dict, score_thr: Optional[float] = None) -> dict:
        try:
            from mmdet.structures import DetDataSample
            img = item["img"].unsqueeze(0).to(self.device)
            h, w = img.shape[-2:]

            img_metas = self._fix_img_metas(item["img_metas"], h, w)
            data_sample = DetDataSample()
            data_sample.set_metainfo(img_metas)

            # Use raw model: DDP/DataParallel cannot gather non-tensor DetDataSample outputs
            results = self._raw_model(img, [data_sample], mode="predict")
            pred = results[0].pred_instances

            if score_thr is None:
                score_thr = self.cfg.get("inference", {}).get("score_thr", 0.05)
            keep_idx = (pred.scores >= score_thr).nonzero(as_tuple=False).squeeze(1).cpu().numpy()

            bboxes = pred.bboxes[keep_idx].cpu().tolist()
            scores = pred.scores[keep_idx].cpu().numpy()
            labels = pred.labels[keep_idx].cpu().numpy()

            masks = []
            if hasattr(pred, "masks") and len(keep_idx) > 0:
                raw = pred.masks[keep_idx]
                mask_arr = raw.masks if hasattr(raw, "masks") else np.asarray(raw.cpu())
                masks = [mask_arr[i].astype(np.uint8) for i in range(len(mask_arr))]

            return {"bboxes": bboxes, "labels": labels, "scores": scores, "masks": masks}
        except Exception as e:
            logger.exception(f"Val forward error ({type(e).__name__}): {e}")
            return {"bboxes": [], "labels": np.zeros(0, int), "scores": np.zeros(0), "masks": []}

    def _compute_val_metrics(self, all_preds, all_gt) -> dict:
        from src.utils.metrics import (
            compute_mask_ap_per_class, compute_ap_per_class, compute_map,
            compute_pr_per_class, compute_confusion_matrix,
        )
        num_classes = self.cfg["model"].get("num_classes", 4)

        pred_scores = [np.asarray(p["scores"]) for p in all_preds]
        pred_labels = [np.asarray(p["labels"], dtype=int) for p in all_preds]
        gt_labels   = [np.asarray(g["gt_labels"], dtype=int) for g in all_gt]

        has_masks = all(len(p["masks"]) == len(p["bboxes"]) for p in all_preds)
        if has_masks:
            pred_masks = [p["masks"] for p in all_preds]
            gt_masks   = [g["gt_masks"] for g in all_gt]
            ap_dict = compute_mask_ap_per_class(
                pred_masks, pred_scores, pred_labels,
                gt_masks, gt_labels,
                num_classes=num_classes, iou_threshold=0.5,
            )
            pr_data = compute_pr_per_class(
                pred_masks, pred_scores, pred_labels,
                gt_masks, gt_labels,
                num_classes=num_classes, iou_threshold=0.5,
            )
            cm = compute_confusion_matrix(
                pred_masks, pred_scores, pred_labels,
                gt_masks, gt_labels,
                num_classes=num_classes, iou_threshold=0.5,
            )
        else:
            logger.warning("Masks missing in some predictions; falling back to box AP for validation")
            pred_bboxes = [np.array(p["bboxes"]).reshape(-1, 4) if p["bboxes"] else np.zeros((0, 4))
                           for p in all_preds]
            gt_bboxes = [g["gt_bboxes"] for g in all_gt]
            ap_dict = compute_ap_per_class(
                pred_bboxes, pred_scores, pred_labels,
                gt_bboxes, gt_labels,
                num_classes=num_classes, iou_threshold=0.5,
            )
            pr_data = {}
            cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

        mAP = compute_map(ap_dict)
        return {"mask_ap": mAP, "ap_per_class": ap_dict, "pr_data": pr_data, "confusion_matrix": cm}

    def _fix_img_metas(self, img_metas, h, w):
        img_metas = dict(img_metas)

        if "scale_factor" in img_metas:
            sf = img_metas["scale_factor"]
            if isinstance(sf, torch.Tensor):
                sf = sf.cpu().numpy()
            sf = np.array(sf).reshape(-1)
            if len(sf) == 4:
                img_metas["scale_factor"] = sf[:2].tolist()
            elif len(sf) == 2:
                img_metas["scale_factor"] = sf.tolist()

        img_metas.setdefault("batch_input_shape", (h, w))
        img_metas.setdefault("img_shape", (h, w))
        img_metas.setdefault("ori_shape", (h, w))
        return img_metas

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        if not self.is_main:
            return
        state = {
            "epoch": self.current_epoch,
            "state_dict": self._raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "is_best": is_best,
            "cfg": self.cfg,
        }
        save_checkpoint(state, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = load_checkpoint(path, map_location=self.device)
        self._raw_model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt and self.optimizer:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.current_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("best_metric", 0.0)
        logger.info(f"Resumed from {path} (epoch {self.current_epoch})")

    # ------------------------------------------------------------------
    # Hooks (rank-0 only)
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        train_cfg = self.cfg.get("training", {})
        self.scheduler = self._build_scheduler(train_cfg)
        self._loss_history: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        if not self.is_main:
            return
        from src.utils.visualization import _denorm
        self._loss_history.append({k: v for k, v in train_metrics.items()})

        metrics_data: dict = {
            "epoch": epoch + 1,
            "train_losses": self.train_loss_history,
            "val_map_history": self.val_map_history,
            "loss_history": self._loss_history,
            "ap_per_class": val_metrics.get("ap_per_class", {}),
        }

        if "pr_data" in val_metrics and val_metrics["pr_data"]:
            metrics_data["pr_data"] = val_metrics["pr_data"]

        if "confusion_matrix" in val_metrics:
            metrics_data["confusion_matrix"] = val_metrics["confusion_matrix"]

        if "vis_raw" in val_metrics and val_metrics["vis_raw"]:
            items = val_metrics["vis_raw"]
            metrics_data["vis_data"] = {
                "images":      [_denorm(v["img_chw"]) for v in items],
                "gt_masks":    [v["gt_masks"]   for v in items],
                "gt_labels":   [v["gt_labels"]  for v in items],
                "pred_masks":  [v["pred_masks"] for v in items],
                "pred_labels": [v["pred_labels"] for v in items],
                "pred_scores": [v["pred_scores"] for v in items],
            }

        save_all_charts(str(self.chart_dir), metrics_data, cfg=self.cfg)
        logger.info(f"Charts updated after epoch {epoch + 1} (loss={train_metrics.get('loss', 0):.4f})")

    def on_train_end(self) -> None:
        if not self.is_main:
            return
        metrics_data = {
            "train_losses": self.train_loss_history,
            "val_map_history": self.val_map_history,
            "loss_history": getattr(self, "_loss_history", []),
        }
        save_all_charts(str(self.chart_dir), metrics_data, cfg=self.cfg)
        logger.info(f"Charts saved to {self.chart_dir}")
