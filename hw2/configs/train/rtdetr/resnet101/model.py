"""
RT-DETRv2 with PResNet-101 backbone (native RT-DETR backbone, pretrained from Paddle).
Architecture matches rtdetrv2_r101vd_6x_coco.yml:
  hidden_dim=384, dim_feedforward=2048, feat_channels=[384,384,384]
"""

import os
import sys
import types as _types

import torch
import torch.nn as nn
from omegaconf import OmegaConf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../.."))
_DETREX_ROOT = os.path.join(_PROJECT_ROOT, "configs", "detrex")

for _p in [_DETREX_ROOT, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Pre-register rtdetr_src as an empty package to skip its __init__.py
# (which imports data/ → faster_coco_eval, not installed).
if "rtdetr_src" not in sys.modules:
    _rtdetr_src_path = os.path.join(_PROJECT_ROOT, "rtdetr_src")
    _stub = _types.ModuleType("rtdetr_src")
    _stub.__path__ = [_rtdetr_src_path]
    _stub.__package__ = "rtdetr_src"
    _stub.__file__ = os.path.join(_rtdetr_src_path, "__init__.py")
    sys.modules["rtdetr_src"] = _stub

# Stub rtdetr_src.data: misc/dist_utils.py does `from ..data import DataLoader`
if "rtdetr_src.data" not in sys.modules:
    import torch.utils.data as _tud
    _data_stub = _types.ModuleType("rtdetr_src.data")
    _data_stub.__path__ = [os.path.join(_PROJECT_ROOT, "rtdetr_src", "data")]
    _data_stub.__package__ = "rtdetr_src.data"
    _data_stub.DataLoader = _tud.DataLoader
    sys.modules["rtdetr_src.data"] = _data_stub

from rtdetr_src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from rtdetr_src.zoo.rtdetr.rtdetrv2_decoder import RTDETRTransformerv2
from rtdetr_src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2
from rtdetr_src.zoo.rtdetr.matcher import HungarianMatcher
from rtdetr_src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
from rtdetr_src.nn.backbone.presnet import PResNet

from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.solver import WarmupParamScheduler
from detectron2.structures import Boxes, Instances
import detectron2.data.transforms as T
from detrex.data import DetrDatasetMapper
from fvcore.common.param_scheduler import MultiStepParamScheduler

NUM_CLASSES = 10


class RTDETRDetectron2(nn.Module):
    """
    RT-DETRv2 wrapper compatible with Detectron2 trainer/tester.
    Uses native PResNet-101 backbone (pretrained weights from Paddle).
    Architecture: hidden_dim=384, dim_feedforward=2048 (R101 spec).
    """

    def __init__(
        self,
        num_classes: int = 10,
        backbone_name: str = "resnet101",  
        backbone_pretrained: bool = True,
        hidden_dim: int = 384,
        dim_feedforward: int = 2048,
        num_queries: int = 300,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 6,
        num_denoising: int = 100,
        pixel_mean=(123.675, 116.280, 103.530),
        pixel_std=(58.395, 57.120, 57.375),
    ):
        super().__init__()

        self.backbone = PResNet(
            depth=101,
            variant="d",
            freeze_at=0,
            return_idx=[1, 2, 3],   # strides [8, 16, 32], channels [512, 1024, 2048]
            num_stages=4,
            freeze_norm=True,
            pretrained=backbone_pretrained,
        )
        in_channels = self.backbone.out_channels  # [512, 1024, 2048]
        feat_strides = self.backbone.out_strides  # [8, 16, 32]

        # ---- encoder ----
        self.encoder = HybridEncoder(
            in_channels=in_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            use_encoder_idx=[2],
            num_encoder_layers=num_encoder_layers,
            nhead=8,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            enc_act="gelu",
            expansion=1.0,
            depth_mult=1.0,
            act="silu",
            eval_spatial_size=None,
        )

        # ---- decoder ----
        feat_channels = [hidden_dim] * len(in_channels)  # [384, 384, 384]
        self.decoder = RTDETRTransformerv2(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            feat_channels=feat_channels,
            feat_strides=feat_strides,
            num_levels=len(in_channels),
            num_points=[4, 4, 4],
            nhead=8,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            num_denoising=num_denoising,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            eval_spatial_size=None,
            eval_idx=-1,
            aux_loss=True,
            cross_attn_method="default",
            query_select_method="default",
        )

        # ---- criterion ----
        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.criterion = RTDETRCriterionv2(
            matcher=matcher,
            weight_dict={"loss_vfl": 1, "loss_bbox": 5, "loss_giou": 2},
            losses=["vfl", "boxes"],
            alpha=0.75,
            gamma=2.0,
            num_classes=num_classes,
            boxes_weight_format="giou",
        )

        # ---- postprocessor ----
        self.postprocessor = RTDETRPostProcessor(
            num_classes=num_classes,
            use_focal_loss=True,
            num_top_queries=num_queries,
        )

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(3, 1, 1))
        self.register_buffer("pixel_std",  torch.tensor(pixel_std).view(3, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        return (img.float() - self.pixel_mean) / self.pixel_std

    def _pad_images(self, images):
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        max_h = (max_h + 31) // 32 * 32
        max_w = (max_w + 31) // 32 * 32
        batch = torch.zeros(len(images), 3, max_h, max_w, device=self.device)
        for i, img in enumerate(images):
            batch[i, :, :img.shape[1], :img.shape[2]] = img
        return batch

    def forward(self, batched_inputs):
        imgs = [self._normalize(x["image"].to(self.device)) for x in batched_inputs]
        img_sizes = [(img.shape[1], img.shape[2]) for img in imgs]
        samples = self._pad_images(imgs)

        if self.training:
            targets = self._build_targets(batched_inputs, img_sizes)
            feats = self.backbone(samples)
            feats = self.encoder(feats)
            out = self.decoder(feats, targets)
            return self.criterion(out, targets)
        else:
            feats = self.backbone(samples)
            feats = self.encoder(feats)
            out = self.decoder(feats)
            orig_sizes = torch.tensor(
                [[x["width"], x["height"]] for x in batched_inputs],
                dtype=torch.float32, device=self.device,
            )
            results = self.postprocessor(out, orig_sizes)
            return self._to_detectron2(results, batched_inputs)

    def _build_targets(self, batched_inputs, img_sizes):
        targets = []
        for inp, (h, w) in zip(batched_inputs, img_sizes):
            instances = inp.get("instances")
            if instances is None or len(instances) == 0:
                targets.append({
                    "labels": torch.zeros(0, dtype=torch.long, device=self.device),
                    "boxes":  torch.zeros(0, 4, device=self.device),
                })
                continue
            gt_boxes   = instances.gt_boxes.tensor.to(self.device)
            gt_classes = instances.gt_classes.to(self.device)
            cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0 / w
            cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0 / h
            bw = (gt_boxes[:, 2] - gt_boxes[:, 0]) / w
            bh = (gt_boxes[:, 3] - gt_boxes[:, 1]) / h
            boxes = torch.stack([cx, cy, bw, bh], dim=1).clamp(0.0, 1.0)
            targets.append({"labels": gt_classes, "boxes": boxes})
        return targets

    @staticmethod
    def _to_detectron2(results, batched_inputs):
        outputs = []
        for result, inp in zip(results, batched_inputs):
            h, w = inp["height"], inp["width"]
            inst = Instances((h, w))
            inst.pred_boxes   = Boxes(result["boxes"])
            inst.pred_classes = result["labels"]
            inst.scores       = result["scores"]
            outputs.append({"instances": inst})
        return outputs


# ---------------------------------------------------------------------------
# LazyConfig objects
# ---------------------------------------------------------------------------

model = L(RTDETRDetectron2)(
    num_classes=NUM_CLASSES,
    backbone_pretrained=True,
    hidden_dim=384,           # R101 spec (vs 256 for R50)
    dim_feedforward=2048,     # R101 spec (vs 1024 for R50)
    num_queries=300,
    num_encoder_layers=1,
    num_decoder_layers=6,
    num_denoising=100,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
)

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(384, 512, 608),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 608),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=2,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="valid", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="valid",
    output_dir="logs/rtdetr/resnet101/0",
)

import torch.optim
from detectron2.solver.build import get_default_optimizer_params

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        weight_decay_norm=0.0,
        lr_factor_func=lambda module_name: 0.1 if "backbone" in module_name else 1.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[202500, 243000],
        num_updates=270000,
    ),
    warmup_length=1000 / 270000,
    warmup_method="linear",
    warmup_factor=0.001,
)

train = dict(
    output_dir="logs/rtdetr/resnet101/0",
    init_checkpoint="",
    max_iter=270000,
    amp=dict(enabled=True),
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    clip_grad=dict(
        enabled=True,
        params=dict(max_norm=0.1, norm_type=2),
    ),
    seed=-1,
    fast_dev_run=dict(enabled=False),
    checkpointer=dict(period=5000, max_to_keep=5),
    eval_period=5000,
    log_period=20,
    wandb=dict(
        enabled=False,
        params=dict(
            dir="./wandb_output",
            project="hw2_detrex",
            name="rtdetr_resnet101_exp0",
        ),
    ),
    model_ema=dict(
        enabled=False,
        decay=0.9998,
        device="",
        use_ema_weights_for_eval_only=False,
    ),
    device="cuda",
)
