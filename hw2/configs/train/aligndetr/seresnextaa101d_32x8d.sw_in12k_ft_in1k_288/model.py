import copy
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../.."))
_DETREX_ROOT = os.path.join(_PROJECT_ROOT, "configs", "detrex")

for _p in [_DETREX_ROOT, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.layers import FrozenBatchNorm2d, ShapeSpec
from detectron2.solver.build import get_default_optimizer_params
from detrex.modeling.backbone import TimmBackbone
import detectron2.data.transforms as T

from detrex.data import DetrDatasetMapper
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.neck import ChannelMapper
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from projects.align_detr.modeling import (
    AlignDETR,
    AlignDETRCriterion,
    MixedMatcher,
    Transformer,
    TransformerDecoder,
    TransformerEncoder,
)

NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------


class _FrozenBN(FrozenBatchNorm2d):
    """FrozenBatchNorm2d that ignores device/dtype kwargs passed by timm."""

    def __init__(self, num_features, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("dtype", None)
        super().__init__(num_features)


class _TimmBackbone(TimmBackbone):
    """TimmBackbone with size_divisibility=64 for seresnextaa anti-aliased ResNet.
    32 is insufficient — blur pooling causes off-by-one (80 vs 81) in residuals."""

    @property
    def size_divisibility(self):
        return 64


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
#
# Backbone  seresnextaa101d_32x8d  (out_indices 2,3,4)
#   p2 :  512 ch  stride  8
#   p3 : 1024 ch  stride 16
#   p4 : 2048 ch  stride 32
#
# Neck  ChannelMapper  (3 → 4 levels, all projected to 256 ch)
#   n2 : 256 ch  stride  8   ← from p2
#   n3 : 256 ch  stride 16   ← from p3
#   n4 : 256 ch  stride 32   ← from p4
#   n5 : 256 ch  stride 64   ← extra level (stride-2 conv on p4)
#
# Transformer  embed_dim=256, 4 feature levels, 6-layer encoder, 6-layer decoder
#   Queries : 900
# ---------------------------------------------------------------------------

_EMBED_DIM = 256
_NUM_HEADS = 8
_FFN_DIM = 2048  # feedforward_dim = 8 × embed_dim
_NUM_ENC_LAYERS = 6
_NUM_DEC_LAYERS = 6
_NUM_LEVELS = 4  # must match ChannelMapper num_outs
_NUM_QUERIES = 900

model = L(AlignDETR)(
    backbone=L(_TimmBackbone)(
        model_name="seresnextaa101d_32x8d.sw_in12k_ft_in1k_288",
        features_only=True,
        pretrained=True,
        in_channels=3,
        out_indices=(2, 3, 4),
        norm_layer=_FrozenBN,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=_EMBED_DIM // 2,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "p2": ShapeSpec(channels=512),
            "p3": ShapeSpec(channels=1024),
            "p4": ShapeSpec(channels=2048),
        },
        in_features=["p2", "p3", "p4"],
        out_channels=_EMBED_DIM,
        num_outs=_NUM_LEVELS,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=_EMBED_DIM),
    ),
    transformer=L(Transformer)(
        encoder=L(TransformerEncoder)(
            embed_dim=_EMBED_DIM,
            num_heads=_NUM_HEADS,
            feedforward_dim=_FFN_DIM,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=_NUM_ENC_LAYERS,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
        ),
        decoder=L(TransformerDecoder)(
            embed_dim=_EMBED_DIM,
            num_heads=_NUM_HEADS,
            feedforward_dim=_FFN_DIM,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=_NUM_DEC_LAYERS,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
        ),
        num_feature_levels=_NUM_LEVELS,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=_EMBED_DIM,
    num_classes=NUM_CLASSES,
    num_queries=_NUM_QUERIES,
    aux_loss=True,
    criterion=L(AlignDETRCriterion)(
        num_classes="${..num_classes}",
        matcher=L(MixedMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        match_num=[2, 2, 2, 2, 2, 2, 1],
        alpha=0.25,
        gamma=2.0,
        tau=1.5,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
    prior_init=0.01,
)

_base_wd = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    _aux_wd = {}
    _aux_wd.update({k + "_enc": v for k, v in _base_wd.items()})
    for _i in range(model.transformer.decoder.num_layers - 1):
        _aux_wd.update({k + f"_{_i}": v for k, v in _base_wd.items()})
    model.criterion.weight_dict = {**model.criterion.weight_dict, **_aux_wd}

# ---------------------------------------------------------------------------
# Dataloader
# All short_edge_length values are multiples of 32 to minimise padding.
# max_size=1344 (32×42) avoids edge cases with anti-aliased striding.
# ---------------------------------------------------------------------------

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="train"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1344,
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
                max_size=1344,
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
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1344),
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
    output_dir="src/logs/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0",
)

# ---------------------------------------------------------------------------
# Optimizer  –  backbone gets 0.1× LR (standard for pretrained backbone)
# ---------------------------------------------------------------------------

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
        milestones=[7500, 9000],
        num_updates=10000,
    ),
    warmup_length=100 / 10000,
    warmup_method="linear",
    warmup_factor=0.001,
)

train = dict(
    output_dir="src/logs/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0",
    init_checkpoint="",
    max_iter=10000,
    amp=dict(enabled=False),
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
    checkpointer=dict(period=1000, max_to_keep=5),
    eval_period=500,
    log_period=20,
    wandb=dict(
        enabled=False,
        params=dict(
            dir="./wandb_output",
            project="hw2_detrex",
            name="align_detr_seresnextaa101d_exp0",
        ),
    ),
    model_ema=dict(
        enabled=False,
        decay=0.999,
        device="",
        use_ema_weights_for_eval_only=False,
    ),
    device="cuda",
)
