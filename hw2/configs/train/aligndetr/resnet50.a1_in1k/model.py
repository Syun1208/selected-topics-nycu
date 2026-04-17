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
from detectron2.layers.nms import batched_nms
from detectron2.solver.build import get_default_optimizer_params
from detectron2.structures import Boxes, Instances
from detrex.modeling.backbone import TimmBackbone
from detrex.layers import PositionEmbeddingSine, box_cxcywh_to_xyxy
import detectron2.data.transforms as T

from detrex.data import DetrDatasetMapper
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


class _AlignDETRWithNMS(AlignDETR):

    def __init__(
        self,
        *args,
        nms_iou_threshold: float = 0.5,
        nms_max_per_image: int = 100,
        nms_pre_topk: int = 1000,
        score_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_max_per_image = nms_max_per_image
        self.nms_pre_topk = nms_pre_topk
        self.score_threshold = score_threshold

    def inference(self, box_cls, box_pred, image_sizes):
        assert len(box_cls) == len(image_sizes)
        results = []

        bs, n_queries, n_cls = box_cls.shape
        prob = box_cls.sigmoid()


        all_scores = prob.view(bs, n_queries * n_cls)
        all_indexes = (
            torch.arange(n_queries * n_cls, device=box_cls.device)
            .unsqueeze(0)
            .expand(bs, -1)
        )
        all_boxes_idx = torch.div(all_indexes, n_cls, rounding_mode="floor")
        all_labels = all_indexes % n_cls


        boxes_xyxy = box_cxcywh_to_xyxy(box_pred)
        boxes_exp = torch.gather(
            boxes_xyxy, 1, all_boxes_idx.unsqueeze(-1).expand(-1, -1, 4)
        )

        for scores_i, labels_i, boxes_i, image_size in zip(
            all_scores, all_labels, boxes_exp, image_sizes
        ):

            if self.score_threshold > 0:
                keep_mask = scores_i > self.score_threshold
                scores_i = scores_i[keep_mask]
                labels_i = labels_i[keep_mask]
                boxes_i = boxes_i[keep_mask]


            topk = min(self.nms_pre_topk, scores_i.shape[0])
            if topk < scores_i.shape[0]:
                pre_idx = scores_i.topk(topk).indices
                scores_i = scores_i[pre_idx]
                labels_i = labels_i[pre_idx]
                boxes_i = boxes_i[pre_idx]


            keep = batched_nms(boxes_i, scores_i, labels_i, self.nms_iou_threshold)
            keep = keep[: self.nms_max_per_image]

            result = Instances(image_size)
            result.pred_boxes = Boxes(boxes_i[keep])
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_i[keep]
            result.pred_classes = labels_i[keep]
            results.append(result)

        return results


class _FrozenBN(FrozenBatchNorm2d):

    def __init__(self, num_features, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("dtype", None)
        super().__init__(num_features)





















_EMBED_DIM = 256
_NUM_HEADS = 8
_FFN_DIM = 2048
_NUM_ENC_LAYERS = 6
_NUM_DEC_LAYERS = 6
_NUM_LEVELS = 4
_NUM_QUERIES = 900

model = L(_AlignDETRWithNMS)(
    nms_iou_threshold=0.5,
    nms_max_per_image=100,
    nms_pre_topk=1000,
    score_threshold=0.0,
    backbone=L(TimmBackbone)(
        model_name="resnet50.a1_in1k",
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
                crop_size=(384, 600),
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
    output_dir="src/logs/resnet50.a1_in1k/0",
)


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
    output_dir="src/logs/resnet50.a1_in1k/0",
    init_checkpoint="",
    max_iter=10000,
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
    checkpointer=dict(period=1000, max_to_keep=5),
    eval_period=500,
    log_period=20,
    wandb=dict(
        enabled=False,
        params=dict(
            dir="./wandb_output",
            project="hw2_detrex",
            name="align_detr_resnet50_a1_in1k_exp0",
        ),
    ),
    model_ema=dict(
        enabled=True,
        decay=0.9998,
        device="",
        use_ema_weights_for_eval_only=True,
    ),
    device="cuda",
)
