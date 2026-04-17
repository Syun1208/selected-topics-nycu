import copy
import math
import os
import sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "../../../.."))
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
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine, box_cxcywh_to_xyxy
from detrex.modeling.losses.giou_loss import giou_loss as _giou_loss
from detrex.utils import inverse_sigmoid
import detectron2.data.transforms as T
from detectron2.layers.nms import batched_nms
from detectron2.modeling import detector_postprocess
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, Instances, pairwise_iou

from detrex.data import DetrDatasetMapper
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from projects.dino.modeling import (
    DINO,
    DINOCriterion,
    DINOTransformer,
    DINOTransformerDecoder,
    DINOTransformerEncoder,
)

NUM_CLASSES = 10


class _FrozenBN(FrozenBatchNorm2d):

    def __init__(self, num_features, **kwargs):
        kwargs.pop("device", None)
        kwargs.pop("dtype", None)
        super().__init__(num_features)


class _TimmBackbone(TimmBackbone):

    @property
    def size_divisibility(self):
        return 32


class _DINO(DINO):

    def preprocess_image(self, batched_inputs):
        from detectron2.structures import ImageList

        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class _DINOWithNMS(_DINO):

    def __init__(self, *args, nms_iou_threshold=0.7, nms_max_per_image=100, nms_pre_topk=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_max_per_image = nms_max_per_image
        self.nms_pre_topk = nms_pre_topk

    def inference(self, box_cls, box_pred, image_sizes):
        assert len(box_cls) == len(image_sizes)
        results = []
        bs, n_queries, n_cls = box_cls.shape
        prob = box_cls.sigmoid()
        all_scores = prob.view(bs, n_queries * n_cls)
        all_indexes = torch.arange(n_queries * n_cls, device=box_cls.device).unsqueeze(0).expand(bs, -1)
        all_boxes_idx = torch.div(all_indexes, n_cls, rounding_mode="floor")
        all_labels = all_indexes % n_cls
        boxes_xyxy = box_cxcywh_to_xyxy(box_pred)
        boxes_exp = torch.gather(boxes_xyxy, 1, all_boxes_idx.unsqueeze(-1).expand(-1, -1, 4))
        for scores_i, labels_i, boxes_i, image_size in zip(all_scores, all_labels, boxes_exp, image_sizes):
            pre_topk = scores_i.topk(min(self.nms_pre_topk, scores_i.shape[0])).indices
            box = boxes_i[pre_topk]
            score = scores_i[pre_topk]
            label = labels_i[pre_topk]
            keep = batched_nms(box, score, label, self.nms_iou_threshold)[: self.nms_max_per_image]
            result = Instances(image_size)
            result.pred_boxes = Boxes(box[keep])
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = score[keep]
            result.pred_classes = label[keep]
            results.append(result)
        return results


def _sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t).pow(gamma) * ce).sum()


def _decode_anchor_boxes(anchors, deltas):
    wa = (anchors[:, 2] - anchors[:, 0]).clamp(min=1e-4)
    ha = (anchors[:, 3] - anchors[:, 1]).clamp(min=1e-4)
    cxa = anchors[:, 0] + wa * 0.5
    cya = anchors[:, 1] + ha * 0.5
    dx, dy, dw, dh = deltas.unbind(-1)
    pred_cx = dx * wa + cxa
    pred_cy = dy * ha + cya
    pred_w = wa * dw.clamp(max=4.0).exp()
    pred_h = ha * dh.clamp(max=4.0).exp()
    return torch.stack([pred_cx - pred_w * 0.5, pred_cy - pred_h * 0.5,
                        pred_cx + pred_w * 0.5, pred_cy + pred_h * 0.5], dim=-1)


class _AnchorHead(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int, num_convs: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        stem: List[nn.Module] = []
        for _ in range(num_convs):
            stem += [
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, in_channels),
                nn.ReLU(inplace=True),
            ]
        self.stem = nn.Sequential(*stem)
        self.cls_head = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.reg_head = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        prior = 0.01
        nn.init.normal_(self.cls_head.weight, std=0.01)
        nn.init.constant_(self.cls_head.bias, -math.log((1 - prior) / prior))
        nn.init.normal_(self.reg_head.weight, std=0.01)
        nn.init.zeros_(self.reg_head.bias)

    def forward(self, features: List[torch.Tensor]):
        cls_logits, bbox_preds = [], []
        for feat in features:
            x = self.stem(feat)
            cls_logits.append(self.cls_head(x))
            bbox_preds.append(self.reg_head(x))
        return cls_logits, bbox_preds


class _DINOWithAnchorHead(_DINOWithNMS):

    def __init__(self, anchor_head: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.anchor_head = anchor_head


        self.anchor_generator = DefaultAnchorGenerator(
            sizes=[[32], [64], [128], [256]],
            aspect_ratios=[[0.5, 1.0, 2.0]],
            strides=[8, 16, 32, 64],
        )
        self.anchor_matcher = Matcher(
            thresholds=[0.4, 0.5],
            labels=[0, -1, 1],
            allow_low_quality_matches=True,
        )
        self.anchor_cls_weight = 1.0
        self.anchor_box_weight = 2.0

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        batch_size, _, H, W = images.tensor.shape

        if self.training:
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for i in range(batch_size):
                ih, iw = batched_inputs[i]["instances"].image_size
                img_masks[i, :ih, :iw] = 0
        else:
            img_masks = images.tensor.new_zeros(batch_size, H, W)


        features = self.backbone(images.tensor)
        multi_level_feats = self.neck(features)


        if self.training:
            cls_logits, bbox_preds = self.anchor_head(multi_level_feats)


        multi_level_masks, multi_level_pos_embeds = [], []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_pos_embeds.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            gt_instances = None
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None

        (inter_states, init_reference, inter_references, enc_state, enc_reference) = \
            self.transformer(
                multi_level_feats, multi_level_masks, multi_level_pos_embeds,
                (input_query_label, input_query_bbox),
                attn_masks=[attn_mask, None],
            )
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        outputs_classes, outputs_coords = [], []
        for lvl in range(inter_states.shape[0]):
            ref = init_reference if lvl == 0 else inter_references[lvl - 1]
            ref = inverse_sigmoid(ref)
            out_cls = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if ref.shape[-1] == 4:
                tmp += ref
            else:
                tmp[..., :2] += ref
            outputs_classes.append(out_cls)
            outputs_coords.append(tmp.sigmoid())

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_meta)

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        output["enc_outputs"] = {
            "pred_logits": self.transformer.decoder.class_embed[-1](enc_state),
            "pred_boxes": enc_reference,
        }

        if self.training:
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict:
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            loss_dict.update(
                self._anchor_losses(cls_logits, bbox_preds, multi_level_feats, gt_instances)
            )
            return loss_dict
        else:
            results = self.inference(output["pred_logits"], output["pred_boxes"], images.image_sizes)
            processed = []
            for r, inp, img_sz in zip(results, batched_inputs, images.image_sizes):
                h = inp.get("height", img_sz[0])
                w = inp.get("width", img_sz[1])
                processed.append({"instances": detector_postprocess(r, h, w)})
            return processed

    def _anchor_losses(self, cls_logits, bbox_preds, feats, gt_instances):
        anchors_per_level = self.anchor_generator(feats)
        all_anchors = torch.cat([a.tensor for a in anchors_per_level])

        B = cls_logits[0].shape[0]
        na = self.anchor_head.num_anchors
        nc = self.num_classes
        flat_cls, flat_box = [], []
        for cls_l, box_l in zip(cls_logits, bbox_preds):
            _, _, H, W = cls_l.shape
            flat_cls.append(cls_l.permute(0, 2, 3, 1).reshape(B, H * W * na, nc))
            flat_box.append(box_l.permute(0, 2, 3, 1).reshape(B, H * W * na, 4))
        pred_cls = torch.cat(flat_cls, dim=1)
        pred_box = torch.cat(flat_box, dim=1)

        total_cls_loss = pred_cls.new_tensor(0.0)
        total_box_loss = pred_cls.new_tensor(0.0)
        num_pos = 0

        for i, gt_inst in enumerate(gt_instances):
            gt_boxes = gt_inst.gt_boxes.tensor
            gt_classes = gt_inst.gt_classes
            tgt_cls = torch.zeros((all_anchors.shape[0], nc), dtype=pred_cls.dtype,
                                  device=all_anchors.device)
            if gt_boxes.shape[0] > 0:
                iou_mat = pairwise_iou(Boxes(gt_boxes), Boxes(all_anchors))
                matched_idxs, anchor_labels = self.anchor_matcher(iou_mat)
                pos_mask = anchor_labels == 1
                if pos_mask.any():
                    tgt_cls[pos_mask, gt_classes[matched_idxs[pos_mask]]] = 1.0
                    decoded = _decode_anchor_boxes(all_anchors[pos_mask], pred_box[i][pos_mask])
                    decoded = torch.stack([
                        decoded[:, 0], decoded[:, 1],
                        torch.max(decoded[:, 2], decoded[:, 0] + 1e-3),
                        torch.max(decoded[:, 3], decoded[:, 1] + 1e-3),
                    ], dim=-1)
                    total_box_loss += _giou_loss(decoded, gt_boxes[matched_idxs[pos_mask]],
                                                 reduction="sum")
                    num_pos += pos_mask.sum().item()
                valid = anchor_labels != -1
                total_cls_loss += _sigmoid_focal_loss(pred_cls[i][valid], tgt_cls[valid])
            else:
                total_cls_loss += _sigmoid_focal_loss(pred_cls[i], tgt_cls)

        norm = max(num_pos, 1)
        return {
            "loss_anchor_cls": self.anchor_cls_weight * total_cls_loss / norm,
            "loss_anchor_box": self.anchor_box_weight * total_box_loss / norm,
        }


_EMBED_DIM = 256
_NUM_HEADS = 8
_FFN_DIM = 2048
_NUM_ENC_LAYERS = 6
_NUM_DEC_LAYERS = 6
_NUM_LEVELS = 4
_NUM_QUERIES = 900

model = L(_DINOWithAnchorHead)(
    anchor_head=L(_AnchorHead)(
        in_channels=_EMBED_DIM,
        num_classes=NUM_CLASSES,
        num_anchors=3,
        num_convs=3,
    ),
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
    transformer=L(DINOTransformer)(
        encoder=L(DINOTransformerEncoder)(
            embed_dim=_EMBED_DIM,
            num_heads=_NUM_HEADS,
            feedforward_dim=_FFN_DIM,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=_NUM_ENC_LAYERS,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=_EMBED_DIM,
            num_heads=_NUM_HEADS,
            feedforward_dim=_FFN_DIM,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=_NUM_DEC_LAYERS,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        num_feature_levels=_NUM_LEVELS,
        two_stage_num_proposals="${..num_queries}",
    ),
    embed_dim=_EMBED_DIM,
    num_classes=NUM_CLASSES,
    num_queries=_NUM_QUERIES,
    aux_loss=True,
    criterion=L(DINOCriterion)(
        num_classes="${..num_classes}",
        matcher=L(HungarianMatcher)(
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
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    device="cuda",
    input_format="RGB",
    vis_period=0,
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


            L(T.ResizeShortestEdge)(short_edge_length=1000, max_size=1344),
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
    output_dir="logs/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0",
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

        milestones=[202500, 243000],
        num_updates=270000,
    ),
    warmup_length=1000 / 270000,
    warmup_method="linear",
    warmup_factor=0.001,
)

train = dict(
    output_dir="logs/dino/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/0",
    init_checkpoint="",
    max_iter=270000,
    amp=dict(enabled=False),
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=True,
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
            name="dino_seresnextaa101d_exp0",
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
