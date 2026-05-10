from typing import Any, Dict

from src.models.cascade_mask_rcnn import (
    _backbone_resnext101,
    _backbone_swin_b,
    _backbone_swin_l,
    _backbone_convnext_l,
    _fpn_neck,
    _rpn_head,
    _CASCADE_BBOX_STAGES,
    _CASCADE_TEST_CFG,
)

NUM_CLASSES = 4


def _htc_roi_head(num_classes: int) -> dict:
    bbox_head = [
        dict(
            type="Shared2FCBBoxHead",
            in_channels=256, fc_out_channels=1024, roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0, 0, 0, 0],
                target_stds=[s, s, s * 2, s * 2],
            ),
            reg_class_agnostic=True,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="SmoothL1Loss", beta=1.0, loss_weight=1.0),
        )
        for s in [0.1, 0.05, 0.033]
    ]


    mask_head = [
        dict(
            type="HTCMaskHead",
            with_conv_res=False,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
        dict(
            type="HTCMaskHead",
            with_conv_res=True,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
        dict(
            type="HTCMaskHead",
            with_conv_res=True,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        ),
    ]


    semantic_head = dict(
        type="FusedSemanticHead",
        num_ins=5,
        fusion_level=1,
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=num_classes + 1,
        ignore_label=255,
        loss_seg=dict(type="CrossEntropyLoss", ignore_index=255, loss_weight=0.2),
    )

    return dict(
        type="HybridTaskCascadeRoIHead",
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        semantic_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8],
        ),
        semantic_head=semantic_head,
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        bbox_head=bbox_head,
        mask_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
        ),
        mask_head=mask_head,
        train_cfg=dict(rcnn=_CASCADE_BBOX_STAGES),
        test_cfg=_CASCADE_TEST_CFG["rcnn"],
    )


_BACKBONE_BUILDERS = {
    "resnext101": _backbone_resnext101,
    "swin_b": _backbone_swin_b,
    "swin_l": _backbone_swin_l,
    "convnext_l": _backbone_convnext_l,
}


def build_htc(cfg: dict) -> dict:
    backbone_name = cfg.get("backbone", "resnext101").lower()
    num_classes = cfg.get("num_classes", NUM_CLASSES)

    if backbone_name not in _BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone '{backbone_name}'. "
                         f"Choices: {list(_BACKBONE_BUILDERS.keys())}")

    bb = _BACKBONE_BUILDERS[backbone_name]()

    model = dict(
        type="HybridTaskCascade",
        **bb,
        rpn_head=_rpn_head(),
        roi_head=_htc_roi_head(num_classes),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                              min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5,
                             neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=0, pos_weight=-1, debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=1000, max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0,
            ),
            rcnn=_CASCADE_BBOX_STAGES,
        ),
        test_cfg=_CASCADE_TEST_CFG,
    )
    return model
