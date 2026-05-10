from typing import Any, Dict


NUM_CLASSES = 4
_CASCADE_BBOX_STAGES = [
    dict(
        assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5,
                      min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
        sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25,
                     neg_pos_ub=-1, add_gt_as_proposals=True),
        mask_size=28, pos_weight=-1, debug=False,
    ),
    dict(
        assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.6, neg_iou_thr=0.6,
                      min_pos_iou=0.6, match_low_quality=False, ignore_iof_thr=-1),
        sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25,
                     neg_pos_ub=-1, add_gt_as_proposals=True),
        mask_size=28, pos_weight=-1, debug=False,
    ),
    dict(
        assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.7,
                      min_pos_iou=0.7, match_low_quality=False, ignore_iof_thr=-1),
        sampler=dict(type="RandomSampler", num=256, pos_fraction=0.25,
                     neg_pos_ub=-1, add_gt_as_proposals=True),
        mask_size=28, pos_weight=-1, debug=False,
    ),
]

_CASCADE_TEST_CFG = dict(
    rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100, mask_thr_binary=0.5,
    ),
)


def _fpn_neck(in_channels):
    return dict(
        type="FPN",
        in_channels=in_channels,
        out_channels=256,
        num_outs=5,
    )


def _rpn_head():
    return dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8], ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
        ),
        bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                        target_means=[0, 0, 0, 0], target_stds=[1, 1, 1, 1]),
        loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
    )


def _cascade_roi_head(num_classes: int):
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
    mask_head = dict(
        type="FCNMaskHead",
        num_convs=4, in_channels=256, conv_out_channels=256,
        num_classes=num_classes,
        loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
    )
    return dict(
        type="CascadeRoIHead",
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
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


def _backbone_resnext101() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="ResNeXt",
            depth=101,
            groups=64,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            with_cp=True,
            norm_cfg=dict(type="BN", requires_grad=True),
            style="pytorch",
            init_cfg=dict(
                type="Pretrained",
                checkpoint="open-mmlab://resnext101_64x4d",
            ),
        ),
        neck=_fpn_neck([256, 512, 1024, 2048]),
    )


_SWIN_B_CKPT = (
    "/home/longpm/.cache/torch/hub/checkpoints/swin_base_patch4_window7_224_22k_backbone.pth"
    if __import__("os").path.exists(
        "/home/longpm/.cache/torch/hub/checkpoints/swin_base_patch4_window7_224_22k_backbone.pth"
    ) else
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth"
)

_SWIN_L_CKPT = (
    "/home/longpm/.cache/torch/hub/checkpoints/swin_large_patch4_window7_224_22k_backbone.pth"
    if __import__("os").path.exists(
        "/home/longpm/.cache/torch/hub/checkpoints/swin_large_patch4_window7_224_22k_backbone.pth"
    ) else
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
)


def _backbone_swin_b() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="SwinTransformer",
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=True,
            init_cfg=dict(
                type="Pretrained",
                checkpoint=_SWIN_B_CKPT,
            ),
        ),
        neck=_fpn_neck([128, 256, 512, 1024]),
    )


def _backbone_swin_l() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="SwinTransformer",
            pretrain_img_size=224,
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=True,
            init_cfg=dict(
                type="Pretrained",
                checkpoint=_SWIN_L_CKPT,
            ),
        ),
        neck=_fpn_neck([192, 384, 768, 1536]),
    )


def _backbone_convnext_l() -> Dict[str, Any]:
    return dict(
        backbone=dict(
            type="mmpretrain.ConvNeXt",
            arch="large",
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.4,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            with_cp=True,
            init_cfg=dict(
                type="Pretrained",
                checkpoint="https://download.openmmlab.com/mmclassification/v0/convnext/"
                           "convnext-large_3rdparty_in21k_20220124-41b5a79f.pth",
                prefix="backbone.",
            ),
        ),
        neck=dict(
            type="FPN",
            in_channels=[192, 384, 768, 1536],
            out_channels=256,
            num_outs=5,
        ),
    )


def _backbone_x101_64x4d() -> Dict[str, Any]:
    """X-101-64x4d without init_cfg — weights come from a full cascade_rcnn checkpoint."""
    return dict(
        backbone=dict(
            type="ResNeXt",
            depth=101,
            groups=64,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            with_cp=True,
            norm_cfg=dict(type="BN", requires_grad=True),
            style="pytorch",
        ),
        neck=_fpn_neck([256, 512, 1024, 2048]),
    )


_BACKBONE_BUILDERS = {
    "resnext101": _backbone_resnext101,
    "x101_64x4d": _backbone_x101_64x4d,
    "swin_b": _backbone_swin_b,
    "swin_l": _backbone_swin_l,
    "convnext_l": _backbone_convnext_l,
}


def build_cascade_mask_rcnn(cfg: dict) -> dict:
    backbone_name = cfg.get("backbone", "resnext101").lower()
    num_classes = cfg.get("num_classes", NUM_CLASSES)

    if backbone_name not in _BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone '{backbone_name}'. "
                         f"Choices: {list(_BACKBONE_BUILDERS.keys())}")

    bb = _BACKBONE_BUILDERS[backbone_name]()

    model = dict(
        type="CascadeRCNN",
        **bb,
        rpn_head=_rpn_head(),
        roi_head=_cascade_roi_head(num_classes),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(type="MaxIoUAssigner", pos_iou_thr=0.7, neg_iou_thr=0.3,
                              min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
                sampler=dict(type="RandomSampler", num=512, pos_fraction=0.5,
                             neg_pos_ub=-1, add_gt_as_proposals=False),
                allowed_border=0, pos_weight=-1, debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=2000, max_per_img=2000,
                nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0,
            ),
            rcnn=_CASCADE_BBOX_STAGES,
        ),
        test_cfg=_CASCADE_TEST_CFG,
    )
    return model
