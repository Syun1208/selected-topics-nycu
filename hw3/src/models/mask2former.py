from typing import Any, Dict

NUM_CLASSES = 4

_SWIN_L_CKPT = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"


def build_mask2former(cfg: dict) -> dict:
    backbone_name = cfg.get("backbone", "swin_l").lower()
    num_classes = cfg.get("num_classes", NUM_CLASSES)

    # Pure instance segmentation: no stuff classes
    num_things_classes = num_classes
    num_stuff_classes = 0
    class_weight = [1.0] * num_things_classes + [0.1]  # +1 for background token

    if backbone_name == "swin_l":
        backbone = dict(
            type="SwinTransformer",
            pretrain_img_size=384,
            embed_dims=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=True,
            convert_weights=True,
            frozen_stages=-1,
            init_cfg=dict(type="Pretrained", checkpoint=_SWIN_L_CKPT),
        )
        in_channels = [192, 384, 768, 1536]
    else:
        raise ValueError(f"Unknown backbone '{backbone_name}' for Mask2Former. Choices: [swin_l]")

    model = dict(
        type="Mask2Former",
        backbone=backbone,
        panoptic_head=dict(
            type="Mask2FormerHead",
            in_channels=in_channels,
            strides=[4, 8, 16, 32],
            feat_channels=256,
            out_channels=256,
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            num_queries=100,
            num_transformer_feat_level=3,
            pixel_decoder=dict(
                type="MSDeformAttnPixelDecoder",
                num_outs=3,
                norm_cfg=dict(type="GN", num_groups=32),
                act_cfg=dict(type="ReLU"),
                encoder=dict(
                    num_layers=6,
                    layer_cfg=dict(
                        self_attn_cfg=dict(
                            embed_dims=256,
                            num_heads=8,
                            num_levels=3,
                            num_points=4,
                            dropout=0.0,
                            batch_first=True,
                        ),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=1024,
                            num_fcs=2,
                            ffn_drop=0.0,
                            act_cfg=dict(type="ReLU", inplace=True),
                        ),
                    ),
                ),
                positional_encoding=dict(num_feats=128, normalize=True),
            ),
            enforce_decoder_input_project=False,
            positional_encoding=dict(num_feats=128, normalize=True),
            transformer_decoder=dict(
                return_intermediate=True,
                num_layers=9,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, dropout=0.0, batch_first=True,
                    ),
                    cross_attn_cfg=dict(
                        embed_dims=256, num_heads=8, dropout=0.0, batch_first=True,
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            loss_cls=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=2.0,
                reduction="mean",
                class_weight=class_weight,
            ),
            loss_mask=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                reduction="mean",
                loss_weight=5.0,
            ),
            loss_dice=dict(
                type="DiceLoss",
                use_sigmoid=True,
                activate=True,
                reduction="mean",
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0,
            ),
        ),
        panoptic_fusion_head=dict(
            type="MaskFormerFusionHead",
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            loss_panoptic=None,
            init_cfg=None,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="HungarianAssigner",
                match_costs=[
                    dict(type="ClassificationCost", weight=2.0),
                    dict(type="CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                    dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type="MaskPseudoSampler"),
        ),
        test_cfg=dict(
            panoptic_on=False,
            semantic_on=False,
            instance_on=True,
            max_per_image=100,
            iou_thr=0.8,
            filter_low_score=True,
        ),
        init_cfg=None,
    )
    return model
