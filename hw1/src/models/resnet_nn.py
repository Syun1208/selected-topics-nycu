from typing import Dict, Any

import timm
import torch
import torch.nn as nn


class ResNetNNClassifier(nn.Module):

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 100,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.backbone.requires_grad_(False)
        backbone_channels = self.backbone.num_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(backbone_channels, num_classes)

        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    @property
    def pretrained_cfg(self) -> Dict[str, Any]:
        return self.backbone.pretrained_cfg

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector before the classifier."""
        x = self.backbone.forward_features(x)  # [B, C, H, W]
        x = self.pool(x)                        # [B, C, 1, 1]
        return torch.flatten(x, 1)              # [B, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.fc(x)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        backbone: str,
        num_classes: int = 100,
        drop_rate: float = 0.0,
    ) -> "ResNetNNClassifier":
        model = cls(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
            drop_rate=drop_rate,
        )
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        return model


if __name__ == "__main__":
    model = ResNetNNClassifier(
        backbone="resnet50",
        num_classes=100,
        pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}  ({total / 1e6:.2f}M)")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
