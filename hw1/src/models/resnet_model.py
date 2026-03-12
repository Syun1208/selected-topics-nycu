from typing import List, Dict, Any

import timm
import torch
import torch.nn as nn


class ResNetClassifier(nn.Module):

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 100,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings before the classifier head."""
        return self.model.forward_features(x)

    @property
    def num_classes(self) -> int:
        return self.model.num_classes

    @property
    def pretrained_cfg(self) -> Dict[str, Any]:
        return self.model.pretrained_cfg

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, backbone: str, num_classes: int
    ) -> "ResNetClassifier":
        model = cls(
            backbone=backbone, num_classes=num_classes, pretrained=False
        )
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        return model
