from typing import Dict, Any, Optional

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


def _make_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class NeuralNetwork(nn.Module):
    """
    Custom image classifier with ~5.3M parameters.

    Architecture: 4-stage residual network with channels [44, 88, 176, 352].
    Designed for 100-class image classification.

    Args:
        num_classes: Number of output classes.
        drop_rate: Dropout rate before the classifier head.
    """

    def __init__(self, num_classes: int = 100, drop_rate: float = 0.0) -> None:
        super().__init__()
        self._num_classes = num_classes

        # Stem: 3 -> 44, stride 2 + maxpool -> /4 spatial
        self.stem = nn.Sequential(
            nn.Conv2d(3, 44, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(44),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = _make_layer(44, 44, num_blocks=2, stride=1)   # /4
        self.layer2 = _make_layer(44, 88, num_blocks=2, stride=2)   # /8
        self.layer3 = _make_layer(88, 176, num_blocks=2, stride=2)  # /16
        self.layer4 = _make_layer(176, 352, num_blocks=2, stride=2) # /32

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(352, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.dropout(x)
        return self.fc(x)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, num_classes: int = 100, drop_rate: float = 0.0
    ) -> "NeuralNetwork":
        model = cls(num_classes=num_classes, drop_rate=drop_rate)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        return model


if __name__ == "__main__":
    model = NeuralNetwork(num_classes=100)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total:,}  ({total / 1e6:.2f}M)")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
