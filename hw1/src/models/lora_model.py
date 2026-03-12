import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Frozen pre-trained weight (not updated during LoRA training)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(
            dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> "LoRALinear":
        lora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        lora.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora.bias.data.copy_(linear.bias.data)
        return lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(
            F.linear(
                self.lora_dropout(x),
                self.lora_A),
            self.lora_B)
        return base + lora * self.scaling

    def merge_weights(self) -> nn.Linear:
        merged = nn.Linear(
            self.in_features, self.out_features, bias=self.bias is not None
        )
        merged.weight.data = self.weight.data + \
            self.scaling * (self.lora_B @ self.lora_A)
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.3f}"
        )


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: List[str],
) -> nn.Module:
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and type(
                child).__name__ in target_modules:
            setattr(
                model,
                name,
                LoRALinear.from_linear(
                    child,
                    rank,
                    alpha,
                    dropout))
        else:
            apply_lora(child, rank, alpha, dropout, target_modules)
    return model


def get_lora_state_dict(model: nn.Module) -> dict:
    return {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }


def count_parameters(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
