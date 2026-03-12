import torch.nn as nn

from data.dataclass import ModelConfig
from models.lora_model import apply_lora
from models.neural_network import NeuralNetwork
from models.resnet_model import ResNetClassifier
from models.resnet_nn import ResNetNNClassifier

_PREFIX_RESNET_NN = "resnet_nn:"
_KEY_NEURAL_NETWORK = "neural_network"


class ModelFactory:

    @classmethod
    def create(cls, cfg: ModelConfig) -> nn.Module:
        if cfg.backbone == _KEY_NEURAL_NETWORK:
            model = NeuralNetwork(
                num_classes=cfg.num_classes,
                drop_rate=cfg.drop_rate,
            )
        elif cfg.backbone.startswith(_PREFIX_RESNET_NN):
            backbone_name = cfg.backbone[len(_PREFIX_RESNET_NN):]
            model = ResNetNNClassifier(
                backbone=backbone_name,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                drop_rate=cfg.drop_rate,
            )
        else:
            model = ResNetClassifier(
                backbone=cfg.backbone,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                drop_rate=cfg.drop_rate,
            )

        if cfg.lora.enabled:
            # Freeze all parameters first, then LoRA will unfreeze A/B
            # matrices.
            for param in model.parameters():
                param.requires_grad_(False)
            apply_lora(
                model,
                rank=cfg.lora.rank,
                alpha=cfg.lora.alpha,
                dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
            )

        return model
