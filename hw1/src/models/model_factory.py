import torch.nn as nn

from data.dataclass import ModelConfig
from models.neural_network import NeuralNetwork
from models.resnet_model import ResNetClassifier
from models.resnet_nn import ResNetNNClassifier

_PREFIX_RESNET_NN = "resnet_nn:"
_KEY_NEURAL_NETWORK = "neural_network"


class ModelFactory:
    """
    Factory for creating model instances from a ModelConfig.

    Backbone naming convention:
        "neural_network"         → NeuralNetwork (custom CNN, no pretrained backbone)
        "resnet_nn:<timm_name>"  → ResNetNNClassifier (pretrained backbone + NN head)
        "<timm_name>"            → ResNetClassifier (standard timm ResNet)
    """

    @classmethod
    def create(cls, cfg: ModelConfig) -> nn.Module:
        if cfg.backbone == _KEY_NEURAL_NETWORK:
            return NeuralNetwork(
                num_classes=cfg.num_classes,
                drop_rate=cfg.drop_rate,
            )
        elif cfg.backbone.startswith(_PREFIX_RESNET_NN):
            backbone_name = cfg.backbone[len(_PREFIX_RESNET_NN):]
            return ResNetNNClassifier(
                backbone=backbone_name,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                drop_rate=cfg.drop_rate,
            )
        else:
            return ResNetClassifier(
                backbone=cfg.backbone,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                drop_rate=cfg.drop_rate,
            )
