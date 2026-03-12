from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseModelWrapper(ABC):

    @abstractmethod
    def build(self, cfg: Any) -> nn.Module: ...

    @abstractmethod
    def load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> nn.Module: ...

    @abstractmethod
    def get_lazy_config(self) -> Any: ...
