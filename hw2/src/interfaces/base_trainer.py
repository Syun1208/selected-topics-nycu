from abc import ABC, abstractmethod
from typing import Any


class BaseTrainer(ABC):

    @abstractmethod
    def setup(self, cfg: Any, resume: bool = False) -> None: ...

    @abstractmethod
    def train(self) -> None: ...

    @abstractmethod
    def save_checkpoint(self, output_dir: str, iteration: int) -> None: ...
