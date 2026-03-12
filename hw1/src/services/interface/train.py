from abc import ABC, abstractmethod

from data.dataclass import EpochResult, TrainConfig


class TrainerInterface(ABC):
    """Abstract interface for model trainers."""

    @abstractmethod
    def setup(self, config: TrainConfig) -> None:
        """Initialize model, optimizer, dataloaders, and scheduler."""

    @abstractmethod
    def train_epoch(self, epoch: int) -> EpochResult:
        """Run one full training + validation epoch and return metrics."""

    @abstractmethod
    def train(self) -> None:
        """Run the full training loop for all epochs."""

    @abstractmethod
    def save_checkpoint(
        self, epoch: int, val_acc: float, is_best: bool
    ) -> None:
        """Persist model weights and training state to disk."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint and return the epoch to resume from."""
