from abc import ABC, abstractmethod
from typing import List

from data.dataclass import Prediction, TestConfig


class TesterInterface(ABC):
    """Abstract interface for model testers."""

    @abstractmethod
    def setup(self, config: TestConfig) -> None:
        """Initialize model and dataloader from config."""

    @abstractmethod
    def predict(self) -> List[Prediction]:
        """Run inference on the test set and return predictions."""

    @abstractmethod
    def save_submission(
        self, predictions: List[Prediction], output_path: str
    ) -> None:
        """Write predictions to a CSV file in submission format."""
