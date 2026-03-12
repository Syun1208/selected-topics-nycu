from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTester(ABC):

    @abstractmethod
    def setup(self, cfg: Any) -> None: ...

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]: ...

    @abstractmethod
    def predict(self, output_dir: str) -> None: ...

    @abstractmethod
    def submission(self, output_dir: str) -> None: ...

    @abstractmethod
    def save_results(self, results: Dict[str, Any], output_dir: str) -> None: ...
