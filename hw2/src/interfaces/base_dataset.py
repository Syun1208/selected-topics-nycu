from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDataset(ABC):

    @abstractmethod
    def register(self) -> None: ...

    @abstractmethod
    def get_dicts(self, split: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]: ...
