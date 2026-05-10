from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class InstanceAnnotation:
    category_id: int
    instance_id: int
    mask: np.ndarray
    bbox: Tuple[float, float, float, float]
    area: float


@dataclass
class TrainSample:
    image_id: str
    image: np.ndarray
    height: int
    width: int
    instances: List[InstanceAnnotation] = field(default_factory=list)


@dataclass
class TestSample:
    image_id: int
    file_name: str
    image: np.ndarray
    height: int
    width: int


@dataclass
class PredictionResult:
    image_id: int
    category_id: int
    bbox: List[float]
    score: float
    segmentation: dict
