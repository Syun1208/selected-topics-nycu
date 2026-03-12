from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BBox:

    x: float
    y: float
    width: float
    height: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.width, self.height]

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Annotation:

    id: int
    image_id: int
    category_id: int
    bbox: BBox
    area: float
    iscrowd: int = 0
    segmentation: List = field(default_factory=list)


@dataclass
class ImageMeta:

    id: int
    file_name: str
    height: int
    width: int


@dataclass
class CategoryMeta:

    id: int
    name: str
    supercategory: str = "digit"


@dataclass
class DatasetMeta:

    name: str
    num_classes: int
    class_names: List[str]
    train_json: str
    train_images_dir: str
    valid_json: Optional[str] = None
    valid_images_dir: Optional[str] = None
    test_images_dir: Optional[str] = None
    register_train: str = "hw2_train"
    register_valid: str = "hw2_valid"
