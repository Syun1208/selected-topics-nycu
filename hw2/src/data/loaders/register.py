import logging
import os
from typing import Optional, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from src.data.models.annotation import DatasetMeta
from src.interfaces.base_dataset import BaseDataset

logger = logging.getLogger(__name__)

HW2_CLASS_NAMES: Tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")


def register_dataset(
    train_json: str,
    train_images_dir: str,
    valid_json: Optional[str] = None,
    valid_images_dir: Optional[str] = None,
    dataset_name_train: str = "train",
    dataset_name_valid: str = "valid",
    class_names: Tuple[str, ...] = HW2_CLASS_NAMES,
) -> DatasetMeta:
    assert os.path.isfile(train_json), f"train.json not found: {train_json }"
    assert os.path.isdir(train_images_dir), f"train images dir not found: {train_images_dir }"
    register_coco_instances(dataset_name_train, {}, train_json, train_images_dir)
    MetadataCatalog.get(dataset_name_train).set(thing_classes=list(class_names))
    logger.info("Registered training dataset '%s'", dataset_name_train)

    if valid_json and valid_images_dir:
        assert os.path.isfile(valid_json), f"valid.json not found: {valid_json }"
        assert os.path.isdir(valid_images_dir), f"valid images dir not found: {valid_images_dir }"
        register_coco_instances(dataset_name_valid, {}, valid_json, valid_images_dir)
        MetadataCatalog.get(dataset_name_valid).set(thing_classes=list(class_names))
        logger.info("Registered validation dataset '%s'", dataset_name_valid)

    return DatasetMeta(
        name="hw2",
        num_classes=len(class_names),
        class_names=list(class_names),
        train_json=train_json,
        train_images_dir=train_images_dir,
        valid_json=valid_json,
        valid_images_dir=valid_images_dir,
        register_train=dataset_name_train,
        register_valid=dataset_name_valid,
    )


class HW2Dataset(BaseDataset):

    def __init__(
        self,
        train_json: str,
        train_images_dir: str,
        valid_json: Optional[str] = None,
        valid_images_dir: Optional[str] = None,
        dataset_name_train: str = "train",
        dataset_name_valid: str = "valid",
    ) -> None:
        self.train_json = train_json
        self.train_images_dir = train_images_dir
        self.valid_json = valid_json
        self.valid_images_dir = valid_images_dir
        self.dataset_name_train = dataset_name_train
        self.dataset_name_valid = dataset_name_valid
        self._meta: Optional[DatasetMeta] = None

    def register(self) -> None:
        self._meta = register_dataset(
            train_json=self.train_json,
            train_images_dir=self.train_images_dir,
            valid_json=self.valid_json,
            valid_images_dir=self.valid_images_dir,
            dataset_name_train=self.dataset_name_train,
            dataset_name_valid=self.dataset_name_valid,
        )

    def get_dicts(self, split: str):
        from detectron2.data import DatasetCatalog

        if split == "train":
            return DatasetCatalog.get(self.dataset_name_train)
        elif split in ("valid", "val"):
            return DatasetCatalog.get(self.dataset_name_valid)
        else:
            raise ValueError(f"Unknown split '{split }'. Expected 'train' or 'valid'.")

    def get_metadata(self):
        return {
            "thing_classes": list(HW2_CLASS_NAMES),
            "num_classes": len(HW2_CLASS_NAMES),
            "dataset_name_train": self.dataset_name_train,
            "dataset_name_valid": self.dataset_name_valid,
        }
