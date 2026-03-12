from typing import Any, Dict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data.transforms_factory import create_transform
from timm.data.config import resolve_model_data_config


def get_train_transforms_v1(**kwargs: Dict[str, Any]) -> transforms.Compose:
    image_size = kwargs.get("image_size", 224)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

def get_train_transforms(**kwargs: Dict[str, Any]) -> transforms.Compose:
    model = kwargs.get("model")
    model_config = resolve_model_data_config(model=model)
    return create_transform(**model_config, is_training=True)

def get_val_transforms_v1(**kwargs: Dict[str, Any]) -> transforms.Compose:
    resize_size = kwargs.get("resize_size", 256)
    crop_size = kwargs.get("crop_size", 224)
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(**kwargs: Dict[str, Any]) -> transforms.Compose:
    model = kwargs.get("model")
    model_config = resolve_model_data_config(model=model)
    return create_transform(**model_config, is_training=False)

class ClassificationDataset(Dataset):
    """Dataset for labeled image classification (train/val)."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        class_dirs = sorted(self.root.iterdir(), key=lambda p: int(p.name))
        self.classes = [d.name for d in class_dirs if d.is_dir()]
        self.class_to_idx = {cls: int(cls) for cls in self.classes}

        for class_dir in class_dirs:
            if not class_dir.is_dir():
                continue
            label = int(class_dir.name)
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    """Dataset for unlabeled test images."""

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.image_paths: List[Path] = sorted(
            p for p in self.root.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # return image tensor and filename without extension
        return image, img_path.stem
