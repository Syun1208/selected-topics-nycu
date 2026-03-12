from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class LoraConfig:
    enabled: bool = False
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: ["Linear"])
    save_lora_only: bool = False


@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 100
    drop_rate: float = 0.2
    checkpoint: Optional[str] = None
    lora: LoraConfig = field(default_factory=LoraConfig)


@dataclass
class DataConfig:
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"
    image_size: int = 224
    resize_size: int = 334
    crop_size: int = 320
    batch_size: int = 64
    num_workers: int = 4
    use_augmentation: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 50
    lr: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    mixup_alpha: float = 0.0
    accumulation_steps: int = 1
    use_class_weights: bool = True
    use_weighted_sampler: bool = True
    freeze_batch_normalize: bool = False


@dataclass
class OutputConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_top_k: int = 3
    submission_file: str = "submission.csv"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class TestConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class BatchResult:
    loss: float
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float


@dataclass
class Prediction:
    image_name: str
    label: int


@dataclass
class AugmentationJob:
    class_name: str
    src_dir: Path
    dst_dir: Path
    target: int
    n_add: int


@dataclass
class PathConfig:
    train_dir: Path = field(default_factory=lambda: Path(
        __file__).resolve().parents[2] / "data" / "train_strong_previous")
    aug_dir: Path = field(default_factory=lambda: Path(
        __file__).resolve().parents[2] / "data" / "train_strong")
