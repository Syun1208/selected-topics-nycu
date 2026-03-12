from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import albumentations as A
from dataclass import AugmentationJob, PathConfig
from typing import Callable, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
import shutil
import random
import argparse
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class AugmentationStrategy(ABC):
    @abstractmethod
    def augment(self, image_rgb: np.ndarray) -> np.ndarray: ...


class AlbumentationsStrategy(AugmentationStrategy):
    def augment(self, image_rgb: np.ndarray) -> np.ndarray:
        pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15,
                rotate_limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.7,
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=30,
                    val_shift_limit=20),
                A.CLAHE(clip_limit=4.0),
                A.RGBShift(
                    r_shift_limit=20,
                    g_shift_limit=20,
                    b_shift_limit=20),
            ], p=0.8),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
                A.MedianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(10.0, 50.0)),
            ], p=0.4),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3),
                A.ElasticTransform(alpha=1, sigma=20),
                A.OpticalDistortion(distort_limit=0.2),
            ], p=0.4),
            A.CoarseDropout(
                max_holes=6, min_holes=1,
                max_height=30, min_height=10,
                max_width=30, min_width=10,
                fill_value=0, p=0.5,
            ),
            A.RandomShadow(p=0.2),
            A.RandomFog(fog_coef_range=(0.1, 0.25), p=0.1),
        ])
        return pipeline(image=image_rgb)["image"]


class TorchvisionStrategy(AugmentationStrategy):
    def __init__(self, apply_probability: float = 0.7):
        self._prob = apply_probability

    def augment(self, image_rgb: np.ndarray) -> np.ndarray:
        if random.random() >= self._prob:
            return image_rgb

        policies = [
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.TrivialAugmentWide(),
            transforms.AugMix(severity=3, mixture_width=3),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        ]
        pil_img = Image.fromarray(image_rgb)
        pil_img = random.choice(policies)(pil_img)
        return np.array(pil_img)


class CompositeAugmentationStrategy(AugmentationStrategy):
    def __init__(self, *strategies: AugmentationStrategy):
        self._strategies = strategies

    def augment(self, image_rgb: np.ndarray) -> np.ndarray:
        arr = image_rgb
        for strategy in self._strategies:
            arr = strategy.augment(arr)
            arr = self._to_rgb(arr)
        return arr

    @staticmethod
    def _to_rgb(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        if arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr


def build_default_augmentor() -> AugmentationStrategy:
    return CompositeAugmentationStrategy(
        TorchvisionStrategy(apply_probability=0.7),
        AlbumentationsStrategy(),
    )


class ImageRepository:
    def __init__(self, image_exts: frozenset[str] = IMAGE_EXTS):
        self._exts = image_exts

    def load_rgb(self, path: Path) -> Optional[np.ndarray]:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def save_rgb(self, arr: np.ndarray, path: Path) -> None:
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    def list_images(self, directory: Path) -> list[Path]:
        return [p for p in directory.iterdir() if p.suffix.lower()
                in self._exts]

    def count_images(self, directory: Path) -> int:
        return sum(1 for p in directory.iterdir()
                   if p.suffix.lower() in self._exts)

    def count_per_class(self, root: Path) -> dict[str, int]:
        return {
            d.name: self.count_images(d)
            for d in sorted(root.iterdir())
            if d.is_dir()
        }

    def copy_all(self, src_dir: Path, dst_dir: Path) -> int:
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for src in src_dir.iterdir():
            if src.suffix.lower() not in self._exts:
                continue
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        return copied


def _worker_process_class(job: AugmentationJob) -> Tuple[str, int]:
    repo = ImageRepository()
    augmentor = build_default_augmentor()

    job.dst_dir.mkdir(parents=True, exist_ok=True)
    src_files = repo.list_images(job.src_dir)
    if not src_files:
        return job.class_name, 0

    for f in src_files:
        shutil.copy2(f, job.dst_dir / f.name)

    existing = len(src_files)
    needed = max(0, job.target - existing)
    generated = _generate_augmented(
        repo,
        augmentor,
        src_files,
        job.dst_dir,
        existing,
        needed)
    return job.class_name, generated


def _worker_add_to_class(job: AugmentationJob) -> Tuple[str, int]:
    repo = ImageRepository()
    augmentor = build_default_augmentor()

    job.dst_dir.mkdir(parents=True, exist_ok=True)
    src_files = repo.list_images(job.src_dir)
    if not src_files:
        return job.class_name, 0

    existing_count = repo.count_images(job.dst_dir)
    generated = _generate_augmented(
        repo, augmentor, src_files, job.dst_dir, existing_count, job.n_add,
        random_suffix=True,
    )
    return job.class_name, generated


def _generate_augmented(
    repo: ImageRepository,
    augmentor: AugmentationStrategy,
    src_files: list[Path],
    dst_dir: Path,
    offset: int,
    count: int,
    random_suffix: bool = False,
) -> int:
    generated = 0
    while generated < count:
        src = random.choice(src_files)
        image_rgb = repo.load_rgb(src)
        if image_rgb is None:
            continue
        aug = augmentor.augment(image_rgb)
        name = _make_name(src, offset + generated, random_suffix)
        repo.save_rgb(aug, dst_dir / name)
        generated += 1
    return generated


def _make_name(src: Path, index: int, random_suffix: bool) -> str:
    if random_suffix:
        return f"aug_{
            index:05d}_{
            src.stem}_x{
            random.randint(
                0,
                9999):04d}{
                    src.suffix}"
    return f"aug_{index:05d}_{src.name}"


class AugmentationOrchestrator:
    def __init__(self, n_workers: int):
        self._n_workers = n_workers

    def run(
        self,
        jobs: list[AugmentationJob],
        worker_fn: Callable[[AugmentationJob], Tuple[str, int]],
        desc: str = "Classes",
    ) -> int:
        total = 0
        with ProcessPoolExecutor(max_workers=self._n_workers) as pool:
            futures = {pool.submit(worker_fn, job): job for job in jobs}
            with tqdm(total=len(jobs), desc=desc) as pbar:
                for fut in as_completed(futures):
                    cls, n = fut.result()
                    total += n
                    pbar.set_postfix(last=cls, gen=total)
                    pbar.update(1)
        return total


class Command(ABC):
    @abstractmethod
    def execute(self) -> None: ...


class CountCommand(Command):
    def __init__(self, directory: Path, repo: ImageRepository):
        self._directory = directory
        self._repo = repo

    def execute(self) -> None:
        counts = self._repo.count_per_class(self._directory)
        for cls, n in counts.items():
            print(f"{cls}: {n}")
        values = list(counts.values())
        print(f"\nTotal classes : {len(counts)}")
        print(f"Total images  : {sum(values)}")
        print(f"Min / Max     : {min(values)} / {max(values)}")


class MergeCommand(Command):
    def __init__(self, paths: PathConfig, repo: ImageRepository):
        self._paths = paths
        self._repo = repo

    def execute(self) -> None:
        if not self._paths.aug_dir.exists():
            print(
                f"{self._paths.aug_dir} does not exist. "
                "Run augmentation first."
            )
            return
        copied = 0
        for class_dir in tqdm(
                sorted(
                    self._paths.train_dir.iterdir()),
                desc="Merging"):
            if class_dir.is_dir():
                copied += self._repo.copy_all(
                    class_dir,
                    self._paths.aug_dir / class_dir.name
                )
        print(
            f"Merged {copied} new files from {
                self._paths.train_dir} into {
                self._paths.aug_dir}.")


class AddCommand(Command):
    def __init__(
            self,
            paths: PathConfig,
            orchestrator: AugmentationOrchestrator,
            n_add: int):
        self._paths = paths
        self._orchestrator = orchestrator
        self._n_add = n_add

    def execute(self) -> None:
        class_names = [
            d.name for d in self._paths.train_dir.iterdir() if d.is_dir()]
        print(
            f"Adding {
                self._n_add} augmented images per class into {
                self._paths.aug_dir}")
        print(
            f"Classes: {
                len(class_names)}, Workers: {
                self._orchestrator._n_workers}")

        jobs = [
            AugmentationJob(
                class_name=cls,
                src_dir=self._paths.train_dir / cls,
                dst_dir=self._paths.aug_dir / cls,
                target=0,
                n_add=self._n_add,
            )
            for cls in class_names
        ]
        total = self._orchestrator.run(jobs, _worker_add_to_class)
        print(f"\nDone. Generated {total} new images.")


class ProcessCommand(Command):
    def __init__(
        self,
        paths: PathConfig,
        repo: ImageRepository,
        orchestrator: AugmentationOrchestrator,
        target: int,
        resume: bool,
    ):
        self._paths = paths
        self._repo = repo
        self._orchestrator = orchestrator
        self._target = target
        self._resume = resume

    def execute(self) -> None:
        class_names = [
            d.name for d in self._paths.train_dir.iterdir() if d.is_dir()]
        class_counts = {
            cls: self._repo.count_images(self._paths.train_dir / cls)
            for cls in class_names
        }
        target = self._target if self._target > 0 else max(
            class_counts.values())

        pending = self._resolve_pending(class_names)
        if pending is None:
            return

        self._print_summary(class_counts, target)

        jobs = [
            AugmentationJob(
                class_name=cls,
                src_dir=self._paths.train_dir / cls,
                dst_dir=self._paths.aug_dir / cls,
                target=target,
                n_add=0,
            )
            for cls in pending
        ]
        total_generated = self._orchestrator.run(jobs, _worker_process_class)

        total_after = sum(
            self._repo.count_images(
                self._paths.aug_dir /
                cls) for cls in class_names)
        print(f"\nDone. Generated {total_generated} new images.")
        print(f"Total images in {self._paths.aug_dir}: {total_after}")

    def _resolve_pending(self, class_names: list[str]) -> Optional[list[str]]:
        if self._resume:
            pending = [
                cls for cls in class_names if not (
                    self._paths.aug_dir /
                    cls).exists()]
            print(
                f"Resuming: {
                    len(class_names) -
                    len(pending)} done, {
                    len(pending)} remaining")
            return pending

        if self._paths.aug_dir.exists():
            n_existing = sum(
                1 for p in self._paths.aug_dir.rglob("*") if p.is_file())
            answer = input(
                f"WARNING: This will DELETE {self._paths.aug_dir} "
                f"({n_existing} files) and recreate it. "
                f"Use --add N to append instead. Continue? [y/N] "
            )
            if answer.strip().lower() != "y":
                print("Aborted.")
                return None
            shutil.rmtree(self._paths.aug_dir)

        self._paths.aug_dir.mkdir(parents=True)
        return class_names

    def _print_summary(
            self, class_counts: dict[str, int], target: int) -> None:
        values = list(class_counts.values())
        print(f"Classes      : {len(class_counts)}")
        print(f"Min / Max    : {min(values)} / {max(values)}")
        print(f"Original dir : {self._paths.train_dir}")
        print(f"Target/class : {target}")
        print(f"Output dir   : {self._paths.aug_dir}")
        print(f"Workers      : {self._orchestrator._n_workers}")


_COUNT_SENTINEL = Path("__count_default__")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=int, default=0,
        help="Target images per class (0 = auto: use class with most images)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge data/train into data/train_strong (skip existing files)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip classes that already exist in train_strong",
    )
    parser.add_argument(
        "--count", type=Path, metavar="DIR", nargs="?",
        const=_COUNT_SENTINEL, default=None,
        help="Count images per class in DIR (default: AUG_DIR) and exit",
    )
    parser.add_argument(
        "--add",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Add N more augmented images per class into existing "
            "AUG_DIR (no deletion)"
        ),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    paths = PathConfig()
    repo = ImageRepository()
    orchestrator = AugmentationOrchestrator(n_workers=args.workers)

    if args.count is not None:
        count_dir = (
            paths.aug_dir if args.count is _COUNT_SENTINEL else args.count
        )
        command: Command = CountCommand(count_dir, repo)
    elif args.merge:
        command = MergeCommand(paths, repo)
    elif args.add > 0:
        command = AddCommand(paths, orchestrator, args.add)
    else:
        command = ProcessCommand(
            paths,
            repo,
            orchestrator,
            args.target,
            args.resume)

    command.execute()


if __name__ == "__main__":
    main()
