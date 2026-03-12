import logging
from typing import List, Optional, Tuple, Union

import detectron2.data.transforms as T

logger = logging.getLogger(__name__)


def build_train_augmentation(
    short_edge_lengths: Tuple[int, ...] = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
    max_size: int = 1333,
    use_crop: bool = True,
    crop_short_edge_lengths: Tuple[int, ...] = (400, 500, 600),
    crop_size: Tuple[int, int] = (384, 600),
) -> List[T.Transform]:
    augmentation = [
        T.RandomFlip(),
        T.ResizeShortestEdge(
            short_edge_length=list(short_edge_lengths),
            max_size=max_size,
            sample_style="choice",
        ),
    ]
    return augmentation


def build_train_augmentation_with_crop(
    short_edge_lengths: Tuple[int, ...] = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
    max_size: int = 1333,
    crop_short_edge_lengths: Tuple[int, ...] = (400, 500, 600),
    crop_size: Tuple[int, int] = (384, 600),
) -> List[T.Transform]:
    return [
        T.RandomFlip(),
        T.ResizeShortestEdge(
            short_edge_length=list(crop_short_edge_lengths),
            sample_style="choice",
        ),
        T.RandomCrop(
            crop_type="absolute_range",
            crop_size=list(crop_size),
        ),
        T.ResizeShortestEdge(
            short_edge_length=list(short_edge_lengths),
            max_size=max_size,
            sample_style="choice",
        ),
    ]


def build_test_augmentation(
    short_edge_length: int = 800,
    max_size: int = 1333,
) -> List[T.Transform]:
    return [
        T.ResizeShortestEdge(
            short_edge_length=short_edge_length,
            max_size=max_size,
        ),
    ]
