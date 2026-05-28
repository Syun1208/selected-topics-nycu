from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
import torch

from data import FilePath
from matchers.base import DetectorFreeMatcher
from matchers.config import CascadeMatcherConfig
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage
from workspace import log


def read_image(path: FilePath) -> np.ndarray:
    return cv2.imread(str(path))


class CascadeMatcher(DetectorFreeMatcher):
    """Run a fast matcher first; escalate to a heavy matcher only on hard pairs.

    Easy pairs (>= ``escalate_threshold`` matches from the fast matcher) keep the
    fast matches. Hard pairs are re-matched with the heavy matcher, whose result
    overwrites the pair's entry in the storage.
    """

    def __init__(self, conf: CascadeMatcherConfig, device: Optional[torch.device] = None):
        # Imported lazily to avoid a circular import with matchers.factory.
        from matchers.factory import create_detector_free_matcher

        self.conf = conf
        self.fast = create_detector_free_matcher(conf.fast, device=device)
        self.heavy = create_detector_free_matcher(conf.heavy, device=device)
        log(
            f"[CascadeMatcher] fast={conf.fast.type} -> heavy={conf.heavy.type} "
            f"(escalate when < {conf.escalate_threshold} matches)"
        )

    def __call__(
        self,
        path1: FilePath,
        path2: FilePath,
        matched_keypoint_storage: MatchedKeypointStorage,
        cropper: Optional[OverlapRegionCropper] = None,
        orientation1: Optional[int] = None,
        orientation2: Optional[int] = None,
        image_reader: Callable = read_image,
    ):
        kwargs = dict(
            cropper=cropper,
            orientation1=orientation1,
            orientation2=orientation2,
            image_reader=image_reader,
        )

        # 1) Fast matcher on every pair.
        self.fast(path1, path2, matched_keypoint_storage, **kwargs)

        n_fast = 0
        if matched_keypoint_storage.has(path1, path2):
            mk1, _ = matched_keypoint_storage.get(path1, path2)
            n_fast = len(mk1)

        # 2) Escalate hard pairs to the heavy matcher (overwrites the entry).
        if n_fast < self.conf.escalate_threshold:
            self.heavy(path1, path2, matched_keypoint_storage, **kwargs)
            n_heavy = 0
            if matched_keypoint_storage.has(path1, path2):
                mk1, _ = matched_keypoint_storage.get(path1, path2)
                n_heavy = len(mk1)
            log(
                f"[CascadeMatcher] HARD {path1}|{path2}: fast={n_fast} "
                f"-> heavy={n_heavy}"
            )
