from __future__ import annotations

import gc
import os
import sys
from typing import Callable, Optional

import cv2
import numpy as np
import torch

from data import FilePath, resolve_model_path
from matchers.base import DetectorFreeMatcher
from matchers.config import GlueStickConfig
from preprocesses.orientation import OrientationNormalizer
from preprocesses.region import OverlapRegionCropper
from storage import MatchedKeypointStorage
from workspace import log

# The GlueStick repo is vendored under models/gluestick/ with an inner
# ``gluestick`` package that uses absolute imports (``from gluestick import ...``),
# so its root must be importable on sys.path.
_GLUESTICK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "gluestick"
)
if _GLUESTICK_DIR not in sys.path:
    sys.path.insert(0, _GLUESTICK_DIR)


def read_image(path: FilePath) -> np.ndarray:
    return cv2.imread(str(path))


class GlueStickMatcher(DetectorFreeMatcher):
    def __init__(self, conf: GlueStickConfig, device: Optional[torch.device] = None):
        from gluestick.models.two_view_pipeline import TwoViewPipeline

        self.conf = conf
        self.device = device or torch.device("cuda")

        gs_weights = str(resolve_model_path(conf.weight_path))
        sp_params = {
            "force_num_keypoints": False,
            "max_num_keypoints": conf.max_num_keypoints,
        }
        if conf.sp_weight_path is not None:
            sp_params["weights"] = str(resolve_model_path(conf.sp_weight_path))

        pipe_conf = {
            "name": "two_view_pipeline",
            "use_lines": True,
            "extractor": {
                "name": "wireframe",
                "sp_params": sp_params,
                "wireframe_params": {
                    "merge_points": True,
                    "merge_line_endpoints": True,
                },
                "max_n_lines": conf.max_n_lines,
            },
            "matcher": {
                "name": "gluestick",
                "weights": gs_weights,
                "trainable": False,
            },
            "ground_truth": {"from_pose_depth": False},
        }
        self.model = TwoViewPipeline(pipe_conf).to(self.device).eval()
        log(f"[GlueStickMatcher] loaded weights={conf.weight_path} ({conf})")

    def _to_gray_resized(self, img_bgr: np.ndarray):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        L = self.conf.resize_long_edge
        if L and max(H, W) > L:
            scale = L / float(max(H, W))
            new_w, new_h = int(round(W * scale)), int(round(H * scale))
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return gray, (new_w / float(W), new_h / float(H))
        return gray, (1.0, 1.0)

    @torch.inference_mode()
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
        from gluestick import batch_to_np, numpy_image_to_torch

        img1, img2 = image_reader(str(path1)), image_reader(str(path2))

        ori1 = OrientationNormalizer.create_if_needed(orientation1)
        ori2 = OrientationNormalizer.create_if_needed(orientation2)

        if cropper:
            cropper.set_original_image(img1, img2)
            img1, img2 = cropper.crop_ndarray_image(img1, img2)
        if ori1:
            ori1.set_original_image(img1)
            img1 = ori1.get_upright_image_ndarray()
        if ori2:
            ori2.set_original_image(img2)
            img2 = ori2.get_upright_image_ndarray()

        g1, (sx1, sy1) = self._to_gray_resized(img1)
        g2, (sx2, sy2) = self._to_gray_resized(img2)
        t1 = numpy_image_to_torch(g1).to(self.device)[None]
        t2 = numpy_image_to_torch(g2).to(self.device)[None]

        pred = batch_to_np(self.model({"image0": t1, "image1": t2}))
        kp0, kp1, m0 = pred["keypoints0"], pred["keypoints1"], pred["matches0"]
        valid = m0 != -1
        mkpts1 = kp0[valid].astype(np.float64)
        mkpts2 = kp1[m0[valid]].astype(np.float64)

        if "matching_scores0" in pred:
            scores = np.asarray(pred["matching_scores0"])[valid].astype(np.float64)
        else:
            scores = np.ones(len(mkpts1), dtype=np.float64)

        # rescale matched keypoints back to the (cropped/uprighted) image coords
        if len(mkpts1):
            mkpts1[:, 0] /= sx1
            mkpts1[:, 1] /= sy1
            mkpts2[:, 0] /= sx2
            mkpts2[:, 1] /= sy2

        order = np.argsort(-scores)
        mkpts1, mkpts2, scores = mkpts1[order], mkpts2[order], scores[order]
        if self.conf.topk is not None:
            k = self.conf.topk
            mkpts1, mkpts2, scores = mkpts1[:k], mkpts2[:k], scores[:k]

        if ori1:
            mkpts1 = ori1.keypoints_to_original_coords_ndarray(mkpts1)
        if ori2:
            mkpts2 = ori2.keypoints_to_original_coords_ndarray(mkpts2)
        if cropper:
            mkpts1, mkpts2 = cropper.convert_cropped_to_original_coordinates(
                mkpts1, mkpts2
            )

        if self.conf.min_matches is None or len(mkpts1) >= self.conf.min_matches:
            matched_keypoint_storage.add(path1, path2, mkpts1, mkpts2, scores=scores)

        # GlueStick leaks ~100 MB/pair (C-level); free what we can each call.
        del t1, t2, pred, kp0, kp1, m0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
