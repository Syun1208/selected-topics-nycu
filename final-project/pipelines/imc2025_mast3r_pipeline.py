from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pandas as pd
import pycolmap
import torch
import tqdm

from clusterings.factory import MASt3RFPSClustering, create_clustering
from colmap import (
    _cam_from_world,
    get_image_id_of_scene_graph_center,
    get_outlier_reconstructions,
    import_into_colmap,
)
from data import DEFAULT_OUTLIER_SCENE_NAME, SAVE_CAMERA_DEBUG_INFO, set_random_seed
from data_schema import DataSchema
from distributed import DistConfig
from matchers.base import run_overlap_region_estimation
from matchers.factory import create_point_tracking_matcher
from matchers.mast3r import MASt3RMatcher
from matchers.mast3r_c2f import MASt3RC2FMatcher
from matchers.mast3r_hybrid import MASt3RHybridMatcher
from models.mast3r.encoder_cache import MASt3REncoderCache
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict_with_scene_clustering,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import IMC2025MASt3RPipelineConfig
from pipelines.matching import run_point_tracking_matching
from pipelines.snapshot import SceneSnapshot
from pipelines.verification import (
    compute_ransac_inlier_counts,
    run_ransac,
    verify_matches,
)
from preprocesses.region import OverlapRegionEstimator
from shortlists.factory import create_shortlist_generator
from storage import (
    InMemoryKeypointStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
)
from utils.camvis import save_camera_debug_info
from utils.imc25.metric import register_by_Horn
from workspace import log


_INCREMENTAL_MAPPER_SCRIPT = (
    Path(__file__).resolve().parent.parent / "tools" / "run_incremental_mapper.py"
)


def _run_incremental_mapping_in_subprocess(
    database_path: str,
    image_path: str,
    output_path: str,
    option_overrides: dict,
    glog_minloglevel: Optional[int] = None,
) -> dict:
    """Run ``pycolmap.incremental_mapping`` out-of-process and reload the
    resulting reconstructions.

    Done out-of-process because the pyceres BA callback can segfault on
    ``PyErr_Fetch`` from a non-Python thread; running in a subprocess isolates
    the crash from the main pipeline.

    ``glog_minloglevel`` (if set) is exported as ``GLOG_minloglevel`` for the
    subprocess to quieten the noisy Ceres "Linear solver failure" WARNING spam
    (2 = hide INFO+WARNING, keep ERROR/FATAL).
    """
    import json
    import os
    import sys

    Path(output_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_INCREMENTAL_MAPPER_SCRIPT),
        "--database-path", str(database_path),
        "--image-path", str(image_path),
        "--output-path", str(output_path),
        "--options-json", json.dumps(option_overrides),
    ]
    env = None
    if glog_minloglevel is not None:
        env = dict(os.environ)
        env["GLOG_minloglevel"] = str(glog_minloglevel)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(
            f"[incremental_mapping] subprocess returned {proc.returncode} "
            "-> treating as no reconstructions"
        )
        return {}

    maps: dict = {}
    out_dir = Path(output_path)
    if not out_dir.exists():
        return maps
    for sub in sorted(out_dir.iterdir(), key=lambda p: p.name):
        if not sub.is_dir() or not sub.name.isdigit():
            continue
        try:
            maps[int(sub.name)] = pycolmap.Reconstruction(str(sub))
        except Exception as e:
            print(f"[incremental_mapping] failed to load {sub}: {e}")
    return maps


def _one_to_one(idxs: np.ndarray) -> np.ndarray:
    """Keep only mutual (one-to-one) matches.

    Given match index pairs ``(a, b)`` (keypoint ``a`` in image1 <-> keypoint
    ``b`` in image2), greedily keep the first occurrence of each ``a`` and each
    ``b`` so the resulting assignment is bijective. MASt3R sparse matches are
    already reciprocal-NN; this mainly removes conflicts introduced when dense
    and sparse matches are merged.
    """
    seen0: set[int] = set()
    seen1: set[int] = set()
    keep = []
    for a, b in idxs:
        a = int(a)
        b = int(b)
        if a in seen0 or b in seen1:
            continue
        seen0.add(a)
        seen1.add(b)
        keep.append((a, b))
    if not keep:
        return idxs[:0]
    return np.asarray(keep, dtype=idxs.dtype).reshape(-1, 2)


class IMC2025MASt3RPipeline(Pipeline):
    def __init__(
        self,
        conf: IMC2025MASt3RPipelineConfig,
        dist_conf: Optional[DistConfig] = None,
        device: Optional[torch.device] = None,
    ):
        set_random_seed(seed=conf.seed)
        dist_conf = dist_conf or DistConfig.single()
        device = device or torch.device("cpu")

        self.dist_conf = dist_conf
        self.device = device
        self.conf = conf

        # Clustering
        clustering = create_clustering(conf.clustering, device=device)
        assert isinstance(clustering, MASt3RFPSClustering)
        self.clustering: MASt3RFPSClustering = clustering

        # Shortlist
        self.shortlist_generator_in_clustering = None
        if self.conf.shortlist_generator_in_clustering:
            self.shortlist_generator_in_clustering = create_shortlist_generator(
                self.conf.shortlist_generator_in_clustering, device=device
            )
        self.shortlist_generator = create_shortlist_generator(
            self.conf.shortlist_generator, device=device
        )

        # Matchers
        self.matcher = MASt3RMatcher(conf.matcher, device=device)
        self.c2f_matcher = None
        if conf.matcher_c2f:
            self.c2f_matcher = MASt3RC2FMatcher(conf.matcher_c2f, device=device)
        self.hybrid_matcher = None
        if conf.matcher_hybrid:
            self.hybrid_matcher = create_point_tracking_matcher(
                conf.matcher_hybrid, device=self.device
            )

        # Preprocessors
        self.overlap_region_estimator = None
        if conf.overlap_region_estimation:
            self.overlap_region_estimator = OverlapRegionEstimator(
                conf.overlap_region_estimation
            )

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("IMC2025MASt3RPipeline starts")

        data_dict = create_data_dict(data_schema, df=df, ignore_gt_scene_label=True)
        # results, num_scenes = init_result_dict(data_dict)
        results = init_result_dict_with_scene_clustering(data_dict)
        log(f"The data list has been loaded. # of datasets: {len(results)}")

        iterator = iterate_scenes(data_dict, data_schema)
        progress_bar = tqdm.tqdm(
            total=len(results),
            desc="IMC2025MASt3RPipeline",
            disable=self.dist_conf.is_slave(),
        )

        # NOTE
        # Iterate run_scene() over "datasets" because "scenes" means "datasets" in IMC2025
        seen_datasets = set()
        for scene in iterator:
            if seen_datasets and scene.dataset not in seen_datasets:
                progress_bar.update(1)
            seen_datasets.add(scene.dataset)
            progress_bar.set_description(
                f"IMC2025MASt3RPipeline::{scene.dataset} ({len(seen_datasets)}/{len(results)})"
            )

            assert isinstance(scene, Scene)
            with scene.create_space() as scene:
                outputs = self.run_scene(
                    scene, progress_bar, save_snapshot=save_snapshot
                )
                results[scene.dataset] = outputs
        progress_bar.update(1)

        df = results_to_submission_df(results, schema="imc2025")
        return df

    def run_scene(
        self, scene: Scene, iterator: tqdm.tqdm, save_snapshot: bool = False
    ) -> dict:
        scene.cache_all_images()

        # Per-scene MASt3R encoder cache shared across clustering + matching.
        # Each unique image is encoded once and reused across every pair it
        # appears in; freed before moving to the next scene to avoid GPU OOM.
        encoder_cache = MASt3REncoderCache()
        self._set_encoder_cache(encoder_cache)
        try:
            return self._run_scene_inner(scene, iterator, save_snapshot)
        finally:
            log(f"[{scene.dataset}] MASt3R encoder cache stats: {encoder_cache.stats()}")
            encoder_cache.release()
            self._set_encoder_cache(None)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _set_encoder_cache(self, cache: Optional[MASt3REncoderCache]) -> None:
        for m in (
            self.matcher,
            self.hybrid_matcher,
            getattr(self.clustering, "matcher", None),
        ):
            if m is not None and hasattr(m, "set_encoder_cache"):
                m.set_encoder_cache(cache)

    def _coarse_verify_prune(
        self,
        clustered_scene: Scene,
        pairs: list[tuple[int, int]],
        image_reader,
        iterator: tqdm.tqdm,
        label: str = "",
    ) -> list[tuple[int, int]]:
        """Coarse geometric verification of shortlist edges.

        Matches every shortlist pair with the coarse MASt3R matcher
        (``self.matcher``), counts RANSAC inliers, and drops edges with fewer
        than ``coarse_verification.min_inliers`` inliers. Runs before the
        expensive hybrid matcher + COLMAP so weak edges never reach
        reconstruction.
        """
        assert self.conf.coarse_verification is not None
        cv_conf = self.conf.coarse_verification

        cv_mkpt = InMemoryMatchedKeypointStorage()
        for i, (idx1, idx2) in enumerate(pairs):
            iterator.set_postfix_str(f"[{label}] coarse verify ({i + 1}/{len(pairs)})")
            path1 = clustered_scene.image_paths[idx1]
            path2 = clustered_scene.image_paths[idx2]
            cropper = clustered_scene.create_overlap_region_cropper(
                path1, path2, cropper_type=self.conf.cropper_type
            )
            self.matcher(
                path1,
                path2,
                cv_mkpt,
                cropper=cropper,
                image_reader=image_reader,
            )

        kp_storage = InMemoryKeypointStorage()
        m_storage = InMemoryMatchingStorage()
        cv_mkpt.to_keypoints_and_matches(
            keypoint_storage=kp_storage,
            matching_storage=m_storage,
            apply_round=self.conf.round_matched_keypoints,
        )
        scores = compute_ransac_inlier_counts(
            kp_storage, m_storage, cv_conf.ransac, progress_bar=iterator
        )

        kept: list[tuple[int, int]] = []
        for idx1, idx2 in pairs:
            k1 = Path(clustered_scene.image_paths[idx1]).name
            k2 = Path(clustered_scene.image_paths[idx2]).name
            n = scores.get(k1, {}).get(k2)
            if n is None:
                n = scores.get(k2, {}).get(k1, 0)
            if n >= cv_conf.min_inliers:
                kept.append((idx1, idx2))

        log(
            f"[{label}] coarse verification: kept {len(kept)}/{len(pairs)} edges "
            f"(min_inliers={cv_conf.min_inliers}), dropped {len(pairs) - len(kept)}"
        )
        return kept

    def _prune_matches_before_colmap(
        self,
        clustered_scene: Scene,
        keypoint_storage: InMemoryKeypointStorage,
        matching_storage: InMemoryMatchingStorage,
        iterator: tqdm.tqdm,
        label: str = "",
    ) -> InMemoryMatchingStorage:
        """Filter raw matches before COLMAP insertion.

        For every image pair: optionally keep mutual (one-to-one) matches, then
        run RANSAC/MAGSAC++ and keep only geometric inlier matches. Pairs left
        with fewer than ``match_prune.min_inliers`` matches are dropped.
        """
        assert self.conf.match_prune is not None
        conf = self.conf.match_prune

        k_storage = keypoint_storage.to_memory()
        m_storage = matching_storage.to_memory()
        new_storage = InMemoryMatchingStorage()

        n_matches_before = 0
        n_matches_after = 0
        n_pairs_before = 0
        n_pairs_after = 0
        count = 0
        for key1, group in m_storage:
            for key2, idxs in group.items():
                count += 1
                iterator.set_postfix_str(f"[{label}] match prune ({count})")
                n_pairs_before += 1
                idxs = np.asarray(idxs)
                n_matches_before += len(idxs)
                if len(idxs) == 0:
                    continue

                if conf.mutual_check:
                    idxs = _one_to_one(idxs)

                try:
                    mkpts0 = k_storage.keypoints[key1][idxs[:, 0]].copy()
                    mkpts1 = k_storage.keypoints[key2][idxs[:, 1]].copy()
                    _, inliers = run_ransac(mkpts0, mkpts1, conf.ransac)
                    inliers = np.asarray(inliers).reshape(-1)
                    if len(inliers) == len(idxs):
                        idxs = idxs[inliers > 0]
                    else:
                        # RANSAC could not be run (too few matches) -> drop pair
                        idxs = idxs[:0]
                except Exception as e:
                    print(f"[match_prune] Error on ({key1}, {key2}): {e}")
                    continue

                if len(idxs) < conf.min_inliers:
                    continue

                if key1 not in new_storage.matches:
                    new_storage.matches[key1] = {}
                new_storage.matches[key1][key2] = idxs.copy()
                n_pairs_after += 1
                n_matches_after += len(idxs)

        log(
            f"[{label}] match prune: pairs {n_pairs_before}->{n_pairs_after}, "
            f"matches {n_matches_before}->{n_matches_after} "
            f"(mutual={conf.mutual_check}, min_inliers={conf.min_inliers}, "
            f"method={conf.ransac.method})"
        )
        return new_storage

    def _grid_filter_matches(
        self,
        keypoint_storage: InMemoryKeypointStorage,
        matching_storage: InMemoryMatchingStorage,
        image_shapes: dict,
        label: str = "",
    ) -> InMemoryMatchingStorage:
        """Grid-based spatial filtering of matches.

        Divide image A into ``grid_size`` x ``grid_size`` cells, bucket each
        match by the cell of its keypoint in image A, and keep at most
        ``top_n_per_cell`` matches per cell. Matches are score-ordered by the
        matcher, so this keeps the top-N best per cell.
        """
        assert self.conf.grid_filter is not None
        G = max(1, self.conf.grid_filter.grid_size)
        topn = max(1, self.conf.grid_filter.top_n_per_cell)

        # image basename -> (H, W)
        shape_by_name = {Path(p).name: hw for p, hw in image_shapes.items()}

        k_storage = keypoint_storage.to_memory()
        m_storage = matching_storage.to_memory()
        new_storage = InMemoryMatchingStorage()

        n_before = 0
        n_after = 0
        for key1, group in m_storage:
            kptsA = k_storage.keypoints.get(key1)
            hwA = shape_by_name.get(key1)
            for key2, idxs in group.items():
                idxs = np.asarray(idxs)
                n_before += len(idxs)
                if len(idxs) == 0:
                    continue
                if kptsA is None or hwA is None:
                    # cannot grid-filter (missing shape) -> keep as is
                    new_storage.matches.setdefault(key1, {})[key2] = idxs.copy()
                    n_after += len(idxs)
                    continue

                H, W = float(hwA[0]), float(hwA[1])
                cw = W / G
                ch = H / G
                counts: dict[int, int] = {}
                keep = []
                for mi in range(len(idxs)):
                    x, y = kptsA[int(idxs[mi, 0])]
                    cx = min(int(x / cw), G - 1) if cw > 0 else 0
                    cy = min(int(y / ch), G - 1) if ch > 0 else 0
                    cell = cx * G + cy
                    c = counts.get(cell, 0)
                    if c < topn:
                        counts[cell] = c + 1
                        keep.append(mi)
                kept = idxs[keep]
                if len(kept) > 0:
                    new_storage.matches.setdefault(key1, {})[key2] = kept.copy()
                    n_after += len(kept)

        log(
            f"[{label}] grid filter ({G}x{G}, top_n={topn}): "
            f"matches {n_before}->{n_after}"
        )
        return new_storage

    def _run_scene_inner(
        self, scene: Scene, iterator: tqdm.tqdm, save_snapshot: bool = False
    ) -> dict:
        pairs_for_clustering = None
        if self.shortlist_generator_in_clustering:
            pairs_for_clustering = self.shortlist_generator_in_clustering(
                scene, progress_bar=iterator
            )

        clustering_result = self.clustering.run(
            scene.image_paths,
            image_reader=scene.get_image,
            pre_computed_pairs=pairs_for_clustering,
            neighbor_metric=self.conf.clustering_neighbor_metric,
        )
        pre_mkpt_storage = cast(
            InMemoryMatchedKeypointStorage,
            clustering_result.get_output("matched_keypoint_storage"),
        )
        pairwise_scores = cast(
            np.ndarray, clustering_result.get_output("pairwise_score")
        )

        clustered_scenes = clustering_result.to_scene_list(
            scene.dataset, scene.data_schema
        )
        num_clusters = len(clustered_scenes)
        clustered_scene_results = []
        for cluster_idx, clustered_scene in enumerate(clustered_scenes):
            scene.make_output_dir_for_child_scene(clustered_scene)
            mkpt_storage = pre_mkpt_storage.clone_subset(clustered_scene.image_paths)

            pairwise_scores_in_cluster = focus_on_cluster(
                clustered_scene, pairwise_scores
            )
            pairs = self.make_pairs(
                clustered_scene,
                pairwise_scores_in_cluster,
                basic_pair_topk=None,
                iterator=iterator,
            )
            log(
                f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                f"# of pairs: {len(pairs)}"
            )

            if self.conf.coarse_verification is not None:
                pairs = self._coarse_verify_prune(
                    clustered_scene,
                    pairs,
                    image_reader=scene.get_image,
                    iterator=iterator,
                    label=f"{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}",
                )

            if self.overlap_region_estimator:
                run_overlap_region_estimation(
                    self.overlap_region_estimator,
                    pairs,
                    clustered_scene,
                    matched_keypoint_storage=mkpt_storage,
                    progress_bar=iterator,
                )
                clustered_scene.make_roi_from_overlap_regions()

            if self.conf.matching_stage_mode == "complementary":
                # Mode: "complementary"
                # ---------------------
                # 1. If a pair has matched keypoints from the clustering stage, re-use the results
                # 2. Otherwise, MASt3RMatcher computes the matching between idx1 and idx2 images
                # Required matchers:
                #   - MASt3RMatcher
                for i, (idx1, idx2) in enumerate(pairs):
                    iterator.set_postfix_str(
                        f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                        f"MASt3R matching ({i + 1}/{len(pairs)})"
                    )
                    path1 = clustered_scene.image_paths[idx1]
                    path2 = clustered_scene.image_paths[idx2]
                    if mkpt_storage.has(path1, path2):
                        continue
                    cropper = clustered_scene.create_overlap_region_cropper(
                        path1, path2, cropper_type=self.conf.cropper_type
                    )
                    self.matcher(
                        path1,
                        path2,
                        mkpt_storage,
                        cropper=cropper,
                        image_reader=scene.get_image,
                    )

                keypoint_storage = InMemoryKeypointStorage()
                matching_storage = InMemoryMatchingStorage()
                mkpt_storage.to_keypoints_and_matches(
                    keypoint_storage=keypoint_storage,
                    matching_storage=matching_storage,
                    apply_round=self.conf.round_matched_keypoints,
                )
            elif self.conf.matching_stage_mode == "c2f_override":
                # Mode: "c2f_override"
                # -------------------------------
                # 1. Matched keypoints from the clustering stage will not be used
                # 2. A pair that has matched keypoints from the clustering stage
                #      -> MASt3RC2FMatcher
                # 3. A pair that does not have matched keypoints from the clustering stage
                #      -> MASt3RMatcher
                # Required matchers:
                #   - MASt3RMatcher
                #   - MASt3RC2FMatcher
                for i, (idx1, idx2) in enumerate(pairs):
                    iterator.set_postfix_str(
                        f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                        f"MASt3R matching ({i + 1}/{len(pairs)})"
                    )
                    path1 = clustered_scene.image_paths[idx1]
                    path2 = clustered_scene.image_paths[idx2]

                    H1, W1 = scene.image_shapes[str(path1)]
                    H2, W2 = scene.image_shapes[str(path2)]
                    enable_c2f = 512 < max(H1, W1) or 512 < max(H2, W2)
                    if mkpt_storage.has(path1, path2) and enable_c2f:
                        assert self.c2f_matcher is not None
                        self.c2f_matcher(
                            path1,
                            path2,
                            mkpt_storage,
                            image_reader=scene.get_image,
                        )
                    else:
                        cropper = clustered_scene.create_overlap_region_cropper(
                            path1, path2, cropper_type=self.conf.cropper_type
                        )
                        self.matcher(
                            path1,
                            path2,
                            mkpt_storage,
                            cropper=cropper,
                            image_reader=scene.get_image,
                        )

                keypoint_storage = InMemoryKeypointStorage()
                matching_storage = InMemoryMatchingStorage()
                mkpt_storage.to_keypoints_and_matches(
                    keypoint_storage=keypoint_storage,
                    matching_storage=matching_storage,
                    apply_round=self.conf.round_matched_keypoints,
                )
            elif self.conf.matching_stage_mode == "hybrid_matcher_override":
                # Mode: "hybrid_matcher_override"
                # -------------------------------
                # 1. Matched keypoints from the clustering stage will not be used
                # 2. MASt3RHybridMatcher computes matchings for each pair
                # Required matchers:
                #   - MASt3RHybridMatcher
                assert self.hybrid_matcher is not None
                assert self.conf.matcher_hybrid
                _mk_storage = InMemoryMatchedKeypointStorage()  # No used
                keypoint_storage = InMemoryKeypointStorage()
                matching_storage = InMemoryMatchingStorage()
                run_point_tracking_matching(
                    self.hybrid_matcher,
                    pairs,
                    clustered_scene,
                    keypoint_storage,
                    matching_storage,
                    _mk_storage,
                    impl_version=self.conf.matcher_hybrid.impl_version,
                    apply_round=self.conf.matcher_hybrid.apply_round,
                    mkpts_decoupling_method="imc2023",
                    matching_filter_conf=self.conf.matcher_hybrid.matching_filter,
                    progress_bar=iterator,
                )
            else:
                raise ValueError(self.conf.matching_stage_mode)

            # Filter matches (mutual check + MAGSAC++) BEFORE inserting into COLMAP
            if self.conf.match_prune is not None:
                matching_storage = self._prune_matches_before_colmap(
                    clustered_scene,
                    keypoint_storage,
                    matching_storage,
                    iterator=iterator,
                    label=f"{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}",
                )

            # Grid-based spatial filtering of matches BEFORE inserting into COLMAP
            if self.conf.grid_filter is not None:
                matching_storage = self._grid_filter_matches(
                    keypoint_storage,
                    matching_storage,
                    image_shapes=scene.image_shapes,
                    label=f"{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}",
                )

            # Add keypoints and matches into COLMAP DB
            database_path = str(clustered_scene.database_path)
            log(
                f"[{clustered_scene.scene}; {cluster_idx + 1}/{num_clusters}] "
                f"COLMAP database path: {database_path}"
            )
            id_mappings = import_into_colmap(
                clustered_scene,
                keypoint_storage,
                matching_storage,
                database_path=database_path,
                camera_model=self.conf.reconstruction.get_camera_model(
                    unique_resolution_num=clustered_scene.get_unique_resolution_num()
                ),
            )

            if len(keypoint_storage.keypoints) == 0:
                # Avoid COLMAP errors
                print("Outlier scene")
                cluster_results, _ = get_outlier_reconstructions(clustered_scene)
                clustered_scene_results.append(cluster_results)
                continue

            if len(matching_storage.matches) == 0:
                # Avoid COLMAP errors
                print("Outlier scene")
                cluster_results, _ = get_outlier_reconstructions(clustered_scene)
                clustered_scene_results.append(cluster_results)
                continue

            # Add two-view geometry into COLMAP DB
            g_storage = verify_matches(
                clustered_scene,
                self.conf.verification,
                keypoint_storage=keypoint_storage,
                matching_storage=matching_storage,
                id_mappings=id_mappings,
                progress_bar=iterator,
            )

            if save_snapshot:
                SceneSnapshot(
                    clustered_scene,
                    keypoint_storage,
                    matching_storage,
                    two_view_geometry_storage=g_storage,
                ).save(pipeline_id=self.pipeline_id)

            maps: dict[int, pycolmap.Reconstruction] = {}
            if self.conf.use_glomap:
                args = [
                    "glomap",
                    "mapper",
                    "--database_path",
                    database_path,
                    "--image_path",
                    str(scene.image_dir),
                    "--output_path",
                    str(scene.reconstruction_dir),
                ]
                glomap_process = subprocess.Popen(args)
                glomap_process.wait()

                if glomap_process.returncode != 0:
                    print(
                        f"Subprocess Error (Return code: {glomap_process.returncode} )"
                    )
                else:
                    maps = {
                        0: pycolmap.Reconstruction(str(scene.reconstruction_dir / "0"))
                    }

            if len(maps) == 0:
                # NOTE
                # (From https://www.kaggle.com/code/eduardtrulls/imc-2023-submission-example/notebook)
                # By default colmap does not generate a reconstruction
                # if less than 10 images are registered. Lower it to 3.
                #
                # We collect IncrementalPipelineOptions as a flat dict and run
                # pycolmap.incremental_mapping in a subprocess. The pyceres BA
                # callback can segfault on PyErr_Fetch from a non-Python
                # thread; isolating it in a subprocess prevents that crash from
                # bringing down the whole pipeline.
                option_overrides: dict = {
                    "num_threads": 1,
                    "min_model_size": (
                        self.conf.reconstruction.mapper_min_model_size or 3
                    ),
                }
                if self.conf.reconstruction.mapper_max_num_models is not None:
                    option_overrides["max_num_models"] = (
                        self.conf.reconstruction.mapper_max_num_models
                    )
                if self.conf.reconstruction.mapper_multiple_models is not None:
                    option_overrides["multiple_models"] = (
                        self.conf.reconstruction.mapper_multiple_models
                    )
                if self.conf.reconstruction.mapper_min_num_matches is not None:
                    option_overrides["min_num_matches"] = (
                        self.conf.reconstruction.mapper_min_num_matches
                    )
                if self.conf.reconstruction.mapper_filter_max_reproj_error is not None:
                    option_overrides["mapper.filter_max_reproj_error"] = (
                        self.conf.reconstruction.mapper_filter_max_reproj_error
                    )
                if self.conf.reconstruction.set_scene_graph_center_node_to_init_image_id1:
                    image_id1 = get_image_id_of_scene_graph_center(
                        clustered_scene, database_path=database_path
                    )
                    if image_id1 is not None:
                        option_overrides["init_image_id1"] = image_id1

                maps = _run_incremental_mapping_in_subprocess(
                    database_path=database_path,
                    image_path=str(clustered_scene.image_dir),
                    output_path=str(clustered_scene.reconstruction_dir),
                    option_overrides=option_overrides,
                    glog_minloglevel=self.conf.reconstruction.mapper_glog_minloglevel,
                )

            cluster_results, clustered_scene_infos = get_best_reconstruction(
                maps, clustered_scene
            )

            if SAVE_CAMERA_DEBUG_INFO:
                print(clustered_scene_infos)
                save_camera_debug_info(
                    cluster_results,
                    clustered_scene,
                    Path(f"extra/camvis/{self.pipeline_id}"),
                    prefix_dict=clustered_scene_infos["localization_by"],
                )

            clustered_scene_results.append(cluster_results)

        scene.release_all()

        outputs = {}
        image_count = 0
        for clustered_scene, cluster_results in zip(
            clustered_scenes, clustered_scene_results
        ):
            image_count += len(list(cluster_results.keys()))
            outputs[clustered_scene.scene] = copy.deepcopy(cluster_results)

        assert image_count == len(scene.image_paths)
        return outputs

    def make_pairs(
        self,
        scene: Scene,
        pairwise_scores: np.ndarray,
        basic_pair_topk: int | None = None,
        iterator: tqdm.tqdm | None = None,
    ) -> list[tuple[int, int]]:
        assert len(scene.image_paths) == len(pairwise_scores)
        assert scene.indices_in_parent_scene is not None

        basic_pairs = make_pairs_from_pairwise_scores(
            pairwise_scores, topk=basic_pair_topk
        )
        suppl_pairs = self.shortlist_generator(
            scene,
            progress_bar=iterator,
        )

        stats = {
            "basic_pair_count": len(basic_pairs),
            "suppl_pair_cand_count": len(suppl_pairs),
            "added_pair_count_from_suppl_pairs": 0,
        }

        pairs = set(basic_pairs)
        for pair in suppl_pairs:
            if pair[0] > pair[1]:
                pair = (pair[1], pair[0])

            if pair in basic_pairs:
                continue

            i, j = pair
            score = pairwise_scores[i, j]
            if score >= 0:
                # score>=0 means that (i, j) has been checked by pre-matching
                continue
            pairs.add(pair)
            stats["added_pair_count_from_suppl_pairs"] += 1

        log(f"[{scene.scene}] make_pairs: {stats}")
        return sorted(list(pairs))


def focus_on_cluster(clustered_scene: Scene, pairwise_scores: np.ndarray) -> np.ndarray:
    assert clustered_scene.indices_in_parent_scene is not None
    keeps = clustered_scene.indices_in_parent_scene
    return pairwise_scores[keeps][:, keeps].copy()


def make_pairs_from_pairwise_scores(
    pairwise_scores: np.ndarray,
    topk: int | None = None,
) -> list[tuple[int, int]]:
    pairs = set()
    for i in range(len(pairwise_scores)):
        ranks = np.argsort(-pairwise_scores[i])
        if topk is not None:
            ranks = ranks[:topk]
        ranked_scores = np.take_along_axis(pairwise_scores[i], ranks, axis=0)

        keeps = ranked_scores > 0
        ranks = ranks[keeps]
        ranked_scores = ranked_scores[keeps]
        # print(ranked_scores)

        for j in ranks:
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))
    return sorted(list(pairs))


def get_best_reconstruction(
    maps: dict[int, pycolmap.Reconstruction],
    scene: Scene,
) -> tuple[dict, dict]:
    images_registered = 0
    best_idx = None
    for idx, rec in maps.items():
        print(idx, rec.summary())
        if len(rec.images) > images_registered:
            images_registered = len(rec.images)
            best_idx = idx

    # #valid reconstructions = reconstructions with at least one registered image
    num_valid_reconstructions = sum(1 for rec in maps.values() if len(rec.images) > 0)
    log(
        f"[Reconstruction] {scene.dataset}/{scene.scene} | "
        f"#reconstructions={len(maps)} "
        f"#valid_reconstructions={num_valid_reconstructions} "
        f"#registered_images(best)={images_registered}"
    )

    if best_idx is None:
        return get_outlier_reconstructions(scene)

    results = {}
    infos = {"localization_by": {}}
    camid_im_map = {}
    for k, im in maps[best_idx].images.items():
        key = scene.data_schema.format_output_key(
            scene.dataset,
            scene.scene,
            im.name,
        )
        metadata = scene.data_schema.get_output_metadata(
            scene.dataset,
            scene.scene,
            im.name,
        )
        results[key] = {
            "R": copy.deepcopy(_cam_from_world(im).rotation.matrix()),
            "t": copy.deepcopy(np.array(_cam_from_world(im).translation)),
            "metadata": metadata,
        }
        infos["localization_by"][key] = "colmap"
        camid_im_map[im.camera_id] = im.name

    try:
        for idx, rec in maps.items():
            u_cameras = []
            g_cameras = []
            if idx == best_idx:
                continue

            for k, im in rec.images.items():
                key = scene.data_schema.format_output_key(
                    scene.dataset,
                    scene.scene,
                    im.name,
                )
                if key in results:
                    g_R = copy.deepcopy(results[key]["R"])
                    g_t = copy.deepcopy(results[key]["t"])
                    g_C = -g_R.T @ g_t

                    u_R = copy.deepcopy(_cam_from_world(im).rotation.matrix())
                    u_t = copy.deepcopy(np.array(_cam_from_world(im).translation))
                    u_C = -u_R.T @ u_t
                    g_cameras.append(g_C.reshape(3, 1))
                    u_cameras.append(u_C.reshape(3, 1))
            if len(g_cameras) < 3:
                print(
                    f"# of cameras that are registered to both rec({idx}) and best({best_idx}): {len(g_cameras)}"
                )
                continue
            g_cameras = np.array(g_cameras).reshape(3, -1)
            u_cameras = np.array(u_cameras).reshape(3, -1)
            inl_cf = 0
            strict_cf = -1
            thresholds = np.array([0.025, 0.05, 0.1, 0.2, 0.5, 1.0])
            model = register_by_Horn(
                u_cameras, g_cameras, np.asarray(thresholds), inl_cf, strict_cf
            )
            T = np.squeeze(model["transf_matrix"][-1])
            # print(T)
            # print(T[:3].shape)
            for k, im in rec.images.items():
                key = scene.data_schema.format_output_key(
                    scene.dataset,
                    scene.scene,
                    im.name,
                )
                if key not in results:
                    Tcw2 = np.eye(4)
                    Tcw2[:3, :3] = copy.deepcopy(_cam_from_world(im).rotation.matrix())
                    Tcw2[:3, 3] = copy.deepcopy(np.array(_cam_from_world(im).translation))
                    Tw2c = np.linalg.inv(Tcw2)
                    Tw1c = np.matmul(T, Tw2c)
                    Tcw1 = np.linalg.inv(Tw1c)
                    results[key]["R"] = copy.deepcopy(Tcw1[:3, :3])
                    results[key]["t"] = copy.deepcopy(Tcw1[:3, 3])
                    infos["localization_by"][key] = "horn"
                    print(f"Registered {key} by alignment to the best reconstruction")
                else:
                    print(
                        f"Registered {key}, but it has already been in the best reconstruction"
                    )
    except Exception as e:
        print(f"Registration failed: {e}")

    # Failures_to_outliers:
    for path in scene.image_paths:
        key1 = scene.data_schema.format_output_key(
            scene.dataset, scene.scene, Path(path).name
        )
        if key1 not in results:
            print(
                f"Reconstruction failed: "
                f"{scene}[{key1}] -> {DEFAULT_OUTLIER_SCENE_NAME}"
            )
            metadata = scene.data_schema.get_output_metadata(
                scene.dataset,
                scene.scene,
                Path(path).name,
            )
            R = np.eye(3) * np.nan
            t = np.zeros(3) * np.nan
            results[key1] = {
                "R": R,
                "t": t,
                "cluster_name": DEFAULT_OUTLIER_SCENE_NAME,
                "metadata": metadata,
            }
            infos["localization_by"][key1] = "fill_nan"

    return results, infos
