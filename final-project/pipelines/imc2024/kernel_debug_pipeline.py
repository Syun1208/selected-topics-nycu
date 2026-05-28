import time
from typing import Optional

import pandas as pd
import pycolmap
import torch
import tqdm

from colmap import get_best_reconstruction, import_into_colmap
from data import set_random_seed, on_kaggle_kernel_rerun
from data_schema import DataSchema
from distributed import DistConfig
from extractor import (
    Line2DFeatureExtractor,
    LocalFeatureExtractor,
    extract_line2d_features_all,
)
from features.factory import create_line2d_feature_handler, create_local_feature_handler
from matchers.base import (
    run_overlap_region_estimation,
)
from matchers.factory import create_detector_free_matcher, create_local_feature_matcher
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import IMC2024PipelineConfig
from pipelines.matching import (
    run_pre_matching,
)
from pipelines.snapshot import SceneSnapshot
from pipelines.stage import (
    run_detector_free_matching_stage,
    run_detector_free_matching_stage_with_hloc_match_dense,
    run_local_feature_extraction_and_matching_stage,
)
from pipelines.verification import verify_matches
from postprocesses.matching_filter import create_matching_filter
from postprocesses.panet import PANetRefiner
from postprocesses.pruning import create_pruner
from preprocesses.deblur import run_deblurring
from preprocesses.orientation import compute_and_register_orientations
from preprocesses.region import OverlapRegionEstimator
from shortlists.factory import create_shortlist_generator, create_shortlist_updater
from storage import (
    concat_matched_keypoints,
    filter_matched_keypoints_by_mask_regions,
    fuse_matching_sets_late,
)
from workspace import log


class KernelDebugPipeline(Pipeline):
    def __init__(
        self,
        conf: IMC2024PipelineConfig,
        dist_conf: Optional[DistConfig] = None,
        device: Optional[torch.device] = None,
    ):
        set_random_seed(seed=conf.seed)
        dist_conf = dist_conf or DistConfig.single()
        device = device or torch.device("cpu")

        self.dist_conf = dist_conf
        self.device = device
        self.conf = conf
        self.shortlist_generator = create_shortlist_generator(
            conf.shortlist_generator, device=device
        )

        self.shortlist_updater = None
        if conf.shortlist_updater:
            self.shortlist_updater = create_shortlist_updater(
                conf.shortlist_updater, device=device
            )

        self.overlap_region_estimator = None
        if conf.overlap_region_estimation:
            self.overlap_region_estimator = OverlapRegionEstimator(
                conf.overlap_region_estimation
            )

        local_feature_extractors = []
        for f_conf in conf.local_features:
            handler = create_local_feature_handler(f_conf, device=device)
            extractor = LocalFeatureExtractor(f_conf, handler)
            local_feature_extractors.append(extractor)
        self.local_feature_extractors = local_feature_extractors

        local_feature_matchers = []
        for _conf in conf.local_feature_matchers:
            matcher = create_local_feature_matcher(_conf, device=device)
            local_feature_matchers.append(matcher)
        self.local_feature_matchers = local_feature_matchers

        detector_free_matchers = []
        for _conf in conf.detector_free_matchers:
            matcher = create_detector_free_matcher(_conf, device=device)
            detector_free_matchers.append(matcher)
        self.detector_free_matchers = detector_free_matchers

        line2d_feature_extractors = []
        for _conf in conf.line2d_features:
            handler = create_line2d_feature_handler(_conf, device=device)
            extractor = Line2DFeatureExtractor(_conf, handler)
            line2d_feature_extractors.append(extractor)
        self.line2d_feature_extractors = line2d_feature_extractors

        self.filter = None
        if conf.filtering:
            self.filter = create_matching_filter(conf.filtering, device=device)
            print("[KernelDebugPipeline] Use filter")

        self.refiner = None
        if conf.refinement:
            self.refiner = PANetRefiner(conf.refinement, device=device)
            print("[KernelDebugPipeline] Use refiner")

        self.pruner = None
        if conf.pruning:
            self.pruner = create_pruner(conf.pruning, device=device)
            print(f"[KernelDebugPipeline] Use pruner: {self.pruner}")

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("KernelDebugPipeline starts")

        data_dict = create_data_dict(data_schema, df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f"The data list has been loaded. # of scenes: {num_scenes}")

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict, data_schema),
            total=num_scenes,
            desc="KernelDebugPipeline",
            disable=self.dist_conf.is_slave(),
        )

        for scene in iterator:
            assert isinstance(scene, Scene)
            with scene.create_space() as scene:
                outputs = self.run_scene(scene, iterator, save_snapshot=save_snapshot)
                results[scene.dataset][scene.scene] = outputs

        if on_kaggle_kernel_rerun():
            while True:
                time.sleep(5)

        df = results_to_submission_df(results)
        return df

    def run_scene(
        self, scene: Scene, iterator: tqdm.tqdm, save_snapshot: bool = False
    ) -> dict:
        scene.cache_all_images()

        if self.conf.deblurring:
            run_deblurring(
                scene, self.conf.deblurring, device=self.device, progress_bar=iterator
            )

        if self.conf.orientation_normalization:
            compute_and_register_orientations(
                scene, self.conf.orientation_normalization, progress_bar=iterator
            )

        if self.conf.pre_matching:
            pairs = self.shortlist_generator(
                scene,
                progress_bar=iterator,
            )
            log(f"[{scene}] # of pairs: {len(pairs)}")

            # Pre-matching
            prematch_mk_storage_list = run_pre_matching(
                self.conf.pre_matching,
                pairs,
                scene,
                device=self.device,
                progress_bar=iterator,
            )

            if self.shortlist_updater:
                pairs = self.shortlist_updater(
                    scene,
                    progress_bar=iterator,
                    mk_storage_list=prematch_mk_storage_list,
                )
                log(f"[{scene}] # of pairs after updating shortlist: {len(pairs)}")

            if self.overlap_region_estimator:
                # Overlap esitmation based on pre-matching
                prematch_mk_storage = concat_matched_keypoints(
                    prematch_mk_storage_list, use_score_if_exists=False
                )
                run_overlap_region_estimation(
                    self.overlap_region_estimator,
                    pairs,
                    scene,
                    matched_keypoint_storage=prematch_mk_storage,
                    progress_bar=iterator,
                )

                if self.conf.masking and self.conf.masking.make_watermark_masks:
                    scene.make_mask_regions_from_overlap_regions(
                        overlap_delta=self.conf.masking.watermark_overlap_delta,
                        border_delta=self.conf.masking.watermark_border_delta,
                    )

                    if self.conf.masking.rerun_overlap_estimation:
                        prematch_mk_storage = filter_matched_keypoints_by_mask_regions(
                            prematch_mk_storage, scene
                        )
                        run_overlap_region_estimation(
                            self.overlap_region_estimator,
                            pairs,
                            scene,
                            matched_keypoint_storage=prematch_mk_storage,
                            progress_bar=iterator,
                        )

                # Convert set of overlap regions to a bbox of an image
                scene.make_roi_from_overlap_regions()
        else:
            pairs = self.shortlist_generator(
                scene,
                progress_bar=iterator,
            )
            log(f"[{scene}] # of pairs: {len(pairs)}")

        # Run line2d feature extraction
        line2d_feature_storage = None
        if self.line2d_feature_extractors:
            line2d_feature_storage = extract_line2d_features_all(
                self.line2d_feature_extractors[0],
                scene,
                progress_bar=iterator
            )

        # Run local feature extraction and matching
        local_feature_keypoint_storages, local_feature_matching_storages = (
            run_local_feature_extraction_and_matching_stage(
                scene,
                pairs,
                self.local_feature_extractors,
                self.local_feature_matchers,
                line2d_feature_storage=line2d_feature_storage,
                progress_bar=iterator,
            )
        )

        scene.release_all()
        return {}
