from pathlib import Path
from typing import Optional

import pandas as pd
import pycolmap
import torch
import tqdm

from colmap import (
    get_best_reconstruction,
    get_image_id_of_scene_graph_center,
    import_into_colmap,
)
from data import SAVE_CAMERA_DEBUG_INFO, set_random_seed
from data_schema import DataSchema
from distributed import DistConfig
from extractor import (
    Line2DFeatureExtractor,
    LocalFeatureExtractor,
)
from features.factory import create_line2d_feature_handler, create_local_feature_handler
from localizers.factory import create_post_localizer
from matchers.base import (
    run_overlap_region_estimation,
)
from matchers.factory import (
    create_detector_free_matcher,
    create_line2d_feature_matcher,
    create_local_feature_matcher,
)
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import IMC2024PipelineConfig
from pipelines.matching import run_pre_matching
from pipelines.snapshot import SceneSnapshot
from pipelines.stage import (
    run_detector_free_matching_stage,
    run_detector_free_matching_stage_with_hloc_match_dense,
    run_line2d_feature_extraction_and_matching_stage,
    run_local_feature_extraction_and_matching_stage,
)
from pipelines.verification import verify_matches
from postprocesses.matching_filter import create_matching_filter
from postprocesses.panet import PANetRefiner
from postprocesses.pruning import create_pruner
from preprocesses.deblur import run_deblurring
from preprocesses.depth import run_depth_estimation
from preprocesses.orientation import compute_and_register_orientations
from preprocesses.region import OverlapRegionEstimator
from preprocesses.segmentation import run_segmentation
from shortlists.factory import create_shortlist_generator, create_shortlist_updater
from storage import (
    concat_matched_keypoints,
    filter_matched_keypoints_by_mask_regions,
    fuse_line2d_matching_sets_late,
    fuse_matching_sets_late,
)
from utils.camvis import save_camera_debug_info
from workspace import log


class IMC2024Pipeline(Pipeline):
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
            print("[IMC2024Pipeline] Use line2d features")
        self.line2d_feature_extractors = line2d_feature_extractors

        line2d_feature_matchers = []
        for _conf in conf.line2d_matchers:
            matcher = create_line2d_feature_matcher(_conf, device=device)
            line2d_feature_matchers.append(matcher)
            print("[IMC2024Pipeline] Use line2d matchers")
        self.line2d_feature_matchers = line2d_feature_matchers

        self.filter = None
        if conf.filtering:
            self.filter = create_matching_filter(conf.filtering, device=device)
            print("[IMC2024Pipeline] Use filter")

        self.refiner = None
        if conf.refinement:
            self.refiner = PANetRefiner(conf.refinement, device=device)
            print("[IMC2024Pipeline] Use refiner")

        self.pruner = None
        if conf.pruning:
            self.pruner = create_pruner(conf.pruning, device=device)
            print(f"[IMC2024Pipeline] Use pruner: {self.pruner}")

        self.post_localizer = None
        if conf.post_localizer:
            self.post_localizer = create_post_localizer(
                conf.post_localizer, device=device
            )
            print(f"[IMC2024Pipeline] Use post localizer: {self.post_localizer}")

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("IMC2024Pipeline starts")

        data_dict = create_data_dict(data_schema, df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f"The data list has been loaded. # of scenes: {num_scenes}")

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict, data_schema),
            total=num_scenes,
            desc="IMC2024Pipeline",
            disable=self.dist_conf.is_slave(),
        )

        for scene in iterator:
            assert isinstance(scene, Scene)
            with scene.create_space() as scene:
                outputs = self.run_scene(scene, iterator, save_snapshot=save_snapshot)
                results[scene.dataset][scene.scene] = outputs

        df = results_to_submission_df(results)
        return df

    def run_scene(
        self, scene: Scene, iterator: tqdm.tqdm, save_snapshot: bool = False
    ) -> dict:
        scene.cache_all_images()

        if self.conf.segmentation:
            if (
                self.conf.segmentation.skip_when_identical_camera_scene
                and scene.get_unique_resolution_num() == 1
            ):
                print("Skip a segmentation process")
            else:
                run_segmentation(
                    scene,
                    self.conf.segmentation,
                    device=self.device,
                    progress_bar=iterator,
                )

        if self.conf.depth_estimation:
            run_depth_estimation(
                scene,
                self.conf.depth_estimation,
                device=self.device,
                progress_bar=iterator,
            )

        if self.conf.deblurring:
            run_deblurring(
                scene, self.conf.deblurring, device=self.device, progress_bar=iterator
            )

        if self.conf.orientation_normalization:
            compute_and_register_orientations(
                scene, self.conf.orientation_normalization, progress_bar=iterator
            )

        line2d_seg_storage = None
        line2d_matching_storage = None

        if self.conf.pre_matching:
            pairs = self.shortlist_generator(
                scene,
                progress_bar=iterator,
            )
            log(f"[{scene}] # of pairs: {len(pairs)}")

            # Line2d matching
            if self.line2d_feature_extractors:
                line2d_seg_storages, line2d_matching_storages = (
                    run_line2d_feature_extraction_and_matching_stage(
                        scene,
                        pairs,
                        self.line2d_feature_extractors,
                        self.line2d_feature_matchers,
                        progress_bar=iterator,
                    )
                )
                line2d_seg_storage, line2d_matching_storage = (
                    fuse_line2d_matching_sets_late(
                        list(zip(line2d_seg_storages, line2d_matching_storages)), scene
                    )
                )

            # Pre-matching
            prematch_mk_storage_list = run_pre_matching(
                self.conf.pre_matching,
                pairs,
                scene,
                line2d_seg_storage=line2d_seg_storage,
                segmentation_conf=self.conf.segmentation,
                device=self.device,
                progress_bar=iterator,
            )

            if self.shortlist_updater:
                pairs = self.shortlist_updater(
                    scene,
                    progress_bar=iterator,
                    mk_storage_list=prematch_mk_storage_list,
                    line2d_seg_storage=line2d_seg_storage,
                    line2d_matching_storage=line2d_matching_storage,
                )
                log(f"[{scene}] # of pairs after updating shortlist: {len(pairs)}")

            # Overlap esitmation based on pre-matching
            if self.overlap_region_estimator:
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

            # Line2d matching
            if self.line2d_feature_extractors:
                line2d_seg_storages, line2d_matching_storages = (
                    run_line2d_feature_extraction_and_matching_stage(
                        scene,
                        pairs,
                        self.line2d_feature_extractors,
                        self.line2d_feature_matchers,
                        progress_bar=iterator,
                    )
                )
                line2d_seg_storage, line2d_matching_storage = (
                    fuse_line2d_matching_sets_late(
                        list(zip(line2d_seg_storages, line2d_matching_storages)), scene
                    )
                )

        # Run local feature extraction and matching
        local_feature_keypoint_storages, local_feature_matching_storages = (
            run_local_feature_extraction_and_matching_stage(
                scene,
                pairs,
                self.local_feature_extractors,
                self.local_feature_matchers,
                line2d_seg_storage=line2d_seg_storage,
                segmentation_conf=self.conf.segmentation,
                progress_bar=iterator,
            )
        )

        # Run detector-free matchers
        if self.conf.hloc_match_dense:
            detector_free_keypoint_storages, detector_free_matching_storages = (
                run_detector_free_matching_stage_with_hloc_match_dense(
                    scene,
                    pairs,
                    self.detector_free_matchers,
                    self.conf.detector_free_matchers,
                    self.conf.hloc_match_dense,
                    local_feature_keypoint_storages=local_feature_keypoint_storages,  # type: ignore
                    progress_bar=iterator,
                )
            )
        else:
            detector_free_keypoint_storages, detector_free_matching_storages = (
                run_detector_free_matching_stage(
                    scene,
                    pairs,
                    self.detector_free_matchers,
                    self.conf.detector_free_matchers,
                    progress_bar=iterator,
                )
            )

        # Concat keypoints and matchings
        keypoint_storages = (
            local_feature_keypoint_storages + detector_free_keypoint_storages
        )
        matching_storages = (
            local_feature_matching_storages + detector_free_matching_storages
        )
        keypoint_storage, matching_storage = fuse_matching_sets_late(
            list(zip(keypoint_storages, matching_storages)), scene
        )

        # Filter matches
        if self.filter:
            self.filter.run(
                keypoint_storage, matching_storage, scene, progress_bar=iterator
            )

        # Refine keypoints
        if self.refiner:
            self.refiner.refine_all(
                scene, keypoint_storage, matching_storage, progress_bar=iterator
            )

        # Add keypoints and matches into COLMAP DB
        database_path = str(scene.database_path)
        id_mappings = import_into_colmap(
            scene,
            keypoint_storage,
            matching_storage,
            database_path=database_path,
            camera_model=self.conf.reconstruction.get_camera_model(
                unique_resolution_num=scene.get_unique_resolution_num()
            ),
        )

        # Add two-view geometry into COLMAP DB
        g_storage = verify_matches(
            scene,
            self.conf.verification,
            keypoint_storage=keypoint_storage,
            matching_storage=matching_storage,
            id_mappings=id_mappings,
            progress_bar=iterator,
        )

        if self.pruner:
            # Prune two-view geometry
            self.pruner(
                scene, g_storage, database_path=database_path, progress_bar=iterator
            )

        if save_snapshot:
            SceneSnapshot(
                scene,
                keypoint_storage,
                matching_storage,
                two_view_geometry_storage=g_storage,
            ).save(pipeline_id=self.pipeline_id)
        scene.release_cached_images()

        # NOTE
        # (From https://www.kaggle.com/code/eduardtrulls/imc-2023-submission-example/notebook)
        # By default colmap does not generate a reconstruction
        # if less than 10 images are registered. Lower it to 3.
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.num_threads = 1
        mapper_options.min_model_size = (
            self.conf.reconstruction.mapper_min_model_size or 3
        )
        if self.conf.reconstruction.mapper_max_num_models is not None:
            mapper_options.max_num_models = (
                self.conf.reconstruction.mapper_max_num_models
            )
        if self.conf.reconstruction.mapper_multiple_models is not None:
            mapper_options.multiple_models = (
                self.conf.reconstruction.mapper_multiple_models
            )
        if self.conf.reconstruction.mapper_min_num_matches is not None:
            mapper_options.min_num_matches = (
                self.conf.reconstruction.mapper_min_num_matches
            )
        if self.conf.reconstruction.mapper_filter_max_reproj_error is not None:
            mapper_options.mapper.filter_max_reproj_error = (
                self.conf.reconstruction.mapper_filter_max_reproj_error
            )
        if self.conf.reconstruction.set_scene_graph_center_node_to_init_image_id1:
            image_id1 = get_image_id_of_scene_graph_center(
                scene, database_path=database_path
            )
            if image_id1 is not None:
                mapper_options.init_image_id1 = image_id1

        # NOTE
        # Doc: https://github.com/colmap/pycolmap/blob/master/pipeline/sfm.cc
        maps = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=str(scene.image_dir),
            output_path=str(scene.reconstruction_dir),
            options=mapper_options,
        )

        outputs, infos = get_best_reconstruction(
            maps,
            scene,
            keypoint_storage,
            matching_storage,
            fill_zero_Rt=self.conf.reconstruction.fill_zero_Rt,
            fill_nan_Rt=self.conf.reconstruction.fill_nan_Rt,
            fill_nearest_position=self.conf.reconstruction.fill_nearest_position,
            use_localize_sfm=self.conf.reconstruction.use_localize_sfm,
            use_localize_pixloc=self.conf.reconstruction.use_localize_pixloc,
            post_localizer=self.post_localizer,
            progress_bar=iterator,
        )
        if SAVE_CAMERA_DEBUG_INFO:
            print(infos)
            save_camera_debug_info(
                outputs,
                scene,
                Path(f"extra/camvis/{self.pipeline_id}"),
                prefix_dict=infos["localization_by"],
            )
        scene.release_all()
        return outputs
