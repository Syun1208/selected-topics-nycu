from typing import Optional

import pandas as pd
import pycolmap
import torch
import tqdm

from colmap import get_best_reconstruction, import_into_colmap
from data import (
    set_random_seed,
)
from data_schema import DataSchema
from distributed import DistConfig
from extractor import LocalFeatureExtractor
from features.factory import create_local_feature_handler
from matchers.factory import create_detector_free_matcher, create_local_feature_matcher
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import SimpleEnsemblePipelineConfig
from pipelines.snapshot import SceneSnapshot
from pipelines.stage import (
    run_detector_free_matching_stage,
    run_detector_free_matching_stage_with_hloc_match_dense,
    run_local_feature_extraction_and_matching_stage,
)
from pipelines.verification import verify_matches
from postprocesses.matching_filter import create_matching_filter
from shortlists.factory import create_shortlist_generator
from storage import (
    fuse_matching_sets_late,
)
from workspace import log


class SimpleEnsemblePipeline(Pipeline):
    def __init__(
        self,
        conf: SimpleEnsemblePipelineConfig,
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

        self.filter = None
        if conf.filtering:
            self.filter = create_matching_filter(conf.filtering, device=device)

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("SimpleEnsemblePipeline starts")

        data_dict = create_data_dict(data_schema, df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f"The data list has been loaded. # of scenes: {num_scenes}")

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict, data_schema),
            total=num_scenes,
            desc="SimpleEnsemblePipeline",
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

        pairs = self.shortlist_generator(scene, progress_bar=iterator)
        log(f"[{scene}] # of pairs: {len(pairs)}")

        # Run local feature extraction and matching
        local_feature_keypoint_storages, local_feature_matching_storages = (
            run_local_feature_extraction_and_matching_stage(
                scene,
                pairs,
                self.local_feature_extractors,
                self.local_feature_matchers,
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
        # NOTE
        # IncrementalMapperOptions:
        # min_num_matches = 15
        # ignore_watermarks = False
        # multiple_models = True
        # max_num_models = 50
        # max_model_overlap = 20
        # min_model_size = 10
        # init_image_id1 = -1
        # init_image_id2 = -1
        # init_num_trials = 200
        # extract_colors = True
        # num_threads = -1
        # min_focal_length_ratio = 0.1
        # max_focal_length_ratio = 10.0
        # max_extra_param = 1.0
        # ba_refine_focal_length = True
        # ba_refine_principal_point = False
        # ba_refine_extra_params = True
        # ba_min_num_residuals_for_multi_threading = 50000
        # ba_local_num_images = 6
        # ba_local_function_tolerance = 0.0
        # ba_local_max_num_iterations = 25
        # ba_global_use_pba = False
        # ba_global_pba_gpu_index = -1
        # ba_global_images_ratio = 1.1
        # ba_global_points_ratio = 1.1
        # ba_global_images_freq = 500
        # ba_global_points_freq = 250000
        # ba_global_function_tolerance = 0.0
        # ba_global_max_num_iterations = 50
        # ba_local_max_refinements = 2
        # ba_local_max_refinement_change = 0.001
        # ba_global_max_refinements = 5
        # ba_global_max_refinement_change = 0.0005
        # snapshot_path =
        # snapshot_images_freq = 0
        # image_names = set()
        # fix_existing_images = False
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
        )
        scene.release_all()
        return outputs
