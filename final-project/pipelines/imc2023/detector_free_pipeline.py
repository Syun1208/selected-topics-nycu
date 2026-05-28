from typing import Optional

import pandas as pd
import pycolmap
import torch
import tqdm

from colmap import get_best_reconstruction, import_into_colmap
from data import set_random_seed
from data_schema import DataSchema
from distributed import DistConfig
from matchers.factory import create_detector_free_matcher
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import DetectorFreePipelineConfig
from pipelines.snapshot import SceneSnapshot
from pipelines.stage import (
    run_detector_free_matching_stage,
    run_detector_free_matching_stage_with_hloc_match_dense,
)
from pipelines.verification import verify_matches
from postprocesses.matching_filter import create_matching_filter
from shortlists.factory import create_shortlist_generator
from workspace import log


class DetectorFreePipeline(Pipeline):
    def __init__(
        self,
        conf: DetectorFreePipelineConfig,
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

        self.matcher = create_detector_free_matcher(conf.matcher, device=device)
        self.filter = None
        if conf.filtering:
            self.filter = create_matching_filter(conf.filtering, device=device)

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("DetectorFreePipeline starts")

        data_dict = create_data_dict(data_schema, df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f"The data list has been loaded. # of scenes: {num_scenes}")

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict, data_schema),
            total=num_scenes,
            desc="DetectorFreePipeline",
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

        # Run detector-free matchers
        if self.conf.hloc_match_dense:
            keypoint_storages, matching_storages = (
                run_detector_free_matching_stage_with_hloc_match_dense(
                    scene,
                    pairs,
                    [self.matcher],
                    [self.conf.matcher],
                    self.conf.hloc_match_dense,
                    progress_bar=iterator,
                )
            )
        else:
            keypoint_storages, matching_storages = run_detector_free_matching_stage(
                scene,
                pairs,
                [self.matcher],
                [self.conf.matcher],
                progress_bar=iterator,
            )

        keypoint_storage = keypoint_storages[0]
        matching_storage = matching_storages[0]

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
            failures_to_outliers=self.conf.reconstruction.failures_to_outliers,
            use_localize_sfm=self.conf.reconstruction.use_localize_sfm,
            use_localize_pixloc=self.conf.reconstruction.use_localize_pixloc,
        )
        scene.release_all()
        return outputs
