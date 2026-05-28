import gc
from typing import List, Optional

import pandas as pd
import pycolmap
import torch
import tqdm

from colmap import (
    get_best_reconstruction,
    import_into_colmap,
)
from data import set_random_seed
from data_schema import DataSchema
from distributed import DistConfig
from extractor import LocalFeatureExtractor, extract_all
from features.factory import create_local_feature_handler
from matchers.factory import create_local_feature_matcher
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import DetectorBasedPipelineConfig
from pipelines.matching import run_local_feature_matching
from pipelines.snapshot import SceneSnapshot
from pipelines.verification import verify_matches
from postprocesses.matching_filter import create_matching_filter
from shortlists.factory import create_shortlist_generator
from storage import (
    InMemoryLocalFeatureStorage,
    InMemoryMatchingStorage,
    fuse_matching_sets_late,
)
from workspace import log


class DetectorBasedPipeline(Pipeline):
    def __init__(
        self,
        conf: DetectorBasedPipelineConfig,
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

        extractors = []
        for f_conf in conf.local_features:
            handler = create_local_feature_handler(f_conf, device=device)
            extractor = LocalFeatureExtractor(f_conf, handler)
            extractors.append(extractor)
        self.extractors = extractors

        self.matcher = create_local_feature_matcher(conf.matcher, device=device)
        self.filter = None
        if conf.filtering:
            self.filter = create_matching_filter(conf.filtering, device=device)

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("DetectorBasedPipeline starts")

        data_dict = create_data_dict(data_schema, df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f"The data list has been loaded. # of scenes: {num_scenes}")

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict, data_schema),
            total=num_scenes,
            desc="DetectorBasedPipeline",
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

        feature_storages: List[InMemoryLocalFeatureStorage] = []
        matching_storages: List[InMemoryMatchingStorage] = []

        for extractor in self.extractors:
            m_storage = InMemoryMatchingStorage()
            f_storage = InMemoryLocalFeatureStorage()

            # Extract local features from all images in the scene
            extract_all(extractor, scene, storage=f_storage, progress_bar=iterator)

            # Match all pairs
            run_local_feature_matching(
                self.matcher,
                pairs,
                scene,
                feature_storage=f_storage,
                matching_storage=m_storage,
                progress_bar=iterator,
            )

            feature_storages.append(f_storage)
            matching_storages.append(m_storage)

        keypoint_storages = [
            storage.to_memory().to_keypoint_storage() for storage in feature_storages
        ]
        keypoint_storage, matching_storage = fuse_matching_sets_late(
            list(zip(keypoint_storages, matching_storages)), scene
        )

        del keypoint_storages
        del matching_storages
        gc.collect()

        if self.filter:
            self.filter.run(
                keypoint_storage, matching_storage, scene, progress_bar=iterator
            )

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
            use_localize_sfm=self.conf.reconstruction.use_localize_sfm,
            use_localize_pixloc=self.conf.reconstruction.use_localize_pixloc,
        )
        scene.release_all()
        return outputs
