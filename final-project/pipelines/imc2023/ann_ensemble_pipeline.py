from pathlib import Path
from typing import List, Optional

import pandas as pd
import pycolmap
import torch
import tqdm

from colmap import get_best_reconstruction, import_into_colmap
from data import (DEFAULT_TMP_DIR, SHOW_MEM_USAGE, DirPath, FilePath,
                  load_train_df, set_random_seed)
from distributed import DistConfig
from extractor import LocalFeatureExtractor, extract_all
from features.base import LocalFeatureHandler
from features.factory import create_local_feature_handler
from matchers.base import (LocalFeatureMatcher, run_detector_free_matching,
                           run_local_feature_matching)
from matchers.factory import (create_detector_free_matcher,
                              create_local_feature_matcher)
from pipelines.base import Pipeline
from pipelines.common import (Scene, create_data_dict, init_result_dict,
                              iterate_scenes, results_to_submission_df)
from pipelines.config import ANNEnsemblePipelineConfig
from pipelines.verification import verify_matches
from postprocesses.matching_filter import create_matching_filter
from shortlists.factory import create_shortlist_generator
from storage import (InMemoryKeypointStorage, InMemoryLocalFeatureStorage,
                     InMemoryMatchedKeypointStorage, InMemoryMatchingStorage,
                     fuse_matching_sets_late)
from workspace import log


class LocalFetureBasedANNEnsemblePipeline(Pipeline):
    def __init__(
        self,
        conf: ANNEnsemblePipelineConfig,
        dist_conf: Optional[DistConfig] = None,
        device: Optional[torch.device] = None
    ):
        set_random_seed(seed=conf.seed)
        dist_conf = dist_conf or DistConfig.single()
        device = device or torch.device('cpu')

        self.dist_conf = dist_conf
        self.device = device
        self.conf = conf

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
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        log(f'SimpleEnsemblePipeline starts')

        data_dict = create_data_dict(df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f'The data list has been loaded. # of scenes: {num_scenes}')

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict),
            total=num_scenes,
            desc='SimpleEnsemblePipeline',
            disable=self.dist_conf.is_slave()
        )

        for scene in iterator:
            assert isinstance(scene, Scene)
            with scene.create_space() as scene:
                outputs = self.run_scene(scene, iterator)
                results[scene.dataset][scene.scene] = outputs
        
        df = results_to_submission_df(results)
        return df

    def run_scene(
        self,
        scene: Scene,
        iterator: tqdm.tqdm
    ) -> dict:
        scene.cache_all_images()

        pairs = self.shortlist_generator(scene, progress_bar=iterator)
        log(f'[{scene}] # of pairs: {len(pairs)}')

        # Run local feature extraction and matching
        local_feature_storages: List[InMemoryLocalFeatureStorage] = []
        local_feature_keypoint_storages: List[InMemoryKeypointStorage] = []
        local_feature_matching_storages: List[InMemoryMatchingStorage] = []
        for extractor, matcher in zip(self.local_feature_extractors,
                                      self.local_feature_matchers):
            m_storage = InMemoryMatchingStorage()
            f_storage = InMemoryLocalFeatureStorage()

            # Extract local features from all images in the scene
            extract_all(extractor, scene,
                        storage=f_storage,
                        progress_bar=iterator)

            # Match all pairs
            run_local_feature_matching(
                matcher, pairs, scene,
                feature_storage=f_storage,
                matching_storage=m_storage,
                progress_bar=iterator
            )
            k_storage = f_storage.to_memory().to_keypoint_storage()
            
            local_feature_storages.append(f_storage)
            local_feature_keypoint_storages.append(k_storage)
            local_feature_matching_storages.append(m_storage)
        
        # Run detector-free matchers
        detector_free_keypoint_storages: List[InMemoryKeypointStorage] = []
        detector_free_matching_storages: List[InMemoryMatchingStorage] = []
        for m_conf, matcher in zip(self.conf.detector_free_matchers,
                                   self.detector_free_matchers):
            mk_storage = InMemoryMatchedKeypointStorage()
            k_storage = InMemoryKeypointStorage()
            m_storage = InMemoryMatchingStorage()

            # Match all pairs
            run_detector_free_matching(
                matcher, pairs, scene,
                matched_keypoint_storage=mk_storage,
                keypoint_storage=k_storage,
                matching_storage=m_storage,
                apply_round=m_conf.apply_round,
                progress_bar=iterator
            )

            detector_free_keypoint_storages.append(k_storage)
            detector_free_matching_storages.append(m_storage)

        keypoint_storages = local_feature_keypoint_storages + detector_free_keypoint_storages
        matching_storages = local_feature_matching_storages + detector_free_matching_storages
        keypoint_storage, matching_storage = fuse_matching_sets_late(
            list(zip(keypoint_storages, matching_storages)),
            scene
        )

        # Filter matches
        if self.filter:
            self.filter.run(keypoint_storage, matching_storage, scene,
                            progress_bar=iterator)

        # Add keypoints and matches into COLMAP DB
        database_path = str(scene.database_path)
        id_mappings = import_into_colmap(
            scene, keypoint_storage, matching_storage,
            database_path=database_path
        )
        
        # Add two-view geometry into COLMAP DB
        verify_matches(scene,
                       self.conf.verification,
                       keypoint_storage=keypoint_storage,
                       matching_storage=matching_storage,
                       id_mappings=id_mappings,
                       progress_bar=iterator)
        
        scene.release_cached_images()

        # NOTE
        # (From https://www.kaggle.com/code/eduardtrulls/imc-2023-submission-example/notebook)
        # By default colmap does not generate a reconstruction
        # if less than 10 images are registered. Lower it to 3.
        mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options.min_model_size = self.conf.reconstruction.mapper_min_model_size or 3
        if self.conf.reconstruction.mapper_multiple_models is not None:
            mapper_options.multiple_models = self.conf.reconstruction.mapper_multiple_models

        # NOTE
        # Doc: https://github.com/colmap/pycolmap/blob/master/pipeline/sfm.cc
        maps = pycolmap.incremental_mapping(database_path=database_path,
                                            image_path=str(scene.image_dir),
                                            output_path=str(scene.reconstruction_dir),
                                            options=mapper_options)
        
        outputs, infos = get_best_reconstruction(
            maps, scene,
            keypoint_storage, matching_storage,
            fill_zero_Rt=True,
            fill_nearest_position=self.conf.reconstruction.fill_nearest_position
        )
        scene.release_all()
        return outputs
