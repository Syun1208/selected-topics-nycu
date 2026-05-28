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
from localizers.factory import create_localizer
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
from pipelines.config import LocalizerPipelineConfig
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
from shortlists.factory import create_shortlist_generator, create_shortlist_updater
from storage import (
    concat_matched_keypoints,
    filter_matched_keypoints_by_mask_regions,
    fuse_line2d_matching_sets_late,
    fuse_matching_sets_late,
)
from utils.camvis import save_camera_debug_info
from workspace import log


class LocalizerPipeline(Pipeline):
    def __init__(
        self,
        conf: LocalizerPipelineConfig,
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

        self.localizer = create_localizer(
            conf.localizer, device=device
        )
        print(f"[LocalizerPipeline] Use localizer: {self.localizer}")

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("LocalizerPipeline starts")

        data_dict = create_data_dict(data_schema, df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f"The data list has been loaded. # of scenes: {num_scenes}")

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict, data_schema),
            total=num_scenes,
            desc="LocalizerPipeline",
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

        pairs = self.shortlist_generator(
            scene,
            progress_bar=iterator,
        )
        log(f"[{scene}] # of pairs: {len(pairs)}")

        if self.conf.depth_estimation:
            run_depth_estimation(
                scene,
                self.conf.depth_estimation,
                device=self.device,
                progress_bar=iterator,
            )

        outputs = self.localizer.localize(scene, pairs)
        if SAVE_CAMERA_DEBUG_INFO:
            save_camera_debug_info(
                outputs,
                scene,
                Path(f"extra/camvis/{self.pipeline_id}"),
            )
        scene.release_all()
        return outputs
