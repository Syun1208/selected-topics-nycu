from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import tqdm
import dust3r.inference
import dust3r.utils.image
import dust3r.image_pairs
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from data import set_random_seed
from data_schema import DataSchema
from distributed import DistConfig
from extractor import (
    Line2DFeatureExtractor,
    extract_line2d_features_all,
)
from features.factory import create_line2d_feature_handler
from matchers.base import (
    run_overlap_region_estimation,
)
from pipelines.base import Pipeline
from pipelines.common import (
    Scene,
    create_data_dict,
    init_result_dict,
    iterate_scenes,
    results_to_submission_df,
)
from pipelines.config import DUSt3RPipelineConfig
from pipelines.matching import (
    run_pre_matching,
)
from preprocesses.region import OverlapRegionEstimator
from shortlists.factory import create_shortlist_generator, create_shortlist_updater
from storage import (
    concat_matched_keypoints,
    filter_matched_keypoints_by_mask_regions,
)
from workspace import log


class DUSt3RPipeline(Pipeline):
    def __init__(
        self,
        conf: DUSt3RPipelineConfig,
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

        line2d_feature_extractors = []
        for _conf in conf.line2d_features:
            handler = create_line2d_feature_handler(_conf, device=device)
            extractor = Line2DFeatureExtractor(_conf, handler)
            line2d_feature_extractors.append(extractor)
        self.line2d_feature_extractors = line2d_feature_extractors

        # Load DUSt3R model
        self.model = AsymmetricCroCo3DStereo.from_pretrained(
            "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        ).to(device)

    def run(
        self, df: pd.DataFrame, data_schema: DataSchema, save_snapshot: bool = False
    ) -> pd.DataFrame:
        log("DUSt3RPipeline starts")

        data_dict = create_data_dict(data_schema, df=df)
        results, num_scenes = init_result_dict(data_dict)
        log(f"The data list has been loaded. # of scenes: {num_scenes}")

        iterator = tqdm.tqdm(
            iterate_scenes(data_dict, data_schema),
            total=num_scenes,
            desc="DUSt3RPipeline",
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

        # Run line2d feature extraction
        line2d_feature_storage = None
        if self.line2d_feature_extractors:
            line2d_feature_storage = extract_line2d_features_all(
                self.line2d_feature_extractors[0], scene, progress_bar=iterator
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
                    line2d_feature_storage=line2d_feature_storage,
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

        # TODO
        pairs = pairs[:20]
        use_image_ids = set()
        for pair in pairs:
            i, j = pair
            use_image_ids.add(i)
            use_image_ids.add(j)
        
        idxs = list(sorted(list(use_image_ids)))
        print(idxs)
        image_paths = [
            str(scene.image_paths[i]) for i in idxs
        ]
        print(image_paths)
        imgs = dust3r.utils.image.load_images(image_paths, size=512)
        assert len(imgs) > 1

        dust3r_image_pairs = dust3r.image_pairs.make_pairs(
            imgs, scene_graph="swin", prefilter=None, symmetrize=True
        )
        output = dust3r.inference.inference(
            dust3r_image_pairs,
            self.model,
            self.device,
            batch_size=8
        )

        with torch.inference_mode(False):
            dust3r_scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
            lr = 0.01
            loss = dust3r_scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=lr)
            print(loss)

        focals = dust3r_scene.get_focals().cpu().numpy()
        cams2world = dust3r_scene.get_im_poses().cpu().numpy()

        outputs = {}
        for path in scene.image_paths:
            key = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(path).name
            )
            outputs[key] = {
                'R': np.eye(3),
                't': np.zeros(3)
            }
        for i, pose in zip(idxs, cams2world):
            key = scene.data_schema.format_output_key(
                scene.dataset, scene.scene, Path(scene.image_paths[i]).name
            )
            #pose = np.linalg.inv(pose)
            outputs[key] = {
                "R": pose[0:3, 0:3],
                "t": pose[0:3, 3]
            }

        if save_snapshot:
            pass
        scene.release_cached_images()

        scene.release_all()
        return outputs
