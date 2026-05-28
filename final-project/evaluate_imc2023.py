from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import pandas as pd

from config import PipelineConfig, SubmissionConfig
from data import (DEFAULT_EVALUATION_ROOT_DIR, FilePath,
                  ROTATION_THRESHOLDS_DEGREES_DICT,
                  TRANSLATION_THRESHOLDS_METERS_DICT, camera_dict_from_test_df,
                  load_train_df)
from kernel import run
from workspace import log

SUBSET_TYPES = (
    'full',
    'tiny',
    'dioscuri', # scene=dioscuri
    'cyprus',   # scene=cyprus
    'wall',     # scene=wall
    'urban',    # dataset subset
    'heritage', # dataset subset
    'haiper'    # dataset subset
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-c', '--conf', default=None,
                        help='Path to the pipeline config file')
    parser.add_argument('--submission-csv', default=None,
                        help='Path to submission.csv to evaluate')
    parser.add_argument('-w', '--overwrite', action='store_true')
    parser.add_argument('-s', '--subset', default='full', choices=SUBSET_TYPES)
    parser.add_argument('--cpu', action='store_true')
    return parser.parse_args()


def find_cached_results(
    conf_path: FilePath,
    subset_type: str
) -> Optional[pd.DataFrame]:
    if not DEFAULT_EVALUATION_ROOT_DIR.exists():
        DEFAULT_EVALUATION_ROOT_DIR.mkdir(parents=True, exist_ok=True)
        return
    
    subset_dir = DEFAULT_EVALUATION_ROOT_DIR / subset_type
    if not subset_dir.exists():
        subset_dir.mkdir(parents=True, exist_ok=True)
        return
    
    cache_file = get_cached_result_csv_path(conf_path, subset_type)
    if not cache_file.exists():
        return
    
    df = pd.read_csv(cache_file)
    return df


def get_cached_result_csv_path(
    conf_path: FilePath,
    subset_type: str
) -> Path:
    subset_dir = DEFAULT_EVALUATION_ROOT_DIR / subset_type
    cache_file = subset_dir / f'{Path(conf_path).stem}.csv'
    return cache_file


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    # Symmetric matrix K.
    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                  [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                  [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                  [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # Quaternion is eigenvector of K that corresponds to largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)
    return q


def evaluate_R_t(R_gt: np.ndarray,
                 t_gt: np.ndarray,
                 R: np.ndarray,
                 t: np.ndarray,
                 eps: float = 1e-15) -> Tuple[np.ndarray, np.ndarray]:
    t = t.flatten()
    t_gt = t_gt.flatten()

    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    GT_SCALE = np.linalg.norm(t_gt)
    t = GT_SCALE * (t / (np.linalg.norm(t) + eps))
    err_t = min(np.linalg.norm(t_gt - t), np.linalg.norm(t_gt + t))
    
    return np.degrees(err_q), err_t


def compute_dR_dT(R1: np.ndarray,
                  T1: np.ndarray,
                  R2: np.ndarray,
                  T2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given absolute (R, T) pairs for two cameras,
       compute the relative pose difference, from the first.
    """
    dR = np.dot(R2, R1.T)
    dT = T2 - np.dot(dR, T1)
    return dR, dT


def compute_mAA(err_q: np.ndarray,
                err_t: np.ndarray,
                ths_q: np.ndarray,
                ths_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Compute the mean average accuracy over a set of thresholds.
       Additionally returns the metric only over rotation and translation.
    '''
    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(ths_q, ths_t):
        cur_acc_q = (err_q <= th_q)
        cur_acc_t = (err_t <= th_t)
        cur_acc = cur_acc_q & cur_acc_t
        
        acc.append(cur_acc.astype(np.float32).mean())
        acc_q.append(cur_acc_q.astype(np.float32).mean())
        acc_t.append(cur_acc_t.astype(np.float32).mean())
    return np.array(acc), np.array(acc_q), np.array(acc_t)


def evaluate_submission(
    df: pd.DataFrame
) -> List[Tuple[str, float]]:
    scenes = df['scene'].unique().tolist()
    gt_df = load_train_df(scenes_to_use=scenes)

    submission_dict = camera_dict_from_test_df(df)
    gt_dict = camera_dict_from_test_df(gt_df)

    metrics_per_dataset = []
    for dataset in gt_dict:
        metrics_per_scene = []
        for scene in gt_dict[dataset]:
            err_q_all = []
            err_t_all = []
            images = [camera for camera in gt_dict[dataset][scene]]
            # Process all pairs in a scene
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    gt_i = gt_dict[dataset][scene][images[i]]
                    gt_j = gt_dict[dataset][scene][images[j]]
                    dR_gt, dT_gt = compute_dR_dT(gt_i.rotmat, gt_i.tvec, gt_j.rotmat, gt_j.tvec)

                    pred_i = submission_dict[dataset][scene][images[i]]
                    pred_j = submission_dict[dataset][scene][images[j]]
                    dR_pred, dT_pred = compute_dR_dT(pred_i.rotmat, pred_i.tvec, pred_j.rotmat, pred_j.tvec)

                    err_q, err_t = evaluate_R_t(dR_gt, dT_gt, dR_pred, dT_pred)
                    err_q_all.append(err_q)
                    err_t_all.append(err_t)

            mAA = []
            mAA_q = []
            mAA_t = []
            for err_q, err_t in zip(err_q_all, err_t_all):
                _mAA, _mAA_q, _mAA_t = compute_mAA(
                    err_q=err_q,
                    err_t=err_t,
                    ths_q=ROTATION_THRESHOLDS_DEGREES_DICT[(dataset, scene)],
                    ths_t=TRANSLATION_THRESHOLDS_METERS_DICT[(dataset, scene)]
                )
                mAA.append(_mAA)
                mAA_q.append(_mAA_q)
                mAA_t.append(_mAA_t)

            log(f'{dataset} / {scene} ({len(images)} images, '
                f'{len(err_q_all)} pairs) -> '
                f'mAA={np.mean(mAA):.06f}, '
                f'mAA_q={np.mean(mAA_q):.06f}, '
                f'mAA_t={np.mean(mAA_t):.06f}')
            metrics_per_scene.append(np.mean(mAA))

        metrics_per_dataset.append(np.mean(metrics_per_scene))
        log(f'{dataset} -> mAA={np.mean(metrics_per_scene):.06f}')

    return list(zip(gt_dict.keys(), metrics_per_dataset))


def main():
    """CLI
    """
    args = parse_args()

    if args.submission_csv:
        df = pd.read_csv(args.submission_csv)
        results = evaluate_submission(df)
        print(results)
        return

    assert args.conf
    conf = SubmissionConfig(
        pipeline=PipelineConfig.load_config(args.conf),
        target_data_type='train'
    )

    if args.subset == 'tiny':
        conf.scenes_to_use = ['fountain']
    elif args.subset == 'urban':
        conf.scenes_to_use = ['kyiv-puppet-theater']
    elif args.subset == 'heritage':
        conf.scenes_to_use = ['dioscuri', 'cyprus', 'wall']
    elif args.subset == 'haiper':
        conf.scenes_to_use = ['bike', 'chairs', 'fountain']
    elif args.subset == 'wall':
        conf.scenes_to_use = ['wall']
    elif args.subset == 'cyprus':
        conf.scenes_to_use = ['cyprus']
    elif args.subset == 'dioscuri':
        conf.scenes_to_use = ['dioscuri']
    
    df = find_cached_results(args.conf, args.subset)
    if args.overwrite or df is None:
        if args.cpu:
            device = torch.device('cpu')
        else:
            device = None
        df = run(conf, env_name='local', device=device)

        csv_path = get_cached_result_csv_path(args.conf, args.subset)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)

    results = evaluate_submission(df)
    print(results)


if __name__ == '__main__':
    main()
