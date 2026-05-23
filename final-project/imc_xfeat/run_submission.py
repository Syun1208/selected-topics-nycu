"""Run the XFeat + COLMAP pipeline on the IMC test set and write ``submission.csv``.

For every dataset folder under ``--test-dir`` the pipeline extracts XFeat
features, matches all image pairs, runs COLMAP Structure-from-Motion and writes
the recovered camera pose (rotation matrix + translation vector) of each image.
Images COLMAP could not register are written with an identity pose.

The output CSV uses the official IMC columns
``image_id, dataset, scene, image, rotation_matrix, translation_vector`` --
the same format as ``sample_submission.csv``.

Run via ``test.sh`` (recommended) or directly:

    python -m imc_xfeat.run_submission --test-dir data/image-matching/test --gpu-ids 0
"""

import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate the IMC submission with the XFeat + COLMAP pipeline.")
    p.add_argument("--test-dir", required=True,
                   help="Folder with <dataset>/<image> subfolders (the IMC test set).")
    p.add_argument("--sample-submission", default="",
                   help="Optional sample_submission.csv (used only for image_id strings).")
    p.add_argument("--output", default="submission.csv", help="Output CSV path.")
    p.add_argument("--weights", default="",
                   help="XFeat weights (default: bundled pretrained xfeat.pt).")
    p.add_argument("--work-dir", default="imc_xfeat/work",
                   help="Scratch directory for COLMAP databases / reconstructions.")
    p.add_argument("--gpu-ids", default="0",
                   help="Comma-separated GPU ids, e.g. '0'. Empty string = CPU.")
    # pipeline hyper-parameters
    p.add_argument("--top-k", type=int, default=4096,
                   help="XFeat keypoints kept per image.")
    p.add_argument("--min-cossim", type=float, default=0.82,
                   help="XFeat descriptor cosine-similarity threshold for a match.")
    p.add_argument("--min-matches", type=int, default=15,
                   help="Minimum matches for an image pair to be kept.")
    p.add_argument("--min-model-size", type=int, default=3,
                   help="Minimum images for a COLMAP sub-reconstruction (cluster).")
    return p.parse_args()


args = parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_ids)

import numpy as np

from imc_xfeat.imc_io import (list_dataset_images, mat_to_str,
                              read_sample_submission, write_submission)
from imc_xfeat.reconstruct import extract_features, match_pairs, reconstruct
from imc_xfeat.xfeat_utils import DEFAULT_WEIGHTS, load_xfeat

IDENTITY_R = mat_to_str(np.eye(3))
ZERO_T = mat_to_str(np.zeros(3))


def build_image_id_map(sample_path):
    """Map ``(dataset, image) -> image_id`` from a sample submission, if available."""
    if not sample_path or not os.path.exists(sample_path):
        return {}
    return {(r["dataset"], r["image"]): r["image_id"]
            for r in read_sample_submission(sample_path)}


def main():
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        raise SystemExit("pycolmap is required. Install it with:  pip install pycolmap")

    datasets = list_dataset_images(args.test_dir)
    if not datasets:
        raise SystemExit(f"No <dataset>/<image> folders found under {args.test_dir}")
    image_id_map = build_image_id_map(args.sample_submission)
    n_images = sum(len(v) for v in datasets.values())
    print(f"[submission] {n_images} images across {len(datasets)} dataset(s): "
          f"{', '.join(datasets)}")

    xfeat = load_xfeat(args.weights or DEFAULT_WEIGHTS, top_k=args.top_k)

    rows, t0 = [], time.time()
    for dataset, names in datasets.items():
        print(f"\n=== dataset: {dataset}  ({len(names)} images) ===")
        image_dir = os.path.join(args.test_dir, dataset)

        poses = {}
        try:
            feats = extract_features(xfeat, image_dir, names, args.top_k)
            pair_matches = match_pairs(xfeat, feats, names,
                                       args.min_cossim, args.min_matches)
            print(f"  kept {len(pair_matches)} candidate image pairs")
            poses = reconstruct(image_dir, names, feats, pair_matches,
                                os.path.join(args.work_dir, dataset),
                                min_model_size=args.min_model_size)
        except Exception as exc:  # keep going: a failed dataset -> identity poses
            print(f"  ERROR processing {dataset}: {exc}")

        for name in names:
            image_id = image_id_map.get((dataset, name), f"{dataset}_{name}_public")
            if name in poses:
                cluster_idx, R, t = poses[name]
                rows.append({
                    "image_id": image_id, "dataset": dataset,
                    "scene": f"cluster{cluster_idx}", "image": name,
                    "rotation_matrix": mat_to_str(R),
                    "translation_vector": mat_to_str(t)})
            else:
                rows.append({
                    "image_id": image_id, "dataset": dataset,
                    "scene": "outliers", "image": name,
                    "rotation_matrix": IDENTITY_R,
                    "translation_vector": ZERO_T})

    write_submission(args.output, rows)
    posed = sum(1 for r in rows if r["scene"] != "outliers")
    print(f"\n[submission] wrote {args.output} | {posed}/{len(rows)} images posed "
          f"| {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
    # pycolmap's native module can abort during interpreter shutdown; the work
    # is already done and flushed, so exit immediately to keep a clean status.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
