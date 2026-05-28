"""Visualize the MASt3R-Hybrid matching inference process, stage by stage.

For a chosen image pair this script runs the MASt3R matcher and renders each
intermediate output of the pipeline:

  Stage 1  Input pair            -> the two shortlisted images
  Stage 2  Raw correspondences   -> all MASt3R dense matches (clear coloured lines)
  Stage 3  Geometric verification-> MAGSAC++/RANSAC inliers (green) vs outliers (red)

Outputs (written to ``--outdir``):
  matching_viz.png      Stage-2 correspondences with thick, high-contrast lines
  matching_stages.png   the three stages stacked vertically (for the report)

Example:
  python report/visualize_matching.py \
      --img1 data/train/imc2023_heritage/cyprus_dsc_6488.png \
      --img2 data/train/imc2023_heritage/cyprus_dsc_6512.png \
      --outdir report/images
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from matchers.config import MASt3RMatcherConfig
from matchers.mast3r import MASt3RMatcher
from storage import InMemoryMatchedKeypointStorage


# ----------------------------------------------------------------------------- #
# Inference
# ----------------------------------------------------------------------------- #
def run_mast3r(img1: str, img2: str, device: torch.device):
    """Run the MASt3R matcher on a pair and return matched keypoints (N,2)x2."""
    conf = MASt3RMatcherConfig.model_validate(
        {
            "mast3r": {"weight_path": "MAST3R", "use_amp": True},
            "size": 512,
            "subsample": 8,
            "pixel_tol": 5,
            "match_threshold": 1.001,
            "min_matches": 1,
        }
    )
    matcher = MASt3RMatcher(conf, device=device)
    store = InMemoryMatchedKeypointStorage()
    matcher(img1, img2, store)
    if not store.has(img1, img2):
        raise RuntimeError("No matches produced for this pair.")
    k1, k2 = store.get(img1, img2)
    return np.asarray(k1, np.float32), np.asarray(k2, np.float32)


def geometric_verification(k1: np.ndarray, k2: np.ndarray):
    """Robust fundamental-matrix fit; return a boolean inlier mask."""
    if len(k1) < 8:
        return np.ones(len(k1), bool)
    try:
        _, mask = cv2.findFundamentalMat(
            k1, k2, cv2.USAC_MAGSAC, 1.0, 0.999, 100000
        )
    except cv2.error:
        _, mask = cv2.findFundamentalMat(k1, k2, cv2.FM_RANSAC, 1.0, 0.999)
    if mask is None:
        return np.ones(len(k1), bool)
    return mask.ravel().astype(bool)


# ----------------------------------------------------------------------------- #
# Drawing helpers
# ----------------------------------------------------------------------------- #
def _stack(img1: np.ndarray, img2: np.ndarray):
    """Pad to equal height and place side by side; return (canvas, x_offset)."""
    h = max(img1.shape[0], img2.shape[0])

    def pad(im):
        c = np.full((h, im.shape[1], 3), 255, np.uint8)
        c[: im.shape[0]] = im
        return c

    a, b = pad(img1), pad(img2)
    return np.hstack([a, b]), a.shape[1]


def draw_matches(
    img1, img2, k1, k2, mask=None, max_lines=60, thickness=2, seed=3
):
    """Draw match lines with a dark outline (for contrast) + a bright core.

    If ``mask`` is given, inliers are green and outliers red; otherwise each
    line gets a distinct bright colour.
    """
    canvas, off = _stack(img1, img2)
    n = len(k1)
    idx = np.linspace(0, n - 1, min(n, max_lines)).astype(int)
    rng = np.random.RandomState(seed)
    for j in idx:
        x1, y1 = k1[j]
        x2, y2 = int(k2[j][0]) + off, int(k2[j][1])
        p, q = (int(x1), int(y1)), (x2, int(y2))
        if mask is not None:
            color = (60, 200, 60) if mask[j] else (60, 60, 230)  # BGR
        else:
            color = tuple(int(c) for c in rng.randint(70, 256, 3))
        # dark outline first, bright line on top -> readable on any background
        cv2.line(canvas, p, q, (20, 20, 20), thickness + 2, cv2.LINE_AA)
        cv2.line(canvas, p, q, color, thickness, cv2.LINE_AA)
        for c in (p, q):
            cv2.circle(canvas, c, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, 5, color, 2, cv2.LINE_AA)
    return canvas


def _banner(canvas, text):
    """Add a title strip above an image panel."""
    bar = np.full((46, canvas.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", default="data/train/imc2023_heritage/cyprus_dsc_6488.png")
    ap.add_argument("--img2", default="data/train/imc2023_heritage/cyprus_dsc_6512.png")
    ap.add_argument("--outdir", default="report/images")
    ap.add_argument("--max_lines", type=int, default=60)
    ap.add_argument("--display_long_edge", type=int, default=900,
                    help="downscale each image to this long edge for the figure")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i1, i2 = cv2.imread(args.img1), cv2.imread(args.img2)

    k1, k2 = run_mast3r(args.img1, args.img2, device)
    inliers = geometric_verification(k1, k2)
    print(f"matches: {len(k1)} | inliers: {int(inliers.sum())} "
          f"({100 * inliers.mean():.1f}%)")

    # Downscale full-resolution images (and their keypoints) for a compact figure.
    def resize_with_kpts(im, kpts, long_edge=args.display_long_edge):
        s = long_edge / max(im.shape[:2])
        if s < 1.0:
            im = cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)),
                            interpolation=cv2.INTER_AREA)
            kpts = kpts * s
        return im, kpts

    i1, k1 = resize_with_kpts(i1, k1)
    i2, k2 = resize_with_kpts(i2, k2)

    # Stage 2: clear correspondences (the report's matching figure)
    viz = draw_matches(i1, i2, k1, k2, max_lines=args.max_lines, thickness=2)
    cv2.imwrite(os.path.join(args.outdir, "matching_viz.png"), viz)

    # Three-stage panel
    pair = _banner(_stack(i1, i2)[0], "Stage 1: input image pair")
    raw = _banner(draw_matches(i1, i2, k1, k2, max_lines=args.max_lines),
                  f"Stage 2: MASt3R correspondences ({len(k1)} raw)")
    ver = _banner(
        draw_matches(i1, i2, k1, k2, mask=inliers, max_lines=args.max_lines),
        f"Stage 3: geometric verification "
        f"(green=inlier {int(inliers.sum())}, red=outlier)",
    )
    w = max(pair.shape[1], raw.shape[1], ver.shape[1])

    def fitw(im):
        if im.shape[1] == w:
            return im
        pad = np.full((im.shape[0], w - im.shape[1], 3), 245, np.uint8)
        return np.hstack([im, pad])

    gap = np.full((16, w, 3), 255, np.uint8)
    stages = np.vstack([fitw(pair), gap, fitw(raw), gap, fitw(ver)])
    cv2.imwrite(os.path.join(args.outdir, "matching_stages.png"), stages)
    print(f"saved matching_viz.png and matching_stages.png to {args.outdir}")


if __name__ == "__main__":
    main()
