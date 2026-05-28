"""Visualize the per-stage outputs of the baseline+moretopk pipeline on a pair.

Following the architecture (and ``baseline+moretopk.yaml``) it runs the real
MASt3R-Hybrid matcher with its ALIKED + SuperPoint extractors on one image pair
and renders the output of each pipeline block as a full-width strip:

  Stage 1.1  Image Pairs            -> the input image pair
  Stage 1.2  Keypoint Detection     -> ALIKED + SuperPoint keypoints
  Stage 2    MASt3R-Hybrid Matcher  -> dense (mkpts) + sparse (matched_idx) matches
  Stage 3    COLMAP Pipeline        -> two-view 3D reconstruction, shown both as a
                                       3D plot AND reprojected (colour=depth) on img1

It also writes an EDA figure contrasting the two Stage-2 branches
(dense ``mkpts`` vs sparse ``matched_idx``).

Outputs:  report/images/matching_stages.png , report/images/eda_dense_sparse.png
"""
from __future__ import annotations

import argparse
import io
import os

import cv2
import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.cm as cm  # noqa: E402

from config import PipelineConfig  # noqa: E402
from matchers.factory import create_point_tracking_matcher  # noqa: E402
from storage import (  # noqa: E402
    InMemoryKeypointStorage,
    InMemoryMatchedKeypointStorage,
    InMemoryMatchingStorage,
)

CFG = "conf/pipeline/imc2025/combination/baseline+moretopk.yaml"
KP_COLORS = [(60, 140, 255), (255, 120, 40)]      # ALIKED orange, SuperPoint blue (BGR)
DENSE_C, SPARSE_C = (60, 200, 60), (40, 120, 240)  # green dense, orange sparse


# --------------------------------------------------------------------------- #
def build_matcher(device):
    cfg = PipelineConfig.parse_obj(yaml.safe_load(open(CFG)))
    ptm = cfg.get_core_config().point_tracking_matchers[0]
    return create_point_tracking_matcher(ptm, device=device), [c.type for c in ptm.local_features]


def run(matcher, p1, p2):
    kpt = InMemoryKeypointStorage(); per = {p1: [], p2: []}
    for p in (p1, p2):
        for ex in matcher.extractors:
            per[p].append(np.asarray(ex(p)[1], np.float32))
        kpt.add(p, np.concatenate(per[p], 0))
    ms, mk = InMemoryMatchingStorage(), InMemoryMatchedKeypointStorage()
    matcher(p1, p2, kpt, ms, mk)
    d1, d2 = mk.get(p1, p2)
    k1, k2 = kpt.get(p1), kpt.get(p2)
    if ms.has(p1, p2):
        idx = ms.get(p1, p2); s1, s2 = k1[idx[:, 0]], k2[idx[:, 1]]
    else:
        s1 = s2 = np.empty((0, 2), np.float32)
    return per, (d1, d2), (s1, s2)


# --------------------------------------------------------------------------- #
def resize_kp(img, kpt_lists, long_edge=640):
    s = long_edge / max(img.shape[:2])
    img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                     interpolation=cv2.INTER_AREA)
    return img, [[k * s for k in lst] for lst in kpt_lists]


def stack(i1, i2):
    h = max(i1.shape[0], i2.shape[0])
    def pad(im): c = np.full((h, im.shape[1], 3), 255, np.uint8); c[: im.shape[0]] = im; return c
    a, b = pad(i1), pad(i2); return np.hstack([a, b]), a.shape[1]


def banner(canvas, text, h=42):
    bar = np.full((h, canvas.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (15, 15, 15), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def draw_lines(canvas, k1, k2, off, color, n=45, t=2):
    if len(k1) == 0: return
    for j in np.linspace(0, len(k1) - 1, min(len(k1), n)).astype(int):
        p = (int(k1[j][0]), int(k1[j][1])); q = (int(k2[j][0]) + off, int(k2[j][1]))
        cv2.line(canvas, p, q, (20, 20, 20), t + 2, cv2.LINE_AA)
        cv2.line(canvas, p, q, color, t, cv2.LINE_AA)
        for c in (p, q):
            cv2.circle(canvas, c, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, 4, color, 1, cv2.LINE_AA)


def to_width(im, w):
    if im.shape[1] == w: return im
    return cv2.resize(im, (w, round(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
def render_recon_panel(rec_dir, w):
    """Render a multi-view COLMAP reconstruction (RGB points + camera poses) from
    two viewpoints, in the style of colmap_output.png; return a BGR image."""
    import pycolmap
    from pathlib import Path
    models = sorted([p for p in Path(rec_dir).iterdir()
                     if p.is_dir() and (p / "points3D.bin").exists()],
                    key=lambda p: int(p.name) if p.name.isdigit() else 0)
    data = []
    for ci, m in enumerate(models):
        rec = pycolmap.Reconstruction(str(m))
        xyz = np.array([p.xyz for p in rec.points3D.values()], float)
        rgb = np.array([p.color for p in rec.points3D.values()], float) / 255
        cams = []
        for im in rec.images.values():
            cfw = im.cam_from_world; cfw = cfw() if callable(cfw) else cfw
            cams.append(np.asarray(cfw.inverse().translation, float))
        if len(xyz):
            data.append((np.array(cams), xyz, rgb))
    allx = np.concatenate([d[1] for d in data], 0)
    allc = np.concatenate([d[0] for d in data], 0)
    n_cam = len(allc)
    c = np.median(np.vstack([allx, allc]), 0)
    r = max(np.percentile(np.linalg.norm(allx - c, axis=1), 75),
            np.linalg.norm(allc - c, axis=1).max() * 1.1)

    def draw(ax, elev, azim, title):
        for cams, xyz, rgb in data:
            keep = np.linalg.norm(xyz - c, axis=1) <= r
            x, col = xyz[keep], rgb[keep]
            if len(x) > 9000:
                sub = np.random.RandomState(0).choice(len(x), 9000, False); x, col = x[sub], col[sub]
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=2.5, c=col, marker=".")
            ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], s=70, c="red", marker="^",
                       edgecolors="black", linewidths=0.8)
        for s, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c): s(m - r, m + r)
        ax.view_init(elev=elev, azim=azim); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(title, fontsize=10)

    fig = plt.figure(figsize=(w / 100, w / 220), dpi=100)
    ax1 = fig.add_subplot(121, projection="3d"); draw(ax1, 18, -60, "perspective view")
    ax2 = fig.add_subplot(122, projection="3d"); draw(ax2, 75, -30, "top-down view")
    fig.suptitle(f"{len(models)} cluster(s),  {n_cam} cameras,  {len(allx)} 3D points",
                 fontsize=10, fontweight="bold")
    buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                                    facecolor="white"); plt.close(fig); buf.seek(0)
    return cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- #
def four_stages(i1, i2, per_ex, dense, sparse, names, rec_dir, out):
    p1k, p2k = per_ex[list(per_ex)[0]], per_ex[list(per_ex)[1]]
    # resize for display (scale all keypoints/matches)
    i1d, (a1, sp1, dd1, ss1) = resize_kp(i1, [p1k[0], p1k[1], dense[0], sparse[0]])
    i2d, (a2, sp2, dd2, ss2) = resize_kp(i2, [p2k[0], p2k[1], dense[1], sparse[1]])

    pair, off = stack(i1d, i2d)
    kp, _ = stack(i1d, i2d)
    for (arr, oset) in [(a1, 0), (a2, off), (sp1, 0), (sp2, off)]:
        col = KP_COLORS[0] if (arr is a1 or arr is a2) else KP_COLORS[1]
        for (x, y) in arr[:: max(1, len(arr) // 500)]:
            cv2.circle(kp, (int(x) + oset, int(y)), 2, col, -1, cv2.LINE_AA)
    mt, _ = stack(i1d, i2d)
    draw_lines(mt, dd1, dd2, off, DENSE_C, n=45)
    draw_lines(mt, ss1, ss2, off, SPARSE_C, n=35)

    # stage 3: the actual multi-view COLMAP reconstruction (colmap_output style)
    s3 = render_recon_panel(rec_dir, pair.shape[1])

    rows = [
        banner(pair, "Stage 1.1  |  Image Pairs output: input pair (img1, img2)"),
        banner(kp, f"Stage 1.2  |  Keypoint Detection: {names[0]} (orange) + {names[1]} (blue)"),
        banner(mt, f"Stage 2  |  MASt3R-Hybrid Matcher: {len(dense[0])} dense (mkpts, green)"
                   f" + {len(sparse[0])} sparse (matched_idx, orange)"),
        banner(s3, "Stage 3  |  COLMAP Pipeline output: 3D reconstruction "
                   "(RGB points) + camera poses (red triangle)"),
    ]
    W = max(r.shape[1] for r in rows)
    gap = np.full((10, W, 3), 255, np.uint8)
    canvas = []
    for r in rows: canvas += [to_width(r, W), gap]
    cv2.imwrite(out, np.vstack(canvas[:-1])); print(f"saved {out}")


def eda(i1, dense, sparse, out):
    d1, s1 = dense[0], sparse[0]
    nd, ns = len(d1), len(s1)
    dd = np.linalg.norm(dense[0] - dense[1], axis=1) if nd else np.array([0.])
    ds = np.linalg.norm(sparse[0] - sparse[1], axis=1) if ns else np.array([0.])
    W = 900; sc = W / i1.shape[1]; disp = cv2.resize(i1, (W, round(i1.shape[0] * sc)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].bar(["dense\n(mkpts)", "sparse\n(matched_idx)"], [nd, ns],
              color=["#37b24d", "#f08c00"], edgecolor="black")
    for i, v in enumerate([nd, ns]): ax[0].text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    ax[0].set_title("(a) #matches per branch", fontweight="bold"); ax[0].set_ylabel("matches")
    ax[1].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
    if nd: ax[1].scatter(d1[:, 0] * sc, d1[:, 1] * sc, s=4, c="#37b24d", label=f"dense ({nd})", alpha=.6)
    if ns: ax[1].scatter(s1[:, 0] * sc, s1[:, 1] * sc, s=16, c="#f08c00", marker="x", label=f"sparse ({ns})")
    ax[1].set_title("(b) spatial coverage on img1", fontweight="bold"); ax[1].axis("off"); ax[1].legend(fontsize=9)
    bins = np.linspace(0, max(dd.max(), ds.max(), 1), 40)
    ax[2].hist(dd, bins=bins, color="#37b24d", alpha=.6, label="dense", density=True)
    ax[2].hist(ds, bins=bins, color="#f08c00", alpha=.6, label="sparse", density=True)
    ax[2].set_title("(c) match displacement |p1-p2| (px)", fontweight="bold")
    ax[2].set_xlabel("displacement (px)"); ax[2].set_ylabel("density"); ax[2].legend()
    fig.suptitle("Stage-2 EDA: dense (mkpts) vs sparse (matched_idx)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", default="data/train/imc2023_heritage/cyprus_dsc_6488.png")
    ap.add_argument("--img2", default="data/train/imc2023_heritage/cyprus_dsc_6512.png")
    ap.add_argument("--rec_dir", default="/tmp/heritage_saved/colmap_rec")
    ap.add_argument("--outdir", default="report/images")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher, names = build_matcher(device)
    per, dense, sparse = run(matcher, args.img1, args.img2)
    print(f"keypoints {names} | dense={len(dense[0])} sparse={len(sparse[0])}")
    i1, i2 = cv2.imread(args.img1), cv2.imread(args.img2)
    four_stages(i1, i2, per, dense, sparse, names, args.rec_dir,
                os.path.join(args.outdir, "matching_stages.png"))
    eda(i1, dense, sparse, os.path.join(args.outdir, "eda_dense_sparse.png"))


if __name__ == "__main__":
    main()
