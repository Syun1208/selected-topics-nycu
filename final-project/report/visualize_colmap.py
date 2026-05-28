"""Visualize the COLMAP-pipeline output: the (reconstruction, cluster_label) list.

A scene's ``colmap_rec`` directory contains one sub-model per disconnected
reconstruction (``0/``, ``1/`` ...); each sub-model is one predicted scene
cluster. This script loads every sub-model and renders, in a single 3D view:

  * the sparse 3D point cloud (coloured by its own RGB, or per cluster), and
  * the recovered camera centres (red markers), i.e. the estimated poses.

Output:  report/images/colmap_output.png

Example:
  python report/visualize_colmap.py --rec_dir /tmp/.../colmap_rec --outdir report/images
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pycolmap  # noqa: E402

CLUSTER_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def camera_center(image) -> np.ndarray:
    """World-space camera centre, robust to pycolmap 3.x/4.x."""
    cfw = image.cam_from_world
    cfw = cfw() if callable(cfw) else cfw
    return np.asarray(cfw.inverse().translation, dtype=float)


def load_model(path: str):
    rec = pycolmap.Reconstruction(path)
    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=float)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=float) / 255.0
    centers = np.array([camera_center(im) for im in rec.images.values()], dtype=float)
    return xyz, rgb, centers


def robust_lims(xyz: np.ndarray, q=2.0):
    """Percentile-based axis limits to ignore far outlier points."""
    lo, hi = np.percentile(xyz, [q, 100 - q], axis=0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec_dir", required=True,
                    help="a colmap_rec directory containing sub-models 0/,1/,...")
    ap.add_argument("--outdir", default="report/images")
    ap.add_argument("--max_points", type=int, default=8000)
    ap.add_argument("--color", choices=["rgb", "cluster"], default="rgb")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    models = sorted(
        [p for p in Path(args.rec_dir).iterdir()
         if p.is_dir() and (p / "points3D.bin").exists()],
        key=lambda p: int(p.name) if p.name.isdigit() else 0,
    )
    if not models:
        raise SystemExit(f"No sub-models found under {args.rec_dir}")

    # pre-load and clean every model once
    data = []
    for ci, m in enumerate(models):
        xyz, rgb, centers = load_model(str(m))
        if len(xyz) == 0:
            continue
        if len(xyz) > args.max_points:
            sel = np.random.RandomState(0).choice(len(xyz), args.max_points, False)
            xyz, rgb = xyz[sel], rgb[sel]
        lo, hi = robust_lims(xyz)
        keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        data.append((ci, xyz[keep], rgb[keep], centers))

    # Centre on the cameras (robust) and zoom into the dense structure around
    # them, dropping far SfM outlier points that otherwise shrink the object.
    cam_all = np.concatenate([d[3] for d in data], 0)
    all_xyz = np.concatenate([d[1] for d in data], 0)
    n_cam = len(cam_all)
    c = np.median(np.concatenate([all_xyz, cam_all], 0), axis=0)
    r = np.percentile(np.linalg.norm(all_xyz - c, axis=1), 70)
    r = max(r, np.linalg.norm(cam_all - c, axis=1).max() * 1.1)

    def draw(ax, elev, azim):
        for ci, xyz, rgb, centers in data:
            keep = np.linalg.norm(xyz - c, axis=1) <= r
            xyz, rgb = xyz[keep], rgb[keep]
            col = rgb if args.color == "rgb" else CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=4.0, c=col, marker=".",
                       depthshade=True, label=f"cluster {ci} ({len(centers)} cams)")
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=110,
                       c=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)], marker="^",
                       edgecolors="black", linewidths=1.0, depthshade=False)
        for setlim, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), c):
            setlim(mid - r, mid + r)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection="3d"); draw(ax1, 18, -60)
    ax2 = fig.add_subplot(122, projection="3d"); draw(ax2, 70, -30)
    ax1.set_title("view 1 (▲ = camera pose)", fontsize=11)
    ax2.set_title("view 2 (top-down)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9, markerscale=4)
    fig.suptitle(f"COLMAP output: {len(models)} cluster(s), {n_cam} cameras, "
                 f"{len(all_xyz)} points", fontsize=13, fontweight="bold")
    out = os.path.join(args.outdir, "colmap_output.png")
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"clusters={len(models)} cameras={n_cam} points={len(all_xyz)} -> {out}")


if __name__ == "__main__":
    main()
