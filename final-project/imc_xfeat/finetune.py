"""Self-supervised fine-tuning of XFeat on the IMC training images.

The IMC-2025 training images come with ground-truth camera poses but **no depth
maps and no camera intrinsics**, so the original XFeat correspondence loss
(which needs depth to lift pixels into 3D) cannot be used directly.

Instead this script fine-tunes XFeat with *self-supervised homography warps*:
every image is warped by a randomly generated homography, which yields exact
dense pixel correspondences for free. Training on these pairs adapts XFeat's
descriptors / reliability head to the visual domain of the IMC scenes
(buildings, statues, gardens, ...) without requiring any extra labels.

Run via ``train.sh`` (recommended) or directly:

    python -m imc_xfeat.finetune --train-dir data/image-matching/train --gpu-ids 0
"""

import argparse
import glob
import os
import random
import time


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune XFeat on IMC images (self-supervised homography warps).")
    # data / weights
    p.add_argument("--train-dir", required=True,
                   help="Folder with <scene>/<image> subfolders (the IMC train set).")
    p.add_argument("--weights", default="",
                   help="Initial XFeat weights (default: bundled accelerated_features/weights/xfeat.pt).")
    p.add_argument("--ckpt-dir", default="imc_xfeat/checkpoints",
                   help="Directory where fine-tuned checkpoints are saved.")
    # hardware
    p.add_argument("--gpu-ids", default="0",
                   help="Comma-separated GPU ids, e.g. '0' or '0,1'. Empty string = CPU.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes.")
    # optimisation hyper-parameters
    p.add_argument("--batch-size", type=int, default=8, help="Images per training step.")
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    p.add_argument("--steps", type=int, default=2000, help="Total number of optimisation steps.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient-norm clipping value.")
    # task hyper-parameters
    p.add_argument("--train-res", default="800,608",
                   help="Training resolution as W,H (both must be divisible by 32).")
    p.add_argument("--difficulty", type=float, default=0.3,
                   help="Homography warp strength (0.1 easy ... 0.5 hard).")
    p.add_argument("--max-corr", type=int, default=2048,
                   help="Max correspondences per image used in the loss.")
    # logging
    p.add_argument("--save-every", type=int, default=500, help="Checkpoint interval (steps).")
    p.add_argument("--log-every", type=int, default=20, help="Console log interval (steps).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


args = parse_args()
# Must be set before torch sees the GPUs.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_ids)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from imc_xfeat.xfeat_utils import DEFAULT_WEIGHTS, load_xfeat_model

IMAGE_GLOBS = ("*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG")


# --------------------------------------------------------------------------
# Loss (copied from accelerated_features/modules/training/losses.py so that we
# do not drag in its MegaDepth / ALIKE imports, which need extra dependencies).
# --------------------------------------------------------------------------
def dual_softmax_loss(X, Y, temp=0.2):
    """Contrastive matching loss over a set of ground-truth correspondences."""
    dist = (X @ Y.t()) * temp
    conf12 = F.log_softmax(dist, dim=1)
    conf21 = F.log_softmax(dist.t(), dim=1)
    with torch.no_grad():
        conf = conf12.exp().max(dim=-1)[0] * conf21.exp().max(dim=-1)[0]
    target = torch.arange(len(X), device=X.device)
    loss = F.nll_loss(conf12, target) + F.nll_loss(conf21, target)
    return loss, conf


# --------------------------------------------------------------------------
# Synthetic homography generation
# --------------------------------------------------------------------------
def random_homography(w, h, mult=0.3):
    """Random homography (rotation + scale + shear + small perspective + shift)."""
    rng = np.random
    scale = min(1.0, mult / 0.3)
    theta = np.radians(rng.uniform(-25, 25) * scale)
    sx, sy = rng.uniform(0.75, 1.25, 2)
    cx, cy = w / 2.0, h / 2.0
    c, s = np.cos(theta), np.sin(theta)

    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)
    scl = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)
    shear = rng.normal(0, 0.1 * mult, 2)
    aff = np.array([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]], float)
    persp = rng.normal(0, 5e-4 * mult, 2)
    prj = np.array([[1, 0, 0], [0, 1, 0], [persp[0], persp[1], 1]], float)
    tx = rng.uniform(-0.12, 0.12) * w * scale
    ty = rng.uniform(-0.12, 0.12) * h * scale
    back = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], float)

    H = back @ scl @ prj @ aff @ rot @ to_origin
    return H.astype(np.float32)


class HomographyPairs(torch.utils.data.Dataset):
    """Yields (image, warped image, homography) triplets for self-supervision."""

    def __init__(self, image_paths, res, difficulty):
        self.paths = image_paths
        self.w, self.h = res
        self.difficulty = difficulty

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if img is None:  # unreadable file -> blank image, will be skipped by the loss
            img = np.zeros((self.h, self.w, 3), np.uint8)
        img = cv2.resize(img, (self.w, self.h))
        H = random_homography(self.w, self.h, self.difficulty)
        warped = cv2.warpPerspective(img, H, (self.w, self.h),
                                     flags=cv2.INTER_LINEAR, borderValue=0)
        t1 = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t2 = torch.from_numpy(warped).permute(2, 0, 1).float() / 255.0
        return t1, t2, torch.from_numpy(H)


def coarse_correspondences(H, w, h):
    """Map each 1/8-resolution cell of image1 to its cell in image2 under ``H``.

    Returns two index arrays into the flattened (H/8 x W/8) coarse grids.
    """
    wc, hc = w // 8, h // 8
    ys, xs = np.mgrid[0:hc, 0:wc]
    # pixel centre of every coarse cell in image1
    px = xs.reshape(-1) * 8 + 4
    py = ys.reshape(-1) * 8 + 4
    pts = np.stack([px, py, np.ones_like(px)], 0).astype(np.float64)
    warped = H.astype(np.float64) @ pts
    warped = warped[:2] / warped[2:3]
    x2, y2 = warped[0], warped[1]

    valid = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)
    cx2 = np.clip((x2 // 8).astype(np.int64), 0, wc - 1)
    cy2 = np.clip((y2 // 8).astype(np.int64), 0, hc - 1)
    i1 = np.arange(hc * wc)[valid]
    i2 = (cy2 * wc + cx2)[valid]
    # keep a single source cell per target cell (approximate one-to-one matching)
    uniq2, first = np.unique(i2, return_index=True)
    return i1[first], uniq2


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    w, h = (int(v) for v in args.train_res.split(","))
    if w % 32 or h % 32:
        raise SystemExit("--train-res W,H must both be divisible by 32")

    gpu_ids = [g for g in args.gpu_ids.split(",") if g.strip()]
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = load_xfeat_model(args.weights or DEFAULT_WEIGHTS, device=device)
    net.train()
    train_net = net
    if use_cuda and torch.cuda.device_count() > 1:
        train_net = torch.nn.DataParallel(net)
        print(f"[finetune] DataParallel over {torch.cuda.device_count()} GPUs")

    paths = []
    for ext in IMAGE_GLOBS:
        paths += glob.glob(os.path.join(args.train_dir, "*", ext))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(f"No images found under {args.train_dir}")
    print(f"[finetune] {len(paths)} images | device={device} | batch={args.batch_size} "
          f"| steps={args.steps} | res={w}x{h}")

    loader = torch.utils.data.DataLoader(
        HomographyPairs(paths, (w, h), args.difficulty),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=use_cuda)

    opt = torch.optim.Adam(train_net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, args.steps // 3), gamma=0.5)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    latest = os.path.join(args.ckpt_dir, "xfeat_imc_latest.pt")

    step, running, t0 = 0, 0.0, time.time()
    done = False
    while not done:
        for t1, t2, Hs in loader:
            t1, t2 = t1.to(device), t2.to(device)
            feats1, _, hmap1 = train_net(t1)
            feats2, _, _ = train_net(t2)
            _, C, hc, wc = feats1.shape

            losses = []
            for b in range(t1.shape[0]):
                i1, i2 = coarse_correspondences(Hs[b].numpy(), w, h)
                if len(i1) < 32:
                    continue
                if len(i1) > args.max_corr:
                    sel = np.random.choice(len(i1), args.max_corr, replace=False)
                    i1, i2 = i1[sel], i2[sel]
                i1 = torch.from_numpy(i1).to(device)
                i2 = torch.from_numpy(i2).to(device)
                f1 = feats1[b].reshape(C, hc * wc).t()[i1]
                f2 = feats2[b].reshape(C, hc * wc).t()[i2]
                loss_ds, conf = dual_softmax_loss(f1, f2)
                # reliability head: high response where matching is confident
                rel = hmap1[b].reshape(-1)[i1]
                loss_kp = F.l1_loss(rel, conf) * 3.0
                losses.append(loss_ds + loss_kp)

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_net.parameters(), args.grad_clip)
            opt.step()
            scheduler.step()

            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"[finetune] step {step:6d}/{args.steps} | "
                      f"loss {running / args.log_every:.4f} | "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | "
                      f"{(time.time() - t0) / step:.2f}s/step")
                running = 0.0
            if step % args.save_every == 0 or step >= args.steps:
                ckpt = os.path.join(args.ckpt_dir, f"xfeat_imc_{step:06d}.pt")
                torch.save(net.state_dict(), ckpt)
                torch.save(net.state_dict(), latest)
                print(f"[finetune] saved {ckpt}")
            if step >= args.steps:
                done = True
                break

    print(f"[finetune] finished {step} steps in {time.time() - t0:.1f}s -> {latest}")


if __name__ == "__main__":
    main()
