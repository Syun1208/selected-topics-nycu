#!/usr/bin/env python3
"""
augment.py - Strong data augmentation for instance segmentation (class balancing).

Augmentation pipeline:
  - HSV color jitter
  - Random horizontal/vertical flip
  - RandomAffine (rotation, scale, translation, shear)
  - Mosaic (4-image grid)              -- disable with --no-mosaic
  - MixUp (alpha-blend two samples)    -- disable with --no-mixup
  - Copy-Paste (paste instances across samples)

Training tip: disable Mosaic & MixUp for the last 10-20 epochs.
  Early epochs  : python augment.py ...               (mosaic + mixup ON)
  Final epochs  : python augment.py ... --no-mosaic --no-mixup
"""

import os
# Must be set before numpy/cv2 are imported to prevent OpenMP thread pool deadlocks
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["NUMEXPR_NUM_THREADS"]  = "1"

import sys
import argparse
import random
import uuid
import shutil
import multiprocessing as mp
from copy import deepcopy


class _Timeout(Exception):
    pass

def _worker(q, fn_args):
    """Subprocess worker: unpickle args, run augment+save, put result in queue."""
    try:
        result = fn_args()
        q.put(("ok", result))
    except Exception as e:
        q.put(("err", str(e)))

def run_with_timeout(fn, seconds=8):
    """Run fn() in a subprocess; raise _Timeout if it exceeds `seconds`.
    Uses multiprocessing so OS-level kill reliably interrupts C extensions."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(q, fn))
    p.start()
    p.join(timeout=seconds)
    if p.is_alive():
        p.kill()
        p.join()
        raise _Timeout()
    try:
        status, value = q.get(timeout=2)
    except Exception:
        raise _Timeout()
    if status == "err":
        raise RuntimeError(value)
    return value

import numpy as np
import tifffile
import cv2
cv2.setNumThreads(1)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_sample(folder: str):
    """Return (image_uint8_HWXC, masks_dict) where masks_dict[cls] = float64 H×W."""
    MAX_SIDE = 1024  # cap large images to avoid slow Mosaic/MixUp on huge inputs
    img = tifffile.imread(os.path.join(folder, "image.tif"))
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        nh, nw = h, w

    masks = {}
    for fname in os.listdir(folder):
        if fname.startswith("class") and fname.endswith(".tif"):
            cls = int(fname[5:-4])
            m = tifffile.imread(os.path.join(folder, fname)).astype(np.float64)
            if (h, w) != (nh, nw):
                m = cv2.resize(m.astype(np.float32), (nw, nh),
                               interpolation=cv2.INTER_NEAREST).astype(np.float64)
            masks[cls] = m
    return img, masks


def save_sample(folder: str, image: np.ndarray, masks: dict):
    # Masks saved as uint16 (IDs fit in 0-65535) with LZW compression.
    # float64 → uint16 + LZW cuts mask file size ~6-8x vs original float64 uncompressed.
    os.makedirs(folder, exist_ok=True)
    tifffile.imwrite(os.path.join(folder, "image.tif"),
                     np.ascontiguousarray(image.astype(np.uint8)),
                     compression='lzw')
    for cls, mask in masks.items():
        m = np.ascontiguousarray(np.clip(mask, 0, 65535).astype(np.uint16))
        tifffile.imwrite(os.path.join(folder, f"class{cls}.tif"), m, compression='lzw')


# ---------------------------------------------------------------------------
# Geometric helpers (always INTER_NEAREST on masks to preserve IDs)
# ---------------------------------------------------------------------------

def _resize_image(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask, w, h):
    return cv2.resize(mask.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(np.float64)


def _resize_masks(masks, w, h):
    return {c: _resize_mask(m, w, h) for c, m in masks.items()}


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def aug_hsv(image: np.ndarray,
            h_gain: float = 0.015,
            s_gain: float = 0.7,
            v_gain: float = 0.4) -> np.ndarray:
    """Apply random HSV jitter to the first 3 channels (RGB)."""
    if image.ndim < 3 or image.shape[2] < 3:
        return image
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    rgb = image[:, :, :3].copy()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    dtype = rgb.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * r[0]) % 180).astype(dtype)
    lut_s = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_v = np.clip(x * r[2], 0, 255).astype(dtype)
    hsv_out = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    result = image.copy()
    result[:, :, :3] = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
    return result


def aug_flip(image: np.ndarray, masks: dict, flip_code: int):
    """flip_code: 0=vertical, 1=horizontal, -1=both."""
    new_img = cv2.flip(image, flip_code)
    new_masks = {c: cv2.flip(m, flip_code) for c, m in masks.items()}
    return new_img, new_masks


def aug_affine(image: np.ndarray, masks: dict,
               max_angle: float = 15.0,
               scale_range=(0.8, 1.2),
               translate_frac: float = 0.1,
               shear_range: float = 5.0):
    """Random rotation + scale + translation + shear around image centre."""
    h, w = image.shape[:2]
    angle  = random.uniform(-max_angle, max_angle)
    scale  = random.uniform(*scale_range)
    tx     = random.uniform(-translate_frac, translate_frac) * w
    ty     = random.uniform(-translate_frac, translate_frac) * h
    shear  = random.uniform(-shear_range, shear_range)

    cx, cy = w / 2, h / 2
    # Rotation + scale matrix
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    # Add shear component
    shear_rad = np.deg2rad(shear)
    M_shear = np.array([[1, np.tan(shear_rad), 0],
                         [0, 1, 0]], dtype=np.float32)
    # Combine: first shear, then rotate/scale/translate
    M3 = np.vstack([M, [0, 0, 1]])
    Ms = np.vstack([M_shear, [0, 0, 1]])
    Mc = (M3 @ Ms)[:2]

    new_img = cv2.warpAffine(image, Mc, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    new_masks = {}
    for c, m in masks.items():
        warped = cv2.warpAffine(m.astype(np.float32), Mc, (w, h),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        new_masks[c] = warped.astype(np.float64)
    return new_img, new_masks


def _gaussian_blur_mask(binary: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    """Soft alpha mask for copy-paste blending (replaces scipy.gaussian)."""
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (ksize, ksize), sigma)
    return blurred.astype(np.float64)


def aug_copy_paste(tgt_img: np.ndarray, tgt_masks: dict,
                   src_pool: list,        # list of (img, masks)
                   target_cls: int,
                   max_paste: int = 3,
                   blend: bool = True) -> tuple:
    """
    Copy instances of `target_cls` from random source samples into target.
    Returns updated (image, masks).
    """
    img = tgt_img.copy()
    masks = deepcopy(tgt_masks)
    h, w = img.shape[:2]

    if target_cls not in masks:
        masks[target_cls] = np.zeros((h, w), dtype=np.float64)
    max_id = float(masks[target_cls].max())

    n_paste = random.randint(1, max_paste)
    for _ in range(n_paste):
        src_img, src_masks = random.choice(src_pool)
        if target_cls not in src_masks:
            continue
        src_mask = src_masks[target_cls]
        instance_ids = [v for v in np.unique(src_mask) if v > 0]
        if not instance_ids:
            continue
        inst_id = random.choice(instance_ids)
        binary = (src_mask == inst_id).astype(np.uint8)

        # Resize source to match target
        if src_img.shape[:2] != (h, w):
            src_img_r = _resize_image(src_img, w, h)
            binary    = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            src_img_r = src_img

        # Blend
        alpha = _gaussian_blur_mask(binary) if blend else binary.astype(np.float64)
        alpha3 = alpha[..., None]
        img = (src_img_r * alpha3 + img * (1 - alpha3)).astype(np.uint8)

        max_id += 1.0
        masks[target_cls][binary > 0] = max_id

    return img, masks


def aug_mosaic(samples: list, out_h: int, out_w: int) -> tuple:
    """
    Combine exactly 4 (img, masks) samples into a 2×2 mosaic.
    Instance IDs are globally shifted per class to avoid collisions.
    Returns (mosaic_img, mosaic_masks).
    """
    assert len(samples) == 4
    cx = int(random.uniform(0.35, 0.65) * out_w)
    cy = int(random.uniform(0.35, 0.65) * out_h)

    all_classes = set()
    for img, masks in samples:
        all_classes.update(masks.keys())

    c = samples[0][0].shape[2] if samples[0][0].ndim == 3 else 1
    mosaic_img = np.zeros((out_h, out_w, c), dtype=np.uint8)
    mosaic_masks = {cls: np.zeros((out_h, out_w), dtype=np.float64) for cls in all_classes}
    id_offsets = {cls: 0.0 for cls in all_classes}

    regions = [
        (0, 0, cy, cx),
        (0, cx, cy, out_w),
        (cy, 0, out_h, cx),
        (cy, cx, out_h, out_w),
    ]

    for (src_img, src_masks), (r0, c0, r1, c1) in zip(samples, regions):
        rh, rw = r1 - r0, c1 - c0
        mosaic_img[r0:r1, c0:c1] = _resize_image(src_img, rw, rh)

        for cls in all_classes:
            if cls in src_masks:
                m = _resize_mask(src_masks[cls], rw, rh)
                max_id = float(m.max())
                shifted = np.where(m > 0, m + id_offsets[cls], 0.0)
                mosaic_masks[cls][r0:r1, c0:c1] = shifted
                if max_id > 0:
                    id_offsets[cls] += max_id

    return mosaic_img, mosaic_masks


def aug_mixup(img1, masks1, img2, masks2, alpha: float = 0.5) -> tuple:
    """
    Alpha-blend img1 and img2. Masks are unioned with shifted IDs.
    Returns (mixed_img, mixed_masks).
    """
    lam = float(np.random.beta(alpha, alpha))
    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])

    i1 = _resize_image(img1, w, h)
    i2 = _resize_image(img2, w, h)
    mixed = (i1 * lam + i2 * (1 - lam)).astype(np.uint8)

    all_classes = set(masks1.keys()) | set(masks2.keys())
    mixed_masks = {}
    for cls in all_classes:
        m = np.zeros((h, w), dtype=np.float64)
        offset = 0.0
        if cls in masks1:
            m1 = _resize_mask(masks1[cls], w, h)
            m += np.where(m1 > 0, m1, 0.0)
            offset = float(m1.max())
        if cls in masks2:
            m2 = _resize_mask(masks2[cls], w, h)
            m2s = np.where(m2 > 0, m2 + offset, 0.0)
            # m1 regions take priority; fill only where m == 0
            m = np.where((m == 0) & (m2s > 0), m2s, m)
        mixed_masks[cls] = m

    return mixed, mixed_masks


# ---------------------------------------------------------------------------
# Composite strong augmentation pipeline
# ---------------------------------------------------------------------------

def strong_augment(src_img: np.ndarray, src_masks: dict,
                   pool: list,
                   use_mosaic: bool = True,
                   use_mixup: bool = True,
                   copy_paste_classes: list = None) -> tuple:
    """
    Apply a random combination of strong augmentations.
    pool: list of (img, masks) for other samples (used in Mosaic/MixUp/CopyPaste).
    Returns (aug_img, aug_masks).
    """
    img, masks = src_img.copy(), deepcopy(src_masks)

    # 1. HSV jitter (always)
    img = aug_hsv(img)

    # 2. Random flip (~70% chance)
    if random.random() < 0.7:
        flip_code = random.choice([0, 1, -1])
        img, masks = aug_flip(img, masks, flip_code)

    # 3. Random affine (~70% chance)
    if random.random() < 0.7:
        img, masks = aug_affine(img, masks)

    # 4. Mosaic (50% if enabled and pool has enough samples)
    if use_mosaic and random.random() < 0.5 and len(pool) >= 3:
        others = random.sample(pool, 3)
        four = [(img, masks)] + others
        img, masks = aug_mosaic(four, src_img.shape[0], src_img.shape[1])

    # 5. MixUp (30% if enabled and mosaic was not applied)
    elif use_mixup and random.random() < 0.3 and len(pool) >= 1:
        other_img, other_masks = random.choice(pool)
        img, masks = aug_mixup(img, masks, other_img, other_masks)

    # 6. Copy-Paste (40% per class, if copy_paste_classes specified)
    if copy_paste_classes:
        for cls in copy_paste_classes:
            if random.random() < 0.4:
                cls_pool = [(pi, pm) for pi, pm in pool if cls in pm]
                if cls_pool:
                    img, masks = aug_copy_paste(img, masks, cls_pool, cls)

    return img, masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Balance instance-segmentation classes via strong augmentation.")
    p.add_argument("--data-dir",   default="/home/caotulab/Desktop/Long/data/train",
                   help="Source training directory (default: %(default)s)")
    p.add_argument("--path-save",  required=True,
                   help="Output directory for the balanced dataset")
    p.add_argument("--target",     type=int, default=None,
                   help="Target number of samples per class (e.g. 500)")
    p.add_argument("--target-instances", type=int, default=None,
                   help="Target instances per class; sample target is computed from per-class density")
    p.add_argument("--gpu-id",     type=int, default=0,
                   help="GPU device ID exposed via CUDA_VISIBLE_DEVICES (default: 0)")
    p.add_argument("--seed",       type=int, default=42, help="Random seed")
    p.add_argument("--no-mosaic",  action="store_true",
                   help="Disable Mosaic augmentation (recommended for last 10-20 training epochs)")
    p.add_argument("--no-mixup",   action="store_true",
                   help="Disable MixUp augmentation (recommended for last 10-20 training epochs)")
    p.add_argument("--copy-only",  action="store_true",
                   help="Only copy originals without generating augmented samples (dry-run layout)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.target is None and args.target_instances is None:
        print("ERROR: must specify --target or --target-instances", file=sys.stderr)
        sys.exit(1)

    data_dir  = args.data_dir
    save_dir  = args.path_save
    use_mosaic = not args.no_mosaic
    use_mixup  = not args.no_mixup

    target_mode = (f"{args.target_instances} instances/class"
                   if args.target_instances else f"{args.target} samples/class")

    print(f"{'='*60}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Save dir   : {save_dir}")
    print(f"  Target     : {target_mode}")
    print(f"  GPU id     : {args.gpu_id}")
    print(f"  Mosaic     : {'OFF' if args.no_mosaic else 'ON'}")
    print(f"  MixUp      : {'OFF' if args.no_mixup else 'ON'}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ load
    sample_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )

    all_samples = []   # list of (folder_name, img, masks)
    for d in tqdm(sample_dirs, desc="Loading", unit="sample"):
        try:
            img, masks = load_sample(os.path.join(data_dir, d))
            all_samples.append((d, img, masks))
        except Exception as e:
            tqdm.write(f"  [WARN] skip {d}: {e}")

    print(f"Loaded {len(all_samples)} samples.\n")

    # ---------------------------------------------------- class distribution
    class_to_indices = {}   # cls -> [index in all_samples]
    for i, (_, _, masks) in enumerate(all_samples):
        for cls in masks:
            class_to_indices.setdefault(cls, []).append(i)

    # Count instances per class (for display and instance-based targets)
    class_instance_counts = {}
    for _, _, masks in all_samples:
        for cls, mask in masks.items():
            n_inst = len([v for v in np.unique(mask) if v > 0])
            class_instance_counts[cls] = class_instance_counts.get(cls, 0) + n_inst

    print("Original class distribution:")
    for cls in sorted(class_to_indices):
        n_samp = len(class_to_indices[cls])
        n_inst = class_instance_counts.get(cls, 0)
        print(f"  class{cls}: {n_samp} samples  |  {n_inst:,} instances  ({n_inst/n_samp:.1f} inst/sample)")
    print()

    # Compute per-class sample targets
    if args.target_instances:
        class_sample_targets = {}
        print("Instance-based sample targets:")
        for cls, indices in sorted(class_to_indices.items()):
            avg_inst = class_instance_counts.get(cls, 1) / len(indices)
            cls_target = max(len(indices), int(np.ceil(args.target_instances / avg_inst)))
            class_sample_targets[cls] = cls_target
            print(f"  class{cls}: {len(indices)} → {cls_target} samples  (avg {avg_inst:.1f} inst/sample)")
        print()
    else:
        class_sample_targets = {cls: args.target for cls in class_to_indices}

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------- copy original samples to output folder
    # Use shutil.copytree to copy raw files directly — avoids decode→re-encode overhead
    for fname, _, _ in tqdm(all_samples, desc="Copying originals", unit="sample"):
        src = os.path.join(data_dir, fname)
        dst = os.path.join(save_dir, fname)
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
    print()

    if args.copy_only:
        print("--copy-only set, skipping augmentation.")
        return

    # ------------------------------------------- augment under-represented classes
    # Pool for augmentation: (img, masks) tuples (no names needed)
    pool_all = [(img, masks) for _, img, masks in all_samples]

    for cls in sorted(class_to_indices):
        existing  = len(class_to_indices[cls])
        cls_target = class_sample_targets[cls]
        needed    = cls_target - existing

        if needed <= 0:
            print(f"class{cls}: {existing} >= {cls_target}, skipping.")
            continue

        cls_indices = class_to_indices[cls]
        cls_pool    = [pool_all[i] for i in cls_indices]

        pbar = tqdm(range(needed), desc=f"class{cls} ({existing}→{cls_target})", unit="img")
        for i in pbar:
            src_img, src_masks = random.choice(cls_pool)
            local_pool = [s for s in pool_all if s is not (src_img, src_masks)]

            new_id = str(uuid.uuid4())
            out_path = os.path.join(save_dir, new_id)

            # Define task inside loop; fork captures current iteration's variables.
            # Both augment AND save run in the subprocess so tifffile LZW write
            # is also covered by the timeout (not just strong_augment).
            def _task():
                a_img, a_masks = strong_augment(
                    src_img, src_masks,
                    pool=local_pool,
                    use_mosaic=use_mosaic,
                    use_mixup=use_mixup,
                    copy_paste_classes=list(src_masks.keys()),
                )
                if cls not in a_masks or float(a_masks[cls].max()) == 0:
                    a_img = aug_hsv(src_img.copy())
                    a_masks = deepcopy(src_masks)
                    a_img, a_masks = aug_flip(a_img, a_masks, 1)
                save_sample(out_path, a_img, a_masks)

            try:
                run_with_timeout(_task, seconds=15)
            except (_Timeout, Exception) as e:
                if isinstance(e, _Timeout):
                    tqdm.write(f"  [WARN] sample {i} timeout (>15s), using fallback.")
                else:
                    tqdm.write(f"  [WARN] augment error (sample {i}): {e}. Using fallback.")
                fb_img = aug_hsv(src_img.copy())
                fb_masks = deepcopy(src_masks)
                fb_img, fb_masks = aug_flip(fb_img, fb_masks, random.choice([0, 1, -1]))
                save_sample(out_path, fb_img, fb_masks)
        print()

    # --------------------------------------------------------- final summary
    print("Final class distribution in output:")
    final = {}
    for d in os.listdir(save_dir):
        dpath = os.path.join(save_dir, d)
        if not os.path.isdir(dpath):
            continue
        for f in os.listdir(dpath):
            if f.startswith("class") and f.endswith(".tif"):
                c = int(f[5:-4])
                final[c] = final.get(c, 0) + 1
    for cls in sorted(final):
        print(f"  class{cls}: {final[cls]} samples")

    print(f"\nDone! Augmented dataset → {save_dir}")


if __name__ == "__main__":
    main()
