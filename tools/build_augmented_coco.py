#!/usr/bin/env python3
"""Convert the per-instance augmented dataset produced by augment.py into a
COCO-format dataset that train.sh / mmdet can consume.

Input layout (one directory per sample, written by augment.py):
    <augmented-train>/<uuid>/image.tif
    <augmented-train>/<uuid>/class<N>.tif   # uint16; 0 = bg, >0 = instance id

Output layout:
    <out>/images/train/<uuid>.tif           # symlink -> <augmented-train>/<uuid>/image.tif
    <out>/images/val/                       # symlink -> <reference-coco>/images/val
    <out>/annotations/train.json            # generated from masks
    <out>/annotations/val.json              # symlink -> <reference-coco>/annotations/val.json
"""

import argparse
import json
import multiprocessing as mp
import os
import os.path as osp
import sys
from functools import partial

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

cv2.setNumThreads(1)


CATEGORIES = [
    {'id': 1, 'name': 'class1', 'supercategory': 'object'},
    {'id': 2, 'name': 'class2', 'supercategory': 'object'},
    {'id': 3, 'name': 'class3', 'supercategory': 'object'},
    {'id': 4, 'name': 'class4', 'supercategory': 'object'},
]

MIN_RING_POINTS = 3  # COCO polygon ring needs >= 3 (x,y) pairs


def _instance_polygons(binary):
    """Return (polygons, area, bbox) for one instance, or None if too small.

    `binary` is a uint8 mask (0/1). Multiple disconnected components are kept
    as separate polygon rings within the same annotation.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if c.shape[0] < MIN_RING_POINTS:
            continue
        polys.append(c.reshape(-1).astype(float).tolist())
    if not polys:
        return None

    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    area = int(binary.sum())
    return polys, area, bbox


def _process_sample(uid, train_dir):
    folder = osp.join(train_dir, uid)
    img_path = osp.join(folder, 'image.tif')
    if not osp.exists(img_path):
        return uid, None, []
    try:
        img = tifffile.imread(img_path)
    except Exception as e:
        sys.stderr.write(f'[WARN] {img_path}: {e}\n')
        return uid, None, []
    h, w = img.shape[:2]

    anns = []
    for fname in sorted(os.listdir(folder)):
        if not (fname.startswith('class') and fname.endswith('.tif')):
            continue
        try:
            cat_id = int(fname[5:-4])
        except ValueError:
            continue
        try:
            mask = tifffile.imread(osp.join(folder, fname))
        except Exception as e:
            sys.stderr.write(f'[WARN] {folder}/{fname}: {e}\n')
            continue
        for inst in (int(v) for v in np.unique(mask) if v > 0):
            res = _instance_polygons((mask == inst).astype(np.uint8))
            if res is None:
                continue
            polys, area, bbox = res
            anns.append({
                'category_id': cat_id,
                'segmentation': polys,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0,
            })

    return uid, {'file_name': f'{uid}.tif', 'width': int(w), 'height': int(h)}, anns


def _link(src, dst):
    """Create/refresh a symlink at `dst` pointing to `src`."""
    if osp.lexists(dst):
        os.unlink(dst)
    os.symlink(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--augmented-train', default='data_augmented/train')
    p.add_argument('--reference-coco',  default='data/custom_coco')
    p.add_argument('--out',             default='data_augmented/custom_coco')
    p.add_argument('--workers', type=int, default=max(1, mp.cpu_count() // 2))
    args = p.parse_args()

    train_dir = osp.abspath(args.augmented_train)
    ref_root  = osp.abspath(args.reference_coco)
    out_root  = osp.abspath(args.out)

    if not osp.isdir(train_dir):
        sys.exit(f'ERROR: --augmented-train not found: {train_dir}')
    if not osp.isdir(ref_root):
        sys.exit(f'ERROR: --reference-coco not found: {ref_root}')

    images_train = osp.join(out_root, 'images', 'train')
    ann_dir      = osp.join(out_root, 'annotations')
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Val split: take it directly from the reference dataset
    _link(osp.join(ref_root, 'images', 'val'),       osp.join(out_root, 'images', 'val'))
    _link(osp.join(ref_root, 'annotations', 'val.json'), osp.join(ann_dir, 'val.json'))

    sample_uuids = sorted(
        d for d in os.listdir(train_dir)
        if osp.isdir(osp.join(train_dir, d))
    )
    print(f'[build_augmented_coco] {len(sample_uuids)} samples -> {out_root}', flush=True)

    for uid in tqdm(sample_uuids, desc='Linking images', unit='img'):
        _link(osp.join(train_dir, uid, 'image.tif'),
              osp.join(images_train, f'{uid}.tif'))

    fn = partial(_process_sample, train_dir=train_dir)
    results = []
    if args.workers > 1:
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.workers) as pool:
            for r in tqdm(pool.imap_unordered(fn, sample_uuids, chunksize=8),
                          total=len(sample_uuids),
                          desc='Extracting polygons', unit='img'):
                results.append(r)
    else:
        for uid in tqdm(sample_uuids, desc='Extracting polygons', unit='img'):
            results.append(fn(uid))

    # Sort for deterministic image/annotation IDs
    results.sort(key=lambda x: x[0])

    images, annotations = [], []
    img_id, ann_id = 1, 1
    for _uid, img_info, anns in results:
        if img_info is None:
            continue
        img_info['id'] = img_id
        images.append(img_info)
        for a in anns:
            a['id'] = ann_id
            a['image_id'] = img_id
            annotations.append(a)
            ann_id += 1
        img_id += 1

    out_ann = osp.join(ann_dir, 'train.json')
    with open(out_ann, 'w') as f:
        json.dump({'images': images,
                   'annotations': annotations,
                   'categories': CATEGORIES}, f)
    print(f'[build_augmented_coco] wrote {out_ann}: '
          f'{len(images)} images, {len(annotations)} annotations', flush=True)


if __name__ == '__main__':
    main()
