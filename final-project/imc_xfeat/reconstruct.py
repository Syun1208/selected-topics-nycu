"""XFeat feature extraction + matching + COLMAP Structure-from-Motion.

Given a folder of images belonging to one IMC dataset, this module:
  1. extracts XFeat keypoints/descriptors for every image,
  2. matches every image pair with XFeat's mutual-nearest-neighbour matcher,
  3. feeds the keypoints + matches into a COLMAP database,
  4. runs COLMAP incremental Structure-from-Motion (via ``pycolmap``).

COLMAP naturally returns one *reconstruction* per connected set of images, so a
dataset whose images come from several scenes yields several reconstructions --
exactly the clustering the IMC submission asks for.
"""

import os

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    def tqdm(x, **kw):
        return x


def extract_features(xfeat, image_dir, image_names, top_k):
    """Run XFeat on every image. Returns ``{name: {keypoints, descriptors}}``."""
    feats = {}
    for name in tqdm(image_names, desc="  XFeat extract"):
        img = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out = xfeat.detectAndCompute(img, top_k=top_k)[0]
        feats[name] = {
            "keypoints": out["keypoints"].cpu().numpy().astype(np.float32),
            "descriptors": out["descriptors"],  # tensor kept on device for matching
        }
    return feats


def match_pairs(xfeat, feats, image_names, min_cossim, min_matches):
    """Exhaustively match all image pairs. Returns ``{(name1, name2): matches}``."""
    names = [n for n in image_names if n in feats]
    pair_matches = {}
    total = len(names) * (len(names) - 1) // 2
    pbar = tqdm(total=total, desc="  XFeat match")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            idx0, idx1 = xfeat.match(feats[names[i]]["descriptors"],
                                     feats[names[j]]["descriptors"],
                                     min_cossim=min_cossim)
            if len(idx0) >= min_matches:
                pair_matches[(names[i], names[j])] = np.stack(
                    [idx0.cpu().numpy(), idx1.cpu().numpy()], axis=1).astype(np.uint32)
            pbar.update(1)
    pbar.close()
    return pair_matches


def _open_database(pycolmap, db_path):
    """Open (or create) a COLMAP database.

    ``pycolmap.Database`` is an abstract interface in pycolmap 4.x -- the
    concrete object must be obtained through the ``Database.open`` factory.
    """
    return pycolmap.Database.open(db_path)


def _set_option(opts, name, value):
    """Best-effort set of an option attribute (names differ between versions)."""
    if hasattr(opts, name):
        try:
            setattr(opts, name, value)
        except Exception:
            pass


def reconstruct(image_dir, image_names, feats, pair_matches, work_dir,
                min_model_size=3, verbose=True):
    """Run COLMAP SfM. Returns ``{name: (cluster_idx, R 3x3, t 3)}`` for posed images."""
    import pycolmap

    os.makedirs(work_dir, exist_ok=True)
    db_path = os.path.join(work_dir, "database.db")
    sfm_dir = os.path.join(work_dir, "sparse")
    pairs_path = os.path.join(work_dir, "pairs.txt")
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(sfm_dir, exist_ok=True)

    posed = [n for n in image_names if n in feats]
    if len(posed) < 3 or not pair_matches:
        if verbose:
            print("  [sfm] too few images/matches, skipping reconstruction")
        return {}

    # 1. create an empty database, then import images -> cameras + images
    #    (import_images requires the database file to already exist)
    _open_database(pycolmap, db_path).close()
    pycolmap.import_images(db_path, image_dir,
                           pycolmap.CameraMode.PER_IMAGE, list(posed))

    # 2. write XFeat keypoints + raw matches into the database
    db = _open_database(pycolmap, db_path)
    name_to_id = {img.name: img.image_id for img in db.read_all_images()}
    for name in posed:
        if name in name_to_id:
            db.write_keypoints(name_to_id[name], feats[name]["keypoints"])
    for (n1, n2), matches in pair_matches.items():
        if n1 in name_to_id and n2 in name_to_id:
            db.write_matches(name_to_id[n1], name_to_id[n2], matches)
    db.close()

    # 3. geometric verification of the matches
    with open(pairs_path, "w") as f:
        for (n1, n2) in pair_matches:
            f.write(f"{n1} {n2}\n")
    pycolmap.verify_matches(db_path, pairs_path)

    # 4. incremental Structure-from-Motion
    opts = pycolmap.IncrementalPipelineOptions()
    _set_option(opts, "min_model_size", min_model_size)
    _set_option(opts, "min_num_matches", 15)
    try:
        recs = pycolmap.incremental_mapping(db_path, image_dir, sfm_dir, opts)
    except Exception as exc:  # pragma: no cover - depends on data
        if verbose:
            print(f"  [sfm] incremental_mapping failed: {exc}")
        return {}

    # 5. collect absolute world-to-camera poses
    poses = {}
    for cluster_idx, rec in recs.items():
        for _, img in rec.images.items():
            if not img.has_pose:
                continue
            cam_from_world = img.cam_from_world()
            R = np.asarray(cam_from_world.rotation.matrix(), dtype=float)
            t = np.asarray(cam_from_world.translation, dtype=float).reshape(3)
            poses[img.name] = (int(cluster_idx), R, t)
    if verbose:
        print(f"  [sfm] {len(recs)} cluster(s), {len(poses)}/{len(posed)} images posed")
    return poses
