"""I/O helpers for the Image Matching Challenge data layout and submission file."""

import csv
import os

import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg")
SUBMISSION_FIELDS = ["image_id", "dataset", "scene", "image",
                     "rotation_matrix", "translation_vector"]


def list_dataset_images(root):
    """Return ``{dataset: [image filename, ...]}`` for a ``root/<dataset>/<image>`` layout."""
    out = {}
    for dataset in sorted(os.listdir(root)):
        ddir = os.path.join(root, dataset)
        if not os.path.isdir(ddir):
            continue
        imgs = sorted(f for f in os.listdir(ddir)
                      if f.lower().endswith(IMAGE_EXTS))
        if imgs:
            out[dataset] = imgs
    return out


def read_sample_submission(path):
    """Read sample_submission.csv into a list of row dicts (order preserved)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_submission(path, rows):
    """Write submission rows (list of dicts) using the official column order."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in SUBMISSION_FIELDS})


def mat_to_str(values):
    """Flatten an array row-major and join with ';' (the IMC submission format)."""
    flat = np.asarray(values, dtype=float).reshape(-1)
    return ";".join("{:.9f}".format(v) for v in flat)


def read_train_labels(path):
    """Read train_labels.csv into ``{(dataset, image): (R 3x3, t 3)}``."""
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            R = np.array([float(v) for v in row["rotation_matrix"].split(";")]).reshape(3, 3)
            t = np.array([float(v) for v in row["translation_vector"].split(";")])
            labels[(row["dataset"], row["image"])] = (R, t)
    return labels
