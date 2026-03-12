import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_coco(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_image_annotations(coco: dict) -> dict:
    mapping = defaultdict(list)
    for ann in coco["annotations"]:
        mapping[ann["image_id"]].append(ann)
    return mapping


def load_cleanvision(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    csv_dir = Path(csv_path).parent
    issue_cols = [c for c in df.columns if c.startswith("is_") and c.endswith("_issue")]
    result = {}
    for _, row in df.iterrows():
        abs_path = (csv_dir / row["filepath"]).resolve()
        result[str(abs_path)] = {col: bool(row[col]) for col in issue_cols}
    return result
