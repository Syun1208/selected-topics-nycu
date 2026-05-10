import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from collections import defaultdict

logger = logging.getLogger(__name__)

NUM_CLASSES = 4


def _load_sample(sample_dir: str) -> Tuple[Optional[np.ndarray], Dict[int, np.ndarray]]:
    image_path = os.path.join(sample_dir, "image.tif")
    if not os.path.exists(image_path):
        return None, {}
    image = tifffile.imread(image_path)
    masks = {}
    for cls_id in range(1, NUM_CLASSES + 1):
        mask_path = os.path.join(sample_dir, f"class{cls_id}.tif")
        if os.path.exists(mask_path):
            masks[cls_id] = tifffile.imread(mask_path).astype(np.float64)
    return image, masks


class DataCleaner:

    def __init__(self, data_dir: str, report_dir: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.report_dir = Path(report_dir) if report_dir else None
        self.sample_ids = sorted(
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        )
        self.issues: Dict[str, List[str]] = defaultdict(list)


    def check_mask_errors(self) -> Dict[str, List[str]]:
        errors = {}
        for sid in self.sample_ids:
            sample_dir = self.data_dir / sid
            _, masks = _load_sample(str(sample_dir))
            sample_errors = []
            for cls_id, mask in masks.items():
                if np.any(np.isnan(mask)):
                    sample_errors.append(f"class{cls_id}: contains NaN")
                if np.any(np.isinf(mask)):
                    sample_errors.append(f"class{cls_id}: contains Inf")
                if np.any(mask < 0):
                    sample_errors.append(f"class{cls_id}: negative values")
                instance_ids = np.unique(mask)
                instance_ids = instance_ids[instance_ids > 0]
                if len(instance_ids) == 0:
                    sample_errors.append(f"class{cls_id}: all-zero mask (no instances)")
            if sample_errors:
                errors[sid] = sample_errors
                self.issues[sid].extend(sample_errors)
        logger.info(f"[check_mask_errors] {len(errors)} samples with mask errors")
        return errors

    def check_bbox_mask_consistency(self, area_ratio_thresh: float = 0.05) -> Dict[str, List[str]]:
        warnings = {}
        for sid in self.sample_ids:
            sample_dir = self.data_dir / sid
            _, masks = _load_sample(str(sample_dir))
            sample_warnings = []
            for cls_id, mask in masks.items():
                instance_ids = np.unique(mask)
                instance_ids = instance_ids[instance_ids > 0]
                for iid in instance_ids:
                    binary = (mask == iid).astype(np.uint8)
                    rows = np.any(binary, axis=1)
                    cols = np.any(binary, axis=0)
                    if not rows.any():
                        continue
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
                    mask_area = binary.sum()
                    ratio = mask_area / max(bbox_area, 1)
                    if ratio < area_ratio_thresh:
                        sample_warnings.append(
                            f"class{cls_id}/inst{int(iid)}: mask/bbox={ratio:.3f} < {area_ratio_thresh}"
                        )
            if sample_warnings:
                warnings[sid] = sample_warnings
                self.issues[sid].extend(sample_warnings)
        logger.info(f"[check_bbox_mask_consistency] {len(warnings)} samples with low mask/bbox ratio")
        return warnings

    def check_class_imbalance(self) -> Dict[str, int]:
        class_instance_counts = defaultdict(int)
        class_sample_counts = defaultdict(int)
        for sid in self.sample_ids:
            sample_dir = self.data_dir / sid
            _, masks = _load_sample(str(sample_dir))
            for cls_id, mask in masks.items():
                instance_ids = np.unique(mask)
                instance_ids = instance_ids[instance_ids > 0]
                class_instance_counts[cls_id] += len(instance_ids)
                class_sample_counts[cls_id] += 1

        report = {}
        for cls_id in range(1, NUM_CLASSES + 1):
            report[f"class{cls_id}_samples"] = class_sample_counts[cls_id]
            report[f"class{cls_id}_instances"] = class_instance_counts[cls_id]

        total = sum(class_instance_counts.values())
        logger.info("[check_class_imbalance]")
        for cls_id in range(1, NUM_CLASSES + 1):
            cnt = class_instance_counts[cls_id]
            logger.info(f"  class{cls_id}: {cnt} instances ({100*cnt/max(total,1):.1f}%)")
        return report

    def check_duplicates(self) -> List[List[str]]:
        hash_to_ids: Dict[str, List[str]] = defaultdict(list)
        for sid in self.sample_ids:
            img_path = self.data_dir / sid / "image.tif"
            if not img_path.exists():
                continue
            with open(img_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            hash_to_ids[h].append(sid)

        duplicates = [ids for ids in hash_to_ids.values() if len(ids) > 1]
        for group in duplicates:
            msg = f"Duplicate images: {group}"
            logger.warning(msg)
            for sid in group[1:]:
                self.issues[sid].append(f"Duplicate of {group[0]}")
        logger.info(f"[check_duplicates] {len(duplicates)} duplicate groups found")
        return duplicates

    def check_object_size_distribution(self) -> Dict[str, dict]:
        class_areas: Dict[int, List[float]] = defaultdict(list)
        for sid in self.sample_ids:
            sample_dir = self.data_dir / sid
            _, masks = _load_sample(str(sample_dir))
            for cls_id, mask in masks.items():
                instance_ids = np.unique(mask)
                instance_ids = instance_ids[instance_ids > 0]
                for iid in instance_ids:
                    area = float((mask == iid).sum())
                    class_areas[cls_id].append(area)

        stats = {}
        for cls_id in range(1, NUM_CLASSES + 1):
            areas = class_areas.get(cls_id, [0])
            areas_arr = np.array(areas)
            stats[f"class{cls_id}"] = {
                "count": len(areas),
                "min": float(areas_arr.min()),
                "max": float(areas_arr.max()),
                "mean": float(areas_arr.mean()),
                "median": float(np.median(areas_arr)),
                "small_pct": float((areas_arr < 32 * 32).mean() * 100),
                "large_pct": float((areas_arr > 96 * 96).mean() * 100),
            }
            logger.info(
                f"  class{cls_id}: n={len(areas)}, mean={stats[f'class{cls_id}']['mean']:.0f}px², "
                f"small={stats[f'class{cls_id}']['small_pct']:.1f}%, "
                f"large={stats[f'class{cls_id}']['large_pct']:.1f}%"
            )
        return stats


    def run_all_checks(self) -> dict:
        logger.info("=" * 60)
        logger.info("Running data cleaning checks...")
        logger.info("=" * 60)
        report = {
            "mask_errors": self.check_mask_errors(),
            "bbox_consistency": self.check_bbox_mask_consistency(),
            "class_imbalance": self.check_class_imbalance(),
            "duplicates": self.check_duplicates(),
            "size_distribution": self.check_object_size_distribution(),
        }
        n_bad = len(self.issues)
        logger.info(f"\nTotal samples with issues: {n_bad} / {len(self.sample_ids)}")
        if self.report_dir:
            import json
            self.report_dir.mkdir(parents=True, exist_ok=True)
            out = self.report_dir / "cleaning_report.json"


            serializable_report = {}
            for k, v in report.items():
                if isinstance(v, dict):
                    serializable_report[k] = {str(kk): vv for kk, vv in v.items()}
                else:
                    serializable_report[k] = v

            with open(out, "w") as f:
                json.dump(serializable_report, f, indent=2)
            logger.info(f"Report saved to {out}")
        return report

    def get_valid_sample_ids(self, allow_warnings: bool = True) -> List[str]:
        bad = set(self.issues.keys())
        valid = [sid for sid in self.sample_ids if sid not in bad]
        logger.info(
            f"Valid samples: {len(valid)} / {len(self.sample_ids)} "
            f"(removed {len(bad)} with issues)"
        )
        return valid
