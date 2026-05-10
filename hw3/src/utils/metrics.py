from typing import Dict, List, Optional, Tuple

import numpy as np


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter) / max(float(union), 1e-6)


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    union_area = area1[:, None] + area2[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


def compute_ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    rec_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for thr in rec_thresholds:
        p = precision[recall >= thr]
        ap += p.max() if p.size > 0 else 0.0
    return ap / 101


def compute_ap_per_class(
    pred_bboxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[int, float]:
    ap_dict: Dict[int, float] = {}
    for cls in range(num_classes):
        all_scores, all_tp, all_fp = [], [], []
        n_gt = 0
        for img_pred_boxes, img_scores, img_pred_labels, img_gt_boxes, img_gt_labels in zip(
            pred_bboxes, pred_scores, pred_labels, gt_bboxes, gt_labels
        ):
            cls_pred_mask = img_pred_labels == cls
            cls_gt_mask = img_gt_labels == cls
            c_pred = img_pred_boxes[cls_pred_mask]
            c_scores = img_scores[cls_pred_mask]
            c_gt = img_gt_boxes[cls_gt_mask]
            n_gt += len(c_gt)

            if len(c_pred) == 0:
                continue
            if len(c_gt) == 0:
                all_scores.extend(c_scores.tolist())
                all_tp.extend([0] * len(c_scores))
                all_fp.extend([1] * len(c_scores))
                continue

            iou_mat = bbox_iou(c_pred, c_gt)
            matched_gt = set()
            order = np.argsort(-c_scores)
            for i in order:
                iou_row = iou_mat[i]
                best_j = int(iou_row.argmax())
                if iou_row[best_j] >= iou_threshold and best_j not in matched_gt:
                    all_tp.append(1)
                    all_fp.append(0)
                    matched_gt.add(best_j)
                else:
                    all_tp.append(0)
                    all_fp.append(1)
                all_scores.append(c_scores[i])

        if n_gt == 0:
            ap_dict[cls] = 0.0
            continue

        if not all_scores:
            ap_dict[cls] = 0.0
            continue

        order = np.argsort(-np.array(all_scores))
        tp_cum = np.cumsum(np.array(all_tp)[order])
        fp_cum = np.cumsum(np.array(all_fp)[order])
        recall = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap_dict[cls] = compute_ap_from_pr(precision, recall)

    return ap_dict


def compute_map(ap_dict: Dict[int, float]) -> float:
    vals = list(ap_dict.values())
    return float(np.mean(vals)) if vals else 0.0


def mask_iou_matrix(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> np.ndarray:
    N, M = len(pred_masks), len(gt_masks)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    # Subsample large masks (1024x1024 → 256x256) to keep matmul tractable.
    # IoU approximation error at stride=4 is <1% for typical cell masks.
    stride = 4 if pred_masks[0].size > 256 * 256 else 1

    pred_flat = np.stack([m[::stride, ::stride].ravel().astype(np.float32) for m in pred_masks])
    gt_flat   = np.stack([m[::stride, ::stride].ravel().astype(np.float32) for m in gt_masks])

    inter = pred_flat @ gt_flat.T                          # (N, M)
    pred_area = pred_flat.sum(axis=1, keepdims=True)       # (N, 1)
    gt_area   = gt_flat.sum(axis=1, keepdims=True).T       # (1, M)
    union = pred_area + gt_area - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


def compute_pr_per_class(
    pred_masks: List[List[np.ndarray]],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_masks: List[List[np.ndarray]],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[int, dict]:
    """Returns per-class dicts with 'precision', 'recall', 'f1', 'ap', 'scores' arrays."""
    result: Dict[int, dict] = {}
    for cls in range(num_classes):
        all_scores_cls, all_tp, all_fp = [], [], []
        n_gt = 0
        for img_pred_masks, img_scores, img_pred_labels, img_gt_masks, img_gt_labels in zip(
            pred_masks, pred_scores, pred_labels, gt_masks, gt_labels
        ):
            cls_pred_idx = np.where(img_pred_labels == cls)[0]
            cls_gt_idx   = np.where(img_gt_labels   == cls)[0]
            c_pred   = [img_pred_masks[i] for i in cls_pred_idx]
            c_scores = img_scores[cls_pred_idx]
            c_gt     = [img_gt_masks[i] for i in cls_gt_idx]
            n_gt += len(c_gt)

            if len(c_pred) == 0:
                continue
            if len(c_gt) == 0:
                all_scores_cls.extend(c_scores.tolist())
                all_tp.extend([0] * len(c_scores))
                all_fp.extend([1] * len(c_scores))
                continue

            iou_mat = mask_iou_matrix(c_pred, c_gt)
            matched_gt = set()
            order = np.argsort(-c_scores)
            for i in order:
                best_j = int(iou_mat[i].argmax())
                if iou_mat[i, best_j] >= iou_threshold and best_j not in matched_gt:
                    all_tp.append(1); all_fp.append(0); matched_gt.add(best_j)
                else:
                    all_tp.append(0); all_fp.append(1)
                all_scores_cls.append(float(c_scores[i]))

        if n_gt == 0 or not all_scores_cls:
            result[cls] = {"precision": np.zeros(1), "recall": np.zeros(1),
                           "f1": np.zeros(1), "ap": 0.0, "scores": np.zeros(1)}
            continue

        order = np.argsort(-np.array(all_scores_cls))
        tp_cum = np.cumsum(np.array(all_tp)[order])
        fp_cum = np.cumsum(np.array(all_fp)[order])
        recall    = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        with np.errstate(invalid="ignore"):
            f1 = np.where((precision + recall) > 0,
                          2 * precision * recall / (precision + recall), 0.0)
        ap = compute_ap_from_pr(precision, recall)
        result[cls] = {
            "precision": precision, "recall": recall,
            "f1": f1, "ap": ap,
            "scores": np.array(all_scores_cls)[order],
        }
    return result


def compute_confusion_matrix(
    pred_masks: List[List[np.ndarray]],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_masks: List[List[np.ndarray]],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> np.ndarray:
    """Returns (num_classes+1) x (num_classes+1) confusion matrix.
    Rows = GT (last row = background / FP), Cols = Pred (last col = missed / FN)."""
    n = num_classes + 1  # +1 for background
    cm = np.zeros((n, n), dtype=np.int64)
    bg = num_classes

    for img_pred_masks, img_scores, img_pred_labels, img_gt_masks, img_gt_labels in zip(
        pred_masks, pred_scores, pred_labels, gt_masks, gt_labels
    ):
        keep = np.where(img_scores >= score_threshold)[0]
        c_pred_masks  = [img_pred_masks[i] for i in keep]
        c_pred_labels = img_pred_labels[keep]

        matched_pred = set()
        matched_gt   = set()

        if len(c_pred_masks) > 0 and len(img_gt_masks) > 0:
            iou_mat = mask_iou_matrix(c_pred_masks, list(img_gt_masks))
            rows, cols = np.where(iou_mat >= iou_threshold)
            # greedy match by descending IoU
            taken_r, taken_c = set(), set()
            pairs = sorted(zip(iou_mat[rows, cols], rows, cols), reverse=True)
            for _, r, c in pairs:
                if r not in taken_r and c not in taken_c:
                    pred_cls = int(c_pred_labels[r])
                    gt_cls   = int(img_gt_labels[c])
                    cm[gt_cls, pred_cls] += 1
                    taken_r.add(r); taken_c.add(c)
                    matched_pred.add(r); matched_gt.add(c)

        for i in range(len(c_pred_masks)):
            if i not in matched_pred:
                cm[bg, int(c_pred_labels[i])] += 1  # FP: pred with no GT

        for j in range(len(img_gt_masks)):
            if j not in matched_gt:
                cm[int(img_gt_labels[j]), bg] += 1  # FN: GT with no pred

    return cm


def compute_mask_ap_per_class(
    pred_masks: List[List[np.ndarray]],
    pred_scores: List[np.ndarray],
    pred_labels: List[np.ndarray],
    gt_masks: List[List[np.ndarray]],
    gt_labels: List[np.ndarray],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[int, float]:
    ap_dict: Dict[int, float] = {}
    for cls in range(num_classes):
        all_scores, all_tp, all_fp = [], [], []
        n_gt = 0
        for img_pred_masks, img_scores, img_pred_labels, img_gt_masks, img_gt_labels in zip(
            pred_masks, pred_scores, pred_labels, gt_masks, gt_labels
        ):
            cls_pred_idx = np.where(img_pred_labels == cls)[0]
            cls_gt_idx   = np.where(img_gt_labels   == cls)[0]
            c_pred   = [img_pred_masks[i] for i in cls_pred_idx]
            c_scores = img_scores[cls_pred_idx]
            c_gt     = [img_gt_masks[i] for i in cls_gt_idx]
            n_gt += len(c_gt)

            if len(c_pred) == 0:
                continue
            if len(c_gt) == 0:
                all_scores.extend(c_scores.tolist())
                all_tp.extend([0] * len(c_scores))
                all_fp.extend([1] * len(c_scores))
                continue

            iou_mat = mask_iou_matrix(c_pred, c_gt)
            matched_gt = set()
            order = np.argsort(-c_scores)
            for i in order:
                best_j = int(iou_mat[i].argmax())
                if iou_mat[i, best_j] >= iou_threshold and best_j not in matched_gt:
                    all_tp.append(1)
                    all_fp.append(0)
                    matched_gt.add(best_j)
                else:
                    all_tp.append(0)
                    all_fp.append(1)
                all_scores.append(float(c_scores[i]))

        if n_gt == 0:
            ap_dict[cls] = 0.0
            continue
        if not all_scores:
            ap_dict[cls] = 0.0
            continue

        order = np.argsort(-np.array(all_scores))
        tp_cum = np.cumsum(np.array(all_tp)[order])
        fp_cum = np.cumsum(np.array(all_fp)[order])
        recall    = tp_cum / max(n_gt, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap_dict[cls] = compute_ap_from_pr(precision, recall)

    return ap_dict


def coco_evaluate(
    coco_gt,
    coco_results: List[dict],
    iou_type: str = "segm",
) -> dict:
    from pycocotools.cocoeval import COCOeval
    if not coco_results:
        return {}

    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, iou_type)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    result = {
        "mAP":    float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_s":  float(stats[3]),
        "mAP_m":  float(stats[4]),
        "mAP_l":  float(stats[5]),
        "mask_ap": float(stats[0]),
    }
    return result


def compute_iou_distribution(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> np.ndarray:
    ious = [mask_iou(p, g) for p, g in zip(pred_masks, gt_masks)]
    return np.array(ious)
