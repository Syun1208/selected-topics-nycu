import os
import json
import cv2
import argparse
import zipfile
import sys
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

_COLORS_BGR = [
    ( 50,  50, 220),
    ( 50, 180,  50),
    (220, 100,  50),
    ( 30, 140, 230),
]
_CLASS_NAMES = ["class1", "class2", "class3", "class4"]


def get_parser():
    parser = argparse.ArgumentParser(description="Instance Segmentation Inference")
    parser.add_argument("--test_folder",  type=str, default="data/test_release")
    parser.add_argument("--output_dir",   type=str, default=None,
                        help="Output dir (default: submissions/<model>/<backbone>/<exp>)")
    parser.add_argument("--config",       type=str, required=True,
                        help="Path to test YAML config")
    parser.add_argument("--checkpoint",   type=str, default=None,
                        help="Override checkpoint path")
    parser.add_argument("--mapping_json", type=str, default="data/test_image_name_to_ids.json")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--gpu_ids",      type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'")
    parser.add_argument("--score_thr",    type=float, default=0.05)
    return parser


def setup_model(args, cfg):
    import yaml
    from src.utils.config import build_output_paths
    from src.services.tester import MMDetTester

    if args.checkpoint:
        cfg.setdefault("paths", {})["checkpoint_path"] = args.checkpoint
    cfg.setdefault("inference", {}).update({
        "device": args.device,
        "score_thr": args.score_thr,
    })
    cfg.setdefault("data", {}).update({
        "test_dir":     args.test_folder,
        "mapping_json": args.mapping_json,
    })
    out_paths = build_output_paths(cfg)
    cfg["paths"].update(out_paths)

    tester = MMDetTester(cfg)
    tester.build_model()
    return tester


def load_image_id_mapping(mapping_json_path):
    with open(mapping_json_path) as f:
        mapping_list = json.load(f)
    name_to_id = {item["file_name"]: item["id"]                      for item in mapping_list}
    size_dict  = {item["file_name"]: (item["height"], item["width"]) for item in mapping_list}
    return name_to_id, size_dict


def _read_image_bgr(image_path):
    if image_path.lower().endswith(".tif") or image_path.lower().endswith(".tiff"):
        import tifffile
        img = tifffile.imread(image_path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] >= 4:
            img = img[:, :, :3]
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=-1)
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.imread(image_path)


def _draw_predictions(image_bgr, boxes, masks, classes, scores):
    vis = image_bgr.astype(np.float32)
    for mask, cls in zip(masks, classes):
        color = _COLORS_BGR[min(int(cls), 3)]
        for c in range(3):
            vis[:, :, c] = np.where(
                mask > 0,
                vis[:, :, c] * 0.55 + color[c] * 0.45,
                vis[:, :, c],
            )
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = _COLORS_BGR[min(int(cls), 3)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{_CLASS_NAMES[min(int(cls), 3)]} {score:.2f}"
        cv2.putText(vis, label, (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


def main():
    args = get_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    import yaml
    import torch
    from mmdet.structures import DetDataSample
    from src.data.loaders.dataset import CellTestDataset
    from torch.utils.data import DataLoader

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tester = setup_model(args, cfg)

    model_name = cfg["model"]["name"]
    backbone   = cfg["model"]["backbone"]
    exp_name   = cfg.get("experiment", {}).get("name", "v1")
    output_dir = args.output_dir or os.path.join("submissions", model_name, backbone, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    visualize_dir = os.path.join(output_dir, "visualize")
    os.makedirs(visualize_dir, exist_ok=True)

    json_output_path = os.path.join(output_dir, "test-results.json")
    zip_output_path  = os.path.join(output_dir, f"{os.path.basename(output_dir)}.zip")

    name_to_id, size_dict = load_image_id_mapping(args.mapping_json)

    img_size    = int(cfg.get("data", {}).get("img_size", 1024))
    num_workers = int(cfg.get("data", {}).get("num_workers", 4))
    dataset = CellTestDataset(args.test_folder, args.mapping_json, img_size=img_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=num_workers, collate_fn=lambda x: x)

    results = []

    for batch in tqdm(loader, desc="Instance prediction"):
        item      = batch[0]
        file_name = item["file_name"]
        image_id  = item["image_id"]

        if file_name not in name_to_id:
            print(f"Skipping: {file_name}")
            continue

        H, W = size_dict[file_name]

        img_np = item["img"].permute(1, 2, 0).numpy().astype(np.float32)
        img_np = (img_np - _MEAN) / _STD
        img_t  = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(tester.device)

        h, w = img_t.shape[-2:]
        raw_metas = dict(item["img_metas"])
        raw_metas.setdefault("batch_input_shape", (h, w))
        raw_metas.setdefault("img_shape", (h, w))
        raw_metas.setdefault("ori_shape", (h, w))
        sf = raw_metas.get("scale_factor")
        if sf is not None:
            sf = np.array(sf).reshape(-1)
            raw_metas["scale_factor"] = sf[:2].tolist() if len(sf) >= 2 else float(sf[0])

        data_sample = DetDataSample()
        data_sample.set_metainfo(raw_metas)

        with torch.no_grad():
            preds = tester.model(img_t, [data_sample], mode="predict")

        pred     = preds[0].pred_instances
        scores   = pred.scores.cpu().numpy()
        keep     = scores >= tester.score_thr

        boxes    = pred.bboxes[keep].cpu().numpy()
        scores_k = scores[keep]
        classes  = pred.labels[keep].cpu().numpy()

        masks_bin = None
        if hasattr(pred, "masks") and keep.any():
            try:
                raw_masks = pred.masks[keep]
                if hasattr(raw_masks, "masks"):
                    masks_bin = raw_masks.masks.cpu().numpy()
                else:
                    masks_bin = raw_masks.cpu().numpy()
            except Exception:
                pass

        # Visualize on original image
        image_bgr = _read_image_bgr(os.path.join(args.test_folder, file_name))
        scale     = img_size / max(H, W)
        vis_boxes = boxes / scale if len(boxes) > 0 else boxes
        vis_masks = []
        if masks_bin is not None:
            for m in masks_bin:
                vis_masks.append(
                    cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                )
        vis = _draw_predictions(image_bgr, vis_boxes, vis_masks, classes, scores_k)
        out_path = os.path.join(visualize_dir, file_name.replace(".tif", ".png"))
        cv2.imwrite(out_path, vis)

        # COCO-format results (masks kept at model output resolution)
        for i, (box, score, cls) in enumerate(zip(boxes, scores_k, classes)):
            bbox = [float(box[0]), float(box[1]),
                    float(box[2] - box[0]), float(box[3] - box[1])]

            segmentation = None
            if masks_bin is not None and i < len(masks_bin):
                mask = masks_bin[i].astype(np.uint8)
                rle  = mask_utils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                segmentation  = {"size": list(rle["size"]), "counts": rle["counts"]}

            results.append({
                "image_id":     int(image_id),
                "bbox":         bbox,
                "score":        float(score),
                "category_id":  int(cls) + 1,
                "segmentation": segmentation,
            })

    results = sorted(results, key=lambda x: x["image_id"])

    with open(json_output_path, "w") as f:
        json.dump(results, f)
    print(f"\n✅ JSON saved to: {json_output_path}")

    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_output_path, arcname=os.path.basename(json_output_path))
    print(f"✅ ZIP archive saved to: {zip_output_path}")


if __name__ == "__main__":
    main()
