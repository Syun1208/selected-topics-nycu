from src.data.dataset import ClassificationDataset, get_val_transforms
import sys
import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image, ImageFile
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm import tqdm
from cleanvision import Imagelab
from cleanlab.outlier import OutOfDistribution
from cleanlab.filter import find_label_issues as _find
from cleanlab.rank import get_label_quality_scores


ImageFile.LOAD_TRUNCATED_IMAGES = True

_HERE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_HERE))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def check_integrity(data_dir: Path, min_size: int = 32) -> pd.DataFrame:
    """Return DataFrame with one row per image; flags corrupted / too-small."""
    records = []
    for class_dir in sorted(
            data_dir.iterdir(), key=lambda p: int(
            p.name) if p.name.isdigit() else 0):
        if not class_dir.is_dir():
            continue
        label = int(class_dir.name)
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                img = Image.open(img_path)
                img.verify()
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                too_small = w < min_size or h < min_size
                records.append(
                    dict(
                        path=str(img_path),
                        label=label,
                        corrupted=False,
                        too_small=too_small,
                        error=None))
            except Exception as e:
                records.append(
                    dict(
                        path=str(img_path),
                        label=label,
                        corrupted=True,
                        too_small=False,
                        error=str(e)))
    df = pd.DataFrame(records)
    n_bad = (df["corrupted"] | df["too_small"]).sum()
    log.info(
        f"[integrity] corrupted={
            df['corrupted'].sum()}  too_small={
            df['too_small'].sum()}  total_flagged={n_bad}")
    return df


def check_cleanvision(image_paths: list[str]) -> pd.DataFrame:
    imagelab = Imagelab(filepaths=image_paths)
    imagelab.find_issues()
    issues = imagelab.issues.copy()
    issue_cols = [c for c in issues.columns if c.startswith("is_")]
    total = issues[issue_cols].any(axis=1).sum()
    log.info(
        f"[cleanvision] images with ≥1 issue: {total} / {len(image_paths)}")
    for col in issue_cols:
        n = int(issues[col].sum())
        if n:
            log.info(f"  {col:<40} {n:5d}")
    return issues


def load_model(model_name: str, checkpoint: Path, num_classes: int, device):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # Strip wrapper prefix "model.*" if present
    if state and all(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    log.info(f"[model] loaded {model_name} from {checkpoint}")
    return model


@torch.no_grad()
def extract_embeddings(
    model,
    data_dir: Path,
    transform,
    batch_size: int,
    device,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
        labels     (N,)
        embeddings (N, C)
        paths      list[str] length N
    """
    ds = ClassificationDataset(root=str(data_dir), transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    paths = [str(p) for p, _ in ds.samples]

    all_labels, all_probs, all_embeds = [], [], []
    feats: dict = {}

    def _hook(module, inp, out):
        feats["embed"] = out.detach().cpu()

    hook = model.global_pool.register_forward_hook(_hook)
    for imgs, labels in tqdm(
            loader, desc="  extracting embeddings", leave=False):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.append(probs)
        all_embeds.append(feats["embed"].numpy())
    hook.remove()

    labels = np.array(all_labels)
    pred_probs = np.vstack(all_probs)
    embeddings = np.vstack(all_embeds)
    log.info(f"[embeddings] shape={embeddings.shape}")
    return labels, pred_probs, embeddings, paths


def find_label_noise(
    labels: np.ndarray,
    pred_probs: np.ndarray,
    paths: list[str],
    quality_threshold: float = None,
) -> pd.DataFrame:
    scores = get_label_quality_scores(labels, pred_probs)
    issue_idx = _find(labels=labels, pred_probs=pred_probs,
                      return_indices_ranked_by="self_confidence")

    pred_labels = np.argmax(pred_probs, axis=1)
    df = pd.DataFrame({
        "path": [paths[i] for i in issue_idx],
        "given_label": labels[issue_idx],
        "predicted_label": pred_labels[issue_idx],
        "quality_score": scores[issue_idx],
    }).sort_values("quality_score").reset_index(drop=True)

    if quality_threshold is not None:
        df = df[df["quality_score"] < quality_threshold]

    log.info(f"[label_noise] suspected issues: {len(df)} / {len(labels)}  "
             f"({100 * len(df) / len(labels):.1f}%)")
    return df


def find_outliers(
    labels: np.ndarray,
    embeddings: np.ndarray,
    paths: list[str],
    percentile: float = 2.0,
) -> pd.DataFrame:
    ood = OutOfDistribution()
    scores = ood.fit_score(features=embeddings, labels=labels)
    threshold = np.percentile(scores, percentile)
    mask = scores < threshold

    df = pd.DataFrame({
        "path": [paths[i] for i in np.where(mask)[0]],
        "label": labels[mask],
        "ood_score": scores[mask],
    }).sort_values("ood_score").reset_index(drop=True)

    log.info(f"[outliers] flagged: {len(df)} (bottom {percentile}%)")
    return df


def find_near_duplicates(
    embeddings: np.ndarray,
    paths: list[str],
    labels: np.ndarray,
    threshold: float = 0.98,
    batch_size: int = 512,
) -> pd.DataFrame:
    """
    Returns DataFrame of near-duplicate pairs (cosine sim ≥ threshold).
    For each pair (a, b) the sample with lower quality is flagged for removal
    (prefer to keep the one from a larger class if cross-class, else keep
    the first encountered).
    """
    emb_norm = normalize(embeddings, axis=1)
    n = len(emb_norm)
    pairs = []

    for i in tqdm(
            range(
                0,
                n,
                batch_size),
            desc="  finding near-dups",
            leave=False):
        batch = emb_norm[i:i + batch_size]
        sims = batch @ emb_norm.T
        for bi, sim_row in enumerate(sims):
            gi = i + bi
            candidates = np.where(
                (sim_row > threshold) & (
                    np.arange(n) > gi))[0]
            for gj in candidates:
                pairs.append(dict(
                    idx_a=gi, idx_b=gj,
                    path_a=paths[gi], path_b=paths[gj],
                    label_a=int(labels[gi]), label_b=int(labels[gj]),
                    similarity=float(sim_row[gj]),
                    cross_class=(labels[gi] != labels[gj]),
                ))

    if not pairs:
        log.info("[near_dups] none found")
        return pd.DataFrame()

    df = pd.DataFrame(pairs).sort_values(
        "similarity",
        ascending=False).reset_index(
        drop=True)
    exact = (df["similarity"] >= 0.999).sum()
    cross = df["cross_class"].sum()
    log.info(
        f"[near_dups] pairs={
            len(df)}  exact(≥0.999)={exact}  cross-class={cross}")
    return df


def _select_dup_to_remove(
        dup_df: pd.DataFrame,
        labels: np.ndarray) -> set[str]:
    """
    For each near-dup pair decide which file to flag for removal:
      - cross-class pair → remove the one from the smaller class
        (its label is more likely wrong / it is less represented)
      - same-class pair  → remove path_b (higher index, arbitrary)
    Returns set of absolute path strings to remove.
    """
    from collections import Counter
    class_counts = Counter(labels.tolist())
    to_remove: set[str] = set()

    for _, row in dup_df.iterrows():
        if row["path_a"] in to_remove or row["path_b"] in to_remove:
            continue   # one already flagged, skip
        if row["cross_class"]:
            # keep sample from the larger class
            if class_counts[row["label_a"]] >= class_counts[row["label_b"]]:
                to_remove.add(row["path_b"])
            else:
                to_remove.add(row["path_a"])
        else:
            to_remove.add(row["path_b"])

    return to_remove


def run_cleaning(
    data_dir: Path,
    checkpoint: Path,
    model_name: str,
    cleaned_dir: Path = None,
    output_dir: Path = None,
    num_classes: int = 100,
    image_size: int = 320,
    resize_size: int = 340,
    batch_size: int = 32,
    num_workers: int = 4,
    # thresholds
    min_pixel_size: int = 32,
    outlier_percentile: float = 2.0,
    dup_threshold: float = 0.98,
    label_quality_threshold: float = None,
    # feature flags
    run_integrity: bool = True,
    run_cleanvision: bool = True,
    run_label_noise: bool = True,
    run_outliers: bool = True,
    run_duplicates: bool = True,
    # action
    dry_run: bool = True,
) -> pd.DataFrame:
    data_dir = data_dir.resolve()

    # Resolve default paths relative to data_dir's parent (hw1/data/)
    if cleaned_dir is None:
        cleaned_dir = data_dir.parent / (data_dir.name + "_cleaned")
    if output_dir is None:
        output_dir = data_dir.parent / (data_dir.name + "_reports")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Source (read-only) : {data_dir}")
    log.info(
        f"Cleaned copy       : {cleaned_dir}  "
        f"(created only when dry_run=False)"
    )
    log.info(f"Reports            : {output_dir}")

    flagged: dict[str, list[str]] = {}   # path (in data_dir) → [reason, …]

    def _flag(path: str, reason: str):
        flagged.setdefault(path, []).append(reason)

    # Collect all valid image paths from the ORIGINAL directory
    all_paths = [
        str(p)
        for cls_dir in sorted(
            data_dir.iterdir(),
            key=lambda p: int(p.name) if p.name.isdigit() else 0
        )
        if cls_dir.is_dir()
        for p in sorted(cls_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if run_integrity:
        log.info("── Step 1: Integrity check ──")
        integrity_df = check_integrity(data_dir, min_size=min_pixel_size)
        integrity_df.to_csv(output_dir / "integrity.csv", index=False)
        for _, row in integrity_df[integrity_df["corrupted"]].iterrows():
            _flag(row["path"], "corrupted")
        for _, row in integrity_df[integrity_df["too_small"]].iterrows():
            _flag(row["path"], f"too_small(<{min_pixel_size}px)")

    if run_cleanvision:
        log.info("── Step 2: CleanVision quality check ──")
        cv_issues = check_cleanvision(all_paths)
        cv_issues.to_csv(output_dir / "cleanvision_issues.csv", index=True)
        issue_cols = [c for c in cv_issues.columns if c.startswith("is_")]
        for path, row in cv_issues.iterrows():
            for col in issue_cols:
                if row[col]:
                    _flag(str(path), col.replace("is_", "cv_"))

    need_model = run_label_noise or run_outliers or run_duplicates
    if need_model:
        log.info("── Step 3: Extracting model embeddings ──")
        transform = get_val_transforms(
            resize_size=resize_size, crop_size=image_size)
        model = load_model(model_name, checkpoint, num_classes, device)
        labels, pred_probs, embeddings, paths = extract_embeddings(
            model, data_dir, transform, batch_size, device, num_workers)

        if run_label_noise:
            log.info("── Step 4: Label noise detection ──")
            noise_df = find_label_noise(
                labels, pred_probs, paths,
                quality_threshold=label_quality_threshold)
            noise_df.to_csv(output_dir / "label_noise.csv", index=False)
            for _, row in noise_df.iterrows():
                _flag(row["path"],
                      f"label_noise(given={int(row['given_label'])}"
                      f"→pred={int(row['predicted_label'])}"
                      f",score={row['quality_score']:.3f})")

        if run_outliers:
            log.info("── Step 5: Outlier detection ──")
            outlier_df = find_outliers(labels, embeddings, paths,
                                       percentile=outlier_percentile)
            outlier_df.to_csv(output_dir / "outliers.csv", index=False)
            for _, row in outlier_df.iterrows():
                _flag(row["path"], f"outlier(ood={row['ood_score']:.4f})")

        if run_duplicates:
            log.info("── Step 6: Near-duplicate detection ──")
            dup_df = find_near_duplicates(embeddings, paths, labels,
                                          threshold=dup_threshold)
            if len(dup_df):
                dup_df.to_csv(output_dir / "near_duplicates.csv", index=False)
                dup_to_remove = _select_dup_to_remove(dup_df, labels)
                for p in dup_to_remove:
                    _flag(p, "near_duplicate")

    report_rows = [
        {"path": p, "reasons": "|".join(reasons), "n_issues": len(reasons)}
        for p, reasons in flagged.items()
    ]
    report_df = pd.DataFrame(report_rows).sort_values(
        ["n_issues", "path"], ascending=[False, True]
    ).reset_index(drop=True)
    report_df.to_csv(output_dir / "cleaning_report.csv", index=False)

    log.info(f"\n{'─' * 55}")
    log.info(f"Total flagged images : {len(report_df)}")
    log.info(f"Report saved to      : {output_dir / 'cleaning_report.csv'}")

    if dry_run:
        log.info("DRY RUN — no clone created, no files modified.")
        return report_df

    if cleaned_dir.exists():
        log.info(f"Removing existing clone at {cleaned_dir}")
        shutil.rmtree(cleaned_dir)
    log.info(f"Cloning {data_dir} → {cleaned_dir} …")
    shutil.copytree(str(data_dir), str(cleaned_dir))
    log.info("Clone complete.")

    removed = 0
    for path_str in report_df["path"]:
        src = Path(path_str)
        try:
            rel = src.relative_to(data_dir)
        except ValueError:
            log.warning(f"Cannot remap path (not under data_dir): {src}")
            continue
        clone_path = cleaned_dir / rel
        if clone_path.exists():
            clone_path.unlink()
            removed += 1
        else:
            log.warning(f"Not found in clone: {clone_path}")

    log.info(f"Deleted {removed} flagged images from clone.")
    log.info(f"Clean dataset at: {cleaned_dir}")

    return report_df


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Clean a labeled image dataset (original is never modified)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir", required=True, type=Path,
                   help="Source dataset directory (read-only).")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--model_name", required=True, type=str)
    p.add_argument("--cleaned_dir", default=None, type=Path,
                   help="Where to write the cleaned clone "
                        "(default: data_dir/../<name>_cleaned).")
    p.add_argument("--output_dir", default=None, type=Path,
                   help="Where to write CSV reports "
                        "(default: data_dir/../<name>_reports).")
    p.add_argument("--num_classes", default=100, type=int)
    p.add_argument("--image_size", default=320, type=int)
    p.add_argument("--resize_size", default=340, type=int)
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    # thresholds
    p.add_argument("--min_pixel_size", default=32, type=int)
    p.add_argument("--outlier_percentile", default=2.0, type=float)
    p.add_argument("--dup_threshold", default=0.98, type=float)
    p.add_argument("--label_quality_threshold", default=None, type=float)
    # feature flags
    p.add_argument("--no_integrity", action="store_true")
    p.add_argument("--no_cleanvision", action="store_true")
    p.add_argument("--no_label_noise", action="store_true")
    p.add_argument("--no_outliers", action="store_true")
    p.add_argument("--no_duplicates", action="store_true")
    # action
    p.add_argument("--dry_run", action="store_true",
                   help="Report only; do not clone or delete files.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_cleaning(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        cleaned_dir=args.cleaned_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        image_size=args.image_size,
        resize_size=args.resize_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_pixel_size=args.min_pixel_size,
        outlier_percentile=args.outlier_percentile,
        dup_threshold=args.dup_threshold,
        label_quality_threshold=args.label_quality_threshold,
        run_integrity=not args.no_integrity,
        run_cleanvision=not args.no_cleanvision,
        run_label_noise=not args.no_label_noise,
        run_outliers=not args.no_outliers,
        run_duplicates=not args.no_duplicates,
        dry_run=args.dry_run,
    )
