import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader


def plot_training_curves(
    csv_path: str,
    save_dir: Optional[str] = None,
) -> None:
    df = pd.read_csv(csv_path)
    save_dir = Path(save_dir) if save_dir else Path(csv_path).parent

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(df["epoch"], df["train_acc"], label="Train")
    axes[1].plot(df["epoch"], df["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(df["epoch"], df["lr"])
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(True)

    fig.tight_layout()
    out = save_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved training curves → {out}")


@torch.no_grad()
def plot_confusion_matrix(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    normalize: bool = True,
) -> None:
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    norm = "true" if normalize else None
    cm = confusion_matrix(all_labels, all_preds, normalize=norm)
    num_classes = cm.shape[0]

    fig_size = max(10, num_classes // 4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )
    disp.plot(ax=ax, colorbar=True, xticks_rotation="vertical",
              values_format=".2f" if normalize else "d")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    fig.tight_layout()

    save_dir = Path(save_dir) if save_dir else Path("runs")
    out = save_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix → {out}")


@torch.no_grad()
def plot_roc_curves(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
) -> None:
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)
    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    aucs = []
    plot_individual = num_classes <= 20
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if plot_individual:
            label = class_names[i] if class_names else f"class {i}"
            ax.plot(fpr, tpr, lw=1, alpha=0.6,
                    label=f"{label} (AUC={roc_auc:.2f})")

    all_fpr = np.unique(np.concatenate(
        [roc_curve(labels_bin[:, i], all_probs[:, i])[0]
         for i in range(num_classes)]
    ))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color="navy", lw=2, linestyle="--",
            label=f"Macro-avg (AUC={macro_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    if plot_individual:
        ax.legend(loc="lower right", fontsize="x-small",
                  ncol=max(1, num_classes // 10))
    else:
        ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    save_dir = Path(save_dir) if save_dir else Path("runs")
    out = save_dir / "roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curves → {out}  (macro AUC={macro_auc:.4f})")


def visualize_run(
    run_dir: str,
    model: Optional[nn.Module] = None,
    data_loader: Optional[DataLoader] = None,
    device: Optional[torch.device] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    run_dir = Path(run_dir)
    csv_path = run_dir / "training.csv"
    if csv_path.exists():
        plot_training_curves(str(csv_path), save_dir=str(run_dir))
    else:
        print(f"No training.csv found in {run_dir}, skipping curves.")

    if model is not None and data_loader is not None and device is not None:
        plot_confusion_matrix(
            model, data_loader, device,
            class_names=class_names, save_dir=str(run_dir),
        )
        if num_classes is not None:
            plot_roc_curves(
                model, data_loader, device, num_classes,
                class_names=class_names, save_dir=str(run_dir),
            )