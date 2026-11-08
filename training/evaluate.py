"""Held-out evaluation and metrics visualization for Deep Transit CNN."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch.utils.data import DataLoader


def predict_loader(model, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            logits.append(model(inputs.to("cpu")).reshape(-1).cpu().numpy())
            labels.append(targets.reshape(-1).cpu().numpy())
    return np.concatenate(labels).astype(int), np.concatenate(logits)


def compute_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    predictions = (probabilities >= threshold).astype(int)
    has_two_classes = np.unique(labels).size == 2
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc_roc": float(roc_auc_score(labels, probabilities)) if has_two_classes else None,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "n_samples": int(len(labels)),
        "n_negative": int(np.sum(labels == 0)),
        "n_positive": int(np.sum(labels == 1)),
        "probabilities": probabilities,
        "predictions": predictions,
    }


def save_evaluation_plots(
    output_dir: str | Path,
    history: dict[str, list[float]],
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.5,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(labels, logits, threshold)
    predictions = metrics["predictions"]
    outputs: list[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.get("train_loss", []), label="Training")
    ax.plot(history.get("validation_loss", []), label="Validation")
    ax.set(xlabel="Epoch", ylabel="Weighted BCE loss", title="Training history")
    ax.legend()
    path = output_dir / "training_loss.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)

    fig, ax = plt.subplots(figsize=(6, 5))
    if np.unique(labels).size == 2:
        probabilities = metrics["probabilities"]
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = metrics["auc_roc"]
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    else:
        ax.text(0.5, 0.5, "ROC needs both classes in the test split", ha="center", va="center")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="False positive rate", ylabel="Recall", title="Held-out ROC", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    path = output_dir / "roc_curve.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["False positive", "Planetary"],
           yticklabels=["False positive", "Planetary"], xlabel="Predicted", ylabel="Catalog label",
           title=f"Confusion matrix (threshold {threshold:.2f})")
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    path = output_dir / "confusion_matrix.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)

    keys = ["accuracy", "precision", "recall", "f1"]
    values = [metrics[key] for key in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, values, color=["#4263eb", "#7048e8", "#2f9e44", "#f08c00"])
    ax.set(ylim=(0, 1), ylabel="Score", title="Held-out classification metrics")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.02, f"{value:.3f}", ha="center")
    path = output_dir / "classification_metrics.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); outputs.append(path)
    return outputs
