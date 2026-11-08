"""Reproducible CPU training entry point for the KOI CNN."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from models.cnn import DeepTransitCNN
from models.inference import load_checkpoint
from .data import TransitDataset
from .evaluate import compute_metrics, predict_loader, save_evaluation_plots

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _epoch(model, loader, criterion, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    running, count = 0.0, 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for inputs, labels in loader:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            per_example = criterion(logits, labels).reshape(-1)
            loss = per_example.mean()
            if training:
                loss.backward()
                optimizer.step()
            running += float(loss.detach()) * len(inputs)
            count += len(inputs)
    return running / max(count, 1)


def train_and_evaluate(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    metrics_dir: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    weight_decay: float = 0.00001,
    dropout: float = 0.4,
    patience: int = 8,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict:
    """Fit on train split, early-stop on validation, evaluate held-out test once."""
    manifest_path = Path(manifest_path).resolve()
    checkpoint_path = Path(checkpoint_path)
    metrics_dir = Path(metrics_dir)
    manifest = pd.read_csv(manifest_path)
    ready = manifest.loc[manifest["status"] == "ready"].copy()
    if ready.empty:
        raise ValueError("Dataset manifest contains no ready examples.")
    if "host_id" not in ready:
        raise ValueError("Dataset manifest lacks host_id required to audit leakage.")
    for split in ("train", "validation", "test"):
        if split not in set(ready["split"]):
            raise ValueError(f"Dataset manifest has no ready examples in the {split} split.")
    host_splits = ready.groupby("host_id")["split"].nunique()
    if (host_splits > 1).any():
        raise ValueError("Host leakage detected: a KIC occurs in multiple data splits.")

    train_dataset = TransitDataset(manifest, "train")
    validation_dataset = TransitDataset(manifest, "validation")
    test_dataset = TransitDataset(manifest, "test")
    train_labels = train_dataset.rows["label"].astype(int).to_numpy()
    class_counts = np.bincount(train_labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("Training split must contain both false-positive and planetary examples.")
    # N/(2*n_class) gives each class equal total weight; counts use only train.
    class_weights = len(train_labels) / (2.0 * class_counts.astype(float))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    seed_everything(seed)
    model = DeepTransitCNN(dropout=dropout).to("cpu")
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {"train_loss": [], "validation_loss": []}
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total, train_count = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            losses = criterion(logits, labels).reshape(-1)
            weights = torch.where(
                labels.reshape(-1) == 1,
                torch.tensor(class_weights[1], dtype=torch.float32),
                torch.tensor(class_weights[0], dtype=torch.float32),
            )
            loss = (losses * weights).mean()
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.detach()) * len(inputs)
            train_count += len(inputs)
        train_loss = train_loss_total / max(train_count, 1)
        validation_loss = _epoch(model, validation_loader, criterion)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        LOGGER.info(
            "Epoch %d/%d: train loss %.6f, validation loss %.6f",
            epoch, epochs, train_loss, validation_loss,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            payload = {
                "format_version": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "history": history,
                "model_config": {"dropout": dropout, "input_shape": [1, 1000]},
                "training_config": {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "seed": seed,
                    "loss": "BCEWithLogitsLoss with inverse-frequency sample weights",
                    "class_weights": class_weights.tolist(),
                    "class_counts_train": class_counts.tolist(),
                },
                "data_provenance": {
                    "manifest": str(manifest_path),
                    "split_counts": {
                        str(name): int(count)
                        for name, count in ready["split"].value_counts().items()
                    },
                    "split_hosts": {
                        split: sorted(ready.loc[ready["split"] == split, "host_id"].astype(str).unique())
                        for split in ("train", "validation", "test")
                    },
                },
                "class_mapping": {"FALSE POSITIVE": 0, "CONFIRMED": 1, "CANDIDATE": 1},
            }
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                LOGGER.info("Early stopping after %d epochs without validation improvement.", patience)
                break

    if best_epoch == 0:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    best_model, checkpoint = load_checkpoint(checkpoint_path)
    test_labels, test_logits = predict_loader(best_model, test_loader)
    result = compute_metrics(test_labels, test_logits, threshold=threshold)
    result.pop("probabilities")
    result.pop("predictions")
    result.update({
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "checkpoint": str(checkpoint_path.resolve()),
        "manifest": str(manifest_path),
        "class_counts": {
            "train_negative": int(class_counts[0]),
            "train_positive": int(class_counts[1]),
        },
        "threshold_selection": "fixed by user/config; test metrics are reported at this threshold",
    })
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (metrics_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_evaluation_plots(metrics_dir, history, test_labels, test_logits, threshold)
    LOGGER.info("Best checkpoint: %s (epoch %d)", checkpoint_path, checkpoint["epoch"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate Deep Transit on CPU")
    parser.add_argument("--manifest", default="data/metadata/dataset_manifest.csv")
    parser.add_argument("--checkpoint", default="models/checkpoints/deep_transit.pt")
    parser.add_argument("--metrics-dir", default="data/metrics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    training = config.get("training", {})
    try:
        result = train_and_evaluate(
            args.manifest,
            args.checkpoint,
            args.metrics_dir,
            epochs=int(training.get("epochs", 50)),
            batch_size=int(training.get("batch_size", 32)),
            learning_rate=float(training.get("learning_rate", 0.0001)),
            weight_decay=float(training.get("weight_decay", 0.00001)),
            dropout=float(training.get("dropout", 0.4)),
            patience=int(training.get("patience", 8)),
            seed=int(config.get("dataset", {}).get("random_seed", 42)),
            threshold=args.threshold,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
