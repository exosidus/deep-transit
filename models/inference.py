"""CPU checkpoint loading and single-example inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .cnn import DeepTransitCNN


@dataclass(frozen=True)
class ClassificationResult:
    planetary_probability: float
    false_positive_probability: float
    predicted_class: str
    threshold: float
    checkpoint: str


def load_checkpoint(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = DeepTransitCNN(dropout=float(checkpoint.get("model_config", {}).get("dropout", 0.4)))
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as exc:
        raise RuntimeError(f"Could not load compatible Deep Transit checkpoint {path}: {exc}") from exc
    model.to("cpu")
    model.eval()
    return model, checkpoint


def classify(model, model_input, *, threshold: float = 0.5, checkpoint: str = "unknown"):
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be between 0 and 1.")
    values = np.asarray(model_input, dtype=np.float32)
    if values.shape == (1, 1000):
        values = values[np.newaxis, ...]
    if values.shape != (1, 1, 1000) or not np.isfinite(values).all():
        raise ValueError("Expected one finite model input with shape (1, 1000).")
    with torch.inference_mode():
        logit = model(torch.from_numpy(values)).reshape(-1)[0]
        probability = float(torch.sigmoid(logit).item())
    return ClassificationResult(
        planetary_probability=probability,
        false_positive_probability=1.0 - probability,
        predicted_class="planetary-like" if probability >= threshold else "false-positive-like",
        threshold=float(threshold),
        checkpoint=str(checkpoint),
    )
