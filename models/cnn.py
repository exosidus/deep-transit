"""CPU-friendly 1D CNN for standardized, phase-folded transit curves."""

from __future__ import annotations

import torch
from torch import nn


class DeepTransitCNN(nn.Module):
    """Specified three-block Conv1d binary classifier returning logits."""

    def __init__(self, dropout: float = 0.4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16000, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (1, 1000):
            raise ValueError(
                f"Expected input shape (batch_size, 1, 1000); got {tuple(inputs.shape)}"
            )
        return self.classifier(self.features(inputs))
