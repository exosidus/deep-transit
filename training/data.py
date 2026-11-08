"""Manifest-backed PyTorch dataset for processed KOI examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TransitDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split: str):
        required = {"status", "split", "example_path", "label", "kepoi_name"}
        missing = required.difference(manifest.columns)
        if missing:
            raise ValueError(f"Dataset manifest is missing required columns: {sorted(missing)}")
        self.rows = manifest.loc[
            (manifest["status"] == "ready") & (manifest["split"] == split)
        ].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No ready examples are assigned to the {split!r} split.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = Path(row["example_path"])
        if not path.exists():
            raise FileNotFoundError(f"Processed example for {row['kepoi_name']} is missing: {path}")
        try:
            with np.load(path, allow_pickle=False) as archive:
                inputs = np.asarray(archive["model_input"], dtype=np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"Could not read processed example {path}: {exc}") from exc
        if inputs.shape != (1, 1000) or not np.isfinite(inputs).all():
            raise ValueError(f"Example {path} must contain a finite (1, 1000) model_input.")
        label = float(row["label"])
        return torch.from_numpy(inputs), torch.tensor([label], dtype=torch.float32)
