"""Build cached, labeled examples from the real Kepler KOI catalog."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import logging

import numpy as np
import pandas as pd

from preprocessing.acquisition import AcquisitionError, fetch_koi_catalog, get_kepler_lightcurve
from preprocessing.pipeline import PreprocessingError, preprocess_lightcurve

LOGGER = logging.getLogger(__name__)
LABELS = {"CONFIRMED": 1, "CANDIDATE": 1, "FALSE POSITIVE": 0}


def grouped_split_assignments(
    frame: pd.DataFrame,
    *,
    group_column: str = "host_id",
    label_column: str = "label",
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole host groups to approximate stratified split proportions."""
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (3,) or np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Train, validation, and test fractions must be positive and sum to 1.")
    rng = np.random.default_rng(seed)
    grouped = []
    for group, rows in frame.groupby(group_column, sort=False):
        counts = np.bincount(rows[label_column].astype(int), minlength=2)
        grouped.append((str(group), len(rows), counts))
    if not grouped:
        return {}
    # Randomized tie ordering keeps assignments reproducible for equally sized hosts.
    random_order = rng.permutation(len(grouped))
    grouped = [grouped[index] for index in random_order]
    grouped.sort(key=lambda item: (-(item[2].max() / item[1]), -item[1]))
    totals = np.zeros((3, 3), dtype=float)  # rows, negative, positive
    desired_rows = fractions * len(frame)
    desired_classes = np.outer(fractions, np.bincount(frame[label_column].astype(int), minlength=2))
    assignments: dict[str, str] = {}
    names = ("train", "validation", "test")
    for group, size, counts in grouped:
        vector = np.array([size, counts[0], counts[1]], dtype=float)
        scores = []
        for split_idx in range(3):
            candidate = totals.copy()
            candidate[split_idx] += vector
            row_error = np.sum(((candidate[:, 0] - desired_rows) / max(len(frame), 1)) ** 2)
            class_error = np.sum(
                ((candidate[:, 1:] - desired_classes) / np.maximum(desired_classes, 1)) ** 2
            )
            scores.append(row_error + class_error)
        best = int(np.argmin(scores))
        totals[best] += vector
        assignments[group] = names[best]
    return assignments


def _valid_cached_example(path: Path, phase_bins: int, signature: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = archive["model_input"]
            return (
                values.shape == (1, phase_bins)
                and np.isfinite(values).all()
                and str(archive["processing_signature"].item()) == signature
            )
    except (OSError, ValueError, KeyError):
        return False


def build_koi_dataset(
    *,
    raw_dir: str | Path,
    processed_dir: str | Path,
    metadata_dir: str | Path,
    archive_table: str = "q1_q17_dr25_koi",
    timeout_seconds: int = 60,
    flux_column: str = "pdcsap_flux",
    quality_bitmask: str = "hard",
    detrending_window_transit_durations: float = 3.0,
    transit_mask_duration_multiplier: float = 1.2,
    phase_bins: int = 1000,
    minimum_valid_bins: int = 50,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    limit: int | None = None,
) -> pd.DataFrame:
    """Download/preprocess KOIs into resumable NPZ examples and a CSV manifest."""
    raw_dir, processed_dir, metadata_dir = map(Path, (raw_dir, processed_dir, metadata_dir))
    for directory in (raw_dir, processed_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    catalog = fetch_koi_catalog(
        metadata_dir / "koi_q1_q17_dr25.csv",
        table=archive_table,
        timeout_seconds=timeout_seconds,
    )
    catalog = catalog.copy()
    catalog["koi_disposition"] = catalog["koi_disposition"].astype(str).str.strip().str.upper()
    catalog = catalog.loc[catalog["koi_disposition"].isin(LABELS)].copy()
    catalog = catalog.dropna(subset=["kepid", "koi_period", "koi_time0bk", "koi_duration"])
    catalog["kepid"] = pd.to_numeric(catalog["kepid"], errors="coerce")
    catalog = catalog.dropna(subset=["kepid"])
    catalog["kepid"] = catalog["kepid"].astype(int)
    if limit is not None:
        catalog = catalog.head(max(0, int(limit)))

    records: list[dict] = []
    for row in catalog.itertuples(index=False):
        kepid = int(row.kepid)
        koi_name = str(row.kepoi_name)
        disposition = str(row.koi_disposition).upper()
        label = LABELS[disposition]
        host_id = f"KIC {kepid}"
        slug = koi_name.replace(".", "-").replace(" ", "")
        output = processed_dir / f"{slug}.npz"
        record = {
            "kepoi_name": koi_name,
            "kepid": kepid,
            "host_id": host_id,
            "disposition": disposition,
            "label": label,
            "period_days": float(row.koi_period),
            "epoch_bkjd": float(row.koi_time0bk),
            "duration_hours": float(row.koi_duration),
            "example_path": str(output),
            "status": "pending",
            "exclusion_reason": "",
        }
        signature_payload = {
            "table": archive_table,
            "kepid": kepid,
            "kepoi_name": koi_name,
            "disposition": disposition,
            "period_days": record["period_days"],
            "epoch_bkjd": record["epoch_bkjd"],
            "duration_hours": record["duration_hours"],
            "flux_column": flux_column,
            "quality_bitmask": quality_bitmask,
            "detrending_window_transit_durations": detrending_window_transit_durations,
            "transit_mask_duration_multiplier": transit_mask_duration_multiplier,
            "phase_bins": phase_bins,
            "minimum_valid_bins": minimum_valid_bins,
        }
        signature = hashlib.sha256(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        record["processing_signature"] = signature
        if _valid_cached_example(output, phase_bins, signature):
            record["status"] = "ready"
            records.append(record)
            continue
        try:
            lightcurve = get_kepler_lightcurve(kepid, raw_dir, flux_column=flux_column)
            processed = preprocess_lightcurve(
                lightcurve,
                period_days=record["period_days"],
                epoch_bkjd=record["epoch_bkjd"],
                duration_hours=record["duration_hours"],
                quality_bitmask=quality_bitmask,
                detrending_window_transit_durations=detrending_window_transit_durations,
                transit_mask_duration_multiplier=transit_mask_duration_multiplier,
                phase_bins=phase_bins,
                minimum_valid_bins=minimum_valid_bins,
            )
            temporary = output.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    model_input=processed.model_input,
                    phase=processed.phase,
                    bin_occupancy=processed.bin_occupancy,
                    kepid=np.int64(kepid),
                    kepoi_name=np.array(koi_name),
                    disposition=np.array(disposition),
                    processing_signature=np.array(signature),
                )
            temporary.replace(output)
            record["status"] = "ready"
        except (AcquisitionError, PreprocessingError, OSError, ValueError) as exc:
            record["status"] = "excluded"
            record["exclusion_reason"] = str(exc)
            LOGGER.warning("Skipping %s: %s", koi_name, exc)
        records.append(record)

    manifest = pd.DataFrame.from_records(records)
    if not manifest.empty:
        ready = manifest.loc[manifest["status"] == "ready"].copy()
        manifest["split"] = ""
        if not ready.empty:
            split_map = grouped_split_assignments(ready, fractions=fractions, seed=seed)
            ready["split"] = ready["host_id"].map(split_map)
            split_by_example = ready.set_index("kepoi_name")["split"]
            manifest["split"] = manifest["kepoi_name"].map(split_by_example).fillna("")
    manifest_path = metadata_dir / "dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    LOGGER.info("Dataset manifest written to %s (%d rows)", manifest_path, len(manifest))
    return manifest
