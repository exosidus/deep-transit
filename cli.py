"""Command-line entry points for Deep Transit data preparation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from preprocessing.acquisition import (
    AcquisitionError,
    fetch_koi_catalog,
    get_kepler_lightcurve,
    resolve_target,
)
from preprocessing.pipeline import PreprocessingError, preprocess_lightcurve
from training.dataset import build_koi_dataset
from visualization.lightcurves import save_lightcurve_stages


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    for group in ("paths", "archive", "preprocessing", "dataset"):
        config.setdefault(group, {})
    return config


def _absolute(base: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def inspect_target(args, config: dict, root: Path) -> int:
    paths = config["paths"]
    archive = config["archive"]
    preprocessing = config["preprocessing"]
    catalog_path = _absolute(root, paths.get("metadata", "data/metadata")) / archive.get(
        "cache_name", "koi_q1_q17_dr25.csv"
    )
    catalog = fetch_koi_catalog(
        catalog_path,
        table=archive.get("koi_table", "q1_q17_dr25_koi"),
        timeout_seconds=int(archive.get("timeout_seconds", 60)),
        refresh=args.refresh_catalog,
    )
    metadata = resolve_target(catalog, args.target)
    lightcurve = get_kepler_lightcurve(
        metadata.kepid,
        _absolute(root, paths.get("raw", "data/raw")),
        flux_column=preprocessing.get("flux_column", "pdcsap_flux"),
        refresh=args.refresh_lightcurve,
    )
    output = _absolute(root, args.output) if args.output else _absolute(
        root, paths.get("figures", "data/figures")
    ) / f"kic-{metadata.kepid}-stages.png"
    if metadata.period_days is None or metadata.epoch_bkjd is None or metadata.duration_hours is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        time = getattr(lightcurve.time, "value", lightcurve.time)
        flux = getattr(lightcurve.flux, "value", lightcurve.flux)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, flux, ".", ms=1, alpha=0.7)
        ax.set(title=f"KIC {metadata.kepid}: raw Kepler observations", xlabel="Time (BKJD)", ylabel="Flux")
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        print(f"KOI ephemeris unavailable; saved raw light curve to {output}")
        print("A separate transit-search procedure is required before classification preprocessing.")
        return 0

    result = preprocess_lightcurve(
        lightcurve,
        period_days=metadata.period_days,
        epoch_bkjd=metadata.epoch_bkjd,
        duration_hours=metadata.duration_hours,
        quality_bitmask=preprocessing.get("quality_bitmask", "hard"),
        detrending_window_transit_durations=float(
            preprocessing.get("detrending_window_transit_durations", 3.0)
        ),
        transit_mask_duration_multiplier=float(
            preprocessing.get("transit_mask_duration_multiplier", 1.2)
        ),
        phase_bins=int(preprocessing.get("phase_bins", 1000)),
        minimum_valid_bins=int(preprocessing.get("minimum_valid_bins", 50)),
    )
    save_lightcurve_stages(result, output, title=f"{metadata.kepoi_name or metadata.host_id} light curve")
    print(f"Target: {metadata.kepoi_name or metadata.host_id}")
    print(f"Catalog disposition: {metadata.disposition or 'unclassified'}")
    print(f"Period: {metadata.period_days:g} days; epoch: {metadata.epoch_bkjd:g} BKJD")
    print(f"Valid observations: {int(result.quality_mask.sum())}/{len(result.time)}")
    print(f"Input tensor shape: {result.model_input.shape}; plot: {output}")
    return 0


def prepare_dataset(args, config: dict, root: Path) -> int:
    paths, archive = config["paths"], config["archive"]
    preprocessing, dataset = config["preprocessing"], config["dataset"]
    fractions = (
        float(dataset.get("train_fraction", 0.70)),
        float(dataset.get("validation_fraction", 0.15)),
        float(dataset.get("test_fraction", 0.15)),
    )
    manifest = build_koi_dataset(
        raw_dir=_absolute(root, paths.get("raw", "data/raw")),
        processed_dir=_absolute(root, paths.get("processed", "data/processed")),
        metadata_dir=_absolute(root, paths.get("metadata", "data/metadata")),
        archive_table=archive.get("koi_table", "q1_q17_dr25_koi"),
        timeout_seconds=int(archive.get("timeout_seconds", 60)),
        flux_column=preprocessing.get("flux_column", "pdcsap_flux"),
        quality_bitmask=preprocessing.get("quality_bitmask", "hard"),
        detrending_window_transit_durations=float(
            preprocessing.get("detrending_window_transit_durations", 3.0)
        ),
        transit_mask_duration_multiplier=float(
            preprocessing.get("transit_mask_duration_multiplier", 1.2)
        ),
        phase_bins=int(preprocessing.get("phase_bins", 1000)),
        minimum_valid_bins=int(preprocessing.get("minimum_valid_bins", 50)),
        fractions=fractions,
        seed=int(dataset.get("random_seed", 42)),
        limit=args.limit,
    )
    if manifest.empty:
        print("No labeled KOI rows were available to prepare.")
    else:
        print(manifest.groupby(["status", "disposition"], dropna=False).size().to_string())
        print(f"Manifest: {_absolute(root, paths.get('metadata', 'data/metadata')) / 'dataset_manifest.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Transit data acquisition and preprocessing")
    parser.add_argument("--config", default="config.yaml", help="YAML configuration file")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Fetch and visualize a Kepler target")
    inspect.add_argument("--target", required=True, help="KIC identifier or KOI name")
    inspect.add_argument("--output", help="Optional output PNG path")
    inspect.add_argument("--refresh-catalog", action="store_true")
    inspect.add_argument("--refresh-lightcurve", action="store_true")
    dataset = commands.add_parser("prepare-dataset", help="Build real, labeled KOI examples")
    dataset.add_argument("--limit", type=int, help="Limit catalog rows for an initial run")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    root = config_path.parent
    try:
        if args.command == "inspect":
            return inspect_target(args, config, root)
        return prepare_dataset(args, config, root)
    except (AcquisitionError, PreprocessingError, OSError, ValueError) as exc:
        logging.getLogger("deep_transit").error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
