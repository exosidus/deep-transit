"""Interactive dashboard, Kepler target search, and candidate-vetting view."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.inference import classify, load_checkpoint
from preprocessing.acquisition import (
    AcquisitionError,
    fetch_koi_catalog,
    get_kepler_lightcurve,
    resolve_target,
)
from preprocessing.pipeline import PreprocessingError, preprocess_lightcurve


def read_config() -> dict:
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def configured_path(config: dict, section: str, key: str, fallback: str) -> Path:
    value = Path(config.get(section, {}).get(key, fallback))
    return value if value.is_absolute() else ROOT / value


@st.cache_resource(show_spinner=False)
def cached_model(checkpoint_path: str, modified_time: float):
    del modified_time  # part of the cache key so replacing a checkpoint reloads it
    return load_checkpoint(checkpoint_path)[0]


def _catalog_metrics(config: dict) -> tuple[dict, dict | None]:
    metadata_dir = configured_path(config, "paths", "metadata", "data/metadata")
    manifest_path = metadata_dir / "dataset_manifest.csv"
    counts = {"total": None, "confirmed": None, "candidate": None, "negative": None}
    if manifest_path.exists():
        try:
            frame = pd.read_csv(manifest_path)
            ready = frame.loc[frame["status"] == "ready"]
            counts = {
                "total": int(len(ready)),
                "confirmed": int((ready["disposition"] == "CONFIRMED").sum()),
                "candidate": int((ready["disposition"] == "CANDIDATE").sum()),
                "negative": int((ready["label"] == 0).sum()),
            }
        except Exception:
            pass
    metric_path = configured_path(config, "paths", "metrics", "data/metrics") / "evaluation.json"
    metrics = None
    if metric_path.exists():
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = None
    return counts, metrics


def _show_dashboard(config: dict) -> None:
    st.header("Deep Transit")
    st.write(
        "A CPU-friendly tool for inspecting real Kepler observations and vetting KOI-like "
        "transit signals. It requires known catalog ephemerides and does not confirm planets."
    )
    counts, metrics = _catalog_metrics(config)
    checkpoint = configured_path(config, "paths", "checkpoints", "models/checkpoints") / "deep_transit.pt"
    model_status = "Available" if checkpoint.is_file() else "Not trained / checkpoint unavailable"
    sample_text = lambda key: "Unavailable" if counts[key] is None else f"{counts[key]:,}"
    cols = st.columns(5)
    cols[0].metric("Ready samples", sample_text("total"))
    cols[1].metric("Confirmed planets", sample_text("confirmed"))
    cols[2].metric("Planet candidates", sample_text("candidate"))
    cols[3].metric("False positives", sample_text("negative"))
    cols[4].metric("Model status", model_status)

    if metrics is None:
        st.info("No evaluation metrics are available yet. Build a dataset and train the model first.")
    else:
        st.subheader("Held-out test metrics")
        st.caption(
            f"Checkpoint: {metrics.get('checkpoint', 'unknown')} · "
            f"{metrics.get('n_samples', 'unknown')} test examples · "
            f"classification threshold {metrics.get('threshold', 0.5):.2f}"
        )
        metric_cols = st.columns(5)
        for col, key in zip(metric_cols, ("accuracy", "precision", "recall", "f1", "auc_roc")):
            value = metrics.get(key)
            col.metric(key.replace("_", " ").title(), "Unavailable" if value is None else f"{value:.3f}")
    st.caption("Catalog disposition and model output are separate pieces of information.")


def _plot_stages(processed) -> None:
    st.subheader("Light curve analysis")
    stages = st.tabs(["Original", "Quality mask", "Detrending", "Phase folded", "1,000 bins"])
    with stages[0]:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=processed.time, y=processed.raw_flux, mode="markers", marker={"size": 3}, name="Raw"))
        fig.update_layout(xaxis_title="Time (BKJD)", yaxis_title="PDCSAP flux")
        st.plotly_chart(fig, use_container_width=True)
    with stages[1]:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=processed.time, y=processed.raw_flux, mode="markers", marker={"size": 3, "opacity": 0.25}, name="All"))
        fig.add_trace(go.Scattergl(x=processed.time, y=processed.masked_flux, mode="markers", marker={"size": 3}, name="Hard quality mask"))
        fig.update_layout(xaxis_title="Time (BKJD)", yaxis_title="PDCSAP flux")
        st.plotly_chart(fig, use_container_width=True)
    with stages[2]:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=processed.time, y=processed.normalized_flux, mode="markers", marker={"size": 3, "opacity": 0.25}, name="Normalized"))
        fig.add_trace(go.Scattergl(x=processed.time, y=processed.trend, mode="lines", name="Running median trend"))
        fig.add_trace(go.Scattergl(x=processed.time, y=processed.detrended_flux, mode="markers", marker={"size": 3}, name="Detrended"))
        fig.update_layout(xaxis_title="Time (BKJD)", yaxis_title="Relative flux")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Median cadence {processed.median_cadence_days:.5f} days; running window {processed.detrending_window_samples} samples.")
    with stages[3]:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=processed.folded_phase, y=processed.folded_flux, mode="markers", marker={"size": 3, "opacity": 0.35}, name="Folded observations"))
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Transit epoch")
        fig.add_vline(x=-0.5, line_dash="dot", line_color="gray", annotation_text="Secondary eclipse region")
        fig.add_vline(x=0.5, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Orbital phase", yaxis_title="Detrended relative flux")
        st.plotly_chart(fig, use_container_width=True)
    with stages[4]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=processed.phase, y=processed.binned_flux, mode="lines", name="Standardized 1,000-bin vector"))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(xaxis_title="Orbital phase", yaxis_title="Standardized flux")
        st.plotly_chart(fig, use_container_width=True)
        nonempty = int(np.count_nonzero(processed.bin_occupancy))
        st.caption(f"{nonempty}/1,000 bins contain observations. Empty bins are cyclically interpolated for the model input.")
    st.markdown(
        "Transit shape is a qualitative cue: a U-shaped dip can occur for a planet, while a sharp V-shaped "
        "eclipse or a secondary event can suggest a binary system. These shapes are not definitive by themselves."
    )


def _search_target(config: dict, identifier: str) -> None:
    paths = config.get("paths", {})
    archive = config.get("archive", {})
    preprocessing = config.get("preprocessing", {})
    catalog_path = configured_path(config, "paths", "metadata", "data/metadata") / archive.get(
        "cache_name", "koi_q1_q17_dr25.csv"
    )
    catalog = fetch_koi_catalog(
        catalog_path,
        table=archive.get("koi_table", "q1_q17_dr25_koi"),
        timeout_seconds=int(archive.get("timeout_seconds", 60)),
    )
    metadata = resolve_target(catalog, identifier)
    lightcurve = get_kepler_lightcurve(
        metadata.kepid,
        configured_path(config, "paths", "raw", "data/raw"),
        flux_column=preprocessing.get("flux_column", "pdcsap_flux"),
    )
    processed = None
    if metadata.period_days is not None and metadata.epoch_bkjd is not None and metadata.duration_hours is not None:
        processed = preprocess_lightcurve(
            lightcurve,
            period_days=metadata.period_days,
            epoch_bkjd=metadata.epoch_bkjd,
            duration_hours=metadata.duration_hours,
            quality_bitmask=preprocessing.get("quality_bitmask", "hard"),
            detrending_window_transit_durations=float(preprocessing.get("detrending_window_transit_durations", 3.0)),
            transit_mask_duration_multiplier=float(preprocessing.get("transit_mask_duration_multiplier", 1.2)),
            phase_bins=int(preprocessing.get("phase_bins", 1000)),
            minimum_valid_bins=int(preprocessing.get("minimum_valid_bins", 50)),
        )
    st.session_state["target_metadata"] = metadata
    st.session_state["processed_lightcurve"] = processed
    st.session_state["raw_lightcurve"] = lightcurve


def _show_search(config: dict) -> None:
    st.header("Target search")
    st.write("Search by KOI name or KIC identifier. A KIC without KOI ephemeris can be plotted but cannot be classified.")
    with st.form("target_search_form"):
        identifier = st.text_input("Kepler target or KOI", placeholder="e.g. KOI-1.01 or 757076")
        submitted = st.form_submit_button("Retrieve and process")
    if submitted:
        if not identifier.strip():
            st.warning("Enter a KOI or KIC identifier.")
        else:
            with st.spinner("Retrieving catalog data and Kepler photometry…"):
                try:
                    _search_target(config, identifier.strip())
                except (AcquisitionError, PreprocessingError, OSError, ValueError, RuntimeError) as exc:
                    st.error(str(exc))
                    st.session_state.pop("processed_lightcurve", None)
                    st.session_state.pop("target_metadata", None)

    metadata = st.session_state.get("target_metadata")
    if metadata is not None:
        st.subheader(metadata.kepoi_name or metadata.host_id)
        st.write(f"Catalog disposition: **{metadata.disposition or 'Unclassified'}**")
        if metadata.period_days is None or metadata.epoch_bkjd is None or metadata.duration_hours is None:
            st.warning("No complete KOI ephemeris is available. A separate transit-search procedure is required before classification.")
        else:
            st.write(
                f"Period: {metadata.period_days:.6g} days · Epoch: {metadata.epoch_bkjd:.6g} BKJD · "
                f"Duration: {metadata.duration_hours:.4g} hours"
            )


def _show_classification(config: dict) -> None:
    st.header("Light curve analysis and classification")
    metadata = st.session_state.get("target_metadata")
    processed = st.session_state.get("processed_lightcurve")
    if metadata is None:
        st.info("Search for a target first.")
        return
    if processed is None:
        st.warning("This target has no complete catalog ephemeris. Classification is unavailable.")
        return
    _plot_stages(processed)

    st.subheader("Exoplanet classification")
    checkpoint_path = configured_path(config, "paths", "checkpoints", "models/checkpoints") / "deep_transit.pt"
    if not checkpoint_path.is_file():
        st.info("No trained checkpoint is available. Prepare data and train the model before scoring targets.")
        return
    threshold = st.slider("Planetary-like classification threshold", 0.0, 1.0, 0.5, 0.01)
    try:
        model = cached_model(str(checkpoint_path), checkpoint_path.stat().st_mtime)
        result = classify(model, processed.model_input, threshold=threshold, checkpoint=str(checkpoint_path))
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not classify this light curve: {exc}")
        return
    cols = st.columns(3)
    cols[0].metric("Planetary-like score", f"{result.planetary_probability:.1%}")
    cols[1].metric("False-positive-like score", f"{result.false_positive_probability:.1%}")
    cols[2].metric("Model label", result.predicted_class)
    st.caption(f"Threshold: {result.threshold:.2f} · checkpoint: {result.checkpoint}")
    st.warning(
        "These probabilities describe resemblance to the labeled training examples. "
        "They are not proof of a planet and do not replace astronomical validation."
    )
    st.write(f"Catalog disposition remains: **{metadata.disposition or 'Unclassified'}**")


def main() -> None:
    st.set_page_config(page_title="Deep Transit", page_icon="🪐", layout="wide")
    config = read_config()
    st.sidebar.title("Deep Transit")
    st.sidebar.caption("Kepler transit candidate vetting")
    dashboard, search, analysis = st.tabs(["Dashboard", "Target Search", "Light Curve Analysis"])
    with dashboard:
        _show_dashboard(config)
    with search:
        _show_search(config)
    with analysis:
        _show_classification(config)


if __name__ == "__main__":
    main()
