"""Quality masking, normalization, transit-aware detrending, and phase bins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class PreprocessingError(ValueError):
    """Raised when data or ephemeris cannot safely be processed."""


@dataclass
class ProcessedLightCurve:
    time: np.ndarray
    raw_flux: np.ndarray
    quality_mask: np.ndarray
    masked_flux: np.ndarray
    normalized_flux: np.ndarray
    trend: np.ndarray
    detrended_flux: np.ndarray
    folded_phase: np.ndarray | None
    phase: np.ndarray | None
    folded_flux: np.ndarray | None
    binned_flux: np.ndarray | None
    bin_occupancy: np.ndarray | None
    model_input: np.ndarray | None
    median_cadence_days: float
    detrending_window_samples: int


def _to_float_array(values) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float).reshape(-1)


def quality_mask_lightcurve(lightcurve, quality_bitmask: str = "hard") -> np.ndarray:
    """Return true for usable rows using Lightkurve's mission quality flags."""
    quality = getattr(lightcurve, "quality", None)
    if quality is None:
        raise PreprocessingError(
            "This light curve has no quality flags; the required hard quality mask cannot be applied."
        )
    try:
        from lightkurve.utils import KeplerQualityFlags

        return np.asarray(
            KeplerQualityFlags.create_quality_mask(quality, bitmask=quality_bitmask),
            dtype=bool,
        )
    except ImportError as exc:
        raise PreprocessingError("Lightkurve is required to interpret Kepler quality flags.") from exc
    except Exception as exc:
        raise PreprocessingError(f"Could not apply the {quality_bitmask!r} quality mask: {exc}") from exc


def normalize_flux(flux: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalize by valid median and return relative flux (F / median(F) - 1)."""
    finite = valid & np.isfinite(flux)
    if not finite.any():
        raise PreprocessingError("No finite, quality-approved flux values are available.")
    baseline = float(np.nanmedian(flux[finite]))
    if not np.isfinite(baseline) or baseline == 0:
        raise PreprocessingError("The median flux is zero or invalid; normalization is undefined.")
    return flux / baseline - 1.0


def phase_fold(time: np.ndarray, flux: np.ndarray, period_days: float, epoch_bkjd: float):
    """Fold data into [-0.5, 0.5) with the supplied transit epoch at phase zero.

    The common modulo expression `((t-t0) mod P)/P - 0.5` puts `t0` at
    phase -0.5. Adding half a period before wrapping rotates that interval so
    the transit is centered at zero while retaining one full orbital cycle.
    """
    if not np.isfinite(period_days) or period_days <= 0:
        raise PreprocessingError("A finite, positive orbital period is required for phase folding.")
    if not np.isfinite(epoch_bkjd):
        raise PreprocessingError("A finite transit epoch is required for phase folding.")
    phase = ((time - epoch_bkjd + period_days / 2) % period_days) / period_days - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def median_bin(phase: np.ndarray, flux: np.ndarray, bins: int = 1000):
    """Median aggregate uniform phase bins and cyclically interpolate empty bins."""
    if bins < 2:
        raise PreprocessingError("At least two phase bins are required.")
    finite = np.isfinite(phase) & np.isfinite(flux)
    phase, flux = phase[finite], flux[finite]
    if phase.size == 0:
        raise PreprocessingError("No finite phase-folded observations are available.")
    edges = np.linspace(-0.5, 0.5, bins + 1)
    indices = np.floor((phase + 0.5) * bins).astype(int)
    indices = np.clip(indices, 0, bins - 1)
    binned = np.full(bins, np.nan)
    occupancy = np.zeros(bins, dtype=np.int64)
    for index in range(bins):
        members = indices == index
        occupancy[index] = int(members.sum())
        if occupancy[index]:
            binned[index] = float(np.median(flux[members]))
    present = np.flatnonzero(np.isfinite(binned))
    if present.size == 0:
        raise PreprocessingError("No phase bins contain observations.")
    # Extend occupied bins by one cycle on either side so interpolation at
    # phase wrap is continuous; preserve original occupancy separately.
    xp = np.r_[present - bins, present, present + bins]
    fp = np.tile(binned[present], 3)
    missing = ~np.isfinite(binned)
    binned[missing] = np.interp(np.flatnonzero(missing), xp, fp)
    mean, std = float(binned.mean()), float(binned.std())
    if not np.isfinite(std) or std <= np.finfo(float).eps:
        raise PreprocessingError("Binned flux is constant; unit-variance scaling is undefined.")
    standardized = (binned - mean) / std
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, standardized, occupancy


def preprocess_lightcurve(
    lightcurve,
    *,
    period_days: float | None,
    epoch_bkjd: float | None,
    duration_hours: float | None,
    quality_bitmask: str = "hard",
    detrending_window_transit_durations: float = 3.0,
    transit_mask_duration_multiplier: float = 1.2,
    phase_bins: int = 1000,
    minimum_valid_bins: int = 50,
) -> ProcessedLightCurve:
    """Run all stages while preserving timestamps and inspectable intermediates."""
    time = _to_float_array(lightcurve.time)
    raw_flux = _to_float_array(lightcurve.flux)
    if time.size != raw_flux.size or time.size < 3:
        raise PreprocessingError("Light curve must contain at least three aligned time/flux samples.")
    order = np.argsort(time)
    time, raw_flux = time[order], raw_flux[order]
    finite = np.isfinite(time) & np.isfinite(raw_flux)
    raw_quality = quality_mask_lightcurve(lightcurve, quality_bitmask)[order]
    quality_mask = raw_quality & finite
    masked_flux = np.where(quality_mask, raw_flux, np.nan)
    normalized = normalize_flux(raw_flux, quality_mask)
    normalized[~quality_mask] = np.nan

    valid_times = time[quality_mask]
    cadence = float(np.nanmedian(np.diff(valid_times))) if valid_times.size > 1 else np.nan
    if not np.isfinite(cadence) or cadence <= 0:
        raise PreprocessingError("Could not estimate a positive cadence from valid timestamps.")
    if duration_hours is None or not np.isfinite(duration_hours) or duration_hours <= 0:
        raise PreprocessingError("A valid KOI transit duration is required for transit-aware detrending.")
    if period_days is None or epoch_bkjd is None:
        raise PreprocessingError(
            "Known orbital period and transit epoch are required; blind transit search is not implemented."
        )

    duration_days = duration_hours / 24.0
    window_days = detrending_window_transit_durations * duration_days
    window_samples = max(3, int(round(window_days / cadence)))
    if window_samples % 2 == 0:
        window_samples += 1
    # Transit region and margin are excluded from the running-median baseline.
    phase = ((time - epoch_bkjd + period_days / 2) % period_days) / period_days - 0.5
    transit_half_width = duration_days * transit_mask_duration_multiplier / (2 * period_days)
    out_of_transit = np.abs(phase) > transit_half_width
    baseline_samples = pd.Series(normalized).where(quality_mask & out_of_transit)
    rolling = baseline_samples.rolling(window=window_samples, center=True, min_periods=1).median()
    trend = rolling.to_numpy(dtype=float)
    known = np.flatnonzero(np.isfinite(trend) & quality_mask)
    if known.size < 2:
        raise PreprocessingError("Too few out-of-transit samples to estimate the stellar trend.")
    trend = np.interp(np.arange(time.size), known, trend[known])
    if np.any(~np.isfinite(trend)) or np.any(np.abs(1.0 + trend) < np.finfo(float).eps):
        raise PreprocessingError("Estimated stellar trend contains invalid or zero baselines.")
    detrended = (1.0 + normalized) / (1.0 + trend) - 1.0
    detrended[~quality_mask] = np.nan

    folded_phase, folded_flux = phase_fold(
        time[quality_mask], detrended[quality_mask], period_days, epoch_bkjd
    )
    phase_centers, binned, occupancy = median_bin(folded_phase, folded_flux, bins=phase_bins)
    if np.count_nonzero(occupancy) < minimum_valid_bins:
        raise PreprocessingError(
            f"Only {np.count_nonzero(occupancy)} of {phase_bins} phase bins have data; "
            f"at least {minimum_valid_bins} are required."
        )
    return ProcessedLightCurve(
        time=time,
        raw_flux=raw_flux,
        quality_mask=quality_mask,
        masked_flux=masked_flux,
        normalized_flux=normalized,
        trend=trend,
        detrended_flux=detrended,
        folded_phase=folded_phase,
        phase=phase_centers,
        folded_flux=folded_flux,
        binned_flux=binned,
        bin_occupancy=occupancy,
        model_input=binned[np.newaxis, :].astype(np.float32),
        median_cadence_days=cadence,
        detrending_window_samples=window_samples,
    )
