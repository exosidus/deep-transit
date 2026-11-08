"""Save inspectable plots for each stage of Kepler light-curve processing."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_lightcurve_stages(processed, output_path: str | Path, *, title: str = "Kepler light curve") -> Path:
    """Write a six-panel figure showing raw through 1,000-bin model input."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    time = processed.time
    axes[0, 0].plot(time, processed.raw_flux, ".", ms=1, alpha=0.65)
    axes[0, 0].set_title("Original observations")
    axes[0, 1].plot(time, processed.masked_flux, ".", ms=1, alpha=0.65)
    axes[0, 1].set_title("Hard quality mask")
    axes[1, 0].plot(time, processed.normalized_flux, ".", ms=1, alpha=0.45, label="normalized")
    axes[1, 0].plot(time, processed.trend, lw=1.2, label="running median trend")
    axes[1, 0].set_title("Trend estimate")
    axes[1, 0].legend()
    axes[1, 1].plot(time, processed.detrended_flux, ".", ms=1, alpha=0.65)
    axes[1, 1].set_title("Detrended observations")
    axes[2, 0].plot(processed.folded_phase, processed.folded_flux, ".", ms=1, alpha=0.25)
    axes[2, 0].set_title("Phase-folded observations")
    axes[2, 0].set_xlabel("Orbital phase")
    axes[2, 1].plot(processed.phase, processed.binned_flux, lw=1)
    axes[2, 1].set_title("1,000-bin standardized model input")
    axes[2, 1].set_xlabel("Orbital phase")
    axes[2, 1].set_ylabel("Standardized flux")
    for ax in axes.flat[:4]:
        ax.set_xlabel("Time (BKJD)")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
