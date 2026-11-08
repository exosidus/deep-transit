"""Retrieve KOI metadata and real Kepler long-cadence photometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

LOGGER = logging.getLogger(__name__)
TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KOI_COLUMNS = "kepid, kepoi_name, koi_disposition, koi_period, koi_time0bk, koi_duration"


class AcquisitionError(RuntimeError):
    """Raised when archive metadata or a light curve cannot be retrieved."""


@dataclass(frozen=True)
class TargetMetadata:
    kepid: int
    kepoi_name: str | None = None
    disposition: str | None = None
    period_days: float | None = None
    epoch_bkjd: float | None = None
    duration_hours: float | None = None

    @property
    def host_id(self) -> str:
        return f"KIC {self.kepid}"


def _identifier_key(value: object) -> str:
    text = str(value).strip().upper().replace("_", " ")
    text = re.sub(r"^(KOI|KIC)\s*[-:]?\s*", "", text)
    text = re.sub(r"^K(?=\d{5,})", "", text)
    key = re.sub(r"\s+", "", text)
    koi_match = re.fullmatch(r"(\d+)\.(\d+)", key)
    if koi_match:
        return f"{int(koi_match.group(1))}.{int(koi_match.group(2)):02d}"
    if key.isdigit():
        return str(int(key))
    return key


def fetch_koi_catalog(
    cache_path: str | Path,
    *,
    table: str = "q1_q17_dr25_koi",
    timeout_seconds: int = 60,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load a cached KOI catalog or fetch the Q1-Q17 DR25 table over TAP."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        try:
            return pd.read_csv(cache_path)
        except (OSError, pd.errors.ParserError) as exc:
            LOGGER.warning("Ignoring unreadable KOI cache %s: %s", cache_path, exc)

    if not re.fullmatch(r"[a-z0-9_]+", table):
        raise ValueError(f"Invalid Exoplanet Archive table name: {table!r}")
    query = f"SELECT {KOI_COLUMNS} FROM {table}"
    params = urllib.parse.urlencode({"query": query, "format": "csv"})
    url = f"{TAP_SYNC_URL}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "DeepTransit/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AcquisitionError(f"Could not retrieve KOI catalog from TAP: {exc}") from exc
    if not payload:
        raise AcquisitionError("The Exoplanet Archive returned an empty KOI catalog.")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(cache_path)
    try:
        catalog = pd.read_csv(cache_path)
    except (OSError, pd.errors.ParserError) as exc:
        cache_path.unlink(missing_ok=True)
        raise AcquisitionError(f"The Exoplanet Archive response was not valid CSV: {exc}") from exc
    required = {"kepid", "kepoi_name", "koi_disposition", "koi_period", "koi_time0bk", "koi_duration"}
    if not required.issubset(catalog.columns):
        cache_path.unlink(missing_ok=True)
        raise AcquisitionError(
            "The Exoplanet Archive response did not contain the expected KOI columns. "
            "Check the selected table and its current schema."
        )
    return catalog


def lookup_koi(catalog: pd.DataFrame, identifier: str | int) -> TargetMetadata | None:
    """Resolve a KIC or KOI identifier to a target's catalog metadata."""
    if catalog.empty:
        return None
    key = _identifier_key(identifier)
    if not key:
        return None
    if key.isdigit():
        matches = catalog.loc[pd.to_numeric(catalog["kepid"], errors="coerce") == int(key)]
    else:
        names = catalog["kepoi_name"].fillna("").map(_identifier_key)
        matches = catalog.loc[names == key]
    if matches.empty:
        return None
    row = matches.iloc[0]

    def number(column: str) -> float | None:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        return None if pd.isna(value) else float(value)

    return TargetMetadata(
        kepid=int(row["kepid"]),
        kepoi_name=str(row["kepoi_name"]) if pd.notna(row.get("kepoi_name")) else None,
        disposition=str(row["koi_disposition"]).strip().upper()
        if pd.notna(row.get("koi_disposition")) else None,
        period_days=number("koi_period"),
        epoch_bkjd=number("koi_time0bk"),
        duration_hours=number("koi_duration"),
    )


def resolve_target(catalog: pd.DataFrame, identifier: str | int) -> TargetMetadata:
    """Resolve KOI identifiers and plain Kepler IDs, including non-KOIs."""
    result = lookup_koi(catalog, identifier)
    if result is not None:
        return result
    key = _identifier_key(identifier)
    if key.isdigit():
        return TargetMetadata(kepid=int(key))
    raise AcquisitionError(
        f"Target {identifier!r} is not in the KOI catalog. Search with a KIC ID "
        "to retrieve photometry without KOI parameters."
    )


def get_kepler_lightcurve(
    kepid: int,
    cache_dir: str | Path,
    *,
    flux_column: str = "pdcsap_flux",
    refresh: bool = False,
):
    """Return stitched real long-cadence Kepler data, using a local FITS cache."""
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise AcquisitionError(
            "Lightkurve is required for photometry downloads. Install requirements.txt."
        ) from exc
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"kic-{int(kepid)}-{flux_column}.fits"
    if cache_path.exists() and not refresh:
        try:
            return lk.read(cache_path)
        except Exception as exc:
            LOGGER.warning("Ignoring unreadable light-curve cache %s: %s", cache_path, exc)
            cache_path.unlink(missing_ok=True)
    try:
        results = lk.search_lightcurve(f"KIC {int(kepid)}", mission="Kepler", cadence="long")
        if len(results) == 0:
            raise AcquisitionError(f"No Kepler long-cadence observations found for KIC {kepid}.")
        collection = results.download_all(flux_column=flux_column, quality_bitmask=None)
        if collection is None or len(collection) == 0:
            raise AcquisitionError(f"Kepler returned no downloadable observations for KIC {kepid}.")
        lightcurve = collection.stitch()
        lightcurve.to_fits(cache_path, overwrite=True)
        return lightcurve
    except AcquisitionError:
        raise
    except Exception as exc:
        raise AcquisitionError(
            f"Failed to download Kepler light curves for KIC {kepid}: {exc}"
        ) from exc
