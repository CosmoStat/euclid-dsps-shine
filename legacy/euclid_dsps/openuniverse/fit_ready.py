"""Build DSPS-compatible OpenUniverse photometry tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.io import ensure_dir

from .filter_curves import OpenUniverseFilterCurve, load_openuniverse_filter_curves
from .photometry import ab0_photon_rate, photon_rate_to_fnu_cgs
from .schema import (
    OU_LSST_ROMAN_14_BANDS,
    normalized_flux_column,
    normalized_flux_truth_column,
    normalized_fluxerr_column,
    normalized_mask_column,
)


def make_openuniverse_fit_ready_table(
    *,
    input_path: str | Path,
    main_path: str | Path,
    output_path: str | Path,
    band_names: Sequence[str] = OU_LSST_ROMAN_14_BANDS,
    filter_root: str | Path = "filters",
    filter_paths: Mapping[str, str | Path] | None = None,
    lensing_mode: str = "unlensed",
    filter_response_mode: str = "dsps_clipped",
) -> dict[str, Any]:
    """Write a DSPS-compatible OpenUniverse table in ``fnu_cgs`` units.

    The input prepared table stores OpenUniverse public photon-rate fluxes.
    The output table preserves those photon columns with explicit names, adds
    lensing columns, and overwrites the standard ``flux_*``/``fluxerr_*``
    columns with equivalent ``fnu_cgs`` values for current DSPS/amortized code.
    """
    if lensing_mode not in {"unlensed", "lensed"}:
        raise ValueError("lensing_mode must be 'unlensed' or 'lensed'")
    if filter_response_mode not in {"dsps_clipped", "native"}:
        raise ValueError("filter_response_mode must be 'dsps_clipped' or 'native'")
    bands = tuple(str(band) for band in band_names)
    frame = pd.read_parquet(input_path)
    main = _read_lensing_columns(main_path)
    out = frame.merge(main, on="galaxy_id", how="left", validate="many_to_one")
    out["mu_lensing"] = compute_lensing_magnification(
        out["convergence"].to_numpy(dtype=float),
        out["shear1"].to_numpy(dtype=float),
        out["shear2"].to_numpy(dtype=float),
    )
    invalid_mu = ~np.isfinite(out["mu_lensing"].to_numpy(dtype=float))
    invalid_mu |= out["mu_lensing"].to_numpy(dtype=float) <= 0.0
    if invalid_mu.any():
        raise ValueError(f"Found {int(invalid_mu.sum())} invalid lensing magnifications")

    native_filters = load_openuniverse_filter_curves(
        bands,
        filter_root=filter_root,
        filter_paths=filter_paths,
    )
    filters = {
        band: _filter_for_response_mode(native_filters[band], filter_response_mode)
        for band in bands
    }
    ab0_rates = {band: ab0_photon_rate(filters[band]) for band in bands}
    mu = out["mu_lensing"].to_numpy(dtype=float)
    use_mu = mu if lensing_mode == "unlensed" else np.ones_like(mu)

    new_columns: dict[str, np.ndarray | str] = {}
    for band in bands:
        truth_col = normalized_flux_truth_column(band)
        flux_col = normalized_flux_column(band)
        err_col = normalized_fluxerr_column(band)
        mask_col = normalized_mask_column(band)
        _require_columns(out, [truth_col, flux_col, err_col, mask_col])

        truth_public = out[truth_col].to_numpy(dtype=float)
        flux_public = out[flux_col].to_numpy(dtype=float)
        err_public = out[err_col].to_numpy(dtype=float)

        new_columns[f"flux_truth_lensed_photon_{band}"] = truth_public
        new_columns[f"flux_lensed_photon_{band}"] = flux_public
        new_columns[f"fluxerr_lensed_photon_{band}"] = err_public

        truth_unlensed = truth_public / mu
        flux_unlensed = flux_public / mu
        err_unlensed = err_public / mu
        new_columns[f"flux_truth_unlensed_photon_{band}"] = truth_unlensed
        new_columns[f"flux_unlensed_photon_{band}"] = flux_unlensed
        new_columns[f"fluxerr_unlensed_photon_{band}"] = err_unlensed

        truth_photon = truth_public / use_mu
        flux_photon = flux_public / use_mu
        err_photon = err_public / use_mu

        new_columns[truth_col] = photon_rate_to_fnu_cgs(truth_photon, ab0_rates[band])
        new_columns[flux_col] = photon_rate_to_fnu_cgs(flux_photon, ab0_rates[band])
        new_columns[err_col] = photon_rate_to_fnu_cgs(err_photon, ab0_rates[band])

    new_columns["openuniverse_source_flux_unit"] = "photon_per_sec_cm2"
    new_columns["photometry_unit"] = "fnu_cgs"
    new_columns["lensing_mode"] = lensing_mode
    replacements = pd.DataFrame(new_columns, index=out.index)
    out = pd.concat(
        [out.drop(columns=[c for c in replacements.columns if c in out]), replacements],
        axis=1,
    ).copy()

    output = Path(output_path)
    ensure_dir(output.parent)
    out.to_parquet(output, index=False)
    manifest = {
        "dataset": "openuniverse_lsst_roman_14_fit_ready",
        "input_path": str(input_path),
        "main_path": str(main_path),
        "output_path": str(output),
        "number_of_rows": int(len(out)),
        "bands": list(bands),
        "photometry_unit": "fnu_cgs",
        "source_flux_unit": "photon_per_sec_cm2",
        "lensing_mode": lensing_mode,
        "filter_response_mode": filter_response_mode,
        "mu_lensing": {
            "median": float(np.nanmedian(out["mu_lensing"])),
            "p01": float(np.nanpercentile(out["mu_lensing"], 1.0)),
            "p99": float(np.nanpercentile(out["mu_lensing"], 99.0)),
        },
        "ab0_photon_rate_by_band": {band: float(ab0_rates[band]) for band in bands},
        "filter_sources": {
            band: {
                "source": filters[band].source,
                "approximate": bool(filters[band].approximate),
            }
            for band in bands
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "notes": [
            "Standard flux_* and fluxerr_* columns are DSPS-compatible fnu_cgs.",
            "Original public photon fluxes are preserved as *_lensed_photon_* columns.",
            "Unlensed photon fluxes divide public photon fluxes by mu_lensing.",
            "dsps_clipped clips filter responses to [0, 1] before AB0 conversion, matching euclid_dsps.filters.load_ascii_filter.",
        ],
    }
    manifest_path = output.with_suffix(".manifest.yaml")
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest


def compute_lensing_magnification(
    convergence: np.ndarray,
    shear1: np.ndarray,
    shear2: np.ndarray,
) -> np.ndarray:
    """Compute weak-lensing magnification from convergence and shear."""
    kappa = np.asarray(convergence, dtype=float)
    g1 = np.asarray(shear1, dtype=float)
    g2 = np.asarray(shear2, dtype=float)
    denominator = np.square(1.0 - kappa) - np.square(g1) - np.square(g2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = 1.0 / denominator
    return mu.astype(float)


def _filter_for_response_mode(
    filter_curve: OpenUniverseFilterCurve,
    response_mode: str,
) -> OpenUniverseFilterCurve:
    if response_mode == "native":
        return filter_curve
    if response_mode != "dsps_clipped":
        raise ValueError("response_mode must be 'dsps_clipped' or 'native'")
    return OpenUniverseFilterCurve(
        band_name=filter_curve.band_name,
        wave_angstrom=filter_curve.wave_angstrom,
        transmission=np.clip(filter_curve.transmission, 0.0, 1.0),
        source=filter_curve.source,
        approximate=filter_curve.approximate,
    )


def _read_lensing_columns(main_path: str | Path) -> pd.DataFrame:
    columns = _available_parquet_columns(main_path)
    shear1_column = "shear1" if "shear1" in columns else "shear_1"
    shear2_column = "shear2" if "shear2" in columns else "shear_2"
    read_columns = ["galaxy_id", "convergence", shear1_column, shear2_column]
    missing = [column for column in read_columns if column not in columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    main = pd.read_parquet(main_path, columns=read_columns)
    rename = {}
    if shear1_column == "shear_1":
        rename["shear_1"] = "shear1"
    if shear2_column == "shear_2":
        rename["shear_2"] = "shear2"
    main = main.rename(columns=rename)
    _require_columns(main, ["galaxy_id", "convergence", "shear1", "shear2"])
    return main[["galaxy_id", "convergence", "shear1", "shear2"]]


def _available_parquet_columns(path: str | Path) -> set[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return set(pd.read_parquet(path).columns)
    return set(pq.ParquetFile(path).schema_arrow.names)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
