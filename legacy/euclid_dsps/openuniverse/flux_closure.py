"""SED-to-flux closure diagnostics for OpenUniverse data products."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir

from .filter_curves import (
    OpenUniverseFilterCurve,
    load_openuniverse_filter_curves,
)
from .photometry import SedFnuUnit, photon_rate_from_fnu_sed
from .schema import (
    OU_LSST_ROMAN_14_BANDS,
    normalized_flux_truth_column,
)
from .sed import read_sed_components

SedWavelengthFrame = Literal["rest", "observer"]


@dataclass(frozen=True)
class SedFluxClosureResult:
    """Tabular outputs of an OpenUniverse SED-to-flux closure run."""

    rows: pd.DataFrame
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    summary: dict[str, Any]


def run_sed_flux_closure(
    *,
    catalog_path: str | Path,
    sed_path: str | Path,
    band_names: Sequence[str] = OU_LSST_ROMAN_14_BANDS,
    filter_curves: Mapping[str, OpenUniverseFilterCurve] | None = None,
    filter_root: str | Path = "filters",
    filter_paths: Mapping[str, str | Path] | None = None,
    allow_approx_filters: bool = False,
    limit: int | None = None,
    id_column: str = "galaxy_id",
    redshift_column: str = "redshift",
    sed_fnu_unit: SedFnuUnit = "native",
    sed_fnu_scale: float = 1.0,
    sed_wavelength_frame: SedWavelengthFrame = "rest",
    calibrate: bool = True,
) -> SedFluxClosureResult:
    """Compare OpenUniverse SED-integrated photon rates to flux-table truth."""
    bands = tuple(str(band) for band in band_names)
    filters = dict(filter_curves or {})
    if not filters:
        filters = load_openuniverse_filter_curves(
            bands,
            filter_root=filter_root,
            filter_paths=filter_paths,
            allow_approx_filters=allow_approx_filters,
        )
    _validate_filter_bands(bands, filters)
    catalog = _read_closure_catalog(
        catalog_path,
        bands=bands,
        id_column=id_column,
        redshift_column=redshift_column,
        limit=limit,
    )
    if catalog.empty:
        raise ValueError("No catalog rows selected for OpenUniverse SED closure")

    sed = read_sed_components(
        sed_path,
        catalog[id_column].to_numpy(dtype=np.int64),
        missing="skip",
    )
    if sed.galaxy_id.size == 0:
        raise ValueError("No requested galaxy ids were found in the SED HDF5 file")
    catalog = catalog[catalog[id_column].isin(set(sed.galaxy_id.tolist()))].copy()
    catalog = catalog.set_index(id_column).loc[sed.galaxy_id].reset_index()
    rows = _closure_rows(
        catalog,
        sed,
        filters,
        bands=bands,
        id_column=id_column,
        redshift_column=redshift_column,
        sed_fnu_unit=sed_fnu_unit,
        sed_fnu_scale=sed_fnu_scale,
        sed_wavelength_frame=sed_wavelength_frame,
    )
    calibration = _band_calibration(rows, calibrate=calibrate)
    rows = rows.merge(calibration[["band", "calibration_factor"]], on="band", how="left")
    rows["sed_flux_photon_calibrated"] = (
        rows["sed_flux_photon_raw"] * rows["calibration_factor"]
    )
    rows["relative_error_calibrated"] = (
        rows["sed_flux_photon_calibrated"] - rows["catalog_flux_photon"]
    ) / rows["catalog_flux_photon"]
    metrics = _closure_metrics(rows, calibration)
    summary = {
        "catalog_path": str(catalog_path),
        "sed_path": str(sed_path),
        "n_requested": int(0 if limit is None else limit),
        "n_objects": int(catalog[id_column].nunique()),
        "n_rows": int(len(rows)),
        "bands": list(bands),
        "sed_fnu_unit": str(sed_fnu_unit),
        "sed_fnu_scale": float(sed_fnu_scale),
        "sed_wavelength_frame": str(sed_wavelength_frame),
        "calibrated": bool(calibrate),
        "uses_approximate_filters": bool(
            any(filters[band].approximate for band in bands)
        ),
        "filter_sources": {
            band: {
                "source": filters[band].source,
                "approximate": bool(filters[band].approximate),
            }
            for band in bands
        },
    }
    return SedFluxClosureResult(
        rows=rows,
        metrics=metrics,
        calibration=calibration,
        summary=summary,
    )


def write_sed_flux_closure_outputs(
    result: SedFluxClosureResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write closure rows, metrics, calibration, and summary to ``output_dir``."""
    out = ensure_dir(output_dir)
    rows_path = out / "sed_flux_closure_rows.parquet"
    metrics_path = out / "sed_flux_closure_metrics.csv"
    calibration_path = out / "sed_flux_closure_calibration.csv"
    summary_path = out / "sed_flux_closure_summary.json"
    result.rows.to_parquet(rows_path, index=False)
    result.metrics.to_csv(metrics_path, index=False)
    result.calibration.to_csv(calibration_path, index=False)
    summary_path.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "rows": str(rows_path),
        "metrics": str(metrics_path),
        "calibration": str(calibration_path),
        "summary": str(summary_path),
    }


def _read_closure_catalog(
    catalog_path: str | Path,
    *,
    bands: tuple[str, ...],
    id_column: str,
    redshift_column: str,
    limit: int | None,
) -> pd.DataFrame:
    columns = [id_column, redshift_column]
    columns.extend(_catalog_flux_column(band) for band in bands)
    frame = pd.read_parquet(catalog_path, columns=list(dict.fromkeys(columns)))
    if limit is not None:
        frame = frame.head(max(int(limit), 0))
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(
            "Closure catalog is missing required columns: " + ", ".join(missing)
        )
    return frame


def _closure_rows(
    catalog: pd.DataFrame,
    sed,
    filters: dict[str, OpenUniverseFilterCurve],
    *,
    bands: tuple[str, ...],
    id_column: str,
    redshift_column: str,
    sed_fnu_unit: SedFnuUnit,
    sed_fnu_scale: float,
    sed_wavelength_frame: SedWavelengthFrame,
) -> pd.DataFrame:
    base_wave = np.asarray(sed.wavelength, dtype=float)
    if base_wave.ndim != 1 or base_wave.size < 2:
        raise ValueError("SED file does not provide a usable wavelength axis")
    rows: list[dict[str, Any]] = []
    for object_index, row in catalog.reset_index(drop=True).iterrows():
        galaxy_id = int(row[id_column])
        redshift = float(row[redshift_column])
        component_sed = np.asarray(sed.sed[object_index], dtype=float)
        total_fnu = np.nansum(component_sed, axis=0)
        if sed_wavelength_frame == "rest":
            sed_wave = base_wave * (1.0 + redshift)
        elif sed_wavelength_frame == "observer":
            sed_wave = base_wave
        else:
            raise ValueError("sed_wavelength_frame must be 'rest' or 'observer'")
        for band in bands:
            predicted = photon_rate_from_fnu_sed(
                sed_wave,
                total_fnu,
                filters[band],
                fnu_unit=sed_fnu_unit,
                fnu_scale=sed_fnu_scale,
            )
            truth = float(row[_catalog_flux_column(band)])
            rows.append(
                {
                    "galaxy_id": galaxy_id,
                    "redshift": redshift,
                    "band": band,
                    "catalog_flux_photon": truth,
                    "sed_flux_photon_raw": predicted,
                    "relative_error_raw": _relative_error(predicted, truth),
                    "filter_source": filters[band].source,
                    "filter_approximate": bool(filters[band].approximate),
                    "sed_wavelength_frame": sed_wavelength_frame,
                    "sed_fnu_unit": str(sed_fnu_unit),
                    "sed_fnu_scale": float(sed_fnu_scale),
                }
            )
    return pd.DataFrame(rows)


def _band_calibration(rows: pd.DataFrame, *, calibrate: bool) -> pd.DataFrame:
    out = []
    for band, group in rows.groupby("band", sort=False):
        predicted = group["sed_flux_photon_raw"].to_numpy(dtype=float)
        truth = group["catalog_flux_photon"].to_numpy(dtype=float)
        valid = np.isfinite(predicted) & np.isfinite(truth) & (predicted > 0.0)
        valid &= truth > 0.0
        if calibrate and valid.any():
            factor = float(np.nanmedian(truth[valid] / predicted[valid]))
        else:
            factor = 1.0
        out.append(
            {
                "band": str(band),
                "calibration_factor": factor,
                "n_calibration_objects": int(valid.sum()),
                "median_catalog_flux_photon": _nanmedian(truth[valid]),
                "median_sed_flux_photon_raw": _nanmedian(predicted[valid]),
            }
        )
    return pd.DataFrame(out)


def _closure_metrics(rows: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    metrics = []
    calibration_by_band = calibration.set_index("band")
    for band, group in rows.groupby("band", sort=False):
        raw = group["relative_error_raw"].to_numpy(dtype=float)
        calibrated = group["relative_error_calibrated"].to_numpy(dtype=float)
        metrics.append(
            {
                "band": str(band),
                "n_objects": int(len(group)),
                "filter_approximate": bool(group["filter_approximate"].any()),
                "calibration_factor": float(
                    calibration_by_band.loc[band, "calibration_factor"]
                ),
                "median_relative_error_raw": _nanmedian(raw),
                "sigma_mad_relative_error_raw": _sigma_mad(raw),
                "median_relative_error_calibrated": _nanmedian(calibrated),
                "sigma_mad_relative_error_calibrated": _sigma_mad(calibrated),
                "p95_abs_relative_error_calibrated": _nanpercentile(
                    np.abs(calibrated),
                    95.0,
                ),
            }
        )
    return pd.DataFrame(metrics)


def _catalog_flux_column(band: str) -> str:
    return normalized_flux_truth_column(str(band))


def _validate_filter_bands(
    bands: tuple[str, ...],
    filters: Mapping[str, OpenUniverseFilterCurve],
) -> None:
    missing = [band for band in bands if band not in filters]
    if missing:
        raise ValueError("Missing filter curves for bands: " + ", ".join(missing))


def _relative_error(predicted: float, truth: float) -> float:
    if not np.isfinite(predicted) or not np.isfinite(truth) or truth == 0.0:
        return float("nan")
    return float((predicted - truth) / truth)


def _nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


def _sigma_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    median = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - median)))


def _nanpercentile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, q))
