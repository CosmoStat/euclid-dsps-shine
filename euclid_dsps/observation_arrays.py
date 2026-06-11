"""Array-based photometry extraction for training-oriented workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .io import microjy_to_flux_fnu_cgs
from .photometry import abmag_to_fnu_cgs, magerr_to_fluxerr_fnu_cgs


@dataclass(frozen=True)
class PhotometryArrays:
    object_id: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    mask: np.ndarray
    band_names: tuple[str, ...]
    truth: dict[str, np.ndarray] | None = None


def photometry_arrays_from_dataframe(
    frame: pd.DataFrame,
    band_configs: list[dict[str, Any]],
    *,
    object_id_column: str | None = None,
) -> PhotometryArrays:
    """Extract configured photometry into ``fnu_cgs`` arrays.

    Negative fluxes are preserved when their associated error is finite and
    positive. The mask identifies bands with finite fluxes and usable errors.
    """
    band_names = tuple(str(band["name"]) for band in band_configs)
    flux_columns = []
    err_columns = []
    mask_columns = []
    for band in band_configs:
        raw_flux = frame[str(band["column"])].astype(float).to_numpy()
        units = str(band.get("units", "fnu_cgs"))
        flux = _convert_flux_to_fnu_cgs(raw_flux, units)
        flux_err = _flux_error_array(frame, band, flux, units)
        mask = np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0.0)
        flux_columns.append(flux)
        err_columns.append(flux_err)
        mask_columns.append(mask)

    if object_id_column and object_id_column in frame:
        object_id = frame[object_id_column].to_numpy()
    else:
        object_id = np.asarray(frame.index.to_numpy(), dtype=np.int64)

    return PhotometryArrays(
        object_id=object_id,
        flux=np.stack(flux_columns, axis=1).astype(np.float32),
        flux_err=np.stack(err_columns, axis=1).astype(np.float32),
        mask=np.stack(mask_columns, axis=1).astype(bool),
        band_names=band_names,
    )


def validate_fs2_band_contract(
    band_configs: list[dict[str, Any]],
    *,
    expected_n_bands: int = 10,
) -> tuple[str, ...]:
    """Validate the first amortized implementation's strict FS2 band contract."""
    names = tuple(str(band["name"]) for band in band_configs)
    expected = (
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "euclid_vis",
        "euclid_nisp_y",
        "euclid_nisp_j",
        "euclid_nisp_h",
    )
    if len(names) != int(expected_n_bands):
        raise ValueError(f"Expected {expected_n_bands} FS2 bands, got {len(names)}")
    if names != expected:
        raise ValueError(
            "Amortized FS2 currently requires bands in exact order "
            f"{expected}; got {names}"
        )
    return names


def _convert_flux_to_fnu_cgs(values: np.ndarray, units: str) -> np.ndarray:
    if units == "fnu_cgs":
        return np.asarray(values, dtype=float)
    if units == "abmag":
        return np.asarray(abmag_to_fnu_cgs(values), dtype=float)
    if units in {"microjy", "ujy"}:
        return np.asarray(microjy_to_flux_fnu_cgs(values), dtype=float)
    raise ValueError(f"Unsupported photometry units: {units}")


def _flux_error_array(
    frame: pd.DataFrame,
    band: dict[str, Any],
    flux_fnu_cgs: np.ndarray,
    flux_units: str,
) -> np.ndarray:
    fallback_sigma_mag = float(band.get("sigma_mag", 0.05))
    fallback = np.asarray(
        magerr_to_fluxerr_fnu_cgs(flux_fnu_cgs, fallback_sigma_mag), dtype=float
    )
    error_column = band.get("error_column")
    if not error_column or str(error_column) not in frame:
        return fallback

    raw_error = frame[str(error_column)].astype(float).to_numpy()
    error_units = str(band.get("error_units", flux_units))
    if error_units == "fnu_cgs":
        error = np.asarray(raw_error, dtype=float)
    elif error_units in {"microjy", "ujy"}:
        error = np.asarray(microjy_to_flux_fnu_cgs(raw_error), dtype=float)
    elif error_units == "abmag":
        error = np.asarray(
            magerr_to_fluxerr_fnu_cgs(flux_fnu_cgs, raw_error), dtype=float
        )
    else:
        raise ValueError(f"Unsupported photometry error units: {error_units}")

    usable = np.isfinite(error) & (error > 0.0)
    return np.where(usable, error, fallback)
