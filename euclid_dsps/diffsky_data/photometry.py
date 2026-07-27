"""Photometry detection and standardization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..photometric_uncertainty import (
    default_m5_depth_error_model,
    flux_error_from_model,
)
from ..photometry import abmag_to_fnu_cgs


@dataclass(frozen=True)
class PhotometryColumnReport:
    native_kind: str
    band_names: tuple[str, ...]
    native_columns: tuple[str, ...]
    unit: str
    warnings: tuple[str, ...] = ()


def detect_photometry_columns(columns: Sequence[str]) -> PhotometryColumnReport:
    names = [str(column).split("/")[-1] for column in columns]
    mag_columns = [name for name in names if _is_composite_mag_column(name)]
    if mag_columns:
        return PhotometryColumnReport(
            native_kind="magnitude",
            band_names=tuple(_standard_band_name(name) for name in mag_columns),
            native_columns=tuple(mag_columns),
            unit="mag(AB)",
        )
    flux_columns = [
        name
        for name in names
        if name.lower().startswith("flux_") or name.lower().endswith("_flux")
    ]
    if flux_columns:
        return PhotometryColumnReport(
            native_kind="flux",
            band_names=tuple(_standard_band_name(name) for name in flux_columns),
            native_columns=tuple(flux_columns),
            unit="unknown_flux_unit",
            warnings=("Flux unit not inferred from names alone.",),
        )
    return PhotometryColumnReport(
        native_kind="none",
        band_names=(),
        native_columns=(),
        unit="none",
        warnings=("No native multi-band photometry columns detected.",),
    )


def standardize_magnitude_photometry(
    data: dict[str, np.ndarray],
    report: PhotometryColumnReport,
    snr: float,
    add_synthetic_errors: bool = True,
    error_model: dict | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame()
    for band, column in zip(report.band_names, report.native_columns, strict=True):
        mag = np.asarray(data[column], dtype=float)
        flux = np.asarray(abmag_to_fnu_cgs(mag), dtype=float)
        valid = np.isfinite(mag) & np.isfinite(flux) & (flux > 0.0) & (mag < 90.0)
        frame[f"mag_{band}"] = mag
        frame[f"flux_{band}"] = flux
        if add_synthetic_errors:
            model = (
                default_m5_depth_error_model() if error_model is None else error_model
            )
            err = flux_error_from_model(flux, model, band_name=band)
            frame[f"fluxerr_{band}"] = err
        frame[f"mask_{band}"] = valid
    return frame


def _is_composite_mag_column(name: str) -> bool:
    lower = name.lower()
    if any(
        part in lower
        for part in ("_bulge", "_disk", "_knots", "nodust", "rest", "grism", "prism")
    ):
        return False
    if lower.startswith("lsst_") and lower in {
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
    }:
        return True
    if lower.startswith("roman_f"):
        return True
    if lower.startswith("lsst_obs_") or lower.startswith("roman_obs_"):
        return True
    return False


def _standard_band_name(name: str) -> str:
    lower = name.lower()
    if lower.startswith("lsst_obs_"):
        return "lsst_" + lower.removeprefix("lsst_obs_")
    if lower.startswith("roman_obs_"):
        return "roman_" + lower.removeprefix("roman_obs_").upper()
    if lower.startswith("roman_f"):
        return "roman_" + name.split("_", 1)[1]
    return lower
