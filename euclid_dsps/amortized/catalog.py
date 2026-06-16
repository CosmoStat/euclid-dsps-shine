"""Catalog table builders for amortized posterior outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.calibration import delta_mag_from_alpha, log10_mass_alpha_corrected


def posterior_samples_frame(
    object_id,
    theta,
    parameter_names: tuple[str, ...],
    logq,
    logprior,
    loglike,
    *,
    row_index=None,
    log_alpha_sed: float = 0.0,
    alpha_sed: float = 1.0,
) -> pd.DataFrame:
    """Return long-form posterior sample rows."""
    object_id = np.asarray(object_id)
    row_index = _optional_row_index(row_index, object_id)
    theta = np.asarray(theta, dtype=float)
    logq = np.asarray(logq, dtype=float)
    logprior = np.asarray(logprior, dtype=float)
    loglike = np.asarray(loglike, dtype=float)
    if theta.ndim != 3:
        raise ValueError(f"theta must be [K,N,D], got {theta.shape}")
    rows = []
    n_samples, n_objects, _ = theta.shape
    for sample_id in range(n_samples):
        for object_index in range(n_objects):
            row = {
                "object_id": object_id[object_index],
                **_row_index_value(row_index, object_index),
                "sample_id": int(sample_id),
                "logq": float(logq[sample_id, object_index]),
                "logprior": float(logprior[sample_id, object_index]),
                "loglike": float(loglike[sample_id, object_index]),
                "log_alpha_sed": float(log_alpha_sed),
                "alpha_sed": float(alpha_sed),
                "delta_mag_global": float(delta_mag_from_alpha(alpha_sed)),
            }
            for param_index, name in enumerate(parameter_names):
                row[name] = float(theta[sample_id, object_index, param_index])
            _add_alpha_corrected_mass(row, alpha_sed)
            rows.append(row)
    return pd.DataFrame(rows)


def posterior_summary_frame(
    object_id,
    theta,
    parameter_names: tuple[str, ...],
    loglike,
    chi2,
    mask,
    *,
    row_index=None,
    log_alpha_sed: float = 0.0,
    alpha_sed: float = 1.0,
) -> pd.DataFrame:
    """Return one posterior summary row per object."""
    object_id = np.asarray(object_id)
    row_index = _optional_row_index(row_index, object_id)
    theta = np.asarray(theta, dtype=float)
    loglike = np.asarray(loglike, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    rows = []
    for object_index, oid in enumerate(object_id):
        row = {
            "object_id": oid,
            **_row_index_value(row_index, object_index),
            "n_valid_bands": int(mask[object_index].sum()),
            "photometric_loglike_mean": float(np.nanmean(loglike[:, object_index])),
            "posterior_predictive_chi2_median": float(
                np.nanmedian(chi2[:, object_index])
            ),
            "flag_nonfinite_loglike": bool(
                not np.all(np.isfinite(loglike[:, object_index]))
            ),
            "log_alpha_sed": float(log_alpha_sed),
            "alpha_sed": float(alpha_sed),
            "delta_mag_global": float(delta_mag_from_alpha(alpha_sed)),
        }
        for param_index, name in enumerate(parameter_names):
            values = theta[:, object_index, param_index]
            row[f"{name}_q16"] = float(np.nanquantile(values, 0.16))
            row[f"{name}_median"] = float(np.nanquantile(values, 0.50))
            row[f"{name}_q84"] = float(np.nanquantile(values, 0.84))
        if "log10_stellar_mass" in parameter_names:
            raw = row.get("log10_stellar_mass_median")
            row["log10_stellar_mass_raw"] = float(raw)
            row["log10_stellar_mass_alpha_corrected"] = float(
                log10_mass_alpha_corrected(raw, alpha_sed)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def posterior_predictive_flux_frame(
    object_id,
    model_flux,
    band_names: tuple[str, ...],
    *,
    row_index=None,
    model_flux_raw=None,
    log_alpha_sed: float = 0.0,
    alpha_sed: float = 1.0,
) -> pd.DataFrame:
    """Return long-form posterior predictive model flux rows."""
    object_id = np.asarray(object_id)
    row_index = _optional_row_index(row_index, object_id)
    model_flux = np.asarray(model_flux, dtype=float)
    model_flux_raw = (
        np.asarray(model_flux_raw, dtype=float)
        if model_flux_raw is not None
        else model_flux
    )
    if model_flux.ndim != 3:
        raise ValueError(f"model_flux must be [K,N,B], got {model_flux.shape}")
    rows = []
    n_samples, n_objects, n_bands = model_flux.shape
    for sample_id in range(n_samples):
        for object_index in range(n_objects):
            for band_index in range(n_bands):
                rows.append(
                    {
                        "object_id": object_id[object_index],
                        **_row_index_value(row_index, object_index),
                        "sample_id": int(sample_id),
                        "band": band_names[band_index],
                        "model_flux_fnu_cgs": float(
                            model_flux[sample_id, object_index, band_index]
                        ),
                        "model_flux_scaled_fnu_cgs": float(
                            model_flux[sample_id, object_index, band_index]
                        ),
                        "model_flux_raw_fnu_cgs": float(
                            model_flux_raw[sample_id, object_index, band_index]
                        ),
                        "log_alpha_sed": float(log_alpha_sed),
                        "alpha_sed": float(alpha_sed),
                    }
                )
    return pd.DataFrame(rows)


def learned_prior_samples_frame(
    x,
    theta,
    parameter_names: tuple[str, ...],
    logprior,
    *,
    log_alpha_sed: float = 0.0,
    alpha_sed: float = 1.0,
) -> pd.DataFrame:
    """Return learned RealNVP prior samples in latent ``x`` and physical ``theta``."""
    x = np.asarray(x, dtype=float)
    theta = np.asarray(theta, dtype=float)
    logprior = np.asarray(logprior, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be [S,D], got {x.shape}")
    if theta.ndim != 2:
        raise ValueError(f"theta must be [S,D], got {theta.shape}")
    if x.shape[0] != theta.shape[0]:
        raise ValueError(f"x and theta must have same sample count, got {x.shape} and {theta.shape}")
    rows = []
    for sample_id in range(theta.shape[0]):
        row = {
            "sample_id": int(sample_id),
            "logprior": float(logprior[sample_id]),
            "log_alpha_sed": float(log_alpha_sed),
            "alpha_sed": float(alpha_sed),
            "delta_mag_global": float(delta_mag_from_alpha(alpha_sed)),
        }
        for x_index in range(x.shape[1]):
            row[f"x_{x_index:02d}"] = float(x[sample_id, x_index])
        for param_index, name in enumerate(parameter_names):
            row[name] = float(theta[sample_id, param_index])
        _add_alpha_corrected_mass(row, alpha_sed)
        rows.append(row)
    return pd.DataFrame(rows)


def _add_alpha_corrected_mass(row: dict, alpha_sed: float) -> None:
    if "log10_stellar_mass" not in row:
        return
    raw = float(row["log10_stellar_mass"])
    row["log10_stellar_mass_raw"] = raw
    row["log10_stellar_mass_alpha_corrected"] = float(
        log10_mass_alpha_corrected(raw, alpha_sed)
    )


def _optional_row_index(row_index, object_id: np.ndarray) -> np.ndarray | None:
    if row_index is None:
        return None
    values = np.asarray(row_index, dtype=np.int64)
    if values.shape[0] != np.asarray(object_id).shape[0]:
        raise ValueError(
            "row_index length must match object_id length: "
            f"{values.shape[0]} vs {np.asarray(object_id).shape[0]}"
        )
    return values


def _row_index_value(row_index: np.ndarray | None, object_index: int) -> dict[str, int]:
    if row_index is None:
        return {}
    return {"row_index": int(row_index[object_index])}
