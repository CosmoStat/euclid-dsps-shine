"""Catalog table builders for amortized posterior outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def posterior_samples_frame(
    object_id,
    theta,
    parameter_names: tuple[str, ...],
    logq,
    logprior,
    loglike,
) -> pd.DataFrame:
    """Return long-form posterior sample rows."""
    object_id = np.asarray(object_id)
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
                "sample_id": int(sample_id),
                "logq": float(logq[sample_id, object_index]),
                "logprior": float(logprior[sample_id, object_index]),
                "loglike": float(loglike[sample_id, object_index]),
            }
            for param_index, name in enumerate(parameter_names):
                row[name] = float(theta[sample_id, object_index, param_index])
            rows.append(row)
    return pd.DataFrame(rows)


def posterior_summary_frame(
    object_id,
    theta,
    parameter_names: tuple[str, ...],
    loglike,
    chi2,
    mask,
) -> pd.DataFrame:
    """Return one posterior summary row per object."""
    object_id = np.asarray(object_id)
    theta = np.asarray(theta, dtype=float)
    loglike = np.asarray(loglike, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    rows = []
    for object_index, oid in enumerate(object_id):
        row = {
            "object_id": oid,
            "n_valid_bands": int(mask[object_index].sum()),
            "photometric_loglike_mean": float(np.nanmean(loglike[:, object_index])),
            "posterior_predictive_chi2_median": float(
                np.nanmedian(chi2[:, object_index])
            ),
            "flag_nonfinite_loglike": bool(
                not np.all(np.isfinite(loglike[:, object_index]))
            ),
        }
        for param_index, name in enumerate(parameter_names):
            values = theta[:, object_index, param_index]
            row[f"{name}_q16"] = float(np.nanquantile(values, 0.16))
            row[f"{name}_median"] = float(np.nanquantile(values, 0.50))
            row[f"{name}_q84"] = float(np.nanquantile(values, 0.84))
        rows.append(row)
    return pd.DataFrame(rows)


def posterior_predictive_flux_frame(
    object_id,
    model_flux,
    band_names: tuple[str, ...],
) -> pd.DataFrame:
    """Return long-form posterior predictive model flux rows."""
    object_id = np.asarray(object_id)
    model_flux = np.asarray(model_flux, dtype=float)
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
                        "sample_id": int(sample_id),
                        "band": band_names[band_index],
                        "model_flux_fnu_cgs": float(
                            model_flux[sample_id, object_index, band_index]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def learned_prior_samples_frame(
    x,
    theta,
    parameter_names: tuple[str, ...],
    logprior,
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
        }
        for x_index in range(x.shape[1]):
            row[f"x_{x_index:02d}"] = float(x[sample_id, x_index])
        for param_index, name in enumerate(parameter_names):
            row[name] = float(theta[sample_id, param_index])
        rows.append(row)
    return pd.DataFrame(rows)
