"""DSPS closure photometry for synthetic Diffsky proposal rows."""

from __future__ import annotations

from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.filters import load_filters
from euclid_dsps.model import dynamic_model_args, load_context
from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
from euclid_dsps.photometric_uncertainty import flux_error_from_model
from euclid_dsps.photometry import abmag_to_fnu_cgs


GROUND_TRUTH_COLUMNS = {
    "z_obs": "redshift_true",
    "log10_stellar_mass": "logsm_true",
    **{
        name: f"{name}_true"
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
        if name not in {"z_obs", "log10_stellar_mass"}
    },
}


def theta_from_truth_frame(frame: pd.DataFrame) -> np.ndarray:
    """Return theta in canonical DIFFSKY_BASIC_PARAMETER_NAMES order."""
    missing = [
        column
        for name in DIFFSKY_BASIC_PARAMETER_NAMES
        for column in [GROUND_TRUTH_COLUMNS[name]]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing closure ground-truth columns: {missing}")
    theta = frame[[GROUND_TRUTH_COLUMNS[name] for name in DIFFSKY_BASIC_PARAMETER_NAMES]]
    arr = theta.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr).all(axis=1)).sum())
        raise ValueError(f"Closure theta contains {bad} non-finite rows")
    return arr


def add_dsps_closure_photometry(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    batch_size: int,
    noise_seed: int,
    flux_error_model: dict[str, Any],
    verbose: bool = False,
) -> pd.DataFrame:
    """Append true/noisy flux, fluxerr, mask, and model magnitude columns."""
    result = frame.copy()
    theta = theta_from_truth_frame(result)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int((config.get("model", {}) or {}).get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    mags = []
    n_batches = int(np.ceil(len(theta) / max(int(batch_size), 1)))
    for start in range(0, len(theta), int(batch_size)):
        batch_index = start // int(batch_size) + 1
        if verbose:
            end = min(start + int(batch_size), len(theta))
            print(
                "[diffsky][photometry] "
                f"DSPS batch {batch_index}/{n_batches} rows {start}:{end}",
                flush=True,
            )
        chunk = jnp.asarray(theta[start : start + int(batch_size)], dtype=jnp.float32)
        pred = model_mags_from_theta_matrix_jax(
            context,
            model_args,
            chunk,
            DIFFSKY_BASIC_PARAMETER_NAMES,
        )
        mags.append(np.asarray(jax.device_get(pred), dtype=float))
    mag_matrix = np.concatenate(mags, axis=0) if mags else np.empty((0, 0))
    rng = np.random.default_rng(int(noise_seed))
    if verbose:
        print(
            "[diffsky][photometry] applying flux error model "
            f"{str(flux_error_model.get('type', flux_error_model.get('kind', 'unknown')))} "
            f"with noise_seed={int(noise_seed)}",
            flush=True,
        )
    for band_index, band in enumerate(config["bands"]):
        band_name = str(band["name"])
        mag = mag_matrix[:, band_index]
        flux_true = np.asarray(abmag_to_fnu_cgs(mag), dtype=float)
        err_model = dict(band.get("error_model", flux_error_model) or flux_error_model)
        fluxerr = flux_error_from_model(flux_true, err_model, band_name=band_name)
        noise = rng.normal(0.0, fluxerr, size=len(result))
        flux = flux_true + noise
        mask = np.isfinite(flux_true) & np.isfinite(fluxerr) & (fluxerr > 0.0)
        result[f"mag_true_{band_name}"] = mag
        result[f"flux_true_{band_name}"] = flux_true
        result[f"flux_{band_name}"] = flux
        result[f"fluxerr_{band_name}"] = fluxerr
        result[f"mask_{band_name}"] = mask
    return result
