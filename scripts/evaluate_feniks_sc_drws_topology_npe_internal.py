#!/usr/bin/env python3
"""Truth-free held-out, SBC and sensitivity validation for one NPE arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.data import load_photometry_arrays_from_config
from euclid_dsps.amortized.decoder import model_flux_from_x
from euclid_dsps.amortized.features import make_encoder_features, read_feature_stats
from euclid_dsps.amortized.latent import latent_spec_from_config, theta_to_x, x_to_theta
from euclid_dsps.amortized.likelihood import photometric_normalized_residual
from euclid_dsps.amortized.npe_validation import (
    flux_error_jacobian_sensitivity,
    held_out_band_predictive_gate,
    mask_held_out_bands,
    summarize_model_generated_rank_calibration,
    summarize_normalized_residuals,
)
from euclid_dsps.amortized.posterior import sample_posterior
from euclid_dsps.amortized.posterior_target import (
    _apply_model_calibration,
    safe_decoder_inputs,
)
from euclid_dsps.amortized.train import (
    LossBatch,
    _repeat_sleep_rows,
    _sample_sleep_noise,
    _sleep_encoder_features,
    _sleep_flux_error,
    _sleep_observed_selection_mask,
    _sleep_runtime_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import dynamic_model_args, load_context


def _residual_frame(
    residual: np.ndarray,
    mask: np.ndarray,
    band_names: tuple[str, ...],
) -> pd.DataFrame:
    values = np.asarray(residual, dtype=float)
    valid = np.broadcast_to(np.asarray(mask, dtype=bool)[None, ...], values.shape)
    draw, obj, band = np.nonzero(valid & np.isfinite(values))
    return pd.DataFrame(
        {
            "draw": draw,
            "object": obj,
            "band": np.asarray(band_names, dtype=object)[band],
            "residual_sigma": values[draw, obj, band],
        }
    )


def _decode(model, x, spec, context, model_args, config):
    safe, valid = safe_decoder_inputs(x, spec)
    raw = model_flux_from_x(safe, spec, context, model_args, spec.names)
    flux = _apply_model_calibration(
        model,
        raw,
        {"calibration": config.get("calibration", {}) or {}},
    )
    valid &= jnp.all(jnp.isfinite(flux), axis=-1)
    return flux, valid


def evaluate(
    *,
    config_path: Path,
    checkpoint: Path,
    feature_stats_path: Path,
    dataset: Path,
    row_indices_path: Path,
    out: Path,
    objects: int,
    posterior_draws: int,
    seed: int,
) -> dict:
    config = load_config(config_path)
    if (config.get("truth", {}) or {}).get("parameter_columns"):
        raise ValueError("internal validation config exposes catalogue truth")
    config = dict(config)
    config["catalog_path"] = str(dataset.resolve())
    rows = np.load(row_indices_path, allow_pickle=False).astype(np.int64)[:objects]
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=max(objects, 1),
        row_indices=rows,
    )
    stats = read_feature_stats(feature_stats_path)
    spec = latent_spec_from_config(config)
    model = load_checkpoint(checkpoint, config)
    context = load_context(config)
    model_args = dynamic_model_args(context)
    likelihood = config["amortized"]["likelihood"]
    sleep = _sleep_runtime_config(config, stats)
    n_objects = int(len(rows))
    candidate_factor = max(4, int(sleep.get("selection_candidate_factor", 1)))
    candidates = n_objects * candidate_factor
    base_batch = LossBatch(
        flux=jnp.asarray(arrays.flux),
        flux_err=jnp.asarray(arrays.flux_err),
        mask=jnp.asarray(arrays.mask),
        features=make_encoder_features(
            jnp.asarray(arrays.flux),
            jnp.asarray(arrays.flux_err),
            stats,
            jnp.asarray(arrays.mask),
        ),
        truth_theta=jnp.zeros((n_objects, 0), dtype=jnp.float32),
    )
    prior_key, noise_key, posterior_key, heldout_key, reference_key = jax.random.split(
        jax.random.PRNGKey(int(seed)), 5
    )
    generated_x = model.prior.sample(prior_key, candidates)
    generated_flux, physical_valid = _decode(
        model, generated_x, spec, context, model_args, config
    )
    generated_mask = _repeat_sleep_rows(base_batch.mask, candidates)
    generated_error = _sleep_flux_error(generated_flux, base_batch, sleep)
    noise, noise_family = _sample_sleep_noise(
        noise_key,
        generated_error,
        sleep=sleep,
        likelihood_config=likelihood,
    )
    noisy_flux = generated_flux + noise
    physical_valid &= jnp.all(jnp.isfinite(noisy_flux), axis=-1)
    physical_valid &= jnp.all(
        jnp.isfinite(generated_error) & (generated_error > 0), axis=-1
    )
    selection_band = sleep.get("selection_band_index")
    if selection_band is not None:
        physical_valid = _sleep_observed_selection_mask(
            noisy_flux,
            physical_valid,
            band_index=int(selection_band),
            flux_min=float(sleep["selection_flux_min_fnu_cgs"]),
            observed_mask=generated_mask,
        )
    selected = np.flatnonzero(np.asarray(jax.device_get(physical_valid)))[:n_objects]
    if len(selected) < n_objects:
        raise ValueError(
            f"model-generated validation accepted {len(selected)}/{n_objects} rows"
        )
    generated_x = jnp.take(generated_x, selected, axis=0)
    generated_flux = jnp.take(generated_flux, selected, axis=0)
    generated_error = jnp.take(generated_error, selected, axis=0)
    generated_mask = jnp.take(generated_mask, selected, axis=0)
    noisy_flux = jnp.take(noisy_flux, selected, axis=0)
    generated_features = _sleep_encoder_features(
        noisy_flux,
        generated_error,
        generated_mask,
        sleep,
    )
    generated_posterior = sample_posterior(
        model,
        posterior_key,
        generated_features,
        int(posterior_draws),
    )
    rank_summary = summarize_model_generated_rank_calibration(
        np.asarray(jax.device_get(generated_posterior.x)),
        np.asarray(jax.device_get(generated_x)),
        parameter_names=spec.names,
        seed=int(seed) + 1,
        maximum_ks=float(
            config["amortized"]["truth_free_validation"].get(
                "maximum_model_generated_pit_ks", 0.12
            )
        ),
        maximum_coverage_ece=float(
            config["amortized"]["truth_free_validation"].get(
                "maximum_model_generated_coverage_ece", 0.12
            )
        ),
    )
    roundtrip = theta_to_x(x_to_theta(generated_x, spec), spec)
    roundtrip_error = float(
        np.max(np.abs(np.asarray(jax.device_get(roundtrip - generated_x))))
    )

    held_names = tuple(
        config["amortized"]["truth_free_validation"].get("held_out_bands", ())
    )
    held_indices = [arrays.band_names.index(name) for name in held_names]
    observed_features, observed_conditioning_mask = mask_held_out_bands(
        jnp.asarray(arrays.flux),
        jnp.asarray(arrays.flux_err),
        jnp.asarray(arrays.mask),
        stats,
        held_indices,
    )
    if bool(np.any(np.asarray(observed_conditioning_mask)[:, held_indices])):
        raise RuntimeError("held-out bands leaked into the conditioning mask")
    observed_q = sample_posterior(
        model,
        heldout_key,
        observed_features,
        int(posterior_draws),
    )
    observed_model_flux, _ = _decode(
        model, observed_q.x, spec, context, model_args, config
    )
    held_mask = np.zeros_like(arrays.mask, dtype=bool)
    held_mask[:, held_indices] = arrays.mask[:, held_indices]
    observed_residual = photometric_normalized_residual(
        obs_flux=jnp.asarray(arrays.flux),
        model_flux=observed_model_flux,
        obs_err=jnp.asarray(arrays.flux_err),
        mask=jnp.asarray(held_mask),
        error_floor_frac=float(likelihood.get("error_floor_frac", 0.0)),
        error_jitter=float(likelihood.get("error_jitter", 0.0)),
    )
    reference_features, reference_conditioning_mask = mask_held_out_bands(
        noisy_flux,
        generated_error,
        generated_mask,
        stats,
        held_indices,
    )
    if bool(np.any(np.asarray(reference_conditioning_mask)[:, held_indices])):
        raise RuntimeError("held-out bands leaked into simulated conditioning")
    reference_q = sample_posterior(
        model,
        reference_key,
        reference_features,
        int(posterior_draws),
    )
    reference_model_flux, _ = _decode(
        model, reference_q.x, spec, context, model_args, config
    )
    reference_held_mask = np.zeros_like(np.asarray(generated_mask), dtype=bool)
    reference_held_mask[:, held_indices] = np.asarray(generated_mask)[:, held_indices]
    reference_residual = photometric_normalized_residual(
        obs_flux=noisy_flux,
        model_flux=reference_model_flux,
        obs_err=generated_error,
        mask=jnp.asarray(reference_held_mask),
        error_floor_frac=float(likelihood.get("error_floor_frac", 0.0)),
        error_jitter=float(likelihood.get("error_jitter", 0.0)),
    )
    observed_residual_summary = summarize_normalized_residuals(
        _residual_frame(
            np.asarray(jax.device_get(observed_residual)),
            held_mask,
            tuple(arrays.band_names),
        )
    )
    reference_residual_summary = summarize_normalized_residuals(
        _residual_frame(
            np.asarray(jax.device_get(reference_residual)),
            reference_held_mask,
            tuple(arrays.band_names),
        )
    )
    heldout = held_out_band_predictive_gate(
        observed_residual_summary,
        reference_residual_summary,
    )

    sensitivity_rows = []
    jacobian_objects = min(8, n_objects)
    for index in range(jacobian_objects):
        result = flux_error_jacobian_sensitivity(
            lambda value: _decode(model, value, spec, context, model_args, config)[0],
            generated_x[index],
            generated_error[index],
            generated_mask[index],
        )
        sensitivity_rows.append(
            {
                "coordinate_norm": np.asarray(
                    jax.device_get(result["coordinate_norm"]), dtype=float
                ),
                "singular_values": np.asarray(
                    jax.device_get(result["singular_values"]), dtype=float
                ),
                "near_zero": np.asarray(
                    jax.device_get(result["near_zero_coordinate"]), dtype=bool
                ),
            }
        )
    coordinate_norm = np.stack([row["coordinate_norm"] for row in sensitivity_rows])
    near_zero = np.stack([row["near_zero"] for row in sensitivity_rows])
    theta = np.asarray(jax.device_get(x_to_theta(generated_posterior.x, spec)))
    lower = np.asarray(spec.lower, dtype=float)
    upper = np.asarray(spec.upper, dtype=float)
    span = np.maximum(upper - lower, 1.0e-12)
    near_boundary = ((theta - lower) / span < 0.01) | ((upper - theta) / span < 0.01)
    sensitivity = {
        "objects": jacobian_objects,
        "coordinates": {
            name: {
                "median_flux_error_jacobian_norm": float(
                    np.median(coordinate_norm[:, i])
                ),
                "minimum_flux_error_jacobian_norm": float(
                    np.min(coordinate_norm[:, i])
                ),
                "near_zero_fraction": float(np.mean(near_zero[:, i])),
                "posterior_near_physical_boundary_fraction": float(
                    np.mean(near_boundary[:, :, i])
                ),
            }
            for i, name in enumerate(spec.names)
        },
        "contract": "DSPS flux Jacobian divided by observed error; no catalogue truth",
    }
    ensure_dir(out)
    payload = {
        "status": "COMPLETE",
        "truth_used": False,
        "objects": n_objects,
        "posterior_draws": int(posterior_draws),
        "noise_family": noise_family,
        "selection": {
            "event": "noisy_lsst_r_ab_lt_29",
            "candidate_factor": candidate_factor,
            "accepted_fraction": float(np.mean(np.asarray(physical_valid))),
        },
        "latent_contract": {
            "parameter_names": list(spec.names),
            "theta_x_roundtrip_max_abs_error": roundtrip_error,
            "sfh_parameters": [name for name in spec.names if name.startswith("sfh_")],
        },
        "model_generated_calibration": rank_summary,
        "held_out_band": heldout,
        "held_out_observed_residuals": observed_residual_summary,
        "held_out_model_generated_reference": reference_residual_summary,
        "sensitivity": sensitivity,
        "catalogue_truth_columns_read": [],
        "scientific_promotion": False,
    }
    write_json(out / "INTERNAL_TRUTH_FREE_VALIDATION.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--feature-stats", dest="feature_stats_path", type=Path, required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--row-indices", dest="row_indices_path", type=Path, required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objects", type=int, default=256)
    parser.add_argument("--posterior-draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260906)
    args = parser.parse_args()
    print(json.dumps(evaluate(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
