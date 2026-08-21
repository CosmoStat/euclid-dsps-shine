#!/usr/bin/env python3
"""Compare parent, forward-selected, inferred and FENIKS truth populations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.infer import _model_flux_from_x_2d_chunks
from euclid_dsps.amortized.latent import x_to_theta
from euclid_dsps.amortized.posterior import sample_posterior
from euclid_dsps.amortized.posterior_target import safe_decoder_inputs
from euclid_dsps.amortized.selection_correction import (
    observed_flux_selection_beta,
    observed_magnitude_flux_limit_jax,
)
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.calibration import (
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    per_band_flux_calibration_config,
)
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.model import dynamic_model_args, load_context
from euclid_dsps.photometric_uncertainty import m5_depth_flux_error_jax


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return float(np.interp(float(q), cumulative, sorted_values))


def _summary(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for distribution, group in frame.groupby("distribution", sort=False):
        weights = group["weight"].to_numpy(dtype=float)
        weights = weights / np.sum(weights)
        for name in names:
            values = group[name].to_numpy(dtype=float)
            mean = float(np.sum(weights * values))
            variance = float(np.sum(weights * (values - mean) ** 2))
            rows.append(
                {
                    "distribution": distribution,
                    "parameter": name,
                    "mean": mean,
                    "std": np.sqrt(max(variance, 0.0)),
                    "q05": _weighted_quantile(values, weights, 0.05),
                    "q50": _weighted_quantile(values, weights, 0.50),
                    "q95": _weighted_quantile(values, weights, 0.95),
                    "effective_rows": float(1.0 / np.sum(weights**2)),
                }
            )
    return pd.DataFrame(rows)


def _correlations(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for distribution, group in frame.groupby("distribution", sort=False):
        values = group.loc[:, names].to_numpy(dtype=float)
        weights = group["weight"].to_numpy(dtype=float)
        weights = weights / np.sum(weights)
        mean = np.sum(weights[:, None] * values, axis=0)
        centered = values - mean
        covariance = (centered * weights[:, None]).T @ centered
        scale = np.sqrt(np.maximum(np.diag(covariance), 1.0e-30))
        correlation = covariance / np.outer(scale, scale)
        for left, left_name in enumerate(names):
            for right, right_name in enumerate(names):
                rows.append(
                    {
                        "distribution": distribution,
                        "parameter_i": left_name,
                        "parameter_j": right_name,
                        "correlation": float(correlation[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def _population_comparisons(
    marginals: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    pairs = {
        "parent_prior_vs_parent_truth": ("parent_prior", "feniks_truth_parent"),
        "forward_selected_prior_vs_selected_truth": (
            "forward_selected_prior",
            "feniks_truth_selected",
        ),
        "catalog_inferred_vs_selected_truth": (
            "catalog_inferred_selected",
            "feniks_truth_selected",
        ),
    }
    rows = []
    for comparison, (candidate_name, truth_name) in pairs.items():
        candidate = marginals.loc[
            marginals["distribution"] == candidate_name
        ].set_index("parameter")
        truth = marginals.loc[marginals["distribution"] == truth_name].set_index(
            "parameter"
        )
        width = np.maximum(truth["q95"] - truth["q05"], 1.0e-12)
        quantile_error = (
            np.abs(candidate[["q05", "q50", "q95"]] - truth[["q05", "q50", "q95"]])
            .mean(axis=1)
            .to_numpy()
            / width.to_numpy()
        )
        candidate_corr = correlations.loc[
            correlations["distribution"] == candidate_name, "correlation"
        ].to_numpy(dtype=float)
        truth_corr = correlations.loc[
            correlations["distribution"] == truth_name, "correlation"
        ].to_numpy(dtype=float)
        std_ratio = candidate["std"].to_numpy() / np.maximum(
            truth["std"].to_numpy(), 1.0e-12
        )
        rows.append(
            {
                "comparison": comparison,
                "median_quantile_l1_over_truth_q90_width": float(
                    np.median(quantile_error)
                ),
                "max_quantile_l1_over_truth_q90_width": float(np.max(quantile_error)),
                "correlation_rmse": float(
                    np.sqrt(np.mean((candidate_corr - truth_corr) ** 2))
                ),
                "median_std_ratio": float(np.median(std_ratio)),
                "min_std_ratio": float(np.min(std_ratio)),
                "max_std_ratio": float(np.max(std_ratio)),
            }
        )
    return pd.DataFrame(rows)


def _distribution_frame(
    theta: np.ndarray,
    names: tuple[str, ...],
    *,
    distribution: str,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(theta), columns=names)
    frame.insert(0, "draw", np.arange(len(frame), dtype=np.int64))
    frame.insert(0, "distribution", distribution)
    if weights is None:
        weights = np.full(len(frame), 1.0 / len(frame), dtype=float)
    normalized = np.asarray(weights, dtype=float)
    normalized /= np.sum(normalized)
    frame["weight"] = normalized
    return frame


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    selected_rows = np.asarray(np.load(args.selected_indices), dtype=np.int64)
    if selected_rows.ndim != 1 or len(selected_rows) == 0:
        raise ValueError("selected-indices must contain at least one row")
    feature_stats = read_feature_stats(args.feature_stats)
    model = load_checkpoint(args.checkpoint, config)
    latent_spec = _latent_spec_for_amortized_config(config)
    names = tuple(latent_spec.names)
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=10_000,
        row_indices=selected_rows,
    )

    key = jax.random.PRNGKey(int(args.seed))
    inferred = []
    order = np.arange(len(arrays.object_id), dtype=np.int64)
    for batch_index, batch in enumerate(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=int(args.batch_size),
            feature_stats=feature_stats,
            order=order,
        )
    ):
        key, sample_key = jax.random.split(key)
        posterior = sample_posterior(
            model,
            sample_key,
            batch.features,
            int(args.q_samples_per_object),
        )
        theta = x_to_theta(posterior.x, latent_spec)
        inferred.append(np.asarray(jax.device_get(theta)).reshape(-1, len(names)))
        print(
            "[feniks-parent-population] "
            f"q_batch={batch_index + 1} objects={len(batch.object_id)}",
            flush=True,
        )
    inferred_theta = np.concatenate(inferred, axis=0)

    key, prior_key = jax.random.split(key)
    prior_x = model.prior.sample(prior_key, int(args.n_prior_samples))
    safe_prior_x, prior_valid = safe_decoder_inputs(prior_x, latent_spec)
    prior_theta = np.asarray(jax.device_get(x_to_theta(prior_x, latent_spec)))
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    prior_flux = _model_flux_from_x_2d_chunks(
        safe_prior_x,
        latent_spec,
        context,
        dynamic_model_args(context),
        names,
        batch_size=int(args.prior_decoder_batch_size),
    )
    calibration = {"calibration": config.get("calibration", {}) or {}}
    global_cfg = global_sed_scale_config(calibration)
    band_cfg = per_band_flux_calibration_config(calibration)
    if global_cfg.enabled:
        prior_flux = apply_global_sed_scale_to_flux(
            prior_flux, model.sed_scale.log_alpha_sed
        )
    if band_cfg.enabled:
        prior_flux = apply_per_band_flux_calibration_to_flux(
            prior_flux, model.band_calibration.log_alpha_band
        )
    band_names = tuple(feature_stats.band_names)
    r_index = band_names.index("lsst_r")
    noise_model = config["synthetic_diffsky"]["flux_error_model"]
    gamma_cfg = noise_model.get("gamma", {}) or {}
    gamma = float(gamma_cfg.get("lsst_r", 0.04 * noise_model.get("default_eta", 1.0)))
    sigma_r = m5_depth_flux_error_jax(
        prior_flux[:, r_index],
        float(noise_model["m5"]["lsst_r"]),
        gamma,
        sigma_sys_mag=float(noise_model.get("sigma_sys_mag", 0.0)),
        min_sigma_fnu_cgs=float(noise_model.get("min_sigma_fnu_cgs", 1.0e-40)),
    )
    beta = observed_flux_selection_beta(
        prior_flux[:, r_index],
        sigma_r,
        observed_magnitude_flux_limit_jax(25.0),
    )
    beta = np.asarray(jax.device_get(beta), dtype=float)
    beta *= np.asarray(jax.device_get(prior_valid), dtype=float)
    if not np.isfinite(beta).all() or np.sum(beta) <= 0.0:
        raise ValueError("forward selection produced invalid or zero total beta")

    truth = pq.read_table(args.dataset, columns=list(names)).to_pandas()
    truth_theta = truth.loc[:, names].to_numpy(dtype=float)
    selected_truth_theta = truth.iloc[selected_rows].loc[:, names].to_numpy(dtype=float)
    frames = [
        _distribution_frame(prior_theta, names, distribution="parent_prior"),
        _distribution_frame(
            prior_theta,
            names,
            distribution="forward_selected_prior",
            weights=beta,
        ),
        _distribution_frame(
            inferred_theta,
            names,
            distribution="catalog_inferred_selected",
        ),
        _distribution_frame(truth_theta, names, distribution="feniks_truth_parent"),
        _distribution_frame(
            selected_truth_theta,
            names,
            distribution="feniks_truth_selected",
        ),
    ]
    combined = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=False)
    combined.to_parquet(args.out / "population_draws.parquet", index=False)
    marginals = _summary(combined, names)
    correlations = _correlations(combined, names)
    comparisons = _population_comparisons(marginals, correlations)
    marginals.to_csv(args.out / "population_marginals.csv", index=False)
    correlations.to_csv(args.out / "population_correlations.csv", index=False)
    comparisons.to_csv(args.out / "population_comparisons.csv", index=False)
    payload = {
        "status": "complete",
        "contract": (
            "parent prior is compared only with parent truth; beta-weighted prior and "
            "catalog q are compared only with observed-r<25 selected truth"
        ),
        "truth_role": "FENIKS closure diagnostics only; never training or selection",
        "distributions": {
            name: int(len(frame)) for name, frame in combined.groupby("distribution")
        },
        "selection": {
            "band": "lsst_r",
            "max_mag_ab": 25.0,
            "probability_model": "gaussian_m5",
            "alpha_mc": float(np.mean(beta)),
            "prior_physical_valid_fraction": float(np.mean(np.asarray(prior_valid))),
        },
        "artifacts": {
            "draws": "population_draws.parquet",
            "marginals": "population_marginals.csv",
            "correlations": "population_correlations.csv",
            "comparisons": "population_comparisons.csv",
        },
    }
    (args.out / "population_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--selected-indices", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-prior-samples", type=int, default=8192)
    parser.add_argument("--q-samples-per-object", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prior-decoder-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260821)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
