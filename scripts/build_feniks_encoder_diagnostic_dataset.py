#!/usr/bin/env python3
"""Build one exact-benchmark dataset mixing observed and true sleep pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.decoder import model_flux_from_x
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import x_to_theta
from euclid_dsps.amortized.posterior_target import safe_decoder_inputs
from euclid_dsps.amortized.train import (
    _repeat_sleep_rows,
    _sample_sleep_noise,
    _sleep_flux_error,
    _sleep_observed_selection_mask,
    _sleep_runtime_config,
)
from euclid_dsps.calibration import (
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    per_band_flux_calibration_config,
)
from euclid_dsps.config import load_config
from euclid_dsps.io import truth_column_from_spec
from scripts.run_feniks_exact_posterior_benchmark import _load_runtime_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _take_stratified(cohort: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministically across every existing observed stratum."""
    if count <= 0 or count > len(cohort):
        raise ValueError(f"count must be in [1, {len(cohort)}]")
    pieces = []
    labels = sorted(cohort["example_key"].astype(str).unique())
    remaining = int(count)
    for index, label in enumerate(labels):
        group = cohort.loc[cohort["example_key"].astype(str).eq(label)]
        groups_left = len(labels) - index
        target = min(len(group), max(1, remaining // groups_left))
        pieces.append(group.iloc[:target])
        remaining -= target
    selected = pd.concat(pieces, ignore_index=False)
    if remaining:
        available = cohort.drop(index=selected.index)
        selected = pd.concat((selected, available.iloc[:remaining]))
    return selected.iloc[:count].reset_index(drop=True)


def _generate_sleep_pairs(
    runtime,
    config,
    *,
    feature_stats: Path,
    count: int,
    seed: int,
):
    stats = read_feature_stats(feature_stats)
    sleep = _sleep_runtime_config(config, stats)
    candidate_count = max(int(count) * 64, 512)
    key = jax.random.PRNGKey(int(seed))
    prior_key, noise_key = jax.random.split(key)
    samples = runtime.model.prior.sample(prior_key, candidate_count)
    safe_samples, valid = safe_decoder_inputs(samples, runtime.latent_spec)
    model_flux = model_flux_from_x(
        safe_samples,
        runtime.latent_spec,
        runtime.context,
        runtime.model_args,
        runtime.latent_spec.names,
    )
    calibration = {"calibration": config.get("calibration", {}) or {}}
    scale_cfg = global_sed_scale_config(calibration)
    band_cfg = per_band_flux_calibration_config(calibration)
    if scale_cfg.enabled:
        model_flux = apply_global_sed_scale_to_flux(
            model_flux, runtime.model.sed_scale.log_alpha_sed
        )
    if band_cfg.enabled and runtime.model.band_calibration is not None:
        model_flux = apply_per_band_flux_calibration_to_flux(
            model_flux, runtime.model.band_calibration.log_alpha_band
        )
    valid &= jnp.all(jnp.isfinite(model_flux), axis=-1)
    mask = _repeat_sleep_rows(runtime.batch.mask, candidate_count)
    flux_err = _sleep_flux_error(model_flux, runtime.batch, sleep)
    noise, noise_family = _sample_sleep_noise(
        noise_key,
        flux_err,
        sleep=sleep,
        likelihood_config=config["amortized"]["likelihood"],
    )
    noisy_flux = model_flux + noise
    valid &= jnp.all(jnp.isfinite(flux_err) & (flux_err > 0.0), axis=-1)
    valid &= jnp.all(jnp.isfinite(noisy_flux), axis=-1)
    valid = _sleep_observed_selection_mask(
        noisy_flux,
        valid,
        band_index=int(sleep["selection_band_index"]),
        flux_min=jnp.asarray(
            sleep["selection_flux_min_fnu_cgs"], dtype=model_flux.dtype
        ),
        observed_mask=mask,
    )
    valid_index = np.flatnonzero(np.asarray(jax.device_get(valid)))
    if len(valid_index) < count:
        raise RuntimeError(
            f"sleep rejection pool retained {len(valid_index)}/{candidate_count}; "
            f"need {count}"
        )
    selected = valid_index[:count]
    selected_x = np.asarray(jax.device_get(samples[selected]))
    theta = np.asarray(
        jax.device_get(x_to_theta(jnp.asarray(selected_x), runtime.latent_spec))
    )
    return {
        "x": selected_x,
        "theta": theta,
        "flux": np.asarray(jax.device_get(noisy_flux[selected])),
        "flux_err": np.asarray(jax.device_get(flux_err[selected])),
        "mask": np.asarray(jax.device_get(mask[selected]), dtype=bool),
        "noise_family": noise_family,
        "candidate_count": candidate_count,
        "selected_count": int(len(valid_index)),
    }


def build(
    *,
    config_path: Path,
    dataset: Path,
    checkpoint: Path,
    feature_stats: Path,
    observed_cohort: Path,
    out: Path,
    n_observed: int,
    n_sleep: int,
    seed: int,
) -> dict[str, object]:
    config = load_config(config_path)
    config["catalog_path"] = str(dataset)
    source_cohort = pd.read_csv(observed_cohort)
    observed = _take_stratified(source_cohort, int(n_observed))
    template_rows = observed["row_index"].to_numpy(dtype=np.int64)
    runtime_args = SimpleNamespace(
        checkpoint=checkpoint,
        feature_stats=feature_stats,
    )
    runtime = _load_runtime_rows(runtime_args, config, template_rows)
    generated = _generate_sleep_pairs(
        runtime,
        config,
        feature_stats=feature_stats,
        count=int(n_sleep),
        seed=seed,
    )

    source = pd.read_parquet(dataset)
    observed_frame = source.iloc[template_rows].copy().reset_index(drop=True)
    template_positions = np.resize(np.arange(len(observed_frame)), int(n_sleep))
    sleep_frame = observed_frame.iloc[template_positions].copy().reset_index(drop=True)
    for band_index, band in enumerate(config["bands"]):
        sleep_frame[str(band["column"])] = generated["flux"][:, band_index]
        sleep_frame[str(band["error_column"])] = generated["flux_err"][:, band_index]
        mask_column = band.get("mask_column")
        if mask_column:
            sleep_frame[str(mask_column)] = generated["mask"][:, band_index]
    truth_cfg = (config.get("truth", {}) or {}).get("parameter_columns", {}) or {}
    for parameter_index, name in enumerate(runtime.latent_spec.names):
        column = truth_column_from_spec(truth_cfg.get(name))
        if not column:
            raise ValueError(f"missing truth column contract for {name}")
        sleep_frame[column] = generated["theta"][:, parameter_index]
    if "object_id" in sleep_frame:
        sleep_frame["object_id"] = [
            f"sleep-synthetic-{index:04d}" for index in range(int(n_sleep))
        ]
    combined = pd.concat((observed_frame, sleep_frame), ignore_index=True)
    out.mkdir(parents=True, exist_ok=False)
    dataset_out = out / "diagnostic_dataset.parquet"
    combined.to_parquet(dataset_out, index=False)

    observed_rows = []
    for order, item in enumerate(observed.itertuples(index=False)):
        observed_rows.append(
            {
                "order": order,
                "example_key": f"observed__{item.example_key}",
                "row_index": order,
                "object_id": str(item.object_id),
                "domain": "observed_catalog",
                "source_row_index": int(item.row_index),
            }
        )
    sleep_rows = [
        {
            "order": int(n_observed) + index,
            "example_key": "sleep_synthetic",
            "row_index": int(n_observed) + index,
            "object_id": f"sleep-synthetic-{index:04d}",
            "domain": "sleep_synthetic",
            "source_row_index": int(template_rows[index % len(template_rows)]),
        }
        for index in range(int(n_sleep))
    ]
    cohort = pd.DataFrame(observed_rows + sleep_rows)
    cohort_out = out / "diagnostic_cohort.csv"
    cohort.to_csv(cohort_out, index=False)
    payload = {
        "status": "complete",
        "contract": "ENCODER_DIAGNOSTIC_ONLY",
        "truth_role": "closure diagnostics only; never training or selection",
        "observed_objects": int(n_observed),
        "sleep_objects": int(n_sleep),
        "sleep_noise_family": generated["noise_family"],
        "sleep_candidate_count": generated["candidate_count"],
        "sleep_selected_count": generated["selected_count"],
        "dataset": str(dataset_out),
        "dataset_sha256": _sha256(dataset_out),
        "cohort": str(cohort_out),
        "cohort_sha256": _sha256(cohort_out),
        "source_dataset": str(dataset),
        "source_dataset_sha256": _sha256(dataset),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "feature_stats": str(feature_stats),
        "feature_stats_sha256": _sha256(feature_stats),
        "seed": int(seed),
    }
    (out / "diagnostic_dataset_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--observed-cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-observed", type=int, default=16)
    parser.add_argument("--n-sleep", type=int, default=16)
    parser.add_argument("--seed", type=int, default=260821)
    args = parser.parse_args()
    build(
        config_path=args.config,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        feature_stats=args.feature_stats,
        observed_cohort=args.observed_cohort,
        out=args.out,
        n_observed=args.n_observed,
        n_sleep=args.n_sleep,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
