#!/usr/bin/env python3
"""Run adaptive q-to-target SMC on a frozen Pop-COSMOS cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.adaptive_smc import (
    build_adaptive_smc_kernels,
    run_adaptive_smc,
)
from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.decoder import model_flux_from_x
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import x_to_theta
from euclid_dsps.amortized.likelihood import (
    photometric_loglike,
    photometric_normalized_residual,
)
from euclid_dsps.amortized.posterior import posterior_log_prob, sample_posterior
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--row-indices", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--object-batch-size", type=int, default=4)
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--target-ess-fraction", type=float, default=0.5)
    parser.add_argument("--max-stages", type=int, default=64)
    parser.add_argument("--mala-steps", type=int, default=2)
    parser.add_argument("--mala-step-size", type=float, default=0.02)
    parser.add_argument("--mala-particle-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=260817)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite SMC output: {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "weighted_particles").mkdir()

    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    cfg = amortized_config(config)
    likelihood = cfg["likelihood"]
    selected_rows = np.asarray(np.load(args.row_indices), dtype=np.int64)[: args.limit]
    if len(selected_rows) != args.limit or len(np.unique(selected_rows)) != args.limit:
        raise ValueError("row-indices must provide LIMIT unique rows")
    feature_stats = read_feature_stats(args.feature_stats)
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=10_000,
        row_indices=selected_rows,
    )
    if arrays.row_index is None:
        raise ValueError("Selected catalog does not expose stable row indices")
    position = {int(value): index for index, value in enumerate(arrays.row_index)}
    missing = [int(value) for value in selected_rows if int(value) not in position]
    if missing:
        raise ValueError(f"Requested rows not loaded: {missing[:10]}")
    order = np.asarray([position[int(value)] for value in selected_rows], dtype=int)

    model = load_checkpoint(args.checkpoint, config)
    latent_spec = _latent_spec_for_amortized_config(config)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    scale_cfg = global_sed_scale_config(config)
    band_cfg = per_band_flux_calibration_config(config)
    band_names = tuple(str(item["name"]) for item in config["bands"])

    def logq_fn(values, features, _flux, _flux_err, _mask):
        return posterior_log_prob(model, features, values)

    def components(values, _features, observed_flux, observed_flux_err, mask):
        raw_flux = model_flux_from_x(
            values,
            latent_spec,
            context,
            model_args,
            latent_spec.names,
        )
        flux = (
            apply_global_sed_scale_to_flux(raw_flux, model.sed_scale.log_alpha_sed)
            if scale_cfg.enabled
            else raw_flux
        )
        if band_cfg.enabled and model.band_calibration is not None:
            flux = apply_per_band_flux_calibration_to_flux(
                flux, model.band_calibration.log_alpha_band
            )
        loglike = photometric_loglike(
            observed_flux,
            flux,
            observed_flux_err,
            mask,
            likelihood_type=str(likelihood["type"]),
            student_t_dof=float(likelihood["student_t_dof"]),
            error_floor_frac=float(likelihood["error_floor_frac"]),
            error_jitter=float(likelihood["error_jitter"]),
        )
        return model.prior.log_prob(values), loglike, flux

    components_jit = jax.jit(components)

    def target_fn(values, *density_args):
        logprior, loglike, _flux = components_jit(values, *density_args)
        return logprior + loglike

    kernels = build_adaptive_smc_kernels(
        proposal_logdensity_fn=logq_fn,
        target_logdensity_fn=target_fn,
        mala_step_size=args.mala_step_size,
    )

    print(
        "[posthoc-smc] "
        f"objects={args.limit} particles={args.particles} "
        f"floor={likelihood['error_floor_frac']} seed={args.seed} "
        f"batch={args.object_batch_size} "
        f"mala_particle_chunk={args.mala_particle_chunk_size}",
        flush=True,
    )
    key = jax.random.PRNGKey(args.seed)
    object_frames: list[pd.DataFrame] = []
    stage_frames: list[pd.DataFrame] = []
    band_accumulator: list[pd.DataFrame] = []
    start_time = time.perf_counter()
    batches = iter_photometry_batches_from_arrays(
        arrays,
        batch_size=args.object_batch_size,
        feature_stats=feature_stats,
        order=order,
    )
    processed = 0
    for batch_index, batch in enumerate(batches):
        key, sample_key, smc_key = jax.random.split(key, 3)
        proposal = sample_posterior(model, sample_key, batch.features, args.particles)
        density_args = (batch.features, batch.flux, batch.flux_err, batch.mask)

        batch_start = time.perf_counter()
        result = run_adaptive_smc(
            key=smc_key,
            initial_particles=proposal.x,
            proposal_logdensity_fn=logq_fn,
            target_logdensity_fn=target_fn,
            target_ess_fraction=args.target_ess_fraction,
            max_stages=args.max_stages,
            mala_steps=args.mala_steps,
            mala_step_size=args.mala_step_size,
            mala_particle_chunk_size=args.mala_particle_chunk_size,
            density_args=density_args,
            kernels=kernels,
        )
        jax.block_until_ready(result.particles)
        logprior, loglike, model_flux = components_jit(result.particles, *density_args)
        theta = x_to_theta(result.particles, latent_spec)
        batch_seconds = time.perf_counter() - batch_start
        particle_frame = _particle_frame(
            batch,
            result,
            theta=np.asarray(theta),
            logprior=np.asarray(logprior),
            loglike=np.asarray(loglike),
            parameter_names=latent_spec.names,
        )
        particle_frame.to_parquet(
            args.out / "weighted_particles" / f"batch_{batch_index:06d}.parquet",
            index=False,
        )
        object_frame, band_frame = _diagnostic_frames(
            batch,
            result,
            model_flux=np.asarray(model_flux),
            likelihood=likelihood,
            band_names=band_names,
            latent_dim=len(latent_spec.names),
        )
        object_frame["batch_seconds"] = batch_seconds
        object_frames.append(object_frame)
        band_accumulator.append(band_frame)
        stage_frames.append(_stage_frame(batch, result))
        processed += len(batch.object_id)
        print(
            "[posthoc-smc] "
            f"batch={batch_index + 1} processed={processed}/{args.limit} "
            f"stages={result.beta_to.shape[0]} seconds={batch_seconds:.1f} "
            f"median_final_ess={object_frame['final_ess_fraction'].median():.3f} "
            f"median_ancestors={object_frame['unique_ancestor_fraction'].median():.3f}",
            flush=True,
        )

    objects = pd.concat(object_frames, ignore_index=True)
    stages = pd.concat(stage_frames, ignore_index=True)
    bands = pd.concat(band_accumulator, ignore_index=True)
    objects.to_parquet(args.out / "smc_object_diagnostics.parquet", index=False)
    stages.to_parquet(args.out / "smc_stage_diagnostics.parquet", index=False)
    bands.to_parquet(
        args.out / "posterior_predictive_band_objects.parquet", index=False
    )
    _summarize_bands(bands).to_csv(
        args.out / "posterior_predictive_by_band.csv", index=False
    )
    summary = _summary(args, config, likelihood, objects, stages, start_time)
    (args.out / "smc_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "support_gate.json").write_text(
        json.dumps(summary["support_gate"], indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.save(args.out / "row_indices.npy", selected_rows)
    (args.out / "DONE").touch()
    print(
        f"[posthoc-smc] complete support={summary['support_gate']['status']} "
        f"-> {args.out}",
        flush=True,
    )


def _particle_frame(batch, result, *, theta, logprior, loglike, parameter_names):
    particles = np.asarray(result.particles)
    n_samples, n_objects, latent_dim = particles.shape
    payload: dict[str, np.ndarray] = {
        "object_id": np.repeat(np.asarray(batch.object_id), n_samples),
        "row_index": np.repeat(np.asarray(batch.row_index), n_samples),
        "sample_id": np.tile(np.arange(n_samples), n_objects),
        "smc_weight": np.asarray(result.weights).T.reshape(-1),
        "smc_logweight": np.asarray(result.log_weights).T.reshape(-1),
        "logq": np.asarray(result.proposal_logdensity).T.reshape(-1),
        "logprior": logprior.T.reshape(-1),
        "loglike": loglike.T.reshape(-1),
        "ancestor_id": np.asarray(result.ancestor_ids).T.reshape(-1),
    }
    theta_object_major = np.swapaxes(theta, 0, 1).reshape(-1, latent_dim)
    x_object_major = np.swapaxes(particles, 0, 1).reshape(-1, latent_dim)
    for index, name in enumerate(parameter_names):
        payload[str(name)] = theta_object_major[:, index]
        payload[f"latent_x_{name}"] = x_object_major[:, index]
    return pd.DataFrame(payload)


def _diagnostic_frames(
    batch, result, *, model_flux, likelihood, band_names, latent_dim
):
    weights = np.asarray(result.weights)
    mask = np.asarray(batch.mask)
    chi = np.asarray(
        photometric_normalized_residual(
            batch.flux,
            jnp.asarray(model_flux),
            batch.flux_err,
            batch.mask,
            error_floor_frac=float(likelihood["error_floor_frac"]),
            error_jitter=float(likelihood["error_jitter"]),
        )
    )
    rows = []
    band_rows = []
    ancestors = np.asarray(result.ancestor_ids)
    for object_index, row_index in enumerate(batch.row_index):
        weight = weights[:, object_index]
        valid = mask[object_index]
        active_stage = (
            result.beta_to[:, object_index] > result.beta_from[:, object_index]
        )
        chi2 = np.sum(np.square(chi[:, object_index, valid]), axis=1)
        n_valid = int(np.sum(valid))
        reduced_denom = max(n_valid - int(latent_dim), 1)
        rows.append(
            {
                "object_id": int(np.asarray(batch.object_id)[object_index]),
                "row_index": int(row_index),
                "log_evidence": float(np.asarray(result.log_evidence)[object_index]),
                "final_ess": float(1.0 / np.sum(np.square(weight))),
                "final_ess_fraction": float(
                    1.0 / np.sum(np.square(weight)) / len(weight)
                ),
                "max_final_weight": float(np.max(weight)),
                "unique_ancestor_fraction": float(
                    len(np.unique(ancestors[:, object_index])) / len(weight)
                ),
                "min_stage_ess_fraction": float(
                    np.min(result.pre_resample_ess[active_stage, object_index])
                    / len(weight)
                ),
                "mean_mala_acceptance": float(
                    np.mean(result.mala_acceptance[active_stage, object_index])
                ),
                "n_stages": int(np.sum(active_stage)),
                "weighted_chi2_per_valid_band": float(
                    np.sum(weight * chi2) / max(n_valid, 1)
                ),
                "weighted_reduced_chi2": float(np.sum(weight * chi2) / reduced_denom),
                "weighted_fraction_abs_gt_5": float(
                    np.sum(
                        weight[:, None] * (np.abs(chi[:, object_index, valid]) > 5.0)
                    )
                    / max(n_valid, 1)
                ),
            }
        )
        for band_index, band in enumerate(band_names):
            if not valid[band_index]:
                continue
            values = chi[:, object_index, band_index]
            band_rows.append(
                {
                    "row_index": int(row_index),
                    "band": band,
                    "weighted_abs_chi": float(np.sum(weight * np.abs(values))),
                    "weighted_frac_abs_gt_5": float(
                        np.sum(weight * (np.abs(values) > 5.0))
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(band_rows)


def _stage_frame(batch, result):
    rows = []
    for stage in range(result.beta_to.shape[0]):
        for object_index, row_index in enumerate(batch.row_index):
            if (
                result.beta_to[stage, object_index]
                <= result.beta_from[stage, object_index]
            ):
                continue
            rows.append(
                {
                    "row_index": int(row_index),
                    "stage": stage + 1,
                    "beta_from": float(result.beta_from[stage, object_index]),
                    "beta_to": float(result.beta_to[stage, object_index]),
                    "pre_resample_ess": float(
                        result.pre_resample_ess[stage, object_index]
                    ),
                    "pre_resample_ess_fraction": float(
                        result.pre_resample_ess[stage, object_index]
                        / result.particles.shape[0]
                    ),
                    "resampled": bool(result.resampled[stage, object_index]),
                    "mala_acceptance": float(
                        result.mala_acceptance[stage, object_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _summarize_bands(frame):
    return (
        frame.groupby("band", sort=False)
        .agg(
            n_objects=("row_index", "size"),
            median_weighted_abs_chi=("weighted_abs_chi", "median"),
            median_weighted_frac_abs_gt_5=("weighted_frac_abs_gt_5", "median"),
        )
        .reset_index()
    )


def _summary(args, config, likelihood, objects, stages, start_time):
    checks = {
        "all_objects_reached_beta_one": bool(
            np.allclose(stages.groupby("row_index")["beta_to"].max(), 1.0)
        ),
        "median_final_ess_fraction_ge_0p2": bool(
            objects["final_ess_fraction"].median() >= 0.2
        ),
        "median_unique_ancestor_fraction_ge_0p05": bool(
            objects["unique_ancestor_fraction"].median() >= 0.05
        ),
        "median_max_final_weight_le_0p1": bool(
            objects["max_final_weight"].median() <= 0.1
        ),
        "median_mala_acceptance_between_0p15_0p8": bool(
            0.15 <= objects["mean_mala_acceptance"].median() <= 0.8
        ),
        "fraction_objects_final_ess_ge_0p1_ge_0p9": bool(
            np.mean(objects["final_ess_fraction"] >= 0.1) >= 0.9
        ),
        "fraction_objects_max_weight_le_0p2_ge_0p9": bool(
            np.mean(objects["max_final_weight"] <= 0.2) >= 0.9
        ),
        "fraction_objects_mala_acceptance_0p05_0p95_ge_0p9": bool(
            np.mean(objects["mean_mala_acceptance"].between(0.05, 0.95)) >= 0.9
        ),
    }
    return {
        "status": "complete",
        "algorithm": "adaptive q-to-target SMC with multinomial resampling and MALA",
        "n_objects": int(len(objects)),
        "particles_per_object": int(args.particles),
        "seed": int(args.seed),
        "likelihood": dict(likelihood),
        "density_space": "unconstrained network latent_x",
        "target_contract": "log p_x(x) + log p(photometry|theta(x))",
        "proposal_contract": "frozen amortized joint q_x(x|photometry)",
        "selection_contract": "photometric diagnostics only; spectroscopy unused",
        "target_ess_fraction": float(args.target_ess_fraction),
        "mala_steps": int(args.mala_steps),
        "mala_step_size": float(args.mala_step_size),
        "mala_particle_chunk_size": int(args.mala_particle_chunk_size),
        "wall_seconds": float(time.perf_counter() - start_time),
        "metrics": {
            "mean_log_evidence": float(objects["log_evidence"].mean()),
            "median_log_evidence": float(objects["log_evidence"].median()),
            "median_final_ess_fraction": float(objects["final_ess_fraction"].median()),
            "median_unique_ancestor_fraction": float(
                objects["unique_ancestor_fraction"].median()
            ),
            "median_max_final_weight": float(objects["max_final_weight"].median()),
            "median_mala_acceptance": float(objects["mean_mala_acceptance"].median()),
            "p10_final_ess_fraction": float(
                objects["final_ess_fraction"].quantile(0.1)
            ),
            "p10_unique_ancestor_fraction": float(
                objects["unique_ancestor_fraction"].quantile(0.1)
            ),
            "p90_max_final_weight": float(objects["max_final_weight"].quantile(0.9)),
            "p10_mala_acceptance": float(objects["mean_mala_acceptance"].quantile(0.1)),
            "p90_mala_acceptance": float(objects["mean_mala_acceptance"].quantile(0.9)),
            "median_chi2_per_valid_band": float(
                objects["weighted_chi2_per_valid_band"].median()
            ),
            "median_reduced_chi2": float(objects["weighted_reduced_chi2"].median()),
            "median_fraction_abs_gt_5": float(
                objects["weighted_fraction_abs_gt_5"].median()
            ),
        },
        "support_gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
        },
        "inputs": {
            "config": _receipt(args.config),
            "dataset": _receipt(args.dataset),
            "checkpoint": _receipt(args.checkpoint),
            "feature_stats": _receipt(args.feature_stats),
            "row_indices": _receipt(args.row_indices),
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "bands": [str(item["name"]) for item in config["bands"]],
    }


def _receipt(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _validate_args(args):
    for path in (
        args.config,
        args.dataset,
        args.checkpoint,
        args.feature_stats,
        args.row_indices,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.limit <= 0 or args.object_batch_size <= 0 or args.particles < 2:
        raise ValueError("limit/batch must be positive and particles must be >= 2")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")


if __name__ == "__main__":
    main()
