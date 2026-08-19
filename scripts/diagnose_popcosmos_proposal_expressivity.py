#!/usr/bin/env python3
"""Diagnose proposal support and test independent-flow expressivity on SMC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from euclid_dsps.amortized.config import amortized_config, require_equinox
from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.decoder import model_flux_from_x
from euclid_dsps.amortized.exact_posterior import normalized_importance_weights
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.likelihood import photometric_loglike
from euclid_dsps.amortized.posterior import posterior_log_prob, sample_posterior
from euclid_dsps.amortized.proposal_expressivity import (
    IndependentFlowMixture,
    count_parameters,
    fit_proposal_candidate,
    independent_mixture_log_prob,
    joint_distribution_metrics,
    sample_independent_mixture,
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

eqx = require_equinox()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--object-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--mixture-mean-offset", type=float, default=0.05)
    parser.add_argument("--proposal-samples", type=int, default=2048)
    parser.add_argument("--decoder-sample-chunk-size", type=int, default=1)
    parser.add_argument("--geometry-draws", type=int, default=256)
    parser.add_argument("--geometry-projections", type=int, default=64)
    parser.add_argument("--min-median-ess-fraction", type=float, default=0.05)
    parser.add_argument("--max-fraction-pareto-k-gt-0p7", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=260819)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (
        args.config,
        args.dataset,
        args.checkpoint,
        args.feature_stats,
        args.panel,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.out}")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"Expected GPU backend, got {jax.default_backend()}")
    if args.proposal_samples < 64 or args.decoder_sample_chunk_size <= 0:
        raise ValueError("proposal_samples must be >=64 and decoder chunk positive")
    args.out.mkdir(parents=True)
    (args.out / "checkpoints").mkdir()

    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    latent_spec = _latent_spec_for_amortized_config(config)
    model = load_checkpoint(args.checkpoint, config)
    panel = pd.read_parquet(args.panel).sort_values("row_index").reset_index(drop=True)
    required_panel = {"row_index", "panel_role", "expressivity_split", "pareto_k"}
    missing = sorted(required_panel - set(panel.columns))
    if missing:
        raise ValueError(f"panel missing columns: {missing}")

    row_indices, particles, weights = _load_banks(args.bank, latent_spec.names)
    if not np.array_equal(row_indices, panel["row_index"].to_numpy(dtype=np.int64)):
        panel = panel.set_index("row_index").loc[row_indices].reset_index()
    batch = _load_batch(
        config,
        args.feature_stats,
        row_indices=row_indices,
    )
    train_indices = np.flatnonzero(panel["expressivity_split"].to_numpy() == "train")
    validation_indices = np.flatnonzero(
        panel["expressivity_split"].to_numpy() == "validation"
    )

    independent = IndependentFlowMixture(
        jax.random.PRNGKey(args.seed + 10),
        model.encoder,
        n_components=args.experts,
        mean_offset=args.mixture_mean_offset,
    )
    print(
        "[proposal-expressivity] "
        f"objects={len(row_indices)} train={len(train_indices)} "
        f"validation={len(validation_indices)} particles={particles.shape[0]} "
        f"proposal_samples={args.proposal_samples}",
        flush=True,
    )
    single_fit = fit_proposal_candidate(
        model,
        model.encoder,
        features=batch.features,
        particles=particles,
        weights=weights,
        train_indices=train_indices,
        validation_indices=validation_indices,
        epochs=args.epochs,
        object_batch_size=args.object_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        progress_label="single_flow_refit",
    )
    mixture_fit = fit_proposal_candidate(
        model,
        independent,
        features=batch.features,
        particles=particles,
        weights=weights,
        train_indices=train_indices,
        validation_indices=validation_indices,
        epochs=args.epochs,
        object_batch_size=args.object_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        progress_label=f"independent_flow_mix{args.experts}",
    )
    single_model = eqx.tree_at(lambda item: item.encoder, model, single_fit.candidate)
    eqx.tree_serialise_leaves(
        args.out / "checkpoints/single_flow_refit.eqx", single_model
    )
    eqx.tree_serialise_leaves(
        args.out / f"checkpoints/independent_flow_mix{args.experts}.eqx",
        mixture_fit.candidate,
    )
    pd.DataFrame(single_fit.history).to_csv(
        args.out / "single_flow_history.csv", index=False
    )
    pd.DataFrame(mixture_fit.history).to_csv(
        args.out / "independent_flow_history.csv", index=False
    )

    keys = jax.random.split(jax.random.PRNGKey(args.seed + 100), 3)
    current_sample = sample_posterior(
        model, keys[0], batch.features, args.proposal_samples
    )
    single_sample = sample_posterior(
        single_model, keys[1], batch.features, args.proposal_samples
    )
    mixture_sample = sample_independent_mixture(
        model,
        mixture_fit.candidate,
        keys[2],
        batch.features,
        args.proposal_samples,
    )
    candidates = {
        "current_single_flow": (
            np.asarray(current_sample.x),
            np.asarray(current_sample.logq),
        ),
        "single_flow_refit": (
            np.asarray(single_sample.x),
            np.asarray(single_sample.logq),
        ),
        f"independent_flow_mix{args.experts}": (
            np.asarray(mixture_sample.x),
            np.asarray(mixture_sample.logq),
        ),
    }
    target_evaluator = _target_evaluator(config, model, latent_spec)
    object_frames = []
    support_summaries = {}
    for candidate_index, (name, (proposal_x, logq)) in enumerate(candidates.items()):
        print(f"[proposal-expressivity] exact target evaluation: {name}", flush=True)
        logprior, loglike = target_evaluator(
            proposal_x,
            batch,
            sample_chunk_size=args.decoder_sample_chunk_size,
            progress_label=name,
        )
        importance = _importance_frame(
            row_indices,
            logprior=logprior,
            loglike=loglike,
            logq=logq,
        )
        target_logq = _logq_on_target(
            name,
            model,
            single_model,
            mixture_fit.candidate,
            batch.features,
            particles,
        )
        weighted_nll = -np.sum(weights * target_logq, axis=0)
        geometry_rows = []
        for object_index, row_index in enumerate(row_indices):
            geometry = joint_distribution_metrics(
                particles[:, object_index],
                weights[:, object_index],
                proposal_x[:, object_index],
                seed=args.seed + 1000 * candidate_index + object_index,
                max_draws=args.geometry_draws,
                n_projections=args.geometry_projections,
            )
            geometry_rows.append({"row_index": int(row_index), **geometry})
        frame = importance.merge(pd.DataFrame(geometry_rows), on="row_index")
        frame["weighted_smc_nll"] = weighted_nll
        frame["candidate"] = name
        frame = frame.merge(
            panel[["row_index", "panel_role", "expressivity_split", "pareto_k"]].rename(
                columns={"pareto_k": "selection_pareto_k"}
            ),
            on="row_index",
            validate="one_to_one",
        )
        object_frames.append(frame)
        support_summaries[name] = _support_summary(
            frame,
            min_ess=args.min_median_ess_fraction,
            max_bad_k=args.max_fraction_pareto_k_gt_0p7,
        )
    objects = pd.concat(object_frames, ignore_index=True)
    objects.to_parquet(args.out / "objectwise_diagnostics.parquet", index=False)
    objects.to_csv(args.out / "objectwise_diagnostics.csv", index=False)

    current = objects.loc[objects["candidate"] == "current_single_flow"]
    diagnosis = _support_diagnosis(current)
    single_name = "single_flow_refit"
    mixture_name = f"independent_flow_mix{args.experts}"
    evidence = _expressivity_evidence(
        objects,
        single_name=single_name,
        mixture_name=mixture_name,
        single_fit=single_fit,
        mixture_fit=mixture_fit,
        support_summaries=support_summaries,
    )
    summary = {
        "status": "complete",
        "experiment_contract": (
            "same updated prior, floor-0.05 likelihood, SMC bank, object split, "
            "optimizer budget and K; only conditional proposal family differs"
        ),
        "cohort_contract": (
            "selected diagnostic panel; evidence can select an architecture but "
            "cannot replace confirmation on a new untouched cohort"
        ),
        "n_objects": int(len(row_indices)),
        "n_train": int(len(train_indices)),
        "n_validation": int(len(validation_indices)),
        "smc_particles_per_object": int(particles.shape[0]),
        "proposal_samples_per_object": int(args.proposal_samples),
        "parameter_counts": {
            "single_flow": count_parameters(model.encoder),
            mixture_name: count_parameters(mixture_fit.candidate),
        },
        "fits": {
            single_name: _fit_summary(single_fit),
            mixture_name: _fit_summary(mixture_fit),
        },
        "ordinary_is": support_summaries,
        "support_diagnosis": diagnosis,
        "expressivity_evidence": evidence,
        "inputs": {
            "config": _receipt(args.config),
            "dataset": _receipt(args.dataset),
            "checkpoint": _receipt(args.checkpoint),
            "feature_stats": _receipt(args.feature_stats),
            "panel": _receipt(args.panel),
            "banks": [_directory_receipt(path) for path in args.bank],
        },
        "artifacts": {
            "single_flow_checkpoint": _receipt(
                args.out / "checkpoints/single_flow_refit.eqx"
            ),
            "independent_mixture_checkpoint": _receipt(
                args.out / f"checkpoints/independent_flow_mix{args.experts}.eqx"
            ),
            "objectwise_diagnostics": _receipt(
                args.out / "objectwise_diagnostics.parquet"
            ),
        },
    }
    (args.out / "support_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "expressivity_evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "proposal_expressivity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out / "DONE").touch()
    print(
        "[proposal-expressivity] "
        f"diagnosis={diagnosis['evidence_status']} "
        f"expressivity={evidence['evidence_status']} "
        f"next={evidence['next_action']} -> {args.out}",
        flush=True,
    )


def _load_batch(config, feature_stats_path, *, row_indices):
    arrays = load_photometry_arrays_from_config(
        config, batch_size=10_000, row_indices=row_indices
    )
    if arrays.row_index is None:
        raise ValueError("selected catalog does not expose stable row indices")
    position = {int(value): index for index, value in enumerate(arrays.row_index)}
    order = np.asarray([position[int(value)] for value in row_indices], dtype=int)
    batch = next(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=len(row_indices),
            feature_stats=read_feature_stats(feature_stats_path),
            order=order,
        )
    )
    if not np.array_equal(np.asarray(batch.row_index), row_indices):
        raise RuntimeError("photometry rows do not match SMC rows")
    return batch


def _load_banks(paths, parameter_names):
    all_particles = []
    all_weights = []
    expected_rows = None
    for root in paths:
        root = Path(root)
        files = _particle_files(root)
        if not files or not (root / "DONE").is_file():
            raise FileNotFoundError(f"incomplete SMC bank: {root}")
        frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        frame = frame.sort_values(["row_index", "sample_id"]).reset_index(drop=True)
        counts = frame.groupby("row_index", sort=True).size()
        if counts.nunique() != 1:
            raise ValueError(f"unequal SMC particles per object: {root}")
        rows = counts.index.to_numpy(dtype=np.int64)
        if expected_rows is None:
            expected_rows = rows
        elif not np.array_equal(expected_rows, rows):
            raise ValueError("SMC bank row cohorts differ")
        n_objects = len(rows)
        n_samples = int(counts.iloc[0])
        columns = [f"latent_x_{name}" for name in parameter_names]
        x = frame[columns].to_numpy(np.float32).reshape(n_objects, n_samples, -1)
        weight = frame["smc_weight"].to_numpy(np.float32).reshape(n_objects, n_samples)
        weight /= np.sum(weight, axis=1, keepdims=True)
        all_particles.append(np.swapaxes(x, 0, 1))
        all_weights.append(np.swapaxes(weight, 0, 1) / len(paths))
    return (
        expected_rows,
        np.concatenate(all_particles, axis=0),
        np.concatenate(all_weights, axis=0),
    )


def _target_evaluator(config, model, latent_spec):
    cfg = amortized_config(config)
    likelihood = cfg["likelihood"]
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

    @jax.jit
    def components(values, observed_flux, observed_flux_err, mask):
        raw_flux = model_flux_from_x(
            values, latent_spec, context, model_args, latent_spec.names
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
        return model.prior.log_prob(values), loglike

    def evaluate(values, batch, *, sample_chunk_size, progress_label):
        priors = []
        likes = []
        for start in range(0, len(values), int(sample_chunk_size)):
            prior, like = components(
                jnp.asarray(values[start : start + int(sample_chunk_size)]),
                batch.flux,
                batch.flux_err,
                batch.mask,
            )
            priors.append(np.asarray(jax.device_get(prior), dtype=np.float64))
            likes.append(np.asarray(jax.device_get(like), dtype=np.float64))
            completed = min(start + int(sample_chunk_size), len(values))
            if completed == len(values) or completed % 128 == 0:
                print(
                    "[proposal-expressivity] "
                    f"target={progress_label} samples={completed}/{len(values)}",
                    flush=True,
                )
        return np.concatenate(priors), np.concatenate(likes)

    return evaluate


def _importance_frame(row_indices, *, logprior, loglike, logq):
    rows = []
    for index, row_index in enumerate(row_indices):
        result = normalized_importance_weights(
            logprior[:, index] + loglike[:, index], logq[:, index]
        )
        rows.append(
            {
                "row_index": int(row_index),
                "raw_ess_fraction": float(result["raw_ess_fraction"]),
                "psis_ess_fraction": float(result["psis_ess"] / len(logq)),
                "pareto_k": float(result["pareto_k"]),
                "max_raw_weight": float(np.max(result["weight"])),
            }
        )
    return pd.DataFrame(rows)


def _logq_on_target(name, model, single_model, mixture, features, particles):
    if name == "current_single_flow":

        def evaluate_logq(x):
            return posterior_log_prob(model, features, x)
    elif name == "single_flow_refit":

        def evaluate_logq(x):
            return posterior_log_prob(single_model, features, x)
    else:

        def evaluate_logq(x):
            return independent_mixture_log_prob(model, mixture, features, x)

    values = []
    for start in range(0, len(particles), 128):
        values.append(
            np.asarray(
                jax.device_get(
                    evaluate_logq(jnp.asarray(particles[start : start + 128]))
                )
            )
        )
    return np.concatenate(values)


def _support_summary(frame, *, min_ess, max_bad_k):
    result = {}
    for split_name, selected in (
        ("all", frame),
        ("train", frame.loc[frame["expressivity_split"] == "train"]),
        ("validation", frame.loc[frame["expressivity_split"] == "validation"]),
    ):
        median_ess = float(selected["raw_ess_fraction"].median())
        bad_k = float(np.mean(selected["pareto_k"] > 0.7))
        result[split_name] = {
            "n_objects": int(len(selected)),
            "median_raw_ess_fraction": median_ess,
            "median_psis_ess_fraction": float(selected["psis_ess_fraction"].median()),
            "fraction_pareto_k_gt_0p7": bad_k,
            "fraction_pareto_k_gt_1": float(np.mean(selected["pareto_k"] > 1.0)),
            "support_status": (
                "PASS" if median_ess >= min_ess and bad_k <= max_bad_k else "FAIL"
            ),
        }
    return result


def _support_diagnosis(current):
    aggregate = _group_metrics(current, "panel_role")
    worst = aggregate["worst_support"]
    control = aggregate["healthy_control"]
    correlations = {
        metric: _finite_spearman(current["pareto_k"], current[metric])
        for metric in (
            "sliced_wasserstein",
            "energy_distance",
            "nearest_cover_ratio",
            "covariance_relative_frobenius",
        )
    }
    checks = {
        "selected_worst_reproduces_failed_is": bool(
            worst["median_raw_ess_fraction"] < 0.05
            and worst["fraction_pareto_k_gt_0p7"] > 0.2
        ),
        "worst_has_larger_sliced_wasserstein": bool(
            worst["median_sliced_wasserstein"]
            > 1.1 * control["median_sliced_wasserstein"]
        ),
        "worst_has_larger_nearest_cover_gap": bool(
            worst["median_nearest_cover_ratio"]
            > 1.1 * control["median_nearest_cover_ratio"]
        ),
        "pareto_k_tracks_joint_geometry": bool(max(correlations.values()) >= 0.2),
    }
    geometry_votes = sum(
        checks[key]
        for key in (
            "worst_has_larger_sliced_wasserstein",
            "worst_has_larger_nearest_cover_gap",
            "pareto_k_tracks_joint_geometry",
        )
    )
    status = (
        "MISSING_SUPPORT_EVIDENCE"
        if checks["selected_worst_reproduces_failed_is"] and geometry_votes >= 2
        else "INCONCLUSIVE"
    )
    return {
        "evidence_status": status,
        "claim_scope": (
            "tests whether poor IS coincides with missing joint target coverage; "
            "it does not by itself identify architecture as the unique cause"
        ),
        "checks": checks,
        "groups": aggregate,
        "spearman_pareto_k_vs_geometry": correlations,
    }


def _expressivity_evidence(
    objects,
    *,
    single_name,
    mixture_name,
    single_fit,
    mixture_fit,
    support_summaries,
):
    validation = objects.loc[objects["expressivity_split"] == "validation"]
    aggregate = _group_metrics(validation, "candidate")
    single = aggregate[single_name]
    mixture = aggregate[mixture_name]
    checks = {
        "mixture_fits_training_smc_better": bool(
            mixture_fit.final_train_nll <= single_fit.final_train_nll - 0.05
        ),
        "mixture_validation_nll_better": bool(
            mixture_fit.best_validation_nll <= single_fit.best_validation_nll - 0.02
        ),
        "mixture_validation_joint_geometry_better": bool(
            mixture["median_sliced_wasserstein"]
            <= 0.9 * single["median_sliced_wasserstein"]
            and mixture["median_nearest_cover_ratio"]
            <= 0.9 * single["median_nearest_cover_ratio"]
        ),
        "mixture_validation_is_better": bool(
            mixture["median_raw_ess_fraction"]
            >= 1.2 * single["median_raw_ess_fraction"]
            and mixture["fraction_pareto_k_gt_0p7"]
            <= single["fraction_pareto_k_gt_0p7"] - 0.05
        ),
    }
    supported = bool(
        checks["mixture_fits_training_smc_better"]
        and checks["mixture_validation_nll_better"]
        and (
            checks["mixture_validation_joint_geometry_better"]
            or checks["mixture_validation_is_better"]
        )
    )
    mixture_support = support_summaries[mixture_name]["validation"]["support_status"]
    if supported and mixture_support == "PASS":
        next_action = "CONFIRM_MIXTURE_ON_NEW_UNTOUCHED_COHORT"
    elif supported:
        next_action = "EXPRESSIVITY_HELPS_BUT_SUPPORT_STILL_FAILS"
    else:
        next_action = "EXPRESSIVITY_NOT_PROVEN_DIAGNOSE_FEATURES_OR_OPTIMIZATION"
    return {
        "evidence_status": (
            "EXPRESSIVITY_BOTTLENECK_SUPPORTED" if supported else "NOT_PROVEN"
        ),
        "claim_scope": (
            "controlled evidence on a selected diagnostic panel; production fast "
            "posterior requires ordinary-IS PASS on a new untouched cohort"
        ),
        "checks": checks,
        "validation_aggregates": aggregate,
        "validation_ordinary_is_status": mixture_support,
        "next_action": next_action,
    }


def _group_metrics(frame, column):
    result = {}
    for name, group in frame.groupby(column, sort=False):
        result[str(name)] = {
            "n_objects": int(len(group)),
            "median_raw_ess_fraction": float(group["raw_ess_fraction"].median()),
            "fraction_pareto_k_gt_0p7": float(np.mean(group["pareto_k"] > 0.7)),
            "median_weighted_smc_nll": float(group["weighted_smc_nll"].median()),
            "median_marginal_wasserstein": float(
                group["marginal_wasserstein"].median()
            ),
            "median_sliced_wasserstein": float(group["sliced_wasserstein"].median()),
            "median_energy_distance": float(group["energy_distance"].median()),
            "median_covariance_relative_frobenius": float(
                group["covariance_relative_frobenius"].median()
            ),
            "median_nearest_cover_ratio": float(group["nearest_cover_ratio"].median()),
        }
    return result


def _fit_summary(result):
    return {
        "initial_train_weighted_nll": result.initial_train_nll,
        "selected_train_weighted_nll": result.final_train_nll,
        "initial_validation_weighted_nll": result.initial_validation_nll,
        "best_validation_weighted_nll": result.best_validation_nll,
        "best_epoch": result.best_epoch,
    }


def _finite_spearman(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return 0.0
    value = float(spearmanr(left[finite], right[finite]).statistic)
    return value if np.isfinite(value) else 0.0


def _particle_files(path):
    path = Path(path)
    direct = sorted((path / "weighted_particles").glob("batch_*.parquet"))
    return direct or sorted(path.glob("shard_*/weighted_particles/batch_*.parquet"))


def _receipt(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}


def _directory_receipt(path):
    path = Path(path)
    return {
        "path": str(path),
        "done": (path / "DONE").is_file(),
        "particle_files": [_receipt(item) for item in _particle_files(path)],
    }


if __name__ == "__main__":
    main()
