"""No-truth report artifacts for a frozen SC-ASMC-EM run."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.io import ensure_dir

from .decoder import model_flux_from_x
from .latent import x_to_theta
from .posterior import sample_posterior
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    OBSERVED_SELECTION_CONTRACT,
    POSTERIOR_METHOD_CODES,
    TARGET_POPULATION_CONTRACT,
    PosteriorBankShard,
    iter_posterior_bank_shards,
    sha256_file,
    validate_posterior_bank_manifest_provenance,
)
from .sc_asmc_config import sc_asmc_em_config_hash, validate_sc_asmc_em_config
from .sc_asmc_training import (
    RuntimeBundle,
    load_sc_model,
    validate_component_checkpoint,
)
from .train import _selection_log_beta_from_prior_samples, save_checkpoint


def generate_sc_asmc_report(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    *,
    run_root: str | Path,
    out_dir: str | Path,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate the frozen, observation-only population and posterior report."""
    validate_sc_asmc_em_config(config)
    root = Path(run_root)
    output = ensure_dir(out_dir)
    receipt_path = output / "report_receipt.json"
    paths = _frozen_component_paths(root)
    feature_stats_path = root / "runtime" / "feature_stats.json"
    for name in ("p0", "p1", "p2", "q0", "q_final"):
        validate_component_checkpoint(paths[name], sha256_file(paths[name]), runtime)
    input_hashes = {
        name: sha256_file(path)
        for name, path in {
            **paths,
            "run_manifest": root / "manifest" / "run_manifest.json",
            "feature_stats": feature_stats_path,
        }.items()
    }
    manifest = json.loads(
        (root / "manifest" / "run_manifest.json").read_text(encoding="utf-8")
    )
    for bank_name, q_name, prior_name in (
        ("bank_em1", "q0", "p0"),
        ("bank_final", "q_final", "p2"),
    ):
        bank_manifest = json.loads(paths[bank_name].read_text(encoding="utf-8"))
        validate_posterior_bank_manifest_provenance(
            bank_manifest,
            expected_fields={
                "dataset_hash": manifest["dataset"]["sha256"],
                "workflow_config_hash": manifest["config_sha256"],
                "q_ema_hash": input_hashes[q_name],
                "prior_checkpoint_hash": input_hashes[prior_name],
            },
        )
    if receipt_path.is_file() and resume:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("input_hashes") != input_hashes:
            raise ValueError("report resume inputs changed")
        _validate_artifact_records(receipt["artifacts"])
        return receipt
    report_config = dict(
        ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
            "report", {}
        )
        or {}
    )
    seed = int((config.get("amortized", {}) or {}).get("training", {}).get("seed", 0))
    prior_count = int(report_config.get("prior_samples", 4096))
    selected_count = int(report_config.get("selected_resamples", prior_count))
    individual_count = int(report_config.get("individual_objects", 32))
    q_samples = int(report_config.get("q_samples_per_object", 128))
    predictive_samples = int(report_config.get("predictive_samples_per_object", 16))
    decoder_batch_size = int(report_config.get("decoder_batch_size", 128))
    if (
        min(
            prior_count, selected_count, individual_count, q_samples, predictive_samples
        )
        <= 0
    ):
        raise ValueError("SC-ASMC-EM report sample counts must be positive")

    frozen_model = _write_frozen_final_model(
        config,
        runtime,
        paths=paths,
        output=output / "frozen",
        feature_stats_path=feature_stats_path,
    )

    population_dir = ensure_dir(output / "population")
    q_reference = paths["q_final"]
    population: dict[str, dict[str, Any]] = {}
    population_arrays: dict[str, dict[str, np.ndarray]] = {}
    for index, label in enumerate(("p0", "p1", "p2")):
        model = load_sc_model(
            config,
            runtime,
            q_checkpoint=q_reference,
            prior_checkpoint=paths[label],
        )
        arrays, summary = _prior_report_arrays(
            model,
            runtime,
            key=jax.random.fold_in(jax.random.PRNGKey(seed), 100 + index),
            n_samples=prior_count,
            selected_resamples=selected_count,
            decoder_batch_size=decoder_batch_size,
        )
        artifact = population_dir / f"prior_{label}_and_selected.npz"
        _atomic_npz(artifact, arrays)
        summary["artifact"] = str(artifact.resolve())
        summary["sha256"] = sha256_file(artifact)
        population[label] = summary
        population_arrays[label] = arrays
        if verbose:
            print(
                f"[sc-asmc][report] {label}: alpha={summary['alpha']:.6f} "
                f"MC_relerr={summary['alpha_mc_relative_error']:.4f}",
                flush=True,
            )
    marginal_path = population_dir / "population_marginals.csv"
    correlation_path = population_dir / "population_correlations.npz"
    _write_population_marginals(
        marginal_path,
        population_arrays,
        runtime.parameter_names,
    )
    _atomic_npz(
        correlation_path,
        {
            f"{label}_{kind}": _weighted_correlation(
                values["theta"],
                None if kind == "parent" else values["beta"],
            )
            for label, values in population_arrays.items()
            for kind in ("parent", "selected")
        },
    )
    marginal_plot, correlation_plot = _write_population_plots(
        population_dir,
        population_arrays,
        runtime.parameter_names,
    )

    report_rows = np.sort(
        np.load(manifest["artifacts"]["preflight_rows"]["path"], allow_pickle=False)[
            :individual_count
        ]
    )
    (
        individual_path,
        predictive_path,
        predictive_summary_path,
        predictive_plot,
    ) = _write_individual_and_predictive_artifacts(
        config,
        runtime,
        paths=paths,
        report_rows=report_rows,
        q_samples=q_samples,
        predictive_samples=predictive_samples,
        decoder_batch_size=decoder_batch_size,
        output=output,
        seed=seed,
    )
    method_summary = summarize_posterior_bank(
        root / "banks" / "em2_p2" / "posterior_bank_manifest.json"
    )
    method_summary["e_step_runtime"] = _e_step_runtime_summary(root)
    method_path = output / "method_runtime_diagnostics.json"
    _atomic_json(method_path, method_summary)
    method_plot = _write_method_plot(output, method_summary)
    selection_path = output / "alpha_score_diagnostics.json"
    _atomic_json(
        selection_path,
        {
            "prior_alpha": population,
            "mstep1": _read_json(root / "mstep1" / "prior_mstep_1_receipt.json")[
                "final_diagnostics"
            ],
            "mstep2": _read_json(root / "mstep2" / "prior_mstep_2_receipt.json")[
                "final_diagnostics"
            ],
            "score_gradient_contract": (
                "sum_k (normalized_beta_k - 1/M) grad log p_eta(x_k)"
            ),
        },
    )

    artifacts = _artifact_records(
        {
            "prior_p0": Path(population["p0"]["artifact"]),
            "prior_p1": Path(population["p1"]["artifact"]),
            "prior_p2": Path(population["p2"]["artifact"]),
            "population_marginals": marginal_path,
            "population_correlations": correlation_path,
            "population_marginals_plot": marginal_plot,
            "population_correlations_plot": correlation_plot,
            "individual_posterior_comparisons": individual_path,
            "posterior_predictive_photometry": predictive_path,
            "posterior_predictive_summary": predictive_summary_path,
            "posterior_predictive_plot": predictive_plot,
            "method_runtime_diagnostics": method_path,
            "method_fractions_plot": method_plot,
            "alpha_score_diagnostics": selection_path,
            "final_model_checkpoint": Path(frozen_model["checkpoint"]["path"]),
            "final_model_sidecar": Path(frozen_model["sidecar"]["path"]),
        }
    )
    report_path = output / "SC_ASMC_EM_REPORT.md"
    _atomic_text(
        report_path,
        _report_markdown(
            manifest=manifest,
            population=population,
            bank_summary=method_summary,
            artifacts=artifacts,
        ),
    )
    artifacts["report"] = _artifact_record(report_path)
    receipt = {
        "status": "PASS",
        "phase": "frozen_no_truth_report",
        "report_code_commit": _git_commit(),
        "training_code_commit": manifest["code"]["commit"],
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": TARGET_POPULATION_CONTRACT,
        "observed_selection": OBSERVED_SELECTION_CONTRACT,
        "scope_limit": "No inference claim is made outside C0.",
        "truth_used": False,
        "truth_columns_requested": [],
        "checkpoints_frozen_before_reporting": True,
        "input_hashes": input_hashes,
        "frozen_model": frozen_model,
        "prior_versions": population,
        "report_rows": report_rows.tolist(),
        "report_row_selection": "first rows of the observed-only stratified preflight",
        "artifacts": artifacts,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_frozen_final_model(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    *,
    paths: dict[str, Path],
    output: Path,
    feature_stats_path: Path,
) -> dict[str, Any]:
    """Serialize the unambiguous post-EM model used by post-freeze tools."""
    model = load_sc_model(
        config,
        runtime,
        q_checkpoint=paths["q_final"],
        prior_checkpoint=paths["p2"],
    )
    checkpoint = output / "sc_asmc_em_final.eqx"
    distill = _read_json(
        paths["q_final"].with_suffix(paths["q_final"].suffix + ".json")
    )
    metric = float(
        distill.get("selection_score", distill.get("heldout_cross_entropy", 0.0))
    )
    save_checkpoint(
        checkpoint,
        model,
        config=config,
        latent_spec=runtime.latent_spec,
        feature_stats=runtime.feature_stats,
        epoch=2,
        metric=metric,
        metric_name="frozen_q1_ema_p2",
    )
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    sidecar = _read_json(sidecar_path)
    sidecar.update(
        {
            "status": "frozen",
            "phase": "post_em2_model_freeze",
            "model_components": "q1_ema + p2",
            "q_checkpoint": str(paths["q_final"].resolve()),
            "q_checkpoint_sha256": sha256_file(paths["q_final"]),
            "prior_checkpoint": str(paths["p2"].resolve()),
            "prior_checkpoint_sha256": sha256_file(paths["p2"]),
            "final_bank_manifest": str(paths["bank_final"].resolve()),
            "final_bank_manifest_sha256": sha256_file(paths["bank_final"]),
            "feature_stats_path": str(feature_stats_path.resolve()),
            "feature_stats_sha256": sha256_file(feature_stats_path),
            "workflow_config_hash": sc_asmc_em_config_hash(config),
            "c0_scope_statement": C0_SCOPE_STATEMENT,
            "target_population": TARGET_POPULATION_CONTRACT,
            "observed_selection": OBSERVED_SELECTION_CONTRACT,
            "truth_used": False,
            "truth_columns_requested": [],
            "training_complete": True,
            "nuts_in_training": False,
        }
    )
    _atomic_json(sidecar_path, sidecar)
    return {
        "model_components": "q1_ema + p2",
        "checkpoint": _artifact_record(checkpoint),
        "sidecar": _artifact_record(sidecar_path),
        "truth_used": False,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
    }


def summarize_posterior_bank(manifest_path: str | Path) -> dict[str, Any]:
    """Stream a bank and summarize hierarchy, movement, and DSPS cost."""
    manifest = _read_json(manifest_path)
    values: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "method",
            "resolved",
            "stage_count",
            "acceptance",
            "ancestor_ess",
            "unique_ancestor_fraction",
            "movement_squared",
            "moved_particle_fraction",
            "dsps_evaluations",
            "ess",
            "particle_count",
            "max_weight",
            "beta_final",
        )
    }
    for shard in iter_posterior_bank_shards(manifest_path):
        for name in values:
            values[name].append(np.asarray(getattr(shard, name)))
    joined = {name: np.concatenate(chunks) for name, chunks in values.items()}
    method = joined["method"]
    count = len(method)
    method_counts = {
        name: int(np.sum(method == code))
        for name, code in POSTERIOR_METHOD_CODES.items()
    }
    return {
        "objects": int(count),
        "manifest_objects": int(manifest["object_count"]),
        "resolved_fraction": float(np.mean(joined["resolved"])),
        "unresolved_fraction": float(np.mean(~joined["resolved"].astype(bool))),
        "method_counts": method_counts,
        "method_fractions": {
            name: value / float(count) for name, value in method_counts.items()
        },
        "stage_count": _finite_summary(joined["stage_count"]),
        "mutation_acceptance": _finite_summary(joined["acceptance"]),
        "ancestry_ess": _finite_summary(joined["ancestor_ess"]),
        "unique_ancestor_fraction": _finite_summary(joined["unique_ancestor_fraction"]),
        "movement_squared": _finite_summary(joined["movement_squared"]),
        "moved_particle_fraction": _finite_summary(joined["moved_particle_fraction"]),
        "posterior_ess_fraction": _finite_summary(
            joined["ess"] / joined["particle_count"]
        ),
        "maximum_weight": _finite_summary(joined["max_weight"]),
        "beta_final": _finite_summary(joined["beta_final"]),
        "dsps_evaluations_per_object": _finite_summary(joined["dsps_evaluations"]),
        "total_dsps_evaluations": int(np.sum(joined["dsps_evaluations"])),
    }


def _prior_report_arrays(
    model: Any,
    runtime: RuntimeBundle,
    *,
    key: jax.Array,
    n_samples: int,
    selected_resamples: int,
    decoder_batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_key, resample_key = jax.random.split(key)
    x = np.asarray(jax.device_get(model.prior.sample(sample_key, int(n_samples))))
    theta = np.asarray(jax.device_get(x_to_theta(jnp.asarray(x), runtime.latent_spec)))
    logprior = np.asarray(jax.device_get(model.prior.log_prob(jnp.asarray(x))))
    selection = runtime.selection_objective_config["selection_correction"]
    log_beta_chunks = []
    for start in range(0, len(x), int(decoder_batch_size)):
        log_beta_chunks.append(
            np.asarray(
                jax.device_get(
                    _selection_log_beta_from_prior_samples(
                        model,
                        jnp.asarray(x[start : start + int(decoder_batch_size)]),
                        runtime.jit_latent_spec,
                        runtime.context,
                        runtime.model_args,
                        runtime.parameter_names,
                        runtime.calibration_config,
                        selection,
                    )
                )
            )
        )
    log_beta = np.concatenate(log_beta_chunks)
    beta = np.exp(log_beta)
    alpha = float(np.mean(beta))
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise RuntimeError("prior report has zero or non-finite selection probability")
    error = float(np.std(beta, ddof=1) / np.sqrt(len(beta))) if len(beta) > 1 else 0.0
    normalized = beta / np.sum(beta)
    rng = np.random.default_rng(
        int(np.asarray(jax.random.randint(resample_key, (), 0, 2**31 - 1)))
    )
    selected_indices = rng.choice(
        len(x), size=int(selected_resamples), replace=True, p=normalized
    )
    arrays = {
        "x": x.astype(np.float32),
        "theta": theta.astype(np.float32),
        "logprior": logprior.astype(np.float64),
        "log_beta": log_beta.astype(np.float64),
        "beta": beta.astype(np.float64),
        "selected_weights": normalized.astype(np.float64),
        "selected_indices": selected_indices.astype(np.int64),
        "selected_x": x[selected_indices].astype(np.float32),
        "selected_theta": theta[selected_indices].astype(np.float32),
    }
    summary = {
        "samples": int(len(x)),
        "selected_resamples": int(selected_resamples),
        "alpha": alpha,
        "alpha_mc_error": error,
        "alpha_mc_relative_error": error / alpha,
        "score_weight_ess": float(1.0 / np.sum(np.square(normalized))),
        "maximum_score_weight": float(np.max(normalized)),
        "finite": bool(
            np.all(np.isfinite(x))
            and np.all(np.isfinite(theta))
            and np.all(np.isfinite(logprior))
            and np.all(np.isfinite(beta))
        ),
    }
    if not summary["finite"]:
        raise RuntimeError("prior report contains non-finite values")
    return arrays, summary


def _write_individual_and_predictive_artifacts(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    *,
    paths: dict[str, Path],
    report_rows: np.ndarray,
    q_samples: int,
    predictive_samples: int,
    decoder_batch_size: int,
    output: Path,
    seed: int,
) -> tuple[Path, Path, Path, Path]:
    em1 = _read_bank_rows(paths["bank_em1"], report_rows)
    final = _read_bank_rows(paths["bank_final"], report_rows)
    if em1.features is None or final.features is None:
        raise ValueError("individual report requires inline bank features")
    q0_model = load_sc_model(
        config,
        runtime,
        q_checkpoint=paths["q0"],
        prior_checkpoint=paths["p0"],
    )
    q_final_model = load_sc_model(
        config,
        runtime,
        q_checkpoint=paths["q_final"],
        prior_checkpoint=paths["p2"],
    )
    q0 = sample_posterior(
        q0_model,
        jax.random.fold_in(jax.random.PRNGKey(seed), 301),
        jnp.asarray(final.features),
        int(q_samples),
    )
    q_final = sample_posterior(
        q_final_model,
        jax.random.fold_in(jax.random.PRNGKey(seed), 302),
        jnp.asarray(final.features),
        int(q_samples),
    )
    individual_dir = ensure_dir(output / "individual")
    individual_path = individual_dir / "posterior_comparisons.npz"
    _atomic_npz(
        individual_path,
        {
            "row_index": final.row_index,
            "object_id": final.object_id.astype(str),
            "raw_q0_x": np.asarray(jax.device_get(q0.x), dtype=np.float32),
            "distilled_q1_x": np.asarray(jax.device_get(q_final.x), dtype=np.float32),
            "corrected_estep_em1_x": em1.particles,
            "corrected_estep_em1_weights": em1.normalized_weights,
            "after_em2_x": final.particles,
            "after_em2_weights": final.normalized_weights,
            "after_em2_method": final.method,
            "after_em2_resolved": final.resolved,
        },
    )

    rng = np.random.default_rng(seed + 303)
    x_predictive = _weighted_particle_draws(final, predictive_samples, rng=rng)
    model_flux = _predictive_model_flux_batched(
        x_predictive,
        runtime,
        decoder_batch_size=int(decoder_batch_size),
    )
    observed = _observed_arrays_in_row_order(runtime, report_rows)
    if observed.truth:
        raise RuntimeError("posterior predictive report loaded truth")
    noise = rng.normal(size=model_flux.shape) * observed.flux_err[None, :, :]
    replicated_flux = model_flux + noise
    normalized_residual = (model_flux - observed.flux[None, :, :]) / observed.flux_err[
        None, :, :
    ]
    predictive_dir = ensure_dir(output / "posterior_predictive")
    predictive_path = predictive_dir / "posterior_predictive_photometry.npz"
    _atomic_npz(
        predictive_path,
        {
            "row_index": report_rows,
            "object_id": final.object_id.astype(str),
            "band_names": np.asarray(observed.band_names, dtype=str),
            "observed_flux": observed.flux,
            "observed_flux_error": observed.flux_err,
            "mask": observed.mask,
            "latent_x": x_predictive,
            "model_flux": model_flux,
            "gaussian_replicated_flux": replicated_flux,
            "normalized_model_residual": normalized_residual,
        },
    )
    summary_path = predictive_dir / "posterior_predictive_summary.csv"
    _write_predictive_summary(
        summary_path,
        normalized_residual,
        observed.mask,
        observed.band_names,
    )
    plot_path = _write_predictive_plot(
        predictive_dir, normalized_residual, observed.band_names
    )
    return individual_path, predictive_path, summary_path, plot_path


def _predictive_model_flux_batched(
    x: np.ndarray,
    runtime: RuntimeBundle,
    *,
    decoder_batch_size: int,
) -> np.ndarray:
    """Decode posterior draws while bounding total sample-object pairs."""
    values = np.asarray(x)
    if values.ndim != 3:
        raise ValueError("predictive latent draws must have shape [samples,objects,dim]")
    if int(decoder_batch_size) <= 0:
        raise ValueError("decoder_batch_size must be positive")
    sample_chunks = []
    for sample_start in range(0, values.shape[0], int(decoder_batch_size)):
        sample_stop = min(
            sample_start + int(decoder_batch_size), values.shape[0]
        )
        sample_count = sample_stop - sample_start
        object_chunk_size = max(1, int(decoder_batch_size) // sample_count)
        object_chunks = []
        for object_start in range(0, values.shape[1], object_chunk_size):
            block = values[
                sample_start:sample_stop,
                object_start : object_start + object_chunk_size,
            ]
            object_chunks.append(
                np.asarray(
                    jax.device_get(
                        model_flux_from_x(
                            jnp.asarray(block),
                            runtime.jit_latent_spec,
                            runtime.context,
                            runtime.model_args,
                            runtime.parameter_names,
                        )
                    )
                )
            )
        sample_chunks.append(np.concatenate(object_chunks, axis=1))
    return np.concatenate(sample_chunks, axis=0)


def _frozen_component_paths(root: Path) -> dict[str, Path]:
    initialization = _read_json(root / "initialization" / "initialization_receipt.json")
    sleep = _read_json(root / "sleep" / "sleep_receipt.json")
    active_path = (
        root / "preflight" / "active_bootstrap" / "active_bootstrap_receipt.json"
    )
    active = _read_json(active_path) if active_path.is_file() else None
    distill = _read_json(root / "distill1" / "q_distillation_em1_receipt.json")
    mstep1 = _read_json(root / "mstep1" / "prior_mstep_1_receipt.json")
    mstep2 = _read_json(root / "mstep2" / "prior_mstep_2_receipt.json")
    result = {
        "p0": Path(initialization["prior_p0"]["path"]),
        "p1": Path(mstep1["prior_checkpoint"]),
        "p2": Path(mstep2["prior_checkpoint"]),
        "q0": Path(
            active["q_ema_checkpoint"]
            if active is not None
            else sleep["q_ema_checkpoint"]
        ),
        "q_final": Path(distill["q_ema_checkpoint"]),
        "bank_em1": root / "banks" / "em1" / "posterior_bank_manifest.json",
        "bank_final": root / "banks" / "em2_p2" / "posterior_bank_manifest.json",
    }
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"frozen report inputs are incomplete: {missing}")
    return result


def _read_bank_rows(manifest_path: Path, rows: np.ndarray) -> PosteriorBankShard:
    desired = np.asarray(rows, dtype=np.int64)
    include = set(desired.tolist())
    pieces: list[PosteriorBankShard] = []
    for shard in iter_posterior_bank_shards(manifest_path):
        indices = np.flatnonzero(
            np.asarray([int(row) in include for row in shard.row_index], dtype=bool)
        )
        if len(indices):
            payload = {}
            for field in fields(PosteriorBankShard):
                value = getattr(shard, field.name)
                if field.name == "feature_reference":
                    payload[field.name] = value
                elif value is None:
                    payload[field.name] = None
                else:
                    payload[field.name] = np.asarray(value)[indices]
            pieces.append(PosteriorBankShard(**payload))
    if not pieces:
        raise ValueError("requested report rows are absent from posterior bank")
    payload = {}
    for field in fields(PosteriorBankShard):
        if field.name == "feature_reference":
            payload[field.name] = pieces[0].feature_reference
        elif getattr(pieces[0], field.name) is None:
            payload[field.name] = None
        else:
            payload[field.name] = np.concatenate(
                [np.asarray(getattr(piece, field.name)) for piece in pieces], axis=0
            )
    combined = PosteriorBankShard(**payload)
    lookup = {int(row): index for index, row in enumerate(combined.row_index)}
    if set(lookup) != include:
        raise ValueError("posterior bank does not cover every requested report row")
    order = np.asarray([lookup[int(row)] for row in desired], dtype=np.int64)
    ordered = {}
    for field in fields(PosteriorBankShard):
        value = getattr(combined, field.name)
        if field.name == "feature_reference" or value is None:
            ordered[field.name] = value
        else:
            ordered[field.name] = np.asarray(value)[order]
    result = PosteriorBankShard(**ordered)
    result.validate()
    return result


def _weighted_particle_draws(
    shard: PosteriorBankShard,
    samples: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    draws = np.empty(
        (int(samples), shard.object_count, shard.latent_dim), dtype=np.float32
    )
    for object_index, count in enumerate(shard.particle_count):
        weights = np.asarray(shard.normalized_weights[object_index, : int(count)])
        indices = rng.choice(int(count), size=int(samples), replace=True, p=weights)
        draws[:, object_index] = shard.particles[object_index, indices]
    return draws


def _observed_arrays_in_row_order(runtime: RuntimeBundle, rows: np.ndarray):
    from .data import load_photometry_arrays_from_config

    arrays = load_photometry_arrays_from_config(
        runtime.config,
        batch_size=10_000,
        row_indices=np.asarray(rows, dtype=np.int64),
    )
    actual = np.asarray(arrays.row_index, dtype=np.int64)
    lookup = {int(row): index for index, row in enumerate(actual)}
    order = np.asarray([lookup[int(row)] for row in rows], dtype=np.int64)
    from dataclasses import replace

    return replace(
        arrays,
        object_id=np.asarray(arrays.object_id)[order],
        row_index=actual[order],
        flux=np.asarray(arrays.flux)[order],
        flux_err=np.asarray(arrays.flux_err)[order],
        mask=np.asarray(arrays.mask)[order],
        truth=None,
    )


def _write_population_marginals(
    path: Path,
    population: dict[str, dict[str, np.ndarray]],
    names: tuple[str, ...],
) -> None:
    rows = []
    for label, arrays in population.items():
        for selected, values in (
            (False, arrays["theta"]),
            (True, arrays["selected_theta"]),
        ):
            for index, name in enumerate(names):
                column = values[:, index]
                rows.append(
                    {
                        "prior": label,
                        "population": "beta_weighted_selected"
                        if selected
                        else "parent_C0",
                        "parameter": name,
                        "mean": float(np.mean(column)),
                        "std": float(np.std(column)),
                        "q05": float(np.quantile(column, 0.05)),
                        "q16": float(np.quantile(column, 0.16)),
                        "q50": float(np.quantile(column, 0.50)),
                        "q84": float(np.quantile(column, 0.84)),
                        "q95": float(np.quantile(column, 0.95)),
                    }
                )
    _atomic_csv(path, rows)


def _weighted_correlation(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("correlation input must be [samples, parameters]")
    if weights is None:
        return np.corrcoef(data, rowvar=False)
    normalized = np.asarray(weights, dtype=np.float64)
    normalized = normalized / np.sum(normalized)
    mean = np.sum(normalized[:, None] * data, axis=0)
    centered = data - mean
    covariance = (centered * normalized[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 1.0e-30))
    return covariance / np.outer(scale, scale)


def _finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": float("nan"), "median": float("nan"), "q95": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def _e_step_runtime_summary(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for iteration in (1, 2):
        receipts = [
            _read_json(path)
            for path in sorted(
                (root / "banks" / f"em{iteration}").glob("worker_*_receipt.json")
            )
        ]
        if not receipts:
            raise FileNotFoundError(f"missing E-step {iteration} worker receipts")
        elapsed = [float(receipt["elapsed_seconds"]) for receipt in receipts]
        objects = sum(int(receipt["rows"]) for receipt in receipts)
        result[f"em{iteration}"] = {
            "workers": len(receipts),
            "objects": objects,
            "sum_worker_seconds": float(np.sum(elapsed)),
            "parallel_wall_seconds_lower_bound": float(np.max(elapsed)),
            "aggregate_objects_per_second": objects
            / max(float(np.max(elapsed)), 1.0e-12),
        }
    return result


def _write_predictive_summary(
    path: Path,
    residual: np.ndarray,
    mask: np.ndarray,
    band_names: tuple[str, ...],
) -> None:
    rows = []
    for index, name in enumerate(band_names):
        values = residual[:, :, index][:, np.asarray(mask[:, index], dtype=bool)]
        rows.append(
            {
                "band": name,
                "median_normalized_residual": float(np.median(values)),
                "rms_normalized_residual": float(np.sqrt(np.mean(values**2))),
                "fraction_abs_lt_1": float(np.mean(np.abs(values) < 1.0)),
                "fraction_abs_lt_3": float(np.mean(np.abs(values) < 3.0)),
                "finite_fraction": float(np.mean(np.isfinite(values))),
            }
        )
    _atomic_csv(path, rows)


def _write_population_plots(
    output: Path,
    population: dict[str, dict[str, np.ndarray]],
    names: tuple[str, ...],
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(names) / 3))
    figure, axes = plt.subplots(rows, 3, figsize=(12, 2.5 * rows), squeeze=False)
    colors = {"p0": "#3b6fb6", "p1": "#d47a24", "p2": "#28846b"}
    for index, name in enumerate(names):
        axis = axes.flat[index]
        for label, arrays in population.items():
            axis.hist(
                arrays["theta"][:, index],
                bins=40,
                density=True,
                histtype="step",
                linewidth=1.1,
                color=colors[label],
                label=label,
            )
            axis.hist(
                arrays["selected_theta"][:, index],
                bins=40,
                density=True,
                histtype="step",
                linestyle="--",
                linewidth=0.9,
                color=colors[label],
            )
        axis.set_title(name, fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in axes.flat[len(names) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7, ncol=3)
    figure.tight_layout()
    marginal_path = output / "population_marginals.png"
    figure.savefig(marginal_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 8), squeeze=False)
    for column, label in enumerate(("p0", "p1", "p2")):
        for row, kind in enumerate(("parent", "selected")):
            weights = None if kind == "parent" else population[label]["beta"]
            correlation = _weighted_correlation(population[label]["theta"], weights)
            image = axes[row, column].imshow(
                correlation, vmin=-1.0, vmax=1.0, cmap="coolwarm"
            )
            axes[row, column].set_title(f"{label} {kind}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="correlation")
    correlation_plot = output / "population_correlations.png"
    figure.savefig(correlation_plot, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return marginal_path, correlation_plot


def _write_predictive_plot(
    output: Path,
    residual: np.ndarray,
    band_names: tuple[str, ...],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.boxplot(
        [residual[:, :, index].ravel() for index in range(len(band_names))],
        showfliers=False,
    )
    axis.axhline(0.0, color="#202020", linewidth=1.0)
    axis.set_xticks(
        np.arange(1, len(band_names) + 1), band_names, rotation=60, ha="right"
    )
    axis.set_ylabel("(model flux - observed flux) / flux error")
    figure.tight_layout()
    path = output / "posterior_predictive_normalized_residuals.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _write_method_plot(output: Path, summary: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(summary["method_fractions"])
    values = [summary["method_fractions"][label] for label in labels]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        labels, values, color=["#3b6fb6", "#28846b", "#d47a24", "#a94848", "#666666"]
    )
    axis.set_ylabel("fraction of selected catalogue")
    axis.set_ylim(0.0, max(1.0, 1.1 * max(values)))
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    path = output / "posterior_method_fractions.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _report_markdown(
    *,
    manifest: dict[str, Any],
    population: dict[str, dict[str, Any]],
    bank_summary: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Selection-Corrected Amortized SMC-EM report",
        "",
        C0_SCOPE_STATEMENT,
        "",
        f"Target population: `{TARGET_POPULATION_CONTRACT}`.",
        f"Explicit selection correction: `{OBSERVED_SELECTION_CONTRACT}`.",
        "Selection enters only `+log(alpha_eta)` in the population-prior loss; it never enters object posterior weights.",
        "No inference claim is made outside C0.",
        "",
        "## No-truth contract",
        "",
        "Training, E-steps, M-steps, preflight, checkpoint selection, and this report requested no truth columns. Truth-aware closure is a separate post-freeze workflow.",
        "",
        "## Population",
        "",
        "| prior | alpha | alpha MC relative error | score-weight ESS | max score weight |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("p0", "p1", "p2"):
        value = population[label]
        lines.append(
            f"| {label} | {value['alpha']:.6g} | {value['alpha_mc_relative_error']:.4g} | "
            f"{value['score_weight_ess']:.2f} | {value['maximum_score_weight']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## E-step hierarchy",
            "",
            f"Resolved fraction: {bank_summary['resolved_fraction']:.4f}.",
            f"Unresolved fraction: {bank_summary['unresolved_fraction']:.4f}.",
            f"Total DSPS evaluations: {bank_summary['total_dsps_evaluations']}.",
            "",
            "Method counts: "
            + ", ".join(
                f"{name}={count}"
                for name, count in bank_summary["method_counts"].items()
            )
            + ".",
            "",
            "## Provenance",
            "",
            f"Dataset SHA256: `{manifest['dataset']['sha256']}`.",
            f"Selected catalogue objects: {manifest['objects']['selected']}.",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend(
        f"- `{name}`: `{record['path']}` (`{record['sha256']}`)"
        for name, record in artifacts.items()
    )
    return "\n".join(lines) + "\n"


def _artifact_records(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: _artifact_record(path) for name, path in paths.items()}


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_artifact_records(records: dict[str, dict[str, Any]]) -> None:
    for name, record in records.items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"SC-ASMC-EM report artifact mismatch: {name}")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty SC-ASMC-EM CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
