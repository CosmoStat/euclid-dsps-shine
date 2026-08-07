"""Post-hoc importance correction and empirical-Bayes prior updates.

The routines in this module operate on joint posterior draws.  They never
replace a posterior by a vector of marginal summaries.  A posterior median is
only produced for the conventional redshift point-estimate metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.config import load_config
from euclid_dsps.io import ensure_dir, write_json

from .config import require_amortized_dependencies
from .exact_posterior import normalized_importance_weights, systematic_resample
from .latent import theta_to_x
from .train import _latent_spec_for_amortized_config, load_checkpoint

eqx, optax = require_amortized_dependencies()


@dataclass(frozen=True)
class PosteriorBank:
    """Dense joint proposal bank and its immutable data contract."""

    frame: pd.DataFrame
    identity_column: str
    parameter_names: tuple[str, ...]
    source_files: tuple[Path, ...]


def posterior_sample_paths(source: str | Path) -> tuple[Path, ...]:
    """Resolve a posterior-sample file, inference directory, or shard directory."""
    source = Path(source)
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(source)
    monolithic = source / "posterior_samples.parquet"
    if monolithic.is_file():
        return (monolithic,)
    shard_dir = source / "posterior_samples"
    if shard_dir.is_dir():
        shards = tuple(sorted(shard_dir.glob("batch_*.parquet")))
        if shards:
            return shards
    shards = tuple(sorted(source.glob("batch_*.parquet")))
    if shards:
        return shards
    raise FileNotFoundError(f"No posterior sample parquet found under {source}")


def load_posterior_bank(
    source: str | Path,
    *,
    parameter_names: Iterable[str] | None = None,
) -> PosteriorBank:
    """Load and validate a long-form joint posterior proposal bank."""
    files = posterior_sample_paths(source)
    frames = [pd.read_parquet(path) for path in files]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    identity_column = "row_index" if "row_index" in frame else "object_id"
    required = {identity_column, "sample_id", "logq", "logprior", "loglike"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Posterior bank is missing required columns: {missing}")
    if parameter_names is None:
        excluded = {
            "object_id",
            "row_index",
            "sample_id",
            "logq",
            "logprior",
            "loglike",
            "log_alpha_sed",
            "alpha_sed",
            "delta_mag_global",
            "log10_stellar_mass_raw",
            "log10_stellar_mass_alpha_corrected",
        }
        parameter_names = tuple(name for name in frame.columns if name not in excluded)
    else:
        parameter_names = tuple(parameter_names)
    missing_parameters = sorted(set(parameter_names) - set(frame.columns))
    if missing_parameters:
        raise ValueError(
            f"Posterior bank is missing latent parameters: {missing_parameters}"
        )
    duplicate = frame.duplicated([identity_column, "sample_id"], keep=False)
    if duplicate.any():
        raise ValueError(
            "Posterior bank has duplicate (identity, sample_id) rows: "
            f"{int(duplicate.sum())}"
        )
    frame = frame.sort_values([identity_column, "sample_id"]).reset_index(drop=True)
    return PosteriorBank(
        frame=frame,
        identity_column=identity_column,
        parameter_names=tuple(parameter_names),
        source_files=files,
    )


def evaluate_checkpoint_logprior(
    frame: pd.DataFrame,
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    parameter_names: tuple[str, ...],
    batch_size: int = 65_536,
) -> np.ndarray:
    """Evaluate an amortized checkpoint's exact prior density on physical draws."""
    config = load_config(config_path)
    latent_spec = _latent_spec_for_amortized_config(config)
    if tuple(latent_spec.names) != tuple(parameter_names):
        raise ValueError(
            "Posterior bank parameter order differs from checkpoint latent spec: "
            f"bank={parameter_names}, checkpoint={tuple(latent_spec.names)}"
        )
    model = load_checkpoint(checkpoint, config)
    log_prob = eqx.filter_jit(model.prior.log_prob)
    theta = frame.loc[:, parameter_names].to_numpy(dtype=np.float32)
    result: list[np.ndarray] = []
    for start in range(0, len(theta), int(batch_size)):
        values = theta_to_x(jnp.asarray(theta[start : start + batch_size]), latent_spec)
        result.append(np.asarray(jax.device_get(log_prob(values)), dtype=np.float64))
    return np.concatenate(result) if result else np.empty(0, dtype=np.float64)


def importance_weight_bank(
    bank: PosteriorBank,
    *,
    target_logprior: np.ndarray | None = None,
    resample_count: int = 128,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute raw/PSIS weights and a joint PSIS-resampled posterior bank."""
    frame = bank.frame.copy()
    source_logprior = pd.to_numeric(frame["logprior"], errors="coerce").to_numpy()
    if target_logprior is None:
        target_logprior = source_logprior.copy()
    target_logprior = np.asarray(target_logprior, dtype=np.float64)
    if target_logprior.shape != (len(frame),):
        raise ValueError(
            f"target_logprior must have shape {(len(frame),)}, got {target_logprior.shape}"
        )
    frame["source_logprior"] = source_logprior
    frame["target_logprior"] = target_logprior
    diagnostics: list[dict[str, Any]] = []
    resampled: list[pd.DataFrame] = []
    for object_number, (identity, group) in enumerate(
        frame.groupby(bank.identity_column, sort=False)
    ):
        index = group.index.to_numpy()
        loglike = pd.to_numeric(group["loglike"], errors="coerce").to_numpy()
        logq = pd.to_numeric(group["logq"], errors="coerce").to_numpy()
        logtarget = loglike + target_logprior[index]
        result = normalized_importance_weights(logtarget, logq)
        raw_weight = np.asarray(result["weight"], dtype=np.float64)
        psis_weight = np.asarray(result["psis_weight"], dtype=np.float64)
        frame.loc[index, "logtarget"] = logtarget
        frame.loc[index, "logweight"] = np.asarray(result["log_weight"])
        frame.loc[index, "raw_weight"] = raw_weight
        frame.loc[index, "psis_weight"] = psis_weight
        chosen = systematic_resample(
            psis_weight,
            int(resample_count),
            seed=int(seed) + int(object_number),
        )
        selected = group.iloc[chosen].copy()
        selected["source_sample_id"] = selected["sample_id"].to_numpy()
        selected["sample_id"] = np.arange(int(resample_count), dtype=np.int64)
        selected["resampling_weight"] = "psis"
        resampled.append(selected)
        finite = np.isfinite(logtarget - logq)
        diagnostics.append(
            {
                bank.identity_column: identity,
                "object_id": (
                    group["object_id"].iloc[0] if "object_id" in group else identity
                ),
                "n_proposal_samples": int(len(group)),
                "n_finite_logweights": int(finite.sum()),
                "raw_ess": float(result["raw_ess"]),
                "raw_ess_fraction": float(result["raw_ess_fraction"]),
                "psis_ess": float(result["psis_ess"]),
                "psis_ess_fraction": float(result["psis_ess"] / len(group)),
                "pareto_k": float(result["pareto_k"]),
                "max_raw_weight": float(np.max(raw_weight)),
                "max_psis_weight": float(np.max(psis_weight)),
                "log_evidence_is": _logmeanexp(
                    np.asarray(result["log_weight"], dtype=np.float64)
                ),
            }
        )
    return (
        frame,
        pd.concat(resampled, ignore_index=True),
        pd.DataFrame(diagnostics),
    )


def weighted_redshift_metrics(
    weighted_samples: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    weight_column: str = "psis_weight",
    z_parameter: str = "z_obs",
    truth_column: str = "redshift_true",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute weighted PIT, coverage, widths, and point photo-z diagnostics."""
    identity = (
        "row_index"
        if "row_index" in weighted_samples and "row_index" in truth
        else "object_id"
    )
    required_samples = {identity, z_parameter, weight_column}
    required_truth = {identity, truth_column}
    if not required_samples <= set(weighted_samples):
        raise ValueError(
            f"Missing posterior columns: {sorted(required_samples - set(weighted_samples))}"
        )
    if not required_truth <= set(truth):
        raise ValueError(
            f"Missing truth columns: {sorted(required_truth - set(truth))}"
        )
    if truth.duplicated(identity).any():
        raise ValueError(f"Truth contains duplicate {identity} values")
    truth_lookup = truth.set_index(identity, drop=False)
    rows: list[dict[str, Any]] = []
    for identity_value, group in weighted_samples.groupby(identity, sort=False):
        if identity_value not in truth_lookup.index:
            continue
        truth_row = truth_lookup.loc[identity_value]
        z_true = float(truth_row[truth_column])
        values = pd.to_numeric(group[z_parameter], errors="coerce").to_numpy()
        weights = pd.to_numeric(group[weight_column], errors="coerce").to_numpy()
        finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
        if not finite.any() or not np.isfinite(z_true) or weights[finite].sum() <= 0.0:
            continue
        values = values[finite]
        weights = weights[finite]
        weights = weights / weights.sum()
        q025, q16, q50, q84, q975 = _weighted_quantile(
            values, weights, (0.025, 0.16, 0.50, 0.84, 0.975)
        )
        delta = (q50 - z_true) / (1.0 + z_true)
        pit = float(
            weights[values < z_true].sum() + 0.5 * weights[values == z_true].sum()
        )
        rows.append(
            {
                identity: identity_value,
                "object_id": truth_row.get(
                    "object_id",
                    group.get("object_id", pd.Series([identity_value])).iloc[0],
                ),
                "z_true": z_true,
                "z_pred_q50": float(q50),
                "z_q025": float(q025),
                "z_q16": float(q16),
                "z_q84": float(q84),
                "z_q975": float(q975),
                "delta_z": float(delta),
                "pit": pit,
                "covered_68": bool(q16 <= z_true <= q84),
                "covered_95": bool(q025 <= z_true <= q975),
                "posterior_width_68": float(q84 - q16),
                "posterior_width_95": float(q975 - q025),
                "n_samples": int(len(values)),
            }
        )
    objects = pd.DataFrame(rows)
    if objects.empty:
        return objects, {"n_objects": 0}
    delta = objects["delta_z"].to_numpy(dtype=float)
    bias = float(np.median(delta))
    pit = objects["pit"].to_numpy(dtype=float)
    summary = {
        "n_objects": int(len(objects)),
        "median_bias": bias,
        "nmad": float(1.4826 * np.median(np.abs(delta - bias))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "outlier_fraction_0p15": float(np.mean(np.abs(delta) > 0.15)),
        "coverage_68": float(objects["covered_68"].mean()),
        "coverage_95": float(objects["covered_95"].mean()),
        "median_width_68": float(objects["posterior_width_68"].median()),
        "median_width_95": float(objects["posterior_width_95"].median()),
        "pit_mean": float(np.mean(pit)),
        "pit_variance": float(np.var(pit)),
        "pit_ks_uniform": _pit_ks_uniform(pit),
        "weight_column": weight_column,
        "point_estimate_contract": "z_pred_q50 is used only for z_true vs z_inferred metrics",
    }
    return objects, summary


def run_importance_correction(
    *,
    posterior: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    target_checkpoint: str | Path | None = None,
    truth_path: str | Path | None = None,
    truth_column: str = "redshift_true",
    resample_count: int = 128,
    seed: int = 42,
    prior_eval_batch_size: int = 65_536,
    min_median_ess_fraction: float = 0.01,
    max_fraction_pareto_k_gt_0p7: float = 0.5,
) -> dict[str, Any]:
    """Run a complete, auditable importance-correction workflow."""
    if int(resample_count) <= 0:
        raise ValueError("resample_count must be positive")
    if int(prior_eval_batch_size) <= 0:
        raise ValueError("prior_eval_batch_size must be positive")
    _validate_support_thresholds(
        min_median_ess_fraction=min_median_ess_fraction,
        max_fraction_pareto_k_gt_0p7=max_fraction_pareto_k_gt_0p7,
    )
    out = Path(out_dir)
    _refuse_nonempty_output(out)
    ensure_dir(out)
    config = load_config(config_path) if config_path is not None else None
    parameter_names = (
        tuple(_latent_spec_for_amortized_config(config).names)
        if config is not None
        else None
    )
    bank = load_posterior_bank(posterior, parameter_names=parameter_names)
    target_logprior = None
    if (config_path is None) != (target_checkpoint is None):
        raise ValueError("--config and --target-checkpoint must be provided together")
    if target_checkpoint is not None:
        target_logprior = evaluate_checkpoint_logprior(
            bank.frame,
            config_path=config_path,
            checkpoint=target_checkpoint,
            parameter_names=bank.parameter_names,
            batch_size=prior_eval_batch_size,
        )
    weighted, resampled, diagnostics = importance_weight_bank(
        bank,
        target_logprior=target_logprior,
        resample_count=resample_count,
        seed=seed,
    )
    weighted_dir = ensure_dir(out / "weighted_samples")
    resampled_dir = ensure_dir(out / "resampled_samples")
    weighted.to_parquet(weighted_dir / "batch_000000.parquet", index=False)
    resampled.to_parquet(resampled_dir / "batch_000000.parquet", index=False)
    diagnostics.to_parquet(out / "importance_diagnostics.parquet", index=False)
    diagnostics.to_csv(out / "importance_diagnostics.csv", index=False)
    median_raw_ess_fraction = float(diagnostics["raw_ess_fraction"].median())
    fraction_bad_k = float(np.nanmean(diagnostics["pareto_k"] > 0.7))
    support_pass = median_raw_ess_fraction >= float(
        min_median_ess_fraction
    ) and fraction_bad_k <= float(max_fraction_pareto_k_gt_0p7)
    support_gate = {
        "status": "PASS" if support_pass else "FAIL",
        "median_raw_ess_fraction": median_raw_ess_fraction,
        "min_median_raw_ess_fraction": float(min_median_ess_fraction),
        "fraction_pareto_k_gt_0p7": fraction_bad_k,
        "max_fraction_pareto_k_gt_0p7": float(max_fraction_pareto_k_gt_0p7),
        "interpretation": (
            "A failed gate makes the importance-corrected result inconclusive; "
            "PSIS cannot restore target support absent from the proposal bank."
        ),
    }
    write_json(out / "support_gate.json", support_gate)
    redshift_summary = None
    redshift_metrics_by_weight: dict[str, Any] | None = None
    if truth_path is not None:
        truth = pd.read_parquet(truth_path)
        redshift_metrics_by_weight = {}
        psis_objects = None
        for label, weight_column in (("raw", "raw_weight"), ("psis", "psis_weight")):
            objects, weight_summary = weighted_redshift_metrics(
                weighted,
                truth,
                weight_column=weight_column,
                truth_column=truth_column,
            )
            objects.to_parquet(
                out / f"redshift_{label}_weighted_objects.parquet", index=False
            )
            objects.to_csv(out / f"redshift_{label}_weighted_objects.csv", index=False)
            write_json(out / f"redshift_{label}_weighted_summary.json", weight_summary)
            redshift_metrics_by_weight[label] = weight_summary
            if label == "psis":
                psis_objects = objects
        # Backward-compatible aliases are explicitly the stabilized PSIS result.
        redshift_summary = redshift_metrics_by_weight["psis"]
        assert psis_objects is not None
        psis_objects.to_parquet(out / "redshift_weighted_objects.parquet", index=False)
        psis_objects.to_csv(out / "redshift_weighted_objects.csv", index=False)
        write_json(out / "redshift_weighted_summary.json", redshift_summary)
    summary = {
        "status": "complete",
        "n_objects": int(diagnostics.shape[0]),
        "n_joint_draws": int(len(weighted)),
        "resample_count_per_object": int(resample_count),
        "target_prior": "checkpoint" if target_checkpoint is not None else "source",
        "median_raw_ess_fraction": median_raw_ess_fraction,
        "median_psis_ess_fraction": float(diagnostics["psis_ess_fraction"].median()),
        "fraction_pareto_k_gt_0p7": fraction_bad_k,
        "fraction_pareto_k_gt_1": float(np.nanmean(diagnostics["pareto_k"] > 1.0)),
        "parameter_names": list(bank.parameter_names),
        "density_space": "network_latent_x",
        "distribution_contract": "All latent comparisons use weighted or resampled joint draws; q50 is restricted to redshift point metrics.",
        "support_gate": support_gate,
        "redshift_metrics": redshift_summary,
        "redshift_metrics_contract": (
            "redshift_metrics is the backward-compatible PSIS alias; "
            "redshift_metrics_by_weight reports raw self-normalized IW and PSIS separately."
        ),
        "redshift_metrics_by_weight": redshift_metrics_by_weight,
        "inputs": {
            "posterior": [_file_receipt(path) for path in bank.source_files],
            "config": _optional_file_receipt(config_path),
            "target_checkpoint": _optional_file_receipt(target_checkpoint),
            "truth": _optional_file_receipt(truth_path),
        },
    }
    write_json(out / "importance_summary.json", summary)
    write_json(out / "DONE", {"status": "complete"})
    return summary


def run_generalized_em(
    *,
    posterior: str | Path,
    config_path: str | Path,
    checkpoint: str | Path,
    out_dir: str | Path,
    iterations: int = 3,
    mstep_epochs: int = 5,
    object_batch_size: int = 64,
    learning_rate: float = 2.0e-5,
    weight_decay: float = 1.0e-6,
    trust_strength: float = 0.05,
    trust_samples: int = 512,
    weight_kind: str = "raw",
    seed: int = 42,
    validation_fraction: float = 0.1,
    min_median_ess_fraction: float = 0.01,
    max_fraction_pareto_k_gt_0p7: float = 0.5,
    allow_low_ess: bool = False,
) -> dict[str, Any]:
    """Fit a learned population prior with fixed-proposal generalized EM."""
    if weight_kind not in {"raw", "psis"}:
        raise ValueError("weight_kind must be 'raw' or 'psis'")
    if int(iterations) <= 0:
        raise ValueError("iterations must be positive")
    if int(mstep_epochs) <= 0:
        raise ValueError("mstep_epochs must be positive")
    if int(object_batch_size) <= 0:
        raise ValueError("object_batch_size must be positive")
    if int(trust_samples) <= 0:
        raise ValueError("trust_samples must be positive")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    _validate_support_thresholds(
        min_median_ess_fraction=min_median_ess_fraction,
        max_fraction_pareto_k_gt_0p7=max_fraction_pareto_k_gt_0p7,
    )
    out = Path(out_dir)
    _refuse_nonempty_output(out)
    ensure_dir(out / "checkpoints")
    config = load_config(config_path)
    latent_spec = _latent_spec_for_amortized_config(config)
    bank = load_posterior_bank(posterior, parameter_names=latent_spec.names)
    model = load_checkpoint(checkpoint, config)
    theta = bank.frame.loc[:, bank.parameter_names].to_numpy(dtype=np.float32)
    x = np.asarray(jax.device_get(theta_to_x(jnp.asarray(theta), latent_spec)))
    identities, inverse = np.unique(
        bank.frame[bank.identity_column].to_numpy(), return_inverse=True
    )
    counts = np.bincount(inverse)
    if len(np.unique(counts)) != 1:
        raise ValueError("Generalized EM requires the same proposal count per object")
    n_objects = int(len(identities))
    n_samples = int(counts[0])
    x = x.reshape(n_objects, n_samples, x.shape[-1])
    loglike = (
        bank.frame["loglike"].to_numpy(dtype=np.float64).reshape(n_objects, n_samples)
    )
    logq = bank.frame["logq"].to_numpy(dtype=np.float64).reshape(n_objects, n_samples)
    train_objects, validation_objects = _object_split(
        n_objects, validation_fraction=validation_fraction, seed=seed
    )
    pd.DataFrame({bank.identity_column: identities[train_objects]}).to_parquet(
        out / "train_object_identities.parquet", index=False
    )
    pd.DataFrame({bank.identity_column: identities[validation_objects]}).to_parquet(
        out / "validation_object_identities.parquet", index=False
    )
    optimizer = optax.adamw(
        learning_rate=float(learning_rate), weight_decay=float(weight_decay)
    )
    prior = model.prior
    history: list[dict[str, Any]] = []
    key = jax.random.PRNGKey(int(seed))
    initial_logprob = _evaluate_prior_array(prior, x)
    initial_train_evidence = _mean_log_evidence(
        loglike[train_objects], logq[train_objects], initial_logprob[train_objects]
    )
    initial_validation_evidence = _mean_log_evidence(
        loglike[validation_objects],
        logq[validation_objects],
        initial_logprob[validation_objects],
    )
    best_iteration = 0
    best_validation_evidence = initial_validation_evidence
    best_source = Path(checkpoint)
    stopping_reason = "requested_iterations_completed"
    for iteration in range(1, int(iterations) + 1):
        prior_logprob = _evaluate_prior_array(prior, x)
        raw_weights, psis_weights, diagnostic = _em_e_step(loglike, logq, prior_logprob)
        median_ess = float(np.median(diagnostic["raw_ess_fraction"]))
        bad_k = float(np.nanmean(diagnostic["pareto_k"] > 0.7))
        gate_pass = median_ess >= float(min_median_ess_fraction) and bad_k <= float(
            max_fraction_pareto_k_gt_0p7
        )
        if not gate_pass and not allow_low_ess:
            diagnostic.to_parquet(
                out / f"e_step_{iteration:03d}_diagnostics.parquet", index=False
            )
            raise RuntimeError(
                "Generalized-EM proposal support gate failed: "
                f"median ESS fraction={median_ess:.4g}, fraction k>0.7={bad_k:.4g}. "
                "Regenerate the proposal bank or pass --allow-low-ess for a diagnostic run."
            )
        weights = raw_weights if weight_kind == "raw" else psis_weights
        old_prior = prior
        opt_state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))
        epoch_losses: list[float] = []
        for epoch in range(int(mstep_epochs)):
            rng = np.random.default_rng(int(seed) + iteration * 10_000 + epoch)
            order = rng.permutation(train_objects)
            for start in range(0, len(order), int(object_batch_size)):
                object_index = order[start : start + int(object_batch_size)]
                key, trust_key = jax.random.split(key)
                trust_x = old_prior.sample(trust_key, (int(trust_samples),))
                loss, grads = _prior_loss_and_grad(
                    prior,
                    jnp.asarray(x[object_index]),
                    jnp.asarray(weights[object_index]),
                    jnp.asarray(trust_x),
                    float(trust_strength),
                )
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(prior, eqx.is_inexact_array)
                )
                prior = eqx.apply_updates(prior, updates)
                epoch_losses.append(float(jax.device_get(loss)))
        updated_logprob = _evaluate_prior_array(prior, x)
        _updated_raw, _updated_psis, updated_diagnostic = _em_e_step(
            loglike, logq, updated_logprob
        )
        updated_median_ess = float(np.median(updated_diagnostic["raw_ess_fraction"]))
        updated_bad_k = float(np.nanmean(updated_diagnostic["pareto_k"] > 0.7))
        updated_gate_pass = updated_median_ess >= float(
            min_median_ess_fraction
        ) and updated_bad_k <= float(max_fraction_pareto_k_gt_0p7)
        train_evidence = _mean_log_evidence(
            loglike[train_objects], logq[train_objects], updated_logprob[train_objects]
        )
        validation_evidence = _mean_log_evidence(
            loglike[validation_objects],
            logq[validation_objects],
            updated_logprob[validation_objects],
        )
        record = {
            "iteration": iteration,
            "mstep_mean_loss": float(np.mean(epoch_losses)),
            "train_mean_log_evidence_is": train_evidence,
            "validation_mean_log_evidence_is": validation_evidence,
            "validation_log_evidence_delta": float(
                validation_evidence - best_validation_evidence
            ),
            "median_raw_ess_fraction": median_ess,
            "fraction_pareto_k_gt_0p7": bad_k,
            "support_gate": "PASS" if gate_pass else "OVERRIDDEN",
            "updated_median_raw_ess_fraction": updated_median_ess,
            "updated_fraction_pareto_k_gt_0p7": updated_bad_k,
            "updated_support_gate": (
                "PASS"
                if updated_gate_pass
                else ("OVERRIDDEN" if allow_low_ess else "FAIL")
            ),
        }
        history.append(record)
        diagnostic.to_parquet(
            out / f"e_step_{iteration:03d}_diagnostics.parquet", index=False
        )
        _write_updated_checkpoint(
            out / "checkpoints" / f"iteration_{iteration:03d}.eqx",
            model,
            prior,
            source_checkpoint=checkpoint,
            metadata=record,
        )
        updated_diagnostic.to_parquet(
            out / f"updated_prior_{iteration:03d}_diagnostics.parquet", index=False
        )
        print(
            "[posthoc-em] "
            f"iteration={iteration}/{iterations} loss={record['mstep_mean_loss']:.6g} "
            f"validation_logz={validation_evidence:.6g} "
            f"median_ess_fraction={median_ess:.4g} pareto_k_bad={bad_k:.4g}",
            flush=True,
        )
        if not updated_gate_pass and not allow_low_ess:
            prior = old_prior
            stopping_reason = "updated_prior_support_gate_failed"
            break
        if np.isfinite(validation_evidence) and validation_evidence > float(
            best_validation_evidence
        ):
            best_iteration = iteration
            best_validation_evidence = validation_evidence
            best_source = out / "checkpoints" / f"iteration_{iteration:03d}.eqx"
        else:
            prior = old_prior
            stopping_reason = "heldout_evidence_did_not_improve"
            break
    shutil.copy2(best_source, out / "checkpoints" / "best.eqx")
    shutil.copy2(
        Path(str(best_source) + ".json"),
        out / "checkpoints" / "best.eqx.json",
    )
    pd.DataFrame(history).to_csv(out / "em_history.csv", index=False)
    summary = {
        "status": "complete",
        "algorithm": "fixed-proposal generalized EM",
        "weight_kind": weight_kind,
        "density_space": "network_latent_x",
        "n_objects": n_objects,
        "proposal_samples_per_object": n_samples,
        "iterations": int(len(history)),
        "requested_iterations": int(iterations),
        "completed_iterations": int(len(history)),
        "mstep_epochs": int(mstep_epochs),
        "trust_strength": float(trust_strength),
        "best_iteration": best_iteration,
        "selected_candidate": (
            "source_prior" if best_iteration == 0 else "updated_prior"
        ),
        "initial_train_mean_log_evidence_is": initial_train_evidence,
        "initial_validation_mean_log_evidence_is": initial_validation_evidence,
        "best_validation_mean_log_evidence_is": best_validation_evidence,
        "stopping_reason": stopping_reason,
        "proposal_refresh_required_if_support_fails": True,
        "distribution_contract": "M-step consumes stopped per-object joint importance weights; no marginal posterior medians are used.",
        "inputs": {
            "posterior": [_file_receipt(path) for path in bank.source_files],
            "config": _file_receipt(Path(config_path)),
            "checkpoint": _file_receipt(Path(checkpoint)),
        },
    }
    write_json(out / "em_summary.json", summary)
    write_json(out / "DONE", {"status": "complete"})
    return summary


@eqx.filter_jit
def _prior_loss_and_grad(prior, x, weights, trust_x, trust_strength):
    def loss_fn(candidate):
        logprob = candidate.log_prob(x)
        weighted_nll = -jnp.mean(jnp.sum(weights * logprob, axis=1))
        trust_cross_entropy = -jnp.mean(candidate.log_prob(trust_x))
        return weighted_nll + trust_strength * trust_cross_entropy

    return eqx.filter_value_and_grad(loss_fn)(prior)


def _evaluate_prior_array(
    prior, x: np.ndarray, batch_size: int = 262_144
) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    evaluate = eqx.filter_jit(prior.log_prob)
    result = []
    for start in range(0, len(flat), int(batch_size)):
        result.append(
            np.asarray(
                jax.device_get(evaluate(jnp.asarray(flat[start : start + batch_size]))),
                dtype=np.float64,
            )
        )
    return np.concatenate(result).reshape(x.shape[:-1])


def _em_e_step(loglike, logq, logprior):
    raw = np.empty_like(loglike, dtype=np.float64)
    psis = np.empty_like(loglike, dtype=np.float64)
    rows = []
    for index in range(loglike.shape[0]):
        result = normalized_importance_weights(
            loglike[index] + logprior[index], logq[index]
        )
        raw[index] = result["weight"]
        psis[index] = result["psis_weight"]
        rows.append(
            {
                "object_index": index,
                "raw_ess": result["raw_ess"],
                "raw_ess_fraction": result["raw_ess_fraction"],
                "psis_ess": result["psis_ess"],
                "psis_ess_fraction": float(result["psis_ess"] / loglike.shape[1]),
                "pareto_k": result["pareto_k"],
                "max_raw_weight": float(np.max(raw[index])),
            }
        )
    return raw, psis, pd.DataFrame(rows)


def _object_split(n_objects: int, *, validation_fraction: float, seed: int):
    if n_objects < 2:
        raise ValueError("Generalized EM requires at least two objects")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_objects)
    n_validation = min(
        n_objects - 1, max(1, int(round(float(validation_fraction) * n_objects)))
    )
    return np.sort(order[n_validation:]), np.sort(order[:n_validation])


def _validate_support_thresholds(
    *,
    min_median_ess_fraction: float,
    max_fraction_pareto_k_gt_0p7: float,
) -> None:
    if not 0.0 <= float(min_median_ess_fraction) <= 1.0:
        raise ValueError("min_median_ess_fraction must lie between 0 and 1")
    if not 0.0 <= float(max_fraction_pareto_k_gt_0p7) <= 1.0:
        raise ValueError("max_fraction_pareto_k_gt_0p7 must lie between 0 and 1")


def _mean_log_evidence(loglike, logq, logprior) -> float:
    logweight = np.asarray(loglike + logprior - logq, dtype=np.float64)
    maximum = np.max(logweight, axis=1, keepdims=True)
    logz = maximum[:, 0] + np.log(np.mean(np.exp(logweight - maximum), axis=1))
    return float(np.mean(logz))


def _write_updated_checkpoint(path, model, prior, *, source_checkpoint, metadata):
    updated_model = eqx.tree_at(lambda value: value.prior, model, prior)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, updated_model)
    source_sidecar = Path(str(source_checkpoint) + ".json")
    sidecar = (
        json.loads(source_sidecar.read_text(encoding="utf-8"))
        if source_sidecar.is_file()
        else {}
    )
    feature_stats = sidecar.get("feature_stats_path")
    if feature_stats and not Path(str(feature_stats)).is_absolute():
        feature_stats = (Path(source_checkpoint).parent / str(feature_stats)).resolve()
        sidecar["feature_stats_path"] = str(feature_stats)
    sidecar["posthoc_empirical_bayes"] = {
        **metadata,
        "source_checkpoint": _file_receipt(Path(source_checkpoint)),
    }
    write_json(str(path) + ".json", sidecar)


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    return np.interp(np.asarray(quantiles, dtype=float), cumulative, values)


def _pit_ks_uniform(values: np.ndarray) -> float:
    from scipy.stats import kstest

    return float(kstest(np.asarray(values, dtype=float), "uniform").statistic)


def _logmeanexp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return float("nan")
    maximum = float(np.max(finite))
    return float(maximum + math.log(np.mean(np.exp(finite - maximum))))


def _file_receipt(path: Path) -> dict[str, Any]:
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


def _optional_file_receipt(path):
    return None if path is None else _file_receipt(Path(path))


def _refuse_nonempty_output(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
