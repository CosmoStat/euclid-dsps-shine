"""Direct distributional projections for selection-aware population inference.

This module deliberately separates two statistical objects:

* the object-equal aggregate of dense joint ``q(theta | y)`` draws for the
  observed-selected catalogue;
* the parent C0 population reconstructed from those draws with weights
  proportional to ``1 / beta(theta)``.

The helpers never reduce an object posterior to a point estimate. Posterior
calibration against truth belongs to object-aligned PIT/MIRA/TARP diagnostics;
the CDF and rank helpers here compare population distributions only.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

from ..filters import load_filters
from ..model import dynamic_model_args, load_context
from .config import require_amortized_dependencies
from .features import read_feature_stats
from .population_vem import require_git_commit, sha256_file
from .train import (
    JitLatentSpec,
    _latent_spec_for_amortized_config,
    _selection_correction_runtime_config,
    _selection_log_beta_from_prior_samples,
)

eqx, _optax = require_amortized_dependencies()


class WeightedDensityMetrics(NamedTuple):
    loss: jnp.ndarray
    weight_sum: jnp.ndarray
    valid_samples: jnp.ndarray
    raw_gradient_norm: jnp.ndarray
    gradients_finite: jnp.ndarray
    update_applied: jnp.ndarray


def require_projection_runtime_commit(
    root: str | Path,
    manifest: dict[str, Any],
    repo: str | Path,
) -> dict[str, Any]:
    """Accept the frozen manifest commit or one narrowly authorized recovery."""
    root = Path(root).resolve()
    manifest_path = root / "RUN_MANIFEST.json"
    manifest_commit = str(manifest["code_commit"])
    recovery_path = root / "CODE_RECOVERY.json"
    if not recovery_path.is_file():
        actual = require_git_commit(repo, manifest_commit)
        return {
            "mode": "manifest",
            "manifest_code_commit": manifest_commit,
            "runtime_code_commit": actual,
        }

    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    submission = json.loads((root / "SUBMISSION.json").read_text(encoding="utf-8"))
    beta_path = root / "BETA_TARGET_COMPLETE.json"
    checks = {
        "status": recovery.get("status") == "AUTHORIZED",
        "scope": recovery.get("scope") == "fit_and_evaluation_only",
        "root": Path(recovery.get("projection_root", "")).resolve() == root,
        "manifest_sha256": recovery.get("manifest_sha256")
        == sha256_file(manifest_path),
        "beta_receipt_sha256": recovery.get("beta_receipt_sha256")
        == sha256_file(beta_path),
        "manifest_code_commit": recovery.get("manifest_code_commit") == manifest_commit,
        "failed_fit_job": str(recovery.get("failed_fit_job"))
        == str(submission.get("fit_job")),
        "beta_banks_reused": recovery.get("beta_banks_reused") is True,
        "truth_used": recovery.get("truth_used") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"invalid population-projection recovery: {failed}")
    actual = require_git_commit(repo, str(recovery["runtime_code_commit"]))
    return {
        "mode": "authorized_recovery",
        "manifest_code_commit": manifest_commit,
        "runtime_code_commit": actual,
        "recovery_receipt": str(recovery_path),
        "recovery_receipt_sha256": sha256_file(recovery_path),
        "failed_fit_job": str(recovery["failed_fit_job"]),
    }


def selection_runtime(config: dict[str, Any], feature_stats_path: str | Path):
    """Build the truth-free DSPS runtime used only to cache beta(theta)."""
    no_truth = copy.deepcopy(config)
    no_truth.setdefault("truth", {})["parameter_columns"] = {}
    stats = read_feature_stats(feature_stats_path)
    filters = load_filters(no_truth["bands"])
    context = load_context(
        no_truth["ssp_path"],
        filters,
        n_sfh_bins=int(no_truth["model"].get("n_sfh_bins", 96)),
        cosmos_config=no_truth.get("cosmos_sed"),
        nebular_emission=no_truth.get("nebular_emission", "ssp_flux"),
        model_config=no_truth.get("model"),
    )
    latent_spec = _latent_spec_for_amortized_config(no_truth)
    jit_spec = JitLatentSpec(
        names=latent_spec.names,
        lower=latent_spec.lower,
        upper=latent_spec.upper,
        raw_center=latent_spec.raw_center,
        raw_scale=latent_spec.raw_scale,
        normalization=latent_spec.normalization,
        transform_family=latent_spec.transform_family,
        transform_location=latent_spec.transform_location,
        transform_lambda=latent_spec.transform_lambda,
    )
    return (
        latent_spec,
        jit_spec,
        context,
        dynamic_model_args(context),
        _selection_correction_runtime_config(no_truth, stats),
        {"calibration": no_truth.get("calibration", {}) or {}},
    )


def evaluate_log_beta(
    model,
    x: np.ndarray,
    runtime,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate log beta in bounded chunks without retaining DSPS activations."""
    latent_spec, jit_spec, context, model_args, selection, calibration = runtime
    pieces = []
    for start in range(0, len(x), int(chunk_size)):
        stop = min(start + int(chunk_size), len(x))
        values = _selection_log_beta_from_prior_samples(
            model,
            jnp.asarray(x[start:stop]),
            jit_spec,
            context,
            model_args,
            latent_spec.names,
            calibration,
            selection,
        )
        pieces.append(np.asarray(jax.device_get(values), dtype=np.float64))
        print(f"[population-projection] beta {stop}/{len(x)}", flush=True)
    return np.concatenate(pieces)


def inverse_selection_weights(
    log_beta: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mean-one ``1 / beta`` weights and stable audit diagnostics."""
    values = np.asarray(log_beta, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("inverse-selection target cannot be empty")
    if np.any(np.isnan(values)) or np.any(values > 1.0e-10):
        raise ValueError("log_beta must be <= 0 and cannot contain NaN")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "inverse-selection target contains beta=0; the parent is not "
            "identified by finite weights"
        )

    log_inverse = -values
    log_mean_inverse = float(logsumexp(log_inverse) - np.log(values.size))
    log_alpha = -log_mean_inverse
    normalized = np.exp(log_inverse - logsumexp(log_inverse))
    mean_one = normalized * values.size
    ess = float(1.0 / np.sum(np.square(normalized)))
    diagnostics = {
        "status": "complete",
        "samples": int(values.size),
        "log_alpha_harmonic": log_alpha,
        "alpha_harmonic": float(np.exp(log_alpha)),
        "ess": ess,
        "ess_fraction": ess / values.size,
        "maximum_normalized_weight": float(np.max(normalized)),
        "log_beta_min": float(np.min(values)),
        "log_beta_median": float(np.quantile(values, 0.5)),
        "log_beta_p01": float(np.quantile(values, 0.01)),
        "log_beta_p99": float(np.quantile(values, 0.99)),
        "weight_contract": "joint-draw weights proportional to 1 / beta(theta)",
        "normalization": "mean-one weights for unbiased minibatch weighted MLE",
    }
    return mean_one.astype(np.float32), diagnostics


def weighted_cdf_values(
    reference: np.ndarray,
    query: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the right-continuous weighted empirical CDF at ``query``."""
    values = np.asarray(reference, dtype=np.float64).reshape(-1)
    points = np.asarray(query, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("reference distribution must be finite and non-empty")
    if not np.all(np.isfinite(points)):
        raise ValueError("CDF query values must be finite")
    if weights is None:
        probability = np.full(values.size, 1.0 / values.size, dtype=np.float64)
    else:
        probability = np.asarray(weights, dtype=np.float64).reshape(-1)
        if probability.shape != values.shape:
            raise ValueError("CDF weights must match the reference values")
        if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
            raise ValueError("CDF weights must be finite and non-negative")
        total = float(np.sum(probability))
        if total <= 0.0:
            raise ValueError("CDF weights must have positive mass")
        probability = probability / total
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative = np.cumsum(probability[order])
    indices = np.searchsorted(sorted_values, points, side="right") - 1
    return np.where(indices >= 0, cumulative[np.maximum(indices, 0)], 0.0)


def uniform_cdf_distance(
    values: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Kolmogorov distance between values in [0, 1] and a uniform CDF."""
    ordered = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("rank values must be finite and non-empty")
    if np.any((ordered < 0.0) | (ordered > 1.0)):
        raise ValueError("rank values must lie in [0, 1]")
    if weights is None:
        probability = np.full(ordered.size, 1.0 / ordered.size)
    else:
        raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        if raw.shape != ordered.shape:
            raise ValueError("rank weights must match rank values")
        if not np.all(np.isfinite(raw)) or np.any(raw < 0.0) or raw.sum() <= 0.0:
            raise ValueError("rank weights must be finite, non-negative, and nonzero")
        order = np.argsort(np.asarray(values, dtype=np.float64).reshape(-1))
        probability = raw[order] / raw.sum()
    cumulative = np.cumsum(probability)
    previous = cumulative - probability
    upper = np.max(cumulative - ordered)
    lower = np.max(ordered - previous)
    return float(max(upper, lower))


def weighted_cdf_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_weights: np.ndarray | None = None,
    right_weights: np.ndarray | None = None,
) -> float:
    """Return the supremum distance between two complete empirical CDFs."""
    grid = np.unique(
        np.concatenate(
            (
                np.asarray(left, dtype=np.float64).reshape(-1),
                np.asarray(right, dtype=np.float64).reshape(-1),
            )
        )
    )
    left_cdf = weighted_cdf_values(left, grid, left_weights)
    right_cdf = weighted_cdf_values(right, grid, right_weights)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def distribution_comparison(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare two 1D population distributions without point summaries."""
    source_values = np.asarray(source, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    if target_weights is None:
        q25, q75 = np.quantile(target_values, [0.25, 0.75])
    else:
        q25, q75 = _weighted_quantiles(
            target_values, target_weights, np.asarray([0.25, 0.75])
        )
    target_iqr = float(q75 - q25)
    target_iqr = max(target_iqr, 1.0e-8)
    ranks = weighted_cdf_values(target_values, source_values, target_weights)
    wasserstein = float(
        wasserstein_distance(
            source_values,
            target_values,
            u_weights=source_weights,
            v_weights=target_weights,
        )
    )
    return {
        "wasserstein": wasserstein,
        "wasserstein_over_target_iqr": wasserstein / target_iqr,
        "cdf_supremum": weighted_cdf_distance(
            source_values,
            target_values,
            left_weights=source_weights,
            right_weights=target_weights,
        ),
        "distribution_rank_uniform_ks": uniform_cdf_distance(ranks, source_weights),
        "target_iqr": target_iqr,
    }


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if weights.shape != values.shape:
        raise ValueError("quantile weights must match values")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0:
        raise ValueError("quantile weights must be finite, non-negative, and nonzero")
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, values[order])


def make_pmap_weighted_density_step(*, optimizer: Any):
    """Build a data-parallel weighted maximum-likelihood flow update."""
    array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(array_axis, array_axis, array_axis, array_axis, array_axis),
        out_axes=(array_axis, array_axis, array_axis),
    )
    def step(prior, optimizer_state, x, weight, valid):
        x = jax.lax.stop_gradient(x)
        weight = jax.lax.stop_gradient(weight)
        valid = jax.lax.stop_gradient(valid)

        def objective(candidate_prior):
            log_prob = candidate_prior.log_prob(x)
            usable = valid & jnp.isfinite(log_prob) & jnp.isfinite(weight)
            usable &= weight >= 0.0
            safe_weight = jnp.where(usable, weight, 0.0)
            local_numerator = jnp.sum(
                safe_weight * jnp.where(jnp.isfinite(log_prob), -log_prob, 0.0)
            )
            local_denominator = jnp.sum(safe_weight)
            numerator = jax.lax.psum(local_numerator, "devices")
            weight_sum = jax.lax.psum(local_denominator, "devices")
            valid_samples = jax.lax.psum(jnp.sum(usable.astype(jnp.int32)), "devices")
            # Parent weights are normalized to have global mean one. Dividing
            # by the sample count keeps uniform minibatches unbiased for the
            # fixed weighted target; per-batch self-normalization would not.
            loss = numerator / jnp.maximum(valid_samples, 1)
            return loss, (weight_sum, valid_samples)

        (loss, auxiliary), gradients = eqx.filter_value_and_grad(
            objective, has_aux=True
        )(prior)
        gradients = _pmean_tree(gradients, "devices")
        gradient_norm = _tree_l2_norm(gradients)
        gradients_finite = _tree_all_finite(gradients)
        weight_sum, valid_samples = auxiliary
        pre_update_ok = (
            jnp.isfinite(loss)
            & gradients_finite
            & (weight_sum > 0.0)
            & (valid_samples > 0)
        )
        safe_gradients = jax.tree_util.tree_map(
            lambda value: (
                jnp.where(pre_update_ok, value, jnp.zeros_like(value))
                if value is not None
                else None
            ),
            gradients,
        )
        updates, proposed_state = optimizer.update(
            safe_gradients,
            optimizer_state,
            eqx.filter(prior, eqx.is_inexact_array),
        )
        proposed_prior = eqx.apply_updates(prior, updates)
        proposed_finite = _tree_all_finite(proposed_prior)
        apply_update = jax.lax.pmin(
            (pre_update_ok & proposed_finite).astype(jnp.int32), "devices"
        ).astype(jnp.bool_)
        prior = _select_tree(proposed_prior, prior, apply_update)
        optimizer_state = _select_tree(proposed_state, optimizer_state, apply_update)
        metrics = WeightedDensityMetrics(
            loss=loss,
            weight_sum=weight_sum,
            valid_samples=valid_samples,
            raw_gradient_norm=gradient_norm,
            gradients_finite=gradients_finite,
            update_applied=apply_update,
        )
        return prior, optimizer_state, metrics

    return step


def _tree_l2_norm(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and eqx.is_inexact_array(value)
    ]
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves))


def _tree_all_finite(tree) -> jnp.ndarray:
    leaves = [
        value
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None and eqx.is_inexact_array(value)
    ]
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))


def _pmean_tree(tree, axis_name: str):
    return jax.tree_util.tree_map(
        lambda value: jax.lax.pmean(value, axis_name) if value is not None else None,
        tree,
    )


def _select_tree(candidate, fallback, condition):
    return jax.tree_util.tree_map(
        lambda proposed, original: (
            jnp.where(condition, proposed, original)
            if eqx.is_array(proposed)
            else proposed
        ),
        candidate,
        fallback,
    )
