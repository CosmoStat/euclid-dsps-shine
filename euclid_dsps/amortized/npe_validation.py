"""Truth-free diagnostics for conditional posterior proposals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats

from .exact_posterior import normalized_importance_weights
from .features import FeatureStats, make_encoder_features

_FORBIDDEN_TRUTH_TOKENS = (
    "truth",
    "true_",
    "_true",
    "ground_truth",
)


def assert_truth_free_columns(columns: Sequence[str]) -> None:
    """Fail before a tuning path consumes catalogue-truth columns."""
    forbidden = [
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in _FORBIDDEN_TRUTH_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"catalogue truth columns are forbidden here: {forbidden}")


def summarize_truth_free_joint_bank(
    frame: pd.DataFrame,
    *,
    parameter_names: Sequence[str],
    identity_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize exact proposal weights without reading any catalogue truth."""
    assert_truth_free_columns(frame.columns)
    required = {
        identity_column,
        "sample_id",
        "logq",
        "logprior",
        "loglike",
        *parameter_names,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"posterior bank lacks required columns: {missing}")
    rows: list[dict[str, Any]] = []
    evidence_halves: list[float] = []
    for identity, group in frame.groupby(identity_column, sort=False):
        group = group.sort_values("sample_id")
        logq = pd.to_numeric(group["logq"], errors="coerce").to_numpy(float)
        logprior = pd.to_numeric(group["logprior"], errors="coerce").to_numpy(float)
        loglike = pd.to_numeric(group["loglike"], errors="coerce").to_numpy(float)
        result = normalized_importance_weights(loglike + logprior, logq)
        raw = np.asarray(result["weight"], dtype=float)
        pareto = float(result["pareto_k"])
        logweight = loglike + logprior - logq
        first = np.arange(len(group)) % 2 == 0
        second = ~first
        logz_first = _logmeanexp(logweight[first])
        logz_second = _logmeanexp(logweight[second])
        evidence_halves.append(abs(logz_first - logz_second))
        values = group.loc[:, parameter_names].to_numpy(float)
        finite_values = np.all(np.isfinite(values), axis=1)
        covariance = (
            np.cov(values[finite_values], rowvar=False)
            if np.sum(finite_values) > 1
            else np.full((len(parameter_names), len(parameter_names)), np.nan)
        )
        rows.append(
            {
                identity_column: identity,
                "draws": int(len(group)),
                "finite_logweights": int(np.isfinite(logweight).sum()),
                "raw_ess": float(result["raw_ess"]),
                "raw_ess_fraction": float(result["raw_ess_fraction"]),
                "pareto_k": pareto,
                "pareto_k_finite": bool(np.isfinite(pareto)),
                "max_raw_weight": float(np.max(raw)),
                "log_evidence_even": logz_first,
                "log_evidence_odd": logz_second,
                "log_evidence_split_abs_delta": abs(logz_first - logz_second),
                "mean_loglike": float(np.nanmean(loglike)),
                "mean_logprior": float(np.nanmean(logprior)),
                "mean_logq": float(np.nanmean(logq)),
                "mean_posterior_std": float(
                    np.nanmean(np.nanstd(values, axis=0, ddof=1))
                ),
                "maximum_abs_posterior_correlation": _maximum_abs_correlation(
                    covariance
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    ess = diagnostics["raw_ess_fraction"].to_numpy(float)
    pareto = diagnostics["pareto_k"].to_numpy(float)
    max_weight = diagnostics["max_raw_weight"].to_numpy(float)
    finite_k = np.isfinite(pareto)
    summary = {
        "objects": int(len(diagnostics)),
        "draws_per_object": sorted(diagnostics["draws"].unique().tolist()),
        "raw_ess": {
            "median": float(np.nanmedian(diagnostics["raw_ess"])),
            "fraction_median": float(np.nanmedian(ess)),
            "fraction_q10": float(np.nanquantile(ess, 0.10)),
            "fraction_below_0p05": float(np.mean(ess < 0.05)),
        },
        "pareto_k": {
            "finite_fraction": float(np.mean(finite_k)),
            "nonfinite_fraction": float(np.mean(~finite_k)),
            "finite_gt_0p7_fraction": float(np.mean(pareto[finite_k] > 0.7))
            if finite_k.any()
            else None,
            "finite_gt_1_fraction": float(np.mean(pareto[finite_k] > 1.0))
            if finite_k.any()
            else None,
            "gt_0p7_or_nonfinite_fraction": float(np.mean(~finite_k | (pareto > 0.7))),
        },
        "maximum_raw_weight": {
            "median": float(np.nanmedian(max_weight)),
            "p90": float(np.nanquantile(max_weight, 0.90)),
        },
        "log_density_decomposition": {
            key: float(np.nanmean(diagnostics[key]))
            for key in ("mean_loglike", "mean_logprior", "mean_logq")
        },
        "independent_draw_stability": {
            "median_abs_log_evidence_delta": float(np.nanmedian(evidence_halves)),
            "p90_abs_log_evidence_delta": float(np.nanquantile(evidence_halves, 0.90)),
        },
        "posterior_geometry": {
            "median_mean_coordinate_std": float(
                np.nanmedian(diagnostics["mean_posterior_std"])
            ),
            "median_maximum_abs_correlation": float(
                np.nanmedian(diagnostics["maximum_abs_posterior_correlation"])
            ),
        },
    }
    return summary, diagnostics


def posterior_support_gate(
    summary: dict[str, Any],
    *,
    minimum_median_ess_fraction: float = 0.05,
    maximum_bad_pareto_fraction: float = 0.20,
    maximum_p90_raw_weight: float = 0.80,
) -> dict[str, Any]:
    """Apply a technical support gate, not a scientific posterior certificate."""
    checks = {
        "median_ess_fraction": bool(
            summary["raw_ess"]["fraction_median"] >= float(minimum_median_ess_fraction)
        ),
        "pareto_tail": bool(
            summary["pareto_k"]["gt_0p7_or_nonfinite_fraction"]
            <= float(maximum_bad_pareto_fraction)
        ),
        "maximum_weight": bool(
            summary["maximum_raw_weight"]["p90"] <= float(maximum_p90_raw_weight)
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "thresholds": {
            "minimum_median_ess_fraction": float(minimum_median_ess_fraction),
            "maximum_pareto_k_gt_0p7_or_nonfinite_fraction": float(
                maximum_bad_pareto_fraction
            ),
            "maximum_p90_raw_weight": float(maximum_p90_raw_weight),
        },
        "interpretation": "technical prerequisite only; not posterior validation",
    }


def summarize_normalized_residuals(frame: pd.DataFrame) -> dict[str, Any]:
    """Report robust and tail-aware photometric residual metrics by band."""
    assert_truth_free_columns(frame.columns)
    candidates = (
        "normalized_residual",
        "standardized_residual",
        "residual_sigma",
    )
    column = next((name for name in candidates if name in frame), None)
    if column is None:
        raise ValueError(f"residual frame lacks one of {candidates}")
    rows = {}
    for band, group in frame.groupby("band", sort=True):
        value = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        value = value[np.isfinite(value)]
        rows[str(band)] = _residual_metrics(value)
    all_values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    all_values = all_values[np.isfinite(all_values)]
    return {"all_bands": _residual_metrics(all_values), "by_band": rows}


def mask_held_out_bands(
    flux: jnp.ndarray,
    flux_err: jnp.ndarray,
    mask: jnp.ndarray,
    feature_stats: FeatureStats,
    held_out_band_indices: Sequence[int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Remove bands from both q features and the conditioning likelihood mask."""
    conditioned_mask = jnp.asarray(mask, dtype=bool)
    for index in held_out_band_indices:
        conditioned_mask = conditioned_mask.at[:, int(index)].set(False)
    features = make_encoder_features(flux, flux_err, feature_stats, conditioned_mask)
    return features, conditioned_mask


def flux_error_jacobian_sensitivity(
    flux_fn: Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    flux_err: jnp.ndarray,
    mask: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Return singular values and coordinate norms of d(flux/error)/dx."""
    jacobian = jax.jacrev(flux_fn)(x)
    scale = jnp.where(mask, jnp.maximum(flux_err, 1.0e-30), jnp.inf)
    normalized = jacobian / scale[:, None]
    normalized = jnp.where(jnp.isfinite(normalized), normalized, 0.0)
    singular = jnp.linalg.svd(normalized, full_matrices=False, compute_uv=False)
    coordinate_norm = jnp.sqrt(jnp.sum(normalized**2, axis=0))
    return {
        "jacobian": normalized,
        "singular_values": singular,
        "coordinate_norm": coordinate_norm,
        "near_zero_coordinate": coordinate_norm <= 1.0e-8,
    }


def explicit_mixture_log_prob(
    component_log_prob: jnp.ndarray,
    component_weight: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the complete density of an explicit defensive mixture."""
    log_prob = jnp.asarray(component_log_prob)
    weight = jnp.asarray(component_weight, dtype=log_prob.dtype)
    if log_prob.ndim < 1 or weight.ndim != 1:
        raise ValueError("mixture log-probabilities need a final component axis")
    if log_prob.shape[-1] != weight.shape[0]:
        raise ValueError("mixture component count and weights differ")
    if bool(np.any(np.asarray(weight) <= 0.0)):
        raise ValueError("all defensive-mixture weights must be positive")
    if not np.isclose(float(np.asarray(jnp.sum(weight))), 1.0, atol=1.0e-6):
        raise ValueError("defensive-mixture weights must sum to one")
    return jax.scipy.special.logsumexp(
        log_prob + jnp.log(weight),
        axis=-1,
    )


def summarize_model_generated_rank_calibration(
    posterior_samples: np.ndarray,
    generated_x: np.ndarray,
    *,
    parameter_names: Sequence[str],
    seed: int,
    maximum_ks: float = 0.08,
    maximum_coverage_ece: float = 0.08,
) -> dict[str, Any]:
    """Finite-K SBC summary for newly generated model parameters.

    The randomized ranks account for the discrete grid induced by K posterior
    draws.  These parameters are generated as part of the validation
    experiment; no catalogue-truth column is accepted by this function.
    """
    samples = np.asarray(posterior_samples, dtype=float)
    generated = np.asarray(generated_x, dtype=float)
    if samples.ndim != 3:
        raise ValueError("posterior samples must have shape [K, N, D]")
    if generated.shape != samples.shape[1:]:
        raise ValueError("generated parameters must have shape [N, D]")
    if samples.shape[-1] != len(parameter_names):
        raise ValueError("parameter names do not match the posterior dimension")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(generated)):
        raise ValueError("model-generated rank inputs must be finite")

    rng = np.random.default_rng(int(seed))
    rows: dict[str, dict[str, float]] = {}
    coverage_errors: list[float] = []
    ks_values: list[float] = []
    for index, name in enumerate(parameter_names):
        values = samples[:, :, index]
        truth = generated[:, index]
        ranks = np.sum(values < truth[None, :], axis=0)
        randomized = (ranks + rng.uniform(size=ranks.shape)) / (samples.shape[0] + 1.0)
        ks = float(stats.kstest(randomized, "uniform").statistic)
        q16, q025 = np.quantile(values, [0.16, 0.025], axis=0)
        q84, q975 = np.quantile(values, [0.84, 0.975], axis=0)
        coverage_68 = float(np.mean((truth >= q16) & (truth <= q84)))
        coverage_95 = float(np.mean((truth >= q025) & (truth <= q975)))
        ece = 0.5 * (abs(coverage_68 - 0.68) + abs(coverage_95 - 0.95))
        ks_values.append(ks)
        coverage_errors.append(ece)
        rows[str(name)] = {
            "pit_ks_uniform": ks,
            "pit_mean": float(np.mean(randomized)),
            "coverage_68": coverage_68,
            "coverage_95": coverage_95,
            "coverage_ece": float(ece),
        }
    maximum_observed_ks = float(np.max(ks_values))
    maximum_observed_ece = float(np.max(coverage_errors))
    checks = {
        "maximum_coordinate_pit_ks": maximum_observed_ks <= float(maximum_ks),
        "maximum_coordinate_coverage_ece": maximum_observed_ece
        <= float(maximum_coverage_ece),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract": "randomized finite-rank SBC on newly simulated parameters",
        "objects": int(samples.shape[1]),
        "samples_per_object": int(samples.shape[0]),
        "maximum_coordinate_pit_ks": maximum_observed_ks,
        "maximum_coordinate_coverage_ece": maximum_observed_ece,
        "thresholds": {
            "maximum_ks": float(maximum_ks),
            "maximum_coverage_ece": float(maximum_coverage_ece),
        },
        "checks": checks,
        "parameters": rows,
        "catalogue_truth_used": False,
    }


def held_out_band_predictive_gate(
    observed: dict[str, Any],
    model_generated_reference: dict[str, Any],
    *,
    maximum_median_abs_excess: float = 0.50,
    maximum_rms_ratio: float = 2.0,
    maximum_fraction_abs_gt_5_excess: float = 0.10,
) -> dict[str, Any]:
    """Compare held-out residuals with the same-model simulation reference."""
    observed_bands = dict(observed.get("by_band", {}) or {})
    reference_bands = dict(model_generated_reference.get("by_band", {}) or {})
    common = sorted(set(observed_bands) & set(reference_bands))
    if not common:
        raise ValueError("held-out validation has no common observed/reference bands")
    rows = {}
    for band in common:
        value = observed_bands[band]
        reference = reference_bands[band]
        if not value.get("count") or not reference.get("count"):
            raise ValueError(f"held-out band {band} has no finite residuals")
        median_excess = float(value["median_abs"] - reference["median_abs"])
        rms_ratio = float(value["rms"] / max(reference["rms"], 1.0e-12))
        tail_excess = float(value["fraction_abs_gt_5"] - reference["fraction_abs_gt_5"])
        checks = {
            "median_abs": median_excess <= float(maximum_median_abs_excess),
            "rms": rms_ratio <= float(maximum_rms_ratio),
            "tail_5sigma": tail_excess <= float(maximum_fraction_abs_gt_5_excess),
        }
        rows[band] = {
            "median_abs_excess": median_excess,
            "rms_ratio": rms_ratio,
            "fraction_abs_gt_5_excess": tail_excess,
            "checks": checks,
        }
    passed = all(all(record["checks"].values()) for record in rows.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "contract": (
            "held-out bands absent from q features and conditioning likelihood; "
            "comparison to a matched model-generated residual reference"
        ),
        "bands": rows,
        "thresholds": {
            "maximum_median_abs_excess": float(maximum_median_abs_excess),
            "maximum_rms_ratio": float(maximum_rms_ratio),
            "maximum_fraction_abs_gt_5_excess": float(maximum_fraction_abs_gt_5_excess),
        },
        "truth_used": False,
    }


def _residual_metrics(value: np.ndarray) -> dict[str, float | int | None]:
    if value.size == 0:
        return {"count": 0, "median_abs": None, "rms": None}
    absolute = np.abs(value)
    return {
        "count": int(value.size),
        "median_abs": float(np.median(absolute)),
        "rms": float(np.sqrt(np.mean(value**2))),
        "abs_q90": float(np.quantile(absolute, 0.90)),
        "abs_q99": float(np.quantile(absolute, 0.99)),
        "fraction_abs_gt_3": float(np.mean(absolute > 3.0)),
        "fraction_abs_gt_5": float(np.mean(absolute > 5.0)),
    }


def _logmeanexp(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=float)
    finite = np.isfinite(value)
    if not finite.any():
        return float("nan")
    maximum = np.max(value[finite])
    return float(maximum + np.log(np.mean(np.exp(value[finite] - maximum))))


def _maximum_abs_correlation(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        return 0.0
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = scale[:, None] * scale[None, :]
    correlation = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0.0,
    )
    np.fill_diagonal(correlation, np.nan)
    return float(np.nanmax(np.abs(correlation))) if correlation.size > 1 else 0.0
