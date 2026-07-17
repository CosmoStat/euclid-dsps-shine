"""Validation-only diagnostics for the spline-15D RealNVP prior."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .spline15d import (
    SPLINE15D_PARAMETER_NAMES,
    inverse_spline15d_flow_coordinates,
)


def exact_truth_hashes(frame: pd.DataFrame) -> np.ndarray:
    """Return stable row hashes in the fixed spline-15D parameter order."""
    return pd.util.hash_pandas_object(
        frame.loc[:, SPLINE15D_PARAMETER_NAMES],
        index=False,
    ).to_numpy(dtype=np.uint64)


def novel_truth_mask(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> np.ndarray:
    """Identify candidate truths that are not exact rows of the reference."""
    reference_hashes = np.unique(exact_truth_hashes(reference))
    return ~np.isin(exact_truth_hashes(candidate), reference_hashes)


def evaluate_generated_prior(
    prior,
    *,
    truth_theta: np.ndarray,
    truth_x: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    sample_count: int,
    seed: int,
    temperature: float = 1.0,
    whitening: dict[str, Any] | None = None,
    atom_half_width: float | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate generated samples and base typicality against validation truth."""
    count = max(int(sample_count), 256)
    key = jax.random.PRNGKey(int(seed))
    prior_x = np.asarray(
        jax.device_get(
            prior.sample_with_temperature(
                key,
                count,
                temperature=float(temperature),
            )
        ),
        dtype=np.float64,
    )
    try:
        prior_theta = inverse_spline15d_flow_coordinates(
            prior_x,
            transforms=transforms,
            whitening=whitening,
            atom_half_width=atom_half_width,
        )
    except ValueError:
        prior_theta = np.full_like(prior_x, np.nan)
    truth_u, _logdet = prior.inverse(jnp.asarray(truth_x, dtype=jnp.float32))
    truth_u = np.asarray(jax.device_get(truth_u), dtype=np.float64)
    metrics = _sample_metrics(
        truth_theta=np.asarray(truth_theta, dtype=np.float64),
        truth_x=np.asarray(truth_x, dtype=np.float64),
        truth_u=truth_u / float(temperature),
        prior_theta=prior_theta,
        prior_x=prior_x,
    )
    metrics["base_temperature"] = float(temperature)
    metrics.update(realnvp_saturation_metrics(prior, truth_x))
    return metrics, prior_theta, prior_x


def evaluate_sample_pair(
    *,
    truth_theta: np.ndarray,
    truth_x: np.ndarray,
    prior_theta: np.ndarray,
    prior_x: np.ndarray,
) -> dict[str, float]:
    """Evaluate already generated samples without a fitted flow."""
    return _sample_metrics(
        truth_theta=np.asarray(truth_theta, dtype=np.float64),
        truth_x=np.asarray(truth_x, dtype=np.float64),
        truth_u=np.asarray(truth_x, dtype=np.float64),
        prior_theta=np.asarray(prior_theta, dtype=np.float64),
        prior_x=np.asarray(prior_x, dtype=np.float64),
    )


def selection_payload(
    metrics: dict[str, float],
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Return a validation-only checkpoint score and hard-gate status."""
    base_std = max(float(metrics["base_std_mean"]), 1.0e-12)
    tail_excess = max(
        float(metrics["prior_normalized_tail_fraction_abs_gt5"])
        - 2.0 * float(metrics["truth_normalized_tail_fraction_abs_gt5"]),
        0.0,
    )
    score = (
        float(metrics["median_ks_normalized"])
        + 0.5 * float(metrics["max_ks_normalized"])
        + float(metrics["correlation_frobenius_physical"])
        / len(SPLINE15D_PARAMETER_NAMES)
        + abs(float(np.log(base_std)))
        + 2.0 * tail_excess
        + float(metrics["negative_z_fraction"])
        + float(metrics["negative_dust_av_fraction"])
        + float(metrics["sliced_wasserstein_normalized"])
        + float(metrics.get("scale_saturation_fraction", 0.0))
    )
    checks = {
        "median_ks": float(metrics["median_ks_normalized"])
        <= float(thresholds["max_median_ks"]),
        "max_ks": float(metrics["max_ks_normalized"])
        <= float(thresholds["max_max_ks"]),
        "correlation": float(metrics["correlation_frobenius_physical"])
        <= float(thresholds["max_correlation_frobenius"]),
        "base_std_low": float(metrics["base_std_mean"])
        >= float(thresholds["min_base_std_mean"]),
        "base_std_high": float(metrics["base_std_mean"])
        <= float(thresholds["max_base_std_mean"]),
        "normalized_tail": float(metrics["prior_normalized_tail_fraction_abs_gt5"])
        <= float(thresholds["max_normalized_tail_fraction"]),
        "negative_z": float(metrics["negative_z_fraction"])
        <= float(thresholds["max_negative_fraction"]),
        "negative_dust_av": float(metrics["negative_dust_av_fraction"])
        <= float(thresholds["max_negative_fraction"]),
        "scale_saturation": float(metrics.get("scale_saturation_fraction", 0.0))
        <= float(thresholds.get("max_scale_saturation_fraction", 1.0)),
        "sliced_wasserstein": float(metrics["sliced_wasserstein_normalized"])
        <= float(thresholds.get("max_sliced_wasserstein", float("inf"))),
    }
    return {
        "metric": float(score),
        "eligible": bool(all(checks.values())),
        "selection_checks_passed": int(sum(checks.values())),
        "selection_checks_total": int(len(checks)),
        **metrics,
    }


def temperature_scan_frame(
    prior,
    *,
    truth_theta: np.ndarray,
    truth_x: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    temperatures: np.ndarray,
    sample_count: int,
    seed: int,
    thresholds: dict[str, float],
    whitening: dict[str, Any] | None = None,
    atom_half_width: float | None = None,
) -> pd.DataFrame:
    """Scan base temperature using validation data only."""
    rows = []
    for temperature in np.asarray(temperatures, dtype=float):
        metrics, _prior_theta, _prior_x = evaluate_generated_prior(
            prior,
            truth_theta=truth_theta,
            truth_x=truth_x,
            transforms=transforms,
            sample_count=sample_count,
            seed=seed,
            temperature=float(temperature),
            whitening=whitening,
            atom_half_width=atom_half_width,
        )
        rows.append(selection_payload(metrics, thresholds=thresholds))
    return pd.DataFrame(rows).sort_values("base_temperature").reset_index(drop=True)


def select_temperature(scan: pd.DataFrame) -> pd.Series:
    """Select the best eligible temperature, with score fallback."""
    eligible = scan.loc[scan["eligible"].astype(bool)]
    candidates = eligible if len(eligible) else scan
    return candidates.sort_values("metric").iloc[0]


def plot_truth_prior_physical_normalized(
    *,
    truth_theta: np.ndarray,
    prior_theta: np.ndarray,
    truth_x: np.ndarray,
    prior_x: np.ndarray,
    path: str,
    title_suffix: str = "",
) -> None:
    """Plot truth/prior overlays in physical and normalized spaces."""
    fig, axes = plt.subplots(len(SPLINE15D_PARAMETER_NAMES), 2, figsize=(13, 44))
    normal_grid = np.linspace(-5.0, 5.0, 500)
    normal_pdf = np.exp(-0.5 * normal_grid**2) / np.sqrt(2.0 * np.pi)
    for index, name in enumerate(SPLINE15D_PARAMETER_NAMES):
        physical_axis, normalized_axis = axes[index]
        _shared_histogram(
            physical_axis,
            truth_theta[:, index],
            prior_theta[:, index],
        )
        physical_axis.set_title(f"{name} - physical", fontsize=9)
        _shared_histogram(
            normalized_axis,
            truth_x[:, index],
            prior_x[:, index],
        )
        normalized_axis.plot(
            normal_grid,
            normal_pdf,
            color="black",
            lw=1.0,
            alpha=0.75,
            label="N(0,1)",
        )
        normalized_axis.set_xlim(-6.0, 6.0)
        ks = _ks_distance(truth_x[:, index], prior_x[:, index])
        normalized_axis.set_title(
            f"{name} - normalized | KS={ks:.3f} | "
            f"std truth/prior={np.std(truth_x[:, index]):.2f}/"
            f"{np.std(prior_x[:, index]):.2f}",
            fontsize=9,
        )
        if index == 0:
            physical_axis.legend(fontsize=8)
            normalized_axis.legend(fontsize=8)
    suffix = f" - {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Spline-15D truth versus learned RealNVP prior{suffix}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _sample_metrics(
    *,
    truth_theta: np.ndarray,
    truth_x: np.ndarray,
    truth_u: np.ndarray,
    prior_theta: np.ndarray,
    prior_x: np.ndarray,
) -> dict[str, float]:
    finite = np.isfinite(prior_theta).all(axis=1) & np.isfinite(prior_x).all(axis=1)
    if not np.any(finite):
        return _failed_metrics()
    prior_theta = prior_theta[finite]
    prior_x = prior_x[finite]
    ks = np.asarray(
        [
            _ks_distance(truth_x[:, index], prior_x[:, index])
            for index in range(truth_x.shape[1])
        ]
    )
    truth_correlation = np.corrcoef(truth_theta, rowvar=False)
    prior_correlation = np.corrcoef(prior_theta, rowvar=False)
    base_std = np.std(truth_u, axis=0)
    base_mean = np.mean(truth_u, axis=0)
    base_correlation = np.corrcoef(truth_u, rowvar=False)
    identity = np.eye(base_correlation.shape[0])
    return {
        "finite_prior_fraction": float(np.mean(finite)),
        "median_ks_normalized": float(np.median(ks)),
        "max_ks_normalized": float(np.max(ks)),
        "correlation_frobenius_physical": float(
            np.linalg.norm(prior_correlation - truth_correlation, ord="fro")
        ),
        "correlation_max_abs_physical": float(
            np.max(np.abs(prior_correlation - truth_correlation))
        ),
        "base_mean_abs_max": float(np.max(np.abs(base_mean))),
        "base_std_mean": float(np.mean(base_std)),
        "base_std_min": float(np.min(base_std)),
        "base_std_max": float(np.max(base_std)),
        "base_correlation_frobenius": float(
            np.linalg.norm(base_correlation - identity, ord="fro")
        ),
        "truth_normalized_tail_fraction_abs_gt5": float(np.mean(np.abs(truth_x) > 5.0)),
        "prior_normalized_tail_fraction_abs_gt5": float(np.mean(np.abs(prior_x) > 5.0)),
        "negative_z_fraction": float(np.mean(prior_theta[:, 0] < 0.0)),
        "negative_dust_av_fraction": float(np.mean(prior_theta[:, 3] < 0.0)),
        "sliced_wasserstein_normalized": _sliced_wasserstein(
            truth_x,
            prior_x,
        ),
    }


def _failed_metrics() -> dict[str, float]:
    return {
        "finite_prior_fraction": 0.0,
        "median_ks_normalized": 1.0,
        "max_ks_normalized": 1.0,
        "correlation_frobenius_physical": float("inf"),
        "correlation_max_abs_physical": float("inf"),
        "base_mean_abs_max": float("inf"),
        "base_std_mean": 0.0,
        "base_std_min": 0.0,
        "base_std_max": 0.0,
        "base_correlation_frobenius": float("inf"),
        "truth_normalized_tail_fraction_abs_gt5": 0.0,
        "prior_normalized_tail_fraction_abs_gt5": 1.0,
        "negative_z_fraction": 1.0,
        "negative_dust_av_fraction": 1.0,
        "sliced_wasserstein_normalized": float("inf"),
    }


def realnvp_saturation_metrics(prior, values: np.ndarray) -> dict[str, float]:
    """Measure RealNVP clamp saturation, or mark it absent for RQ splines."""
    import jax.numpy as jnp

    from euclid_dsps.amortized.flows import RQSplineCouplingPrior, _flow_permutation

    if isinstance(prior, RQSplineCouplingPrior):
        return {
            "scale_saturation_fraction": 0.0,
            "shift_saturation_fraction": 0.0,
        }

    value = jnp.asarray(values, dtype=jnp.float32)
    scale_flags = []
    shift_flags = []
    for index, layer in reversed(tuple(enumerate(prior.layers))):
        permutation = _flow_permutation(prior.latent_dim, index, prior.permutation)
        value = jnp.take(value, jnp.argsort(permutation), axis=-1)
        mask = layer.mask.astype(value.dtype)
        active = np.asarray(1.0 - mask, dtype=bool)
        log_scale, shift = layer._scale_shift(value * mask)
        log_scale = np.asarray(jax.device_get(log_scale))[:, active]
        shift = np.asarray(jax.device_get(shift))[:, active]
        scale_flags.append(
            (np.abs(log_scale) > 0.9 * float(layer.scale_clamp)).reshape(-1)
        )
        shift_flags.append(
            (np.abs(shift) > 0.9 * float(layer.shift_clamp)).reshape(-1)
        )
        value, _logdet = layer.inverse(value)
    return {
        "scale_saturation_fraction": float(np.mean(np.concatenate(scale_flags))),
        "shift_saturation_fraction": float(np.mean(np.concatenate(shift_flags))),
    }


def _sliced_wasserstein(
    truth: np.ndarray,
    prior: np.ndarray,
    *,
    n_projections: int = 64,
) -> float:
    rng = np.random.default_rng(1729)
    directions = rng.normal(size=(truth.shape[1], int(n_projections)))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    truth_projection = np.asarray(truth) @ directions
    prior_projection = np.asarray(prior) @ directions
    probabilities = np.linspace(0.005, 0.995, 200)
    truth_quantiles = np.quantile(truth_projection, probabilities, axis=0)
    prior_quantiles = np.quantile(prior_projection, probabilities, axis=0)
    return float(np.mean(np.abs(truth_quantiles - prior_quantiles)))


def _shared_histogram(axis, truth: np.ndarray, prior: np.ndarray) -> None:
    finite_truth = np.asarray(truth)[np.isfinite(truth)]
    finite_prior = np.asarray(prior)[np.isfinite(prior)]
    pooled = np.concatenate((finite_truth, finite_prior))
    low, high = np.quantile(pooled, [0.002, 0.998])
    bins = np.linspace(low, high, 65) if high > low else 30
    axis.hist(finite_truth, bins=bins, density=True, alpha=0.45, label="truth")
    axis.hist(
        finite_prior,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.5,
        label="RealNVP",
    )


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    pooled = np.concatenate((left, right))
    left_cdf = np.searchsorted(left, pooled, side="right") / max(len(left), 1)
    right_cdf = np.searchsorted(right, pooled, side="right") / max(len(right), 1)
    return float(np.max(np.abs(left_cdf - right_cdf)))
