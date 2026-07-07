"""Small OpenUniverse validation metrics used by CLI reports and tests."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


def compute_photoz_metrics(
    z_samples: np.ndarray,
    z_truth: np.ndarray,
) -> dict[str, float | int]:
    """Compute standard photo-z metrics from posterior redshift samples."""
    samples = np.asarray(z_samples, dtype=float)
    truth = np.asarray(z_truth, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError(f"z_samples must be [K,N] or [N], got {samples.shape}")
    if samples.shape[1] != truth.shape[0]:
        raise ValueError(
            f"z_samples object dimension {samples.shape[1]} does not match "
            f"z_truth length {truth.shape[0]}"
        )
    z_pred = np.nanmedian(samples, axis=0)
    valid = np.isfinite(z_pred) & np.isfinite(truth)
    if not valid.any():
        raise ValueError("No finite redshift samples/truth values available")
    delta_z = (z_pred[valid] - truth[valid]) / (1.0 + truth[valid])
    median = float(np.nanmedian(delta_z))
    sigma_mad = float(1.4826 * np.nanmedian(np.abs(delta_z - median)))
    pit = redshift_pit(samples[:, valid], truth[valid])
    return {
        "n_objects": int(valid.sum()),
        "median_delta_z": median,
        "sigma_mad": sigma_mad,
        "outlier_fraction_015": float(np.mean(np.abs(delta_z) > 0.15)),
        "rmse_delta_z": float(np.sqrt(np.nanmean(delta_z**2))),
        "coverage_68": central_interval_coverage(samples[:, valid], truth[valid], 0.68),
        "coverage_95": central_interval_coverage(samples[:, valid], truth[valid], 0.95),
        "pit_ks_stat": ks_uniform_statistic(pit),
    }


def redshift_pit(z_samples: np.ndarray, z_truth: np.ndarray) -> np.ndarray:
    """Return PIT values ``P(z_sample < z_true)`` per object."""
    samples = np.asarray(z_samples, dtype=float)
    truth = np.asarray(z_truth, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"z_samples must be [K,N], got {samples.shape}")
    if samples.shape[1] != truth.shape[0]:
        raise ValueError("z_samples and z_truth object counts differ")
    return np.mean(samples < truth[None, :], axis=0)


def central_interval_coverage(
    samples: np.ndarray,
    truth: np.ndarray,
    probability: float,
) -> float:
    """Return fraction of truth values inside the central posterior interval."""
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be in (0, 1)")
    lower_q = 0.5 * (1.0 - probability)
    upper_q = 1.0 - lower_q
    lower = np.nanquantile(samples, lower_q, axis=0)
    upper = np.nanquantile(samples, upper_q, axis=0)
    truth = np.asarray(truth, dtype=float)
    valid = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(truth)
    if not valid.any():
        return float("nan")
    return float(
        np.mean((truth[valid] >= lower[valid]) & (truth[valid] <= upper[valid]))
    )


def ks_uniform_statistic(values: np.ndarray) -> float:
    """One-sample Kolmogorov-Smirnov statistic against U(0,1)."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    n = x.size
    cdf_upper = np.arange(1, n + 1, dtype=float) / n
    cdf_lower = np.arange(0, n, dtype=float) / n
    return float(np.max(np.maximum(cdf_upper - x, x - cdf_lower)))


def compute_prior_overlap_metrics(
    truth_values: np.ndarray,
    posterior_values: np.ndarray,
    prior_values: np.ndarray,
    *,
    name: str = "parameter",
) -> dict[str, Any]:
    """Compare truth, aggregate posterior, and learned-prior 1D samples."""
    truth = _finite_1d(truth_values)
    posterior = _finite_1d(posterior_values)
    prior = _finite_1d(prior_values)
    if truth.size == 0 or posterior.size == 0 or prior.size == 0:
        raise ValueError("truth, posterior, and prior samples must be non-empty")
    lower68, upper68 = np.nanquantile(prior, [0.16, 0.84])
    lower95, upper95 = np.nanquantile(prior, [0.025, 0.975])
    payload = {
        "parameter": name,
        "n_truth": int(truth.size),
        "n_posterior": int(posterior.size),
        "n_prior": int(prior.size),
        "ks_truth_posterior": ks_2sample_statistic(truth, posterior),
        "ks_truth_prior": ks_2sample_statistic(truth, prior),
        "truth_within_prior_68": float(
            np.mean((truth >= lower68) & (truth <= upper68))
        ),
        "truth_within_prior_95": float(
            np.mean((truth >= lower95) & (truth <= upper95))
        ),
        "truth_outside_prior_95": float(np.mean((truth < lower95) | (truth > upper95))),
    }
    if importlib.util.find_spec("scipy") is not None:
        from scipy.stats import wasserstein_distance

        payload["wasserstein_truth_posterior"] = float(
            wasserstein_distance(truth, posterior)
        )
        payload["wasserstein_truth_prior"] = float(wasserstein_distance(truth, prior))
    else:
        payload["wasserstein_truth_posterior"] = None
        payload["wasserstein_truth_prior"] = None
    return payload


def ks_2sample_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic without requiring scipy."""
    x = _finite_1d(a)
    y = _finite_1d(b)
    if x.size == 0 or y.size == 0:
        return float("nan")
    values = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(np.sort(x), values, side="right") / x.size
    cdf_y = np.searchsorted(np.sort(y), values, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _finite_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).ravel()
    return out[np.isfinite(out)]
