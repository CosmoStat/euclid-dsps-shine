"""Empirical distribution checks retaining signed biases and all tail draws."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def w1_parts(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Exact empirical W1 contributions on quantile ranges [0,.01,.99,1].

    Integrate |F^-1(p)-G^-1(p)| over the union of empirical CDF breakpoints.
    This partitions the full W1, unlike trimming and renormalizing samples.
    """
    a, b = np.sort(predicted), np.sort(truth)
    if not len(a) or not len(b) or not np.isfinite(np.r_[a, b]).all():
        raise ValueError("W1 requires nonempty finite samples")
    edges = np.unique(
        np.r_[
            np.arange(len(a) + 1) / len(a), np.arange(len(b) + 1) / len(b), 0.01, 0.99
        ]
    )
    mid = (edges[1:] + edges[:-1]) / 2
    delta = abs(
        a[np.minimum((mid * len(a)).astype(int), len(a) - 1)]
        - b[np.minimum((mid * len(b)).astype(int), len(b) - 1)]
    )
    return np.bincount(
        np.searchsorted([0.01, 0.99], mid), weights=delta * np.diff(edges), minlength=3
    )


def tail_table(predicted: np.ndarray, truth: np.ndarray, names) -> pd.DataFrame:
    predicted, truth = np.asarray(predicted), np.asarray(truth)
    if (
        predicted.ndim != 2
        or truth.ndim != 2
        or predicted.shape[1] != len(names)
        or truth.shape[1] != len(names)
        or min(len(predicted), len(truth)) < 2
        or not np.isfinite(predicted).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Expected finite sample matrices with matching columns")
    rows = []
    for k, name in enumerate(names):
        a, b = predicted[:, k], truth[:, k]
        scale = max(float(np.subtract(*np.percentile(b, [75, 25]))), 1e-8)
        parts = w1_parts(a, b) / scale
        lo, hi = np.quantile(b, [0.01, 0.99])
        row = dict(
            parameter=name,
            w1_over_iqr=float(parts.sum()),
            lower_tail_w1=float(parts[0]),
            central_w1=float(parts[1]),
            upper_tail_w1=float(parts[2]),
            signed_mean_bias_over_iqr=float((a.mean() - b.mean()) / scale),
            signed_median_bias_over_iqr=float((np.median(a) - np.median(b)) / scale),
            predicted_below_truth_q01=float(np.mean(a < lo)),
            predicted_above_truth_q99=float(np.mean(a > hi)),
            truth_zero_fraction=float(np.mean(b == 0)),
            predicted_zero_fraction=float(np.mean(a == 0)),
        )
        for label, values in (("truth", b), ("predicted", a)):
            for q in (0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1):
                row[f"{label}_q{q:g}"] = float(np.quantile(values, q))
        rows.append(row)
    return pd.DataFrame(rows)


def physical_sw(predicted: np.ndarray, truth: np.ndarray, seed: int) -> float:
    """5D SW in truth-IQR units, with fixed directions for paired comparisons."""
    a, b = np.asarray(predicted), np.asarray(truth)
    if a.shape[1] != 5 or b.shape[1] != 5:
        raise ValueError("Physical sliced Wasserstein requires exactly five columns")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite samples")
    scale = np.maximum(np.subtract(*np.percentile(b, [75, 25], axis=0)), 1e-8)
    directions = np.random.default_rng(seed).normal(size=(64, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return float(
        np.mean(
            [wasserstein_distance(a / scale @ d, b / scale @ d) for d in directions]
        )
    )
