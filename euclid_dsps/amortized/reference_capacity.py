"""Read-only finite-mixture capacity diagnostics, not a production population fit.

For unit direction d, component j has analytic CDF
F_j(t) = sum_l A_lj Phi((t - d.anchor_l) / sqrt(sum_k d_k^2 h_lk^2)).
The LP minimizes max_m |sum_j u_j F_j(t_m) - F_train(t_m)|.
This certifies only the chosen projected-CDF features, NOT optimal SW or full
15D representability. No classifier or approximate posterior enters this fit.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.special import ndtr
from scipy.stats import wasserstein_distance

from .native_reference import kernel_bandwidths


def cdf_design(basis, train, random_directions=16, thresholds=25, seed=0):
    x = np.asarray(train, dtype=float)
    anchors = np.asarray(basis["anchors"], dtype=float)[:, :5]
    a = np.asarray(basis["conditional"], dtype=float)
    h = kernel_bandwidths(basis)[:, :5]
    if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all():
        raise ValueError("Finite 5D train coordinates required")
    if not np.isfinite(anchors).all() or not np.isfinite(a).all():
        raise ValueError("Invalid reference kernels")
    if a.shape[0] != len(anchors) or np.any(a < 0) or not np.allclose(a.sum(0), 1):
        raise ValueError("Components must be normalized")
    if random_directions < 0 or thresholds < 3:
        raise ValueError("Invalid feature grid")
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(random_directions, 5))
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    directions = np.vstack([np.eye(5), random])
    grid, design, target = [], [], []
    for direction in directions:
        projected = x @ direction
        t = np.quantile(projected, np.linspace(0.01, 0.99, thresholds))
        projected_std = np.sqrt(h**2 @ direction**2)
        values = ndtr((t[:, None] - (anchors @ direction)[None]) / projected_std) @ a
        design.append(np.clip(values, 0, 1))  # Roundoff in normalized CDF sums only.
        target.append(np.searchsorted(np.sort(projected), t, side="right") / len(x))
        grid.append(t)
    return np.vstack(design), np.concatenate(target), directions, np.asarray(grid)


def empirical_features(x, directions, grid):
    return np.concatenate(
        [
            np.searchsorted(np.sort(x @ d), t, side="right") / len(x)
            for d, t in zip(directions, grid, strict=True)
        ]
    )


def fit_cdf_weights(design, target):
    f, y = np.asarray(design), np.asarray(target)
    if (
        f.ndim != 2
        or y.shape != (len(f),)
        or not np.isfinite(f).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Finite compatible CDF features required")
    if np.any((f < 0) | (f > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("CDF values must be probabilities")
    k = f.shape[1]
    result = linprog(
        np.r_[np.zeros(k), 1.0],
        A_ub=np.vstack([np.c_[f, -np.ones(len(f))], np.c_[-f, -np.ones(len(f))]]),
        b_ub=np.r_[y, -y],
        A_eq=np.array([np.r_[np.ones(k), 0.0]]),
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"CDF capacity LP failed: {result.message}")
    u = result.x[:k]
    gap = float(np.max(np.abs(f @ u - y)))
    if abs(u.sum() - 1) > 1e-7 or np.any(u < -1e-9) or gap > result.fun + 1e-7:
        raise RuntimeError("CDF capacity LP failed primal checks")
    return u, dict(
        train_cdf_max_error=gap,
        lp_objective=float(result.fun),
        solver_message=result.message,
        truth_used_for_training=True,
        production_prior_modified=False,
    )


def observable_tail_metrics(predicted, truth, weights):
    p, t, w = (np.asarray(v, dtype=float) for v in (predicted, truth, weights))
    if p.ndim != 1 or t.ndim != 1 or w.shape != p.shape or not len(t):
        raise ValueError("One-dimensional fluxes and compatible weights required")
    if (
        not all(np.isfinite(v).all() for v in (p, t, w))
        or np.any(w < 0)
        or w.sum() <= 0
    ):
        raise ValueError("Finite fluxes and positive weight mass required")
    w = w / w.sum()
    scale = max(float(np.subtract(*np.quantile(t, [0.75, 0.25]))), 1e-30)
    threshold = float(np.quantile(t, 0.999))
    tail = p > threshold
    raw = float(wasserstein_distance(p, t, u_weights=w) / scale)
    # Upper winsorization is a diagnostic only: no data/target/weights are changed.
    capped = float(
        wasserstein_distance(
            np.minimum(p, threshold), np.minimum(t, threshold), u_weights=w
        )
        / scale
    )
    return dict(
        raw_w1_over_iqr=raw,
        upper_capped_diagnostic_w1_over_iqr=capped,
        asinh_w1=float(
            wasserstein_distance(
                np.arcsinh(p / scale), np.arcsinh(t / scale), u_weights=w
            )
        ),
        truth_flux_iqr=scale,
        truth_q999=threshold,
        predicted_mass_above_truth_q999=float(w[tail].sum()),
        truth_mass_above_q999=float(np.mean(t > threshold)),
        predicted_upper_excess_over_iqr=float(w @ np.maximum(p - threshold, 0) / scale),
        truth_upper_excess_over_iqr=float(
            np.mean(np.maximum(t - threshold, 0)) / scale
        ),
        effective_rows=float(1 / (w @ w)),
    )
