"""TRAIN-only component capacity in physical units; never population inference."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def physical_cdf_design(
    draws: np.ndarray, truth: np.ndarray, directions: int, grid_size: int, seed: int
) -> tuple:
    """Finite-grid approximation to average projected physical W1 / truth IQR.

    Grids cover the full finite draw range; no upper-tail clipping. Component
    CDFs are Monte Carlo estimates, not analytic density certificates. Evaluation
    must use different draws, and none of these targets may train production u.
    """
    draws, truth = np.asarray(draws, float), np.asarray(truth, float)
    if draws.ndim != 3 or draws.shape[2] != 5 or truth.ndim != 2 or truth.shape[1] != 5:
        raise ValueError("Require (components,draws,5) and (objects,5)")
    if min(draws.shape[:2]) < 1 or len(truth) < 2:
        raise ValueError("Nonempty components and at least two target rows required")
    if (
        not np.isfinite(draws).all()
        or not np.isfinite(truth).all()
        or directions < 1
        or grid_size < 3
    ):
        raise ValueError("Finite samples and positive feature grid required")
    scale = np.maximum(np.subtract(*np.quantile(truth, [0.75, 0.25], axis=0)), 1e-8)
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(directions, 5))
    random /= np.linalg.norm(random, axis=1, keepdims=True)
    vectors = np.r_[np.eye(5), random]
    x, t = draws / scale, truth / scale
    features, targets, weights = [], [], []
    for direction in vectors:
        projected, actual = x @ direction, np.sort(t @ direction)
        # Both target and reference quantiles resolve peaks and long tails.
        grid = np.unique(
            np.r_[
                np.quantile(projected, np.linspace(0, 1, grid_size)),
                np.quantile(actual, np.linspace(0, 1, grid_size)),
            ]
        )
        if len(grid) < 2:
            continue  # Identical point masses contribute exactly zero distance.
        middle = 0.5 * (grid[:-1] + grid[1:])
        features.append(
            np.stack(
                [
                    np.searchsorted(np.sort(p), middle, side="right") / len(p)
                    for p in projected
                ],
                axis=1,
            )
        )
        targets.append(np.searchsorted(actual, middle, side="right") / len(actual))
        weights.append(np.diff(grid) / len(vectors))
    if not features:
        raise ValueError("Degenerate physical design: all projections constant")
    return np.vstack(features), np.concatenate(targets), np.concatenate(weights)


def fit_physical_weights(
    features: np.ndarray,
    target: np.ndarray,
    quadrature: np.ndarray,
    cdf_features: np.ndarray,
    cdf_target: np.ndarray,
    cdf_limit: float,
    maximum_seconds: float = 600,
) -> tuple[np.ndarray, dict]:
    """Convex absolute-CDF-area LP with the original analytic TRAIN CDF cap.

    min sum_m delta_t_m |F_m u - target_m|; u in simplex.
    |CDF_original u - CDF_train| <= cdf_limit protects the old criterion.
    No validation/test truth enters this solver.
    """
    f, y, q, a, b = [
        np.asarray(v, float)
        for v in (features, target, quadrature, cdf_features, cdf_target)
    ]
    if (
        f.ndim != 2
        or y.shape != (len(f),)
        or q.shape != y.shape
        or a.ndim != 2
        or a.shape[1] != f.shape[1]
        or b.shape != (len(a),)
    ):
        raise ValueError("Incompatible CDF design shapes")
    if (
        not f.size
        or not a.size
        or not np.isfinite(maximum_seconds)
        or maximum_seconds <= 0
    ):
        raise ValueError("Nonempty design and positive finite solver budget required")
    if (
        not all(np.isfinite(v).all() for v in (f, y, q, a, b))
        or np.any(q <= 0)
        or cdf_limit < 0
        or not np.isfinite(cdf_limit)
    ):
        raise ValueError("Finite CDFs and positive quadrature required")
    if any(np.any((v < 0) | (v > 1)) for v in (f, y, a, b)):
        raise ValueError("CDFs must be probabilities")
    n, k = f.shape
    eye = sparse.eye(n, format="csr")
    z = sparse.csr_matrix((len(a), n))
    constraints = sparse.vstack(
        [
            sparse.hstack([f, -eye]),
            sparse.hstack([-f, -eye]),
            sparse.hstack([a, z]),
            sparse.hstack([-a, z]),
        ],
        format="csr",
    )
    result = linprog(
        np.r_[np.zeros(k), q],
        A_ub=constraints,
        b_ub=np.r_[y, -y, b + cdf_limit, -b + cdf_limit],
        A_eq=sparse.csr_matrix(np.r_[np.ones(k), np.zeros(n)][None]),
        b_eq=[1.0],
        bounds=(0, None),
        method="highs",
        options=dict(time_limit=float(maximum_seconds)),
    )
    if not result.success:
        raise RuntimeError(f"Physical capacity LP failed: {result.message}")
    u = result.x[:k]
    value = float(q @ abs(f @ u - y))
    gap = float(abs(a @ u - b).max())
    if (
        abs(u.sum() - 1) > 1e-7
        or u.min() < -1e-9
        or gap > cdf_limit + 1e-7
        or abs(value - result.fun) > 1e-7
    ):
        raise RuntimeError("Physical capacity LP failed primal checks")
    return u, dict(
        objective=value,
        train_cdf_max=gap,
        train_cdf_cap=float(cdf_limit),
        solver_message=result.message,
        iterations=int(result.nit),
        truth_diagnostic_only=True,
    )
