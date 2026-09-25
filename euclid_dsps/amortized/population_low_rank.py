"""Smooth low-rank corrections of finite population-component weights."""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import LinearConstraint, linprog, minimize, minimize_scalar
from scipy.sparse.csgraph import connected_components

from euclid_dsps.amortized.forward_population import simplex


def _weighted_orthonormalize(vectors, weights, *, tolerance=1e-10):
    """Center and orthonormalize columns under a discrete reference measure."""
    vectors = np.asarray(vectors, dtype=np.float64)
    weights = simplex(weights)
    if vectors.ndim != 2 or vectors.shape[0] != len(weights):
        raise ValueError("mode candidates and reference weights do not match")
    result = []
    for candidate in vectors.T:
        vector = candidate - np.dot(weights, candidate)
        for previous in result:
            vector -= np.dot(weights, vector * previous) * previous
        norm = np.sqrt(np.dot(weights, vector * vector))
        if norm > tolerance:
            result.append(vector / norm)
    if not result:
        raise ValueError("no nonconstant spectral modes survived")
    return np.column_stack(result)


def build_spectral_modes(
    embedding,
    reference_weights,
    *,
    neighbors=12,
    maximum_rank=64,
    tail_component=0,
):
    """Build truth-free smooth modes on a component-similarity graph.

    The broad tail receives one explicit contrast. Remaining modes are ordered
    eigenvectors of the unnormalized graph Laplacian on non-tail components.
    Every returned mode has zero mean and unit variance under the selected
    reference weights, so ``reference * (1 + modes @ coefficients)`` remains
    normalized before enforcing positivity.
    """
    embedding = np.asarray(embedding, dtype=np.float64)
    reference_weights = simplex(reference_weights)
    if embedding.ndim != 2 or len(embedding) != len(reference_weights):
        raise ValueError("embedding and reference weights do not match")
    components = len(reference_weights)
    if not 0 <= tail_component < components:
        raise ValueError("invalid tail component")
    retained = np.flatnonzero(np.arange(components) != tail_component)
    if not 1 <= neighbors < len(retained):
        raise ValueError("neighbors must be between 1 and non-tail components - 1")
    if not 1 <= maximum_rank < components:
        raise ValueError("maximum_rank must be between 1 and components - 1")

    points = embedding[retained]
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    neighbor_distances = np.take_along_axis(distances, nearest, axis=1)
    positive = neighbor_distances[
        np.isfinite(neighbor_distances) & (neighbor_distances > 0)
    ]
    if not len(positive):
        raise ValueError("component embedding has no positive neighbor distances")
    bandwidth = float(np.median(positive))
    affinity = np.zeros_like(distances)
    rows = np.repeat(np.arange(len(retained)), neighbors)
    columns = nearest.ravel()
    affinity[rows, columns] = np.exp(-0.5 * (distances[rows, columns] / bandwidth) ** 2)
    affinity = np.maximum(affinity, affinity.T)
    np.fill_diagonal(affinity, 0.0)
    graph_components = int(connected_components(affinity, directed=False)[0])
    if graph_components != 1:
        raise ValueError(
            f"component graph is disconnected ({graph_components} components)"
        )

    laplacian = np.diag(affinity.sum(axis=1)) - affinity
    eigenvalues, eigenvectors = eigh(laplacian, check_finite=True)
    spectral_count = min(maximum_rank - 1, len(retained) - 1)
    candidates = []
    tail = np.zeros(components, dtype=np.float64)
    tail[tail_component] = 1.0
    candidates.append(tail)
    for index in range(1, spectral_count + 1):
        mode = np.zeros(components, dtype=np.float64)
        mode[retained] = eigenvectors[:, index]
        candidates.append(mode)
    modes = _weighted_orthonormalize(np.column_stack(candidates), reference_weights)
    if modes.shape[1] < maximum_rank:
        raise RuntimeError(
            f"requested {maximum_rank} modes but constructed {modes.shape[1]}"
        )
    return modes[:, :maximum_rank], {
        "bandwidth": bandwidth,
        "neighbors": int(neighbors),
        "graph_components": graph_components,
        "laplacian_eigenvalues": eigenvalues,
    }


def low_rank_selected_weights(reference, modes, coefficients):
    """Return the affine selected-population correction for coefficients."""
    reference = simplex(reference)
    modes = np.asarray(modes, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if modes.shape != (len(reference), len(coefficients)):
        raise ValueError("low-rank mode and coefficient shapes do not match")
    weights = reference * (1.0 + modes @ coefficients)
    if not np.isfinite(weights).all():
        raise ValueError("non-finite low-rank selected weights")
    return weights


def fit_low_rank_selected_weights(
    log_probability,
    reference,
    modes,
    *,
    strength=0.0,
    observation_weights=None,
    weight_floor=1e-10,
    tolerance=2e-7,
    maximum_iterations=2000,
    initial_coefficients=None,
    maximum_seconds=None,
):
    """Fit a convex smooth correction in a fixed low-rank affine subspace.

    For selected-reference weights ``c`` and centered modes ``Phi``, this fits

    ``v = c * (1 + Phi @ gamma)``

    by maximizing the selected mixture likelihood with ``v >= weight_floor``.
    The objective is concave in ``gamma``; the optional quadratic penalty is
    convex after sign reversal. Parent weights are deliberately not returned
    here because the caller must apply the explicit selection correction.
    """
    log_probability = np.asarray(log_probability, dtype=np.float64)
    reference = simplex(reference)
    modes = np.asarray(modes, dtype=np.float64)
    if log_probability.ndim != 2 or log_probability.shape[1] != len(reference):
        raise ValueError("classifier probabilities and reference do not match")
    if modes.ndim != 2 or modes.shape[0] != len(reference) or modes.shape[1] < 1:
        raise ValueError("at least one matching low-rank mode is required")
    if not np.isfinite(log_probability).all() or not np.isfinite(modes).all():
        raise ValueError("finite classifier probabilities and modes required")
    if strength < 0 or not np.isfinite(strength):
        raise ValueError("strength must be finite and nonnegative")
    if not 0 <= weight_floor < reference.min():
        raise ValueError("weight_floor must be below every reference weight")
    centered_error = np.max(np.abs(reference @ modes))
    if centered_error > 1e-8:
        raise ValueError(f"modes are not reference-centered: {centered_error}")

    if observation_weights is None:
        observations = np.ones(len(log_probability), dtype=np.float64)
    else:
        observations = np.asarray(observation_weights, dtype=np.float64)
        if observations.shape != (len(log_probability),) or np.any(
            ~np.isfinite(observations) | (observations < 0)
        ):
            raise ValueError("invalid observation weights")
    if observations.sum() <= 0:
        raise ValueError("positive total observation weight required")
    observations /= observations.sum()

    log_ratio = log_probability - np.log(reference)[None, :]
    correction = reference[:, None] * modes
    rank = modes.shape[1]

    # exp(log_ratio - row_max) @ (c + correction @ gamma) is affine in gamma.
    # Cache the projection once instead of repeating NxJ log/exp operations at
    # every optimizer iteration. Row offsets preserve the original objective.
    row_offset = log_ratio.max(axis=1)
    scaled_ratio = np.exp(log_ratio - row_offset[:, None])
    baseline = scaled_ratio @ reference
    projected = scaled_ratio @ correction
    started = time.monotonic()
    if maximum_seconds is not None and (
        not np.isfinite(maximum_seconds) or maximum_seconds <= 0
    ):
        raise ValueError("maximum_seconds must be positive and finite")

    def terms(coefficients):
        if maximum_seconds is not None and time.monotonic() - started > maximum_seconds:
            raise TimeoutError("low-rank solver reached its per-fit time budget")
        selected = low_rank_selected_weights(reference, modes, coefficients)
        values = baseline + projected @ coefficients
        # Close to a boundary, avoid cancellation between the affine terms.
        sensitive = values < 1e-8 * baseline
        if np.any(sensitive):
            values[sensitive] = scaled_ratio[sensitive] @ selected
        return selected, values

    def objective(coefficients):
        selected, mixture = terms(coefficients)
        if np.any(selected <= 0) or np.any(mixture <= 0):
            return np.inf
        values = np.log(mixture) + row_offset
        penalty = 0.5 * strength * np.mean(coefficients * coefficients)
        return -float(observations @ values) + penalty

    def gradient(coefficients):
        _, mixture = terms(coefficients)
        result = -(projected.T @ (observations / mixture))
        if strength:
            result += strength * coefficients / rank
        return result

    constraint = LinearConstraint(
        correction,
        weight_floor - reference,
        np.full(len(reference), np.inf),
    )
    if initial_coefficients is None:
        initial = np.zeros(rank, dtype=np.float64)
    else:
        initial = np.asarray(initial_coefficients, dtype=np.float64)
        if initial.shape != (rank,) or not np.isfinite(initial).all():
            raise ValueError("invalid initial low-rank coefficients")
        if np.any(
            low_rank_selected_weights(reference, modes, initial) < weight_floor - 1e-12
        ):
            raise ValueError("initial low-rank coefficients are infeasible")
    result = minimize(
        objective,
        initial,
        jac=gradient,
        constraints=(constraint,),
        method="SLSQP",
        options={
            "ftol": min(float(tolerance) * 0.1, 1e-10),
            "maxiter": int(maximum_iterations),
            "disp": False,
        },
    )
    coefficients = np.asarray(result.x, dtype=np.float64)

    def certificate(current):
        grad = gradient(current)
        vertex = linprog(
            grad,
            A_ub=-correction,
            b_ub=reference - weight_floor,
            bounds=(None, None),
            method="highs",
        )
        if not vertex.success:
            raise RuntimeError(f"low-rank KKT oracle failed: {vertex.message}")
        return float(grad @ (current - vertex.x)), vertex.x, grad

    kkt_gap, vertex, grad = certificate(coefficients)
    polish_iterations = 0
    while kkt_gap > tolerance and polish_iterations < 200:
        direction = vertex - coefficients
        line = minimize_scalar(
            lambda step, start=coefficients, delta=direction: objective(
                start + step * delta
            ),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not line.success or line.x <= 1e-12:
            break
        coefficients = coefficients + float(line.x) * direction
        kkt_gap, vertex, grad = certificate(coefficients)
        polish_iterations += 1

    selected = low_rank_selected_weights(reference, modes, coefficients)
    primal_violation = float(max(0.0, weight_floor - selected.min()))
    kkt_gap = max(primal_violation, kkt_gap)
    active = selected <= weight_floor + max(1e-8, 10 * tolerance)
    if not result.success or kkt_gap > tolerance:
        raise RuntimeError(
            "low-rank population solver not converged: "
            f"success={result.success} KKT={kkt_gap:.3g} {result.message}"
        )
    selected = simplex(np.maximum(selected, weight_floor))
    return (
        selected,
        coefficients,
        {
            "solver_message": str(result.message),
            "iterations": int(result.nit),
            "polish_iterations": int(polish_iterations),
            "kkt_gap": float(kkt_gap),
            "gradient_infinity_norm": float(np.max(np.abs(grad))),
            "active_weight_fraction": float(active.mean()),
            "minimum_selected_weight": float(selected.min()),
            "coefficient_l2": float(np.linalg.norm(coefficients)),
            "penalty_strength": float(strength),
            "objective": float(objective(coefficients)),
        },
    )
