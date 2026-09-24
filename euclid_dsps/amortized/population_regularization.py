"""Regularized selected-mixture fits for population-identifiability audits."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar

from euclid_dsps.amortized.forward_population import simplex


def fit_selected_weights_kl(
    log_classifier,
    frequencies,
    *,
    strength: float,
    alpha=None,
    eligible=None,
    weak_parent_mass: float = 0.05,
    observation_weights=None,
    tolerance: float = 2e-6,
    maxiter: int = 4000,
    polish_maxiter: int = 2000,
    weight_floor: float = 1e-12,
):
    """Fit selected weights with ``strength * KL(v || c)`` regularization.

    ``c`` is the selected reference frequency. Shrinking ``v`` toward ``c``
    therefore shrinks the selection-corrected parent toward its broad reference
    parent without using truth. The likelihood remains convex in ``v``.
    """
    logc = np.asarray(log_classifier, np.float64)
    c = simplex(frequencies)
    if strength <= 0 or not np.isfinite(strength):
        raise ValueError("strength must be finite and positive")
    if not 0 <= weight_floor < 1 / len(c):
        raise ValueError("invalid weight floor")
    if (
        logc.ndim != 2
        or logc.shape[1] != len(c)
        or np.any(c <= 0)
        or not np.isfinite(logc).all()
    ):
        raise ValueError("finite log classifier and positive frequencies required")

    logd = logc - np.log(c)
    logd -= np.max(logd, axis=1, keepdims=True)
    d = np.exp(np.maximum(logd, -700))
    ow = simplex(
        np.ones(len(d)) if observation_weights is None else observation_weights
    )
    if len(ow) != len(d):
        raise ValueError("observation weights mismatch")

    constraint = None
    if eligible is not None and not np.all(eligible):
        a = np.asarray(alpha, dtype=float)
        if np.any(a <= 0) or not np.any(eligible) or not 0 <= weak_parent_mass < 1:
            raise ValueError("unidentified selection support")
        constraint = (~np.asarray(eligible, bool) - weak_parent_mass) / a
        constraint /= max(np.abs(constraint).max(), 1.0)

    def objective(v):
        denom = np.maximum(d @ v, 1e-300)
        likelihood = -np.sum(ow * np.log(denom))
        ratio = np.maximum(v, weight_floor) / c
        penalty = strength * np.sum(v * np.log(ratio))
        gradient = -(d.T @ (ow / denom)) + strength * (np.log(ratio) + 1)
        return float(likelihood + penalty), gradient

    constraints = [
        dict(type="eq", fun=lambda v: v.sum() - 1, jac=lambda v: np.ones_like(v))
    ]
    start = c.copy()
    if constraint is not None:
        constraints.append(
            dict(type="ineq", fun=lambda v: -constraint @ v, jac=lambda v: -constraint)
        )
        start = simplex(c * np.asarray(eligible) + weight_floor)
    bounds = [(weight_floor, 1.0)] * len(c)
    result = minimize(
        objective,
        start,
        jac=True,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=maxiter),
    )
    v = simplex(np.maximum(result.x, weight_floor))

    def certificate(weights):
        loss, gradient = objective(weights)
        vertex = linprog(
            gradient,
            A_ub=None if constraint is None else constraint[None],
            b_ub=None if constraint is None else [0.0],
            A_eq=np.ones((1, len(c))),
            b_eq=[1.0],
            bounds=bounds,
            method="highs",
        )
        if not vertex.success:
            raise RuntimeError("regularized population KKT oracle failed")
        return loss, float(gradient @ (weights - vertex.x)), vertex.x

    loss, gap, vertex = certificate(v)
    initial_gap = gap
    restart_iterations = 0
    if gap > tolerance:

        def scaled_objective(weights):
            value, gradient = objective(weights)
            return 1000 * value, 1000 * gradient

        restart = minimize(
            scaled_objective,
            v,
            jac=True,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options=dict(ftol=1e-12, maxiter=maxiter),
        )
        restart_iterations = int(restart.nit)
        candidate = simplex(np.maximum(restart.x, weight_floor))
        candidate_loss, candidate_gap, candidate_vertex = certificate(candidate)
        if (
            constraint is None or constraint @ candidate <= 1e-8
        ) and candidate_loss <= loss:
            v, loss, gap, vertex = (
                candidate,
                candidate_loss,
                candidate_gap,
                candidate_vertex,
            )

    polish_iterations = 0
    while gap > tolerance and polish_iterations < polish_maxiter:
        direction = vertex - v
        search = minimize_scalar(
            lambda step, current=v, delta=direction: objective(current + step * delta)[
                0
            ],
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        candidate = simplex(np.maximum(v + float(search.x) * direction, weight_floor))
        candidate_loss, candidate_gap, candidate_vertex = certificate(candidate)
        if candidate_loss > loss + 1e-12 or np.array_equal(candidate, v):
            break
        v, loss, gap, vertex = (
            candidate,
            candidate_loss,
            candidate_gap,
            candidate_vertex,
        )
        polish_iterations += 1

    feasible = constraint is None or constraint @ v <= 1e-8
    if gap > tolerance or not feasible or not np.isfinite(loss):
        raise RuntimeError(
            f"regularized population solver not converged: gap={gap}, {result.message}"
        )
    return v, {
        "strength": float(strength),
        "kkt_gap": max(gap, 0.0),
        "iterations": int(result.nit),
        "initial_kkt_gap": max(float(initial_gap), 0.0),
        "scaled_restart_iterations": restart_iterations,
        "polish_iterations": polish_iterations,
        "solver_message": str(result.message),
        "support_constraint_active": constraint is not None,
        "weak_parent_mass_cap": float(weak_parent_mass),
        "weight_floor": float(weight_floor),
    }
