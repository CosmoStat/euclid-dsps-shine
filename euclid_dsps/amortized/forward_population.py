"""Posterior-free selected-mixture likelihood and normalized 15D reference basis.

This module deliberately has no dependency on an encoder, RWS or posterior bank.
All densities below are in the repository's invertible latent-x coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
from scipy.stats import norm, qmc

PHYSICAL = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)


def simplex(value):
    value = np.asarray(value, dtype=np.float64)
    if (
        value.ndim != 1
        or not np.isfinite(value).all()
        or np.any(value < 0)
        or value.sum() <= 0
    ):
        raise ValueError("finite nonnegative nonempty weights required")
    return value / value.sum()


@dataclass(frozen=True)
class PhysicalBasis:
    """Analytically normalized overlapping Gaussians, fixed N(0,I) SFH reference.

    g_j(x)=N(x_a; mu_j, sigma_j^2 I) N(x_b;0,I).
    p0=mean_j g_j and p_u/p0 depends ONLY on x_a. Coordinate transforms are
    elementwise, so this is also a correction depending only on physical a.
    The identity-reference conditional b|a is intentionally not learned.
    """

    names: tuple[str, ...]
    centers: np.ndarray
    scales: np.ndarray

    @property
    def indices(self):
        return np.array([self.names.index(name) for name in PHYSICAL])

    @property
    def components(self):
        return len(self.centers)

    @classmethod
    def create(cls, names, components=128, seed=20260920, width=0.7, extent=1.5):
        if len(names) != 15 or not set(PHYSICAL).issubset(names):
            raise ValueError("15 coordinates including all physical names required")
        if components < 4 or width <= 0 or extent <= 0:
            raise ValueError("invalid reference basis geometry")
        exponent = int(np.ceil(np.log2(components - 1)))
        unit = qmc.Sobol(5, scramble=True, seed=seed).random_base2(exponent)[
            : components - 1
        ]
        centers = np.vstack([np.zeros(5), extent * norm.ppf(np.clip(unit, 0.01, 0.99))])
        scales = np.full((components, 5), width)
        scales[0] = 2.5  # Explicit broad tail component, not a q-derived component.
        return cls(tuple(names), centers, scales)

    def sample(self, rng, count, weights=None, labels=None):
        if labels is None:
            labels = rng.choice(
                self.components,
                count,
                p=simplex(np.ones(self.components) if weights is None else weights),
            )
        labels = np.asarray(labels, dtype=int)
        if labels.shape != (count,) or np.any(
            (labels < 0) | (labels >= self.components)
        ):
            raise ValueError("invalid component labels")
        x = rng.normal(size=(count, len(self.names)))
        x[:, self.indices] = (
            self.centers[labels] + x[:, self.indices] * self.scales[labels]
        )
        return x, labels

    def component_log_prob(self, x):
        x = np.asarray(x, dtype=np.float64)
        a = x[..., self.indices]
        nuisance = np.delete(x, self.indices, axis=-1)
        physical = -0.5 * np.sum(
            ((a[..., None, :] - self.centers) / self.scales) ** 2
            + 2 * np.log(self.scales)
            + np.log(2 * np.pi),
            axis=-1,
        )
        return (
            physical - 0.5 * np.sum(nuisance**2 + np.log(2 * np.pi), axis=-1)[..., None]
        )

    def log_prob(self, x, weights):
        w = simplex(weights)
        with np.errstate(divide="ignore"):
            return logsumexp(self.component_log_prob(x) + np.log(w), axis=-1)


def parent_from_selected(selected_weights, alpha):
    v = simplex(selected_weights)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != v.shape or np.any(
        ~np.isfinite(alpha) | (alpha <= 0) | (alpha > 1)
    ):
        raise ValueError("selection efficiencies must be finite in (0,1]")
    return simplex(v / alpha)


def selected_from_parent(parent_weights, alpha):
    u = simplex(parent_weights)
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != u.shape or np.any(
        ~np.isfinite(alpha) | (alpha < 0) | (alpha > 1)
    ):
        raise ValueError("invalid selection efficiencies")
    return simplex(u * alpha)


def selection_efficiencies(
    labels, selected, components, *, min_selected=128, min_alpha=0.001
):
    labels, selected = np.asarray(labels), np.asarray(selected, dtype=bool)
    n = np.bincount(labels, minlength=components).astype(float)
    k = np.bincount(labels[selected], minlength=components).astype(float)
    if np.any(n == 0):
        raise ValueError("every reference component needs simulations")
    alpha = k / n
    # Wilson lower confidence bound: do not invert an unresolved tail efficiency.
    z = 1.96
    lower = (
        alpha
        + z * z / (2 * n)
        - z * np.sqrt(alpha * (1 - alpha) / n + z * z / (4 * n * n))
    ) / (1 + z * z / n)
    eligible = (k >= min_selected) & (lower >= min_alpha)
    return dict(attempts=n, successes=k, alpha=alpha, lower95=lower, eligible=eligible)


def fit_selected_weights(
    log_classifier,
    frequencies,
    *,
    alpha=None,
    eligible=None,
    weak_parent_mass=0.05,
    observation_weights=None,
    tolerance=2e-6,
    maxiter=2000,
    polish_maxiter=10000,
):
    """Concave ML on a simplex; optional LINEAR parent-mass support constraints.

    d_ij=C_j/c_j. Weak components remain in the parent; sum_weak u_j <= cap
    becomes sum_j v_j (1_weak-cap)/alpha_j <= 0, hence remains convex.
    Return only when feasibility and a Frank-Wolfe/KKT dual gap pass.
    """
    from scipy.optimize import linprog

    logc = np.asarray(log_classifier, dtype=np.float64)
    c = simplex(frequencies)
    if (
        logc.ndim != 2
        or logc.shape[1] != len(c)
        or np.any(c <= 0)
        or not np.isfinite(logc).all()
    ):
        raise ValueError(
            "finite log classifier and positive selected frequencies required"
        )
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
            raise ValueError("unidentified selection support: enlarge reference bank")
        constraint = (~np.asarray(eligible, bool) - weak_parent_mass) / a
        constraint /= max(np.abs(constraint).max(), 1.0)

    def objective(v):
        denom = np.maximum(d @ v, 1e-300)
        return -np.sum(ow * np.log(denom)), -(d.T @ (ow / denom))

    constraints = [
        dict(type="eq", fun=lambda v: v.sum() - 1, jac=lambda v: np.ones_like(v))
    ]
    start = c.copy()
    if constraint is not None:
        constraints.append(
            dict(type="ineq", fun=lambda v: -constraint @ v, jac=lambda v: -constraint)
        )
        start = simplex(c * np.asarray(eligible))
    result = minimize(
        objective,
        start,
        jac=True,
        bounds=[(0.0, 1.0)] * len(c),
        constraints=constraints,
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=maxiter),
    )
    v = simplex(np.maximum(result.x, 0))

    def certificate(weights):
        loss, gradient = objective(weights)
        vertex = linprog(
            gradient,
            A_ub=None if constraint is None else constraint[None],
            b_ub=None if constraint is None else [0.0],
            A_eq=np.ones((1, len(c))),
            b_eq=[1.0],
            bounds=(0, 1),
            method="highs",
        )
        if not vertex.success:
            raise RuntimeError("KKT oracle failed")
        return loss, float(gradient @ (weights - vertex.x)), vertex.x

    loss, gap, vertex = certificate(v)
    initial_gap = gap
    restart_iterations = 0
    if gap > tolerance and polish_maxiter > 0:
        # SLSQP's absolute objective-change tolerance is NOT a stationarity
        # tolerance. Rescale the same objective, then certify in original units.
        def scaled_objective(weights):
            value, gradient = objective(weights)
            return 1000 * value, 1000 * gradient

        restart = minimize(
            scaled_objective,
            v,
            jac=True,
            bounds=[(0.0, 1.0)] * len(c),
            constraints=constraints,
            method="SLSQP",
            options=dict(ftol=1e-12, maxiter=maxiter),
        )
        restart_iterations = int(restart.nit)
        candidate = simplex(np.maximum(restart.x, 0))
        if (constraint is None or constraint @ candidate <= 1e-10) and objective(
            candidate
        )[0] <= loss:
            v = candidate
            loss, gap, vertex = certificate(v)

    polish_iterations = 0
    # Frank-Wolfe follows a feasible segment of the SAME polytope. Exact
    # one-dimensional line search reduces the objective without relaxing KKT
    # or the weak-parent-mass constraint, even on simplex boundary solutions.
    while gap > tolerance and polish_iterations < polish_maxiter:
        if constraint is not None and constraint @ v > 1e-8:
            break
        direction = vertex - v
        denom = np.maximum(d @ v, 1e-300)
        change = d @ direction

        def slope(step, denom=denom, change=change):
            return -float(
                np.sum(ow * change / np.maximum(denom + step * change, 1e-300))
            )

        if slope(0) >= 0:
            break
        step = 1.0 if slope(1) <= 0 else brentq(slope, 0.0, 1.0, xtol=1e-15, rtol=1e-14)
        candidate = simplex(v + step * direction)
        if np.array_equal(candidate, v):
            break
        candidate_loss, candidate_gap, candidate_vertex = certificate(candidate)
        if candidate_loss > loss + 1e-12:
            raise RuntimeError("population polishing increased objective")
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
            f"population solver not converged: KKT gap={gap}, {result.message}"
        )
    return v, dict(
        kkt_gap=max(gap, 0.0),
        mean_negative_log_likelihood_shifted=float(loss),
        iterations=int(result.nit),
        solver_message=str(result.message),
        initial_kkt_gap=max(float(initial_gap), 0.0),
        scaled_restart_iterations=restart_iterations,
        polish_iterations=polish_iterations,
        weak_parent_mass_cap=weak_parent_mass,
        support_constraint_active=bool(
            constraint is not None and abs(constraint @ v) < 1e-6
        ),
    )
