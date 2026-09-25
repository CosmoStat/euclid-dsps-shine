import numpy as np
import pytest

from euclid_dsps.amortized.population_low_rank import (
    build_spectral_modes,
    fit_low_rank_selected_weights,
    low_rank_selected_weights,
)


def test_spectral_modes_are_reference_centered_and_weighted_orthonormal():
    rng = np.random.default_rng(11)
    embedding = rng.normal(size=(18, 6))
    reference = rng.uniform(0.2, 2.0, size=18)
    reference /= reference.sum()

    modes, diagnostics = build_spectral_modes(
        embedding,
        reference,
        neighbors=5,
        maximum_rank=8,
        tail_component=0,
    )

    assert modes.shape == (18, 8)
    np.testing.assert_allclose(reference @ modes, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        modes.T @ (reference[:, None] * modes), np.eye(8), atol=1e-10
    )
    assert diagnostics["graph_components"] == 1
    assert diagnostics["bandwidth"] > 0


def test_low_rank_convex_fit_recovers_known_selected_mixture():
    rng = np.random.default_rng(12)
    components = 9
    reference = np.full(components, 1.0 / components)
    raw = rng.normal(size=(components, 3))
    raw -= raw.mean(axis=0)
    modes, _ = np.linalg.qr(raw)
    modes -= reference @ modes
    modes /= np.sqrt(np.sum(reference[:, None] * modes**2, axis=0))
    truth_coefficients = np.array([0.07, -0.04, 0.03])
    truth = low_rank_selected_weights(reference, modes, truth_coefficients)

    labels = rng.choice(components, size=60000, p=truth)
    probability = np.full((len(labels), components), 1e-5)
    probability[np.arange(len(labels)), labels] = 1.0
    probability /= probability.sum(axis=1, keepdims=True)
    fitted, coefficients, diagnostics = fit_low_rank_selected_weights(
        np.log(probability),
        reference,
        modes,
        tolerance=2e-6,
    )

    np.testing.assert_allclose(fitted.sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(coefficients, truth_coefficients, atol=0.015)
    np.testing.assert_allclose(fitted, truth, atol=0.01)
    assert diagnostics["kkt_gap"] <= 2e-6


def test_equal_selection_efficiency_leaves_parent_equal_to_selected():
    from euclid_dsps.amortized.forward_population import parent_from_selected

    selected = np.array([0.1, 0.3, 0.6])
    parent = parent_from_selected(selected, np.full(3, 0.2))

    np.testing.assert_allclose(parent, selected)


def test_cached_solver_matches_independent_log_space_reference():
    from scipy.optimize import LinearConstraint, minimize
    from scipy.special import logsumexp

    rng = np.random.default_rng(55)
    c = np.array([0.1, 0.2, 0.3, 0.4])
    modes = rng.normal(size=(4, 2))
    modes -= c @ modes
    logc = rng.normal(scale=4, size=(500, 4))
    logc -= logsumexp(logc, axis=1, keepdims=True)
    observation_weights = rng.uniform(0.1, 2.0, len(logc))
    observation_weights /= observation_weights.sum()
    correction = c[:, None] * modes
    strength = 0.02

    def objective(gamma):
        v = c + correction @ gamma
        if (v <= 0).any():
            return np.inf
        return -observation_weights @ logsumexp(
            logc - np.log(c) + np.log(v), axis=1
        ) + 0.5 * strength * np.mean(gamma**2)

    reference = minimize(
        objective,
        np.zeros(2),
        method="SLSQP",
        constraints=[LinearConstraint(correction, 1e-10 - c, np.inf)],
        options=dict(ftol=1e-12, maxiter=1000),
    )
    v, gamma, certificate = fit_low_rank_selected_weights(
        logc,
        c,
        modes,
        strength=strength,
        observation_weights=observation_weights,
        tolerance=2e-6,
    )
    assert reference.success
    np.testing.assert_allclose(v, c + correction @ reference.x, atol=1e-5)
    np.testing.assert_allclose(certificate["objective"], objective(gamma), atol=1e-12)
    shifted, _, _ = fit_low_rank_selected_weights(
        logc - 10000.0,
        c,
        modes,
        strength=strength,
        observation_weights=observation_weights,
        tolerance=2e-6,
    )
    np.testing.assert_allclose(shifted, v, atol=1e-7)


def test_solver_time_budget_is_explicit(monkeypatch):
    from euclid_dsps.amortized import population_low_rank as module

    ticks = iter([0.0, 2.0, 3.0, 4.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="time budget"):
        fit_low_rank_selected_weights(
            np.log(np.array([[0.9, 0.1], [0.2, 0.8]])),
            [0.5, 0.5],
            [[-1.0], [1.0]],
            maximum_seconds=1.0,
        )
