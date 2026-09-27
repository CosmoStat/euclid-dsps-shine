import numpy as np
import pytest

from euclid_dsps.amortized.physical_capacity import (
    fit_physical_weights,
    physical_cdf_design,
)
from scripts.feniks_reference_to_parent import weighted_ks


def test_known_mixture_and_original_cap_are_respected():
    f = np.array([[0.1, 0.7, 0.9], [0.8, 0.2, 0.7], [0.3, 0.9, 0.5]])
    actual = np.array([0.2, 0.3, 0.5])
    target = f @ actual
    u, info = fit_physical_weights(f, target, np.ones(3), f, target, 0.001)
    np.testing.assert_allclose(u, actual, atol=1e-7)
    assert info["objective"] < 1e-7 and info["truth_diagnostic_only"]
    # A conflicting physical target cannot bypass the original analytic CDF constraint.
    f = np.array([[0.0, 1.0]])
    u, info = fit_physical_weights(f, [0.9], [1.0], f, [0.2], 0.01)
    assert u[1] == pytest.approx(0.21) and info["train_cdf_max"] <= 0.0100001


def test_cdf_quadrature_recovers_target_family_and_keeps_tails():
    rng = np.random.default_rng(5)
    draws = np.stack([rng.normal(j, 0.25, (1000, 5)) for j in (0.0, 3.0)])
    truth = np.concatenate([draws[0, :200], draws[1, :800]])
    f, y, q = physical_cdf_design(draws, truth, 3, 51, 11)
    u, _ = fit_physical_weights(f, y, q, np.ones((1, 2)), [1.0], 0.0)
    np.testing.assert_allclose(u, [0.2, 0.8], atol=0.03)
    assert np.all((f >= 0) & (f <= 1)) and (q > 0).all()
    extreme = draws.copy()
    extreme[0, -1, 0] = 1e6
    saved = extreme.copy()
    _, _, tailq = physical_cdf_design(extreme, truth, 3, 51, 11)
    assert tailq.sum() > q.sum() * 100
    np.testing.assert_array_equal(extreme, saved)


@pytest.mark.parametrize("bad", [np.nan, -1.0, np.inf])
def test_invalid_solver_budget_rejected(bad):
    with pytest.raises(ValueError):
        fit_physical_weights([[0, 1]], [0.4], [1], [[0, 1]], [0.4], 0.01, bad)


def test_invalid_and_degenerate_designs_rejected():
    with pytest.raises(ValueError):
        physical_cdf_design(np.ones((2, 3, 4)), np.ones((3, 5)), 2, 9, 1)
    with pytest.raises(ValueError, match="Degenerate"):
        physical_cdf_design(np.ones((2, 3, 5)), np.ones((3, 5)), 2, 9, 1)
    with pytest.raises(ValueError):
        fit_physical_weights([[0, 2]], [0.4], [1], [[0, 1]], [0.4], 0.01)
    with pytest.raises(RuntimeError):
        fit_physical_weights([[0, 1]], [0.4], [1], [[0, 0]], [1.0], 0.01)


def test_weighted_cdf_distance_respects_measure_and_ties():
    assert weighted_ks([0, 1], [0, 0, 0, 1], [3, 1]) == 0
    assert weighted_ks([0, 1], [0, 1], [3, 1]) == 0.25
    assert weighted_ks([1, 0, 0], [0, 0, 0, 1], [1, 1, 2]) == 0
    with pytest.raises(ValueError):
        weighted_ks([0, 1], [0, 1], [0, 0])
