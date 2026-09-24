import numpy as np

from euclid_dsps.amortized.forward_population import fit_selected_weights
from euclid_dsps.amortized.population_regularization import fit_selected_weights_kl


def test_kl_regularization_preserves_simplex_and_prevents_boundary_collapse():
    logc = np.log(np.array([[0.999, 0.001]] * 90 + [[0.01, 0.99]] * 10))
    unregularized, _ = fit_selected_weights(logc, [0.5, 0.5])
    regularized, receipt = fit_selected_weights_kl(logc, [0.5, 0.5], strength=0.1)

    np.testing.assert_allclose(regularized.sum(), 1.0)
    assert np.all(regularized > 0)
    assert regularized.min() > unregularized.min()
    assert receipt["kkt_gap"] <= 2e-6


def test_kl_regularization_retains_parent_support_constraint():
    logc = np.log(np.array([[0.01, 0.99]] * 200 + [[0.9, 0.1]] * 10))
    alpha = np.array([0.8, 0.001])
    selected, receipt = fit_selected_weights_kl(
        logc,
        [0.5, 0.5],
        strength=0.01,
        alpha=alpha,
        eligible=[True, False],
        weak_parent_mass=0.05,
    )
    parent = selected / alpha
    parent /= parent.sum()

    assert parent[1] <= 0.050001
    assert receipt["support_constraint_active"]
