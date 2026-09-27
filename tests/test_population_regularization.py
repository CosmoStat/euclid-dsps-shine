import numpy as np
import pytest
from scipy.optimize import OptimizeResult, brentq
from scipy.special import logsumexp

from euclid_dsps.amortized import population_regularization as module
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


@pytest.mark.parametrize("support", [False, True])
def test_regularized_polishing_certifies_premature_optimizer_success(
    monkeypatch, support
):
    def stalled(fun, x0, **kwargs):
        return OptimizeResult(
            x=np.asarray(x0),
            nit=1,
            success=True,
            message="Optimization terminated successfully",
        )

    monkeypatch.setattr(module, "minimize", stalled)
    logc = np.log([[0.99, 0.01]] * 25 + [[0.01, 0.99]] * 75)
    args = (
        dict(alpha=[0.8, 0.01], eligible=[True, False], weak_parent_mass=0.05)
        if support
        else {}
    )
    with pytest.raises(RuntimeError, match="not converged"):
        fit_selected_weights_kl(
            logc, [0.5, 0.5], strength=0.01, polish_maxiter=0, **args
        )
    v, diagnostics = fit_selected_weights_kl(logc, [0.5, 0.5], strength=0.01, **args)
    assert diagnostics["initial_kkt_gap"] > 2e-6
    assert diagnostics["kkt_gap"] <= 2e-6 and diagnostics["polish_iterations"] > 0
    np.testing.assert_allclose(v.sum(), 1)
    assert np.all(v >= 1e-12 - 1e-15)
    if support:
        parent = v / args["alpha"]
        assert parent[1] / parent.sum() <= 0.050001
    else:
        d = np.exp(logc)
        delta = d[:, 0] - d[:, 1]

        def slope(p):
            return -(delta / (d[:, 1] + p * delta)).mean() + 0.01 * np.log(p / (1 - p))

        optimum = brentq(slope, 1e-12, 1 - 1e-12)
        np.testing.assert_allclose(v, [optimum, 1 - optimum], atol=1e-7)


@pytest.mark.parametrize("support", [False, True])
def test_sharp_256_component_ratios_meet_original_kkt_tolerance(support):
    # Deterministic numerical regression, not remote data. Old bounded-loss
    # polishing exhausted 2000 steps with gap 9.63e-5 on the unconstrained case.
    rng = np.random.default_rng(19)
    logits = rng.normal(0, 8, (1024, 256))
    logits[:, :64] += 10
    logc = logits - logsumexp(logits, axis=1, keepdims=True)
    alpha = np.ones(256)
    alpha[:8] = 0.01
    args = dict(alpha=alpha, eligible=alpha > 0.1) if support else {}
    v, diagnostics = fit_selected_weights_kl(
        logc, np.full(256, 1 / 256), strength=0.003, **args
    )
    assert diagnostics["kkt_gap"] <= 2e-6
    np.testing.assert_allclose(v.sum(), 1)
    if support:
        parent = (v / alpha) / (v / alpha).sum()
        assert parent[:8].sum() <= 0.050001
    else:
        assert diagnostics["pairwise_iterations"] > 0
        d = np.exp(logc - logc.max(axis=1, keepdims=True))
        g = -(d.T @ (np.full(len(d), 1 / len(d)) / (d @ v))) + 0.003 * (
            np.log(v * 256) + 1
        )
        dual_gap = g @ (v - 1e-12) - (1 - 256e-12) * g.min()
        assert dual_gap <= 2e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tolerance=0),
        dict(tolerance=float("nan")),
        dict(maxiter=0),
        dict(polish_maxiter=-1),
    ],
)
def test_invalid_solver_budgets_fail(kwargs):
    with pytest.raises(ValueError):
        fit_selected_weights_kl(
            np.log([[0.5, 0.5]]), [0.5, 0.5], strength=0.01, **kwargs
        )
