from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from euclid_dsps.amortized.forward_population import (
    PHYSICAL,
    PhysicalBasis,
    fit_selected_weights,
    parent_from_selected,
    selected_from_parent,
    selection_efficiencies,
)

NAMES = (*PHYSICAL, *(f"sfh_{i}" for i in range(10)))


def test_parent_selection_algebra():
    v = np.array([0.5, 0.5])
    np.testing.assert_allclose(parent_from_selected(v, [0.2, 0.8]), [0.8, 0.2])
    np.testing.assert_allclose(parent_from_selected(v, [0.2, 0.2]), v)
    u = np.array([0.2, 0.3, 0.5])
    alpha = np.array([0.1, 0.7, 0.9])
    np.testing.assert_allclose(
        parent_from_selected(selected_from_parent(u, alpha), alpha), u
    )
    assert np.isclose(selected_from_parent(u, alpha).sum(), 1)
    with pytest.raises(ValueError):
        parent_from_selected(v, [0, 1])


def test_basis_normalized_and_nuisance_correction_independent():
    basis = PhysicalBasis.create(NAMES, 8, seed=12)
    rng = np.random.default_rng(34)
    x, labels = basis.sample(rng, 100000)
    u = np.arange(1, 9) / 36
    logp0 = basis.log_prob(x, np.ones(8))
    ratio = np.exp(basis.log_prob(x, u) - logp0)
    assert abs(ratio.mean() - 1) < 0.012
    y = x[:100].copy()
    y[:, 5:] += 10
    np.testing.assert_allclose(
        basis.log_prob(x[:100], u) - logp0[:100],
        basis.log_prob(y, u) - basis.log_prob(y, np.ones(8)),
        atol=1e-12,
    )
    assert x.shape == (100000, 15)
    assert np.all(np.std(x[:, 5:], axis=0) > 0.98)
    assert len(np.unique(labels)) == 8


def test_exact_synthetic_mixture_recovers_parent():
    rng = np.random.default_rng(93)
    means = np.array([-3.0, 0.0, 3.0])
    alpha = np.array([0.2, 0.7, 0.95])
    u = np.array([0.45, 0.25, 0.30])
    v = selected_from_parent(u, alpha)
    c = selected_from_parent(np.ones(3), alpha)
    labels = rng.choice(3, 50000, p=v)
    x = means[labels] + rng.normal(size=len(labels))
    loglik = -0.5 * (x[:, None] - means) ** 2
    logc = loglik + np.log(c)
    logc -= logsumexp(logc, axis=1, keepdims=True)
    fitted, receipt = fit_selected_weights(logc, c)
    assert abs(fitted.sum() - 1) < 1e-12
    assert receipt["kkt_gap"] < 2e-6
    np.testing.assert_allclose(fitted, v, atol=0.015)
    reconstructed = parent_from_selected(fitted, alpha)
    assert abs(reconstructed.sum() - 1) < 1e-12
    np.testing.assert_allclose(reconstructed, u, atol=0.02)


def test_implemented_15d_family_oracle_recovers_parent_after_observed_cut():
    from scipy.stats import norm

    basis = PhysicalBasis.create(NAMES, 8, seed=14)
    u = np.arange(1, 9) / 36
    x, _ = basis.sample(np.random.default_rng(71), 100000, weights=u)
    # Every component must have observable support for a finite-sample recovery
    # invariant (a near-invisible component cannot pass a tight accuracy gate).
    threshold = -2.0
    x = x[x[:, basis.indices[0]] > threshold]
    alpha = norm.sf((threshold - basis.centers[:, 0]) / basis.scales[:, 0])
    # Observe theta without noise in this algebraic invariant. Selection is
    # deterministic in observed x; classifier has the exact uniform-reference law.
    logits = basis.component_log_prob(x)
    logc = logits - logsumexp(logits, axis=1, keepdims=True)
    v, _ = fit_selected_weights(logc, alpha / alpha.sum())
    np.testing.assert_allclose(parent_from_selected(v, alpha), u, atol=0.025)


def test_support_constraint_bounds_parent_mass_not_selected_mass():
    logc = np.log(np.array([[0.01, 0.99]] * 200 + [[0.9, 0.1]] * 10))
    alpha = np.array([0.8, 0.001])
    v, receipt = fit_selected_weights(
        logc, [0.5, 0.5], alpha=alpha, eligible=[True, False], weak_parent_mass=0.05
    )
    u = parent_from_selected(v, alpha)
    assert u[1] <= 0.050001
    assert receipt["support_constraint_active"]


@pytest.mark.parametrize("support", [False, True])
def test_stalled_slsqp_is_polished_without_relaxing_kkt(monkeypatch, support):
    from scipy.optimize import OptimizeResult

    from euclid_dsps.amortized import forward_population as module

    def premature_success(fun, x0, **unused):
        return OptimizeResult(
            x=np.asarray(x0),
            nit=1,
            success=True,
            message="Optimization terminated successfully",
        )

    monkeypatch.setattr(module, "minimize", premature_success)
    logc = np.log(np.array([[0.99, 0.01]] * 25 + [[0.01, 0.99]] * 75))
    args = (
        dict(alpha=[0.8, 0.01], eligible=[True, False], weak_parent_mass=0.05)
        if support
        else {}
    )
    with pytest.raises(RuntimeError, match="KKT gap"):
        fit_selected_weights(logc, [0.5, 0.5], polish_maxiter=0, **args)
    v, receipt = fit_selected_weights(logc, [0.5, 0.5], **args)
    assert receipt["initial_kkt_gap"] > 2e-6
    assert receipt["kkt_gap"] <= 2e-6
    assert receipt["polish_iterations"] > 0
    np.testing.assert_allclose(v.sum(), 1)
    assert np.all(v >= 0)
    if support:
        assert parent_from_selected(v, args["alpha"])[1] <= 0.050001
    else:
        np.testing.assert_allclose(
            v, [(0.25 - 0.01) / 0.98, (0.75 - 0.01) / 0.98], atol=1e-5
        )


def test_efficiencies_use_rejected_draws():
    result = selection_efficiencies(
        np.repeat([0, 1], 1000),
        np.r_[np.arange(1000) < 100, np.arange(1000) < 800],
        2,
        min_selected=10,
    )
    np.testing.assert_allclose(result["alpha"], [0.1, 0.8])
    assert result["eligible"].all()


def test_population_cannot_call_q_and_training_targets_are_forward_draws():
    from scripts.feniks_forward_population import population, train_posterior

    text = inspect.getsource(population)
    for forbidden in (
        "sample_posterior",
        "log_prob(model",
        "sample_independent_mixture",
        "load_checkpoint",
    ):
        assert forbidden not in text
    text = inspect.getsource(train_posterior)
    assert 'train["x"][ti]' in text
    for forbidden in ("stratified_proposal", "normalized_weights", "sample_posterior"):
        assert forbidden not in text


def test_physical_summary_excludes_sfh():
    from scripts.feniks_avi_em_factorial import _physical_closure_cells

    data = pd.DataFrame(
        [
            dict(variant="Q4_P4", population=p, group=g, median_wasserstein_over_iqr=v)
            for p in ("parent", "selected")
            for g, v in (("physical", 0.2), ("sfh", 100.0))
        ]
    )
    result = _physical_closure_cells(data)
    assert result.loc["Q4_P4", "parent_physical_w1_over_iqr"] == 0.2
    assert result.loc["Q4_P4", "selected_physical_w1_over_iqr"] == 0.2


def test_calibration_uses_all_15_dimensions():
    from scripts.report_feniks_forward_population import calibration

    rng = np.random.default_rng(59)
    truth = rng.normal(size=(1000, 15))
    draws = rng.normal(size=(1000, 512, 15))
    frame, _, _ = calibration(draws, truth, NAMES)
    assert (frame.group == "physical").sum() == 5
    assert (frame.group == "sfh").sum() == 10
    assert abs(frame.coverage_68.mean() - 0.68) < 0.025
    assert abs(frame.coverage_95.mean() - 0.95) < 0.025


def test_jax_parent_matches_numpy():
    import jax

    from euclid_dsps.amortized.forward_population_runtime import PopulationPrior

    basis = PhysicalBasis.create(NAMES, 8)
    u = np.arange(1, 9) / 36
    prior = PopulationPrior(basis, u)
    x = np.asarray(prior.sample(jax.random.PRNGKey(9), 32))
    assert x.shape == (32, 15)
    np.testing.assert_allclose(prior.log_prob(x), basis.log_prob(x, u), rtol=2e-6)
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import LatentSpec, x_to_theta

    spec = LatentSpec(NAMES, jnp.full(15, -10.0), jnp.full(15, 10.0))
    theta = x_to_theta(x, spec)
    jac = jax.vmap(jax.jacfwd(lambda z: x_to_theta(z, spec)))(jnp.asarray(x))
    logdet = jnp.linalg.slogdet(jac)[1]
    np.testing.assert_allclose(
        prior.log_prob_theta(theta, spec), prior.log_prob(x) - logdet, rtol=1e-5
    )


def test_supervised_classifier_learns_and_resumes(tmp_path):
    import jax
    import jax.numpy as jnp

    from scripts.feniks_forward_population import (
        classifier_template,
        classify,
        supervised_fit,
    )

    rng = np.random.default_rng(9)
    labels = np.arange(1024) % 2
    features = (4 * labels[:, None] - 2 + rng.normal(size=(1024, 3))).astype("float32")
    settings = dict(
        width=16,
        depth=1,
        seed=91,
        learning_rate=0.01,
        batch_size=64,
        epochs=8,
        validation_limit=256,
    )
    net = classifier_template(3, 2, settings)

    def loss(net, x, y):
        return -jnp.take_along_axis(
            jax.nn.log_softmax(jax.vmap(net)(x)), y[:, None], axis=1
        )[:, 0]

    trained = supervised_fit(
        net,
        loss,
        features[:768],
        labels[:768],
        (features[768:], labels[768:]),
        settings,
        tmp_path,
    )
    predicted = classify(trained, features[768:])
    assert -predicted[np.arange(256), labels[768:]].mean() < 0.15
    resumed = supervised_fit(
        net,
        loss,
        features[:768],
        labels[:768],
        (features[768:], labels[768:]),
        settings,
        tmp_path,
    )
    np.testing.assert_array_equal(predicted, classify(resumed, features[768:]))


def test_real_15d_spline_supervised_gradient_and_samples():
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.avi_experiments import log_prob
    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
    from euclid_dsps.amortized.proposal_expressivity import (
        IndependentFlowMixture,
        sample_independent_mixture,
    )

    enc = ConditionalFlowEncoder(
        jax.random.PRNGKey(1),
        input_dim=6,
        latent_dim=15,
        hidden_sizes=(8,),
        family="rq_spline",
        n_layers=2,
        hidden_size=8,
        n_bins=4,
        tail_bound=12.0,
        output_space="latent_x",
        transport_float64=True,
        activation="gelu",
        log_std_min=-5.0,
        log_std_max=3.0,
        initial_log_std=0.0,
    )
    model = AmortizedModel(
        encoder=enc, prior=StandardNormalPrior(latent_dim=15), sed_scale=None
    )
    candidate = IndependentFlowMixture(jax.random.PRNGKey(2), enc, n_components=2)
    f = jnp.ones((3, 6))
    truth = jax.random.normal(jax.random.PRNGKey(3), (1, 3, 15))
    loss, grad = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda net: -jnp.mean(log_prob(model, net, f, truth)))
    )(candidate)
    assert np.isfinite(loss)
    assert all(
        np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad) if eqx.is_array(x)
    )
    draws = sample_independent_mixture(model, candidate, jax.random.PRNGKey(4), f, 16).x
    assert draws.shape == (16, 3, 15)
    assert np.isfinite(draws).all()


def test_forward_adapter_noise_selection_and_nuisance(monkeypatch):
    from types import SimpleNamespace

    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.features import FeatureStats
    from euclid_dsps.amortized.forward_population_runtime import simulator
    from euclid_dsps.amortized.latent import LatentSpec

    # Only replace expensive DSPS photometry; use real transforms, m5 errors,
    # Gaussian survey noise, feature construction and observed selection.
    def flux(x, *unused):
        return 1e-30 * jnp.exp(0.1 * x[:, :3] + 0.05 * x[:, 5:8])

    monkeypatch.setattr("euclid_dsps.amortized.decoder.model_flux_from_x", flux)
    monkeypatch.setattr(
        "euclid_dsps.amortized.posterior_target._apply_model_calibration",
        lambda m, f, c: f,
    )
    spec = LatentSpec(NAMES, jnp.full(15, -10.0), jnp.full(15, 10.0))
    stats = FeatureStats(np.full(3, 1e-30), np.full(3, 1e-31), ("lsst_r", "g", "i"))
    rt = SimpleNamespace(
        latent_spec=spec,
        context=None,
        model_args=None,
        parameter_names=NAMES,
        calibration_config={},
        feature_stats=stats,
        likelihood_config={"type": "student_t", "student_t_dof": 2},
    )
    obs = dict(
        m5=jnp.full(3, 29.0),
        gamma=jnp.full(3, 0.039),
        noise_family="gaussian",
        selection_band_index=0,
        selection_flux_min_fnu_cgs=1e-30,
    )
    simulate = simulator(None, rt, obs)
    x = jnp.zeros((4096, 15))
    data = simulate(x, jax.random.PRNGKey(53))
    assert data["theta"].shape == (4096, 15)
    assert data["features"].shape == (4096, 6)
    assert np.all(data["valid"])
    np.testing.assert_array_equal(data["selected"], data["flux"][:, 0] > 1e-30)
    residual = (np.asarray(data["flux"]) - 1e-30) / np.asarray(data["errors"])
    assert abs(residual.std() - 1) < 0.04
    other = simulate(x.at[:, 5].set(1), jax.random.PRNGKey(53))
    assert not np.array_equal(data["flux"], other["flux"])
