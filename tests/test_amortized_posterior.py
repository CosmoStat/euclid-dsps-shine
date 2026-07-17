from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import pytest

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")

if HAS_EQUINOX:
    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import RealNVPPrior, StandardNormalPrior
    from euclid_dsps.amortized.posterior import (
        ConditionalFlowEncoder,
        posterior_log_prob,
        sample_posterior,
    )
    from euclid_dsps.amortized.train import JitLatentSpec, LossBatch, _loss_with_metrics
    from euclid_dsps.calibration import GlobalSedScaleState


@pytest.mark.parametrize("family", ["realnvp", "rq_spline"])
def test_conditional_flow_roundtrip_and_log_prob(family: str) -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(0),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(16,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family=family,
        n_layers=4,
        hidden_size=12,
        n_bins=8,
        init_scale=0.0,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((5, 6), dtype=jnp.float32)
    posterior = sample_posterior(model, jax.random.PRNGKey(1), features, 3)
    evaluated = jax.vmap(lambda value: posterior_log_prob(model, features, value))(
        posterior.x
    )

    assert posterior.x.shape == (3, 5, 4)
    assert posterior.logq.shape == (3, 5)
    assert jnp.all(jnp.isfinite(posterior.logq))
    assert jnp.allclose(evaluated, posterior.logq, atol=2.0e-4)


def test_conditional_flow_gradients_are_finite() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(0),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.01,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((3, 6), dtype=jnp.float32)
    truth = jnp.zeros((3, 4), dtype=jnp.float32)
    grad = jax.grad(lambda value: -posterior_log_prob(model, value, truth).mean())(
        features
    )
    assert jnp.all(jnp.isfinite(grad))


def test_conditional_posterior_density_includes_prior_transport_jacobian() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(0),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.05,
    )
    prior = RealNVPPrior(
        jax.random.PRNGKey(1),
        latent_dim=4,
        n_layers=2,
        hidden_size=8,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=prior,
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((3, 6), dtype=jnp.float32)
    posterior = sample_posterior(model, jax.random.PRNGKey(2), features, 2)
    evaluated = jax.vmap(lambda value: posterior_log_prob(model, features, value))(
        posterior.x
    )

    assert jnp.allclose(evaluated, posterior.logq, atol=3.0e-4)


def test_npe_objective_uses_truth_and_skips_decoder() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(0),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.0,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    batch = LossBatch(
        flux=jnp.ones((3, 2)),
        flux_err=jnp.ones((3, 2)),
        mask=jnp.ones((3, 2), dtype=bool),
        features=jnp.ones((3, 6)),
        truth_theta=0.5 * jnp.ones((3, 4)),
    )
    spec = JitLatentSpec(
        names=("a", "b", "c", "d"),
        lower=jnp.zeros(4),
        upper=jnp.ones(4),
        raw_center=jnp.zeros(4),
        raw_scale=jnp.ones(4),
    )
    loss, metrics = _loss_with_metrics(
        model,
        batch,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(1),
        1,
        0.0,
        {},
        {},
        {"mode": "neural_posterior_estimation"},
    )
    assert jnp.isfinite(loss)
    assert metrics["finite_fraction"] == 1.0
    assert metrics["residual_rms"] == 0.0
