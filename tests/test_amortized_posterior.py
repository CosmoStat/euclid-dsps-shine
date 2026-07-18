from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import pytest

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")

if HAS_EQUINOX:
    import equinox as eqx

    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import RealNVPPrior, StandardNormalPrior
    from euclid_dsps.amortized.posterior import (
        ConditionalFlowEncoder,
        posterior_log_prob,
        sample_posterior,
    )
    from euclid_dsps.amortized.train import (
        JitLatentSpec,
        LossBatch,
        _loss_with_metrics,
        _prior_mstep_loss,
        _training_update_phase,
    )
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


def test_independent_conditional_flow_does_not_transport_through_prior() -> None:
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
        output_space="latent_x",
    )
    prior_a = RealNVPPrior(
        jax.random.PRNGKey(1), latent_dim=4, n_layers=2, hidden_size=8
    )
    prior_b = RealNVPPrior(
        jax.random.PRNGKey(2), latent_dim=4, n_layers=2, hidden_size=8
    )
    model_a = AmortizedModel(
        encoder=encoder,
        prior=prior_a,
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    model_b = AmortizedModel(
        encoder=encoder,
        prior=prior_b,
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((3, 6), dtype=jnp.float32)
    key = jax.random.PRNGKey(3)
    posterior_a = sample_posterior(model_a, key, features, 2)
    posterior_b = sample_posterior(model_b, key, features, 2)
    evaluated = jax.vmap(
        lambda value: posterior_log_prob(model_a, features, value)
    )(posterior_a.x)

    assert jnp.allclose(posterior_a.x, posterior_b.x)
    assert jnp.allclose(posterior_a.logq, posterior_b.logq)
    assert not jnp.allclose(posterior_a.logprior, posterior_b.logprior)
    assert jnp.allclose(evaluated, posterior_a.logq, atol=3.0e-4)


def test_independent_flow_kl_has_nonzero_prior_gradient() -> None:
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
        output_space="latent_x",
    )
    sed_scale = GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0))
    features = jnp.ones((3, 6), dtype=jnp.float32)
    key = jax.random.PRNGKey(3)

    def kl(prior):
        model = AmortizedModel(encoder=encoder, prior=prior, sed_scale=sed_scale)
        posterior = sample_posterior(model, key, features, 2)
        return jnp.mean(posterior.logq - posterior.logprior)

    prior = RealNVPPrior(
        jax.random.PRNGKey(1),
        latent_dim=4,
        n_layers=2,
        hidden_size=8,
        init="identity",
        init_scale=0.0,
    )
    grads = eqx.filter_grad(kl)(prior)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in leaves)


def test_prior_mstep_is_finite_without_decoder() -> None:
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
        output_space="latent_x",
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=RealNVPPrior(
            jax.random.PRNGKey(1), latent_dim=4, n_layers=2, hidden_size=8
        ),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    batch = LossBatch(
        flux=jnp.ones((3, 2)),
        flux_err=jnp.ones((3, 2)),
        mask=jnp.ones((3, 2), dtype=bool),
        features=jnp.ones((3, 6)),
        truth_theta=jnp.zeros((3, 0)),
    )
    spec = JitLatentSpec(
        names=("a", "b", "c", "d"),
        lower=jnp.zeros(4),
        upper=jnp.ones(4),
        raw_center=jnp.zeros(4),
        raw_scale=jnp.ones(4),
    )
    loss, metrics = _prior_mstep_loss(
        model, batch, spec, jax.random.PRNGKey(2), 1, {}
    )
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["prior_mstep_nll"])
    assert metrics["model_flux_mean"] == 0.0


def test_hybrid_objective_combines_elbo_npe_and_prior_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import euclid_dsps.amortized.train as train_module

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
        output_space="latent_x",
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

    def fake_negative_elbo(*args, **kwargs):
        value = jnp.asarray(2.0)
        return value, {"loss": value}

    monkeypatch.setattr(train_module, "negative_elbo", fake_negative_elbo)
    loss, metrics = _loss_with_metrics(
        model,
        batch,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(1),
        1,
        1.0,
        {},
        {},
        {
            "mode": "hybrid_elbo",
            "npe_weight": 50.0,
            "prior_truth_weight": 1.0,
        },
    )
    expected = 2.0 + 50.0 * metrics["npe_nll"] + metrics["prior_truth_nll"]
    assert jnp.allclose(loss, expected)
    assert metrics["npe_weight"] == 50.0
    assert metrics["prior_truth_weight"] == 1.0


def test_variational_em_schedule_counts_encoder_and_prior_epochs() -> None:
    prior_cfg = {
        "source": "joint_realnvp",
        "train_jointly": True,
        "update_schedule": "variational_em",
        "update_every_epochs": 4,
    }
    phases = [
        _training_update_phase(prior_cfg, epoch=epoch, train_prior=True)
        for epoch in range(1, 151)
    ]
    assert phases[:5] == ["encoder", "encoder", "encoder", "encoder", "prior"]
    assert phases.count("encoder") == 120
    assert phases.count("prior") == 30


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
