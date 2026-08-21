from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")

if HAS_EQUINOX:
    import equinox as eqx

    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.features import FeatureStats, make_encoder_features
    from euclid_dsps.amortized.flows import RealNVPPrior, StandardNormalPrior
    from euclid_dsps.amortized.posterior import (
        ConditionalFlowEncoder,
        defensive_mixture_log_prob,
        defensive_posterior_proposal,
        posterior_encoder_state,
        posterior_entropy_diagnostics,
        posterior_log_prob,
        posterior_reference_from_base_mean,
        sample_posterior,
        sample_posterior_from_state,
    )
    from euclid_dsps.amortized.train import (
        JitLatentSpec,
        LossBatch,
        _encoder_epoch_index,
        _estimate_selection_log_alpha,
        _evaluation_metrics,
        _loss_and_grads_jit,
        _loss_with_metrics,
        _normalized_particle_weights,
        _prior_mstep_loss,
        _sample_sleep_noise,
        _sleep_encoder_features,
        _sleep_flux_error,
        _sleep_m5_flux_error,
        _sleep_observed_selection_mask,
        _training_update_phase,
        _wake_update_active,
    )
    from euclid_dsps.calibration import (
        GlobalSedScaleState,
        PerBandFluxCalibrationState,
    )
    from euclid_dsps.photometry import abmag_to_fnu_cgs


def test_defensive_mixture_uses_complete_component_density() -> None:
    log_density = jnp.log(
        jnp.asarray(
            [
                [[0.2, 0.6], [0.5, 0.1]],
                [[0.7, 0.3], [0.1, 0.8]],
                [[0.1, 0.1], [0.4, 0.1]],
            ]
        )
    )
    fractions = jnp.asarray([0.5, 0.3, 0.2])
    actual = defensive_mixture_log_prob(log_density, fractions)
    expected = jnp.log(jnp.sum(fractions[:, None, None] * jnp.exp(log_density), axis=0))
    assert jnp.allclose(actual, expected, atol=1.0e-7)


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


def test_autoregressive_spline_conditional_flow_has_exact_density() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(40),
        input_dim=6,
        latent_dim=3,
        hidden_sizes=(12,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="autoregressive_rq_spline",
        n_layers=2,
        hidden_size=12,
        n_bins=6,
        init_scale=0.1,
        output_space="latent_x",
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=3),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jax.random.normal(jax.random.PRNGKey(41), (4, 6))
    posterior = sample_posterior(model, jax.random.PRNGKey(42), features, 5)
    evaluated = jax.vmap(lambda value: posterior_log_prob(model, features, value))(
        posterior.x
    )

    assert encoder.family == "autoregressive_rq_spline"
    assert posterior.x.shape == (5, 4, 3)
    assert jnp.all(jnp.isfinite(posterior.logq))
    assert jnp.allclose(evaluated, posterior.logq, atol=5.0e-4)


def test_passband_set_context_is_direct_and_differentiable() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(43),
        input_dim=8,
        latent_dim=3,
        hidden_sizes=(12,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="realnvp",
        n_layers=2,
        hidden_size=12,
        output_space="latent_x",
        context_encoder_type="passband_set_transformer",
        set_n_bands=4,
        set_token_dim=12,
        set_context_dim=10,
        set_num_heads=3,
        set_num_layers=2,
    )
    features = jax.random.normal(jax.random.PRNGKey(44), (2, 8))
    mean, log_std = encoder(features)
    context = encoder.flow_context(features, mean, log_std)
    gradient = jax.grad(lambda value: jnp.sum(encoder.flow_context(value)))(features[0])

    assert mean.shape == (2, 3)
    assert log_std.shape == (2, 3)
    assert context.shape == (2, 10)
    assert context.shape[-1] != 2 * mean.shape[-1]
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0.0


def test_precomputed_encoder_state_matches_composed_sampling() -> None:
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
    features = jnp.ones((3, 6), dtype=jnp.float32)
    key = jax.random.PRNGKey(7)
    composed = sample_posterior(model, key, features, 5)
    state = posterior_encoder_state(model, features)
    split = sample_posterior_from_state(model, key, state, 5)
    for composed_value, split_value in zip(composed, split, strict=True):
        assert jnp.array_equal(composed_value, split_value)


def test_tempered_posterior_samples_match_exact_tempered_log_prob() -> None:
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
        init_scale=0.2,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((3, 6), dtype=jnp.float32)
    key = jax.random.PRNGKey(8)
    unit = sample_posterior(model, key, features, 5)
    tempered = sample_posterior(
        model,
        key,
        features,
        5,
        base_temperature=1.5,
    )
    evaluated = jax.vmap(
        lambda value: posterior_log_prob(
            model,
            features,
            value,
            base_temperature=1.5,
        )
    )(tempered.x)

    assert not jnp.allclose(unit.x, tempered.x)
    assert jnp.allclose(evaluated, tempered.logq, atol=2.0e-4)


def test_antithetic_gaussian_posterior_pairs_base_noise() -> None:
    from euclid_dsps.amortized.encoder import GaussianEncoder

    encoder = GaussianEncoder(
        jax.random.PRNGKey(0),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((3, 6), dtype=jnp.float32)
    mean, _log_std = encoder(features)
    posterior = sample_posterior(
        model,
        jax.random.PRNGKey(1),
        features,
        4,
        sample_strategy="antithetic",
    )

    assert jnp.allclose(posterior.x[:2] + posterior.x[2:], 2.0 * mean[None, ...])


def test_tempered_posterior_sample_density_is_exact() -> None:
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
    features = jnp.ones((3, 6), dtype=jnp.float32)
    posterior = sample_posterior(
        model,
        jax.random.PRNGKey(1),
        features,
        3,
        base_temperature=2.0,
    )
    evaluated = jax.vmap(
        lambda value: posterior_log_prob(model, features, value, base_temperature=2.0)
    )(posterior.x)

    assert jnp.allclose(evaluated, posterior.logq, atol=3.0e-4)


def test_posterior_reference_pushes_base_mean_through_conditional_flow() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(7),
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
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((3, 6), dtype=jnp.float32)
    mean, log_std = encoder(features)
    expected, _ = encoder.forward(mean, jnp.concatenate([mean, log_std], axis=-1))

    actual = posterior_reference_from_base_mean(model, features)

    assert jnp.allclose(actual, expected)


def test_mixture_conditional_flow_sample_density_is_exact() -> None:
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
        base_components=2,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((3, 6), dtype=jnp.float32)
    posterior = sample_posterior(model, jax.random.PRNGKey(1), features, 5)
    evaluated = jax.vmap(lambda x: posterior_log_prob(model, features, x))(posterior.x)

    assert posterior.x.shape == (5, 3, 4)
    assert jnp.all(jnp.isfinite(posterior.logq))
    assert jnp.allclose(evaluated, posterior.logq, atol=3.0e-4)


def test_sleep_noise_matches_student_t_likelihood() -> None:
    error = jnp.ones((4096, 2), dtype=jnp.float32)
    noise, family = _sample_sleep_noise(
        jax.random.PRNGKey(4),
        error,
        sleep={"noise_family": "match_likelihood"},
        likelihood_config={"type": "student_t", "student_t_dof": 2.0},
    )

    assert family == "student_t"
    assert jnp.all(jnp.isfinite(noise))
    assert jnp.quantile(jnp.abs(noise), 0.99) > 3.0


def test_sleep_noise_can_remain_gaussian_with_student_t_likelihood() -> None:
    error = jnp.ones((4096, 1), dtype=jnp.float32)
    noise, family = _sample_sleep_noise(
        jax.random.PRNGKey(6),
        error,
        sleep={"noise_family": "gaussian"},
        likelihood_config={"type": "student_t", "student_t_dof": 2.0},
    )
    assert family == "gaussian"
    assert jnp.quantile(jnp.abs(noise), 0.99) < 3.0


def test_sleep_selection_uses_post_noise_observed_flux() -> None:
    model_flux = jnp.asarray([[0.8], [1.2]])
    noise = jnp.asarray([[0.3], [-0.3]])
    noisy_flux = model_flux + noise
    selected = _sleep_observed_selection_mask(
        noisy_flux,
        jnp.asarray([True, True]),
        band_index=0,
        flux_min=1.0,
    )
    assert selected.tolist() == [True, False]
    assert bool(model_flux[0, 0] < 1.0 < noisy_flux[0, 0])
    assert bool(noisy_flux[1, 0] < 1.0 < model_flux[1, 0])


def test_rws_weights_ignore_common_selection_normalizer() -> None:
    logweight = jnp.asarray(
        [[-2.0, 0.5], [-0.5, -1.0], [1.5, 0.2]],
        dtype=jnp.float32,
    )
    log_alpha = jnp.asarray([-0.7, -2.3], dtype=jnp.float32)
    baseline, valid = _normalized_particle_weights(logweight)
    corrected, corrected_valid = _normalized_particle_weights(
        logweight - log_alpha[None, :]
    )
    assert jnp.array_equal(valid, corrected_valid)
    assert jnp.allclose(baseline, corrected, atol=1.0e-7)


def test_selection_log_alpha_reaches_flow_prior_through_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import euclid_dsps.amortized.train as train_module

    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(0),
        input_dim=4,
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
    prior = RealNVPPrior(
        jax.random.PRNGKey(1),
        latent_dim=4,
        n_layers=2,
        hidden_size=8,
        init="identity",
        init_scale=0.0,
    )
    sed_scale = GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0))
    spec = JitLatentSpec(
        names=("a", "b", "c", "d"),
        lower=jnp.zeros(4),
        upper=jnp.ones(4),
        raw_center=jnp.zeros(4),
        raw_scale=jnp.ones(4),
    )
    flux_limit = 3.631e-30
    monkeypatch.setattr(
        train_module,
        "model_flux_from_x",
        lambda x, *_args, **_kwargs: flux_limit + 8.0e-31 * x[..., :1],
    )
    objective = {
        "selection_correction": {
            "enabled": True,
            "rng_seed": 17,
            "common_random_numbers": True,
            "band_index": 0,
            "flux_limit_fnu_cgs": flux_limit,
            "max_mag_ab": 25.0,
            "m5": 27.5,
            "gamma": 0.039,
            "sigma_sys_mag": 0.005,
            "min_sigma_fnu_cgs": 1.0e-40,
            "n_prior_samples": 1024,
            "prior_sample_batch_size": 64,
        }
    }

    def objective_for_prior(candidate_prior):
        model = AmortizedModel(
            encoder=encoder,
            prior=candidate_prior,
            sed_scale=sed_scale,
        )
        return _estimate_selection_log_alpha(
            model,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(9),
            {},
            objective,
        )[0]

    model = AmortizedModel(encoder=encoder, prior=prior, sed_scale=sed_scale)
    log_alpha, metrics = _estimate_selection_log_alpha(
        model,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(9),
        {},
        objective,
    )
    grads = eqx.filter_grad(objective_for_prior)(prior)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    assert jnp.isfinite(log_alpha)
    assert 0.0 < metrics["selection/alpha"] < 1.0
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in leaves)


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
    evaluated = jax.vmap(lambda value: posterior_log_prob(model_a, features, value))(
        posterior_a.x
    )

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
    loss, metrics = _prior_mstep_loss(model, batch, spec, jax.random.PRNGKey(2), 1, {})
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
        {
            "calibration": {
                "per_band_zero_points": {
                    "enabled": True,
                    "mode": "learn_per_band",
                    "trainable": True,
                    "prior_sigma_mag": 0.05,
                }
            }
        },
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


def test_stochastic_elbo_adds_selection_only_for_trainable_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import euclid_dsps.amortized.train as train_module

    calls = 0

    def fake_negative_elbo(*_args, **_kwargs):
        value = jnp.asarray(2.0)
        return value, {"loss": value}

    def fake_selection(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        log_alpha = jnp.asarray(-0.4)
        return log_alpha, {
            "selection/alpha": jnp.exp(log_alpha),
            "selection/log_alpha": log_alpha,
        }

    monkeypatch.setattr(train_module, "negative_elbo", fake_negative_elbo)
    monkeypatch.setattr(
        train_module,
        "_estimate_selection_log_alpha",
        fake_selection,
    )
    objective = {
        "mode": "stochastic_elbo",
        "update_phase": "joint",
        "prior_train_jointly": True,
        "selection_correction": {"enabled": True},
    }
    loss, metrics = _loss_with_metrics(
        object(),
        object(),
        object(),
        None,
        None,
        (),
        jax.random.PRNGKey(1),
        1,
        1.0,
        {},
        {},
        objective,
    )
    assert jnp.isclose(loss, 1.6)
    assert jnp.isclose(metrics["selection/log_alpha"], -0.4)
    assert calls == 1

    frozen = {**objective, "prior_train_jointly": False}
    frozen_loss, frozen_metrics = _loss_with_metrics(
        object(),
        object(),
        object(),
        None,
        None,
        (),
        jax.random.PRNGKey(1),
        1,
        1.0,
        {},
        {},
        frozen,
    )
    assert jnp.isclose(frozen_loss, 2.0)
    assert frozen_metrics["selection/enabled"] == 0.0
    assert calls == 1


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


def test_periodic_wake_uses_encoder_epoch_count_under_vem() -> None:
    prior_cfg = {
        "source": "joint_realnvp",
        "train_jointly": True,
        "update_schedule": "variational_em",
        "update_every_epochs": 4,
    }
    objective = {
        "mode": "periodic_wake",
        "wake": {"start_encoder_epoch": 40, "every_encoder_epochs": 4},
    }
    active_raw_epochs = []
    for epoch in range(1, 56):
        phase = _training_update_phase(prior_cfg, epoch=epoch, train_prior=True)
        encoder_epoch = _encoder_epoch_index(prior_cfg, epoch=epoch, train_prior=True)
        if _wake_update_active(
            objective,
            encoder_epoch=encoder_epoch,
            update_phase=phase,
        ):
            active_raw_epochs.append(epoch)

    assert active_raw_epochs[:2] == [49, 54]


def test_periodic_wake_loss_is_finite_and_reports_ess(
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
        prior=RealNVPPrior(
            jax.random.PRNGKey(2),
            latent_dim=4,
            n_layers=2,
            hidden_size=8,
            init="identity",
            init_scale=0.0,
        ),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
        band_calibration=PerBandFluxCalibrationState(log_alpha_band=jnp.zeros(2)),
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

    def fake_model_flux(x, *_args, **_kwargs):
        return x[..., :2]

    monkeypatch.setattr(train_module, "model_flux_from_x", fake_model_flux)
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
        {"type": "student_t", "student_t_dof": 2.0},
        {
            "calibration": {
                "per_band_zero_points": {
                    "enabled": True,
                    "mode": "learn_per_band",
                    "trainable": True,
                    "prior_sigma_mag": 0.05,
                }
            }
        },
        {
            "mode": "periodic_wake",
            "wake_active": True,
            "wake": {
                "n_particles": 4,
                "n_tempered_particles": 1,
                "base_temperature": 2.0,
            },
        },
    )

    assert jnp.isfinite(loss)
    assert metrics["wake_active"] == 1.0
    assert 0.25 <= metrics["wake_ess_fraction_mean"] <= 1.0
    assert jnp.isfinite(metrics["calibration_mstep_nll_per_band"])

    validation_metrics, object_metrics = _evaluation_metrics(
        model,
        batch,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(1),
        1,
        1.0,
        {"type": "student_t", "student_t_dof": 2.0},
        {
            "calibration": {
                "per_band_zero_points": {
                    "enabled": True,
                    "mode": "learn_per_band",
                    "trainable": True,
                    "prior_sigma_mag": 0.05,
                }
            }
        },
        {
            "mode": "periodic_wake",
            "wake_active": True,
            "wake": {
                "n_particles": 4,
                "n_tempered_particles": 1,
                "base_temperature": 2.0,
            },
        },
    )
    assert validation_metrics["effective_n_samples"] == 4
    assert object_metrics["negative_loglike"].shape == (3,)
    assert jnp.all(jnp.isfinite(object_metrics["negative_loglike"]))

    def wake_loss(candidate):
        return _loss_with_metrics(
            candidate,
            batch,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(1),
            1,
            1.0,
            {"type": "student_t", "student_t_dof": 2.0},
            {
                "calibration": {
                    "per_band_zero_points": {
                        "enabled": True,
                        "mode": "learn_per_band",
                        "trainable": True,
                        "prior_sigma_mag": 0.05,
                    }
                }
            },
            {
                "mode": "periodic_wake",
                "wake_active": True,
                "wake": {
                    "n_particles": 4,
                    "n_tempered_particles": 1,
                    "base_temperature": 2.0,
                },
            },
        )[0]

    grads = eqx.filter_grad(wake_loss)(model)
    encoder_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.encoder) if leaf is not None
    ]
    prior_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.prior) if leaf is not None
    ]
    assert encoder_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in encoder_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in encoder_leaves)
    assert all(jnp.allclose(leaf, 0.0) for leaf in prior_leaves)
    assert grads.band_calibration is not None
    assert jnp.any(jnp.abs(grads.band_calibration.log_alpha_band) > 0.0)


def test_smc_wake_is_finite_and_reports_sampler_diagnostics(
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
        base_components=2,
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
        band_calibration=PerBandFluxCalibrationState(log_alpha_band=jnp.zeros(2)),
    )
    batch = LossBatch(
        flux=jnp.zeros((2, 2)),
        flux_err=0.5 * jnp.ones((2, 2)),
        mask=jnp.ones((2, 2), dtype=bool),
        features=jnp.ones((2, 6)),
        truth_theta=jnp.zeros((2, 0)),
    )
    spec = JitLatentSpec(
        names=("a", "b", "c", "d"),
        lower=jnp.zeros(4),
        upper=jnp.ones(4),
        raw_center=jnp.zeros(4),
        raw_scale=jnp.ones(4),
    )
    monkeypatch.setattr(
        train_module, "model_flux_from_x", lambda x, *_args, **_kwargs: x[..., :2]
    )
    loss, metrics = _loss_with_metrics(
        model,
        batch,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(7),
        1,
        1.0,
        {"type": "student_t", "student_t_dof": 2.0, "error_floor_frac": 0.0},
        {
            "calibration": {
                "per_band_zero_points": {
                    "enabled": True,
                    "mode": "learn_per_band",
                    "trainable": True,
                    "prior_sigma_mag": 0.05,
                }
            }
        },
        {
            "mode": "reweighted_wake_sleep",
            "wake_active": True,
            "wake": {
                "sampler": "smc",
                "n_particles": 4,
                "smc_temperatures": [0.0, 0.5, 1.0],
                "smc_mala_steps": 1,
                "smc_mala_step_size": 0.02,
                "prior_loss_weight": 1.0,
            },
        },
    )

    assert jnp.isfinite(loss)
    assert float(metrics["smc_active"]) == 1.0
    assert 0.0 <= float(metrics["smc_mala_acceptance_mean"]) <= 1.0
    assert float(metrics["wake_ess_mean"]) > 0.0
    assert metrics["wake_all_nonfinite_fraction"] == 0.0
    assert jnp.isfinite(metrics["calibration_mstep_nll_per_band"])

    def smc_loss(candidate):
        return _loss_with_metrics(
            candidate,
            batch,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(7),
            1,
            1.0,
            {"type": "student_t", "student_t_dof": 2.0, "error_floor_frac": 0.0},
            {
                "calibration": {
                    "per_band_zero_points": {
                        "enabled": True,
                        "mode": "learn_per_band",
                        "trainable": True,
                        "prior_sigma_mag": 0.05,
                    }
                }
            },
            {
                "mode": "reweighted_wake_sleep",
                "wake_active": True,
                "wake": {
                    "sampler": "smc",
                    "n_particles": 4,
                    "smc_temperatures": [0.0, 0.5, 1.0],
                    "smc_mala_steps": 1,
                    "smc_mala_step_size": 0.02,
                    "prior_loss_weight": 1.0,
                },
            },
        )[0]

    grads = eqx.filter_grad(smc_loss)(model)
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert grads.band_calibration is not None
    assert jnp.any(jnp.abs(grads.band_calibration.log_alpha_band) > 0.0)


def test_reweighted_wake_updates_encoder_and_learned_prior(
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
        prior=RealNVPPrior(
            jax.random.PRNGKey(2),
            latent_dim=4,
            n_layers=2,
            hidden_size=8,
            init="identity",
            init_scale=0.0,
        ),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    batch = LossBatch(
        flux=jnp.ones((3, 2)),
        flux_err=0.1 * jnp.ones((3, 2)),
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
    monkeypatch.setattr(
        train_module, "model_flux_from_x", lambda x, *_args, **_kwargs: x[..., :2]
    )

    def loss_fn(candidate):
        return _loss_with_metrics(
            candidate,
            batch,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(1),
            1,
            1.0,
            {"type": "gaussian", "error_floor_frac": 0.0},
            {},
            {
                "mode": "reweighted_wake_sleep",
                "wake_active": True,
                "wake": {
                    "n_particles": 4,
                    "n_tempered_particles": 1,
                    "base_temperature": 2.0,
                    "train_prior": True,
                    "prior_loss_weight": 1.0,
                },
            },
        )[0]

    loss = loss_fn(model)
    grads = eqx.filter_grad(loss_fn)(model)
    encoder_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.encoder) if leaf is not None
    ]
    prior_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.prior) if leaf is not None
    ]
    assert jnp.isfinite(loss)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in encoder_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in prior_leaves)

    base_objective = {
        "mode": "reweighted_wake_sleep",
        "wake_active": True,
        "wake": {
            "n_particles": 4,
            "n_tempered_particles": 1,
            "base_temperature": 2.0,
            "train_prior": True,
            "prior_loss_weight": 1.0,
        },
    }

    def evaluate(objective):
        return _loss_with_metrics(
            model,
            batch,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(1),
            1,
            1.0,
            {"type": "gaussian", "error_floor_frac": 0.0},
            {},
            objective,
        )

    baseline_loss, baseline_metrics = evaluate(base_objective)
    disabled_loss, disabled_metrics = evaluate(
        {**base_objective, "selection_correction": {"enabled": False}}
    )
    assert jnp.array_equal(baseline_loss, disabled_loss)
    assert jnp.array_equal(
        baseline_metrics["wake_ess_fraction_mean"],
        disabled_metrics["wake_ess_fraction_mean"],
    )

    log_alpha = jnp.asarray(-0.4)

    def fake_selection(*_args, **_kwargs):
        return log_alpha, {
            "selection/alpha": jnp.exp(log_alpha),
            "selection/log_alpha": log_alpha,
            "selection/beta_mean": jnp.asarray(0.2),
        }

    monkeypatch.setattr(
        train_module,
        "_estimate_selection_log_alpha",
        fake_selection,
    )
    corrected_loss, corrected_metrics = evaluate(
        {**base_objective, "selection_correction": {"enabled": True}}
    )
    assert jnp.array_equal(
        corrected_metrics["wake_ess_fraction_mean"],
        baseline_metrics["wake_ess_fraction_mean"],
    )
    assert jnp.array_equal(
        corrected_metrics["wake_weight_max_mean"],
        baseline_metrics["wake_weight_max_mean"],
    )
    assert jnp.isclose(corrected_loss - baseline_loss, log_alpha)
    assert jnp.isclose(
        corrected_metrics["prior_mstep_nll"]
        - corrected_metrics["prior_mstep_nll_uncorrected"],
        log_alpha,
    )


def test_reweighted_sleep_is_finite_and_only_trains_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import euclid_dsps.amortized.train as train_module

    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(0),
        input_dim=4,
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
            jax.random.PRNGKey(2),
            latent_dim=4,
            n_layers=2,
            hidden_size=8,
            init="identity",
            init_scale=0.0,
        ),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    batch = LossBatch(
        flux=jnp.ones((3, 2)),
        flux_err=0.1 * jnp.ones((3, 2)),
        mask=jnp.ones((3, 2), dtype=bool),
        features=jnp.ones((3, 4)),
        truth_theta=jnp.zeros((3, 0)),
    )
    spec = JitLatentSpec(
        names=("a", "b", "c", "d"),
        lower=jnp.zeros(4),
        upper=jnp.ones(4),
        raw_center=jnp.zeros(4),
        raw_scale=jnp.ones(4),
    )
    monkeypatch.setattr(
        train_module,
        "model_flux_from_x",
        lambda x, *_args, **_kwargs: 1.0e-30 * (1.0 + 0.1 * x[..., :2]),
    )
    objective = {
        "mode": "reweighted_wake_sleep",
        "wake_active": False,
        "sleep": {
            "enabled": True,
            "m5": (25.0, 25.0),
            "gamma": (0.04, 0.04),
            "sigma_sys_mag": 0.005,
            "min_sigma_fnu_cgs": 1.0e-40,
            "feature_flux_scale": (1.0e-30, 1.0e-30),
            "feature_err_scale": (1.0e-31, 1.0e-31),
            "flux_transform": "asinh",
        },
    }

    def loss_fn(candidate):
        return _loss_with_metrics(
            candidate,
            batch,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(1),
            1,
            1.0,
            {"type": "gaussian", "error_floor_frac": 0.0},
            {},
            objective,
        )[0]

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
        {"type": "gaussian", "error_floor_frac": 0.0},
        {},
        objective,
    )
    grads = eqx.filter_grad(loss_fn)(model)
    prior_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.prior) if leaf is not None
    ]
    assert jnp.isfinite(loss)
    assert metrics["sleep_active"] == 1.0
    assert metrics["sleep_physical_valid_fraction"] == 1.0
    assert all(jnp.allclose(leaf, 0.0) for leaf in prior_leaves)


def test_sleep_m5_noise_and_features_match_catalog_contract() -> None:
    flux = jnp.asarray([[1.0e-30, -2.0e-31], [3.0e-31, 2.0e-30]], dtype=jnp.float32)
    config = {
        "m5": (25.0, 26.0),
        "gamma": (0.039, 0.041),
        "sigma_sys_mag": 0.005,
        "min_sigma_fnu_cgs": 1.0e-40,
        "feature_flux_scale": (8.0e-31, 1.2e-30),
        "feature_err_scale": (2.0e-31, 3.0e-31),
        "flux_transform": "asinh",
    }
    actual = _sleep_m5_flux_error(flux, config)
    compiled = jax.jit(lambda value: _sleep_m5_flux_error(value, config))(flux)

    unit = 1.0e-32
    flux64 = np.asarray(flux, dtype=np.float64) / unit
    m5 = np.asarray(config["m5"], dtype=np.float64)
    gamma = np.asarray(config["gamma"], dtype=np.float64)
    f5 = np.asarray(abmag_to_fnu_cgs(m5), dtype=np.float64) / unit
    sigma2 = (0.04 - gamma) * np.abs(flux64) * f5 + gamma * f5**2
    sys_frac = np.expm1(np.log(10.0) * config["sigma_sys_mag"] / 2.5)
    expected = np.sqrt(sigma2 + (sys_frac * np.abs(flux64)) ** 2) * unit
    assert jnp.allclose(
        actual,
        jnp.asarray(expected, dtype=jnp.float32),
        rtol=2.0e-6,
        atol=0.0,
    )
    assert jnp.all(jnp.isfinite(compiled))
    assert jnp.allclose(
        compiled,
        jnp.asarray(expected, dtype=jnp.float32),
        rtol=2.0e-6,
        atol=0.0,
    )

    mask = jnp.asarray([[True, False], [True, True]])
    features = _sleep_encoder_features(flux, actual, mask, config)
    stats = FeatureStats(
        flux_scale=np.asarray(config["feature_flux_scale"], dtype=np.float32),
        err_scale=np.asarray(config["feature_err_scale"], dtype=np.float32),
        band_names=("a", "b"),
        flux_transform="asinh",
    )
    err_scale_safe = jnp.maximum(jnp.asarray(stats.err_scale), 1.0e-30)
    expected = make_encoder_features(
        jnp.where(mask, flux, 0.0),
        jnp.where(mask, actual, err_scale_safe),
        stats,
    )
    assert jnp.array_equal(features, expected)


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


def test_observed_catalog_sleep_errors_are_covariates_not_latent_functions() -> None:
    batch = LossBatch(
        flux=jnp.ones((2, 3)),
        flux_err=jnp.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        mask=jnp.ones((2, 3), dtype=bool),
        features=jnp.zeros((2, 6)),
        truth_theta=jnp.zeros((2, 0)),
    )
    config = {
        "error_model": "observed_catalog",
        "feature_err_scale": (1.0, 1.0, 1.0),
    }
    first = _sleep_flux_error(jnp.zeros((6, 3)), batch, config)
    second = _sleep_flux_error(100.0 * jnp.ones((6, 3)), batch, config)
    expected = jnp.tile(batch.flux_err, (3, 1))
    assert jnp.array_equal(first, expected)
    assert jnp.array_equal(second, expected)


def test_full_posterior_entropy_includes_conditional_flow_logdet() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(50),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=3.0,
        initial_log_std=0.0,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.2,
        output_space="latent_x",
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    metrics = posterior_entropy_diagnostics(
        model,
        jnp.ones((3, 6)),
        jax.random.PRNGKey(51),
        n_samples=32,
    )
    reconstructed = (
        metrics["posterior_base_entropy"] + metrics["posterior_residual_logdet_mean"]
    )
    assert jnp.allclose(metrics["posterior_full_entropy_mc"], reconstructed)
    assert jnp.isfinite(metrics["posterior_residual_logdet_q05"])
    assert jnp.isfinite(metrics["posterior_residual_logdet_q95"])


def test_defensive_sampler_scores_every_draw_under_complete_mixture() -> None:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(52),
        input_dim=6,
        latent_dim=3,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=3.0,
        initial_log_std=0.0,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        output_space="latent_x",
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=3),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jnp.ones((2, 6))
    components = [
        {"source": "posterior", "temperature": 1.0, "fraction": 0.5},
        {"source": "posterior", "temperature": 2.0, "fraction": 0.3},
        {"source": "prior", "fraction": 0.2},
    ]
    proposal = defensive_posterior_proposal(
        model,
        jax.random.PRNGKey(53),
        features,
        16,
        components,
    )
    component_log_prob = jnp.stack(
        (
            posterior_log_prob(model, features, proposal.x, base_temperature=1.0),
            posterior_log_prob(model, features, proposal.x, base_temperature=2.0),
            model.prior.log_prob(proposal.x),
        )
    )
    expected = defensive_mixture_log_prob(
        component_log_prob,
        proposal.component_fractions,
    )
    assert jnp.allclose(proposal.logproposal, expected, atol=1.0e-6)


def test_prior_only_wake_has_no_encoder_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import euclid_dsps.amortized.train as train_module

    model = AmortizedModel(
        encoder=ConditionalFlowEncoder(
            jax.random.PRNGKey(54),
            input_dim=6,
            latent_dim=4,
            hidden_sizes=(8,),
            activation="gelu",
            log_std_min=-4.0,
            log_std_max=3.0,
            initial_log_std=0.0,
            family="realnvp",
            n_layers=2,
            hidden_size=8,
            output_space="latent_x",
        ),
        prior=RealNVPPrior(
            jax.random.PRNGKey(55),
            latent_dim=4,
            n_layers=2,
            hidden_size=8,
            init="identity",
            init_scale=0.0,
        ),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    batch = LossBatch(
        flux=jnp.ones((3, 2)),
        flux_err=0.1 * jnp.ones((3, 2)),
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
    monkeypatch.setattr(
        train_module, "model_flux_from_x", lambda x, *_args, **_kwargs: x[..., :2]
    )

    def loss_fn(candidate):
        return _loss_with_metrics(
            candidate,
            batch,
            spec,
            None,
            None,
            spec.names,
            jax.random.PRNGKey(56),
            1,
            1.0,
            {"type": "gaussian", "error_floor_frac": 0.0},
            {},
            {
                "mode": "reweighted_wake_sleep",
                "wake_active": True,
                "wake": {
                    "n_particles": 8,
                    "n_tempered_particles": 1,
                    "base_temperature": 2.0,
                    "train_encoder": False,
                    "train_prior": True,
                    "proposal": {
                        "components": [
                            {
                                "source": "posterior",
                                "temperature": 1.0,
                                "fraction": 0.50,
                            },
                            {
                                "source": "posterior",
                                "temperature": 2.0,
                                "fraction": 0.25,
                            },
                            {
                                "source": "posterior",
                                "temperature": 4.0,
                                "fraction": 0.15,
                            },
                            {"source": "prior", "fraction": 0.10},
                        ]
                    },
                },
            },
        )[0]

    grads = eqx.filter_grad(loss_fn)(model)
    encoder_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.encoder) if leaf is not None
    ]
    prior_leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads.prior) if leaf is not None
    ]
    assert all(jnp.all(leaf == 0.0) for leaf in encoder_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0.0) for leaf in prior_leaves)

    (jit_loss, jit_metrics), jit_grads = _loss_and_grads_jit(
        model,
        batch,
        spec,
        None,
        None,
        spec.names,
        jax.random.PRNGKey(56),
        1,
        1.0,
        {"type": "gaussian", "error_floor_frac": 0.0},
        {},
        {
            "mode": "reweighted_wake_sleep",
            "wake_active": True,
            "wake": {
                "n_particles": 8,
                "n_tempered_particles": 1,
                "base_temperature": 2.0,
                "train_encoder": False,
                "train_prior": True,
                "proposal": {
                    "components": [
                        {
                            "source": "posterior",
                            "temperature": 1.0,
                            "fraction": 0.50,
                        },
                        {
                            "source": "posterior",
                            "temperature": 2.0,
                            "fraction": 0.25,
                        },
                        {
                            "source": "posterior",
                            "temperature": 4.0,
                            "fraction": 0.15,
                        },
                        {"source": "prior", "fraction": 0.10},
                    ]
                },
            },
        },
    )
    assert jnp.isfinite(jit_loss)
    assert jit_metrics["wake_train_encoder"] == 0.0
    assert all(
        jnp.all(leaf == 0.0)
        for leaf in jax.tree_util.tree_leaves(jit_grads.encoder)
        if leaf is not None
    )
