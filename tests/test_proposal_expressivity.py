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
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.posterior import (
        ConditionalFlowEncoder,
        posterior_log_prob,
    )
    from euclid_dsps.amortized.proposal_expressivity import (
        IndependentFlowMixture,
        count_parameters,
        fit_proposal_candidate,
        independent_mixture_log_prob,
        joint_distribution_metrics,
        sample_independent_mixture,
    )
    from euclid_dsps.calibration import GlobalSedScaleState


def _model():
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(1),
        input_dim=3,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-6.0,
        log_std_max=2.0,
        initial_log_std=-1.0,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.1,
        output_space="latent_x",
    )
    return AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=2),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )


def test_identical_independent_experts_reproduce_single_flow_density() -> None:
    model = _model()
    mixture = IndependentFlowMixture(
        jax.random.PRNGKey(2), model.encoder, n_components=2, mean_offset=0.05
    )
    mixture = eqx.tree_at(
        lambda item: item.experts,
        mixture,
        (model.encoder, model.encoder),
    )
    features = jnp.ones((4, 3), dtype=jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(3), (7, 4, 2))

    expected = posterior_log_prob(model, features, x)
    actual = independent_mixture_log_prob(model, mixture, features, x)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1.0e-5)


def test_independent_mixture_samples_report_exact_density() -> None:
    model = _model()
    mixture = IndependentFlowMixture(
        jax.random.PRNGKey(4), model.encoder, n_components=2, mean_offset=0.1
    )
    features = jnp.ones((3, 3), dtype=jnp.float32)

    sample = sample_independent_mixture(
        model, mixture, jax.random.PRNGKey(5), features, 11
    )
    evaluated = independent_mixture_log_prob(model, mixture, features, sample.x)

    assert sample.x.shape == (11, 3, 2)
    assert sample.component.shape == (11, 3)
    np.testing.assert_allclose(
        np.asarray(sample.logq), np.asarray(evaluated), atol=1e-5
    )
    assert count_parameters(mixture) > count_parameters(model.encoder)


def test_joint_distribution_metrics_detect_shifted_missing_coverage() -> None:
    rng = np.random.default_rng(7)
    target = rng.normal(size=(512, 3))
    weights = np.full(512, 1.0 / 512.0)
    matched = target + 0.02 * rng.normal(size=target.shape)
    shifted = rng.normal(loc=3.0, size=(512, 3))

    matched_metrics = joint_distribution_metrics(
        target, weights, matched, seed=8, max_draws=128, n_projections=32
    )
    shifted_metrics = joint_distribution_metrics(
        target, weights, shifted, seed=8, max_draws=128, n_projections=32
    )

    assert shifted_metrics["sliced_wasserstein"] > matched_metrics["sliced_wasserstein"]
    assert shifted_metrics["energy_distance"] > matched_metrics["energy_distance"]
    assert (
        shifted_metrics["nearest_cover_ratio"] > matched_metrics["nearest_cover_ratio"]
    )


def test_fit_rejects_overlapping_controlled_split() -> None:
    model = _model()
    features = jnp.ones((4, 3), dtype=jnp.float32)
    particles = jnp.zeros((8, 4, 2), dtype=jnp.float32)
    weights = jnp.full((8, 4), 1.0 / 8.0)

    with pytest.raises(ValueError, match="overlap"):
        fit_proposal_candidate(
            model,
            model.encoder,
            features=features,
            particles=particles,
            weights=weights,
            train_indices=np.asarray([0, 1]),
            validation_indices=np.asarray([1, 2]),
            epochs=1,
            object_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            seed=9,
        )
