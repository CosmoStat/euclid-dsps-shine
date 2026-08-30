from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.adaptive_bridge_smc import (
    AdaptiveBridgeSMCConfig,
    AdaptiveBridgeSMCResult,
    adaptive_next_beta,
    ancestor_ess_from_ids,
    epsilon_random_walk_mh,
    mixing_failure_mask,
    particle_movement_diagnostics,
    run_adaptive_bridge_smc,
)
from euclid_dsps.amortized.adaptive_smc_training import (
    SMCPosteriorBatch,
    apply_prior_macro_update,
    batched_prior_data_mstep_terms,
    make_component_optimizer,
    merge_hard_fallback,
    primary_posterior_batch,
    smc_prior_mstep_terms,
    smc_q_distillation_loss,
    snapshot_model,
    tree_all_finite,
)

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None

LOG_2PI = jnp.log(2.0 * jnp.pi)


def _isotropic_normal_log_prob(x, mean, scale):
    dimension = x.shape[-1]
    return -0.5 * (
        jnp.sum(jnp.square((x - mean) / scale), axis=-1)
        + dimension * LOG_2PI
        + 2.0 * dimension * jnp.log(scale)
    )


def _scaled_identity_transport(scale: float, dimension: int):
    logdet = dimension * jnp.log(jnp.asarray(scale))

    def forward(epsilon):
        return scale * epsilon, jnp.full(epsilon.shape[:-1], logdet)

    def inverse(x):
        return x / scale, jnp.full(x.shape[:-1], logdet)

    return forward, inverse


@pytest.mark.parametrize("active", [False, True])
def test_epsilon_move_preserves_state_dtype_with_upcasting_transport(active) -> None:
    particles = jnp.zeros((4, 1, 2), dtype=jnp.float32)

    def log_density(value):
        return -jnp.sum(jnp.square(value), axis=-1)

    def forward(value):
        promoted = value.astype(jnp.float64)
        return promoted, jnp.zeros(promoted.shape[:-1], dtype=jnp.float64)

    def inverse(value):
        promoted = value.astype(jnp.float64)
        return promoted, jnp.zeros(promoted.shape[:-1], dtype=jnp.float64)

    moved, accepted = jax.jit(
        lambda value: epsilon_random_walk_mh(
            key=jax.random.PRNGKey(7),
            particles=value,
            beta=jnp.ones((1,), dtype=jnp.float32),
            object_mask=jnp.asarray([active]),
            log_r0_fn=log_density,
            log_target_fn=log_density,
            epsilon_to_x_fn=forward,
            x_to_epsilon_fn=inverse,
            rw_scale=0.1,
        )
    )(particles)

    assert moved.dtype == particles.dtype
    assert accepted.dtype == jnp.bool_


def _sample_two_scale_mixture(key, shape, *, narrow, broad, narrow_fraction):
    component_key, noise_key = jax.random.split(key)
    narrow_component = jax.random.bernoulli(
        component_key,
        narrow_fraction,
        shape[:-1] + (1,),
    )
    noise = jax.random.normal(noise_key, shape)
    return jnp.where(narrow_component, narrow * noise, broad * noise)


def _two_scale_mixture_log_prob(
    x,
    *,
    narrow,
    broad,
    narrow_fraction,
):
    narrow_log_prob = _isotropic_normal_log_prob(x, 0.0, narrow)
    broad_log_prob = _isotropic_normal_log_prob(x, 0.0, broad)
    return jnp.logaddexp(
        jnp.log(narrow_fraction) + narrow_log_prob,
        jnp.log1p(-narrow_fraction) + broad_log_prob,
    )


def _weighted_global_moments(result):
    particles = np.asarray(result.final_particles)
    weights = np.asarray(result.final_normalized_weights)
    n_objects = particles.shape[1]
    flat_particles = particles.transpose(1, 0, 2).reshape(-1, particles.shape[-1])
    flat_weights = weights.T.reshape(-1) / n_objects
    mean = np.sum(flat_weights[:, None] * flat_particles, axis=0)
    centered = flat_particles - mean
    covariance = (centered.T * flat_weights) @ centered
    return mean, covariance


def test_analytic_1d_bridge_smc_recovers_narrow_proposal_failure() -> None:
    n_particles = 64
    n_objects = 24
    proposal_fraction = 0.90
    sample_key, smc_key = jax.random.split(jax.random.PRNGKey(1))
    initial = _sample_two_scale_mixture(
        sample_key,
        (n_particles, n_objects, 1),
        narrow=0.35,
        broad=2.0,
        narrow_fraction=proposal_fraction,
    )

    def log_r0(x):
        return _two_scale_mixture_log_prob(
            x,
            narrow=0.35,
            broad=2.0,
            narrow_fraction=proposal_fraction,
        )

    def log_target(x):
        return _isotropic_normal_log_prob(x, 1.0, 1.0)

    forward, inverse = _scaled_identity_transport(0.35, 1)
    result = run_adaptive_bridge_smc(
        key=smc_key,
        initial_particles=initial,
        log_r0_fn=log_r0,
        log_target_fn=log_target,
        epsilon_to_x_fn=forward,
        x_to_epsilon_fn=inverse,
        config=AdaptiveBridgeSMCConfig(
            n_particles=n_particles,
            hard_min_mutation_acceptance=0.0,
        ),
    )
    static_log_weights = log_target(initial) - log_r0(initial)
    static_weights = jax.nn.softmax(static_log_weights, axis=0)
    static_ess = 1.0 / jnp.sum(jnp.square(static_weights), axis=0)
    mean, covariance = _weighted_global_moments(result)

    assert float(jnp.median(static_ess) / n_particles) < 0.25
    assert np.all(np.asarray(result.beta_final) == 1.0)
    assert not np.any(np.asarray(result.hard_object_flag))
    assert abs(mean[0] - 1.0) < 0.20
    assert abs(covariance[0, 0] - 1.0) < 0.30
    assert abs(float(jnp.mean(result.logZ_estimate))) < 0.20


def test_correlated_gaussian_bridge_recovers_orientation() -> None:
    n_particles = 64
    n_objects = 24
    dimension = 2
    correlation = 0.90
    covariance = jnp.asarray(
        [[1.0, correlation], [correlation, 1.0]], dtype=jnp.float32
    )
    precision = jnp.linalg.inv(covariance)
    logdet = jnp.linalg.slogdet(covariance)[1]
    sample_key, smc_key = jax.random.split(jax.random.PRNGKey(2))
    initial = _sample_two_scale_mixture(
        sample_key,
        (n_particles, n_objects, dimension),
        narrow=0.55,
        broad=2.0,
        narrow_fraction=0.90,
    )

    def log_r0(x):
        return _two_scale_mixture_log_prob(
            x,
            narrow=0.55,
            broad=2.0,
            narrow_fraction=0.90,
        )

    def log_target(x):
        return -0.5 * (
            jnp.einsum("...i,ij,...j->...", x, precision, x)
            + dimension * LOG_2PI
            + logdet
        )

    forward, inverse = _scaled_identity_transport(0.55, dimension)
    result = run_adaptive_bridge_smc(
        key=smc_key,
        initial_particles=initial,
        log_r0_fn=log_r0,
        log_target_fn=log_target,
        epsilon_to_x_fn=forward,
        x_to_epsilon_fn=inverse,
        config=AdaptiveBridgeSMCConfig(
            n_particles=n_particles,
            hard_min_mutation_acceptance=0.0,
        ),
    )
    mean, estimated_covariance = _weighted_global_moments(result)
    estimated_correlation = estimated_covariance[0, 1] / np.sqrt(
        estimated_covariance[0, 0] * estimated_covariance[1, 1]
    )

    assert np.max(np.abs(mean)) < 0.15
    assert estimated_correlation > 0.72
    assert np.all(np.asarray(result.beta_final) == 1.0)


def test_15d_bridge_improves_over_static_is_with_k64() -> None:
    n_particles = 64
    n_objects = 16
    dimension = 15
    proposal_scale = 0.75
    sample_key, smc_key = jax.random.split(jax.random.PRNGKey(3))
    initial = proposal_scale * jax.random.normal(
        sample_key,
        (n_particles, n_objects, dimension),
    )

    def log_r0(x):
        return _isotropic_normal_log_prob(x, 0.0, proposal_scale)

    def log_target(x):
        return _isotropic_normal_log_prob(x, 0.0, 1.0)

    forward, inverse = _scaled_identity_transport(proposal_scale, dimension)
    result = run_adaptive_bridge_smc(
        key=smc_key,
        initial_particles=initial,
        log_r0_fn=log_r0,
        log_target_fn=log_target,
        epsilon_to_x_fn=forward,
        x_to_epsilon_fn=inverse,
        config=AdaptiveBridgeSMCConfig(
            n_particles=n_particles,
            hard_min_mutation_acceptance=0.0,
        ),
    )
    static_weights = jax.nn.softmax(log_target(initial) - log_r0(initial), axis=0)
    static_ess = 1.0 / jnp.sum(jnp.square(static_weights), axis=0)
    _mean, estimated_covariance = _weighted_global_moments(result)

    assert float(jnp.median(static_ess) / n_particles) < 0.30
    assert float(jnp.median(result.final_ess) / n_particles) > 0.70
    assert 0.70 < float(np.mean(np.diag(estimated_covariance))) < 1.20
    assert np.all(np.asarray(result.beta_final) == 1.0)


def test_defensive_bridge_retains_both_modes() -> None:
    n_particles = 64
    n_objects = 32
    sample_key, smc_key = jax.random.split(jax.random.PRNGKey(4))
    initial = _sample_two_scale_mixture(
        sample_key,
        (n_particles, n_objects, 1),
        narrow=0.60,
        broad=3.0,
        narrow_fraction=0.90,
    )

    def log_r0(x):
        return _two_scale_mixture_log_prob(
            x,
            narrow=0.60,
            broad=3.0,
            narrow_fraction=0.90,
        )

    def log_target(x):
        left = jnp.log(0.5) + _isotropic_normal_log_prob(x, -2.0, 0.5)
        right = jnp.log(0.5) + _isotropic_normal_log_prob(x, 2.0, 0.5)
        return jnp.logaddexp(left, right)

    forward, inverse = _scaled_identity_transport(0.60, 1)
    result = run_adaptive_bridge_smc(
        key=smc_key,
        initial_particles=initial,
        log_r0_fn=log_r0,
        log_target_fn=log_target,
        epsilon_to_x_fn=forward,
        x_to_epsilon_fn=inverse,
        config=AdaptiveBridgeSMCConfig(
            n_particles=n_particles,
            max_stages=12,
            hard_min_mutation_acceptance=0.0,
        ),
    )
    left_mass = jnp.mean(
        jnp.sum(
            result.final_normalized_weights
            * (result.final_particles[..., 0] < 0.0),
            axis=0,
        )
    )

    assert 0.30 < float(left_mass) < 0.70
    assert np.all(np.asarray(result.beta_final) == 1.0)


def test_adaptive_beta_is_monotone_and_matches_conditional_ess() -> None:
    key = jax.random.PRNGKey(5)
    log_ratio = 4.0 * jax.random.normal(key, (64, 3))
    log_weights = jnp.full((64, 3), -jnp.log(64.0))
    beta = jnp.asarray([0.0, 0.2, 0.8])
    next_beta, conditional_ess = adaptive_next_beta(
        log_weights,
        log_ratio,
        beta,
        target_fraction=0.75,
    )

    assert jnp.all(next_beta >= beta)
    assert jnp.all(next_beta <= 1.0)
    assert jnp.all(conditional_ess >= 0.75 * 64 - 1.0e-3)


def test_ancestor_ess_distinguishes_balanced_collapsed_and_dominant_lines() -> None:
    ancestors = jnp.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
            [5, 0, 0],
            [6, 0, 1],
            [7, 0, 2],
        ],
        dtype=jnp.int32,
    )
    ess, fraction = ancestor_ess_from_ids(ancestors)
    dominant_expected = 1.0 / (0.75**2 + 2.0 * 0.125**2)

    assert np.asarray(ess) == pytest.approx([8.0, 1.0, dominant_expected])
    assert np.asarray(fraction) == pytest.approx(
        [1.0, 1.0 / 8.0, dominant_expected / 8.0]
    )


def test_mixing_gate_rejects_low_ancestry_even_when_particles_moved() -> None:
    failure, poor_acceptance, poor_ancestry, poor_movement = mixing_failure_mask(
        mutation_acceptance=jnp.asarray([0.30, 0.30, 0.05]),
        mutation_proposed=jnp.asarray([True, True, True]),
        ancestor_ess_fraction=jnp.asarray([0.01, 0.01, 0.50]),
        epsilon_squared_jump=jnp.asarray([1.0, 0.0, 1.0]),
        min_mutation_acceptance=0.10,
        min_ancestor_ess_fraction=0.05,
        min_epsilon_squared_jump=1.0e-4,
    )

    assert np.array_equal(np.asarray(poor_acceptance), [False, False, True])
    assert np.array_equal(np.asarray(poor_ancestry), [True, True, False])
    assert np.array_equal(np.asarray(poor_movement), [False, True, False])
    assert np.array_equal(np.asarray(failure), [True, True, True])


def test_particle_movement_median_exposes_mostly_cloned_population() -> None:
    initial = jnp.zeros((8, 1, 2))
    final = initial.at[0, 0].set(jnp.asarray([20.0, 0.0]))
    accepted = jnp.zeros((8, 1), dtype=jnp.bool_).at[0, 0].set(True)
    mean_jump, median_jump, moved_fraction, unchanged_fraction = (
        particle_movement_diagnostics(final, initial, accepted)
    )

    assert float(mean_jump[0]) == pytest.approx(50.0)
    assert float(median_jump[0]) == 0.0
    assert float(moved_fraction[0]) == pytest.approx(1.0 / 8.0)
    assert float(unchanged_fraction[0]) == pytest.approx(7.0 / 8.0)


def test_epsilon_rw_mh_preserves_analytic_target() -> None:
    n_particles = 8192
    initial = jax.random.normal(jax.random.PRNGKey(6), (n_particles, 1, 1))
    forward, inverse = _scaled_identity_transport(1.0, 1)
    moved, accepted = epsilon_random_walk_mh(
        key=jax.random.PRNGKey(7),
        particles=initial,
        beta=jnp.ones((1,)),
        object_mask=jnp.ones((1,), dtype=jnp.bool_),
        log_r0_fn=lambda x: _isotropic_normal_log_prob(x, 0.0, 1.0),
        log_target_fn=lambda x: _isotropic_normal_log_prob(x, 0.0, 1.0),
        epsilon_to_x_fn=forward,
        x_to_epsilon_fn=inverse,
        rw_scale=0.60,
    )

    assert abs(float(jnp.mean(moved))) < 0.04
    assert abs(float(jnp.var(moved)) - 1.0) < 0.06
    assert 0.50 < float(jnp.mean(accepted)) < 0.95


def test_low_acceptance_adapts_rw_scale_and_records_particle_movement() -> None:
    n_particles = 512
    dimension = 15
    initial = jax.random.normal(
        jax.random.PRNGKey(71),
        (n_particles, 2, dimension),
    )
    forward, inverse = _scaled_identity_transport(1.0, dimension)
    result = run_adaptive_bridge_smc(
        key=jax.random.PRNGKey(72),
        initial_particles=initial,
        log_r0_fn=lambda x: _isotropic_normal_log_prob(x, 0.0, 1.0),
        log_target_fn=lambda x: _isotropic_normal_log_prob(x, 0.0, 1.0),
        epsilon_to_x_fn=forward,
        x_to_epsilon_fn=inverse,
        config=AdaptiveBridgeSMCConfig(
            n_particles=n_particles,
            steps_after_resample=0,
            final_steps_at_beta1=4,
            rw_scale=1.0,
            rw_scale_min=0.1,
            rw_scale_max=2.0,
            hard_min_mutation_acceptance=0.0,
        ),
    )

    expected = jnp.exp(result.mutation_acceptance - 0.30)
    assert jnp.allclose(result.final_rw_scale, expected, atol=2.0e-6)
    assert jnp.allclose(
        result.mutation_acceptance_path[0],
        result.mutation_acceptance,
        atol=1.0e-7,
    )
    assert jnp.all(result.unique_ancestor_fraction == 1.0)
    assert jnp.all(result.epsilon_squared_jump > 0.0)
    assert jnp.all(jnp.isfinite(result.epsilon_squared_jump))
    assert jnp.all(result.median_epsilon_squared_jump >= 0.0)
    assert jnp.all(result.moved_particle_fraction >= 0.0)
    assert jnp.all(result.moved_particle_fraction <= 1.0)
    assert jnp.allclose(
        result.unchanged_from_ancestor_fraction,
        1.0 - result.moved_particle_fraction,
    )


def test_unreachable_budget_marks_hard_without_fake_gradient_sample() -> None:
    n_particles = 64
    initial = jax.random.normal(jax.random.PRNGKey(8), (n_particles, 2, 1))
    forward, inverse = _scaled_identity_transport(1.0, 1)
    result = run_adaptive_bridge_smc(
        key=jax.random.PRNGKey(9),
        initial_particles=initial,
        log_r0_fn=lambda x: _isotropic_normal_log_prob(x, 0.0, 1.0),
        log_target_fn=lambda x: _isotropic_normal_log_prob(x, 30.0, 0.05),
        epsilon_to_x_fn=forward,
        x_to_epsilon_fn=inverse,
        config=AdaptiveBridgeSMCConfig(
            n_particles=n_particles,
            max_stages=1,
            steps_after_resample=0,
            final_steps_at_beta1=0,
            hard_min_mutation_acceptance=0.0,
        ),
    )

    assert np.all(np.asarray(result.hard_object_flag))
    assert np.all(np.asarray(result.beta_final) < 1.0)
    assert np.all(np.isfinite(np.asarray(result.final_normalized_weights)))


def _fake_smc_result(*, n_particles: int, n_objects: int, hard):
    particles = jnp.arange(n_particles * n_objects, dtype=jnp.float32).reshape(
        n_particles, n_objects, 1
    )
    weights = jnp.full((n_particles, n_objects), 1.0 / n_particles)
    hard = jnp.asarray(hard, dtype=jnp.bool_)
    return AdaptiveBridgeSMCResult(
        final_particles=particles,
        final_normalized_weights=weights,
        final_log_weights=jnp.log(weights),
        beta_final=jnp.where(hard, 0.5, 1.0),
        beta_path=jnp.zeros((2, n_objects)),
        conditional_ess_path=jnp.zeros((1, n_objects)),
        ess_path=jnp.zeros((1, n_objects)),
        resampled_path=jnp.zeros((1, n_objects), dtype=jnp.bool_),
        mutation_acceptance_path=jnp.zeros((1, n_objects)),
        final_ess=jnp.full((n_objects,), float(n_particles)),
        final_max_weight=jnp.full((n_objects,), 1.0 / n_particles),
        number_of_stages=jnp.ones((n_objects,), dtype=jnp.int32),
        number_of_resamples=jnp.zeros((n_objects,), dtype=jnp.int32),
        mutation_acceptance=jnp.full((n_objects,), 0.3),
        final_rw_scale=jnp.full((n_objects,), 0.6),
        unique_ancestor_fraction=jnp.ones((n_objects,)),
        ancestor_ess=jnp.full((n_objects,), float(n_particles)),
        ancestor_ess_fraction=jnp.ones((n_objects,)),
        epsilon_squared_jump=jnp.ones((n_objects,)),
        median_epsilon_squared_jump=jnp.ones((n_objects,)),
        moved_particle_fraction=jnp.ones((n_objects,)),
        unchanged_from_ancestor_fraction=jnp.zeros((n_objects,)),
        poor_acceptance=jnp.zeros((n_objects,), dtype=jnp.bool_),
        poor_ancestry=jnp.zeros((n_objects,), dtype=jnp.bool_),
        poor_movement=jnp.zeros((n_objects,), dtype=jnp.bool_),
        mixing_failure=jnp.zeros((n_objects,), dtype=jnp.bool_),
        hard_object_flag=hard,
        finite_target_fraction=jnp.ones((n_objects,)),
        logZ_estimate=jnp.arange(n_objects, dtype=jnp.float32),
        ancestor_ids=jnp.zeros((n_particles, n_objects), dtype=jnp.int32),
    )


def test_hard_fallback_only_replaces_successful_queued_objects() -> None:
    primary_result = _fake_smc_result(n_particles=4, n_objects=3, hard=[False, True, True])
    primary = primary_posterior_batch(primary_result)
    fallback = _fake_smc_result(n_particles=8, n_objects=2, hard=[False, True])
    merged = merge_hard_fallback(
        key=jax.random.PRNGKey(20),
        primary=primary,
        fallback=fallback,
        hard_object_indices=jnp.asarray([1, 2]),
    )

    assert np.array_equal(np.asarray(merged.eligible), [True, True, False])
    assert np.array_equal(np.asarray(merged.fallback_attempted), [False, True, True])
    assert np.array_equal(np.asarray(merged.fallback_succeeded), [False, True, False])
    assert jnp.allclose(merged.normalized_weights[:, 1], 0.25)
    assert jnp.array_equal(merged.particles[:, 2], primary.particles[:, 2])


@pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")
def test_conditional_realnvp_standard_transport_roundtrip_and_jacobian() -> None:
    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.posterior import (
        ConditionalFlowEncoder,
        posterior_log_prob,
        posterior_standard_base_to_x,
        posterior_x_to_standard_base,
    )
    from euclid_dsps.calibration import GlobalSedScaleState

    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(10),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(16,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=3.0,
        initial_log_std=0.0,
        family="realnvp",
        n_layers=4,
        hidden_size=16,
        init_scale=0.1,
        output_space="latent_x",
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=4),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jax.random.normal(jax.random.PRNGKey(11), (3, 6))
    epsilon = jax.random.normal(jax.random.PRNGKey(12), (7, 3, 4))
    forward = posterior_standard_base_to_x(model, features, epsilon)
    inverse = posterior_x_to_standard_base(model, features, forward.value)
    logq = posterior_log_prob(model, features, forward.value)
    standard_log_prob = _isotropic_normal_log_prob(epsilon, 0.0, 1.0)

    assert jnp.allclose(inverse.value, epsilon, atol=2.0e-5)
    assert jnp.allclose(
        inverse.logabsdet_dx_depsilon,
        forward.logabsdet_dx_depsilon,
        atol=2.0e-5,
    )
    assert jnp.allclose(
        logq,
        standard_log_prob - forward.logabsdet_dx_depsilon,
        atol=2.0e-4,
    )


@pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")
def test_smc_losses_stop_particles_and_separate_q_from_prior() -> None:
    import equinox as eqx

    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import RealNVPPrior
    from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
    from euclid_dsps.calibration import GlobalSedScaleState

    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(30),
        input_dim=6,
        latent_dim=4,
        hidden_sizes=(12,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=3.0,
        initial_log_std=0.0,
        family="realnvp",
        n_layers=2,
        hidden_size=12,
        init_scale=0.1,
        output_space="latent_x",
    )
    prior = RealNVPPrior(
        jax.random.PRNGKey(31),
        latent_dim=4,
        n_layers=2,
        hidden_size=12,
        permutation="roll",
        init="identity",
    )
    model = AmortizedModel(
        encoder=encoder,
        prior=prior,
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    features = jax.random.normal(jax.random.PRNGKey(32), (3, 6))
    particles = jax.random.normal(jax.random.PRNGKey(33), (5, 3, 4))
    posterior = SMCPosteriorBatch(
        particles=particles,
        normalized_weights=jnp.full((5, 3), 0.2),
        eligible=jnp.asarray([True, True, False]),
        beta_final=jnp.ones((3,)),
        final_ess=jnp.full((3,), 5.0),
        final_max_weight=jnp.full((3,), 0.2),
        mutation_acceptance=jnp.full((3,), 0.3),
        final_rw_scale=jnp.full((3,), 0.6),
        unique_ancestor_fraction=jnp.ones((3,)),
        ancestor_ess=jnp.full((3,), 5.0),
        ancestor_ess_fraction=jnp.ones((3,)),
        epsilon_squared_jump=jnp.ones((3,)),
        median_epsilon_squared_jump=jnp.ones((3,)),
        moved_particle_fraction=jnp.ones((3,)),
        unchanged_from_ancestor_fraction=jnp.zeros((3,)),
        poor_acceptance=jnp.zeros((3,), dtype=jnp.bool_),
        poor_ancestry=jnp.zeros((3,), dtype=jnp.bool_),
        poor_movement=jnp.zeros((3,), dtype=jnp.bool_),
        mixing_failure=jnp.zeros((3,), dtype=jnp.bool_),
        logZ_estimate=jnp.zeros((3,)),
        fallback_attempted=jnp.zeros((3,), dtype=jnp.bool_),
        fallback_succeeded=jnp.zeros((3,), dtype=jnp.bool_),
    )
    q_loss, model_grads = eqx.filter_value_and_grad(
        lambda candidate: smc_q_distillation_loss(
            candidate, features, posterior
        )[0]
    )(model)
    particle_grads = jax.grad(
        lambda values: smc_q_distillation_loss(
            model,
            features,
            posterior._replace(particles=values),
        )[0]
    )(particles)
    encoder_norm = sum(
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(model_grads.encoder)
        if leaf is not None
    )
    prior_norm = sum(
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(model_grads.prior)
        if leaf is not None
    )
    trust_samples = prior.sample(jax.random.PRNGKey(34), 32)
    prior_loss, prior_grads = eqx.filter_value_and_grad(
        lambda candidate: smc_prior_mstep_terms(
            candidate,
            prior,
            posterior,
            trust_samples,
        ).data_nll
    )(prior)
    prior_mstep_grad_norm = sum(
        jnp.sum(jnp.square(leaf))
        for leaf in jax.tree_util.tree_leaves(prior_grads)
        if leaf is not None
    )

    assert jnp.isfinite(q_loss)
    assert jnp.isfinite(prior_loss)
    assert encoder_norm > 0.0
    assert prior_norm == 0.0
    assert prior_mstep_grad_norm > 0.0
    assert jnp.all(particle_grads == 0.0)


@pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")
def test_realnvp_prior_data_and_trust_gradients_are_finite_at_extremes() -> None:
    import equinox as eqx

    from euclid_dsps.amortized.flows import RealNVPPrior

    dimension = 4
    prior = RealNVPPrior(
        jax.random.PRNGKey(61),
        latent_dim=dimension,
        n_layers=4,
        hidden_size=16,
        permutation="roll",
        init="identity",
    )
    ordinary = jax.random.normal(jax.random.PRNGKey(62), (14, 3, dimension))
    extremes = jnp.asarray(
        [
            [[8.0, -8.0, 6.0, -6.0]] * 3,
            [[-7.0, 7.0, -5.0, 5.0]] * 3,
        ],
        dtype=jnp.float32,
    )
    particles = jnp.concatenate((ordinary, extremes), axis=0)
    particle_count = particles.shape[0]
    posterior = SMCPosteriorBatch(
        particles=particles,
        normalized_weights=jnp.full((particle_count, 3), 1.0 / particle_count),
        eligible=jnp.ones((3,), dtype=jnp.bool_),
        beta_final=jnp.ones((3,)),
        final_ess=jnp.full((3,), float(particle_count)),
        final_max_weight=jnp.full((3,), 1.0 / particle_count),
        mutation_acceptance=jnp.full((3,), 0.3),
        final_rw_scale=jnp.full((3,), 0.6),
        unique_ancestor_fraction=jnp.ones((3,)),
        ancestor_ess=jnp.full((3,), float(particle_count)),
        ancestor_ess_fraction=jnp.ones((3,)),
        epsilon_squared_jump=jnp.ones((3,)),
        median_epsilon_squared_jump=jnp.ones((3,)),
        moved_particle_fraction=jnp.ones((3,)),
        unchanged_from_ancestor_fraction=jnp.zeros((3,)),
        poor_acceptance=jnp.zeros((3,), dtype=jnp.bool_),
        poor_ancestry=jnp.zeros((3,), dtype=jnp.bool_),
        poor_movement=jnp.zeros((3,), dtype=jnp.bool_),
        mixing_failure=jnp.zeros((3,), dtype=jnp.bool_),
        logZ_estimate=jnp.zeros((3,)),
        fallback_attempted=jnp.zeros((3,), dtype=jnp.bool_),
        fallback_succeeded=jnp.zeros((3,), dtype=jnp.bool_),
    )
    trust_samples = prior.sample(jax.random.PRNGKey(63), 16_384)

    data_value, data_gradient = eqx.filter_value_and_grad(
        lambda candidate: smc_prior_mstep_terms(
            candidate, prior, posterior, trust_samples
        ).data_nll
    )(prior)
    batched_data_value, batched_data_gradient = eqx.filter_value_and_grad(
        lambda candidate: batched_prior_data_mstep_terms(
            candidate,
            posterior,
            object_batch_size=2,
        ).data_nll
    )(prior)
    trust_value, trust_gradient = eqx.filter_value_and_grad(
        lambda candidate: smc_prior_mstep_terms(
            candidate, prior, posterior, trust_samples
        ).prior_kl_old_new
    )(prior)
    updates = jax.tree_util.tree_map(
        lambda gradient: -1.0e-4 * gradient if gradient is not None else None,
        data_gradient,
    )
    proposed = eqx.apply_updates(prior, updates)
    proposed_kl = smc_prior_mstep_terms(
        proposed, prior, posterior, trust_samples
    ).prior_kl_old_new

    assert jnp.isfinite(data_value)
    assert jnp.allclose(batched_data_value, data_value, rtol=1.0e-6, atol=1.0e-6)
    for batched_leaf, full_leaf in zip(
        jax.tree_util.tree_leaves(batched_data_gradient),
        jax.tree_util.tree_leaves(data_gradient),
        strict=True,
    ):
        if batched_leaf is not None:
            assert jnp.allclose(batched_leaf, full_leaf, rtol=2.0e-5, atol=2.0e-5)
    assert tree_all_finite(data_gradient)
    assert tree_all_finite(batched_data_gradient)
    assert trust_value == pytest.approx(0.0, abs=1.0e-7)
    assert tree_all_finite(trust_gradient)
    assert jnp.isfinite(proposed_kl)
    assert float(proposed_kl) >= -1.0e-4


@pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")
def test_prior_rejection_attributes_nonfinite_selection_gradient() -> None:
    import equinox as eqx

    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.flows import RealNVPPrior
    from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
    from euclid_dsps.calibration import GlobalSedScaleState

    latent_dim = 2
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(80),
        input_dim=4,
        latent_dim=latent_dim,
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
    prior = RealNVPPrior(
        jax.random.PRNGKey(81),
        latent_dim=latent_dim,
        n_layers=2,
        hidden_size=8,
        permutation="roll",
        init="identity",
    )
    model = AmortizedModel(
        encoder,
        prior,
        GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    posterior = SMCPosteriorBatch(
        particles=jax.random.normal(jax.random.PRNGKey(82), (8, 4, latent_dim)),
        normalized_weights=jnp.full((8, 4), 1.0 / 8.0),
        eligible=jnp.ones((4,), dtype=jnp.bool_),
        beta_final=jnp.ones((4,)),
        final_ess=jnp.full((4,), 8.0),
        final_max_weight=jnp.full((4,), 1.0 / 8.0),
        mutation_acceptance=jnp.full((4,), 0.3),
        final_rw_scale=jnp.full((4,), 0.6),
        unique_ancestor_fraction=jnp.ones((4,)),
        ancestor_ess=jnp.full((4,), 8.0),
        ancestor_ess_fraction=jnp.ones((4,)),
        epsilon_squared_jump=jnp.ones((4,)),
        median_epsilon_squared_jump=jnp.ones((4,)),
        moved_particle_fraction=jnp.ones((4,)),
        unchanged_from_ancestor_fraction=jnp.zeros((4,)),
        poor_acceptance=jnp.zeros((4,), dtype=jnp.bool_),
        poor_ancestry=jnp.zeros((4,), dtype=jnp.bool_),
        poor_movement=jnp.zeros((4,), dtype=jnp.bool_),
        mixing_failure=jnp.zeros((4,), dtype=jnp.bool_),
        logZ_estimate=jnp.zeros((4,)),
        fallback_attempted=jnp.zeros((4,), dtype=jnp.bool_),
        fallback_succeeded=jnp.zeros((4,), dtype=jnp.bool_),
    )
    optimizer = make_component_optimizer(
        learning_rate=1.0e-4,
        gradient_clip_norm=5.0,
        weight_decay=0.0,
    )
    optimizer_state = optimizer.init(eqx.filter(prior, eqx.is_inexact_array))

    selection_calls = []

    def nonfinite_selection_gradient(candidate, _key):
        selection_calls.append(True)
        leaf = next(
            value
            for value in jax.tree_util.tree_leaves(candidate.prior)
            if eqx.is_inexact_array(value) and value.size
        )
        centered = leaf.reshape(-1)[0] - jax.lax.stop_gradient(leaf.reshape(-1)[0])
        log_alpha = -0.5 + jnp.sqrt(jnp.square(centered))
        return log_alpha, {
            "selection/alpha_mc_relative_error": jnp.asarray(0.0)
        }

    _model, _state, metrics = apply_prior_macro_update(
        model=model,
        prior_snapshot=snapshot_model(model),
        optimizer=optimizer,
        optimizer_state=optimizer_state,
        posterior=posterior,
        trust_key=jax.random.PRNGKey(83),
        selection_key=jax.random.PRNGKey(84),
        selection_log_alpha_fn=nonfinite_selection_gradient,
        trust_samples=32,
        trust_strength=0.2,
        max_kl_per_dimension=0.05,
        max_alpha_mc_relative_error=0.15,
        gradient_clip_norm=5.0,
    )

    assert jnp.isfinite(metrics.loss)
    assert not metrics.grads_finite
    assert metrics.component_diagnostics_evaluated
    assert metrics.data_grads_finite
    assert not metrics.selection_grads_finite
    assert metrics.trust_grads_finite
    assert not metrics.update_applied
    assert metrics.rejection_code == 1
    assert len(selection_calls) == 1
