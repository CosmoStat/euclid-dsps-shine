from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.adaptive_smc_training import (
    ordinary_importance_diagnostics,
)
from euclid_dsps.amortized.posterior_bank import POSTERIOR_METHOD_CODES
from euclid_dsps.amortized.sc_asmc_em import (
    GENERALIZED_EM_PHASES,
    dispatch_posterior_hierarchy,
    evaluate_budget_preflight,
    generalized_em_pseudocode,
    sc_asmc_parameter_count,
    select_active_bootstrap_rows,
    stratified_preflight_indices,
    validate_phase_isolation,
)


def test_ordinary_is_fast_path_uses_likelihood_prior_and_proposal_only() -> None:
    loglike = jnp.zeros((64, 3))
    logprior = jnp.zeros((64, 3))
    logproposal = jnp.zeros((64, 3))

    result = ordinary_importance_diagnostics(loglike, logprior, logproposal)
    perturbed = ordinary_importance_diagnostics(loglike, logprior, logproposal)

    assert jnp.all(result.accepted)
    assert jnp.allclose(result.ess, 64.0)
    assert jnp.allclose(result.max_weight, 1.0 / 64.0)
    # Selection beta is deliberately absent from this public object-weight API.
    assert result._fields == perturbed._fields


def test_hierarchy_dispatch_is_selective_and_hard_only() -> None:
    dispatch = dispatch_posterior_hierarchy(
        np.asarray([True, False, False, False, False]),
        np.asarray([False, True, False, False, False]),
        np.asarray([False, False, True, False, False]),
        np.asarray([False, False, False, True, False]),
    )

    np.testing.assert_array_equal(
        dispatch.method,
        [
            POSTERIOR_METHOD_CODES["IS"],
            POSTERIOR_METHOD_CODES["primary SMC"],
            POSTERIOR_METHOD_CODES["fallback SMC"],
            POSTERIOR_METHOD_CODES["extended SMC"],
            POSTERIOR_METHOD_CODES["unresolved"],
        ],
    )
    np.testing.assert_array_equal(dispatch.primary_attempted, [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(dispatch.fallback_attempted, [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(dispatch.extended_attempted, [0, 0, 0, 1, 1])


def _preflight(dispatch, *, attempt: int, budget: float = 1.0e9):
    n = len(dispatch.method)
    return evaluate_budget_preflight(
        dispatch,
        elapsed_seconds=10.0,
        dsps_evaluations=np.full(n, 100.0),
        stage_count=np.ones(n),
        mutation_acceptance=np.full(n, 0.4),
        ancestry_ess=np.full(n, 32.0),
        movement_squared=np.full(n, 0.1),
        beta_final=np.ones(n),
        full_catalogue_objects=8_376,
        parallel_shards=4,
        job_budget_seconds=budget,
        attempt=attempt,
    )


def test_preflight_gate_pass_active_bootstrap_and_abort() -> None:
    success = dispatch_posterior_hierarchy(
        np.ones(512, dtype=bool),
        np.zeros(512, dtype=bool),
        np.zeros(512, dtype=bool),
        np.zeros(512, dtype=bool),
    )
    failure = dispatch_posterior_hierarchy(
        np.zeros(512, dtype=bool),
        np.zeros(512, dtype=bool),
        np.zeros(512, dtype=bool),
        np.zeros(512, dtype=bool),
    )

    assert _preflight(success, attempt=1).status == "PASS"
    assert _preflight(failure, attempt=1).status == "ACTIVE_BOOTSTRAP"
    assert _preflight(failure, attempt=2).status == "ABORT"
    assert not _preflight(success, attempt=1, budget=1.0).continue_full_catalogue


def test_active_bootstrap_selects_requested_hard_count() -> None:
    direct = np.zeros(512, dtype=bool)
    direct[:300] = True
    dispatch = dispatch_posterior_hierarchy(
        direct,
        np.zeros(512, dtype=bool),
        np.zeros(512, dtype=bool),
        np.zeros(512, dtype=bool),
    )
    rows = select_active_bootstrap_rows(
        np.arange(512),
        dispatch,
        ess_fraction=np.linspace(1.0, 0.0, 512),
        max_weight=np.linspace(0.0, 1.0, 512),
        stage_count=np.arange(512),
        count=128,
    )

    assert rows.shape == (128,)
    assert np.all(rows >= 300)


def test_stratified_preflight_uses_only_observed_arrays() -> None:
    rng = np.random.default_rng(4)
    rows = np.arange(900, dtype=np.int64)
    flux = rng.normal(5.0, 2.0, size=(900, 18))
    error = rng.uniform(0.1, 1.0, size=(900, 18))

    selected = stratified_preflight_indices(
        rows,
        flux,
        error,
        r_band_index=3,
        flux_limit=2.0,
    )

    assert selected.shape == (512,)
    assert len(np.unique(selected)) == 512
    assert np.all(np.isin(selected, rows))


def test_exact_two_iteration_and_phase_isolation_contracts() -> None:
    assert GENERALIZED_EM_PHASES.count("e_step_1") == 1
    assert GENERALIZED_EM_PHASES.count("e_step_2") == 1
    assert len(generalized_em_pseudocode()) == 9
    validate_phase_isolation(
        "prior_m_step",
        q_frozen=True,
        prior_frozen=False,
        posterior_bank_frozen=True,
        trainable_components=("prior",),
    )
    with pytest.raises(ValueError, match="phase isolation violation"):
        validate_phase_isolation(
            "prior_m_step",
            q_frozen=False,
            prior_frozen=False,
            posterior_bank_frozen=True,
            trainable_components=("q", "prior"),
        )


def test_two_em_iteration_gaussian_toy_stops_after_second_update() -> None:
    grid = np.linspace(-4.0, 5.0, 4096)
    observed = np.asarray([1.1, 1.3, 1.6, 1.8, 2.0])
    prior_mean = 0.0
    history = [prior_mean]
    for _iteration in range(2):
        logprior = -0.5 * (grid - prior_mean) ** 2
        posterior_means = []
        for value in observed:
            loglike = -0.5 * ((value - grid) / 0.35) ** 2
            weights = np.exp(loglike + logprior - np.max(loglike + logprior))
            weights /= np.sum(weights)
            posterior_means.append(float(np.sum(weights * grid)))
        prior_mean = float(np.mean(posterior_means))
        history.append(prior_mean)

    assert len(history) == 3
    assert abs(history[-1] - np.mean(observed)) < abs(history[0] - np.mean(observed))
    assert GENERALIZED_EM_PHASES.count("prior_m_step_1") == 1
    assert GENERALIZED_EM_PHASES.count("prior_m_step_2") == 1
    assert "prior_m_step_3" not in GENERALIZED_EM_PHASES


def test_final_parameter_counts() -> None:
    assert sc_asmc_parameter_count(input_dim=36) == {
        "q_trunk_and_heads": 1_769_886,
        "q_conditional_realnvp": 662_196,
        "q_total": 2_432_082,
        "prior_total": 1_179_888,
        "total": 3_611_970,
    }
    assert sc_asmc_parameter_count(input_dim=54)["total"] == 3_621_186


HAS_EQUINOX = importlib.util.find_spec("equinox") is not None


@pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")
def test_residual_q_shapes_direct_context_density_and_identity_init() -> None:
    from euclid_dsps.amortized.elbo import AmortizedModel
    from euclid_dsps.amortized.encoder import ResidualPhotometryEncoder
    from euclid_dsps.amortized.flows import StandardNormalPrior
    from euclid_dsps.amortized.posterior import (
        ConditionalFlowEncoder,
        posterior_log_prob,
        sample_posterior,
    )
    from euclid_dsps.calibration import GlobalSedScaleState

    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(5),
        input_dim=36,
        latent_dim=15,
        hidden_sizes=(16,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=2.5,
        initial_log_std=0.0,
        family="realnvp",
        n_layers=6,
        hidden_size=256,
        scale_clamp=0.45,
        shift_clamp=3.0,
        init_scale=0.0,
        output_space="latent_x",
        context_encoder_type="residual_photometry",
        residual_trunk_width=512,
        residual_blocks=3,
        residual_representation_width=256,
        residual_context_dim=128,
        permutation="alternating_roll",
    )
    assert isinstance(encoder.base, ResidualPhotometryEncoder)
    features = jax.random.normal(jax.random.PRNGKey(6), (2, 36))
    mean, log_std, context = encoder.base.encode(features)
    base = jax.random.normal(jax.random.PRNGKey(7), (3, 2, 15))
    transformed, logdet = encoder.forward(base, context)
    recovered, inverse_logdet = encoder.inverse(transformed, context)
    assert mean.shape == (2, 15)
    assert log_std.shape == (2, 15)
    assert context.shape == (2, 128)
    assert jnp.allclose(log_std, 0.0)
    assert jnp.allclose(transformed, base, atol=1.0e-7)
    assert jnp.allclose(logdet, 0.0)
    assert jnp.allclose(recovered, base, atol=1.0e-7)
    assert jnp.allclose(inverse_logdet, 0.0)
    model = AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=15),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )
    posterior = sample_posterior(model, jax.random.PRNGKey(8), features, 3)
    evaluated = jax.vmap(lambda x: posterior_log_prob(model, features, x))(posterior.x)
    assert jnp.allclose(evaluated, posterior.logq, atol=2.0e-4)
    assert jnp.all(jnp.isfinite(posterior.logq))
    # Context is a direct photometric head, not concat(mean, log_std).
    assert encoder.context_dim == 128 != 2 * encoder.latent_dim
