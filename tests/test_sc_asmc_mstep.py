from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np

from euclid_dsps.amortized.posterior_bank import POSTERIOR_METHOD_CODES
from euclid_dsps.amortized.sc_asmc_mstep import _posterior_from_bank_shard
from tests.test_posterior_bank import _shard


def test_bank_to_mstep_batch_preserves_dense_weights_and_resolution() -> None:
    shard = _shard()
    posterior = _posterior_from_bank_shard(shard, np.asarray([0, 2]))

    assert posterior.particles.shape == (4, 2, 2)
    assert posterior.normalized_weights.shape == (4, 2)
    assert jnp.allclose(jnp.sum(posterior.normalized_weights, axis=0), 1.0)
    assert jnp.all(posterior.eligible)
    assert not bool(posterior.fallback_attempted[0])
    assert bool(posterior.fallback_attempted[1])
    assert int(shard.method[2]) == POSTERIOR_METHOD_CODES["fallback SMC"]


def test_prior_mstep_uses_four_device_global_centered_score() -> None:
    code = textwrap.dedent(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import log_ndtr

        from euclid_dsps.amortized.adaptive_smc_training import (
            SMCPosteriorBatch,
            make_component_optimizer,
            make_pmap_prior_macro_step,
            snapshot_model,
        )
        from euclid_dsps.amortized.elbo import AmortizedModel
        from euclid_dsps.amortized.flows import RealNVPPrior
        from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
        from euclid_dsps.amortized.sc_asmc_mstep import _shard_prior_posterior
        from euclid_dsps.amortized.sc_asmc_training import (
            _replicate_tree,
            _unreplicate_tree,
        )
        from euclid_dsps.calibration import GlobalSedScaleState

        devices = tuple(jax.local_devices())
        assert len(devices) == 4, devices
        k, n, dimension = 8, 8, 2
        encoder = ConditionalFlowEncoder(
            jax.random.PRNGKey(1), input_dim=4, latent_dim=dimension,
            hidden_sizes=(8,), activation="gelu", log_std_min=-4.0,
            log_std_max=2.5, initial_log_std=0.0, family="realnvp",
            n_layers=2, hidden_size=8, output_space="latent_x",
        )
        prior = RealNVPPrior(
            jax.random.PRNGKey(2), latent_dim=dimension, n_layers=2,
            hidden_size=8, permutation="alternating_roll", init="identity",
            init_scale=0.0,
        )
        model = AmortizedModel(
            encoder, prior, GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0))
        )
        posterior = SMCPosteriorBatch(
            particles=jax.random.normal(jax.random.PRNGKey(3), (k, n, dimension)),
            normalized_weights=jnp.full((k, n), 1.0 / k),
            eligible=jnp.ones(n, dtype=bool), beta_final=jnp.ones(n),
            final_ess=jnp.full((n,), k), final_max_weight=jnp.full((n,), 1.0/k),
            mutation_acceptance=jnp.full((n,), 0.3), final_rw_scale=jnp.full((n,), 0.2),
            unique_ancestor_fraction=jnp.ones(n), ancestor_ess=jnp.full((n,), k),
            ancestor_ess_fraction=jnp.ones(n), epsilon_squared_jump=jnp.ones(n),
            median_epsilon_squared_jump=jnp.ones(n), moved_particle_fraction=jnp.ones(n),
            unchanged_from_ancestor_fraction=jnp.zeros(n),
            poor_acceptance=jnp.zeros(n, dtype=bool),
            poor_ancestry=jnp.zeros(n, dtype=bool),
            poor_movement=jnp.zeros(n, dtype=bool),
            mixing_failure=jnp.zeros(n, dtype=bool), logZ_estimate=jnp.zeros(n),
            fallback_attempted=jnp.zeros(n, dtype=bool),
            fallback_succeeded=jnp.zeros(n, dtype=bool),
        )
        optimizer = make_component_optimizer(
            learning_rate=1.0e-4, gradient_clip_norm=5.0, weight_decay=0.0
        )
        state = optimizer.init(eqx.filter(model.prior, eqx.is_inexact_array))
        step = make_pmap_prior_macro_step(
            optimizer=optimizer,
            selection_log_beta_fn=lambda _model, samples: log_ndtr(samples[:, 0]),
            total_selection_samples=64,
            total_trust_samples=64,
            trust_strength=0.2,
            max_kl_per_dimension=10.0,
            max_alpha_mc_relative_error=1.0,
            gradient_clip_norm=5.0,
            n_devices=4,
        )
        updated, _state, metrics = step(
            _replicate_tree(model, devices),
            _replicate_tree(snapshot_model(model), devices),
            _replicate_tree(state, devices),
            _shard_prior_posterior(posterior, 4),
            jax.random.split(jax.random.PRNGKey(4), 4),
            jax.random.split(jax.random.PRNGKey(5), 4),
        )
        metric = _unreplicate_tree(metrics)
        assert bool(metric.update_applied)
        assert bool(metric.selection_score_weights_finite)
        assert jnp.isfinite(metric.selection_grad_norm)
        assert 0.0 < metric.selection_score_weight_ess <= 64.0
        assert 0.0 < metric.selection_maximum_score_weight <= 1.0
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(updated.prior) if eqx.is_array(leaf)]
        assert all(jnp.allclose(leaf[0], leaf[1:]) for leaf in leaves)
        print("PASS")
        """
    )
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS" in completed.stdout


def test_prior_mstep_global_gradient_is_device_count_invariant() -> None:
    code = textwrap.dedent(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from euclid_dsps.amortized.adaptive_smc_training import (
            SMCPosteriorBatch,
            make_component_optimizer,
            make_pmap_prior_macro_step,
            snapshot_model,
        )
        from euclid_dsps.amortized.elbo import AmortizedModel
        from euclid_dsps.amortized.flows import RealNVPPrior
        from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
        from euclid_dsps.amortized.sc_asmc_mstep import _shard_prior_posterior
        from euclid_dsps.amortized.sc_asmc_training import (
            _replicate_tree,
            _unreplicate_tree,
        )
        from euclid_dsps.calibration import GlobalSedScaleState

        devices = tuple(jax.local_devices())
        n_devices = len(devices)
        k, n, dimension = 8, 8, 2
        encoder = ConditionalFlowEncoder(
            jax.random.PRNGKey(1), input_dim=4, latent_dim=dimension,
            hidden_sizes=(8,), activation="gelu", log_std_min=-4.0,
            log_std_max=2.5, initial_log_std=0.0, family="realnvp",
            n_layers=2, hidden_size=8, output_space="latent_x",
        )
        prior = RealNVPPrior(
            jax.random.PRNGKey(2), latent_dim=dimension, n_layers=2,
            hidden_size=8, permutation="alternating_roll", init="identity",
            init_scale=0.0,
        )
        model = AmortizedModel(
            encoder, prior, GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0))
        )
        particles = jax.random.normal(jax.random.PRNGKey(3), (k, n, dimension))
        posterior = SMCPosteriorBatch(
            particles=particles,
            normalized_weights=jnp.full((k, n), 1.0 / k),
            eligible=jnp.ones(n, dtype=bool), beta_final=jnp.ones(n),
            final_ess=jnp.full((n,), k), final_max_weight=jnp.full((n,), 1.0/k),
            mutation_acceptance=jnp.full((n,), 0.3), final_rw_scale=jnp.full((n,), 0.2),
            unique_ancestor_fraction=jnp.ones(n), ancestor_ess=jnp.full((n,), k),
            ancestor_ess_fraction=jnp.ones(n), epsilon_squared_jump=jnp.ones(n),
            median_epsilon_squared_jump=jnp.ones(n), moved_particle_fraction=jnp.ones(n),
            unchanged_from_ancestor_fraction=jnp.zeros(n),
            poor_acceptance=jnp.zeros(n, dtype=bool),
            poor_ancestry=jnp.zeros(n, dtype=bool),
            poor_movement=jnp.zeros(n, dtype=bool),
            mixing_failure=jnp.zeros(n, dtype=bool), logZ_estimate=jnp.zeros(n),
            fallback_attempted=jnp.zeros(n, dtype=bool),
            fallback_succeeded=jnp.zeros(n, dtype=bool),
        )
        optimizer = make_component_optimizer(
            learning_rate=1.0e-4, gradient_clip_norm=100.0, weight_decay=0.0
        )
        state = optimizer.init(eqx.filter(model.prior, eqx.is_inexact_array))
        step = make_pmap_prior_macro_step(
            optimizer=optimizer,
            selection_log_beta_fn=lambda _model, samples: jnp.zeros(samples.shape[0]),
            total_selection_samples=64,
            total_trust_samples=64,
            trust_strength=0.0,
            max_kl_per_dimension=10.0,
            max_alpha_mc_relative_error=1.0,
            gradient_clip_norm=100.0,
            n_devices=n_devices,
        )
        _updated, _state, metrics = step(
            _replicate_tree(model, devices),
            _replicate_tree(snapshot_model(model), devices),
            _replicate_tree(state, devices),
            _shard_prior_posterior(posterior, n_devices),
            jax.random.split(jax.random.PRNGKey(4), n_devices),
            jax.random.split(jax.random.PRNGKey(5), n_devices),
        )
        metric = _unreplicate_tree(metrics)
        print(float(metric.data_grad_norm))
        """
    )

    norms = []
    for device_count in (1, 4):
        env = dict(os.environ)
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"
        env["JAX_PLATFORMS"] = "cpu"
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        norms.append(float(completed.stdout.strip().splitlines()[-1]))

    assert norms[0] > 0.0
    assert np.isclose(norms[0], norms[1], rtol=1.0e-5, atol=1.0e-6)
