from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import log_ndtr

from euclid_dsps.amortized.adaptive_smc_trainer import (
    _config_without_truth,
    _final_training_receipt,
    _smoke_training_config,
    adaptive_smc_configs,
    adaptive_training_config,
)
from euclid_dsps.amortized.adaptive_smc_training import (
    _select_exact_mixture_candidates,
)
from euclid_dsps.amortized.elbo import objective_mode, objective_uses_truth
from euclid_dsps.amortized.train import architecture_summary
from euclid_dsps.config import load_config
from scripts.estimate_feniks_adaptive_smc_cost import estimate

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None
pytestmark = pytest.mark.skipif(not HAS_EQUINOX, reason="equinox is not installed")


def _production_config():
    return load_config(
        "configs/experiments/"
        "feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml"
    )


def test_final_config_is_single_architecture_broad_prior_no_truth_contract() -> None:
    config = _production_config()
    runtime = _config_without_truth(config)
    cfg = runtime["amortized"]
    primary, fallback, proposal = adaptive_smc_configs(runtime)
    training = adaptive_training_config(runtime)

    assert runtime["truth"] == {"parameter_columns": {}}
    assert cfg["latent"]["normalization"] == "standardized_logit"
    assert cfg["latent"]["normalization_checkpoint"] is None
    assert cfg["latent"]["center_source"] == "fit_initial"
    assert cfg["prior"]["source"] == "joint_realnvp"
    assert cfg["prior"]["checkpoint"] is None
    assert cfg["prior"]["init"] == "identity"
    assert cfg["encoder"]["flow_family"] == "realnvp"
    assert cfg["encoder"]["base_components"] == 1
    assert cfg["objective"]["mode"] == "adaptive_smc_wake"
    assert cfg["objective"]["sleep"]["error_model"] == "observed_catalog"
    assert primary.n_particles == 64
    assert primary.max_stages == 8
    assert primary.rw_adapt_target_acceptance == pytest.approx(0.30)
    assert fallback.n_particles == 128
    assert fallback.max_stages == 12
    assert fallback.steps_after_resample == 4
    assert fallback.final_steps_at_beta1 == 2
    assert proposal.normalized_fractions().tolist() == pytest.approx(
        [0.70, 0.20, 0.10]
    )
    assert proposal.posterior_temperature == 1.5
    assert training.bootstrap_sleep_epochs == 12
    assert training.observed_sweeps == 3
    assert training.sleep_replay_every_smc_updates == 4
    assert training.q_gradient_clip_norm == 20.0
    assert training.prior_gradient_clip_norm == 5.0
    assert training.smoke_min_bootstrap_updates == 128
    assert training.min_validation_q_is_ess_fraction == pytest.approx(0.05)
    assert training.max_validation_q_is_max_weight == pytest.approx(0.80)
    assert cfg["training"]["best_checkpoint_metric"] == (
        "validation_smc_cross_entropy"
    )


def test_adaptive_smc_mode_is_checkpoint_and_inference_metadata_compatible() -> None:
    config = _production_config()
    objective = config["amortized"]["objective"]

    assert objective_mode(objective) == "adaptive_smc_wake"
    assert not objective_uses_truth(objective)
    summary = architecture_summary(config)
    assert summary["objective"]["loss"] == "adaptive_smc_wake"
    assert summary["objective"]["kl_estimator"] == (
        "adaptive_bridge_smc_inclusive_distillation"
    )
    assert summary["objective"]["adaptive_smc"]["n_particles"] == 64
    assert summary["objective"]["adaptive_smc"]["hard_fallback"][
        "n_particles"
    ] == 128


def test_smoke_runs_a_minimum_number_of_fresh_sleep_updates() -> None:
    production = adaptive_training_config(_production_config())
    smoke = _smoke_training_config(production, train_objects=96)

    assert smoke.micro_batch_size == 32
    assert smoke.bootstrap_sleep_epochs == 43
    assert smoke.bootstrap_sleep_epochs * 3 >= 128
    assert smoke.observed_sweeps == 1


def test_final_receipt_requires_q_only_importance_support() -> None:
    config = _production_config()
    training = adaptive_training_config(config)
    primary, fallback, proposal = adaptive_smc_configs(config)
    runtime = SimpleNamespace(
        parameter_names=("x0", "x1"),
        train_arrays=SimpleNamespace(flux=np.zeros((96, 2))),
        validation_arrays=SimpleNamespace(flux=np.zeros((32, 2))),
        latent_spec=SimpleNamespace(normalization="standardized_logit"),
    )
    validation = {
        "selection_alpha": 0.3,
        "selection_alpha_mc_relative_error": 0.05,
        "median_mutation_acceptance": 0.3,
        "posterior_full_entropy_mc": 12.0,
        "median_beta_final": 1.0,
        "median_final_ess_fraction": 0.8,
        "hard_fraction_after_fallback": 0.1,
        "median_unique_ancestor_fraction": 0.5,
        "median_epsilon_squared_jump": 1.0,
        "validation_q_is_ess_fraction": 0.05,
        "validation_q_is_max_weight": 0.8,
        "validation_smc_cross_entropy": 10.0,
    }
    prior_rows = [
        {
            "update_applied": True,
            "prior_kl_proposed": 0.01,
            "grads_finite": True,
            "raw_grad_norm": 1.0,
            "rejection_code": 0,
        }
    ]
    log_rows = [
        {"phase": "bootstrap_sleep", "grad_clipped": False},
        {
            "phase": "observed_smc",
            "q_grad_clipped": False,
            "q_update_applied": True,
        },
    ]
    common = dict(
        runtime=runtime,
        training=training,
        primary=primary,
        fallback=fallback,
        proposal=proposal,
        prior_rows=prior_rows,
        log_rows=log_rows,
        best_cross_entropy=10.0,
        best_epoch_label="observed_sweep_1",
        elapsed_seconds=1.0,
        smoke=True,
        alpha_preflight={"finite": True, "nonzero": True},
    )

    passing = _final_training_receipt(validation_rows=[validation], **common)
    collapsed = _final_training_receipt(
        validation_rows=[
            {
                **validation,
                "validation_q_is_ess_fraction": 1.0 / 64.0,
                "validation_q_is_max_weight": 0.99,
            }
        ],
        **common,
    )

    assert passing["status"] == "PASS"
    assert collapsed["status"] == "FAIL"
    assert not collapsed["checks"]["q_only_is_ess_fraction_adequate"]
    assert not collapsed["checks"]["q_only_is_not_single_weight_dominated"]


def test_selection_normalizer_does_not_change_normalized_object_weights() -> None:
    raw_log_weights = jnp.asarray([-8.0, -4.0, -3.0, -1.0])
    log_alpha = jnp.asarray(-0.73)
    baseline = jax.nn.softmax(raw_log_weights)
    selected = jax.nn.softmax(raw_log_weights - log_alpha)
    assert jnp.allclose(baseline, selected, atol=1.0e-7)


def test_production_initial_particles_follow_exact_nominal_mixture() -> None:
    count = 100_000
    candidates = jnp.broadcast_to(
        jnp.arange(3, dtype=jnp.float32)[:, None, None, None],
        (3, count, 1, 1),
    )
    selected, component = _select_exact_mixture_candidates(
        jax.random.PRNGKey(55),
        candidates,
        jnp.asarray([0.70, 0.20, 0.10]),
    )
    frequencies = np.bincount(np.asarray(component).reshape(-1), minlength=3) / count

    assert frequencies == pytest.approx([0.70, 0.20, 0.10], abs=0.005)
    assert jnp.array_equal(selected[..., 0], component.astype(selected.dtype))


def test_cost_estimate_counts_mutation_and_final_target_evaluations(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifests": {
                    "train": {"count": 512},
                    "validation": {"count": 64},
                }
            }
        ),
        encoding="utf-8",
    )
    payload = estimate(
        Path(
            "configs/experiments/"
            "feniks_selfsup_adaptive_smcwake_parentprior_selection_r25.yaml"
        ),
        manifest,
    )
    per_object = payload["smc_per_object_per_sweep"]

    assert per_object["easy_zero_resamples"] == 128
    assert per_object["typical_one_resample"] == 256
    assert per_object["two_resamples"] == 384
    assert per_object["absolute_configured_primary_plus_fallback_upper_bound"] == 7680


def test_selection_corrected_toy_recovers_parent_mean() -> None:
    rng = np.random.default_rng(260821)
    parent = rng.normal(0.0, 1.0, size=200_000)
    observed = parent + rng.normal(0.0, 0.5, size=parent.size)
    selected = jnp.asarray(parent[observed > 0.0], dtype=jnp.float32)
    survey_scale = jnp.sqrt(jnp.asarray(1.0 + 0.5**2))

    def fit(corrected: bool):
        mean = jnp.asarray(0.0)
        for _ in range(400):
            gradient = jax.grad(
                lambda value: 0.5 * jnp.mean(jnp.square(selected - value))
                + (
                    log_ndtr(value / survey_scale)
                    if corrected
                    else jnp.asarray(0.0)
                )
            )(mean)
            mean = mean - 0.05 * gradient
        return float(mean)

    uncorrected = fit(False)
    corrected = fit(True)
    assert uncorrected > 0.60
    assert abs(corrected) < 0.08
    assert abs(corrected) < abs(uncorrected)


def test_two_cpu_device_smc_updates_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    code = textwrap.dedent(
        f"""
        import pathlib
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import log_ndtr
        from euclid_dsps.amortized.adaptive_bridge_smc import (
            AdaptiveBridgeSMCConfig, run_adaptive_bridge_smc
        )
        from euclid_dsps.amortized.adaptive_smc_training import (
            SMCPosteriorBatch, apply_prior_macro_update,
            make_component_optimizer, make_pmap_q_smc_step, snapshot_model
        )
        from euclid_dsps.amortized.adaptive_smc_trainer import (
            _replicate_model_for_pmap, _replicate_tree, _shard_posterior,
            _unreplicate_tree
        )
        from euclid_dsps.amortized.elbo import AmortizedModel
        from euclid_dsps.amortized.flows import RealNVPPrior
        from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
        from euclid_dsps.amortized.selection_correction import (
            selection_log_alpha_from_log_beta
        )
        from euclid_dsps.calibration import GlobalSedScaleState

        assert len(jax.local_devices()) == 2, jax.local_devices()
        K, N, D = 64, 4, 2
        qscale = 0.7
        initial = qscale * jax.random.normal(jax.random.PRNGKey(0), (2, K, N, D))
        keys = jax.random.split(jax.random.PRNGKey(1), 2)

        @jax.pmap
        def toy_smc(key, particles):
            def lp(x, scale):
                return -0.5 * jnp.sum((x / scale) ** 2, axis=-1) - D * jnp.log(scale)
            def forward(epsilon):
                return qscale * epsilon, jnp.full(epsilon.shape[:-1], D * jnp.log(qscale))
            def inverse(x):
                return x / qscale, jnp.full(x.shape[:-1], D * jnp.log(qscale))
            return run_adaptive_bridge_smc(
                key=key,
                initial_particles=particles,
                log_r0_fn=lambda x: lp(x, qscale),
                log_target_fn=lambda x: lp(x, 1.0),
                epsilon_to_x_fn=forward,
                x_to_epsilon_fn=inverse,
                config=AdaptiveBridgeSMCConfig(
                    n_particles=K,
                    hard_min_mutation_acceptance=0.0,
                ),
            )

        result = toy_smc(keys, initial)
        assert jnp.all(result.beta_final == 1.0)
        encoder = ConditionalFlowEncoder(
            jax.random.PRNGKey(2), input_dim=6, latent_dim=D,
            hidden_sizes=(8,), activation='gelu', log_std_min=-4.0,
            log_std_max=3.0, initial_log_std=0.0, family='realnvp',
            n_layers=2, hidden_size=8, output_space='latent_x'
        )
        prior = RealNVPPrior(
            jax.random.PRNGKey(3), latent_dim=D, n_layers=2,
            hidden_size=8, permutation='roll', init='identity'
        )
        model = AmortizedModel(
            encoder, prior, GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0))
        )
        posterior = SMCPosteriorBatch(
            particles=result.final_particles.transpose(1, 0, 2, 3).reshape(K, 2*N, D),
            normalized_weights=result.final_normalized_weights.transpose(1, 0, 2).reshape(K, 2*N),
            eligible=jnp.ones((2*N,), dtype=bool), beta_final=jnp.ones((2*N,)),
            final_ess=jnp.full((2*N,), K), final_max_weight=jnp.full((2*N,), 1/K),
            mutation_acceptance=jnp.full((2*N,), 0.3),
            final_rw_scale=jnp.full((2*N,), 0.6),
            unique_ancestor_fraction=jnp.ones((2*N,)),
            epsilon_squared_jump=jnp.ones((2*N,)),
            logZ_estimate=jnp.zeros((2*N,)),
            fallback_attempted=jnp.zeros((2*N,), dtype=bool),
            fallback_succeeded=jnp.zeros((2*N,), dtype=bool),
        )
        qopt = make_component_optimizer(
            learning_rate=1e-3, gradient_clip_norm=5.0, weight_decay=0.0
        )
        qstate = qopt.init(eqx.filter(model.encoder, eqx.is_inexact_array))
        devices = tuple(jax.local_devices())
        pstep = make_pmap_q_smc_step(optimizer=qopt, gradient_clip_norm=5.0)
        features = jax.random.normal(jax.random.PRNGKey(4), (2*N, 6)).reshape(2, N, 6)
        model_rep, qstate_rep, _metrics, step_metrics = pstep(
            _replicate_tree(model, devices), _replicate_tree(qstate, devices),
            features, _shard_posterior(posterior, 2)
        )
        assert jnp.all(step_metrics.update_applied)
        model = _unreplicate_tree(model_rep)
        popt = make_component_optimizer(
            learning_rate=1e-4, gradient_clip_norm=5.0, weight_decay=0.0
        )
        pstate = popt.init(eqx.filter(model.prior, eqx.is_inexact_array))
        def selection(candidate, key):
            base = jax.random.normal(jax.random.PRNGKey(99), (256, D))
            values, _ = candidate.prior.forward(base)
            return selection_log_alpha_from_log_beta(log_ndtr(values[:, 0]))
        model, pstate, prior_metrics = apply_prior_macro_update(
            model=model, prior_snapshot=snapshot_model(model), optimizer=popt,
            optimizer_state=pstate, posterior=posterior,
            trust_key=jax.random.PRNGKey(5), selection_key=jax.random.PRNGKey(6),
            selection_log_alpha_fn=selection, trust_samples=128,
            trust_strength=0.2, max_kl_per_dimension=0.05,
            max_alpha_mc_relative_error=0.15, gradient_clip_norm=5.0
        )
        assert prior_metrics.update_applied
        model_rep = _replicate_model_for_pmap(model, devices)
        model_rep, qstate_rep, _metrics, second_step_metrics = pstep(
            model_rep, qstate_rep, features, _shard_posterior(posterior, 2)
        )
        assert jnp.all(second_step_metrics.update_applied)
        model = _unreplicate_tree(model_rep)
        bundle = (model, _unreplicate_tree(qstate_rep), pstate)
        path = pathlib.Path({str(tmp_path / 'state.eqx')!r})
        eqx.tree_serialise_leaves(path, bundle)
        restored = eqx.tree_deserialise_leaves(path, bundle)
        leaves_a = [x for x in jax.tree_util.tree_leaves(bundle) if eqx.is_array(x)]
        leaves_b = [x for x in jax.tree_util.tree_leaves(restored) if eqx.is_array(x)]
        assert all(jnp.array_equal(a, b) for a, b in zip(leaves_a, leaves_b, strict=True))
        print('PASS')
        """
    )
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS" in completed.stdout
