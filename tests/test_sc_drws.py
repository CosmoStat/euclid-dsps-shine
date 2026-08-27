from __future__ import annotations

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

from euclid_dsps.amortized import sc_drws_trainer
from euclid_dsps.amortized.elbo import AmortizedModel
from euclid_dsps.amortized.flows import StandardNormalPrior
from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
from euclid_dsps.amortized.sc_drws import (
    HARD_EXPANSION_PROPOSAL,
    JOINT_PROPOSAL,
    WARMUP_PROPOSAL,
    DefensiveImportanceBatch,
    ImportanceDiagnostics,
    SCDrwsSchedule,
    contains_selection_in_object_weights,
    deterministic_multiple_mixture,
    entropy_penalty_factor,
    expand_defensive_importance,
    flow_scale_clamp,
    log_std_floor,
    parent_to_selected_weights,
    prior_support_gate,
    q_weight_temperature,
    run_defensive_importance,
)
from euclid_dsps.amortized.sc_drws_trainer import (
    SCDrwsTrainingState,
    _load_state,
    _macro_slices,
    _pack_first_pass,
    _save_state,
    validate_sc_drws_config,
)
from euclid_dsps.calibration import GlobalSedScaleState
from euclid_dsps.config import load_config
from scripts.estimate_feniks_sc_drws_cost import estimate


def _model() -> AmortizedModel:
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(1),
        input_dim=4,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4.0,
        log_std_max=2.5,
        initial_log_std=0.25,
        family="realnvp",
        n_layers=2,
        hidden_size=8,
        init_scale=0.0,
        output_space="latent_x",
    )
    return AmortizedModel(
        encoder=encoder,
        prior=StandardNormalPrior(latent_dim=2),
        sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)),
    )


def test_variance_control_schedules_reach_exact_endpoints() -> None:
    schedule = SCDrwsSchedule()
    assert float(log_std_floor(1, schedule)) == pytest.approx(-1.5)
    assert float(log_std_floor(100, schedule)) == pytest.approx(-4.0)
    assert float(flow_scale_clamp(1, final_value=0.45, schedule=schedule)) == pytest.approx(0.15)
    assert float(flow_scale_clamp(60, final_value=0.45, schedule=schedule)) == pytest.approx(0.45)
    assert float(q_weight_temperature(1, schedule)) == pytest.approx(0.5)
    assert float(q_weight_temperature(80, schedule)) == pytest.approx(1.0)
    assert entropy_penalty_factor(91, schedule) == 0.0


def test_fixed_particle_mixtures_use_realized_component_counts() -> None:
    realized = JOINT_PROPOSAL.realized(128)
    counts = np.asarray([item[2] * 128 for item in realized.components])
    np.testing.assert_allclose(counts, np.round(counts))
    assert counts.sum() == 128
    combined = deterministic_multiple_mixture(
        realized,
        HARD_EXPANSION_PROPOSAL.realized(384),
        first_count=128,
        additional_count=384,
    )
    assert sum(item[2] for item in combined.components) == pytest.approx(1.0)


def test_defensive_proposal_is_antithetic_and_complete_density_is_finite() -> None:
    model = _model()
    features = jnp.ones((3, 4))
    result = run_defensive_importance(
        model_snapshot=model,
        features=features,
        key=jax.random.PRNGKey(2),
        n_particles=64,
        proposal=WARMUP_PROPOSAL,
        logtarget_fn=lambda x: model.prior.log_prob(x),
    )
    assert result.particles.shape == (64, 3, 2)
    assert bool(jnp.all(jnp.isfinite(result.logproposal)))
    np.testing.assert_allclose(
        np.asarray(result.diagnostics.normalized_weights.sum(axis=0)), 1.0
    )


def test_adaptive_k128_to_k512_recomputes_complete_mis_density() -> None:
    model = _model()
    features = jnp.ones((2, 4))
    def target(x):
        return model.prior.log_prob(x - 1.0)
    first = run_defensive_importance(
        model_snapshot=model,
        features=features,
        key=jax.random.PRNGKey(3),
        n_particles=128,
        proposal=JOINT_PROPOSAL,
        logtarget_fn=target,
    )
    expanded = expand_defensive_importance(
        model_snapshot=model,
        features=features,
        key=jax.random.PRNGKey(4),
        first=first,
        first_proposal=JOINT_PROPOSAL,
        additional_proposal=HARD_EXPANSION_PROPOSAL,
        additional_particles=384,
        logtarget_fn=target,
    )
    assert expanded.expanded_particles.shape == (512, 2, 2)
    assert bool(jnp.all(jnp.isfinite(expanded.expanded_logproposal)))
    np.testing.assert_allclose(
        np.asarray(expanded.expanded_diagnostics.normalized_weights.sum(axis=0)),
        1.0,
        atol=1e-6,
    )


def test_hard_expansion_host_pack_is_writable() -> None:
    particles = jnp.zeros((128, 2, 3))
    weights = jnp.full((128, 2), 1.0 / 128.0)
    first = DefensiveImportanceBatch(
        particles=particles,
        logproposal=jnp.zeros((128, 2)),
        diagnostics=ImportanceDiagnostics(
            normalized_weights=weights,
            logweight=jnp.zeros((128, 2)),
            ess=jnp.asarray([1.0, 2.0]),
            ess_fraction=jnp.asarray([1.0 / 128.0, 2.0 / 128.0]),
            max_weight=jnp.asarray([1.0, 0.5]),
            finite=jnp.asarray([True, True]),
            hard=jnp.asarray([True, True]),
        ),
    )

    packed = _pack_first_pass(first, maximum_particles=512)
    mutable_keys = (
        "particles",
        "weights",
        "q_eligible",
        "prior_eligible",
        "finite",
        "ess_fraction",
        "max_weight",
        "expanded",
        "unresolved",
    )
    assert all(packed[key].flags.writeable for key in mutable_keys)

    selected = np.asarray([0, 1])
    packed["ess_fraction"][selected] = [0.2, 0.3]
    packed["max_weight"][selected] = [0.4, 0.5]
    packed["finite"][selected] = [True, False]
    packed["expanded"][selected] = True
    packed["unresolved"][selected] = [False, True]
    packed["prior_eligible"][selected] = [True, False]
    np.testing.assert_allclose(packed["ess_fraction"], [0.2, 0.3])


def test_selection_is_excluded_from_object_weights_and_selected_prior_is_derived() -> None:
    assert not contains_selection_in_object_weights("loglike + logprior - logr")
    assert contains_selection_in_object_weights("loglike + logprior + log_beta - logr")
    weights, alpha = parent_to_selected_weights(jnp.asarray([0.0, 0.5, 1.0]))
    assert float(alpha) == pytest.approx(0.5)
    np.testing.assert_allclose(np.asarray(weights), [0.0, 1.0 / 3.0, 2.0 / 3.0])


def test_prior_support_gate_fails_closed_on_unresolved_or_dominant_weights() -> None:
    accepted = prior_support_gate(
        ess_fraction=[0.1] * 10,
        max_weight=[0.2] * 10,
        finite=[True] * 10,
        unresolved=[False] * 10,
        minimum_finite_objects=8,
        minimum_median_ess_fraction=0.05,
        maximum_median_weight=0.9,
        maximum_unresolved_fraction=0.02,
    )
    assert accepted.accepted
    rejected = prior_support_gate(
        ess_fraction=[0.1] * 10,
        max_weight=[0.2] * 10,
        finite=[True] * 10,
        unresolved=[True] + [False] * 9,
        minimum_finite_objects=8,
        minimum_median_ess_fraction=0.05,
        maximum_median_weight=0.9,
        maximum_unresolved_fraction=0.02,
    )
    assert not rejected.accepted
    assert rejected.reason == "unresolved_fraction_above_gate"


@pytest.mark.parametrize(
    "name",
    ("feniks_sc_drws_r29_historical.yaml", "feniks_sc_drws_r29_current.yaml"),
)
def test_sc_drws_configs_are_truth_free_and_selection_corrected(name: str) -> None:
    config = load_config(f"configs/experiments/{name}")
    assert validate_sc_drws_config(config)["status"] == "PASS"
    source = config["amortized"]["sc_drws"]
    assert source["truth_allowed"] is False
    assert source["phase_b"]["likelihood"] == "gaussian"
    assert source["hard_mis"]["maximum_particles"] == 512
    assert config["amortized"]["latent"]["normalization"] == "bounded_mixed_warp"


def test_four_device_pmap_q_update_regression() -> None:
    code = textwrap.dedent(
        """
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import optax
        from euclid_dsps.amortized.adaptive_smc_trainer import _replicate_model_for_pmap
        from euclid_dsps.amortized.elbo import AmortizedModel
        from euclid_dsps.amortized.flows import StandardNormalPrior
        from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
        from euclid_dsps.amortized.sc_drws import make_pmap_sc_drws_q_step
        from euclid_dsps.calibration import GlobalSedScaleState

        devices = jax.local_devices()
        assert len(devices) == 4
        encoder = ConditionalFlowEncoder(
            jax.random.PRNGKey(1), input_dim=4, latent_dim=2,
            hidden_sizes=(8,), activation='gelu', log_std_min=-4.0,
            log_std_max=2.5, initial_log_std=0.25, family='realnvp',
            n_layers=2, hidden_size=8, init_scale=0.0,
            output_space='latent_x')
        model = AmortizedModel(
            encoder=encoder, prior=StandardNormalPrior(latent_dim=2),
            sed_scale=GlobalSedScaleState(log_alpha_sed=jnp.asarray(0.0)))
        optimizer = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(1e-4))
        state = optimizer.init(eqx.filter(model.encoder, eqx.is_inexact_array))
        replicate = lambda tree: _replicate_model_for_pmap(tree, tuple(devices))
        step = make_pmap_sc_drws_q_step(optimizer=optimizer, gradient_clip_norm=10.0)
        features = jnp.ones((4, 2, 4))
        particles = jax.random.normal(jax.random.PRNGKey(2), (4, 8, 2, 2))
        weights = jnp.ones((4, 8, 2)) / 8.0
        mask = jnp.ones((4, 2), dtype=bool)
        keys = jax.random.split(jax.random.PRNGKey(3), 4)
        _model, _state, metrics, _details = step(
            replicate(model), replicate(state), features, particles, weights,
            mask, keys, 0.5, -1.5, 0.15, 0.0, 1.0, 0.02, 0.0)
        assert bool(jnp.all(metrics.grads_finite))
        assert bool(jnp.all(metrics.update_applied))
        print('SC_DRWS_PMAP_PASS')
        """
    )
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "SC_DRWS_PMAP_PASS" in result.stdout


def test_launchers_encode_sixteen_h100_pilot_and_resumable_training() -> None:
    root = Path(__file__).resolve().parents[1]
    submit = (root / "scripts/submit_feniks_rws_recovery.sh").read_text()
    pilot = (root / "scripts/feniks_rws_recovery_pilot_h100.slurm").read_text()
    confirmation = (
        root / "scripts/feniks_rws_recovery_confirm_h100.slurm"
    ).read_text()
    full = (root / "scripts/feniks_sc_drws_full_h100.slurm").read_text()
    inference = (root / "scripts/feniks_sc_drws_inference_h100.slurm").read_text()
    entrypoint = (root / "scripts/train_feniks_sc_drws.py").read_text()
    assert "--array=0-3%4" in submit
    assert "#SBATCH --gres=gpu:4" in pilot
    assert "full_dataset_not_submitted=1" in submit
    assert all("--resume-state" in worker for worker in (pilot, confirmation, full))
    assert "--require-full-dataset" in full
    gpu_workers = (pilot, confirmation, full, inference)
    for worker in gpu_workers:
        assert "export EUCLID_DSPS_JAX_PLATFORMS=cuda" in worker
        assert "export EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0" in worker
        assert "export EUCLID_DSPS_REQUIRE_GPU=1" in worker
        assert "export EUCLID_DSPS_EXPECTED_GPU_NAME=NVIDIA" in worker
    runtime_bootstrap = entrypoint.index("apply_jax_runtime_env(")
    trainer_import = entrypoint.index(
        "from euclid_dsps.amortized.sc_drws_trainer import train_feniks_sc_drws"
    )
    assert runtime_bootstrap < trainer_import


def test_training_resume_state_round_trip_and_provenance_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sc_drws_trainer, "latent_spec_hash", lambda _value: "latent")
    monkeypatch.setattr(sc_drws_trainer, "feature_stats_hash", lambda _value: "features")
    runtime = SimpleNamespace(latent_spec=object(), feature_stats=object())
    state = SCDrwsTrainingState(
        model=jnp.asarray([1.0, 2.0]),
        ema_encoder=jnp.asarray([3.0]),
        q_warmup_optimizer_state=jnp.asarray([4.0]),
        q_joint_optimizer_state=jnp.asarray([5.0]),
        prior_optimizer_state=jnp.asarray([6.0]),
        epoch=jnp.asarray(17),
        wake_updates=jnp.asarray(4),
        prior_updates=jnp.asarray(2),
        random_key=jax.random.PRNGKey(9),
        reference_entropy=jnp.asarray(12.5),
    )
    path = tmp_path / "latest.eqx"
    config = {"amortized": {"sc_drws": {"name": "test"}}}
    _save_state(path, state, config=config, runtime=runtime)
    loaded = _load_state(path, state, config=config, runtime=runtime)
    assert int(loaded.epoch) == 17
    np.testing.assert_allclose(np.asarray(loaded.model), [1.0, 2.0])
    with pytest.raises(ValueError, match="provenance mismatch"):
        _load_state(path, state, config={"changed": True}, runtime=runtime)


def test_cost_estimate_accounts_for_sleep_and_hard_expansion() -> None:
    value = estimate(100, 0.2, 1000.0)
    costs = value["latent_object_dsps_evaluations"]
    assert costs["phase_a_wake_k64"] == 96_000
    assert costs["phase_b_first_pass_k128"] == 384_000
    assert costs["phase_b_hard_additional_k384"] == 230_400
    assert costs["selected_sleep_candidate_factor_8"] == 108_000
    assert value["calibrated_runtime"]["hours_per_seed_four_h100"] > 0


def test_prior_macro_slices_cover_every_object_without_tiny_tail() -> None:
    slices = _macro_slices(2100, 1024, 128)
    assert slices == [(0, 1024), (1024, 2100)]
    covered = np.concatenate([np.arange(start, stop) for start, stop in slices])
    np.testing.assert_array_equal(covered, np.arange(2100))
