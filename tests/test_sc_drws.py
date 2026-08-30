from __future__ import annotations

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

from euclid_dsps.amortized import sc_drws_trainer
from euclid_dsps.amortized import train as amortized_train
from euclid_dsps.amortized.elbo import AmortizedModel
from euclid_dsps.amortized.flows import StandardNormalPrior
from euclid_dsps.amortized.posterior import ConditionalFlowEncoder
from euclid_dsps.amortized.sc_drws import (
    HARD_EXPANSION_PROPOSAL,
    JOINT_PROPOSAL,
    WARMUP_HARD_EXPANSION_PROPOSAL,
    WARMUP_PROPOSAL,
    DefensiveImportanceBatch,
    ImportanceDiagnostics,
    SCDrwsSchedule,
    adaptive_q_weight_temperature,
    contains_selection_in_object_weights,
    deterministic_multiple_mixture,
    dominant_component_labels,
    entropy_penalty_factor,
    expand_defensive_importance,
    flow_gradient_multiplier,
    flow_scale_clamp,
    log_std_floor,
    parent_to_selected_weights,
    population_posterior_stability,
    prior_support_gate,
    proposal_component_labels,
    q_weight_temperature,
    run_defensive_importance,
    tempered_q_weights,
    update_kind_for_epoch,
    warmup_cosine_learning_rate,
)
from euclid_dsps.amortized.sc_drws_trainer import (
    SCDrwsTrainingState,
    _apply_prior_updates,
    _load_state,
    _macro_slices,
    _pack_first_pass,
    _save_components,
    _save_state,
    _support_probe_is_better,
    _truncate_csv_after_epoch,
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
    assert float(
        flow_scale_clamp(1, final_value=0.45, schedule=schedule)
    ) == pytest.approx(0.15)
    assert float(
        flow_scale_clamp(60, final_value=0.45, schedule=schedule)
    ) == pytest.approx(0.45)
    assert float(q_weight_temperature(1, schedule)) == pytest.approx(0.5)
    assert float(q_weight_temperature(80, schedule)) == pytest.approx(1.0)
    assert entropy_penalty_factor(91, schedule) == 0.0


def test_production_curriculum_schedules_are_slow_then_exact() -> None:
    schedule = SCDrwsSchedule(
        sleep_only_bootstrap_epochs=16,
        flow_freeze_epochs=12,
        flow_thaw_end_epoch=40,
        flow_final_gradient_multiplier=0.5,
        q_weight_temperature_wake_updates=640,
        adaptive_q_temperature=True,
        q_temperature_minimum_ess_fraction=0.025,
        q_temperature_low_support_cap=0.65,
        q_temperature_force_start_epoch=100,
        q_temperature_force_end_epoch=135,
    )
    assert all(
        update_kind_for_epoch(epoch, schedule) == "sleep" for epoch in range(1, 17)
    )
    assert update_kind_for_epoch(20, schedule) == "wake"
    assert float(flow_gradient_multiplier(12, schedule)) == 0.0
    assert float(flow_gradient_multiplier(40, schedule)) == pytest.approx(0.5)
    assert float(
        adaptive_q_weight_temperature(
            400, epoch=80, median_ess_fraction=1.0 / 64.0, schedule=schedule
        )
    ) == pytest.approx(0.65)
    assert float(
        adaptive_q_weight_temperature(
            400, epoch=135, median_ess_fraction=1.0 / 64.0, schedule=schedule
        )
    ) == pytest.approx(1.0)


def test_q_tempering_uses_logweights_before_normalized_weight_underflow() -> None:
    normalized = jnp.asarray([[1.0], [0.0]], dtype=jnp.float32)
    logweights = jnp.asarray([[0.0], [-20.0]], dtype=jnp.float32)
    tempered = tempered_q_weights(
        normalized,
        0.5,
        exact_logweights=logweights,
    )
    exact = tempered_q_weights(
        normalized,
        1.0,
        exact_logweights=logweights,
    )
    assert float(tempered[1, 0]) > 0.0
    np.testing.assert_allclose(
        np.asarray(exact),
        np.asarray(jax.nn.softmax(logweights, axis=0)),
    )


def test_phase_local_learning_rate_has_warmup_peak_and_cosine_floor() -> None:
    values = dict(
        initial_value=3.0e-6,
        peak_value=5.0e-5,
        end_value=1.0e-5,
        warmup_steps=10,
        total_steps=101,
    )
    assert float(warmup_cosine_learning_rate(0, **values)) == pytest.approx(3.0e-6)
    assert float(warmup_cosine_learning_rate(10, **values)) == pytest.approx(5.0e-5)
    assert float(warmup_cosine_learning_rate(100, **values)) == pytest.approx(1.0e-5)


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


def test_dominant_weight_component_attribution_matches_deterministic_draws() -> None:
    labels = proposal_component_labels(WARMUP_PROPOSAL, 64)
    assert labels.shape == (64,)
    assert np.sum(labels == "posterior_t1") == 35
    weights = np.zeros((64, 2))
    weights[0, 0] = 1.0
    weights[-1, 1] = 1.0
    np.testing.assert_array_equal(
        dominant_component_labels(weights, labels),
        ["posterior_t1", "prior"],
    )
    assert proposal_component_labels(WARMUP_HARD_EXPANSION_PROPOSAL, 192).shape == (
        192,
    )


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
        "logweights",
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


def test_hard_expansion_pack_preserves_importance_precision() -> None:
    first = DefensiveImportanceBatch(
        particles=np.zeros((2, 1, 3), dtype=np.float64),
        logproposal=np.zeros((2, 1), dtype=np.float64),
        diagnostics=ImportanceDiagnostics(
            normalized_weights=np.full((2, 1), 0.5, dtype=np.float64),
            logweight=np.zeros((2, 1), dtype=np.float64),
            ess=np.asarray([2.0]),
            ess_fraction=np.asarray([1.0]),
            max_weight=np.asarray([0.5]),
            finite=np.asarray([True]),
            hard=np.asarray([False]),
        ),
    )
    packed = _pack_first_pass(first, maximum_particles=4)
    assert packed["particles"].dtype == np.float64
    assert packed["weights"].dtype == np.float64
    assert packed["logweights"].dtype == np.float64


def test_sc_drws_checkpoint_sidecar_loads_with_legacy_latent_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "legacy.eqx"
    model = _model()
    sc_drws_trainer.eqx.tree_serialise_leaves(checkpoint, model)
    checkpoint.with_suffix(".eqx.json").write_text(
        '{"latent_transform_hash": "bounded-mixed-warp-hash"}'
    )
    monkeypatch.setattr(
        amortized_train, "_latent_spec_for_amortized_config", lambda _config: object()
    )
    monkeypatch.setattr(
        amortized_train, "latent_spec_hash", lambda _spec: "bounded-mixed-warp-hash"
    )
    monkeypatch.setattr(
        amortized_train,
        "build_amortized_model",
        lambda _config, _key, *, latent_spec: _model(),
    )

    restored = amortized_train.load_checkpoint(checkpoint, {"amortized": {}})

    assert isinstance(restored, AmortizedModel)
    assert isinstance(restored.prior, StandardNormalPrior)

    checkpoint.with_suffix(".eqx.json").write_text(
        '{"latent_transform_hash": "wrong-hash"}'
    )
    with pytest.raises(ValueError, match="latent normalization hash"):
        amortized_train.load_checkpoint(checkpoint, {"amortized": {}})


def test_sc_drws_checkpoint_sidecar_writes_generic_latent_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _model()
    monkeypatch.setattr(
        sc_drws_trainer, "latent_spec_hash", lambda _spec: "latent-hash"
    )
    monkeypatch.setattr(
        sc_drws_trainer,
        "latent_spec_to_jsonable",
        lambda _spec: {"normalization": "bounded_mixed_warp"},
    )
    monkeypatch.setattr(
        sc_drws_trainer, "feature_stats_hash", lambda _stats: "feature-hash"
    )
    runtime = SimpleNamespace(latent_spec=object(), feature_stats=object())

    records = _save_components(
        tmp_path,
        model=model,
        ema_encoder=model.encoder,
        config={},
        runtime=runtime,
        epoch=8,
        reference_entropy=2.0,
    )

    sidecar = records["raw_model"]
    assert sidecar["latent_spec_hash"] == "latent-hash"
    assert sidecar["latent_transform_hash"] == "latent-hash"
    assert sidecar["latent_spec"] == {"normalization": "bounded_mixed_warp"}


def test_selection_is_excluded_from_object_weights_and_selected_prior_is_derived() -> (
    None
):
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


def test_population_first_prior_gate_keeps_finite_one_particle_objects() -> None:
    accepted = prior_support_gate(
        ess_fraction=[1.0 / 512.0] * 10,
        max_weight=[1.0] * 10,
        finite=[True] * 10,
        unresolved=[True] * 10,
        minimum_finite_objects=1,
        minimum_median_ess_fraction=0.05,
        maximum_median_weight=0.9,
        maximum_unresolved_fraction=0.02,
        population_first=True,
    )
    assert accepted.accepted
    assert accepted.finite_objects == 10
    assert accepted.median_ess_fraction == pytest.approx(1.0 / 512.0)
    assert accepted.median_max_weight == pytest.approx(1.0)
    assert accepted.unresolved_fraction == pytest.approx(1.0)


def test_population_stability_uses_dense_weighted_draws() -> None:
    particles = np.asarray(
        [
            [[-1.0], [1.0], [-0.8], [1.2]],
            [[1.0], [-1.0], [0.8], [-1.2]],
        ]
    )
    weights = np.full((2, 4), 0.5)
    metrics = population_posterior_stability(
        particles, weights, np.asarray([True] * 4)
    )
    assert metrics["population_finite_objects"] == 4
    assert metrics["population_split_mean_standardized_rms"] == pytest.approx(0.0)
    assert np.isfinite(metrics["population_split_std_log_ratio_rms"])


def test_population_first_prior_update_uses_all_finite_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_update(**kwargs):
        calls.append(np.asarray(kwargs["posterior"].eligible))
        metrics = SimpleNamespace(
            update_applied=np.asarray(True),
            rejection_code=np.asarray(0),
            loss=np.asarray(1.0),
            data_nll=np.asarray(0.5),
            selection_log_alpha=np.asarray(-0.2),
            selection_alpha=np.asarray(0.8),
            selection_grads_finite=np.asarray(True),
            data_grads_finite=np.asarray(True),
            trust_grads_finite=np.asarray(True),
            prior_kl_proposed=np.asarray(0.1),
        )
        return kwargs["model"], kwargs["optimizer_state"], metrics

    monkeypatch.setattr(sc_drws_trainer, "apply_prior_macro_update", fake_update)
    particles = np.zeros((4, 4, 2), dtype=np.float32)
    weights = np.zeros((4, 4), dtype=np.float32)
    weights[0] = 1.0
    gate, _state, _model, updates, rows = _apply_prior_updates(
        model="model",
        optimizer="optimizer",
        optimizer_state="state",
        selection_fn="selection",
        particles=particles,
        weights=weights,
        ess_fraction=np.full(4, 0.25),
        max_weight=np.ones(4),
        finite=np.ones(4, dtype=bool),
        unresolved=np.ones(4, dtype=bool),
        prior_cfg={
            "population_first": True,
            "population_stability_diagnostics": True,
            "minimum_finite_objects": 1,
            "minimum_median_ess_fraction": 0.05,
            "maximum_median_weight": 0.9,
            "maximum_unresolved_fraction": 0.02,
            "maximum_alpha_mc_relative_error": 0.15,
            "maximum_kl_per_dimension": 0.05,
            "enforce_alpha_mc_relative_error": False,
            "enforce_hard_trust_region": False,
            "updates_per_macro": 1,
            "trust_samples": 8,
            "trust_strength": 0.2,
        },
        key=jax.random.PRNGKey(1),
        prior_updates=0,
        epoch=61,
        prior_snapshot="snapshot",
    )
    assert gate.accepted
    assert updates == 1
    assert len(calls) == 1
    assert calls[0].tolist() == [True] * 4
    assert rows[0]["population_first"] is True
    assert rows[0]["unresolved_fraction"] == pytest.approx(1.0)
    assert rows[0]["update_applied"] is True


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
    assert config["amortized"]["inference"]["write_truth_snapshot"] is False
    assert config["amortized"]["inference"]["write_truth_diagnostics"] is False


@pytest.mark.parametrize(
    ("name", "layers", "width"),
    (
        ("feniks_sc_drws_r29_historical_production.yaml", 4, 128),
        ("feniks_sc_drws_r29_current_production.yaml", 6, 256),
    ),
)
def test_sc_drws_full_profile_adds_anti_collapse_without_changing_architecture(
    name: str, layers: int, width: int
) -> None:
    config = load_config(f"configs/experiments/{name}")
    assert validate_sc_drws_config(config)["status"] == "PASS"
    amortized = config["amortized"]
    source = amortized["sc_drws"]
    assert amortized["encoder"]["flow_layers"] == layers
    assert amortized["encoder"]["flow_hidden_size"] == width
    assert source["profile"] == "full_production_anti_collapse_v1"
    assert source["phase_a"]["sleep_only_bootstrap_epochs"] == 16
    assert source["phase_a_hard_mis"]["maximum_particles"] == 256
    assert source["optimizer"]["schedule"]["enabled"] is True
    assert source["prior_update"]["population_first"] is True
    assert source["prior_update"]["minimum_finite_objects"] == 1
    assert source["prior_update"]["enforce_alpha_mc_relative_error"] is False
    assert source["prior_update"]["enforce_hard_trust_region"] is False
    assert source["checkpoint_safety"]["rollback_enabled"] is False
    assert source["checkpoint_safety"]["restore_best_at_end"] is False


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
        logweights = jnp.zeros((4, 8, 2))
        mask = jnp.ones((4, 2), dtype=bool)
        keys = jax.random.split(jax.random.PRNGKey(3), 4)
        _model, _state, metrics, _details = step(
            replicate(model), replicate(state), features, particles, weights,
            logweights, mask, keys, 0.5, -1.5, 0.15, 0.0, 1.0, 0.02, 0.0, 0.0)
        assert bool(jnp.all(metrics.grads_finite))
        assert bool(jnp.all(metrics.update_applied))
        updated = [
            value[0] for value in jax.tree_util.tree_leaves(_model.encoder.layers)
            if eqx.is_array(value)
        ]
        original = [
            value for value in jax.tree_util.tree_leaves(model.encoder.layers)
            if eqx.is_array(value)
        ]
        assert all(bool(jnp.array_equal(left, right)) for left, right in zip(updated, original))
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
    full_submit = (root / "scripts/submit_feniks_sc_drws_full.sh").read_text()
    pilot = (root / "scripts/feniks_rws_recovery_pilot_h100.slurm").read_text()
    confirmation = (root / "scripts/feniks_rws_recovery_confirm_h100.slurm").read_text()
    full = (root / "scripts/feniks_sc_drws_full_h100.slurm").read_text()
    full_monitor = (root / "scripts/monitor_feniks_sc_drws_full.sh").read_text()
    inference = (root / "scripts/feniks_sc_drws_inference_h100.slurm").read_text()
    entrypoint = (root / "scripts/train_feniks_sc_drws.py").read_text()
    assert "--array=0-3%4" in submit
    assert "#SBATCH --gres=gpu:4" in pilot
    assert "full_dataset_not_submitted=1" in submit
    assert 'ALLOW_UNCONFIRMED_FULL="${ALLOW_UNCONFIRMED_FULL:-0}"' in full_submit
    assert "EXPLICIT_UNCONFIRMED_FULL_OVERRIDE" in full_submit
    assert "FULL_LAUNCH_AUTHORIZATION.json" in full_submit
    assert "FULL_AUTHORIZATION_RECEIPT" in full
    assert "FULL_AUTHORIZATION_RECEIPT" in full_monitor
    assert "--array=0 --output" in full_submit
    assert "#SBATCH --array=0\n" in full
    assert "--array=0-1" not in full_submit
    assert "#SBATCH --array=0-1" not in full
    assert "SEEDS=(260826)" in full
    assert "SEEDS=(260826)" in full_monitor
    assert "260827" not in full_submit
    assert "260827" not in full
    assert "260827" not in full_monitor
    assert all("--resume-state" in worker for worker in (pilot, confirmation, full))
    assert "--require-full-dataset" in full
    assert 'VALIDATION_INDICES="$MANIFEST_ROOT/confirmation_indices.npy"' in full
    assert '--validation-catalog "$TEST_CATALOG"' in full
    assert "explicit_cross_catalog_train_validation_no_truth" in (
        root / "euclid_dsps/amortized/train.py"
    ).read_text()
    assert 'parser.add_argument("--validation-catalog", type=Path)' in entrypoint
    assert "feniks_sc_drws_r29_historical_production.yaml" in full
    assert "feniks_sc_drws_r29_current_production.yaml" in full
    assert "full_production_anti_collapse_v1" in full
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


def test_full_manifest_uses_confirmation_for_cross_catalog_validation(
    tmp_path: Path,
) -> None:
    full_train = tmp_path / "full_train_indices.npy"
    confirmation = tmp_path / "confirmation_indices.npy"
    np.save(full_train, np.asarray([0, 1, 2], dtype=np.int64))
    np.save(confirmation, np.asarray([0, 1], dtype=np.int64))
    manifest = tmp_path / "manifest.json"
    payload = {
        "c0_scope_statement": sc_drws_trainer.C0_SCOPE_STATEMENT,
        "truth_used_for_training_or_checkpoint_selection": False,
        "selection": {
            "max_mag_ab": 29.0,
            "configured_train_retained_fraction": 0.95,
        },
        "manifests": {
            "full_train": {
                "path": str(full_train),
                "count": 3,
                "sha256": sc_drws_trainer._sha256(full_train),
            },
            "confirmation": {
                "path": str(confirmation),
                "count": 2,
                "sha256": sc_drws_trainer._sha256(confirmation),
            },
        },
        "final_full_dataset_contract": {"expected_rows": 3},
    }
    manifest.write_text(json.dumps(payload))

    result = sc_drws_trainer._validate_manifest(
        manifest,
        train_indices_file=full_train,
        validation_indices_file=confirmation,
        require_full_dataset=True,
    )

    assert result["final_full_dataset_contract"]["expected_rows"] == 3


def test_training_routes_validation_catalog_only_to_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_catalog = tmp_path / "test.parquet"

    def validate_manifest(
        _path,
        *,
        train_indices_file,
        validation_indices_file,
        require_full_dataset,
    ):
        assert train_indices_file == "train.npy"
        assert validation_indices_file == "confirmation.npy"
        assert require_full_dataset is True
        return {}

    def prepare_runtime(
        _config,
        _out,
        *,
        train_indices_file,
        validation_indices_file,
        validation_catalog_path,
    ):
        assert train_indices_file == "train.npy"
        assert validation_indices_file == "confirmation.npy"
        assert validation_catalog_path == validation_catalog
        raise RuntimeError("runtime-routing-pass")

    monkeypatch.setattr(sc_drws_trainer, "validate_sc_drws_config", lambda _: {})
    monkeypatch.setattr(sc_drws_trainer, "_validate_manifest", validate_manifest)
    monkeypatch.setattr(
        sc_drws_trainer,
        "prepare_adaptive_training_runtime",
        prepare_runtime,
    )

    with pytest.raises(RuntimeError, match="runtime-routing-pass"):
        sc_drws_trainer.train_feniks_sc_drws(
            {},
            out_dir=tmp_path / "out",
            train_indices_file="train.npy",
            validation_indices_file="confirmation.npy",
            validation_catalog_path=validation_catalog,
            manifest_file="manifest.json",
            require_full_dataset=True,
        )


def test_training_resume_state_round_trip_and_provenance_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sc_drws_trainer, "latent_spec_hash", lambda _value: "latent")
    monkeypatch.setattr(
        sc_drws_trainer, "feature_stats_hash", lambda _value: "features"
    )
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


def test_resume_truncates_logs_after_last_durable_epoch(tmp_path: Path) -> None:
    path = tmp_path / "training.csv"
    path.write_text(
        "epoch,batch,loss\n"
        "62,1,2.0\n"
        "63,1,1.5\n"
        "64,1,1.4\n"
        "64,2,1.3\n",
        encoding="utf-8",
    )

    _truncate_csv_after_epoch(path, 63)

    assert path.read_text(encoding="utf-8") == (
        "epoch,batch,loss\n"
        "62,1,2.0\n"
        "63,1,1.5\n"
    )
    _truncate_csv_after_epoch(tmp_path / "missing.csv", 63)


def test_cost_estimate_accounts_for_sleep_and_hard_expansion() -> None:
    value = estimate(100, 0.2, 1000.0)
    costs = value["latent_object_dsps_evaluations"]
    assert costs["phase_a_wake_k64"] == 96_000
    assert costs["phase_b_first_pass_k128"] == 384_000
    assert costs["phase_b_hard_additional_k384"] == 230_400
    assert costs["selected_sleep_candidate_factor_8"] == 108_000
    assert value["calibrated_runtime"]["hours_per_seed_four_h100"] > 0


def test_production_cost_includes_phase_a_rescue_and_support_probes() -> None:
    costs = estimate(100, 0.2, None, production=True)["latent_object_dsps_evaluations"]
    assert costs["phase_a_wake_k64"] == 70_400
    assert costs["phase_a_hard_additional_k192"] == 42_240
    assert costs["truth_free_gaussian_k128_support_probes"] == 491_520


def test_prior_macro_slices_cover_every_object_without_tiny_tail() -> None:
    slices = _macro_slices(2100, 1024, 128)
    assert slices == [(0, 1024), (1024, 2100)]
    covered = np.concatenate([np.arange(start, stop) for start, stop in slices])
    np.testing.assert_array_equal(covered, np.arange(2100))


def test_support_probe_prefers_ess_then_lower_dominant_weight() -> None:
    incumbent = {
        "median_ess_fraction": 0.04,
        "median_max_weight": 0.7,
        "unresolved_fraction": 0.2,
    }
    assert _support_probe_is_better(
        {
            "median_ess_fraction": 0.05,
            "median_max_weight": 0.8,
            "unresolved_fraction": 0.3,
        },
        incumbent,
    )
    assert not _support_probe_is_better(
        {
            "median_ess_fraction": 0.04,
            "median_max_weight": 0.9,
            "unresolved_fraction": 0.2,
        },
        incumbent,
    )
    assert _support_probe_is_better(
        {
            **incumbent,
            "epoch": 72,
        },
        {
            **incumbent,
            "epoch": 68,
        },
    )
