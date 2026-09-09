import copy
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_model import _synthetic_context

from euclid_dsps import model as sed
from euclid_dsps.amortized.latent import (
    LatentSpec,
    latent_spec_hash,
    latent_spec_to_jsonable,
    theta_to_x,
    x_to_theta,
    x_to_theta_log_abs_det_jacobian,
)
from euclid_dsps.amortized.likelihood import photometric_loglike
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    posterior_log_target,
)
from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES
from scripts import feniks_precision_night as night


def test_latent_precision_version_roundtrip_and_jacobian():
    legacy = LatentSpec(
        ("z_obs", "dust_av"),
        jnp.array([0.01, 0.0], dtype=jnp.float32),
        jnp.array([4.0, 3.0], dtype=jnp.float32),
        normalization="bounded_mixed_warp",
        transform_family=jnp.array([0, 0]),
        transform_location=jnp.zeros(2),
        transform_lambda=jnp.ones(2),
    )
    spec = replace(legacy, arithmetic_precision="float64_v1")
    x = jnp.array([0.5, -0.2], dtype=jnp.float64)
    assert x_to_theta(x, legacy).dtype == jnp.float32
    assert x_to_theta(x, spec).dtype == jnp.float64
    assert "arithmetic_precision" not in latent_spec_to_jsonable(legacy)
    assert latent_spec_hash(legacy) != latent_spec_hash(spec)
    np.testing.assert_allclose(theta_to_x(x_to_theta(x, spec), spec), x, atol=1e-12)
    _, logdet = jnp.linalg.slogdet(jax.jacfwd(lambda p: x_to_theta(p, spec))(x))
    np.testing.assert_allclose(
        logdet, x_to_theta_log_abs_det_jacobian(x, spec), atol=1e-12
    )


def test_likelihood_precision_and_analytic_gradient():
    y, error = jnp.array([[2e-29]]), jnp.array([[0.2e-29]])

    def loss(f, precision):
        return photometric_loglike(
            y,
            f,
            error,
            jnp.ones((1, 1), bool),
            likelihood_type="gaussian",
            error_floor_frac=0,
            arithmetic_precision=precision,
        ).sum()

    f = jnp.array([[1.5e-29]])
    assert loss(f, "float32_legacy").dtype == jnp.float32
    assert loss(f, "float64_v1").dtype == jnp.float64
    ad = jax.grad(lambda p: loss(p, "float64_v1"))(f)
    np.testing.assert_allclose(ad, (y - f) / error**2, rtol=1e-7)


@pytest.mark.parametrize("igm", ["none", "fsps_madau95"])
def test_production_full_target_all_fifteen_coordinates(igm):
    cfg = dict(
        sfh_model="spline15d",
        agn_model="none",
        dust_model="prospector_fsps",
        igm_model=igm,
        stellar_metallicity_model="lognormal_mdf_fixed_scatter",
        stellar_metallicity_scatter_dex=0.2,
        photometry_integrator="merged_gauss4_v1",
        mdf_weight_precision="float64_v1",
    )
    context = _synthetic_context(cfg)
    theta = jnp.array([0.5, 9.3, -0.2, 0.3, -0.2] + [0.02] * 10)
    params = dict(zip(SPLINE15D_PARAMETER_NAMES, theta, strict=True))
    before = sed.run_spline15d_model_jax(context, params).model_mags
    corrected = copy.copy(context)
    corrected.model_config = dict(cfg, spline_precision="float64_v1")
    spec = LatentSpec(
        SPLINE15D_PARAMETER_NAMES,
        jnp.array([0.01, 7.0, -2.0, 0.0, -2.0] + [-3.0] * 10),
        jnp.array([4.0, 12.0, 1.0, 3.0, 1.0] + [3.0] * 10),
        arithmetic_precision="float64_v1",
    )
    point = theta_to_x(theta, spec)
    from euclid_dsps.amortized.decoder import model_flux_from_x

    args = sed.dynamic_model_args(corrected)
    center = model_flux_from_x(point, spec, corrected, args, spec.names)
    observation = PosteriorObservation(
        center[None, :] * 1.1,
        abs(center[None, :]) * 0.1,
        jnp.ones_like(center[None, :], bool),
    )
    model = SimpleNamespace(
        prior=SimpleNamespace(log_prob=lambda x: -0.5 * jnp.sum(x * x, axis=-1))
    )

    def target(x):
        return posterior_log_target(
            model,
            x[None, None, :],
            observation,
            spec,
            corrected,
            args,
            spec.names,
            dict(type="gaussian", arithmetic_precision="float64_v1"),
        )

    f = jax.jit(lambda x: target(x).logtarget.sum())
    assert target(point).model_flux.dtype == jnp.float64
    assert target(point).loglike.dtype == jnp.float64
    ad = jax.jit(jax.grad(f))(point)
    fd = []
    for i in range(15):
        d = jnp.eye(15)[i] * 1e-5
        fd.append((f(point + d) - f(point - d)) / 2e-5)
    np.testing.assert_allclose(ad, fd, rtol=0.003, atol=0.01)
    np.testing.assert_array_equal(
        before, sed.run_spline15d_model_jax(context, params).model_mags
    )
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            sed.spline_numerical_dtype(corrected.model_config)


def test_night_fails_closed_before_any_command(tmp_path, monkeypatch):
    monkeypatch.setattr(
        night.subprocess, "run", lambda *a, **kw: pytest.fail("training started")
    )
    result = night.continue_night(
        tmp_path,
        {},
        {},
        None,
        None,
        None,
        dict(candidate_numerical_checks="NOT_PASSED"),
        1.0,
    )
    assert result["status"] == "BLOCKED_NUMERICAL_QUALIFICATION"
    assert not result["training_started"]
    assert not result["population_training_started"]


def test_night_gate_requires_all_six_cases():
    with pytest.raises(ValueError):
        night.require_qualification(dict(candidate_numerical_checks="PASS", cases=[]))
    report = dict(
        candidate_numerical_checks="PASS",
        cases=[dict(variant="spline64", numerical_checks="PASS") for _ in range(6)],
    )
    night.require_qualification(report)
    report["cases"][-1]["numerical_checks"] = "NOT_PASSED"
    with pytest.raises(ValueError):
        night.require_qualification(report)


def test_night_cohorts_preserve_split_and_hashes(tmp_path):
    source, root = tmp_path / "source", tmp_path / "night"
    source.mkdir()
    cohorts = {}
    for name, values in dict(
        train=np.arange(20),
        validation=np.arange(20, 40),
        validation_pilot=np.arange(20, 24),
    ).items():
        p = source / f"{name}.npy"
        np.save(p, values)
        cohorts[name] = dict(path=str(p), sha256=night.sha256_file(p))
    night.write(source / "RUN_MANIFEST.json", dict(cohorts=cohorts))
    recipe = dict(
        seed=1,
        train_objects=8,
        validation_objects=4,
        tracking_objects=2,
        smoke_train_objects=2,
        smoke_validation_objects=2,
    )
    selected = night.select_cohorts(root, source, recipe)
    values = {k: np.load(v["path"]) for k, v in selected.items()}
    assert not np.intersect1d(values["train"], values["validation"]).size
    assert not np.intersect1d(values["tracking"], values["validation"]).size
    assert set(values["smoke_train"]) <= set(values["train"])
    with (source / "train.npy").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        night.select_cohorts(root, source, recipe)


def test_night_command_timeout_and_no_shell(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(night.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    commands = night.Commands(tmp_path, 10)
    commands.run("smoke", ["python", "path with spaces"])
    assert calls[0][0][0] == ["python", "path with spaces"]
    assert 0 < calls[0][1]["timeout"] <= 10
    assert calls[0][1]["check"]
    commands.deadline = 0
    with pytest.raises(TimeoutError):
        commands.run("never", ["false"])
    assert len(calls) == 1


def tiny_model_config():
    from euclid_dsps.config import load_config
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import observed_only_config

    config = observed_only_config(
        load_config(
            "configs/experiments/feniks_sc_drws_r29_frozen_parent_topology_sleep_npe.yaml"
        )
    )
    a = config["amortized"]
    a["latent"].pop("normalization_checkpoint", None)
    a["prior"].update(source="joint_realnvp", n_layers=2, hidden_size=8)
    a["encoder"].update(
        context_encoder="base_moments",
        hidden_sizes=[8],
        flow_hidden_size=8,
        residual_trunk_width=8,
        residual_representation_width=8,
        residual_context_dim=8,
    )
    return config


def test_real_checkpoint_migration_and_numerical_contract(tmp_path):
    from euclid_dsps.amortized.features import FeatureStats
    from euclid_dsps.amortized.latent import latent_spec_from_config
    from euclid_dsps.amortized.train import (
        _array_tree_sha256,
        build_amortized_model,
        load_checkpoint,
        save_checkpoint,
    )

    old = tiny_model_config()
    config = night.candidate_configuration(old)
    model = build_amortized_model(old, jax.random.PRNGKey(1))
    stats = FeatureStats(
        np.ones(18), np.ones(18), tuple(b["name"] for b in old["bands"])
    )
    path = tmp_path / "new.eqx"
    save_checkpoint(
        path,
        model,
        config=config,
        latent_spec=latent_spec_from_config(config),
        feature_stats=stats,
        epoch=0,
        metric=0,
    )
    restored = load_checkpoint(path, config)
    assert _array_tree_sha256(restored) == _array_tree_sha256(model)
    with pytest.raises(ValueError, match="numerical contract"):
        load_checkpoint(path, old)
    different = copy.deepcopy(config)
    different["amortized"]["likelihood"]["arithmetic_precision"] = "float32_legacy"
    with pytest.raises(ValueError, match="numerical contract"):
        load_checkpoint(path, different)


@pytest.mark.parametrize("fail_smoke", [False, True])
def test_night_stage_order_and_smoke_stops_followups(tmp_path, monkeypatch, fail_smoke):
    from pathlib import Path

    from euclid_dsps.amortized.features import FeatureStats, write_feature_stats
    from euclid_dsps.amortized.latent import latent_spec_from_config
    from euclid_dsps.amortized.train import build_amortized_model

    config = night.candidate_configuration(tiny_model_config())
    model = build_amortized_model(config, jax.random.PRNGKey(4))
    stats = FeatureStats(
        np.ones(18), np.ones(18), tuple(b["name"] for b in config["bands"])
    )
    stats_path = tmp_path / "stats.json"
    write_feature_stats(stats_path, stats)
    recipe = dict(
        seed=1,
        train_objects=8,
        validation_objects=4,
        tracking_objects=2,
        smoke_train_objects=2,
        smoke_validation_objects=2,
        maximum_seconds=10,
        smoke_candidates=4,
        sleep_candidates=8,
        observed_draws=4,
        observed_weight=0.001,
        jax_batch_size=2,
        decoder_batch_size=2,
        learning_rate=1e-5,
        sleep_epochs=1,
        elbo_epochs=1,
        tracking_draws=256,
    )
    manifest = dict(
        night_recipe=recipe,
        source_root="unused",
        source=dict(feature_stats=str(stats_path)),
        dataset=dict(path="observed.parquet"),
    )
    monkeypatch.setattr(
        night,
        "select_cohorts",
        lambda *a: {
            k: dict(path=k + ".npy")
            for k in (
                "train",
                "validation",
                "smoke_train",
                "smoke_validation",
                "tracking",
            )
        },
    )
    stages = []

    def command(self, stage, args):
        stages.append(stage)
        if stage.startswith("summarize_"):
            night.write(
                Path(args[-1]) / "TRUTH_FREE_POSTERIOR_VALIDATION.json",
                {"status": "MOCK"},
            )

    monkeypatch.setattr(night.Commands, "run", command)

    def certify(out, *a, **kw):
        if out.parent.name == "smoke" and fail_smoke:
            raise ValueError("smoke failed")
        return out / "checkpoints/best.eqx"

    monkeypatch.setattr(night, "certify_training", certify)
    report = dict(
        candidate_numerical_checks="PASS",
        cases=[dict(variant="spline64", numerical_checks="PASS") for _ in range(6)],
    )
    if fail_smoke:
        with pytest.raises(ValueError, match="smoke failed"):
            night.continue_night(
                tmp_path,
                manifest,
                config,
                latent_spec_from_config(config),
                model,
                stats,
                report,
                1.0,
            )
        assert stages == ["train_smoke"]
    else:
        result = night.continue_night(
            tmp_path,
            manifest,
            config,
            latent_spec_from_config(config),
            model,
            stats,
            report,
            1.0,
        )
        assert stages[:3] == ["train_smoke", "train_B", "train_C"]
        assert stages[-1] == "internal_C"
        assert result["status"] == "PRECISION_NIGHT_DIAGNOSTIC_COMPLETE"
        assert (
            not result["scientific_promotion"]
            and not result["population_training_started"]
        )
