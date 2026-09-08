from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.elbo import AmortizedModel
from euclid_dsps.amortized.flows import StandardNormalPrior
from euclid_dsps.amortized.local_vi_diagnostic import (
    Budget,
    BudgetExceeded,
    analytic_controls,
    assert_close,
    gradient_audit,
    initialize,
    log_prob,
    make_step,
    perturb,
    sample,
)
from euclid_dsps.amortized.posterior import (
    ConditionalFlowEncoder,
    conditional_flow_topology,
    sample_posterior,
)
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    PosteriorTargetValues,
)
from euclid_dsps.amortized.train import _array_tree_sha256
from euclid_dsps.calibration import GlobalSedScaleState


def model_fixture():
    encoder = ConditionalFlowEncoder(
        jax.random.PRNGKey(1),
        input_dim=6,
        latent_dim=2,
        hidden_sizes=(8,),
        activation="gelu",
        log_std_min=-4,
        log_std_max=2,
        initial_log_std=0,
        n_layers=4,
        hidden_size=8,
        output_space="latent_x",
        context_encoder_type="residual_photometry",
        residual_trunk_width=8,
        residual_blocks=1,
        residual_representation_width=8,
        residual_context_dim=4,
    )
    return AmortizedModel(
        encoder, StandardNormalPrior(latent_dim=2), GlobalSedScaleState(jnp.array(0.0))
    )


def test_local_flow_matches_original_and_restored_density(tmp_path):
    model = model_fixture()
    features = jnp.ones((1, 6))
    initial, context = initialize(model, features)
    key = jax.random.PRNGKey(5)
    x, logq = sample(model.encoder, initial, context, key, 32)
    expected = sample_posterior(model, key, features, 32)
    np.testing.assert_allclose(x, expected.x, atol=1e-6)
    np.testing.assert_allclose(logq, expected.logq, atol=1e-6)
    path = tmp_path / "local.eqx"
    eqx.tree_serialise_leaves(path, initial)
    restored = eqx.tree_deserialise_leaves(path, initial)
    np.testing.assert_allclose(
        log_prob(model.encoder, restored, context, x), logq, atol=1e-5
    )
    assert _array_tree_sha256(initial.layers) == _array_tree_sha256(restored.layers)


def test_local_optimization_updates_base_and_couplings_without_changing_parent():
    model = model_fixture()
    initial, context = initialize(model, jnp.ones((1, 6)))
    fingerprint = _array_tree_sha256(model)
    y = jnp.array([[1.0, -1.0]])

    def target(x, observation):
        lp = model.prior.log_prob(x)
        ll = -0.5 * jnp.sum(((x - observation) / 0.3) ** 2, axis=-1)
        return PosteriorTargetValues(lp + ll, ll, lp, jnp.ones(lp.shape, bool), x, x)

    optimizer, step = make_step(model.encoder, target, draws=16, learning_rate=0.01)
    state = optimizer.init(eqx.filter(initial, eqx.is_inexact_array))
    parameters = initial

    def independent_loss(p):
        x, lq = sample(model.encoder, p, context, jax.random.PRNGKey(888), 2048)
        return float(jnp.mean(lq - target(x, y).logtarget))

    before = independent_loss(initial)
    for index in range(80):
        parameters, state, metrics = step(
            parameters, state, context, y, jax.random.PRNGKey(100 + index)
        )
        assert bool(metrics["finite"])
    assert independent_loss(parameters) < before - 1
    assert _array_tree_sha256(model) == fingerprint
    assert not np.allclose(parameters.mean, initial.mean)
    assert _array_tree_sha256(parameters.layers) != _array_tree_sha256(initial.layers)
    changed_encoder = eqx.tree_at(lambda e: e.layers, model.encoder, parameters.layers)
    assert conditional_flow_topology(changed_encoder) == conditional_flow_topology(
        model.encoder
    )
    # Exact analytic Gaussian target has mean y/(1+.09), covariance .09/1.09 I.
    x, _ = sample(model.encoder, parameters, context, jax.random.PRNGKey(999), 4096)
    np.testing.assert_allclose(np.mean(x, axis=0), np.asarray(y) / 1.09, atol=0.2)


def test_nonfinite_draw_is_not_silently_removed():
    model = model_fixture()
    initial, context = initialize(model, jnp.ones((1, 6)))

    def target(x, observation):
        lp = -jnp.sum(x * x, axis=-1).at[0].set(jnp.inf)
        return PosteriorTargetValues(lp, lp, lp, jnp.isfinite(lp), x, x)

    optimizer, step = make_step(model.encoder, target)
    _, _, metrics = step(
        initial,
        optimizer.init(eqx.filter(initial, eqx.is_inexact_array)),
        context,
        jnp.zeros((1, 2)),
        jax.random.PRNGKey(1),
    )
    assert not bool(metrics["finite"])


def test_rank_controls_detect_ignoring_data():
    result = analytic_controls()
    assert result["ignores_data"]["marginal"]["status"] == "PASS"
    assert result["ignores_data"]["projection"]["status"] == "PASS"
    assert result["ignores_data"]["loglike"]["status"] == "FAIL"
    assert all(v["status"] == "PASS" for v in result["exact_posterior"].values())


def test_budget_limits_before_work(monkeypatch):
    budget = Budget(10, 8)
    budget.charge(4)
    budget.charge(4, gradient=True)
    with pytest.raises(BudgetExceeded, match="evaluation"):
        budget.charge(1)
    assert budget.forward == budget.gradient == 4
    budget.started -= 11
    with pytest.raises(BudgetExceeded, match="wall-clock"):
        budget.charge(0)


def test_parity_failures_are_explicit():
    with pytest.raises(ValueError, match="mismatch"):
        assert_close("check", [1.0], [2.0])
    with pytest.raises(ValueError, match="nonfinite"):
        assert_close("check", [np.nan], [1.0])


def test_gradient_audit_refines_truncation_without_relaxing_tolerances():
    def objective(x):
        return {"f": jnp.sum(x + 10000 * x**3)}

    result = gradient_audit(objective, jnp.zeros(1), jnp.ones(1), Budget(60, 100))
    assert result["status"] == "PASS"
    assert result["atol"] == 0.1 and result["rtol"] == 0.05
    values = result["components"]["f"]["samples"]
    assert abs(values[1]["central_difference"] - values[2]["central_difference"]) > 0.5
    assert result["components"]["f"]["selected_step"] < 0.005


def test_gradient_audit_rejects_wrong_ad_and_unresolved_float32():
    def wrong(x):
        return {"f": jnp.sum(jax.lax.stop_gradient(2 * x))}

    result = gradient_audit(wrong, jnp.zeros(1), jnp.ones(1), Budget(60, 100))
    assert result["status"] == "FAIL"
    assert result["components"]["f"]["selected_step"] is not None

    def unresolved(x):
        return {"f": jnp.asarray(1e8 + 2 * jnp.sum(x), dtype=jnp.float32)}

    result = gradient_audit(unresolved, jnp.zeros(1), jnp.ones(1), Budget(60, 100))
    assert result["status"] == "INCONCLUSIVE"
    assert result["components"]["f"]["selected_step"] is None


def test_gradient_audit_checks_components_not_only_cancelling_total():
    def objective(x):
        value = jnp.sum(jax.lax.stop_gradient(2 * x))
        return dict(loglike=value, logprior=-value, logtarget=value - value)

    result = gradient_audit(objective, jnp.zeros(1), jnp.ones(1), Budget(60, 100))
    assert result["status"] == "FAIL"
    assert result["components"]["logtarget"]["status"] == "PASS"
    assert result["components"]["loglike"]["status"] == "FAIL"


def test_gradient_audit_failure_saves_contract_evidence(tmp_path, monkeypatch):
    # Reuse the complete contract fixture, but inject a recorded inconclusive audit.
    import scripts.run_feniks_sc_drws_local_vi_diagnostic as runner

    monkeypatch.setattr(
        runner,
        "gradient_audit",
        lambda *a: {
            "status": "INCONCLUSIVE",
            "components": {"loglike": {"autodiff": float("nan")}},
        },
    )
    with pytest.raises(ValueError, match="gradient audit INCONCLUSIVE"):
        test_contract_audit_with_analytic_decoder(tmp_path, monkeypatch)
    assert runner.read(tmp_path / "CONTRACT_AUDIT.json")["status"] == "INCONCLUSIVE"
    assert (
        runner.read(tmp_path / "GRADIENT_AUDIT.json")["components"]["loglike"][
            "autodiff"
        ]
        is None
    )


def test_perturbation_preserves_coupling_structure():
    model = model_fixture()
    p, _ = initialize(model, jnp.ones((1, 6)))
    second = perturb(p, jax.random.PRNGKey(4))
    assert not np.allclose(p.mean, second.mean)
    assert _array_tree_sha256(p.layers) == _array_tree_sha256(second.layers)


def test_truth_config_and_noise_guard():
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import check_config

    with pytest.raises(ValueError, match="truth"):
        check_config({"truth": {"parameter_columns": {"z": "z_true"}}})


def test_observed_config_removes_legacy_truth_without_altering_physics(tmp_path):
    import copy

    import yaml

    from euclid_dsps.config import load_config
    from euclid_dsps.io import required_catalog_columns
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import observed_only_config

    source = load_config(
        "configs/experiments/feniks_sc_drws_r29_frozen_parent_topology_sleep_npe.yaml"
    )
    original = copy.deepcopy(source)
    config = observed_only_config(source)
    assert source == original
    assert config["amortized"] == source["amortized"]
    assert config["bands"] == source["bands"]
    assert config["fit"] == source["fit"]
    columns = required_catalog_columns(config)
    assert "redshift_true" not in columns
    assert "z_obs" not in columns
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert required_catalog_columns(load_config(path)) == columns


def test_contract_audit_with_analytic_decoder(tmp_path, monkeypatch):
    import json

    import scripts.run_feniks_sc_drws_local_vi_diagnostic as runner
    from euclid_dsps.amortized.features import FeatureStats
    from euclid_dsps.amortized.latent import LatentSpec
    from euclid_dsps.amortized.posterior_target import posterior_log_target

    model = model_fixture()
    spec = LatentSpec(("x0", "x1"), jnp.array([-10.0, -10.0]), jnp.array([10.0, 10.0]))
    stats = FeatureStats(np.ones(2), np.ones(2), ("b0", "b1"), append_mask=True)
    sleep = dict(
        feature_flux_scale=np.ones(2),
        feature_err_scale=np.ones(2),
        append_mask=True,
        flux_transform="asinh",
        error_epsilon=1e-6,
    )
    monkeypatch.setattr(runner, "_sleep_runtime_config", lambda c, s: sleep)
    config = {"amortized": {"likelihood": {"type": "gaussian"}}}

    def flux(x, *args, **kwargs):
        return 2 * x + 1

    monkeypatch.setattr(runner, "_model_flux_from_x_sample_chunks", flux)

    def target(x, obs):
        return posterior_log_target(
            model,
            x,
            obs,
            spec,
            None,
            None,
            spec.names,
            config["amortized"]["likelihood"],
            model_flux_fn=flux,
        )

    cache = tmp_path / "cache.npz"
    x = jnp.array([[0.1, 0.2], [-0.2, 0.3]])
    np.savez(cache, x=x, model_flux=flux(x))
    (tmp_path / "RUN_MANIFEST.json").write_text(
        json.dumps({"cache": {"path": str(cache)}})
    )
    observation = PosteriorObservation(
        jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 2), dtype=bool)
    )
    result = runner.contract_audit(
        tmp_path,
        model,
        config,
        stats,
        spec,
        None,
        None,
        observation,
        target,
        Budget(120, 100),
    )
    assert result["status"] == "PASS"
    assert result["truth_used"] is False


def test_high_ess_does_not_certify_missing_mode():
    from scipy.integrate import quad
    from scipy.special import logsumexp
    from scipy.stats import norm

    # Normalized two-mode target; a direct proposal missing one mode looks stable.
    rng = np.random.default_rng(9)
    x = rng.normal(-8, 0.5, 4096)
    logq = norm.logpdf(x, -8, 0.5)
    logp = np.logaddexp(logq, norm.logpdf(x, 8, 0.5)) - np.log(2)
    w = np.exp(logp - logq)
    ess = w.sum() ** 2 / np.sum(w**2)
    logz = logsumexp(logp - logq) - np.log(len(x))
    integral = quad(
        lambda z: 0.5 * (norm.pdf(z, -8, 0.5) + norm.pdf(z, 8, 0.5)), -12, 12
    )[0]
    assert integral == pytest.approx(1)
    assert ess / len(x) > 0.99
    assert logz == pytest.approx(-np.log(2))


@pytest.mark.parametrize("isolation", [False, True])
def test_prepare_and_complete_runner_with_mock_physics(
    tmp_path, monkeypatch, isolation
):
    """Exercise real parquet/receipts/optimization, replacing DSPS and GPU only."""
    import pandas as pd
    import yaml

    import scripts.run_feniks_sc_drws_local_vi_diagnostic as runner
    from euclid_dsps.amortized.features import FeatureStats, write_feature_stats
    from euclid_dsps.amortized.latent import LatentSpec
    from euclid_dsps.amortized.posterior_target import posterior_log_target
    from euclid_dsps.config import load_config

    source, root = tmp_path / "source", tmp_path / "diagnostic"
    source.mkdir()
    config = load_config(
        "configs/experiments/feniks_sc_drws_r29_frozen_parent_topology_sleep_npe.yaml"
    )
    config["bands"] = [
        band for band in config["bands"] if band["name"] in ("lsst_u", "lsst_r")
    ]
    config_path = source / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    dataset = source / "data.parquet"
    # Deliberately no catalogue truth columns: the inherited config references them.
    pd.DataFrame(
        dict(
            object_id=range(4),
            flux_lsst_u=[100.0, 101.0, 102.0, 103.0],
            flux_lsst_r=[100.0, 101.0, 102.0, 103.0],
            fluxerr_lsst_u=np.ones(4),
            fluxerr_lsst_r=np.ones(4),
        )
    ).to_parquet(dataset)
    stats_path = source / "stats.json"
    stats = FeatureStats(
        np.full(2, 100.0), np.ones(2), ("lsst_u", "lsst_r"), append_mask=True
    )
    write_feature_stats(stats_path, stats)
    model = model_fixture()
    checkpoint = source / "model.eqx"
    eqx.tree_serialise_leaves(checkpoint, model)
    artifact = dict(
        status="COMPLETE",
        prior_bitwise_unchanged=True,
        truth_used_for_training_or_checkpoint_selection=False,
    )
    for name, path in (
        ("checkpoint", checkpoint),
        ("config", config_path),
        ("feature_stats", stats_path),
    ):
        artifact[name], artifact[name + "_sha256"] = str(path), runner.sha256_file(path)
    runner.write(source / "arms/B/ARM_COMPLETE.json", artifact)
    cohorts = {}
    for name, values in (("train", [0, 1]), ("validation_pilot", [2, 3])):
        path = source / f"{name}.npy"
        np.save(path, values)
        cohorts[name] = dict(path=str(path), sha256=runner.sha256_file(path))
    runner.write(
        source / "RUN_MANIFEST.json",
        dict(
            cohorts=cohorts,
            dataset=dict(path=str(dataset), sha256=runner.sha256_file(dataset)),
        ),
    )

    def flux(x, *args, **kwargs):
        return 100 + 0.2 * x

    cache = source / "cache.npz"
    cache_x = jnp.array([[0.1, 0.2], [-0.1, -0.2]])
    np.savez(cache, x=cache_x, model_flux=flux(cache_x))
    sidecar = cache.with_suffix(".npz.json")
    runner.write(
        sidecar, dict(prior_fingerprint_sha256=_array_tree_sha256(model.prior))
    )
    runner.write(
        source / "TRAINING_CACHE_FROZEN.json",
        dict(
            path=str(cache),
            sha256=runner.sha256_file(cache),
            sidecar_sha256=runner.sha256_file(sidecar),
        ),
    )
    runner.prepare(
        root, source, objects=2, steps=1, draws=32, gradient_isolation=isolation
    )
    with pytest.raises(FileExistsError):
        runner.prepare(root, source, objects=2, steps=1, draws=32)
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args: model)
    spec = LatentSpec(("x0", "x1"), jnp.array([-10.0, -10.0]), jnp.array([10.0, 10.0]))
    monkeypatch.setattr(runner, "latent_spec_from_config", lambda c: spec)
    monkeypatch.setattr(runner, "load_filters", lambda *args: None)
    monkeypatch.setattr(runner, "load_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "dynamic_model_args", lambda c: None)
    monkeypatch.setattr(runner, "_model_flux_from_x_sample_chunks", flux)
    monkeypatch.setattr(
        runner,
        "posterior_log_target",
        lambda *args: posterior_log_target(*args, model_flux_fn=flux),
    )
    result = runner.run(root)
    if isolation:
        assert result["status"] == "GRADIENT_ISOLATION_COMPLETE"
        assert result["local_optimization_started"] is False
        assert not (root / "cases").exists()
        assert len(pd.read_csv(root / "gradient_isolation.csv")) == 36
        assert runner.read(root / "RUN_MANIFEST.json")[
            "allocation_gpu_hours"
        ] == pytest.approx(1 / 3)
        return
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert result["cases_complete"] == 4
    assert result["prior_bitwise_unchanged"] is True
    assert result["population_training_started"] is False
    assert len(pd.read_csv(root / "paired_comparison.csv")) == 12
    assert (root / "paired_comparison.png").stat().st_size > 1000
    assert len(list(root.glob("cases/*/start_*/parameters.eqx"))) == 8


def test_replicated_evaluation_is_density_based(tmp_path):
    from scripts.run_feniks_sc_drws_local_vi_diagnostic import evaluate_distribution

    model = model_fixture()
    p, context = initialize(model, jnp.ones((1, 6)))
    # Simple target has exact q weights; physical exports use a valid latent spec.
    from euclid_dsps.amortized.latent import LatentSpec

    spec = LatentSpec(("x0", "x1"), jnp.array([-10.0, -10.0]), jnp.array([10.0, 10.0]))

    def target(x, observation):
        lq = log_prob(model.encoder, p, context, x)
        zeros = jnp.zeros_like(lq)
        return PosteriorTargetValues(lq, zeros, lq, jnp.ones(lq.shape, bool), x, x)

    budget = Budget(60, 1000)
    result = evaluate_distribution(
        tmp_path,
        model.encoder,
        p,
        context,
        PosteriorObservation(
            jnp.zeros((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 2), bool)
        ),
        target,
        spec,
        budget,
        10,
        32,
    )
    assert result["raw_ess"]["median"] == pytest.approx(64, abs=0.001)
    assert result["replicate_abs_log_evidence_delta"] < 1e-5
    assert budget.forward == 64
    assert result["scientific_promotion"] is False
