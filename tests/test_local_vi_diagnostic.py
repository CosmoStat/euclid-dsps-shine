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


@pytest.mark.parametrize(
    "isolation", [False, True, "decomposition", "reference", "full"]
)
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
    cache_x = jnp.array([[0.1, 0.2], [-0.1, -0.2], [0.2, 0.1]])
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
    reference_path = None
    if isolation == "full":
        from types import SimpleNamespace

        from euclid_dsps.amortized import decoder_qualification

        reference_path = tmp_path / "reference"
        runner.write(
            reference_path / "PHOTOMETRY_REFERENCE.json",
            dict(numerical_reference_checks="PASS", points=[{}, {}, {}]),
        )
        runner.write(
            reference_path / "RUN_MANIFEST.json",
            dict(
                source_root=str(source),
                source_manifest_sha256=runner.sha256_file(source / "RUN_MANIFEST.json"),
            ),
        )
        runner.write(
            reference_path / "FINAL.json",
            dict(
                status="PHOTOMETRY_REFERENCE_COMPLETE",
                artifacts={
                    "PHOTOMETRY_REFERENCE.json": dict(
                        sha256=runner.sha256_file(
                            reference_path / "PHOTOMETRY_REFERENCE.json"
                        )
                    )
                },
            ),
        )
        monkeypatch.setattr(decoder_qualification, "STEPS", (0.02, 0.01, 0.005))
    runner.prepare(
        root,
        source,
        objects=2,
        steps=1,
        draws=32,
        gradient_isolation=isolation is True,
        redshift_decomposition=isolation == "decomposition",
        photometry_reference=isolation == "reference",
        full_decoder_reference=reference_path,
    )
    with pytest.raises(FileExistsError):
        runner.prepare(root, source, objects=2, steps=1, draws=32)
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args: model)
    names = ("log10_stellar_metallicity", "x1") if isolation == "full" else ("x0", "x1")
    spec = LatentSpec(names, jnp.array([-10.0, -10.0]), jnp.array([10.0, 10.0]))
    monkeypatch.setattr(runner, "latent_spec_from_config", lambda c: spec)
    monkeypatch.setattr(runner, "load_filters", lambda *args: None)
    monkeypatch.setattr(runner, "load_context", lambda *args, **kwargs: None)
    if isolation == "full":
        monkeypatch.setattr(
            runner,
            "load_context",
            lambda *args, **kwargs: SimpleNamespace(
                model_config=config["model"], z_sun=0.02
            ),
        )
    monkeypatch.setattr(runner, "dynamic_model_args", lambda c: None)
    monkeypatch.setattr(runner, "_model_flux_from_x_sample_chunks", flux)
    monkeypatch.setattr(
        runner,
        "posterior_log_target",
        lambda *args: posterior_log_target(*args, model_flux_fn=flux),
    )
    if isolation == "decomposition":
        from euclid_dsps.amortized import redshift_decomposition as rd

        def mock_decompose(context, spec, points, observation, budget, **kwargs):
            assert points.shape == (3, 2)
            assert kwargs["canonical_flux"](points[0]).shape == (1, 1, 2)
            rows = [dict(branch="mock", point_index=0, ad=1.0, fd=1.0)]
            kwargs["progress"](0, "mock", rows, [])
            return dict(status="REDSHIFT_DECOMPOSITION_COMPLETE", points=[]), rows

        monkeypatch.setattr(rd, "decompose_redshift", mock_decompose)
    if isolation == "reference":
        from euclid_dsps.amortized import photometry_reference as pr

        def mock_export(root, context, spec, points, observation, budget, **kwargs):
            assert points.shape == (3, 2)
            np.savez(root / "FIXED_SPECTRA.npz", x=np.asarray(points))
            return dict(x=np.asarray(points))

        def mock_analyze(arrays, budget, **kwargs):
            kwargs["progress"](
                0, "point_complete", [dict(point_index=0, method="mock")], []
            )
            return dict(status="PHOTOMETRY_REFERENCE_COMPLETE", points=[]), []

        monkeypatch.setattr(pr, "export_spectra", mock_export)
        monkeypatch.setattr(pr, "analyze_snapshot", mock_analyze)
    result = runner.run(root)
    if isolation == "full":
        assert result["status"] == "FULL_DECODER_QUALIFICATION_COMPLETE"
        report = runner.read(root / "FULL_DECODER_QUALIFICATION.json")
        assert len(report["cases"]) == 10
        assert report["prior_bitwise_unchanged"] is True
        assert report["npe_training_started"] is False
        assert report["old_flux_bank_reuse_authorized"] is False
        assert (root / "candidate_config.yaml").is_file()
        assert runner.read(root / "RUN_MANIFEST.json")["allocation_gpu_hours"] == 1.5
        from scripts.summarize_feniks_sc_drws_decoder_qualification import summarize

        assert summarize(root)["status"] == result["status"]
        import euclid_dsps.model as sed

        monkeypatch.setattr(
            sed, "_context_ssp_lgmet", lambda ctx: jnp.linspace(-4, -1, 8)
        )
        precision_root = tmp_path / "precision"
        runner.prepare(precision_root, source, mdf_precision_reference=root)
        result64 = runner.run(precision_root)
        assert result64["local_optimization_started"] is False
        assert result64["population_training_started"] is False
        precision_report = summarize(precision_root)
        assert precision_report["identical_source_points_verified"] is True
        assert precision_report["variant_labels"] == ["merged_mdf32", "merged_mdf64"]
        assert (precision_root / "MDF_WEIGHT_PROBES.json").is_file()
        # Exercise the residual-only follow-up on a synthetic unresolved check.
        candidate_case = next(
            c for c in precision_report["cases"] if c["variant"] == "merged_mdf64"
        )
        pending = next(
            c for c in candidate_case["checks"] if c["component"] == "centered_loglike"
        )
        pending["status"] = "INCONCLUSIVE"
        precision_path = precision_root / "FULL_DECODER_QUALIFICATION.json"
        runner.write(precision_path, precision_report)
        precision_final = runner.read(precision_root / "FINAL.json")
        precision_final["artifacts"][precision_path.name]["sha256"] = (
            runner.sha256_file(precision_path)
        )
        runner.write(precision_root / "FINAL.json", precision_final)
        residual_root = tmp_path / "residual"
        prepared = runner.prepare(
            residual_root, source, target_resolution_reference=precision_root
        )
        assert prepared["maximum_decoder_evaluations"] == 1000
        assert prepared["allocation_gpu_hours"] == 0.75
        residual_final = runner.run(residual_root)
        assert residual_final["status"] == "TARGET_RESOLUTION_COMPLETE"
        assert residual_final["population_training_started"] is False
        from scripts.summarize_feniks_sc_drws_target_resolution import (
            summarize as summarize_residual,
        )

        residual_report = summarize_residual(residual_root)
        assert len(residual_report["checks"]) == 1
        assert residual_report["checks"][0]["status"] == "PASS"
        # Mock the expensive branch collector, exercising the new linked mode,
        # immutable inputs, artifacts, CPU replay and failure on tampering.
        from euclid_dsps.amortized import redshift_precision

        residual_report["checks"] = [
            dict(
                point_index=4,
                coordinate="z_obs",
                component="lsst_z",
                status="INCONCLUSIVE",
                ad=0.2,
            )
        ]
        runner.write(residual_root / "TARGET_RESOLUTION.json", residual_report)
        rf = runner.read(residual_root / "FINAL.json")
        rf["artifacts"]["TARGET_RESOLUTION.json"]["sha256"] = runner.sha256_file(
            residual_root / "TARGET_RESOLUTION.json"
        )
        runner.write(residual_root / "FINAL.json", rf)

        def branch_collect(*args, progress, **kwargs):
            case = dict(
                component="lsst_z",
                anchor=[1.0],
                observed=[1.0],
                sigma=[1.0],
                flux_jvp=[2.0],
                band_index=0,
                flux_dtype="float64",
                ad=2.0,
                point_index=4,
                coordinate="z_obs",
                source_ad_delta=0.0,
                atol=0.01,
                rtol=0.01,
                samples=[
                    dict(
                        step=h,
                        actual_plus_step=h,
                        actual_minus_step=h,
                        plus=[1 + 2 * h],
                        minus=[1 - 2 * h],
                    )
                    for h in redshift_precision.STEPS
                ],
            )
            value = dict(
                branches=[
                    dict(
                        name="zpath64_full",
                        floating_dtypes=["float64"],
                        max_center_delta_sigma=0.0,
                        snapshot=dict(cases=[case], expected_checks=1),
                    )
                ],
                expected_branches=1,
                physical_z=1.0,
                dz_dx=0.5,
                source_ad_delta=0.0,
            )
            progress(value)
            return value

        monkeypatch.setattr(redshift_precision, "collect", branch_collect)
        branch_root = tmp_path / "redshift_precision"
        prepared = runner.prepare(
            branch_root, source, redshift_precision_reference=residual_root
        )
        assert prepared["maximum_decoder_evaluations"] == 1000
        final_branch = runner.run(branch_root)
        assert final_branch["status"] == "REDSHIFT_PRECISION_COMPLETE"
        assert final_branch["local_optimization_started"] is False
        from scripts.summarize_feniks_sc_drws_redshift_precision import (
            summarize as summarize_branch,
        )

        assert summarize_branch(branch_root)["branches"][0]["all_checks_passed"]
        from scripts import feniks_precision_night as night

        night_root = tmp_path / "night"
        prepared_night = runner.prepare(
            night_root, source, precision_night_reference=branch_root
        )
        assert prepared_night["mode"] == "precision_night"
        assert prepared_night["allocation_gpu_hours"] == 10
        continued = []

        def capture_continuation(*args):
            continued.append(args)
            return {"status": "MOCK_NIGHT_CONTINUATION"}

        monkeypatch.setattr(night, "continue_night", capture_continuation)
        assert runner.run(night_root)["status"] == "MOCK_NIGHT_CONTINUATION"
        assert len(continued) == 1
        cfg = continued[0][2]
        assert cfg["model"]["spline_precision"] == "float64_v1"
        assert cfg["amortized"]["likelihood"]["arithmetic_precision"] == "float64_v1"
        assert continued[0][6]["variant_labels"] == ["legacy", "spline64"]
        (residual_root / "TARGET_RESOLUTION_SNAPSHOT.json").write_text("{}")
        with pytest.raises(ValueError, match="resolution reference artifact changed"):
            runner.verify_redshift_precision_reference(residual_root, source)
        with pytest.raises(ValueError, match="changed artifact"):
            summarize_residual(residual_root)
        (root / "QUALIFICATION_POINTS.npz").write_bytes(b"changed")
        with pytest.raises(ValueError, match="decoder reference artifact changed"):
            runner.verify_decoder_reference(root, source)
        (reference_path / "PHOTOMETRY_REFERENCE.json").write_text("{}")
        with pytest.raises(ValueError, match="reference artifact changed"):
            runner.verify_photometry_reference(reference_path, source)
        return
    if isolation == "reference":
        assert result["status"] == "PHOTOMETRY_REFERENCE_COMPLETE"
        assert not (root / "cases").exists()
        assert (root / "FIXED_SPECTRA.npz").exists()
        assert result["scientific_promotion"] is False
        assert runner.read(root / "RUN_MANIFEST.json")["mode"] == "photometry_reference"
        assert runner.read(root / "RUN_MANIFEST.json")["allocation_gpu_hours"] == 0.75
        return
    if isolation == "decomposition":
        assert result["status"] == "REDSHIFT_DECOMPOSITION_COMPLETE"
        assert result["local_optimization_started"] is False
        assert not (root / "cases").exists()
        assert runner.read(root / "RUN_MANIFEST.json")["allocation_gpu_hours"] == 0.75
        assert (root / "redshift_decomposition.csv").is_file()
        return
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
