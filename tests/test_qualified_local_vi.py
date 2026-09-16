from __future__ import annotations

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import yaml
from test_local_vi_diagnostic import model_fixture

from scripts import feniks_qualified_local_vi as follow
from scripts.analyze_feniks_sc_drws_precision_night import (
    analyze,
    calibration_summary,
    heldout_summary,
)


@pytest.fixture(autouse=True)
def x64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def night(tmp_path):
    from euclid_dsps.amortized.features import FeatureStats, write_feature_stats
    from euclid_dsps.amortized.latent import latent_spec_from_config, latent_spec_hash
    from euclid_dsps.amortized.train import _array_tree_sha256
    from euclid_dsps.config import load_config
    from euclid_dsps.model import photometry_numerics
    from scripts.feniks_precision_night import candidate_configuration

    root = tmp_path / "night"
    root.mkdir()
    model = model_fixture()
    cfg = candidate_configuration(
        load_config(
            "configs/experiments/feniks_sc_drws_r29_frozen_parent_topology_sleep_npe.yaml"
        )
    )
    cfg["bands"] = [b for b in cfg["bands"] if b["name"] in ("lsst_u", "lsst_r")]
    (root / "configs").mkdir()
    (root / "configs/C.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    dataset = root / "data.parquet"
    pd.DataFrame(
        dict(
            object_id=range(6),
            flux_lsst_u=np.full(6, 100.0),
            flux_lsst_r=np.full(6, 100.0),
            fluxerr_lsst_u=np.ones(6),
            fluxerr_lsst_r=np.ones(6),
        )
    ).to_parquet(dataset)
    train = root / "arms/C/train"
    (train / "checkpoints").mkdir(parents=True)
    checkpoint = train / "checkpoints/best.eqx"
    eqx.tree_serialise_leaves(checkpoint, model)
    follow.write(checkpoint.with_suffix(".eqx.json"), {})
    stats = FeatureStats(
        np.full(2, 100.0), np.ones(2), ("lsst_u", "lsst_r"), append_mask=True
    )
    write_feature_stats(train / "feature_stats.json", stats)
    follow.write(
        root / "arms/C/ARM_COMPLETE.json",
        dict(
            status="COMPLETE",
            prior_bitwise_unchanged=True,
            truth_used=False,
            checkpoint_sha256=follow.sha256_file(checkpoint),
            checkpoint_sidecar_sha256=follow.sha256_file(
                checkpoint.with_suffix(".eqx.json")
            ),
            feature_stats_sha256=follow.sha256_file(train / "feature_stats.json"),
        ),
    )
    follow.write(
        root / "NIGHT_FINAL.json",
        dict(
            status="PRECISION_NIGHT_DIAGNOSTIC_COMPLETE",
            truth_used=False,
            population_training_started=False,
            scientific_promotion=False,
        ),
    )
    follow.write(
        root / "FULL_DECODER_QUALIFICATION.json",
        dict(
            candidate_numerical_checks="PASS",
            cases=[dict(variant="spline64", numerical_checks="PASS") for _ in range(6)],
        ),
    )
    follow.write(
        root / "MIGRATION.json",
        dict(
            status="PASS",
            frozen_array_sha256=_array_tree_sha256(
                (model.prior, model.sed_scale, model.band_calibration)
            ),
        ),
    )
    follow.write(
        root / "RUN_MANIFEST.json",
        dict(
            source_root=str(tmp_path / "balanced"),
            dataset=dict(path=str(dataset), sha256=follow.sha256_file(dataset)),
        ),
    )
    (root / "cohorts").mkdir()
    np.save(root / "cohorts/tracking.npy", [2, 3, 4, 5])
    np.save(root / "cohorts/train.npy", [0, 1])
    (root / "cache").mkdir()
    x = np.array([[0.1, 0.2], [-0.1, -0.2]], dtype=np.float32)
    np.savez(
        root / "cache/precision64_sleep.npz",
        x=x,
        model_flux=100 + 0.2 * x.astype(np.float64),
    )
    follow.write(
        root / "cache/precision64_sleep.npz.json",
        dict(
            generator="direct_frozen_parent",
            catalogue_truth_used=False,
            stored_flux_dtype="float64",
            photometry_numerics=photometry_numerics(cfg["model"]),
            latent_spec_hash=latent_spec_hash(latent_spec_from_config(cfg)),
            prior_fingerprint_sha256=_array_tree_sha256(model.prior),
        ),
    )
    seal(root)
    return root, model


def seal(root):
    follow.write(
        root / "NIGHT_ARTIFACTS.json",
        {
            str(p.relative_to(root)): follow.sha256_file(p)
            for p in root.rglob("*")
            if p.is_file() and p.name != "NIGHT_ARTIFACTS.json"
        },
    )


@pytest.mark.parametrize(
    "name",
    ["NIGHT_FINAL.json", "configs/C.yaml", "arms/C/train/checkpoints/best.eqx.json"],
)
def test_reject_changed_night(night, name):
    root, _ = night
    (root / name).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        follow.verify_night(root)


def test_require_qualified_not_merely_completed(night):
    root, _ = night
    report = follow.read(root / "FULL_DECODER_QUALIFICATION.json")
    report["cases"][0]["numerical_checks"] = "INCONCLUSIVE"
    follow.write(root / "FULL_DECODER_QUALIFICATION.json", report)
    seal(root)
    with pytest.raises(ValueError, match="qualification"):
        follow.verify_night(root)


@pytest.mark.parametrize("controlled", [False, True, "long"])
def test_followup_prepare_and_real_local_optimizer_mock_decoder(
    night, monkeypatch, controlled
):
    import scripts.run_feniks_sc_drws_local_vi_diagnostic as runner
    from euclid_dsps.amortized.latent import LatentSpec
    from euclid_dsps.amortized.posterior_target import posterior_log_target

    root, model = night
    out = root.parent / "local"
    manifest = runner.prepare(
        out,
        root.parent / "balanced",
        objects=2,
        steps=1,
        draws=32,
        qualified_night_root=root,
        controlled_optimization=controlled is True,
    )
    if controlled == "long":
        manifest.update(follow.long_optimization_recipe())
        # Exercise the real long-mode runner with a bounded mock-decoder smoke.
        manifest.update(steps=2, trajectory_steps=[1], evaluation_draws=32)
        follow.write(out / "RUN_MANIFEST.json", manifest)
    if controlled is True:
        manifest["steps"] = 2
        manifest["trajectory_steps"] = [1]
        follow.write(out / "RUN_MANIFEST.json", manifest)
    assert manifest["mode"] == "local_vi"
    assert manifest["qualified_night"]["arm"] == "C"
    assert manifest["population_training_started"] is False
    assert not any("truth" in c for c in manifest["catalogue_columns"])
    with pytest.raises(FileExistsError):
        follow.prepare(out, root.parent / "balanced", root)
    with pytest.raises(ValueError, match="bounded"):
        follow.prepare(root.parent / "bad", root.parent / "balanced", root, objects=16)
    spec = LatentSpec(
        ("x0", "x1"),
        jnp.full(2, -10.0),
        jnp.full(2, 10.0),
        arithmetic_precision="float64_v1",
    )
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner, "load_checkpoint", lambda *a: model)
    monkeypatch.setattr(runner, "latent_spec_from_config", lambda c: spec)
    monkeypatch.setattr(runner, "load_filters", lambda *a: None)
    monkeypatch.setattr(runner, "load_context", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "dynamic_model_args", lambda *a: None)

    def flux(x, *a, **kw):
        return 100 + 0.2 * x.astype(jnp.float64)

    monkeypatch.setattr(runner, "_model_flux_from_x_sample_chunks", flux)
    monkeypatch.setattr(
        runner,
        "posterior_log_target",
        lambda *a: posterior_log_target(*a, model_flux_fn=flux),
    )
    result = runner.run(out)
    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert result["cases_complete"] == 4
    assert result["prior_bitwise_unchanged"] is True
    assert result["scientific_promotion"] is False
    summary = follow.read(out / "cases/observed_000/start_0/SUMMARY.json")
    assert summary["pooled_draws"] == 64
    assert len(summary["independent_replicates"]) == 2
    bank = np.load(out / "cases/observed_000/start_0/direct_draws.npz")
    assert not np.array_equal(bank["x"][:32], bank["x"][32:])
    if controlled == "long":
        metadata = follow.read(out / "cases/observed_000/start_0/REGIME.json")
        assert metadata["name"] == "long_mc32"
        assert metadata["gradient_draws"] == 32
        assert (out / "cases/observed_000/start_0/step_0001/SUMMARY.json").exists()
        import shutil

        from scripts.feniks_support_probe import pin_source, verify_source

        with pytest.raises(ValueError, match="step 512"):
            pin_source(out, manifest, long_replay=True)
        # Supply production-length metadata around this two-step test fixture.
        manifest["steps"] = 512
        follow.write(out / "RUN_MANIFEST.json", manifest)
        reference = pin_source(out, manifest, long_replay=True)
        assert reference["source_starts"] == [0, 1]
        replay = root.parent / "replay"
        replay.mkdir()
        for name in ("config.yaml", "observed_rows.npy"):
            shutil.copyfile(out / name, replay / name)
        follow.write(
            replay / "RUN_MANIFEST.json",
            dict(
                manifest,
                method="qualified_long_replay_v1",
                support_probe=reference,
                steps=0,
                trajectory_steps=[],
                optimization_regimes=[
                    dict(name="final_replay", factor=1.0, mixture=False)
                ],
            ),
        )
        monkeypatch.setattr(
            runner, "make_step", lambda *a, **k: pytest.fail("replay must not optimize")
        )
        replay_result = runner.run(replay)
        assert replay_result["cases_complete"] == 4
        assert replay_result["budget"]["gradient_evaluations"] <= 5
        verify_source(reference)
        import subprocess
        import sys

        readback = subprocess.run(
            [
                sys.executable,
                "scripts/summarize_feniks_sc_drws_long_replay.py",
                str(replay),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "Final checkpoints only" in readback.stdout
        assert "both" in readback.stdout
        fresh = np.load(replay / "cases/observed_000/start_0/direct_draws.npz")
        assert not np.array_equal(fresh["x"], bank["x"])
        (out / "cases/observed_000/start_0/parameters.eqx").write_bytes(b"changed")
        with pytest.raises(ValueError, match="source changed"):
            verify_source(reference)
    if controlled is True:
        from scripts.summarize_feniks_sc_drws_controlled_local_vi import collect

        trajectory = collect(out)
        assert len(trajectory) == 4 * (1 + 6 * 2)
        assert set(trajectory.regime) == {"amortized", "original", "slow", "slow_mc16"}
        assert set(trajectory.step) == {0, 1, 2}
        metadata = follow.read(out / "cases/observed_000/start_4/REGIME.json")
        assert metadata["gradient_draws"] == 16
        assert metadata["learning_rate"] == 0.0001
        assert metadata["initialization"] == 0
        import shutil

        from scripts.audit_feniks_saved_draws import audit
        from scripts.feniks_support_probe import pin_source, verify_source

        reference = pin_source(out, manifest)
        verify_source(reference)
        audit_frame = audit(out, root.parent / "cpu_audit")
        assert len(audit_frame) == len(trajectory)
        assert set(audit_frame.status) == {"DESCRIPTIVE_ONLY"}
        probe = root.parent / "probe"
        probe.mkdir()
        for name in ("config.yaml", "observed_rows.npy"):
            shutil.copyfile(out / name, probe / name)
        probe_manifest = dict(
            manifest,
            support_probe=reference,
            steps=0,
            trajectory_steps=[],
            optimization_regimes=[
                dict(name="x1", factor=1.0, mixture=False),
                dict(name="mix", factor=1.5, mixture=True),
            ],
        )
        follow.write(probe / "RUN_MANIFEST.json", probe_manifest)
        monkeypatch.setattr(
            runner, "make_step", lambda *a, **k: pytest.fail("probe must not optimize")
        )
        probe_result = runner.run(probe)
        assert probe_result["cases_complete"] == 4
        assert probe_result["prior_bitwise_unchanged"]
        assert probe_result["budget"]["gradient_evaluations"] <= 5
        draws = np.load(probe / "cases/observed_000/amortized/direct_draws.npz")
        old_draws = np.load(out / "cases/observed_000/amortized/direct_draws.npz")
        assert not np.array_equal(draws["x"], old_draws["x"])
        receipt = out / "cases/observed_000/start_4/REGIME.json"
        receipt.write_text("{}")
        with pytest.raises(ValueError, match="source changed"):
            verify_source(reference)


def test_calibration_audit_does_not_reclassify_small_sample_fail():
    value = dict(
        objects=64,
        parameters={str(i): {} for i in range(32)},
        status="FAIL",
        maximum_coordinate_pit_ks=0.2,
        maximum_coordinate_coverage_ece=0.14,
        thresholds=dict(maximum_ks=0.12),
    )
    result = calibration_summary(value)
    assert result["status_as_recorded"] == "FAIL"
    assert result["simultaneous_95pct_dkw_bound"] > 0.2


def test_relative_heldout_pass_warns_on_bad_reference():
    value = dict(
        held_out_observed_residuals=dict(by_band=dict(b=dict(rms=2.0))),
        held_out_model_generated_reference=dict(by_band=dict(b=dict(rms=1000.0))),
        held_out_band=dict(status="PASS", bands=dict(b=dict(rms_ratio=0.002))),
    )
    before = copy.deepcopy(value)
    result = heldout_summary(value)
    assert result["bands"]["b"]["reference_dominated_warning"]
    assert result["status_as_recorded"] == "PASS"
    assert value == before


def test_cpu_readback_full_artifact_schema(night):
    root, _ = night
    calibration = dict(
        objects=64,
        parameters={"z": {}},
        status="FAIL",
        maximum_coordinate_pit_ks=0.2,
        maximum_coordinate_coverage_ece=0.14,
        thresholds=dict(maximum_ks=0.12),
    )
    internal = dict(
        model_generated_calibration=calibration,
        joint_projection_calibration=calibration,
        held_out_observed_residuals=dict(by_band=dict(b=dict(rms=2.0))),
        held_out_model_generated_reference=dict(by_band=dict(b=dict(rms=1000.0))),
        held_out_band=dict(status="PASS", bands=dict(b=dict(rms_ratio=0.002))),
    )
    final = follow.read(root / "NIGHT_FINAL.json")
    final["summaries"] = {}
    for arm in "ABC":
        final["summaries"][arm] = dict(support={}, technical_gate=dict(status="FAIL"))
        follow.write(
            root / f"validation/{arm}/internal/INTERNAL_TRUTH_FREE_VALIDATION.json",
            internal,
        )
        if arm != "A":
            follow.write(root / f"arms/{arm}/train/training_summary.json", {})
            pd.DataFrame(dict(epoch=[1], loss=[2.0])).to_csv(
                root / f"arms/{arm}/train/training_log.csv", index=False
            )
    follow.write(root / "NIGHT_FINAL.json", final)
    seal(root)
    before = follow.sha256_file(root / "NIGHT_FINAL.json")
    result = analyze(root)
    assert result["status"] == "READBACK_COMPLETE"
    assert result["decoder_calls"] == 0
    assert result["arms"]["C"]["heldout"]["bands"]["b"]["reference_dominated_warning"]
    assert follow.sha256_file(root / "NIGHT_FINAL.json") == before


def test_preparation_rejects_training_overlap(night):
    root, _ = night
    np.save(root / "cohorts/train.npy", [2])
    seal(root)
    with pytest.raises(ValueError, match="overlap"):
        follow.prepare(root.parent / "bad", root.parent / "balanced", root, objects=2)


def test_preparation_rejects_reordered_parameter_coordinates(night):
    root, _ = night
    cfg = follow.load_config(root / "configs/C.yaml")
    # Sorting YAML changes the config_free_parameters coordinate order.
    (root / "configs/C.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    seal(root)
    with pytest.raises(ValueError, match="qualified simulator"):
        follow.prepare(root.parent / "bad", root.parent / "balanced", root, objects=2)
