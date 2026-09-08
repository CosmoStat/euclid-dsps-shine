from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from euclid_dsps.amortized.npe_validation import (
    summarize_model_generated_rank_calibration,
    summarize_projected_rank_calibration,
)
from euclid_dsps.amortized.train import _sleep_conditioning_mask
from scripts import run_feniks_sc_drws_balanced_npe as runner


def test_conditioning_mask_is_opt_in_and_preserves_selection():
    mask = jnp.ones((2000, 4), dtype=bool).at[0, 0].set(False)
    key = jax.random.PRNGKey(2)
    np.testing.assert_array_equal(_sleep_conditioning_mask(mask, key, {}), mask)
    config = {
        "conditioning_mask_probability": 0.25,
        "conditioning_mask_indices": (0, 2),
        "selection_band_index": 1,
    }
    actual = np.asarray(_sleep_conditioning_mask(mask, key, config))
    np.testing.assert_array_equal(actual[:, 1], mask[:, 1])
    np.testing.assert_array_equal(actual[:, 3], mask[:, 3])
    assert not actual[0, 0]
    assert 0.20 < np.mean(~actual[:, 2]) < 0.30
    with pytest.raises(ValueError, match="selection band"):
        _sleep_conditioning_mask(mask, key, {**config, "selection_band_index": 2})


def test_joint_projection_detects_wrong_dependence_despite_correct_marginals():
    rng = np.random.default_rng(31)
    truth = rng.multivariate_normal([0, 0], [[1, 0.99], [0.99, 1]], size=2048)
    incorrect = rng.normal(size=(128, 2048, 2))
    marginal = summarize_model_generated_rank_calibration(
        incorrect,
        truth,
        parameter_names=("x", "y"),
        seed=2,
        maximum_ks=0.06,
        maximum_coverage_ece=0.06,
    )
    assert marginal["status"] == "PASS"
    joint = summarize_projected_rank_calibration(
        incorrect,
        truth,
        scale=np.ones(2),
        seed=2,
        maximum_ks=0.06,
        maximum_coverage_ece=0.06,
    )
    assert joint["status"] == "FAIL"
    correct = rng.multivariate_normal([0, 0], [[1, 0.99], [0.99, 1]], size=(128, 2048))
    assert (
        summarize_projected_rank_calibration(
            correct,
            truth,
            scale=np.ones(2),
            seed=2,
            maximum_ks=0.06,
            maximum_coverage_ece=0.06,
        )["status"]
        == "PASS"
    )


def valid_summary(nll=20):
    return {
        "truth_used": False,
        "tracking": {"technical_gate": {"status": "PASS"}},
        "support": {"technical_gate": {"status": "PASS"}},
        "internal": {
            "truth_used": False,
            "simulation_sha256": "common",
            "simulation_nll": nll,
            **{
                name: {"status": "PASS"}
                for name in (
                    "held_out_band",
                    "model_generated_calibration",
                    "joint_projection_calibration",
                    "masked_model_generated_calibration",
                    "masked_joint_projection_calibration",
                )
            },
        },
    }


def test_selection_requires_joint_and_support_and_sleep_preservation():
    rows = {arm: valid_summary() for arm in runner.ARMS}
    rows["A"]["internal"]["simulation_nll"] = 0
    rows["C"]["internal"]["simulation_nll"] = 19
    assert runner.choose(rows, 0.5)["winner"] == "C"
    rows["C"]["internal"]["joint_projection_calibration"]["status"] = "FAIL"
    assert runner.choose(rows, 0.5)["winner"] == "B"
    rows["B"]["support"]["technical_gate"]["status"] = "FAIL"
    rows["D"]["internal"]["simulation_nll"] = 21
    assert runner.choose(rows, 0.5)["winner"] is None
    rows["A"]["internal"]["simulation_sha256"] = "different"
    with pytest.raises(ValueError, match="same simulated"):
        runner.choose(rows, 0.5)


def test_reserved_confirmation_excludes_train_and_development():
    available = runner.split_confirmation(np.arange(10, 30), [10, 12], np.arange(10), 4)
    assert not np.intersect1d(available, [10, 12]).size
    with pytest.raises(ValueError, match="disjoint"):
        runner.split_confirmation([1, 2], [], [1], 1)


def test_no_candidate_skips_confirmation_and_blocks_population(tmp_path):
    runner.write(tmp_path / "CANDIDATE_FROZEN.json", {"winner": None})
    assert runner.confirm(tmp_path, {})["status"] == "SKIPPED"
    result = runner.close(tmp_path, {})
    assert result["posterior_technical_ready"] is False
    assert result["population_training_started"] is False
    assert result["scientific_promotion"] is False


def test_confirmation_cannot_replace_winner_after_failure(tmp_path):
    runner.write(tmp_path / "CANDIDATE_FROZEN.json", {"winner": "C"})
    summary = valid_summary()
    summary["support"]["technical_gate"]["status"] = "FAIL"
    runner.write(tmp_path / "CONFIRMATION_COMPLETE.json", summary)
    result = runner.close(tmp_path, {})
    assert result["candidate"] == "C"
    assert result["posterior_technical_ready"] is False


def test_submission_dependencies_and_restart_do_not_duplicate_jobs(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setenv("REPO_DIR", str(tmp_path))
    monkeypatch.setenv("BALANCED_LOG_ROOT", str(tmp_path / "logs"))
    monkeypatch.setenv("BALANCED_ENV", str(tmp_path / "latest.env"))
    monkeypatch.setattr(runner, "require_git_commit", lambda *a: None)
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        return str(100 + len(calls)) + "\n"

    monkeypatch.setattr(runner.subprocess, "check_output", sbatch)
    manifest = {"code_commit": "abc"}
    result = runner.submit(root, manifest)
    assert len(calls) == 6
    assert "--dependency=afterok:102" in calls[2]
    assert "--array=0-3%4" in calls[2]
    assert result["maximum_concurrent_gpus"] == 4
    runner.submit(root, manifest)
    assert len(calls) == 6


def test_partial_training_is_preserved(tmp_path, monkeypatch):
    out = tmp_path / "arms/B/train"
    out.mkdir(parents=True)
    (out / "keep").write_text("partial")
    monkeypatch.setattr(runner, "verify_cache", lambda root: None)
    with pytest.raises(FileExistsError, match="partial training preserved"):
        runner.train(tmp_path, {}, "B")
    assert (out / "keep").read_text() == "partial"


@pytest.mark.parametrize("arm", ("S", "B", "C", "D"))
def test_training_command_matches_single_gpu_allocation(tmp_path, monkeypatch, arm):
    from euclid_dsps.amortized.train import _resolve_data_parallel_training

    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}")
    runner.write(tmp_path / "SED_SMOKE_COMPLETE.json", {"status": "COMPLETE"})
    source = {"checkpoint": "source.eqx", "feature_stats": "stats.json"}
    if arm != "S":
        runner.write(tmp_path / "arms/S/ARM_COMPLETE.json", source)
    manifest = {
        "source": source,
        "configs": {arm: {"path": str(cfg)}},
        "dataset": {"path": "observed.parquet"},
        "cohorts": {name: {"path": name + ".npy"} for name in ("train", "validation")},
        "recipe": {"anchor_epochs": 24, "candidate_epochs": 8, "seed": 260908},
    }
    monkeypatch.setattr(runner, "verify_cache", lambda root: None)
    monkeypatch.setattr(runner, "verified_artifacts", lambda value: value)
    monkeypatch.setattr(runner, "certify_training", lambda root, arm: {"arm": arm})
    calls = []
    monkeypatch.setattr(runner, "cli", lambda *args: calls.append(args))
    monkeypatch.setattr(jax, "local_devices", lambda: (object(),))
    assert runner.train(tmp_path, manifest, arm) == {"arm": arm}
    assert len(calls) == 1
    command = calls[0]
    mode = command[command.index("--data-parallel") + 1]
    assert mode == "single"
    actual = _resolve_data_parallel_training({"data_parallel": mode}, jax_batch_size=256)
    assert actual["enabled"] is False
    assert actual["per_device_batch_size"] == 256
    assert "--freeze-prior" in command


def test_prepare_reserves_confirmation_and_disables_truth_in_every_config(
    tmp_path, monkeypatch
):
    source_root = tmp_path / "pilot"
    source_root.mkdir()
    cfg = {
        "truth": {"parameter_columns": {}},
        "amortized": {
            "objective": {
                "mode": "reweighted_wake_sleep",
                "sleep": {"enabled": True},
                "wake": {"train_encoder": False, "train_prior": False},
            },
            "prior": {"train_jointly": False},
            "training": {},
        },
    }
    (source_root / "config.yaml").write_text(yaml.safe_dump(cfg))
    (source_root / "model.eqx").write_text("fixture checkpoint")
    (source_root / "feature_stats.json").write_text("{}")
    artifact = {
        "status": "COMPLETE",
        "prior_bitwise_unchanged": True,
        "truth_used_for_training_or_checkpoint_selection": False,
        "topology": {"minimum_transform_count": 2},
    }
    for key, name in (
        ("config", "config.yaml"),
        ("checkpoint", "model.eqx"),
        ("feature_stats", "feature_stats.json"),
    ):
        artifact[key] = str(source_root / name)
        artifact[key + "_sha256"] = runner.sha256_file(source_root / name)
    runner.write(source_root / "arms/B/ARM_COMPLETE.json", artifact)
    rows = {
        "train": np.arange(10),
        "validation": np.arange(10, 1290),
        "validation_pilot": np.arange(10, 266),
        "support_pilot": np.arange(10, 138),
    }
    cohorts = {}
    for key, values in rows.items():
        path = source_root / f"{key}.npy"
        np.save(path, values)
        cohorts[key] = {"path": str(path)}
    dataset = source_root / "dataset.parquet"
    dataset.write_text("observed-only fixture")
    runner.write(
        source_root / "RUN_MANIFEST.json",
        {
            "source": artifact,
            "cohorts": cohorts,
            "dataset": {"path": str(dataset), "sha256": runner.sha256_file(dataset)},
        },
    )
    monkeypatch.setattr(
        runner, "load_config", lambda path: yaml.safe_load(Path(path).read_text())
    )
    monkeypatch.setattr(
        runner,
        "_representative_observed_rows",
        lambda config, rows, count: (rows[:count], {"truth_used": False}),
    )
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "abcd\n")
    root = tmp_path / "balanced"
    manifest = runner.prepare(
        source_root,
        root,
        Path("configs/experiments/feniks_sc_drws_r29_balanced_npe.yaml"),
        Path.cwd(),
    )
    confirmation = np.load(manifest["cohorts"]["confirmation"]["path"])
    val = np.load(manifest["cohorts"]["validation"]["path"])
    assert not np.intersect1d(confirmation, val).size
    assert not np.intersect1d(confirmation, rows["validation_pilot"]).size
    assert len(confirmation) == 256
    for arm, entry in manifest["configs"].items():
        resolved = yaml.safe_load(Path(entry["path"]).read_text())
        if arm != "A":
            assert resolved["amortized"]["training"]["data_parallel"] == "single"
        inference = resolved["amortized"]["inference"]
        assert inference["write_truth_snapshot"] is False
        assert inference["write_truth_diagnostics"] is False
        from euclid_dsps.amortized.catalog_identity import configured_redshift_column

        assert configured_redshift_column(resolved) is None
        assert resolved["amortized"]["data"]["use_redshift_for_split"] is False
        assert (
            "lsst_r"
            not in resolved["amortized"]["truth_free_validation"]["held_out_bands"]
        )
        assert resolved["amortized"]["truth_free_validation"]["joint_projections"] == 32
    with pytest.raises(FileExistsError):
        runner.prepare(
            source_root,
            root,
            Path("configs/experiments/feniks_sc_drws_r29_balanced_npe.yaml"),
            Path.cwd(),
        )


def test_strict_observed_elbo_does_not_drop_invalid_draws(monkeypatch):
    from types import SimpleNamespace

    from euclid_dsps.amortized import train as training

    monkeypatch.setattr(
        training,
        "sample_posterior",
        lambda *a: SimpleNamespace(x=jnp.zeros((4, 2, 1)), logq=jnp.ones((4, 2))),
    )
    valid = jnp.ones((4, 2), dtype=bool).at[0, 0].set(False)
    monkeypatch.setattr(
        training,
        "posterior_log_target",
        lambda *a: SimpleNamespace(
            physical_valid=valid,
            loglike=jnp.zeros((4, 2)),
            logprior=jnp.zeros((4, 2)),
            logtarget=jnp.zeros((4, 2)),
        ),
    )
    args = (None, SimpleNamespace(features=None), None, None, None, None, None, {}, {})
    old, _ = training.observed_reverse_kl_loss(*args, n_samples=4)
    strict, _ = training.observed_reverse_kl_loss(
        *args, n_samples=4, require_all_finite=True
    )
    assert np.isfinite(old)
    assert np.isposinf(strict)


def test_pure_and_mixed_sleep_use_same_key_when_explicitly_requested(monkeypatch):
    from euclid_dsps.amortized import train as training

    keys = []

    def fake_sleep(*args, **kwargs):
        keys.append(np.asarray(args[6]))
        return jnp.asarray(1.0), {}

    monkeypatch.setattr(training, "_model_generated_sleep_loss", fake_sleep)
    metrics = {
        key: jnp.asarray(0.0)
        for key in (
            "reverse_kl",
            "negative_loglike",
            "loglike_mean",
            "logprior_mean",
            "logq_mean",
            "physical_valid_fraction",
            "decoder_evaluations",
        )
    }
    monkeypatch.setattr(
        training,
        "observed_reverse_kl_loss",
        lambda *a, **k: (jnp.asarray(0.0), metrics),
    )
    args = (None, None, None, None, None, None, jax.random.PRNGKey(5), {}, {})
    common = {"common_sleep_random_numbers": True, "sleep_weight": 1.0, "weight": 0.001}
    for enabled in (False, True):
        training._reweighted_sleep_objective_loss(
            *args, {"observed_elbo": {**common, "enabled": enabled}}
        )
    np.testing.assert_array_equal(keys[0], keys[1])


def test_comparison_writes_diagnostic_table_and_nonblank_plot(tmp_path):
    from PIL import Image

    rows = {arm: valid_summary() for arm in runner.ARMS}
    for value in rows.values():
        value["support"]["support"] = {
            "raw_ess": {"fraction_median": 0.02, "fraction_q10": 0.001},
            "pareto_k": {
                "gt_0p7_or_nonfinite_fraction": 0.8,
                "nonfinite_fraction": 0.1,
            },
            "maximum_raw_weight": {"p90": 0.98},
        }
        for key in (
            "joint_projection_calibration",
            "masked_joint_projection_calibration",
        ):
            value["internal"][key]["maximum_coordinate_pit_ks"] = 0.15
    runner.write_comparison(tmp_path, rows)
    assert len((tmp_path / "matched_validation.csv").read_text().splitlines()) == 5
    pixels = np.asarray(Image.open(tmp_path / "matched_validation.png"))
    assert pixels.shape[0] > 100 and pixels.std() > 10
