"""Real small classifier/flow recovery; synthetic DSPS boundary only."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import feniks_overnight_recovery as rec
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish


@pytest.fixture(scope="module")
def completed_sources(tmp_path_factory):
    from test_feniks_conditional_parent import config, fixture_source
    from test_feniks_parent_to_posterior import configuration

    tmp = tmp_path_factory.mktemp("overnight_sources")
    source, dataset = fixture_source(tmp)
    conditional, posterior = tmp / "conditional", tmp / "posterior"
    rec.cp.initialize(source, conditional, config(tmp))
    rec.cp.prepare(conditional)
    for task in range(2):
        rec.cp.relabel(conditional, task)
    rec.cp.population(conditional)
    rec.ptp.initialize(source, posterior, configuration(tmp), True)
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            rec.ptp, "photometer", lambda *_: lambda t: np.full((len(t), 1), 9e-32)
        )
        for task in range(2):
            rec.ptp.bank(posterior, task)
    rec.ptp.train(posterior)
    rec.ptp.evaluate(posterior)
    (posterior / "report").mkdir()
    write(posterior / "report/DECISION.json", dict(ready_for_production=False))
    finish(
        posterior / "report",
        [posterior / "report/DECISION.json"],
        sha(posterior / "MANIFEST.json"),
    )
    return conditional, posterior


def config(tmp_path):
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_overnight_recovery.yaml").read_text()
    )
    cfg["continuation"].update(
        epochs=2, block_epochs=1, validation_objects=16, validation_draws=16
    )
    cfg["audit"]["extreme_draws"] = 2
    cfg["audit"]["backend"] = "cpu"  # Fixture evaluations are generated on CPU.
    path = tmp_path / "recovery.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_tail_metrics_do_not_hide_a_rare_extreme():
    from euclid_dsps.amortized.posterior_tail_audit import extreme_positions, tail_table

    truth = np.random.default_rng(0).normal(size=(128, 15))
    draws = np.repeat(truth[:, None], 64, axis=1)
    draws[7, 4, 9] = 1e18
    table = tail_table(draws, truth, rec.ci.NAMES)
    assert table.iloc[9].far_draw_fraction == pytest.approx(1 / (128 * 64))
    assert table.iloc[9].far_object_fraction == pytest.approx(1 / 128)
    assert table.iloc[9].w1_over_truth_iqr > 1e12
    assert tuple(extreme_positions(draws, truth, 1)[0]) == (7, 4)
    draws[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite joint"):
        tail_table(draws, truth, rec.ci.NAMES)


def test_replay_inputs_preserve_stored_float32_context_and_validate_metadata():
    from euclid_dsps.amortized.posterior_tail_audit import replay_inputs

    original = np.array([[0.125, -3.0]], dtype=np.float32)
    reconstructed = np.nextafter(original, np.float32(np.inf))
    saved = dict(
        features=original,
        sampling_backend=np.array("gpu"),
        sampling_seed=np.array(31),
        sampling_batch_size=np.array(16),
    )
    kwargs = dict(backend="gpu", expected_backend="gpu", seed=31)
    features, info = replay_inputs(saved, reconstructed, **kwargs)
    np.testing.assert_array_equal(features, original)
    assert features.dtype == np.float32
    assert info["recomputed_feature_max_abs_error"] > 0
    assert info["feature_source"] == "saved_evaluation"
    for changes, message in (
        ({"sampling_backend": "cpu"}, "Saved evaluation backend"),
        ({"sampling_seed": 32}, "sampling_seed"),
        ({"sampling_batch_size": 8}, "sampling_batch_size"),
        ({"features": np.full((1, 2), np.nan)}, "Finite matching"),
        ({"features": np.zeros((2, 2))}, "Finite matching"),
    ):
        with pytest.raises(ValueError, match=message):
            replay_inputs(dict(saved, **changes), reconstructed, **kwargs)
    with pytest.raises(ValueError, match="Replay backend mismatch"):
        replay_inputs(saved, reconstructed, **dict(kwargs, backend="cpu"))
    # Legacy artifacts may reconstruct contexts only on the declared backend.
    features, info = replay_inputs({}, reconstructed, **kwargs)
    np.testing.assert_array_equal(features, reconstructed)
    assert info["saved_backend"] is None
    assert info["feature_source"] == "legacy_reconstruction"
    with pytest.raises(ValueError, match="Replay backend mismatch"):
        replay_inputs({}, reconstructed, **dict(kwargs, backend="cpu"))


def test_bounded_float64_rounding_is_not_a_sampling_error():
    import json

    from euclid_dsps.amortized.coherent_coordinates import to_theta
    from euclid_dsps.amortized.posterior_tail_audit import replay_coordinates

    spec = dict(
        names=rec.ci.NAMES,
        bounded=[True] + [False] * 14,
        lower=[0.0] * 15,
        upper=[1.0] * 15,
        location=[0.0] * 15,
        width=[1.0] * 15,
        center=[0.0] * 15,
        scale=[1.0] * 15,
    )
    x = np.zeros((1, 15))
    x[0, 0] = 34.0
    theta = np.asarray(to_theta(x, spec))
    table, check = replay_coordinates(x, theta, spec, tolerance=0.001)
    assert check["maximum_replay_latent_error"] > 0.001
    assert check["bounded_quantization_compatible_coordinates"] == 1
    assert check["passed"] and not check["stored_latent_available"]
    assert table.iloc[0].replay_theta == theta[0, 0]
    _, check = replay_coordinates(x, theta, spec, tolerance=0.001, saved_x=x)
    assert check["passed"] and check["maximum_stored_latent_error"] == 0
    corrupt_x = x.copy()
    corrupt_x[0, 0] -= 0.01
    _, check = replay_coordinates(x, theta, spec, tolerance=0.001, saved_x=corrupt_x)
    assert not check["passed"]  # Never excuse a discrepancy in directly saved x.
    corrupt_theta = theta.copy()
    corrupt_theta[0, 0] -= 1e-9
    _, check = replay_coordinates(x, corrupt_theta, spec, tolerance=0.001)
    assert not check["passed"]
    corrupt_theta = theta.copy()
    corrupt_theta[0, 9] = 1e18
    _, check = replay_coordinates(x, corrupt_theta, spec, tolerance=0.001)
    assert not check["passed"]  # SFH has no bounded-coordinate exemption.
    corrupt_theta = theta.copy()
    corrupt_theta[0, 0] = 1.0
    _, check = replay_coordinates(x, corrupt_theta, spec, tolerance=0.001)
    assert not check["passed"]  # A saturated endpoint is not finite latent evidence.
    json.dumps(check, allow_nan=False)
    saturated_x = x.copy()
    saturated_x[0, 0] = 100.0
    _, check = replay_coordinates(saturated_x, theta, spec, tolerance=0.001)
    assert not check["passed"] and check["nonfinite_coordinates"] == 1


def test_actual_recovery_and_bounded_training_preserve_sources(
    completed_sources, tmp_path, monkeypatch
):
    conditional, posterior = completed_sources
    before = {
        p: sha(p)
        for folder in (conditional, posterior)
        for p in folder.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "recovery"
    rec.initialize(conditional, posterior, root, config(tmp_path))
    monkeypatch.setattr(
        rec.ci,
        "bounded_fit",
        lambda *a, **kw: pytest.fail("Classifier or original q retrained"),
    )
    monkeypatch.setattr(
        rec.ptp, "bank", lambda *a, **kw: pytest.fail("No DSPS/new bank allowed")
    )
    rec.recover_parent(root)
    rec.audit(root)
    decision = read(root / "audit/DECISION.json")
    assert decision["safe_to_optimize"]
    assert decision["runtime"]["backend"] == "cpu"
    assert decision["runtime"]["x64_enabled"]
    for label in ("in_model", "coherent_target"):
        numerics = decision["checks"][label]
        assert numerics["replay_graph"] == "evaluate_posterior_x_only_v1"
        assert numerics["xonly_full_graph_max_delta"] >= 0.0
        assert numerics["inputs"]["feature_source"] == "saved_evaluation"
        extremes = pd.read_csv(root / f"audit/{label}/extreme_draws.csv")
        assert {
            "full_graph_replay_latent_error",
            "xonly_full_graph_delta",
        }.issubset(extremes.columns)
    assert not read(root / "audit/DECISION.json")["scientific_pass"]
    rec.train(root)
    baseline_nll = read(posterior / "posterior/STOP.json")["best_nll"]
    assert read(root / "posterior/INITIAL_VALIDATION.json")["nll"] == pytest.approx(
        baseline_nll, abs=1e-5
    )
    assert read(root / "posterior/training_measure.json")["optimizer_reset"]
    assert read(root / "posterior/RESUME.json")["epoch"] == 2
    assert len(list((root / "milestones").glob("epoch_*/FINAL.json"))) == 2
    rec.evaluate(root)
    rec.report(root)
    assert not read(root / "report/DECISION.json")["ready_for_production"]
    assert (root / "report/tails.png").is_file()
    assert len(pd.read_csv(root / "report/calibration_comparison.csv")) == 60
    assert len(pd.read_csv(root / "report/tail_comparison.csv")) == 120
    digest = sha(root / "MANIFEST.json")
    assert complete(root / "report", digest)
    assert before == {p: sha(p) for p in before}
    rec.recover_parent(root)
    rec.audit(root)
    rec.train(root)
    rec.evaluate(root)
    rec.report(root)
    assert before == {p: sha(p) for p in before}
    result = subprocess.run(
        ["bash", "scripts/watch_feniks_overnight_recovery.sh", str(root), "--once"],
        env=dict(
            os.environ, PATH=f"{Path(sys.executable).parent}:" + os.environ["PATH"]
        ),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0 and "no production approval" in result.stdout
    assert "backend=cpu" in result.stdout and "failed_coords=0" in result.stdout


def test_audit_replays_saved_contexts_despite_reconstruction_drift(
    completed_sources, tmp_path, monkeypatch
):
    root = tmp_path / "drift"
    rec.initialize(*completed_sources, root, config(tmp_path))
    original = rec.cohort_features

    def drifted(*args):
        features, truth = original(*args)
        return features + np.float32(0.25), truth

    monkeypatch.setattr(rec, "cohort_features", drifted)
    rec.audit(root)
    decision = read(root / "audit/DECISION.json")
    assert decision["safe_to_optimize"]
    for check in decision["checks"].values():
        assert check["inputs"]["feature_source"] == "saved_evaluation"
        assert check["inputs"]["recomputed_feature_max_abs_error"] > 0.2


def test_audit_rejects_wrong_backend_before_replaying(completed_sources, tmp_path):
    path = config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg["audit"]["backend"] = "gpu"
    path.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "wrong_backend"
    rec.initialize(*completed_sources, root, path)
    with pytest.raises(ValueError, match="Replay backend mismatch"):
        rec.audit(root)
    assert not (root / "audit/FINAL.json").exists()
    assert not (root / "audit/DECISION.json").exists()
    runtime = read(root / "audit/runtime.json")
    assert runtime["backend"] == "cpu" and runtime["expected_backend"] == "gpu"


def test_negative_numerical_gate_blocks_without_training(
    completed_sources, tmp_path, monkeypatch
):
    root = tmp_path / "blocked"
    rec.initialize(*completed_sources, root, config(tmp_path))
    digest = sha(root / "MANIFEST.json")
    write(root / "audit/DECISION.json", dict(safe_to_optimize=False))
    finish(root / "audit", [root / "audit/DECISION.json"], digest)
    monkeypatch.setattr(
        rec.ptp,
        "selected_rows",
        lambda *_: pytest.fail("Training bank read despite audit FAIL"),
    )
    with pytest.raises(ValueError, match="Passed numerical audit"):
        rec.train(root)
    with pytest.raises(ValueError, match="Numerical audit failed"):
        rec.schedule(root)


def test_resume_after_training_before_milestone_validation(
    completed_sources, tmp_path, monkeypatch
):
    root = tmp_path / "interrupted"
    rec.initialize(*completed_sources, root, config(tmp_path))
    digest = sha(root / "MANIFEST.json")
    write(root / "audit/DECISION.json", dict(safe_to_optimize=True))
    finish(root / "audit", [root / "audit/DECISION.json"], digest)
    original = rec.evaluate_model

    def interrupted(*args, **kwargs):
        raise RuntimeError("injected milestone interruption")

    monkeypatch.setattr(rec, "evaluate_model", interrupted)
    with pytest.raises(RuntimeError, match="milestone interruption"):
        rec.train(root)
    state = read(root / "posterior/RESUME.json")
    assert state["epoch"] == 1
    assert sha(root / "posterior" / state["state_file"]) == state["state_sha256"]
    monkeypatch.setattr(rec, "evaluate_model", original)
    rec.train(root)
    assert pd.read_json(
        root / "posterior/training.jsonl", lines=True
    ).epoch.tolist() == [1, 2]
    assert read(root / "posterior/INITIAL_VALIDATION.json")["optimizer_reset"]
    assert complete(root / "posterior", digest)


def test_failed_replay_is_not_a_pass(completed_sources, tmp_path, monkeypatch):
    import json

    import equinox as eqx

    from euclid_dsps.amortized import structured_population

    _, source = completed_sources
    _, pc, digest = rec.ptp.settings(source)
    _, spec, _ = rec.ci.require_reference(source, digest)
    with np.load(source / "evaluation/in_model/draws.npz") as f:
        draws, truth, positions = (
            f[k] for k in ("draws", "truth", "evaluation_positions")
        )
        assert f["latent_x"].shape == draws.shape
    features, _ = rec.cohort_features(source, pc, digest, "in_model", positions)
    model = eqx.tree_deserialise_leaves(
        source / "posterior/best.eqx", rec.ptp.template(pc, features.shape[1])
    )
    draws[0, 0, 9] = 1e18
    _, audit = rec.replay_extremes(
        model,
        features,
        draws,
        truth,
        spec,
        pc["seed"] + 20,
        yaml.safe_load(config(tmp_path).read_text()),
    )
    assert not audit["passed"]
    monkeypatch.setattr(
        structured_population,
        "transport_audit",
        lambda *a, **kw: dict(
            passed=False, tolerance=1e-5, experts=[dict(inverse_error=float("nan"))]
        ),
    )
    _, audit = rec.replay_extremes(
        model,
        features,
        draws,
        truth,
        spec,
        pc["seed"] + 20,
        yaml.safe_load(config(tmp_path).read_text()),
    )
    assert not audit["passed"] and audit["nonfinite_metrics"] == [
        "expert_0.inverse_error"
    ]
    json.dumps(audit, allow_nan=False)


def test_reaudit_reuses_only_matching_completed_parent(
    completed_sources, tmp_path, monkeypatch
):
    root, new = tmp_path / "previous", tmp_path / "new"
    rec.initialize(*completed_sources, root, config(tmp_path))
    rec.recover_parent(root)
    write(root / "audit/DECISION.json", dict(safe_to_optimize=False))
    finish(root / "audit", [root / "audit/DECISION.json"], sha(root / "MANIFEST.json"))
    before = {p: sha(p) for p in root.rglob("*") if p.is_file()}
    rec.initialize(*completed_sources, new, config(tmp_path), reuse_parent=root)
    monkeypatch.setattr(
        rec.cp, "population", lambda *_: pytest.fail("Parent recomputed")
    )
    rec.recover_parent(new)
    assert complete(new / "parent", sha(new / "MANIFEST.json"))
    assert not (new / "parent/report").is_symlink()
    assert sha(new / "parent/report/DECISION.json") == sha(
        root / "parent/report/DECISION.json"
    )
    assert not (new / "audit/FINAL.json").exists()
    rec.audit(new)
    assert read(new / "audit/DECISION.json")["safe_to_optimize"]
    assert (new / "audit/coherent_target/replay_coordinates.csv").is_file()
    assert before == {p: sha(p) for p in before}
    write(new / "parent/report/DECISION.json", dict(corrupted=True))
    with pytest.raises(ValueError, match="Prepared configuration changed"):
        rec.settings(new)
    m = read(root / "MANIFEST.json")
    m["source_posterior"] = str(tmp_path / "wrong_source")
    write(root / "MANIFEST.json", m)
    with pytest.raises(ValueError, match="matching recovery"):
        rec.initialize(
            *completed_sources, tmp_path / "bad", config(tmp_path), reuse_parent=root
        )


def test_slurm_parallel_gate_frozen_resume_and_no_mem(completed_sources, tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls, counter = tmp_path / "calls", tmp_path / "counter"
    counter.write_text("900")
    for name, body in (
        (
            "sbatch",
            f'n=$(<{counter}); n=$((n+1)); echo "$n" > {counter}; printf "%s\\n" "$*" >> {calls}; echo "$n"',
        ),
        ("squeue", 'echo "${LIVE:-}"; exit "${QUEUE_STATUS:-0}"'),
        ("scancel", "exit 99"),
    ):
        path = bindir / name
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{Path(sys.executable).parent}:" + os.environ["PATH"],
        SCRATCH=str(tmp_path / "scratch"),
    )
    root = tmp_path / "avi_overnight_recovery_20260928_090000"
    command = ["bash", "scripts/submit_feniks_overnight_recovery.sh"]
    p = subprocess.run(
        command + [*map(str, completed_sources), str(root), str(config(tmp_path))],
        env=env,
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 4
    assert "--dependency" not in lines[0] and "--dependency" not in lines[1]
    assert "--gres" not in lines[0] and "--gres" not in lines[1]
    assert "--dependency=afterok:902" in lines[2] and "--gres=gpu:1" in lines[2]
    assert "--dependency=afterany:901:902:903" in lines[3]
    assert "--mem" not in "\n".join(lines)
    assert (
        sha(Path(str(root) + ".code.tar")) == (root / "CODE_SHA256").read_text().strip()
    )
    for patch in ({"LIVE": "903"}, {"QUEUE_STATUS": "1"}):
        p = subprocess.run(
            command + ["--resume", str(root)],
            env=dict(env, **patch),
            text=True,
            capture_output=True,
        )
        assert p.returncode != 0
        assert len(calls.read_text().splitlines()) == 4
    digest = sha(root / "MANIFEST.json")
    finish(root / "parent", [], digest)
    write(root / "audit/DECISION.json", dict(safe_to_optimize=True))
    finish(root / "audit", [root / "audit/DECISION.json"], digest)
    p = subprocess.run(
        command + ["--resume", str(root)], env=env, text=True, capture_output=True
    )
    assert p.returncode == 0, p.stderr
    assert len(calls.read_text().splitlines()) == 6
    assert "--job-name=feniks_recover_train" in calls.read_text().splitlines()[4]
    previous = tmp_path / "completed_parent"
    rec.initialize(*completed_sources, previous, config(tmp_path))
    rec.recover_parent(previous)
    write(previous / "audit/DECISION.json", dict(safe_to_optimize=False))
    finish(
        previous / "audit",
        [previous / "audit/DECISION.json"],
        sha(previous / "MANIFEST.json"),
    )
    target = tmp_path / "reaudited"
    p = subprocess.run(
        command + ["--reaudit", str(previous), str(target), str(config(tmp_path))],
        env=env,
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr
    lines = calls.read_text().splitlines()[6:]
    assert len(lines) == 3  # Audit -> train -> report, no repeated parent job.
    assert "--job-name=feniks_recover_audit" in lines[0] and "--gres" not in lines[0]
    assert "--dependency=afterok:907" in lines[1]
    assert "--dependency=afterany:907:908" in lines[2]
    gpu_config = config(tmp_path)
    cfg = yaml.safe_load(gpu_config.read_text())
    cfg["audit"]["backend"] = "gpu"
    gpu_config.write_text(yaml.safe_dump(cfg))
    p = subprocess.run(
        command
        + ["--reaudit", str(previous), str(tmp_path / "gpu_audit"), str(gpu_config)],
        env=env,
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr
    lines = calls.read_text().splitlines()[-3:]
    assert "--job-name=feniks_recover_audit" in lines[0]
    assert "--gres=gpu:1" in lines[0] and "--constraint=h100" in lines[0]
    assert "--dependency=afterok:910" in lines[1]
    assert "--dependency=afterany:910:911" in lines[2]
    assert "--gres" not in lines[2] and "--mem" not in "\n".join(lines)
