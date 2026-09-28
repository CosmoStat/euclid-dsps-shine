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
    assert read(root / "audit/DECISION.json")["safe_to_optimize"]
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
