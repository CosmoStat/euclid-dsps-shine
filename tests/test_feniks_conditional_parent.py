"""Real tiny classifier/solver tests; DSPS is not invoked by this pipeline."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import feniks_coherent_inference as ci
from scripts import feniks_conditional_parent as cp
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish


def fixture_source(tmp_path):
    from test_feniks_parent_to_posterior import source_fixture

    from euclid_dsps.amortized.coherent_coordinates import to_theta
    from euclid_dsps.amortized.native_reference import sample_basis

    source, dataset = source_fixture(tmp_path)
    manifest = read(source / "MANIFEST.json")
    cfg = manifest["settings"]
    cfg.update(
        bank=dict(shards=2, rows_per_shard=4096, checkpoint_rows=2048),
        classifier=dict(
            width=16,
            depth=1,
            epochs=2,
            batch_size=128,
            learning_rate=0.002,
            validation_limit=128,
            fixed_validation=True,
        ),
        population=dict(
            min_alpha=0.001,
            min_selected=1,
            weak_parent_mass=0.05,
            penalties=[0.01, 0.1],
        ),
        stopping=dict(
            block_epochs=1, minimum_epochs=2, patience_blocks=1, minimum_gain=0.001
        ),
        evaluation=dict(population_draws=512),
        parent_contracts=dict(
            maximum_physical_sw=0.05,
            maximum_physical_marginal_w1=0.1,
            maximum_alpha_error=0.03,
            maximum_observable_ks=0.05,
            maximum_tail_probability=0.005,
            maximum_ratio_moment_error=0.03,
            minimum_classifier_gain=0.05,
        ),
    )
    (source / "experiment.yaml").write_text(yaml.safe_dump(cfg))
    manifest["frozen_files"]["experiment.yaml"] = sha(source / "experiment.yaml")
    write(source / "MANIFEST.json", manifest)
    digest = sha(source / "MANIFEST.json")
    for stage in ("reference", "population", "qualification", "report"):
        record = read(source / stage / "FINAL.json")
        record["contract"] = digest
        write(source / stage / "FINAL.json", record)
    basis, spec, _ = ci.require_reference(source, digest)
    for task in range(2):
        shard = source / "banks" / f"shard_{task:03d}"
        shard.mkdir(parents=True)
        files = []
        for start in (0, 2048):
            out = shard / f"block_{start:07d}"
            out.mkdir()
            seed = cfg["seed"] + task * 10000000 + start + 1000
            labels = np.arange(2048) % 2
            x = sample_basis(basis, labels, seed)
            theta = np.asarray(to_theta(x, spec))
            rng = np.random.default_rng(seed + 10)
            selected = rng.random(len(x)) < (0.6 + 0.15 * (x[:, 5] > 0))
            roles = rng.choice(5, len(x), p=[0.7, 0.1, 0.05, 0.05, 0.1])
            np.savez(
                out / "bank.npz",
                x=x,
                theta=theta,
                component=labels,
                selected=selected,
                role=roles,
                features=x[:, [0, 5, 6]],
                flux=np.exp(x[:, :1]) * 1e-30,
            )
            finish(out, [out / "bank.npz"], digest)
            files.append(out / "FINAL.json")
        finish(shard, files, digest)
    frame = pd.read_parquet(dataset / "dataset/selected_r29/test.parquet")
    for split in ("train", "validation"):
        frame.to_parquet(dataset / f"dataset/selected_r29/{split}.parquet")
    (dataset / "dataset/parent").mkdir()
    frame.to_parquet(dataset / "dataset/parent/validation.parquet")
    return source, dataset


def config(tmp_path):
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_conditional_parent.yaml").read_text()
    )
    cfg["classifier"]["epochs"] = 2
    p = tmp_path / "conditional.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_relabel_exact_replay_no_dsps_and_missing_only_resume(tmp_path, monkeypatch):
    source, _ = fixture_source(tmp_path)
    root = tmp_path / "new"
    cp.initialize(source, root, config(tmp_path))
    monkeypatch.setattr(ci, "photometer", lambda *_: pytest.fail("No DSPS permitted"))
    cp.prepare(root)
    save = np.savez
    calls = []

    def interrupted(path, **data):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("interrupted relabel")
        return save(path, **data)

    monkeypatch.setattr(np, "savez", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        cp.relabel(root, 0)
    first = root / "banks/shard_000/block_0000000/bank.npz"
    saved_hash = sha(first)
    monkeypatch.setattr(np, "savez", save)
    cp.relabel(root, 0)
    cp.relabel(root, 1)
    assert sha(first) == saved_hash
    for path in (root / "banks").glob("shard_*/block_*/bank.npz"):
        with np.load(path) as new, np.load(source / path.relative_to(root)) as old:
            for key in ("x", "theta", "features", "flux", "role", "selected"):
                np.testing.assert_array_equal(new[key], old[key])
            np.testing.assert_array_equal(new["component"] // 2, old["component"])
        assert read(path.parent / "FINAL.json")["exact_anchor_replay"]
    monkeypatch.setattr(
        cp,
        "sample_basis",
        lambda *_args, **_kwargs: pytest.fail("Completed replay reran"),
    )
    cp.relabel(root, 0)


def test_wrong_replay_fails_closed(tmp_path, monkeypatch):
    source, _ = fixture_source(tmp_path)
    root = tmp_path / "new"
    cp.initialize(source, root, config(tmp_path))
    cp.prepare(root)
    original = cp.sample_basis

    def changed(*args, **kwargs):
        x, a, n = original(*args, **kwargs)
        x[0, 0] += 1e-14
        return x, a, n

    monkeypatch.setattr(cp, "sample_basis", changed)
    with pytest.raises(ValueError, match="Anchor replay mismatch"):
        cp.relabel(root, 0)
    assert not list((root / "banks").rglob("bank.npz"))


def test_actual_classifier_fit_blind_inputs_tied_control_report(tmp_path, monkeypatch):
    source, dataset = fixture_source(tmp_path)
    before = {
        p: sha(p)
        for folder in (source, dataset)
        for p in folder.rglob("*")
        if p.is_file()
    }
    root = tmp_path / "new"
    cp.initialize(source, root, config(tmp_path))
    cp.report(root)
    assert read(root / "report/BLOCKED.json")["missing"] == [
        "reference",
        "population",
        "tied",
    ]
    real_read = pd.read_parquet

    def photometry_only(path, **kwargs):
        assert kwargs["columns"] == ["flux_lsst_r", "fluxerr_lsst_r", "mask_lsst_r"]
        return real_read(path, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", photometry_only)
    monkeypatch.setattr(ci, "photometer", lambda *_: pytest.fail("No DSPS permitted"))
    monkeypatch.setattr(
        ci, "posterior", lambda *_: pytest.fail("No individual q permitted")
    )
    cp.prepare(root)
    cp.relabel(root, 0)
    cp.relabel(root, 1)
    cp.population(root)
    for stage in ("population", "tied"):
        parent = read(root / stage / "parent.json")
        u, v, a = [np.asarray(parent[key]) for key in ("u", "v", "alpha")]
        np.testing.assert_allclose(v, u * a / (u @ a), atol=1e-10)
        assert u.sum() == pytest.approx(1) and v.sum() == pytest.approx(1)
        assert pd.read_csv(root / stage / "regularization.csv").kkt_gap.max() <= 2e-6
    hist = pd.read_json(root / "population/training.jsonl", lines=True)
    assert np.isfinite(hist.validation_nll).all()
    assert hist.train_nll.iloc[-1] < hist.train_nll.iloc[0]
    monkeypatch.setattr(
        ci, "bounded_fit", lambda *_: pytest.fail("Completed classifier retrained")
    )
    cp.population(root)
    monkeypatch.setattr(pd, "read_parquet", real_read)
    cp.report(root)
    decision = read(root / "report/DECISION.json")
    assert (
        not decision["population_uses_target_truth"]
        and not decision["ready_for_production"]
    )
    assert complete(root / "report", sha(root / "MANIFEST.json"))
    assert len(pd.read_csv(root / "report/marginals.csv")) == 90
    assert (root / "report/parent_physical.png").stat().st_size > 10000
    assert before == {p: sha(p) for p in before}


def slurm_environment(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls, counter = tmp_path / "calls", tmp_path / "counter"
    counter.write_text("700")
    for name, script in {
        "sbatch": f'#!/bin/bash\nn=$(<{counter}); n=$((n+1)); echo "$n" > {counter}\nprintf "%s\\n" "$*" >> {calls}\necho "$n"\n',
        "squeue": '#!/bin/bash\necho "${LIVE:-}"\nexit "${QUEUE_STATUS:-0}"\n',
        "scancel": "#!/bin/bash\nexit 99\n",
    }.items():
        p = bindir / name
        p.write_text(script)
        p.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{Path(sys.executable).parent}:" + os.environ["PATH"],
        SCRATCH=str(tmp_path / "scratch"),
    )
    return env, calls


def test_slurm_frozen_code_dag_resume_no_mem_or_cancel(tmp_path):
    source, _ = fixture_source(tmp_path)
    env, calls = slurm_environment(tmp_path)
    root = tmp_path / "avi_conditional_parent_20260927_230000"
    command = ["bash", "scripts/submit_feniks_conditional_parent.sh"]
    result = subprocess.run(
        command + [str(source), str(root), str(config(tmp_path))],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = calls.read_text().splitlines()
    assert len(lines) == 4
    assert "--gres" not in lines[0] and "--gres" not in lines[1]
    assert "--array=0,1%4" in lines[1] and "afterok:701" in lines[1]
    assert "--gres=gpu:1" in lines[2] and "afterok:702" in lines[2]
    assert "afterany:701:702:703" in lines[3] and "--gres" not in lines[3]
    assert "--mem" not in calls.read_text()
    assert (
        sha(Path(str(root) + ".code.tar")) == (root / "CODE_SHA256").read_text().strip()
    )
    for patch in ({"LIVE": "702_1"}, {"QUEUE_STATUS": "1"}):
        result = subprocess.run(
            command + ["--resume", str(root)],
            env=dict(env, **patch),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and len(calls.read_text().splitlines()) == 4
    cp.prepare(root)
    cp.relabel(root, 0)
    result = subprocess.run(
        command + ["--resume", str(root)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 7
    assert "--array=1%4" in calls.read_text().splitlines()[4]
    watched = subprocess.run(
        ["bash", "scripts/watch_feniks_conditional_parent.sh", str(root), "--once"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert watched.returncode == 0 and "Recycled bank" in watched.stdout


def test_overnight_submits_both_and_reuses_active_branches(tmp_path):
    source, _ = fixture_source(tmp_path)
    env, calls = slurm_environment(tmp_path)
    command = ["bash", "scripts/submit_feniks_overnight.sh"]
    night = tmp_path / "night"
    result = subprocess.run(
        command + [str(source), str(night)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert len(calls.read_text().splitlines()) == 8
    assert (night / "NIGHT.env").is_file()
    # A second invocation adopts the matching-source roots, not new simulations.
    other = tmp_path / "second-night"
    active = dict(env, LIVE="702_1\n705_2")
    result = subprocess.run(
        command + [str(source), str(other)], env=active, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("existing active jobs retained") == 2
    assert len(calls.read_text().splitlines()) == 8
    assert (night / "NIGHT.env").read_text() == (other / "NIGHT.env").read_text()
    result = subprocess.run(
        command + ["--resume", str(night)],
        env=dict(env, QUEUE_STATUS="1"),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and len(calls.read_text().splitlines()) == 8
    result = subprocess.run(
        ["bash", "scripts/watch_feniks_overnight.sh", str(night), "--once"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "BLIND SFH-CONDITIONAL" in result.stdout and "FROZEN BASELINE" in result.stdout
    )


def test_blind_source_and_config_immutability(tmp_path):
    source, _ = fixture_source(tmp_path)
    root = tmp_path / "new"
    cp.initialize(source, root, config(tmp_path))
    with (root / "experiment.yaml").open("a") as f:
        f.write("\nchanged: true\n")
    with pytest.raises(ValueError, match="configuration changed"):
        cp.prepare(root)


def test_weak_subcomponent_support_stops_before_training(tmp_path, monkeypatch):
    source, _ = fixture_source(tmp_path)
    root = tmp_path / "new"
    cp.initialize(source, root, config(tmp_path))
    cp.prepare(root)
    cp.relabel(root, 0)
    cp.relabel(root, 1)
    efficiency = cp.selection_efficiencies

    def unresolved(*args, **kwargs):
        result = efficiency(*args, **kwargs)
        result["eligible"][0] = False
        return result

    monkeypatch.setattr(cp, "selection_efficiencies", unresolved)
    monkeypatch.setattr(
        ci,
        "population",
        lambda *_: pytest.fail("Training started before support passed"),
    )
    with pytest.raises(ValueError, match="Split support insufficient"):
        cp.population(root)
    assert not read(root / "population/split_support.json")["passed"]
    assert not (root / "population/best.eqx").exists()
