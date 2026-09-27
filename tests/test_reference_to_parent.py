"""Bounded end-to-end contract smoke with mocked photometry, not DSPS validation."""

import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_feniks_coherent_inference import fixture as coherent_fixture

from euclid_dsps.amortized.coherent_coordinates import fit_coordinates, to_theta, to_x
from scripts import feniks_coherent_inference as ci
from scripts import feniks_reference_to_parent as pipeline
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import finish


def fixture(tmp_path, monkeypatch, *, qualify=True):
    old, data, _ = coherent_fixture(tmp_path, monkeypatch, positive_dust=True)
    ci.reference(old)
    redesign = tmp_path / "redesign"
    redesign.mkdir()
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_reference_redesign.yaml").read_text()
    )
    cfg["evaluation"].update(random_directions=2, thresholds=7)
    # Permissive gates ONLY in the end-to-end fixture; real criteria unchanged.
    if qualify:
        cfg["contracts"].update(
            maximum_cdf_error=1,
            maximum_physical_sw=100,
            maximum_physical_marginal_w1=100,
        )
    else:
        cfg["contracts"].update(
            maximum_cdf_error=1e-15,
            maximum_physical_sw=1e-15,
            maximum_physical_marginal_w1=1e-15,
            sampling_multiplier=1e-15,
        )
    (redesign / "experiment.yaml").write_text(yaml.safe_dump(cfg))
    write(
        redesign / "MANIFEST.json",
        dict(
            source=str(old),
            settings=cfg,
            frozen_files={"experiment.yaml": sha(redesign / "experiment.yaml")},
            source_files={str(old / "MANIFEST.json"): sha(old / "MANIFEST.json")},
        ),
    )
    digest = sha(redesign / "MANIFEST.json")
    model = redesign / "local_256/model"
    model.mkdir(parents=True)
    with np.load(old / "reference/basis.npz") as f:
        basis = dict(f)
    theta = np.asarray(
        to_theta(basis["anchors"], read(old / "reference/coordinates.json"))
    )
    spec = fit_coordinates(
        theta,
        ci.NAMES,
        -np.ones(15),
        np.ones(15),
        positive=["dust_av"],
        affine=["log10_stellar_mass"],
    )
    basis["anchors"] = np.asarray(to_x(theta, spec))
    np.savez(model / "basis.npz", **basis)
    write(model / "coordinates.json", spec)
    finish(model, [model / "basis.npz", model / "coordinates.json"], digest)
    for name in ("report", "replay"):
        (redesign / name).mkdir()
        finish(redesign / name, [], digest, replay_identity_pass=True)
    new = yaml.safe_load(
        Path("configs/experiments/feniks_reference_to_parent.yaml").read_text()
    )
    new["capacity"].update(
        draws_per_component=128, directions=2, grid_size=9, evaluation_draws=400
    )
    new["bank"].update(shards=2, rows_per_shard=2048, checkpoint_rows=1024)
    new["classifier"]["epochs"] = 2
    new["evaluation"]["population_draws"] = 400
    config = tmp_path / "next.yaml"
    config.write_text(yaml.safe_dump(new))
    root = tmp_path / "next"
    pipeline.initialize(redesign, root, config)
    return root, old, data, redesign


def test_full_guarded_blind_population_path_resume_and_no_truth_weights(
    tmp_path, monkeypatch, capsys
):
    root, old, data, source = fixture(tmp_path, monkeypatch)
    cfg = read(root / "MANIFEST.json")["settings"]
    assert "posterior" not in cfg and "kernel_bandwidth" not in cfg["reference"]
    assert "oracle_minutes" not in cfg["resources"]
    original = {
        p: sha(p) for d in (old, data, source) for p in d.rglob("*") if p.is_file()
    }
    real_read = pd.read_parquet

    def development_only(path, *args, **kwargs):
        assert Path(path).name != "test.parquet"
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", development_only)
    pipeline.report(root)
    assert (root / "report/BLOCKED.json").exists()
    pipeline.qualify(root)
    assert read(root / "qualification/FINAL.json")["passed"]
    assert sha(root / "reference/basis.npz") == sha(
        source / "local_256/model/basis.npz"
    )
    assert not (root / "population/parent.json").exists()
    for i in range(2):
        pipeline.bank(root, i)

    def blind_read(path, *args, **kwargs):
        assert "/dataset/selected_r29/" in str(path)
        assert kwargs["columns"] == ci.observation_columns(["lsst_r"])
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", blind_read)
    pipeline.population(root)
    parent = read(root / "population/parent.json")
    assert not parent["target_truth_used"] and not parent["population_uses_q"]
    np.testing.assert_allclose(np.sum(parent["u"]), 1)
    np.testing.assert_allclose(np.sum(parent["v"]), 1)
    np.testing.assert_allclose(
        np.array(parent["u"]) * parent["alpha"] / parent["alpha_parent"], parent["v"]
    )
    monkeypatch.setattr(pd, "read_parquet", development_only)
    pipeline.report(root)
    assert not read(root / "report/FINAL.json")["ready_for_production"]
    assert not (root / "posterior").exists()
    times = {
        p: p.stat().st_mtime_ns
        for p in root.rglob("*")
        if p.is_file() and p.name != "selected_support.json"
    }
    pipeline.qualify(root)
    pipeline.population(root)
    pipeline.report(root)
    for i in range(2):
        pipeline.bank(root, i)
    assert times == {p: p.stat().st_mtime_ns for p in times}
    assert original == {p: sha(p) for p in original}
    watched = subprocess.run(
        ["bash", "scripts/watch_feniks_reference_to_parent.sh", str(root), "--once"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "DONE" in watched.stdout and "artifacts" not in watched.stdout
    capsys.readouterr()
    pipeline.schedule(root)
    assert "NEED_POPULATION=0" in capsys.readouterr().out
    (source / "local_256/model/coordinates.json").write_text("{}")
    with pytest.raises(ValueError, match="Immutable source changed"):
        pipeline.schedule(root)


def test_failed_qualification_blocks_expensive_work(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch, qualify=False)
    with pytest.raises(SystemExit):
        pipeline.qualify(root)
    for fn in (lambda r: pipeline.bank(r, 0), pipeline.population, pipeline.schedule):
        with pytest.raises(ValueError):
            fn(root)
    pipeline.report(root)
    assert "qualification_failed" in read(root / "report/BLOCKED.json")["missing"]
    assert not list((root / "banks").iterdir())


def test_no_posterior_or_diagnostic_weight_feedback():
    code = inspect.getsource(pipeline.population)
    assert "diagnostic_weights" not in code and "posterior" not in code
    source = inspect.getsource(ci.population)
    assert "NAMES" not in source and "sample_basis" not in source


def test_submission_guarded_dag_and_missing_only_resume(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python").write_text("""#!/usr/bin/env python3
import os,sys
from pathlib import Path
mode=sys.argv[3]; root=Path(sys.argv[sys.argv.index('--root')+1])
if mode=='init': (root/'logs').mkdir(parents=True)
elif mode=='schedule':
    print('QUALIFY_MINUTES=30\\nBANK_MINUTES=120\\nPOPULATION_MINUTES=180\\nREPORT_MINUTES=20\\nCPU_THREADS=4\\nBANK_CONCURRENCY=4')
    print('NEED_QUALIFICATION=0\\nMISSING_BANKS=2\\nNEED_POPULATION=1\\nNEED_REPORT=1' if os.environ.get('RESUMING') else 'NEED_QUALIFICATION=1\\nMISSING_BANKS=0,1,2,3\\nNEED_POPULATION=1\\nNEED_REPORT=1')
""")
    (fake / "sbatch").write_text("""#!/usr/bin/env python3
import os,sys,json
from pathlib import Path
p=Path(os.environ['CALL_LOG']); lines=p.read_text().splitlines() if p.exists() else []
p.write_text('\\n'.join(lines+[json.dumps(sys.argv[1:])])+'\\n'); print(800+len(lines))
""")
    (fake / "squeue").write_text('#!/bin/bash\nprintf "%s" "${ACTIVE_TEST:-}"\n')
    for p in fake.iterdir():
        p.chmod(0o755)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for directory in ("scripts", "euclid_dsps", "configs", "Data"):
        (checkout / directory).mkdir()
    for name in (
        "submit_feniks_reference_to_parent.sh",
        "feniks_reference_to_parent.slurm",
    ):
        shutil.copy2(repo / "scripts" / name, checkout / "scripts" / name)
    (checkout / "pyproject.toml").write_text("")
    (checkout / "cfg.yaml").write_text("{}")
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "run"
    env = dict(
        os.environ,
        PATH=f"{fake}:" + os.environ["PATH"],
        SCRATCH=str(tmp_path / "scratch"),
        CALL_LOG=str(tmp_path / "calls"),
    )
    command = ["bash", "scripts/submit_feniks_reference_to_parent.sh"]
    subprocess.run(
        command + [str(source), str(root), "cfg.yaml"],
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
    )
    calls = [json.loads(x) for x in (tmp_path / "calls").read_text().splitlines()]
    assert len(calls) == 4 and "--array=0,1,2,3%4" in calls[1]
    assert (
        "--dependency=afterok:800" in calls[1]
        and "--dependency=afterok:801" in calls[2]
    )
    assert "--dependency=afterany:800:801:802" in calls[3]
    assert "--account=jrx@cpu" in calls[0] and "--account=jrx@cpu" in calls[3]
    for call in calls[1:3]:
        assert "--kill-on-invalid-dep=yes" in call and "--gres=gpu:1" in call
    assert not any("--mem" in arg for call in calls for arg in call)
    subprocess.run(
        command + ["--resume", str(root)],
        cwd=checkout,
        env=dict(env, RESUMING="1"),
        check=True,
        capture_output=True,
    )
    calls = [json.loads(x) for x in (tmp_path / "calls").read_text().splitlines()]
    assert len(calls) == 7 and "--array=2%4" in calls[4]
    failed = subprocess.run(
        command + ["--resume", str(root)],
        cwd=checkout,
        env=dict(env, ACTIVE_TEST="804_2"),
        capture_output=True,
    )
    assert failed.returncode and b"still active" in failed.stderr
