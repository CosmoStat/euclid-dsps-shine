"""Bounded end-to-end contract smoke with mocked photometry, not DSPS validation."""

import copy
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
from euclid_dsps.amortized.reference_redesign import qualification
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


@pytest.mark.parametrize("explore", [False, True])
def test_full_guarded_blind_population_path_resume_and_no_truth_weights(
    tmp_path, monkeypatch, capsys, explore
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
    if explore:
        # Certificate fixture tests orchestration, not capacity on mock photometry.
        make_cdf_failure(root)
        original.update({p: sha(p) for p in root.rglob("*") if p.is_file()})
        new_root = tmp_path / "exploratory"
        pipeline.initialize_exploratory(root, new_root)
        root = new_root
        assert not read(root / "qualification/FINAL.json")["passed"]
        with pytest.raises(SystemExit):
            pipeline.qualify(root)  # Never relabel the original qualification PASS.
        capsys.readouterr()
        pipeline.schedule(root)
        assert "NEED_QUALIFICATION=0" in capsys.readouterr().out
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
    final = read(root / "report/FINAL.json")
    assert not final["ready_for_production"]
    assert final["strict_qualification_passed"] is (not explore)
    assert final["exploratory_continuation"] is explore
    assert not final["capacity_control"]["used_for_population_fitting"]
    assert final["capacity_control"]["same_reserved_bank"]
    comparison = pd.read_csv(root / "report/observable_predictive_comparison.csv")
    assert set(comparison.model) == {"learned_parent", "truth_assisted_capacity"}
    assert not (root / "posterior").exists()
    times = {
        p: p.stat().st_mtime_ns
        for p in root.rglob("*")
        if p.is_file() and p.name != "selected_support.json"
    }
    if not explore:
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
    assert ("CDF FAIL retained" in watched.stdout) is explore
    capsys.readouterr()
    pipeline.schedule(root)
    assert "NEED_POPULATION=0" in capsys.readouterr().out
    (source / "local_256/model/coordinates.json").write_text("{}")
    with pytest.raises(ValueError, match="Immutable source changed"):
        pipeline.schedule(root)


def cdf_certificate():
    contracts = dict(
        maximum_cdf_error=0.03,
        maximum_physical_sw=0.035,
        maximum_physical_marginal_w1=0.1,
        sampling_multiplier=2,
    )
    baseline = dict(cdf_max=0.0162, physical_sw=0.014303)
    row = dict(
        objective="physical_cdf",
        validation_cdf_max=0.0338331456,
        physical_sw=0.0276976,
        physical_marginal_max=0.047019,
        finite_linear_mass_moment=True,
        log10_mean_mass=9.28237,
    )
    row["gates"], row["limits"] = qualification(row, baseline, contracts)
    row["passed"] = False
    return dict(passed=False, results=[row], baseline=baseline), dict(
        qualification_contracts=contracts
    )


def make_cdf_failure(root):
    record, config = cdf_certificate()
    m = read(root / "MANIFEST.json")
    m["settings"].update(config)
    (root / "experiment.yaml").write_text(yaml.safe_dump(m["settings"]))
    m["frozen_files"] = {"experiment.yaml": sha(root / "experiment.yaml")}
    write(root / "MANIFEST.json", m)
    for stage in ("reference", "qualification"):
        saved = read(root / stage / "FINAL.json")
        saved["contract"] = sha(root / "MANIFEST.json")
        if stage == "qualification":
            saved.update(record)
        write(root / stage / "FINAL.json", saved)


def test_small_cdf_excess_is_execution_admission_not_scientific_pass():
    record, cfg = cdf_certificate()
    before = copy.deepcopy(record)
    admission = pipeline.exploratory_cdf_admission(record, cfg)
    assert admission["cdf_excess"] == pytest.approx(0.0014331456)
    assert not admission["strict_qualification_passed"]
    assert not admission["production_promotion"] and record == before


@pytest.mark.parametrize(
    "change",
    [
        "sw",
        "mass",
        "marginal",
        "large_cdf",
        "loose_baseline",
        "nan",
        "stale_gate",
        "already_passed",
    ],
)
def test_exploratory_admission_rejects_other_failures_and_invalid_certificates(change):
    record, cfg = cdf_certificate()
    row = record["results"][0]
    if change == "sw":
        row["physical_sw"] = 0.1
    if change == "mass":
        row["finite_linear_mass_moment"] = False
    if change == "marginal":
        row["physical_marginal_max"] = 0.2
    if change == "large_cdf":
        row["validation_cdf_max"] = 0.035
    if change == "loose_baseline":
        record["baseline"]["cdf_max"] = 0.1
        row["validation_cdf_max"] = 0.201
    if change == "nan":
        row["physical_sw"] = float("nan")
    row["gates"], row["limits"] = qualification(
        row, record["baseline"], cfg["qualification_contracts"]
    )
    if change == "stale_gate":
        row["gates"]["validation_cdf"] = True
    if change == "already_passed":
        record["passed"] = True
    with pytest.raises(ValueError):
        pipeline.exploratory_cdf_admission(record, cfg)


def test_import_requires_completed_intact_source_and_new_root(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch)
    new = tmp_path / "new"
    with pytest.raises(ValueError, match="Completed qualification"):
        pipeline.initialize_exploratory(root, new)
    assert not new.exists()
    pipeline.qualify(root)
    make_cdf_failure(root)
    with pytest.raises(ValueError, match="new root"):
        pipeline.initialize_exploratory(root, root)
    with pytest.raises(ValueError, match="new root"):
        pipeline.initialize_exploratory(root, root / "nested")
    (root / "reference/coordinates.json").write_text("{}")
    with pytest.raises(ValueError, match="Corrupt completed artifact"):
        pipeline.initialize_exploratory(root, new)
    assert not new.exists()


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


@pytest.mark.parametrize("explore", [False, True])
def test_submission_guarded_dag_and_missing_only_resume(tmp_path, explore):
    repo = Path(__file__).resolve().parents[1]
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python").write_text("""#!/usr/bin/env python3
import os,sys
from pathlib import Path
mode=sys.argv[3]; root=Path(sys.argv[sys.argv.index('--root')+1])
if mode in ('init', 'init-exploratory'): (root/'logs').mkdir(parents=True)
elif mode=='schedule':
    print('QUALIFY_MINUTES=30\\nBANK_MINUTES=120\\nPOPULATION_MINUTES=180\\nREPORT_MINUTES=20\\nCPU_THREADS=4\\nBANK_CONCURRENCY=4')
    print('NEED_QUALIFICATION=0\\nMISSING_BANKS=2\\nNEED_POPULATION=1\\nNEED_REPORT=1' if os.environ.get('RESUMING') else 'NEED_QUALIFICATION='+('0' if os.environ.get('EXPLORING') else '1')+'\\nMISSING_BANKS=0,1,2,3\\nNEED_POPULATION=1\\nNEED_REPORT=1')
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
        EXPLORING="1" if explore else "",
    )
    command = ["bash", "scripts/submit_feniks_reference_to_parent.sh"]
    subprocess.run(
        command
        + (
            ["--explore-cdf", str(source), str(root)]
            if explore
            else [str(source), str(root), "cfg.yaml"]
        ),
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
    )
    calls = [json.loads(x) for x in (tmp_path / "calls").read_text().splitlines()]
    first = 0 if explore else 1
    assert len(calls) == 4 - int(explore)
    assert "--array=0,1,2,3%4" in calls[first]
    if not explore:
        assert "--dependency=afterok:800" in calls[1]
        assert "--account=jrx@cpu" in calls[0]
    else:
        assert not any("--dependency" in a for a in calls[0])
        assert not any("FENIKS_RTP_MODE=qualify" in a for c in calls for a in c)
    assert f"--dependency=afterok:{800 + first}" in calls[first + 1]
    dependencies = ":".join(str(800 + i) for i in range(first + 2))
    assert f"--dependency=afterany:{dependencies}" in calls[-1]
    assert "--account=jrx@cpu" in calls[-1]
    assert "--kill-on-invalid-dep=yes" in calls[first + 1]
    assert all("--gres=gpu:1" in c for c in calls[first : first + 2])
    assert not any("--mem" in arg for call in calls for arg in call)
    subprocess.run(
        command + ["--resume", str(root)],
        cwd=checkout,
        env=dict(env, RESUMING="1"),
        check=True,
        capture_output=True,
    )
    calls = [json.loads(x) for x in (tmp_path / "calls").read_text().splitlines()]
    assert len(calls) == 7 - int(explore) and "--array=2%4" in calls[4 - int(explore)]
    failed = subprocess.run(
        command + ["--resume", str(root)],
        cwd=checkout,
        env=dict(env, ACTIVE_TEST=f"{804 - int(explore)}_2"),
        capture_output=True,
    )
    assert failed.returncode and b"still active" in failed.stderr
