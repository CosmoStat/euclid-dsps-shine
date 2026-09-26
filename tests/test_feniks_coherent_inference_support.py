"""Regression for native Av>6, old coordinate compatibility and frozen-run repair."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_feniks_coherent_inference import fixture

from euclid_dsps.amortized.coherent_coordinates import (
    fit_coordinates,
    log_abs_det_dtheta_dx,
    to_theta,
    to_x,
    validate_theta,
)
from scripts import feniks_coherent_inference as pipeline
from scripts.feniks_avi_experiments import read, sha, write
from scripts.repair_feniks_coherent_inference import check, repair


def test_positive_dust_roundtrip_jacobian_and_normalized_measure():
    import jax
    import jax.numpy as jnp
    from scipy.integrate import quad
    from scipy.stats import norm

    theta = np.random.default_rng(71).uniform(-0.7, 0.7, (128, 15))
    theta[:, 3] = np.geomspace(1e-10, 1e3, len(theta))
    original = theta.copy()
    spec = fit_coordinates(
        theta, pipeline.NAMES, [-1] * 15, [1] * 15, positive=["dust_av"]
    )
    assert spec["version"] == "coherent_coordinates_v2"
    assert spec["positive"][3] and not spec["bounded"][3]
    x = to_x(theta, spec)
    np.testing.assert_allclose(to_theta(x, spec), theta, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(original, theta)
    det = jax.vmap(
        lambda a: jnp.linalg.slogdet(jax.jacfwd(lambda b: to_theta(b, spec))(a))[1]
    )(x[:4])
    np.testing.assert_allclose(log_abs_det_dtheta_dx(x[:4], spec), det, atol=1e-10)
    center, scale = spec["center"][3], spec["scale"][3]
    # A standard normal x pushes forward to a normalized lognormal Av on (0,inf).
    # Split the physical integral across scales; one adaptive (0,inf) call
    # misses the mass near zero for this intentionally very broad stress case.
    edges = np.exp(center + scale * np.linspace(-10, 10, 161))
    integral = sum(
        quad(lambda t: norm.pdf((np.log(t) - center) / scale) / (scale * t), lo, hi)[0]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    )
    assert integral == pytest.approx(1, abs=2e-7)
    for invalid in (0.0, -0.1):
        bad = theta[:1].copy()
        bad[:, 3] = invalid
        with pytest.raises(ValueError, match="POSITIVE support: dust_av"):
            validate_theta(bad, spec)
    assert not spec["clipping"] and not spec["dequantization"]


def test_v1_unchanged_and_open_support_error_reports_range():
    theta = np.random.default_rng(4).uniform(-0.7, 0.7, (128, 15))
    spec = fit_coordinates(theta, pipeline.NAMES, [-1] * 15, [1] * 15)
    assert spec["version"] == "coherent_coordinates_v1" and "positive" not in spec
    np.testing.assert_allclose(to_theta(to_x(theta, spec), spec), theta, atol=1e-12)
    theta[0, 3] = 7.59
    with pytest.raises(ValueError, match=r"OPEN support: dust_av; range=.*7.59"):
        validate_theta(theta, spec)


def test_reference_accepts_dust_tail_and_records_support(tmp_path, monkeypatch):
    root, source, _ = fixture(tmp_path, monkeypatch, positive_dust=True)
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    pipeline.reference(root)
    spec = read(root / "reference/coordinates.json")
    assert (
        spec["positive"][3]
        and spec["fitted_on"] == "independent_unweighted_native_reference"
    )
    support = pd.read_csv(root / "reference/support.csv").set_index("parameter")
    assert support.loc["dust_av", "maximum"] > 6
    assert support.loc["dust_av", "transform"] == "log"
    with np.load(root / "reference/basis.npz") as data:
        assert len(data["anchors"]) == 128
    pipeline.bank(root, 0)
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}


def legacy_run(tmp_path, monkeypatch):
    root, source, _ = fixture(tmp_path, monkeypatch, positive_dust=True)
    m = read(root / "MANIFEST.json")
    m["settings"]["reference"].pop("positive")
    m["settings"]["reference"]["bounds"]["dust_av"] = [1e-6, 6.0]
    (root / "experiment.yaml").write_text(yaml.safe_dump(m["settings"]))
    m["frozen_files"]["experiment.yaml"] = sha(root / "experiment.yaml")
    write(root / "MANIFEST.json", m)
    (root / "CODE_DIR").write_text("/frozen/old/code\n")
    (root / "CODE_SHA256").write_text("old-sha\n")
    (root / "JOBS.env").write_text("export ALL_JOBS=217287,217289\n")
    (root / "report").mkdir()
    write(root / "report/BLOCKED.json", dict(status="BLOCKED"))
    return root, source


def test_repair_archives_originals_changes_only_dust_and_resumes(tmp_path, monkeypatch):
    root, source = legacy_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="OPEN support: dust_av"):
        pipeline.reference(root)
    original = read(root / "MANIFEST.json")
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    archive = tmp_path / "new.code.tar"
    archive.write_bytes(b"test snapshot")
    repair(root, Path.cwd(), archive)
    receipt = read(root / "REFERENCE_REPAIR.json")
    backup = Path(receipt["backup"])
    assert receipt["status"] == "COMPLETE"
    assert read(backup / "MANIFEST.json") == original
    assert (backup / "reference/support.csv").is_file()
    assert (backup / "report/BLOCKED.json").is_file()
    m, cfg, _ = pipeline.settings(root)
    expected = original["settings"]
    expected["reference"]["positive"] = ["dust_av"]
    del expected["reference"]["bounds"]["dust_av"]
    assert cfg == expected and m["source_files"] == original["source_files"]
    assert (root / "CODE_SHA256").read_text().strip() == sha(archive)
    pipeline.reference(root)
    assert (root / "reference/FINAL.json").is_file()
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        check(root)


@pytest.mark.parametrize(
    "path",
    [
        "reference/FINAL.json",
        "banks/shard_000/bank.npz",
        "oracle/RESUME.json",
        "population/best.eqx",
        "posterior/training.jsonl",
        "report/FINAL.json",
    ],
)
def test_repair_refuses_any_reusable_computation(tmp_path, monkeypatch, path):
    root, _ = legacy_run(tmp_path, monkeypatch)
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("existing")
    with pytest.raises(ValueError, match="Cannot"):
        check(root)
    assert not (root / "recovery").exists()


@pytest.mark.parametrize("output,code", [("217287", 0), ("217289_[0-3]", 0), ("", 1)])
def test_repair_shell_refuses_active_jobs_and_scheduler_failure(
    tmp_path, monkeypatch, output, code
):
    root, _ = legacy_run(tmp_path, monkeypatch)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, content in {
        "sbatch": "exit 99",
        "squeue": f"printf '%s\\n' '{output}'\nexit {code}",
    }.items():
        path = bindir / name
        path.write_text(f"#!/bin/bash\n{content}\n")
        path.chmod(0o755)
    env = dict(
        os.environ, PATH=f"{bindir}:{Path(sys.executable).parent}:{os.environ['PATH']}"
    )
    result = subprocess.run(
        ["bash", "scripts/repair_feniks_coherent_inference.sh", str(root)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not (root / "recovery").exists()
    assert not (root / "REFERENCE_REPAIR.json").exists()


def test_interrupted_repair_blocks_normal_resume(tmp_path, monkeypatch):
    root, _ = legacy_run(tmp_path, monkeypatch)
    write(root / "REFERENCE_REPAIR.json", dict(status="PREPARING"))
    with pytest.raises(ValueError, match="repair interrupted"):
        pipeline.settings(root)


def test_repair_shell_freezes_new_code_and_resubmits_same_root(tmp_path, monkeypatch):
    root, source = legacy_run(tmp_path, monkeypatch)
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    bindir = tmp_path / "bin"
    bindir.mkdir()
    counter, calls = tmp_path / "counter", tmp_path / "calls"
    counter.write_text("800")
    sbatch = bindir / "sbatch"
    sbatch.write_text(
        f'#!/bin/bash\nn=$(<{counter}); n=$((n+1)); echo "$n" > {counter}\nprintf "%s\\n" "$*" >> {calls}\necho "$n"\n'
    )
    sbatch.chmod(0o755)
    squeue = bindir / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n")
    squeue.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bindir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
        SCRATCH=str(tmp_path / "scratch"),
    )
    result = subprocess.run(
        ["bash", "scripts/repair_feniks_coherent_inference.sh", str(root)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    receipt = read(root / "REFERENCE_REPAIR.json")
    code = Path((root / "CODE_DIR").read_text().strip())
    assert code != Path.cwd() and code.is_dir()
    assert (
        code / "euclid_dsps/amortized/coherent_coordinates.py"
    ).read_bytes() == Path("euclid_dsps/amortized/coherent_coordinates.py").read_bytes()
    assert receipt["new_code_sha256"] == sha(Path(receipt["new_code_archive"]))
    assert len(calls.read_text().splitlines()) == 6
    assert "afterok:801" in calls.read_text()
    assert str(root) in result.stdout
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
