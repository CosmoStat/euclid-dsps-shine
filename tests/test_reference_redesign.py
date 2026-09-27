"""Normalization, backward compatibility, and bounded qualification contracts."""

import inspect
import json
import os
import subprocess
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from test_coherent_reference_audit import basis_fixture, make_run

from euclid_dsps.amortized.coherent_coordinates import (
    fit_coordinates,
    log_abs_det_dtheta_dx,
    to_theta,
    to_x,
)
from euclid_dsps.amortized.native_reference import log_prob, sample_basis
from euclid_dsps.amortized.reference_capacity import cdf_design, empirical_features
from euclid_dsps.amortized.reference_redesign import (
    apply_saved_noise,
    local_basis,
    mass_moment,
    qualification,
    replay_normals,
)
from euclid_dsps.synthetic_diffsky.coherent_parent import observe
from scripts import feniks_reference_redesign as pipeline
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import finish


def config():
    return yaml.safe_load(
        Path("configs/experiments/feniks_reference_redesign.yaml").read_text()
    )


def test_affine_mass_roundtrip_density_jacobian_and_legacy_unchanged():
    theta = np.random.default_rng(43).uniform(-0.7, 0.7, (1000, 15))
    theta[:, 1] += 10
    theta[:, 3] = np.exp(theta[:, 3])
    args = (theta, pipeline.NAMES, -np.ones(15), np.ones(15))
    old = fit_coordinates(*args, positive=["dust_av"])
    legacy = np.asarray(to_x(theta, old))
    np.testing.assert_array_equal(legacy, to_x(theta, dict(old, affine=[False] * 15)))
    spec = fit_coordinates(*args, positive=["dust_av"], affine=["log10_stellar_mass"])
    x = to_x(theta, spec)
    np.testing.assert_allclose(to_theta(x, spec), theta, atol=1e-12)
    jac = np.asarray(jax.jacfwd(lambda z: to_theta(z, spec))(x[0]))
    assert log_abs_det_dtheta_dx(x[0], spec) == pytest.approx(
        np.linalg.slogdet(jac)[1], abs=1e-10
    )
    basis = dict(
        anchors=np.zeros((2, 15)),
        conditional=np.array([[1, 0], [0, 1]]),
        bandwidth=np.ones((2, 15)) * 0.1,
    )
    assert mass_moment(basis, old, [0.5, 0.5])["finite_linear_mass_moment"] is False
    moment = mass_moment(basis, spec, [0.5, 0.5])
    mu = spec["location"][1] + spec["width"][1] * spec["center"][1]
    sigma = spec["width"][1] * spec["scale"][1] * 0.1
    assert moment["log10_mean_mass"] == pytest.approx(mu + 0.5 * np.log(10) * sigma**2)
    from scipy.integrate import quad
    from scipy.stats import norm

    # Physical mass density includes the affine inverse Jacobian, not just p_x.
    width = spec["width"][1] * spec["scale"][1]
    integral = quad(lambda t: norm.pdf((t - mu) / width) / width, -np.inf, np.inf)[0]
    assert integral == pytest.approx(1, abs=1e-9)
    with pytest.raises(ValueError, match="real-line"):
        fit_coordinates(*args, affine=["z_obs"])


def test_scalar_rng_stream_is_bitwise_unchanged_and_aux_reconstructs():
    basis = basis_fixture()
    labels = np.arange(100) % 3
    rng = np.random.default_rng(123)
    ids = np.empty(100, int)
    for j in np.unique(labels):
        mask = np.flatnonzero(labels == j)
        ids[mask] = rng.choice(9, len(mask), p=basis["conditional"][:, j])
    expected = basis["anchors"][ids] + float(basis["bandwidth"]) * rng.normal(
        size=(100, 15)
    )
    actual, anchors, epsilon = sample_basis(basis, labels, 123, return_aux=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(anchors, ids)
    np.testing.assert_array_equal(
        actual, basis["anchors"][anchors] + basis["bandwidth"] * epsilon
    )


def test_diagonal_density_and_projected_cdf_are_normalized():
    basis = basis_fixture()
    basis["bandwidth"] = np.random.default_rng(3).uniform(0.1, 0.5, (9, 15))
    u = np.array([0.2, 0.3, 0.5])
    x = sample_basis(basis, np.arange(600) % 3, 35)
    weights = basis["conditional"] @ u
    expected = logsumexp(
        np.stack(
            [
                multivariate_normal.logpdf(x, mean=a, cov=np.diag(h**2)) + np.log(w)
                for a, h, w in zip(
                    basis["anchors"], basis["bandwidth"], weights, strict=True
                )
            ]
        ),
        axis=0,
    )
    np.testing.assert_allclose(log_prob(basis, x, u, chunk=40), expected, atol=1e-12)
    f, _, directions, grid = cdf_design(basis, x[:, :5], 3, 7)
    for j in range(3):
        draws = sample_basis(basis, np.full(25000, j), 56 + j)
        np.testing.assert_allclose(
            f[:, j], empirical_features(draws[:, :5], directions, grid), atol=0.016
        )


def test_local_gates_preserve_joint_anchors_and_full_stochastic_support():
    anchors = np.random.default_rng(3).normal(size=(350, 15))
    cfg = config()["local"]
    basis = local_basis(anchors, 20, cfg, 12)
    np.testing.assert_array_equal(basis["anchors"], anchors)
    np.testing.assert_allclose(basis["conditional"].sum(0), 1, atol=1e-14)
    np.testing.assert_allclose(basis["conditional"][:, -1], 1 / 350)
    assert np.any(basis["conditional"][:, :-1] == 0)  # No compulsory broad floor.
    np.testing.assert_allclose(basis["bandwidth"][:, 5:], 0.15)
    x, ids, eps = sample_basis(basis, np.arange(100) % 20, 4, return_aux=True)
    np.testing.assert_allclose(x, anchors[ids] + basis["bandwidth"][ids] * eps)
    assert np.all(np.std(x - anchors[ids], axis=0) > 0)
    with pytest.raises(ValueError, match="neighbor"):
        local_basis(anchors, 20, dict(cfg, gate_neighbors=1), 12)


def test_noise_replay_preserves_original_per_band_rng():
    flux = np.random.default_rng(14).uniform(1e-29, 1e-28, (80, 2))
    bands = [
        dict(name="lsst_r"),
        dict(name="lsst_g", error_model=dict(type="fractional_snr", snr=30)),
    ]
    noise = dict(type="m5_depth", m5=27.5, gamma=0.039)
    frame = observe(
        pd.DataFrame(index=np.arange(80)),
        flux,
        bands,
        noise,
        seed=4,
        selection=dict(band="lsst_r", max_mag_ab=29),
    )
    rows = [7, 11, 31]
    actual, err = apply_saved_noise(
        flux[rows], bands, noise, replay_normals(80, rows, 2, 4)
    )
    np.testing.assert_array_equal(
        actual, frame[["flux_lsst_r", "flux_lsst_g"]].to_numpy()[rows]
    )
    np.testing.assert_array_equal(
        err, frame[["fluxerr_lsst_r", "fluxerr_lsst_g"]].to_numpy()[rows]
    )


def fixture(tmp_path):
    source, data = make_run(tmp_path)
    spec = read(source / "reference/coordinates.json")
    rng = np.random.default_rng(38)
    a = rng.uniform(size=(300, 3))
    a /= a.sum(0)
    basis = dict(
        anchors=rng.normal(size=(300, 15)), conditional=a, bandwidth=np.array(0.15)
    )
    np.savez(source / "reference/basis.npz", **basis)
    noise = dict(type="m5_depth", m5=27.5, gamma=0.039)
    write(data / "noise.json", noise)
    bands = read(data / "decoder.json")["bands"]
    selection = dict(band="lsst_r", max_mag_ab=29)

    def predict(theta):
        return (np.exp(theta[:, 0]) * (1 + 0.01 * theta[:, 1] ** 2) * 1e-28)[:, None]

    manifest = read(source / "MANIFEST.json")
    manifest["selection"] = selection
    for path in (data / "dataset").rglob("*.parquet"):
        frame = pd.read_parquet(path)
        frame["flux_lsst_r"] = predict(frame[pipeline.NAMES].to_numpy())[:, 0]
        frame.to_parquet(path)
        manifest["source_files"][str(path)] = sha(path)
    manifest["source_files"][str(data / "noise.json")] = sha(data / "noise.json")
    write(source / "MANIFEST.json", manifest)
    digest = sha(source / "MANIFEST.json")
    ref = source / "reference"
    finish(
        ref,
        [ref / p for p in ("basis.npz", "coordinates.json", "feature_stats.json")],
        digest,
    )
    finish(source / "population", [source / "population/parent.json"], digest)
    x = sample_basis(basis, np.arange(900) % 3, 1003)
    theta = np.asarray(to_theta(x, spec))
    frame = observe(
        pd.DataFrame(index=np.arange(900)),
        predict(theta),
        bands,
        noise,
        seed=1004,
        selection=selection,
    )
    block = source / "banks/shard_000/block_0000000"
    np.savez(
        block / "bank.npz",
        x=x,
        theta=theta,
        flux=frame[["flux_lsst_r"]].to_numpy(),
        selected=frame.selected_r29.to_numpy(),
        role=np.arange(900) % 5,
        component=np.arange(900) % 3,
    )
    finish(block, [block / "bank.npz"], digest)
    finish(block.parent, [block / "FINAL.json"], digest)
    (source / "report").mkdir()
    finish(source / "report", [], digest)
    cfg = config()
    cfg["evaluation"] = dict(random_directions=2, thresholds=7, population_draws=400)
    cfg["replay"].update(extremes=2, controls=2)
    # Exercise the positive handoff; these permissive fixture gates are NOT production defaults.
    cfg["contracts"].update(
        maximum_cdf_error=1, maximum_physical_sw=10, maximum_physical_marginal_w1=10
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return source, data, path, predict


def test_pipeline_all_cells_replay_handoff_resume_and_no_test_access(
    tmp_path, monkeypatch, capsys
):
    source, data, cfg, predict = fixture(tmp_path)
    original = {p: sha(p) for d in (source, data) for p in d.rglob("*") if p.is_file()}
    real_read = pd.read_parquet

    def guard(path, *args, **kwargs):
        assert Path(path).name != "test.parquet", "No test-truth design selection"
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guard)
    monkeypatch.setattr(pipeline, "photometer", lambda *_: predict)
    out = tmp_path / "redesign"
    pipeline.initialize(source, out, cfg)
    pipeline.report(out)
    assert (out / "report/BLOCKED.json").exists()
    pipeline.prepare(out)
    for i in range(4):
        pipeline.capacity(out, i)
    pipeline.replay(out)
    pipeline.report(out)
    assert not (out / "report/BLOCKED.json").exists()
    f = read(out / "report/FINAL.json")
    assert f["ready_for_reference_bank_benchmark"] and not f["ready_for_production"]
    assert f["selected_candidate"] == "affine_mass"
    handoff = read(out / "report/REFERENCE_CANDIDATE.json")
    assert "u" not in handoff and handoff["diagnostic_weights_forbidden"]
    assert read(out / "replay/FINAL.json")["maximum_noise_scaled_replay_error"] < 1e-12
    times = {
        p: p.stat().st_mtime_ns
        for p in out.rglob("*")
        if p.is_file() and "report" not in p.parts and p.name != "ROADMAP_STATUS.md"
    }
    pipeline.prepare(out)
    for i in range(4):
        pipeline.capacity(out, i)
    pipeline.replay(out)
    assert times == {p: p.stat().st_mtime_ns for p in times}
    assert original == {p: sha(p) for p in original}
    watched = subprocess.run(
        ["bash", "scripts/watch_feniks_reference_redesign.sh", str(out), "--once"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "Candidate: affine_mass" in watched.stdout and "artifacts" not in watched.stdout
    )
    capsys.readouterr()
    pipeline.schedule(out)
    assert "NEED_PREPARE=0" in capsys.readouterr().out
    (data / "noise.json").write_text("{}")
    with pytest.raises(ValueError, match="Immutable source changed"):
        pipeline.schedule(out)


def test_nonfinite_metrics_fail_closed_and_no_q_in_capacity():
    metrics = dict(
        finite_linear_mass_moment=True,
        validation_cdf_max=np.nan,
        physical_sw=0.01,
        physical_marginal_max=0.01,
    )
    gates, _ = qualification(
        metrics, dict(cdf_max=0.01, physical_sw=0.01), config()["contracts"]
    )
    assert not all(gates.values())
    code = inspect.getsource(pipeline.capacity)
    assert (
        "test.parquet" not in code
        and "supervised_fit" not in code
        and "sample_posterior" not in code
    )


def test_submission_parallel_dag_and_resume_only_missing_cells(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fake = tmp_path / "bin"
    fake.mkdir()
    # Keep this a shell orchestration test; the real Python pipeline is tested above.
    (fake / "python").write_text("""#!/usr/bin/env python3
import os,sys
from pathlib import Path
mode=sys.argv[3]; root=Path(sys.argv[sys.argv.index('--root')+1])
if mode=='init':
    (root/'logs').mkdir(parents=True)
elif mode=='schedule':
    print('PREPARE_MINUTES=20\\nCAPACITY_MINUTES=25\\nREPLAY_MINUTES=15\\nREPORT_MINUTES=5\\nCPU_THREADS=4\\nCAPACITY_CONCURRENCY=4')
    print('NEED_PREPARE=0\\nMISSING_CELLS=2\\nNEED_REPLAY=0\\nNEED_REPORT=1' if os.environ.get('RESUMING') else 'NEED_PREPARE=1\\nMISSING_CELLS=0,1,2,3\\nNEED_REPLAY=1\\nNEED_REPORT=1')
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
    # A small isolated archive; source code tests use the real repository separately.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for directory in ("scripts", "euclid_dsps", "configs", "Data"):
        (checkout / directory).mkdir()
    for name in (
        "submit_feniks_reference_redesign.sh",
        "feniks_reference_redesign.slurm",
    ):
        (checkout / "scripts" / name).write_text((repo / "scripts" / name).read_text())
    (checkout / "pyproject.toml").write_text("")
    config_path = checkout / "cfg.yaml"
    config_path.write_text("{}")
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "redesign"
    env = dict(
        os.environ,
        PATH=f"{fake}:" + os.environ["PATH"],
        SCRATCH=str(tmp_path / "scratch"),
        CALL_LOG=str(tmp_path / "calls"),
    )
    script = "scripts/submit_feniks_reference_redesign.sh"
    subprocess.run(
        ["bash", script, str(source), str(root), str(config_path)],
        cwd=checkout,
        env=env,
        check=True,
        capture_output=True,
    )
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert len(calls) == 4
    assert "--array=0,1,2,3%4" in calls[1]
    assert (
        "--dependency=afterok:800" in calls[1]
        and "--dependency=afterok:800" in calls[2]
    )
    assert "--dependency=afterany:800:801:802" in calls[3]
    assert "--gres=gpu:1" in calls[2] and "--time=15" in calls[2]
    assert "--account=jrx@cpu" in calls[1] and not any(
        "--mem" in arg for call in calls for arg in call
    )
    subprocess.run(
        ["bash", script, "--resume", str(root)],
        cwd=checkout,
        env=dict(env, RESUMING="1"),
        check=True,
        capture_output=True,
    )
    calls = [json.loads(line) for line in (tmp_path / "calls").read_text().splitlines()]
    assert len(calls) == 6 and "--array=2%4" in calls[4]
    assert "--dependency=afterany:804" in calls[5]
    failed = subprocess.run(
        ["bash", script, "--resume", str(root)],
        cwd=checkout,
        env=dict(env, ACTIVE_TEST="804"),
        capture_output=True,
    )
    assert failed.returncode != 0 and b"still active" in failed.stderr
