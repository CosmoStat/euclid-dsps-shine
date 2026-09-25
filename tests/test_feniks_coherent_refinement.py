"""Bounded continuation, empirical tails, precision and HPC recovery contracts."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import wasserstein_distance

from euclid_dsps.amortized.distribution_diagnostics import (
    physical_sw,
    tail_table,
    w1_parts,
)
from scripts import feniks_coherent_refinement as pipeline
from scripts.feniks_avi_experiments import read, sha


@pytest.mark.parametrize("n,m", [(13, 7), (100, 100), (1, 8)])
def test_tail_partition_is_full_empirical_w1(n, m):
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=n), rng.normal(size=m)
    a[-1] = 100
    parts = w1_parts(a, b)
    assert parts.sum() == pytest.approx(wasserstein_distance(a, b), abs=1e-12)
    assert (parts >= 0).all()
    np.testing.assert_array_equal(w1_parts(a, a), 0)


def test_signed_bias_and_physical_group():
    truth = np.arange(100.0).reshape(20, 5)
    t = tail_table(truth - 10, truth, list("abcde"))
    assert (t.signed_mean_bias_over_iqr < 0).all()
    assert (t.signed_median_bias_over_iqr < 0).all()
    assert physical_sw(truth, truth, 7) == 0
    with pytest.raises(ValueError, match="five"):
        physical_sw(np.ones((8, 15)), np.ones((8, 15)), 7)
    with pytest.raises(ValueError, match="finite"):
        tail_table(np.full((10, 5), np.inf), truth, list("abcde"))


def test_zero_precision_attribution():
    from euclid_dsps.prior_learning.spline15d_schema import SFH_CONTRAST_NAMES

    k32 = pd.DataFrame(
        {f"spline_log_sfr_{k:02d}": [-14.0, -3.0, -2.0] for k in range(11)}
    )
    k64 = k32.copy()
    for k in range(11):
        k64.iloc[1, k] += k * 1e-10
    a = pd.DataFrame(np.diff(k32, axis=1), columns=SFH_CONTRAST_NAMES)
    b = pd.DataFrame(np.diff(k64, axis=1), columns=SFH_CONTRAST_NAMES)
    table = pipeline.precision_table(a, a, k32, b, k64)
    assert table.equal_floor_knots_count.eq(1).all()
    assert table.zero32_nonzero64_count.eq(1).all()
    assert table.zero32_zero64_count.eq(2).all()
    assert table.floor64_count.eq(1).all()


def test_native_sfh_precision_is_explicit_and_default_unchanged(monkeypatch):
    from collections import namedtuple

    import jax.numpy as jnp

    from euclid_dsps import model

    D = namedtuple(
        "D", "lgmcrit lgy_at_mcrit indx_lo indx_hi lg_qt qlglgdt lg_drop lg_rejuv"
    )
    M = namedtuple("M", "logm0 logtc early_index late_index t_peak")
    captured = []

    def calc(d, m, time, **kw):
        captured.append((d.lgmcrit.dtype, time.dtype, float(d.lgmcrit)))
        return jnp.array([-1.0, 2.0], dtype=time.dtype)

    monkeypatch.setattr(
        model,
        "_import_diffstar_api",
        lambda: (calc, D(*([1.0] * 8)), D, M(*([1.0] * 5)), 0.16),
    )
    params = dict(diffstar_lgmcrit=1.0000000001)
    a = model.build_diffsky_basic_sfh_table_jax(jnp.array([0.1, 1.0]), 1.0, params)
    b = model.build_diffsky_basic_sfh_table_jax(
        jnp.array([0.1, 1.0]), 1.0, params, numerical_dtype=jnp.float64
    )
    assert captured[0] == (np.dtype("float32"), np.dtype("float32"), 1.0)
    assert captured[1] == (np.dtype("float64"), np.dtype("float64"), 1.0000000001)
    assert a[0] == pytest.approx(1e-14) and b[0] == pytest.approx(1e-14)


def test_float64_truth_loading_does_not_round_native_parameters():
    from euclid_dsps.synthetic_diffsky.photometry import (
        GROUND_TRUTH_COLUMNS,
        theta_from_truth_frame,
    )

    frame = pd.DataFrame({col: [1.0000000001] for col in GROUND_TRUTH_COLUMNS.values()})
    assert (theta_from_truth_frame(frame) == 1.0).all()
    assert (theta_from_truth_frame(frame, dtype=np.float64) == 1.0000000001).all()


def prepared(tmp_path, monkeypatch):
    import equinox as eqx
    from test_feniks_coherent_representation import prepared as prepare_source

    from euclid_dsps.amortized.structured_population import factor_template
    from scripts.feniks_coherent_parent import finish, settings

    source, _ = prepare_source(tmp_path, monkeypatch)
    _, cfg, digest = settings(source)
    for i, name in enumerate(pipeline.FACTORS):
        out = source / name
        out.mkdir()
        model = factor_template(
            cfg["flow"], 5 if i == 0 else 10, 1 if i == 0 else 5, cfg["seed"] + i
        )
        eqx.tree_serialise_leaves(out / "best.eqx", model)
        pd.DataFrame(
            [dict(epoch=1, train_nll=30.0, validation_nll=30.0, best_nll=30.0)]
        ).to_json(out / "training.jsonl", orient="records", lines=True)
        finish(out, [out / "best.eqx", out / "training.jsonl"], digest)
    for name in ("report", "sfh_zeros"):
        (source / name).mkdir()
        finish(source / name, [], digest)
    with np.load(source / "cache/validation.npz") as f:
        np.savez(source / "report/draws.npz", x=f["x"] + 0.01, theta=f["theta"] + 0.01)
    config = tmp_path / "followup.yaml"
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_coherent_refinement.yaml").read_text()
    )
    cfg.update(epochs=2, milestones=[0, 1, 2], draws=128)
    config.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "followup"
    pipeline.prepare(source, root, config)
    return root, source


def test_freezes_inputs_and_uses_best_not_last(tmp_path, monkeypatch):
    root, source = prepared(tmp_path, monkeypatch)
    m, _, _ = pipeline.settings(root)
    assert m["conditional_frozen"]
    assert str(source / "physical/best.eqx") in m["source_files"]
    assert not any("state_" in p for p in m["source_files"])
    (source / "sfh_conditional/best.eqx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Immutable source"):
        pipeline.settings(root)


def test_audit_training_recovery_and_report_smoke(tmp_path, monkeypatch):
    from euclid_dsps.prior_learning import spline15d
    from scripts import feniks_coherent_representation

    root, source = prepared(tmp_path, monkeypatch)
    before = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}

    def projection(original, precision="float32", **kw):
        names = read(source / "coordinates.json")["names"]
        contrasts = original[names[5:]].to_numpy()
        values = np.c_[np.zeros(len(original)), np.cumsum(contrasts, axis=1)]
        knots = pd.DataFrame(
            values, columns=[f"spline_log_sfr_{k:02d}" for k in range(11)]
        )
        return original[names].copy(), knots

    monkeypatch.setattr(spline15d, "project_diffsky_frame_to_spline15d", projection)
    pipeline.audit(root)
    assert read(root / "audit/FINAL.json")["replay_pass"]
    real_metrics = pipeline.checkpoint_metrics

    def interrupt(model, source, cfg, out, epoch):
        if epoch == 1:
            raise RuntimeError("synthetic interruption after optimizer checkpoint")
        return real_metrics(model, source, cfg, out, epoch)

    monkeypatch.setattr(pipeline, "checkpoint_metrics", interrupt)
    real_load = np.load

    def train_load(path, *args, **kw):
        assert str(path) != str(source / "cache/test.npz"), "Test leaked into training"
        return real_load(path, *args, **kw)

    with monkeypatch.context() as training_patch:
        training_patch.setattr(np, "load", train_load)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            pipeline.fit(root)
        assert read(root / "physical/RESUME.json")["epoch"] == 1
        monkeypatch.setattr(pipeline, "checkpoint_metrics", real_metrics)
        pipeline.fit(root)
    h = pd.read_json(root / "physical/training.jsonl", lines=True)
    assert h.epoch.to_list() == [1, 2]
    assert h.learning_rate.eq(0.000003).all()
    assert read(root / "physical/INITIAL_VALIDATION.json")["optimizer_reset"]
    assert read(root / "physical/transport.json")["passed"]
    monkeypatch.setattr(feniks_coherent_representation, "plots", lambda *args: None)
    pipeline.report(root)
    final = read(root / "report/FINAL.json")
    assert final["status"] == "COMPLETE" and not final["ready_for_production"]
    assert final["conditional_frozen"] and not final["posterior_trained"]
    assert (root / "report/physical_before_after.png").stat().st_size > 1000
    with np.load(root / "report/joint_draws.npz") as f:
        assert f["theta"].shape == (128, 15)
    unchanged = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    pipeline.fit(root)
    pipeline.audit(root)
    pipeline.report(root)
    assert unchanged == {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    assert before == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}


def test_launcher_resume_and_failed_watcher(tmp_path, monkeypatch):
    root, _ = prepared(tmp_path, monkeypatch)
    (root / "CODE_DIR").write_text(str(Path.cwd()))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    for name, body in {
        "sbatch": 'printf "%s\\n" "$*" >> "$CALLS"\nprintf "12345\\n"\n',
        "squeue": "exit 0\n",
        "sacct": "printf '12345|TIMEOUT|\\n'\n",
    }.items():
        p = bin_dir / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(0o755)
    (bin_dir / "python").symlink_to(Path(os.sys.executable))
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls))
    command = [
        "bash",
        "scripts/submit_feniks_coherent_refinement.sh",
        "--resume",
        str(root),
    ]
    subprocess.run(command, check=True, env=env, capture_output=True)
    submitted = calls.read_text().splitlines()
    assert len(submitted) == 3
    assert sum("--gres=gpu:1" in line for line in submitted) == 1
    assert "--time=60" in submitted[0]
    assert "--dependency=afterany:" in submitted[2]
    assert "--dependency" not in submitted[1]
    assert all("--mem" not in line for line in submitted)
    watched = subprocess.run(
        ["bash", "scripts/watch_feniks_coherent_refinement.sh", str(root), "--once"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    ).stdout
    assert "TIMEOUT" in watched and "Truth-trained oracle" in watched
    _, _, digest = pipeline.settings(root)
    (root / "audit").mkdir()
    pipeline.finish(root / "audit", [], digest)
    calls.unlink()
    subprocess.run(command, check=True, env=env, capture_output=True)
    assert len(calls.read_text().splitlines()) == 2
