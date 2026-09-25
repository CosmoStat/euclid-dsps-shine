"""Support, measure, truth isolation and restart invariants; no scientific pilot."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from euclid_dsps.amortized.coherent_coordinates import (
    fit_coordinates,
    log_abs_det_dtheta_dx,
    to_theta,
    to_x,
    validate_theta,
)
from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES
from scripts import feniks_coherent_representation as pipeline
from scripts.feniks_avi_experiments import read, sha, write


def coordinates():
    train = np.random.default_rng(27).uniform(-0.7, 0.7, (128, 15))
    train[:, 1] += 10  # deliberately outside the old mass box
    train[0, 5:] = 100
    train[1:10, -1] = 0
    return train, fit_coordinates(
        train, SPLINE15D_PARAMETER_NAMES, [-1.0] * 15, [1.0] * 15
    )


def test_unbounded_coordinates_preserve_tail_rows_and_zeros():
    train, spec = coordinates()
    before = train.copy()
    validate_theta(train, spec)
    restored = np.asarray(to_theta(to_x(train, spec), spec))
    np.testing.assert_allclose(restored, train, atol=1e-11)
    np.testing.assert_array_equal(train, before)
    assert not spec["clipping"] and not spec["dequantization"]
    assert sum(spec["bounded"]) == 4
    heldout = train[:1].copy()
    heldout[0, 1] = -50
    heldout[0, -1] = -500
    np.testing.assert_allclose(to_theta(to_x(heldout, spec), spec), heldout, atol=1e-10)
    heldout[0, 0] = 1
    with pytest.raises(ValueError, match="OPEN support"):
        validate_theta(heldout, spec)
    assert not np.isfinite(to_x(heldout, spec)).all()  # never silently clip


def test_coordinate_jacobian_and_normalized_measure():
    import jax
    import jax.numpy as jnp
    from scipy.special import ndtr

    train, spec = coordinates()
    x = to_x(train[:4], spec)
    actual = log_abs_det_dtheta_dx(x, spec)
    expected = jax.vmap(
        lambda v: jnp.linalg.slogdet(jax.jacfwd(lambda y: to_theta(y, spec))(v))[1]
    )(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    # A normalized base pushes forward to the full declared support, not a
    # clipped/truncated density. CDF endpoints for bounded and real-line axes.
    for k in range(15):
        t = train[:2].copy()
        if spec["bounded"][k]:
            t[:, k] = [-1 + 1e-12, 1 - 1e-12]
        else:
            t[:, k] = [-1e20, 1e20]
        transformed = np.asarray(to_x(t, spec))[:, k]
        assert np.diff(ndtr(transformed)).item() == pytest.approx(1, abs=1e-8)


def prepared(tmp_path, monkeypatch):
    from test_audit_feniks_coherent_parent import dataset_fixture
    from test_feniks_clean_parent import small_settings

    source, old = dataset_fixture(tmp_path)
    # The fixture has one synthetic band; add only projection metadata.
    decoder = read(source / "decoder.json")
    decoder["model"] = dict(n_sfh_bins=80)
    write(source / "decoder.json", decoder)
    m = read(source / "MANIFEST.json")
    m["frozen_files"]["decoder.json"] = sha(source / "decoder.json")
    write(source / "MANIFEST.json", m)
    receipt = read(source / "report/FINAL.json")
    receipt["contract"] = sha(source / "MANIFEST.json")
    write(source / "report/FINAL.json", receipt)
    cfg = yaml.safe_load(
        Path("configs/experiments/feniks_coherent_representation.yaml").read_text()
    )
    cfg["flow"].update(small_settings(), epochs=1, batch_size=128, validation_limit=128)
    cfg["draws"] = 128
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    root = tmp_path / "representation"
    pipeline.prepare(source, old, root, config)
    return root, source


def test_preparation_uses_train_only_and_freezes_all_inputs(tmp_path, monkeypatch):
    root, source = prepared(tmp_path, monkeypatch)
    spec = read(root / "coordinates.json")
    frame = pd.read_parquet(source / "dataset/parent/train.parquet")
    expected = fit_coordinates(
        frame[list(SPLINE15D_PARAMETER_NAMES)].to_numpy(),
        SPLINE15D_PARAMETER_NAMES,
        [-1.0] * 15,
        [1.0] * 15,
    )
    assert spec == expected
    assert not read(root / "MANIFEST.json")["dataset_modified"]
    with np.load(root / "cache/test.npz") as f:
        assert len(f["theta"]) == 300
    (root / "coordinates.json").write_text("{}")
    with pytest.raises(ValueError, match="configuration changed"):
        pipeline.settings(root)


def test_sfh_provenance_distinguishes_floor_and_nonfloor_knots():
    names = list(SPLINE15D_PARAMETER_NAMES)[5:]
    knots = pd.DataFrame(
        {f"spline_log_sfr_{k:02d}": [-30.0, -5.0, float(k)] for k in range(11)}
    )
    contrasts = pd.DataFrame(np.diff(knots.to_numpy(), axis=1), columns=names)
    result = pipeline.projection_zero_table(contrasts, contrasts, knots)
    assert result.replay_zero_count.eq(2).all()
    assert result.equal_floor_knots_count.eq(1).all()
    assert result.equal_nonfloor_knots_count.eq(1).all()
    assert result.max_replay_error.eq(0).all()


def test_two_factor_training_report_and_resume_smoke(tmp_path, monkeypatch):
    root, source = prepared(tmp_path, monkeypatch)
    original = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    for task in (0, 1):
        pipeline.fit(root, task)
    _, _, digest = pipeline.settings(root)
    zero_dir = root / "sfh_zeros"
    zero_dir.mkdir()
    pipeline.finish(zero_dir, [], digest, replay_pass=True)
    # Exercise actual numerical report; figures get a separate lightweight smoke.
    monkeypatch.setattr(pipeline, "plots", lambda *args: None)
    pipeline.report(root)
    final = read(root / "report/FINAL.json")
    assert final["dimensions"] == 15
    assert not final["ready_for_production"] and not final["population_recovered"]
    assert np.isfinite(final["heldout_nll_theta"])
    before = {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    pipeline.fit(root, 0)
    pipeline.report(root)
    assert before == {str(p): sha(p) for p in root.rglob("*") if p.is_file()}
    assert original == {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    (root / "report/FINAL.json").unlink()
    (root / "physical/FINAL.json").unlink()
    with pytest.raises(RuntimeError, match="Incomplete stages"):
        pipeline.report(root)
    assert read(root / "report/BLOCKED.json")["missing"] == ["physical"]
    history_hash = sha(root / "physical/training.jsonl")
    pipeline.fit(root, 0)  # resume completed epochs, missing only the final receipt
    assert sha(root / "physical/training.jsonl") == history_hash
    assert read(root / "physical/FINAL.json")["epoch"] == 1


def test_launcher_resume_missing_only_and_no_mem_option(tmp_path, monkeypatch):
    root, _ = prepared(tmp_path, monkeypatch)
    _, _, digest = pipeline.settings(root)
    (root / "physical").mkdir()
    pipeline.finish(root / "physical", [], digest)
    (root / "CODE_DIR").write_text(str(Path.cwd()))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    for name, body in {
        "sbatch": 'printf "%s\\n" "$*" >> "$CALLS"\nprintf "12345\\n"\n',
        "squeue": "exit 0\n",
    }.items():
        p = bin_dir / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(0o755)
    (bin_dir / "python").symlink_to(Path(os.sys.executable))
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls))
    subprocess.run(
        [
            "bash",
            "scripts/submit_feniks_coherent_representation.sh",
            "--resume",
            str(root),
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    submitted = calls.read_text()
    assert "--array=1%2" in submitted
    assert "--mem" not in submitted
    assert "--dependency=afterany:" in submitted
    assert "--time=90" in submitted
    assert read(root / "MANIFEST.json")["settings"]["flow"]["epochs"] == 1
