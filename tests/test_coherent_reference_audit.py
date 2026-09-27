"""Cheap numerical/provenance checks, not another scientific training run."""

import inspect
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.native_reference import sample_basis
from euclid_dsps.amortized.reference_capacity import (
    cdf_design,
    empirical_features,
    fit_cdf_weights,
    observable_tail_metrics,
)
from scripts import audit_feniks_coherent_reference as audit
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import finish


def basis_fixture():
    anchors = np.random.default_rng(12).normal(size=(9, 15))
    conditional = np.zeros((9, 3))
    for j in range(3):
        conditional[j * 3 : (j + 1) * 3, j] = 1 / 3
    return dict(anchors=anchors, conditional=conditional, bandwidth=np.array(0.2))


def test_analytic_cdf_matches_full15d_kernel_simulation():
    basis = basis_fixture()
    train = sample_basis(basis, np.arange(6000) % 3, 44)
    f, _, directions, grid = cdf_design(basis, train[:, :5], 2, 11)
    for j in range(3):
        draws = sample_basis(basis, np.full(30000, j), 45 + j)
        empirical = empirical_features(draws[:, :5], directions, grid)
        np.testing.assert_allclose(f[:, j], empirical, atol=0.015)
    assert np.all(np.diff(f.reshape(7, 11, 3), axis=1) >= -1e-12)


def test_cdf_lp_recovers_known_weights_and_certifies_restriction():
    f = np.array([[0.1, 0.7, 0.9], [0.8, 0.2, 0.7], [0.3, 0.9, 0.5]])
    truth = np.array([0.2, 0.3, 0.5])
    u, result = fit_cdf_weights(f, f @ truth)
    np.testing.assert_allclose(u, truth, atol=1e-7)
    assert result["train_cdf_max_error"] < 1e-7
    _, restricted = fit_cdf_weights(np.array([[0.2, 0.3]]), [0.9])
    assert restricted["train_cdf_max_error"] == pytest.approx(0.6)


@pytest.mark.parametrize("a", [np.full((9, 3), 0.2), np.full((9, 3), np.nan)])
def test_unnormalized_or_nonfinite_components_rejected(a):
    basis = basis_fixture()
    basis["conditional"] = a
    with pytest.raises(ValueError):
        cdf_design(basis, basis["anchors"][:, :5])


def test_tail_metric_uses_weights_and_retains_rare_extremes():
    truth = np.linspace(-1, 1, 1001)
    predicted = np.r_[truth, 1e6]
    weights = np.r_[np.ones(1001) * 0.99 / 1001, 0.01]
    row = observable_tail_metrics(predicted, truth, weights)
    assert row["raw_w1_over_iqr"] > 9000
    assert row["upper_capped_diagnostic_w1_over_iqr"] < 0.02
    assert row["predicted_mass_above_truth_q999"] > 0.01
    assert row["asinh_w1"] < 0.2
    assert predicted[-1] == 1e6  # No in-place clipping.
    zero = observable_tail_metrics(truth, truth, np.ones(len(truth)))
    assert zero["raw_w1_over_iqr"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("weights", [[1, -1], [0, 0], [1, np.nan]])
def test_bad_flux_weights_fail(weights):
    with pytest.raises(ValueError):
        observable_tail_metrics([1, 2], [1, 2], weights)


def make_run(tmp_path):
    from euclid_dsps.amortized.coherent_coordinates import fit_coordinates, to_theta
    from euclid_dsps.amortized.features import (
        compute_feature_stats,
        feature_stats_to_json,
    )

    root, source = tmp_path / "run", tmp_path / "data"
    root.mkdir()
    source.mkdir()
    rng = np.random.default_rng(11)
    theta = rng.uniform(-0.7, 0.7, size=(1000, 15))
    spec = fit_coordinates(theta, audit.NAMES, np.full(15, -1), np.ones(15))
    basis = basis_fixture()
    files = {}
    for split in ("train", "validation", "test"):
        x = sample_basis(
            basis, rng.choice(3, 600, p=[0.2, 0.3, 0.5]), int(rng.integers(10000))
        )
        frame = pd.DataFrame(np.asarray(to_theta(x, spec)), columns=audit.NAMES)
        frame["flux_lsst_r"] = np.exp(frame.z_obs)
        for kind in ("parent", "selected_r29"):
            path = source / "dataset" / kind / f"{split}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path)
            files[str(path)] = sha(path)
    write(source / "decoder.json", dict(bands=[dict(name="lsst_r")]))
    files[str(source / "decoder.json")] = sha(source / "decoder.json")
    cfg = dict(seed=3, bank=dict(shards=1, rows_per_shard=900, checkpoint_rows=900))
    write(
        root / "MANIFEST.json",
        dict(source=str(source), source_files=files, settings=cfg, frozen_files={}),
    )
    digest = sha(root / "MANIFEST.json")
    ref = root / "reference"
    ref.mkdir()
    np.savez(ref / "basis.npz", **basis)
    write(ref / "coordinates.json", spec)
    stats = compute_feature_stats(
        np.ones((100, 1)),
        np.ones((100, 1)),
        np.ones((100, 1), bool),
        ("lsst_r",),
        append_mask=True,
    )
    write(ref / "feature_stats.json", feature_stats_to_json(stats))
    finish(ref, sorted(ref.iterdir()), digest)
    pop = root / "population"
    pop.mkdir()
    write(pop / "parent.json", dict(u=[0.6, 0.3, 0.1]))
    finish(pop, [pop / "parent.json"], digest)
    shard = root / "banks/shard_000"
    block = shard / "block_0000000"
    block.mkdir(parents=True)
    np.savez(
        block / "bank.npz",
        flux=rng.lognormal(size=(900, 1)),
        selected=np.ones(900, bool),
        role=np.arange(900) % 5,
        component=np.arange(900) % 3,
    )
    finish(block, [block / "bank.npz"], digest)
    finish(shard, [block / "FINAL.json"], digest)
    return root, source


def test_full_read_only_audit_resume_and_source_immutability(tmp_path):
    root, source = make_run(tmp_path)
    original = {p: sha(p) for d in (root, source) for p in d.rglob("*") if p.is_file()}
    out = tmp_path / "audit"
    audit.run(root, out, draws=400)
    assert read(out / "report/FINAL.json")["ready_for_production"] is False
    assert read(out / "capacity/FINAL.json")["truth_used_for_training"] is True
    assert read(out / "tails/FINAL.json")["resampling_used"] is False
    times = {
        p: p.stat().st_mtime_ns
        for d in (out / "capacity", out / "tails")
        for p in d.iterdir()
    }
    audit.run(root, out, draws=400)
    assert times == {p: p.stat().st_mtime_ns for p in times}
    assert original == {p: sha(p) for p in original}
    with pytest.raises(ValueError, match="inputs changed"):
        audit.run(root, out, draws=401)
    with pytest.raises(ValueError, match="separate audit root"):
        audit.run(root, root / "audit", draws=400)
    (source / "decoder.json").write_text("{}")
    with pytest.raises(ValueError, match="Immutable source changed"):
        audit.run(root, tmp_path / "changed", draws=400)


def test_capacity_has_no_posterior_or_classifier_training():
    source = inspect.getsource(audit.capacity)
    assert "supervised_fit" not in source and "classifier" not in source
    assert "posterior" not in source
    assert "train.parquet" in source
    assert source.index("fit_cdf_weights") < source.index("test.parquet")


def test_available_receipts_distinguish_missing_from_corruption(tmp_path):
    from scripts.analyze_feniks_coherent_inference import verify_available

    write(tmp_path / "MANIFEST.json", dict(test=True))
    digest = sha(tmp_path / "MANIFEST.json")
    for stage in ("reference", "population", "oracle", "posterior", "report"):
        p = tmp_path / stage
        p.mkdir()
        write(p / "data.json", dict(a=1))
        finish(p, [p / "data.json"], digest)
    (tmp_path / "oracle/data.json").unlink()
    result = verify_available(tmp_path)
    assert len(result["verified"]) == 4 and result["unavailable"] == [
        "oracle/data.json"
    ]
    (tmp_path / "posterior/data.json").write_text("{}")
    with pytest.raises(ValueError, match="Artifact changed"):
        verify_available(tmp_path)


def test_frozen_cpu_submission_and_resume(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    (repo / "scripts").mkdir()
    for name in (
        "submit_feniks_coherent_reference_audit.sh",
        "feniks_coherent_reference_audit.slurm",
    ):
        (repo / "scripts" / name).write_text((script_dir / name).read_text())
    for directory in ("euclid_dsps", "configs", "Data"):
        (repo / directory).mkdir()
    (repo / "pyproject.toml").write_text("")
    source = tmp_path / "source"
    (source / "report").mkdir(parents=True)
    (source / "reference").mkdir()
    (source / "report/FINAL.json").write_text("{}")
    (source / "reference/basis.npz").write_text("fixture")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$CALL_LOG"\necho 12345\n'
    )
    (bin_dir / "squeue").write_text('#!/bin/bash\nprintf "%s" "${ACTIVE_TEST:-}"\n')
    for p in bin_dir.iterdir():
        p.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        SCRATCH=str(tmp_path / "scratch"),
        CALL_LOG=str(tmp_path / "args"),
    )
    out = tmp_path / "audit"
    script = "scripts/submit_feniks_coherent_reference_audit.sh"
    subprocess.run(
        ["bash", script, str(source), str(out)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    args = (tmp_path / "args").read_text()
    assert "--account=jrx@cpu" in args and "--cpus-per-task=4" in args
    assert "--mem" not in args and "--gres" not in args
    assert "--time=00:30:00" in args
    subprocess.run(
        ["bash", script, "--resume", str(out)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    failed = subprocess.run(
        ["bash", script, "--resume", str(out)],
        cwd=repo,
        env=dict(env, ACTIVE_TEST="12345"),
        capture_output=True,
    )
    assert failed.returncode != 0 and b"still active" in failed.stderr
