from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.feniks_avi_inference import joint_resample, sample_frame


def test_prepare_requires_all_completed_arms_and_finite_truth(tmp_path, monkeypatch):
    import pandas as pd

    from euclid_dsps.amortized.avi_experiments import ARMS
    from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
    from scripts import feniks_avi_experiments as runner
    from scripts.feniks_avi_inference import prepare

    training = tmp_path / "training"
    training.mkdir()
    catalog = tmp_path / "catalog.parquet"
    truth = pd.DataFrame(np.ones((4, 15)), columns=FENIKS_SPLINE15D_PARAMETERS)
    truth.to_parquet(catalog)
    np.save(training / "validation.npy", [1, 3])
    runner.write(training / "MANIFEST.json", {"validation_catalog": str(catalog)})
    monkeypatch.setattr(
        runner, "check_inputs", lambda root: runner.read(root / "MANIFEST.json")
    )
    for arm in ARMS:
        d = training / "arms" / arm.name
        d.mkdir(parents=True)
        (d / "encoder.eqx").write_bytes(b"fixture")
        runner.write(
            d / "FINAL.json",
            {
                "status": "TRAINING_COMPLETE",
                "manifest_sha256": runner.sha(training / "MANIFEST.json"),
            },
        )
    out = tmp_path / "out"
    prepare(training, out, 256)
    frame = pd.read_parquet(out / "inference_truth.parquet")
    assert frame.row_index.tolist() == [1, 3]
    assert runner.read(out / "MANIFEST.json")["objects"] == 2
    with pytest.raises(FileExistsError):
        prepare(training, out, 256)
    truth.iloc[1, 0] = np.nan
    truth.to_parquet(catalog)
    with pytest.raises(ValueError, match="finite canonical truth"):
        prepare(training, tmp_path / "bad", 256)


def test_joint_resampling_keeps_correlations_and_duplicates():
    theta = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
    draws, ids = joint_resample(
        theta, np.array([0.0, 1.0, 0.0]), 256, np.random.default_rng(2)
    )
    np.testing.assert_array_equal(draws, np.tile(theta[1], (256, 1)))
    assert len(np.unique(ids)) == 1
    with pytest.raises(ValueError):
        joint_resample(theta, np.zeros(3), 256, np.random.default_rng(2))


def test_frame_identity_and_sample_order():
    theta = np.arange(24.0).reshape(3, 4, 2)
    frame = sample_frame(theta, np.array([7, 9, 20]), ("a", "b"))
    assert frame.object_id.tolist() == [7] * 4 + [9] * 4 + [20] * 4
    assert frame.sample_id.tolist() == [0, 1, 2, 3] * 3
    np.testing.assert_array_equal(frame[["a", "b"]].to_numpy(), theta.reshape(-1, 2))


def test_comparison_and_truth_overlay_plots(tmp_path):
    import pandas as pd
    from PIL import Image

    from scripts.feniks_avi_inference import plot_comparison

    names = ["source", "B_experts", "E_experts_elbo"]
    pd.DataFrame(
        {"object_id": [4, 7], "row_index": [4, 7], "a": [0.0, 1.0], "b": [1.0, 2.0]}
    ).to_parquet(tmp_path / "inference_truth.parquet")
    for name in names:
        dest = tmp_path / "arms" / name
        dest.mkdir(parents=True)
        pd.DataFrame(
            {
                "row_index": [4, 7],
                "ess_fraction": [0.1, 0.2],
                "max_weight": [0.2, 0.1],
                "raw_predictive_rms": [1.0, 2.0],
                "is_predictive_rms": [0.8, 1.0],
            }
        ).to_csv(dest / "metrics.csv", index=False)
        for kind in ("raw", "is"):
            sample_frame(
                np.random.default_rng(4).normal(size=(2, 16, 2)), [4, 7], ["a", "b"]
            ).to_parquet(dest / f"{kind}_0.parquet")
    plot_comparison(tmp_path, {"arms": names})
    for path in [tmp_path / "comparison.png", tmp_path / "marginals" / "row_4.png"]:
        pixels = np.asarray(Image.open(path))
        assert pixels.std() > 10


def test_residual_inference_four_devices(tmp_path):
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        "PYTHONPATH": ".:tests",
    }
    result = subprocess.run(
        [sys.executable, "tests/avi_inference_worker.py", str(tmp_path / "run")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
