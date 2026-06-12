from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

ssp_inr_data = pytest.importorskip("euclid_dsps.experimental.ssp_inr.data")
ssp_inr_evaluate = pytest.importorskip("euclid_dsps.experimental.ssp_inr.evaluate")
ssp_inr_train = pytest.importorskip("euclid_dsps.experimental.ssp_inr.train")

load_spectral_grid = ssp_inr_data.load_spectral_grid
evaluate_experiment = ssp_inr_evaluate.evaluate_experiment
train_experiment = ssp_inr_train.train_experiment


def test_spectral_grid_loader_subsets_synthetic_ssp(tmp_path: Path) -> None:
    path = _write_synthetic_ssp(tmp_path / "ssp.h5")
    grid = load_spectral_grid(path, "ssp_flux", max_curves=5, max_wave=11, seed=3)

    assert grid.full_shape == (3, 4, 32)
    assert grid.flux.shape == (5, 11)
    assert grid.curve_coords.shape == (5, 2)
    assert grid.axis_names == ("ssp_lgmet", "ssp_lg_age_gyr", "ssp_wave")


def test_latent_inr_train_and_eval_write_artifacts(tmp_path: Path) -> None:
    path = _write_synthetic_ssp(tmp_path / "ssp.h5")
    train_out = tmp_path / "train"
    eval_out = tmp_path / "eval"

    summary = train_experiment(
        asset=path,
        dataset="ssp_flux",
        model="latent_basis_mlp",
        out=train_out,
        max_curves=8,
        max_wave=16,
        steps=2,
        batch_size=4,
        hidden_width=8,
        hidden_layers=1,
        basis_k=3,
        progress=False,
    )
    checkpoint = Path(summary["checkpoint"])
    assert checkpoint.exists()
    assert (train_out / "loss_curve.png").exists()
    assert (train_out / "reconstruction_examples_full.png").exists()
    assert (train_out / "reconstruction_examples_useful_900_50000A.png").exists()

    result = evaluate_experiment(
        asset=path,
        dataset="ssp_flux",
        out=eval_out,
        checkpoints=[checkpoint],
        log_svd_k=[2],
        max_curves=8,
        max_wave=16,
        timing_repeats=1,
    )

    assert len(result["models"]) == 3
    assert (eval_out / "metrics.json").exists()
    assert (eval_out / "metrics_summary.csv").exists()
    assert (eval_out / "reconstruction_examples_full.png").exists()
    assert (eval_out / "reconstruction_examples_useful_900_50000A.png").exists()
    assert (eval_out / "error_histograms.png").exists()
    assert (eval_out / "error_histograms_useful_wave.png").exists()
    assert (eval_out / "error_histograms_useful_peak1em04.png").exists()
    assert (eval_out / "error_ecdf_useful_peak1em04.png").exists()
    assert (eval_out / "wavelength_error_profile_useful_peak1em04.png").exists()
    assert (eval_out / "curve_error_heatmap_useful_peak1em04.png").exists()
    assert (eval_out / "worst_reconstruction_examples_useful_peak1em04.png").exists()
    assert (eval_out / "runtime_size_tradeoff.png").exists()
    assert (eval_out / "report.md").exists()


def test_compressed_coeff_mlp_train_and_eval(tmp_path: Path) -> None:
    path = _write_synthetic_ssp(tmp_path / "ssp.h5")
    baseline = _write_synthetic_compressed_ssp(tmp_path / "ssp_basis.h5")
    train_out = tmp_path / "coeff_train"
    eval_out = tmp_path / "coeff_eval"

    summary = train_experiment(
        asset=path,
        dataset="ssp_flux",
        model="compressed_coeff_mlp",
        out=train_out,
        coeff_baseline=baseline,
        coeff_loss="coeff",
        max_curves=12,
        max_wave=16,
        steps=2,
        batch_size=4,
        hidden_width=8,
        hidden_layers=1,
        progress=False,
    )

    checkpoint = Path(summary["checkpoint"])
    assert checkpoint.exists()
    assert summary["model_kind"] == "compressed_coeff_mlp"

    result = evaluate_experiment(
        asset=path,
        dataset="ssp_flux",
        out=eval_out,
        checkpoints=[checkpoint],
        compressed_baselines=[baseline],
        max_curves=12,
        max_wave=16,
        timing_repeats=1,
    )

    kinds = {row["model_kind"] for row in result["models"]}
    assert "compressed_coeff_mlp" in kinds
    assert "existing_low_rank_compression" in kinds
    assert (eval_out / "error_ecdf_useful_peak1em04.png").exists()
    assert (eval_out / "wavelength_error_profile_useful_peak1em04.png").exists()
    assert (eval_out / "curve_error_heatmap_useful_peak1em04.png").exists()
    assert (eval_out / "worst_reconstruction_examples_useful_peak1em04.png").exists()


def _write_synthetic_ssp(path: Path) -> Path:
    lgmet = np.asarray([-2.0, -1.0, 0.0], dtype=np.float32)
    lg_age = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float32)
    wave = np.linspace(1000.0, 9000.0, 32, dtype=np.float32)
    met_term = 1.0 + 0.1 * np.arange(len(lgmet), dtype=np.float32)[:, None, None]
    age_term = 1.0 + 0.05 * np.arange(len(lg_age), dtype=np.float32)[None, :, None]
    wave_term = 1.0 + 0.2 * np.sin(wave[None, None, :] / 1200.0)
    flux = 1.0e-3 * met_term * age_term * wave_term
    with h5py.File(path, "w") as handle:
        handle["ssp_lgmet"] = lgmet
        handle["ssp_lg_age_gyr"] = lg_age
        handle["ssp_wave"] = wave
        handle["ssp_flux"] = flux.astype(np.float32)
    return path


def _write_synthetic_compressed_ssp(path: Path) -> Path:
    lgmet = np.asarray([-2.0, -1.0, 0.0], dtype=np.float32)
    lg_age = np.asarray([-3.0, -2.0, -1.0, 0.0], dtype=np.float32)
    wave = np.linspace(1000.0, 9000.0, 32, dtype=np.float32)
    basis = np.stack(
        [
            np.ones_like(wave),
            np.sin(wave / 1200.0),
        ],
        axis=0,
    ).astype(np.float32)
    coeff = np.empty((len(lgmet), len(lg_age), 2), dtype=np.float32)
    for met_index in range(len(lgmet)):
        for age_index in range(len(lg_age)):
            amplitude = 1.0e-3 * (1.0 + 0.1 * met_index) * (1.0 + 0.05 * age_index)
            coeff[met_index, age_index, 0] = amplitude
            coeff[met_index, age_index, 1] = 0.2 * amplitude
    with h5py.File(path, "w") as handle:
        handle["ssp_lgmet"] = lgmet
        handle["ssp_lg_age_gyr"] = lg_age
        handle["ssp_wave"] = wave
        handle["ssp_basis"] = basis
        handle["ssp_coeff"] = coeff.astype(np.float16)
        handle["ssp_scale"] = np.ones(coeff.shape[:-1], dtype=np.float32)
    return path
