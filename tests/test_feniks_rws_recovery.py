from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.elbo import objective_uses_truth
from euclid_dsps.config import load_config
from euclid_dsps.photometry import abmag_to_fnu_cgs
from scripts.build_feniks_rws_recovery_manifests import build
from scripts.evaluate_feniks_rws_recovery import (
    CANDIDATES,
    SEEDS,
    finalize_confirmation,
    predictive_metrics,
    select_pilot,
    support_tail_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"


@pytest.mark.parametrize(
    ("name", "likelihood", "context", "layers", "width"),
    (
        (
            "feniks_rws_recovery_r25_historical_t2.yaml",
            "student_t",
            "base_moments",
            4,
            128,
        ),
        (
            "feniks_rws_recovery_r25_historical_gaussian_eval.yaml",
            "gaussian",
            "base_moments",
            4,
            128,
        ),
        (
            "feniks_rws_recovery_r25_current_t2.yaml",
            "student_t",
            "residual_photometry",
            6,
            256,
        ),
        (
            "feniks_rws_recovery_r25_current_gaussian_eval.yaml",
            "gaussian",
            "residual_photometry",
            6,
            256,
        ),
    ),
)
def test_recovery_configs_are_truth_free_fixed_prior_rws(
    name: str,
    likelihood: str,
    context: str,
    layers: int,
    width: int,
) -> None:
    raw = load_config(CONFIG_DIR / name)
    cfg = amortized_config(raw)

    assert raw["truth"].get("parameter_columns", {}) == {}
    assert cfg["objective"]["mode"] == "reweighted_wake_sleep"
    assert not objective_uses_truth(cfg["objective"])
    assert cfg["prior"]["train_jointly"] is False
    assert cfg["objective"]["wake"]["train_prior"] is False
    assert cfg["objective"]["wake"]["train_encoder"] is True
    assert cfg["likelihood"]["type"] == likelihood
    assert cfg["encoder"]["context_encoder"] == context
    assert cfg["encoder"]["flow_layers"] == layers
    assert cfg["encoder"]["flow_hidden_size"] == width
    assert cfg["training"]["epochs"] == 180


def _catalog(path: Path, *, rows: int) -> None:
    threshold = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    pd.DataFrame(
        {
            "flux_lsst_r": threshold
            * np.where(np.arange(rows) % 5 == 0, 0.5, 2.0),
            # These columns deliberately exist but must never be read by build().
            "redshift_true": np.linspace(0.0, 3.0, rows),
            "truth_parameter": np.arange(rows, dtype=float),
        }
    ).to_parquet(path, index=False)


def test_manifest_builder_uses_selected_observations_and_disjoint_test_cohorts(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    _catalog(train, rows=40)
    _catalog(test, rows=30)
    out = tmp_path / "manifests"

    receipt = build(
        train_catalog=train,
        test_catalog=test,
        out=out,
        validation_objects=5,
        pilot_objects=6,
        confirmation_objects=10,
        seed=9,
    )
    training = np.load(out / "train_indices.npy")
    validation = np.load(out / "validation_indices.npy")
    pilot = np.load(out / "pilot_indices.npy")
    confirmation = np.load(out / "confirmation_indices.npy")
    threshold = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    train_frame = pd.read_parquet(train, columns=["flux_lsst_r"])
    test_frame = pd.read_parquet(test, columns=["flux_lsst_r"])

    assert set(training).isdisjoint(set(validation))
    assert set(pilot).isdisjoint(set(confirmation))
    assert np.all(train_frame.iloc[training].flux_lsst_r.to_numpy() > threshold)
    assert np.all(train_frame.iloc[validation].flux_lsst_r.to_numpy() > threshold)
    assert np.all(test_frame.iloc[pilot].flux_lsst_r.to_numpy() > threshold)
    assert np.all(test_frame.iloc[confirmation].flux_lsst_r.to_numpy() > threshold)
    assert receipt["truth_columns_requested"] == []
    assert receipt["truth_used_for_training_or_checkpoint_selection"] is False


def test_predictive_gate_uses_every_dense_draw(tmp_path: Path) -> None:
    root = tmp_path / "inference"
    root.mkdir()
    summary = pd.DataFrame(
        {
            "row_index": [4, 4, 5, 5],
            "band": ["g", "r", "g", "r"],
            "obs_flux_fnu_cgs": [0.0, 0.0, 0.0, 0.0],
            "obs_err_fnu_cgs": [1.0, 1.0, 1.0, 1.0],
            "valid": [True, True, True, True],
        }
    )
    summary.to_parquet(root / "posterior_predictive_residual_summary.parquet")
    dense = []
    for row in (4, 5):
        for band in ("g", "r"):
            for sample, model_flux in enumerate((-1.0, 1.0)):
                dense.append(
                    {
                        "row_index": row,
                        "sample_id": sample,
                        "band": band,
                        "model_flux_fnu_cgs": model_flux,
                    }
                )
    pd.DataFrame(dense).to_parquet(root / "posterior_predictive_flux.parquet")

    result = predictive_metrics(root, out=tmp_path / "gate")

    assert result["objects"] == 2
    assert result["draws"] == 8
    assert result["median_band_rms"] == pytest.approx(1.0)
    assert result["median_absolute_band_bias"] == pytest.approx(0.0)
    assert result["status"] == "PASS"


def test_support_tail_gate_rejects_a_bad_lower_tail(tmp_path: Path) -> None:
    good = pd.DataFrame(
        {
            "n_proposal_samples": [2048] * 10,
            "n_finite_logweights": [2048] * 10,
            "raw_ess_fraction": [0.08] * 10,
            "max_raw_weight": [0.04] * 10,
        }
    )
    good.to_parquet(tmp_path / "importance_diagnostics.parquet")
    assert support_tail_metrics(tmp_path)["status"] == "PASS"

    bad = good.copy()
    bad.loc[:1, "raw_ess_fraction"] = 1.0 / 2048.0
    bad.loc[:1, "max_raw_weight"] = 0.95
    bad.to_parquet(tmp_path / "importance_diagnostics.parquet")
    result = support_tail_metrics(tmp_path)
    assert result["status"] == "FAIL"
    assert result["fraction_raw_ess_below_0p01"] == pytest.approx(0.2)


def _run_summary(candidate: str, seed: int, *, status: str, ess: float) -> dict:
    return {
        "status": status,
        "candidate": candidate,
        "seed": seed,
        "checkpoint": f"/{candidate}/{seed}/best.eqx",
        "feature_stats": f"/{candidate}/{seed}/feature_stats.json",
        "exact_gaussian_ordinary_iw": {
            "status": status,
            "objects": 512,
            "draws": 512 * 2048,
            "median_raw_ess_fraction": ess,
            "fraction_pareto_k_gt_0p7": 0.1,
            "fraction_pareto_k_gt_1": 0.01,
        },
        "exact_gaussian_posterior_predictive": {
            "status": status,
            "median_band_rms": 1.2,
        },
    }


def test_promotion_requires_both_seeds_then_independent_confirmation(
    tmp_path: Path,
) -> None:
    for candidate in CANDIDATES:
        for seed in SEEDS:
            run = tmp_path / candidate / f"seed_{seed}"
            run.mkdir(parents=True)
            passed = candidate == "historical_4x128"
            (run / "pilot_summary.json").write_text(
                json.dumps(
                    _run_summary(
                        candidate,
                        seed,
                        status="PASS" if passed else "FAIL",
                        ess=0.2 if passed else 0.01,
                    )
                )
            )

    pilot = select_pilot(tmp_path)

    assert pilot["status"] == "PASS"
    assert pilot["selected_candidate"] == "historical_4x128"
    for seed in SEEDS:
        run = tmp_path / "historical_4x128" / f"seed_{seed}"
        (run / "confirmation_summary.json").write_text(
            json.dumps(
                _run_summary(
                    "historical_4x128", seed, status="PASS", ess=0.18
                )
            )
        )
    final = finalize_confirmation(tmp_path)

    assert final["status"] == "PASS"
    assert final["ready_for_smc_diversity_benchmark"] is True
    assert final["ready_for_population_prior_update"] is False
    assert final["ready_for_full_catalogue"] is False
    assert (tmp_path / "RWS_RECOVERY_PASS.json").is_file()


def test_submitter_encodes_smoke_pilot_confirmation_and_safe_cache() -> None:
    submitter = (ROOT / "scripts" / "submit_feniks_rws_recovery.sh").read_text()
    worker = (ROOT / "scripts" / "feniks_rws_recovery_pilot_h100.slurm").read_text()

    assert "--array=0-3%4" in submitter
    assert 'afterok:$SMOKE_JOB' in submitter
    assert 'afterok:$PILOT_JOB' in submitter
    assert 'afterok:$PILOT_GATE_JOB' in submitter
    assert 'afterok:$CONFIRM_JOB' in submitter
    assert "--confirmation-objects 2000" in submitter
    assert 'EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"' in worker
    assert 'JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"' in worker
    assert "--freeze-prior" in worker
    assert "--posterior-samples \"$SUPPORT_SAMPLES\"" in worker
    assert "--min-median-ess-fraction 0.05" in worker
    assert "--max-fraction-pareto-k-gt-0p7 0.20" in worker
