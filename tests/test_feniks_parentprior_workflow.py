from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.photometry import abmag_to_fnu_cgs
from scripts.build_feniks_encoder_diagnostic_dataset import _take_stratified
from scripts.build_feniks_parentprior_r25_manifests import build
from scripts.evaluate_feniks_parent_population import (
    _correlations,
    _distribution_frame,
    _population_comparisons,
    _summary,
)
from scripts.validate_feniks_encoder_diagnostic import validate as validate_encoder
from scripts.validate_feniks_parentprior_exact import validate as validate_exact
from scripts.validate_feniks_parentprior_training import validate as validate_training


def _catalog(path: Path, *, rows: int) -> None:
    limit = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    frame = pd.DataFrame(
        {
            "object_id": [f"object-{index}" for index in range(rows)],
            "flux_lsst_r": limit * np.linspace(0.5, 2.0, rows),
            "fluxerr_lsst_r": limit * np.linspace(0.05, 0.2, rows),
            "mask_lsst_r": np.ones(rows, dtype=bool),
            "z_obs": np.linspace(0.0, 3.0, rows),
            "flux_true_lsst_r": 100.0 * limit * np.ones(rows),
        }
    )
    frame.to_parquet(path, index=False)


def test_parentprior_manifests_use_only_observed_r_cut(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    _catalog(train, rows=40)
    _catalog(test, rows=32)
    out = tmp_path / "manifests"
    payload = build(
        train_catalog=train,
        test_catalog=test,
        out=out,
        validation_fraction=0.2,
        n_exact=8,
        seed=17,
    )
    train_rows = np.load(out / "train_indices.npy")
    validation_rows = np.load(out / "validation_indices.npy")
    selected_test_rows = np.load(out / "selected_test_indices.npy")
    exact_rows = np.load(out / "exact_probe_indices.npy")
    frame = pd.read_parquet(train, columns=["flux_lsst_r"])
    limit = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    assert set(train_rows).isdisjoint(set(validation_rows))
    assert np.all(frame.iloc[train_rows]["flux_lsst_r"].to_numpy() > limit)
    assert np.all(frame.iloc[validation_rows]["flux_lsst_r"].to_numpy() > limit)
    test_frame = pd.read_parquet(test, columns=["flux_lsst_r"])
    assert np.all(test_frame.iloc[selected_test_rows]["flux_lsst_r"].to_numpy() > limit)
    assert set(exact_rows).issubset(set(selected_test_rows))
    assert len(exact_rows) == 8
    assert sum(payload["exact_probe_strata"].values()) == 8
    manifest = json.loads((out / "manifest.json").read_text())
    assert "flux_true_lsst_r" not in manifest["observed_columns_read"]
    assert "z_obs" not in manifest["observed_columns_read"]


def test_encoder_diagnostic_stratification_preserves_observed_groups() -> None:
    cohort = pd.DataFrame(
        {
            "example_key": ["easy"] * 4 + ["low_snr"] * 4 + ["near_cut"] * 4,
            "row_index": np.arange(12),
        }
    )
    selected = _take_stratified(cohort, 6)
    assert selected["example_key"].value_counts().to_dict() == {
        "easy": 2,
        "low_snr": 2,
        "near_cut": 2,
    }


def test_population_comparisons_keep_parent_and_selected_contracts_separate() -> None:
    names = ("a", "b")
    values = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 4.0]])
    frames = [
        _distribution_frame(values, names, distribution="parent_prior"),
        _distribution_frame(values, names, distribution="forward_selected_prior"),
        _distribution_frame(values, names, distribution="catalog_inferred_selected"),
        _distribution_frame(values, names, distribution="feniks_truth_parent"),
        _distribution_frame(values, names, distribution="feniks_truth_selected"),
    ]
    combined = pd.concat(frames, ignore_index=True)
    comparisons = _population_comparisons(
        _summary(combined, names),
        _correlations(combined, names),
    ).set_index("comparison")
    assert set(comparisons.index) == {
        "parent_prior_vs_parent_truth",
        "forward_selected_prior_vs_selected_truth",
        "catalog_inferred_vs_selected_truth",
    }
    assert np.allclose(comparisons["median_quantile_l1_over_truth_q90_width"], 0.0)
    assert np.allclose(comparisons["correlation_rmse"], 0.0)


def test_training_validator_rejects_nan_or_missing_prior_updates(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train"
    checkpoints = train / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "best.eqx").write_bytes(b"checkpoint")
    (train / "selection_gradient_preflight.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "forward_finite": True,
                "prior_gradients_finite": True,
            }
        )
    )
    (train / "training_summary.json").write_text(
        json.dumps(
            {
                "train_rows": 8,
                "validation_rows": 2,
                "best_checkpoint_metric": "validation_sleep_nll",
                "effective_latent_spec": {"normalization": "spline15d_mixed"},
                "selection_correction": {"enabled": True},
                "sleep": {"error_model": "observed_catalog"},
                "wake": {"train_encoder": False, "train_prior": True},
                "best_loss": 1.0,
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifests": {
                    "train": {"count": 8},
                    "validation": {"count": 2},
                }
            }
        )
    )
    rows = [
        {
            "split": "train",
            "update_phase": "encoder_sleep",
            "epoch": 24,
            "prior_raw_grad_norm": 0.0,
            "encoder_raw_grad_norm": 1.0,
            "encoder_grad_clipped_fraction": 0.0,
            "posterior_full_entropy_mc": 2.0,
            "grads_finite": 1.0,
            "update_applied": 1.0,
        },
        {
            "split": "train",
            "update_phase": "prior_wake",
            "epoch": 25,
            "prior_raw_grad_norm": 1.0,
            "encoder_raw_grad_norm": 0.0,
            "wake_prior_update_applied": 0.0,
            "wake_median_ess_fraction": 0.05,
            "selection/evaluated": 0.0,
            "selection/alpha": 1.0,
            "selection/alpha_mc_relative_error": 0.0,
            "grads_finite": 1.0,
            "update_applied": 0.0,
        },
    ]
    pd.DataFrame(rows).to_csv(train / "training_log.csv", index=False)
    payload = validate_training(train=train, manifest=manifest, smoke=False)
    assert payload["status"] == "FAIL"
    assert payload["checks"]["wake_prior_update_applied"] is False

    rows[1].update(
        {
            "wake_prior_update_applied": 1.0,
            "selection/evaluated": 1.0,
            "selection/alpha": 0.4,
            "selection/alpha_mc_relative_error": 0.05,
            "update_applied": 1.0,
        }
    )
    pd.DataFrame(rows).to_csv(train / "training_log.csv", index=False)
    payload = validate_training(train=train, manifest=manifest, smoke=False)
    assert payload["status"] == "PASS"

    rows[1]["grads_finite"] = 0.0
    pd.DataFrame(rows).to_csv(train / "training_log.csv", index=False)
    payload = validate_training(train=train, manifest=manifest, smoke=False)
    assert payload["status"] == "FAIL"
    assert payload["checks"]["all_training_gradients_finite"] is False


def test_recovery_launcher_preserves_epoch24_state_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "scripts/submit_feniks_parentprior_sleepnpe_recovery.sh"
    ).read_text()
    slurm = (root / "scripts/feniks_parentprior_sleepnpe_h100.slurm").read_text()
    assert "checkpoints/epoch_0024.eqx" in launcher
    assert 'START_EPOCH="${START_EPOCH:-25}"' in launcher
    assert "SOURCE_SMOKE_MANIFEST_ROOT" in launcher
    assert "--initial-checkpoint" in slurm
    assert "--fixed-feature-stats" in slurm
    assert "optimizer_state_resumed=0" in slurm


def test_exact_finalizer_requires_is_target_and_population_gates(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exact"
    galaxy = root / "galaxies/00_typical_row7"
    population = root / "population"
    tarp = root / "calibration/tarp"
    mira = root / "calibration/mira"
    galaxy.mkdir(parents=True)
    population.mkdir()
    tarp.mkdir(parents=True)
    mira.mkdir(parents=True)
    pd.DataFrame(
        {
            "importance_raw_ess_fraction": [0.20],
            "importance_pareto_k": [0.4],
            "defensive_importance_raw_ess_fraction": [0.30],
            "defensive_importance_pareto_k": [0.3],
        }
    ).to_parquet(root / "scoreboard.parquet", index=False)
    pd.DataFrame([{"order": 0, "example_key": "typical", "row_index": 7}]).to_parquet(
        root / "cohort.parquet", index=False
    )
    (root / "benchmark_summary.json").write_text(
        json.dumps({"all_nuts_rhat_pass": True})
    )
    (galaxy / "fit_bounds_diagnostics.json").write_text(
        json.dumps({"nuts": {"fraction_of_samples_outside_fit_bounds": 0.0}})
    )
    (galaxy / "posterior_geometry_diagnostics.json").write_text(
        json.dumps(
            {
                "encoder": {"generalized_variance_ratio_max": 1.2},
                "defensive_encoder": {"generalized_variance_ratio_max": 1.1},
            }
        )
    )
    (population / "population_summary.json").write_text(
        json.dumps(
            {
                "selection": {
                    "alpha_mc": 0.4,
                    "prior_physical_valid_fraction": 1.0,
                }
            }
        )
    )
    pd.DataFrame(
        {
            "comparison": [
                "parent_prior_vs_parent_truth",
                "forward_selected_prior_vs_selected_truth",
                "catalog_inferred_vs_selected_truth",
            ],
            "median_quantile_l1_over_truth_q90_width": [0.1, 0.1, 0.1],
            "correlation_rmse": [0.1, 0.1, 0.1],
            "min_std_ratio": [0.8, 0.8, 0.8],
        }
    ).to_csv(population / "population_comparisons.csv", index=False)
    models = ["q", "q_is", "defensive_is", "nuts"]
    pd.DataFrame(
        {
            "model": models,
            "group": ["full_15d"] * len(models),
            "coverage_rmse": [0.08, 0.07, 0.06, 0.05],
            "coverage_max_abs_error": [0.16, 0.14, 0.12, 0.10],
        }
    ).to_csv(tarp / "tarp_summary.csv", index=False)
    pd.DataFrame(
        {
            "model": models,
            "group": ["full_15d"] * len(models),
            "score": [0.64, 0.65, 0.66, 2.0 / 3.0],
            "delta_from_ideal": [
                0.64 - 2.0 / 3.0,
                0.65 - 2.0 / 3.0,
                0.66 - 2.0 / 3.0,
                0.0,
            ],
            "theoretical_sigma": [0.04] * len(models),
        }
    ).to_csv(mira / "mira_scores.csv", index=False)
    payload = validate_exact(root=root)
    assert payload["status"] == "PASS"
    assert payload["ready_for_production"] is True

    failed_tarp = pd.read_csv(tarp / "tarp_summary.csv")
    failed_tarp.loc[failed_tarp["model"].eq("q"), "coverage_rmse"] = 0.50
    failed_tarp.to_csv(tarp / "tarp_summary.csv", index=False)
    payload = validate_exact(root=root)
    assert payload["status"] == "FAIL"
    assert payload["checks"]["q_tarp_close_to_nuts"] is False


def test_encoder_diagnostic_separates_sleep_and_observed_closure(
    tmp_path: Path,
) -> None:
    root = tmp_path / "diagnostic"
    root.mkdir()
    cohort_rows = []
    score_rows = []
    for order, domain in enumerate(("observed_catalog", "sleep_synthetic")):
        example = f"example_{order}"
        cohort_rows.append(
            {
                "order": order,
                "example_key": example,
                "row_index": order,
                "object_id": f"object-{order}",
                "domain": domain,
            }
        )
        score_rows.append(
            {
                "domain": domain,
                "nuts_max_rhat": 1.0,
                "importance_raw_ess_fraction": 0.01 if order == 0 else 0.20,
                "importance_pareto_k": 1.0 if order == 0 else 0.2,
                "defensive_importance_raw_ess_fraction": (0.02 if order == 0 else 0.30),
                "defensive_importance_pareto_k": 0.9 if order == 0 else 0.1,
            }
        )
        galaxy = root / "galaxies" / f"{order:02d}_{example}_row{order}"
        galaxy.mkdir(parents=True)
        (galaxy / "fit_bounds_diagnostics.json").write_text(
            json.dumps({"nuts": {"fraction_of_samples_outside_fit_bounds": 0.0}})
        )
        (galaxy / "posterior_geometry_diagnostics.json").write_text(
            json.dumps(
                {
                    "encoder": {"generalized_variance_ratio_max": 1.5},
                    "defensive_encoder": {"generalized_variance_ratio_max": 1.2},
                }
            )
        )
    pd.DataFrame(cohort_rows).to_parquet(root / "cohort.parquet", index=False)
    pd.DataFrame(score_rows).to_parquet(root / "scoreboard.parquet", index=False)
    agreement = []
    for domain in ("observed_catalog", "sleep_synthetic"):
        for method in ("Encoder", "Encoder + IS", "Defensive + IS"):
            agreement.append(
                {
                    "domain": domain,
                    "method": method,
                    "wasserstein_to_nuts_in_nuts_std": 0.2,
                    "std_ratio_to_nuts": 1.0,
                    "nuts_standardized_mean_offset": 0.1,
                }
            )
    pd.DataFrame(agreement).to_parquet(
        root / "posterior_agreement.parquet", index=False
    )
    (root / "contract.json").write_text(
        json.dumps({"analysis_contract": "ENCODER_DIAGNOSTIC_ONLY"})
    )
    models = ["q", "q_is", "defensive_is", "nuts"]
    for domain in ("observed_catalog", "sleep_synthetic"):
        tarp = root / f"calibration_{domain}/tarp"
        mira = root / f"calibration_{domain}/mira"
        tarp.mkdir(parents=True)
        mira.mkdir(parents=True)
        pd.DataFrame(
            {
                "model": models,
                "group": ["full_15d"] * 4,
                "coverage_rmse": [0.1] * 4,
                "coverage_max_abs_error": [0.2] * 4,
            }
        ).to_csv(tarp / "tarp_summary.csv", index=False)
        pd.DataFrame(
            {
                "model": models,
                "group": ["full_15d"] * 4,
                "score": [2 / 3] * 4,
                "delta_from_ideal": [0.0] * 4,
            }
        ).to_csv(mira / "mira_scores.csv", index=False)
    payload = validate_encoder(root=root)
    assert payload["status"] == "complete"
    assert payload["scientific_diagnosis"] == "SIMULATION_TO_OBSERVATION_GAP"
    assert payload["ready_for_prior_promotion"] is False
    assert (
        payload["domains"]["sleep_synthetic"]["q_only_importance"]["status"] == "PASS"
    )


def test_encoder_diagnostic_launcher_is_not_blocked_by_prior_validation() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "scripts/submit_feniks_encoder_diagnostic_exact.sh").read_text()
    exact = (root / "scripts/feniks_parentprior_exact_h100.slurm").read_text()
    prepare = (
        root / "scripts/feniks_encoder_diagnostic_prepare_h100.slurm"
    ).read_text()
    assert "parentprior_training_validation.json" not in launcher
    assert "prior PASS is not required" in launcher
    assert "ENCODER_DIAGNOSTIC_ONLY=1" in launcher
    assert '"${ENCODER_DIAGNOSTIC_ONLY:-0}" != "1"' in exact
    assert "EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD=0" in prepare
