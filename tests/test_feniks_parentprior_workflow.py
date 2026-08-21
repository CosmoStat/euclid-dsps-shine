from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.photometry import abmag_to_fnu_cgs
from scripts.build_feniks_parentprior_r25_manifests import build
from scripts.evaluate_feniks_parent_population import (
    _correlations,
    _distribution_frame,
    _population_comparisons,
    _summary,
)
from scripts.validate_feniks_parentprior_exact import validate as validate_exact


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
