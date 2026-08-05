from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.mira import evaluate_feniks_mira
from euclid_dsps.amortized.tarp import evaluate_feniks_tarp

ROOT = Path(__file__).resolve().parents[1]


def _load_comparator():
    path = ROOT / "scripts/compare_redshift_calibration_runs.py"
    spec = importlib.util.spec_from_file_location(
        "redshift_calibration_comparator", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sparse_redshift_inference(
    path: Path,
    *,
    posterior_offset: float,
) -> None:
    object_ids = np.arange(800, 808)
    truth_values = np.asarray([np.nan, 0.3, np.nan, 0.8, 1.1, np.nan, np.nan, 1.7])
    truth = pd.DataFrame(
        {
            "object_id": object_ids,
            "row_index": np.arange(100, 108),
            "redshift_true": truth_values,
        }
    )
    path.mkdir(parents=True)
    truth.to_parquet(path / "inference_truth.parquet", index=False)

    n_samples = 8
    centers = np.where(np.isfinite(truth_values), truth_values, 0.9)
    rng = np.random.default_rng(42)
    samples = pd.DataFrame(
        {
            "object_id": np.repeat(object_ids, n_samples),
            "row_index": np.repeat(np.arange(100, 108), n_samples),
            "sample_id": np.tile(np.arange(n_samples), len(object_ids)),
            "z_obs": np.repeat(centers + posterior_offset, n_samples)
            + rng.normal(0.0, 0.08, len(object_ids) * n_samples),
        }
    )
    samples.to_parquet(path / "posterior_samples.parquet", index=False)


def test_sparse_mapped_redshift_truth_runs_paired_mira_and_tarp(
    tmp_path: Path,
) -> None:
    rws26 = tmp_path / "rws26"
    rws24 = tmp_path / "rws24"
    _write_sparse_redshift_inference(rws26, posterior_offset=0.0)
    _write_sparse_redshift_inference(rws24, posterior_offset=0.03)
    posterior_specs = [("rws26", rws26), ("rws24", rws24)]
    shared = {
        "truth_path": rws26 / "inference_truth.parquet",
        "posterior_specs": posterior_specs,
        "samples_per_object": 8,
        "parameters": ("z_obs",),
        "truth_column_map": {"z_obs": "redshift_true"},
        "drop_nonfinite_truth": True,
        "seed": 17,
    }

    mira_out = tmp_path / "mira"
    mira = evaluate_feniks_mira(
        **shared,
        out_dir=mira_out,
        num_regions=6,
        num_bootstrap=12,
    )
    tarp_out = tmp_path / "tarp"
    tarp = evaluate_feniks_tarp(
        **shared,
        out_dir=tarp_out,
        num_alpha_bins=4,
        num_bootstrap=12,
    )

    for summary in (mira, tarp):
        assert summary["status"] == "complete"
        assert summary["num_objects"] == 4
        assert summary["num_posterior_samples"] == 8
        assert summary["selected_sample_ids"] == list(range(8))
        assert summary["score_groups"] == ["marginal_z_obs"]
        assert summary["primary_group"] == "marginal_z_obs"
        assert {row["model"] for row in summary["primary"]} == {"rws26", "rws24"}
        assert summary["full_15d"] == []

    contributions = pd.read_parquet(mira_out / "mira_object_contributions.parquet")
    assert set(contributions["object_id"]) == {801, 803, 804, 807}
    assert len(pd.read_csv(mira_out / "mira_pairwise_differences.csv")) == 1
    assert len(pd.read_csv(tarp_out / "tarp_pairwise_differences.csv")) == 1
    manifest = json.loads((mira_out / "mira_manifest.json").read_text())
    assert manifest["truth_selection"] == {
        "column_map": {"z_obs": "redshift_true"},
        "drop_nonfinite_parameters": True,
        "selected_objects": 4,
    }


def test_cross_cohort_comparator_reads_redshift_rows_and_writes_plot(
    tmp_path: Path,
) -> None:
    comparator = _load_comparator()
    mira = pd.DataFrame(
        {
            "model": ["rws26", "rws24"],
            "group": ["marginal_z_obs", "marginal_z_obs"],
            "num_objects": [100, 100],
            "num_posterior_samples": [128, 128],
            "score": [0.66, 0.64],
            "ideal_score": [2 / 3, 2 / 3],
            "bootstrap_mean": [0.661, 0.641],
            "bootstrap_std": [0.01, 0.02],
            "bootstrap_q025": [0.64, 0.60],
            "bootstrap_q975": [0.68, 0.68],
        }
    )
    tarp = pd.DataFrame(
        {
            "model": ["rws26", "rws24"],
            "group": ["marginal_z_obs", "marginal_z_obs"],
            "num_objects": [100, 100],
            "num_posterior_samples": [128, 128],
            "atc": [0.01, -0.02],
            "ks_pvalue": [0.4, 0.3],
            "coverage_rmse": [0.03, 0.04],
            "coverage_max_abs_error": [0.06, 0.08],
            "bootstrap_atc_mean": [0.011, -0.019],
            "bootstrap_atc_std": [0.003, 0.004],
            "bootstrap_atc_q025": [0.005, -0.027],
            "bootstrap_atc_q975": [0.017, -0.011],
        }
    )
    mira_path = tmp_path / "mira_scores.csv"
    tarp_path = tmp_path / "tarp_summary.csv"
    coverage_path = tmp_path / "tarp_coverage.csv"
    mira.to_csv(mira_path, index=False)
    tarp.to_csv(tarp_path, index=False)
    coverage = pd.DataFrame(
        [
            {
                "model": model,
                "group": "marginal_z_obs",
                "alpha": alpha,
                "ecp": alpha + offset,
                "bootstrap_q025": max(0.0, alpha + offset - 0.02),
                "bootstrap_q975": min(1.0, alpha + offset + 0.02),
            }
            for model, offset in [("rws26", 0.01), ("rws24", -0.01)]
            for alpha in [0.0, 0.5, 1.0]
        ]
    )
    coverage.to_csv(coverage_path, index=False)

    frame = comparator._read_context(
        context="cosmos_public_specz",
        truth_scope="test cohort",
        mira_path=mira_path,
        tarp_path=tarp_path,
    )
    assert frame["model"].tolist() == ["rws26", "rws24"]
    assert frame["mira_score"].tolist() == [0.66, 0.64]
    assert frame["tarp_atc"].tolist() == [0.01, -0.02]
    coverage = comparator._read_tarp_coverage(
        coverage_path, context="cosmos_public_specz"
    )
    plot = tmp_path / "comparison.png"
    comparator._write_plot(frame, plot, tarp_coverage=coverage)
    assert plot.is_file() and plot.stat().st_size > 0
    assert plot.with_suffix(".pdf").is_file()


def test_dashboard_contract_requires_all_four_redshift_runs() -> None:
    comparator = _load_comparator()
    runs = [
        ("cosmos_public_specz", "rws26"),
        ("cosmos_public_specz", "rws24"),
        ("feniks_synthetic", "rws_k8_t2_seed2"),
        ("feniks_synthetic", "rws_k8_t2_seed3"),
    ]
    frame = pd.DataFrame(
        {
            "context": [context for context, _ in runs],
            "model": [model for _, model in runs],
            "mira_score": [0.63, 0.62, 0.67, 0.66],
            "tarp_atc": [-0.03, -0.05, 0.01, 0.0],
        }
    )
    coverage = pd.DataFrame(
        [
            {"context": context, "model": model, "alpha": alpha}
            for context, model in runs
            for alpha in (0.0, 0.5, 1.0)
        ]
    )
    comparator._validate_dashboard_contract(frame, coverage)

    with np.testing.assert_raises_regex(ValueError, "run contract mismatch"):
        comparator._validate_dashboard_contract(frame.iloc[:-1], coverage)
