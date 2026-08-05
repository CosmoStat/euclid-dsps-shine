from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_dashboard():
    path = ROOT / "scripts/build_popcosmos_publication_dashboard.py"
    spec = importlib.util.spec_from_file_location("popcosmos_dashboard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timing(*, predictive: float) -> dict[str, object]:
    return {
        "status": "complete",
        "backend": "gpu",
        "n_objects": 100,
        "posterior_samples": 128,
        "feature_construction_seconds": 2.0,
        "steady_state": {
            "encoder_only": {"median_seconds_per_object": 2.0e-6},
            "posterior_draws": {"median_seconds_per_object": 8.0e-6},
            "posterior_predictive": {
                "median_seconds_per_object": predictive,
            },
        },
    }


def _write_feniks_metrics(path: Path, *, offset: float) -> None:
    path.mkdir()
    pd.DataFrame(
        {
            "n_objects": [5000],
            "median_bias": [0.01 + offset],
            "sigma_mad": [0.05 + offset],
            "rmse": [0.13 + offset],
            "outlier_fraction_0p15": [0.12 + offset],
            "coverage_68": [0.65 - offset],
        }
    ).to_csv(path / "photoz_metrics.csv", index=False)


def test_complete_dashboard_writes_all_artifacts(tmp_path: Path) -> None:
    dashboard = _load_dashboard()
    feniks_paths = [tmp_path / f"feniks_{index}" for index in range(3)]
    for index, path in enumerate(feniks_paths):
        _write_feniks_metrics(path, offset=0.01 * index)
    matched = tmp_path / "matched"
    matched.mkdir()
    pd.DataFrame(
        [
            {
                "method": method,
                "n_spec": 1395,
                "median_bias": value,
                "nmad": 0.02 + value,
                "rmse": 0.20 + value,
                "outlier_fraction_0p15": 0.07 + value,
                "coverage_68": 0.48 - value,
            }
            for method, value in (("rws26", 0.001), ("rws24", 0.002), ("popcosmos", 0.0))
        ]
    ).to_csv(matched / "redshift_method_metrics.csv", index=False)
    accuracy = dashboard._read_accuracy_metrics(
        feniks_rws=feniks_paths[0],
        feniks_rws_mixture=feniks_paths[1],
        feniks_smcwake=feniks_paths[2],
        matched_metrics=matched,
    )
    assert accuracy["method"].tolist() == [
        "feniks_rws_k8",
        "feniks_rws_mix_k8",
        "feniks_smcwake_k4",
        "rws26",
        "rws24",
        "popcosmos",
        "popcosmos_paper",
    ]

    calibration = pd.DataFrame(
        [
            {
                "context": context,
                "model": model,
                "mira_score": 0.66,
                "mira_bootstrap_q025": 0.64,
                "mira_bootstrap_q975": 0.68,
                "tarp_atc": 0.01,
                "tarp_bootstrap_atc_q025": 0.0,
                "tarp_bootstrap_atc_q975": 0.02,
            }
            for context, model in sorted(dashboard.EXPECTED_CALIBRATION_RUNS)
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "context": context,
                "model": model,
                "alpha": alpha,
                "ecp": alpha,
                "bootstrap_q025": max(0.0, alpha - 0.02),
                "bootstrap_q975": min(1.0, alpha + 0.02),
            }
            for context, model in sorted(dashboard.EXPECTED_CALIBRATION_RUNS)
            for alpha in (0.0, 0.5, 1.0)
        ]
    )
    speed = dashboard._speed_summary(
        _timing(predictive=0.28),
        _timing(predictive=0.26),
        popcosmos_gpu_seconds=15.0,
    )
    assert speed["variants"]["rws26"]["popcosmos_throughput_ratio"] > 49.0

    plot = tmp_path / "dashboard.png"
    dashboard._write_dashboard(
        accuracy=accuracy,
        calibration=calibration,
        tarp_coverage=coverage,
        speed=speed,
        path=plot,
    )
    assert plot.is_file() and plot.stat().st_size > 0
    assert plot.with_suffix(".pdf").is_file()
