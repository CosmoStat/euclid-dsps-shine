from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.diagnostics import _write_multi_overlay_corner_plot

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observed_flux_cohort_and_panels_do_not_read_truth(tmp_path: Path) -> None:
    prepare = _load_script("prepare_feniks_sc_drws_population_posterior.py")
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(
        {
            "object_id": np.arange(1000, 1200),
            "flux_lsst_r": rng.lognormal(size=200),
            "z_obs": rng.normal(size=200),
        }
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    frame.to_parquet(first, index=False)
    changed = frame.copy()
    changed["z_obs"] = changed["z_obs"].sample(frac=1.0, random_state=9).to_numpy()
    changed.to_parquet(second, index=False)
    selected = np.arange(5, 195, dtype=np.int64)

    left, left_panels = prepare._select_observed_flux_quantiles(
        first,
        selected,
        flux_column="flux_lsst_r",
        id_column="object_id",
        objects=64,
        panels=8,
    )
    right, right_panels = prepare._select_observed_flux_quantiles(
        second,
        selected,
        flux_column="flux_lsst_r",
        id_column="object_id",
        objects=64,
        panels=8,
    )

    assert left["row_index"].is_unique
    assert left["panel"].sum() == 8
    assert left["observed_flux_rank_fraction"].is_monotonic_increasing
    assert left["row_index"].tolist() == right["row_index"].tolist()
    assert np.array_equal(left_panels, right_panels)
    assert "z_obs" not in left


def test_prepare_freezes_truth_free_inference_and_observed_cohort(
    tmp_path: Path, monkeypatch
) -> None:
    prepare = _load_script("prepare_feniks_sc_drws_population_posterior.py")
    benchmark = tmp_path / "benchmark"
    calibration = tmp_path / "calibration"
    benchmark.mkdir()
    calibration.mkdir()
    dataset = tmp_path / "test.parquet"
    pd.DataFrame(
        {
            "object_id": np.arange(100),
            "flux_lsst_r": np.linspace(1.0, 100.0, 100),
            "z_truth": np.linspace(0.0, 2.0, 100),
        }
    ).to_parquet(dataset, index=False)
    selected_rows = calibration / "selected.npy"
    np.save(selected_rows, np.arange(10, 90, dtype=np.int64), allow_pickle=False)
    calibration_manifest = calibration / "RUN_MANIFEST.json"
    calibration_manifest.write_text(
        json.dumps(
            {
                "banks": {
                    "q_evaluation": {
                        "cohort_path": str(selected_rows),
                        "cohort_sha256": prepare.sha256_file(selected_rows),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    files = {}
    for name in (
        "parent.eqx",
        "parent.eqx.json",
        "source.eqx",
        "source.eqx.json",
        "features.json",
        "candidate.yaml",
        "source.yaml",
        "truth.yaml",
    ):
        files[name] = tmp_path / name
        files[name].write_text(name, encoding="utf-8")
    source = {
        "checkpoint": str(files["source.eqx"]),
        "checkpoint_sha256": prepare.sha256_file(files["source.eqx"]),
        "checkpoint_sidecar": str(files["source.eqx.json"]),
        "checkpoint_sidecar_sha256": prepare.sha256_file(files["source.eqx.json"]),
        "feature_stats": str(files["features.json"]),
        "feature_stats_sha256": prepare.sha256_file(files["features.json"]),
        "calibration_manifest": str(calibration_manifest),
        "calibration_manifest_sha256": prepare.sha256_file(calibration_manifest),
    }
    parent = {
        "checkpoint": str(files["parent.eqx"]),
        "checkpoint_sha256": prepare.sha256_file(files["parent.eqx"]),
        "checkpoint_sidecar": str(files["parent.eqx.json"]),
        "checkpoint_sidecar_sha256": prepare.sha256_file(files["parent.eqx.json"]),
        "config": str(files["candidate.yaml"]),
        "config_sha256": prepare.sha256_file(files["candidate.yaml"]),
    }
    manifest = {
        "source": source,
        "config": {
            "path": str(files["source.yaml"]),
            "sha256": prepare.sha256_file(files["source.yaml"]),
        },
        "truth_config": {
            "path": str(files["truth.yaml"]),
            "sha256": prepare.sha256_file(files["truth.yaml"]),
        },
        "datasets": {
            "test": {
                "path": str(dataset),
                "sha256": prepare.sha256_file(dataset),
                "selected_objects": 80,
            }
        },
    }
    (benchmark / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (benchmark / "PROJECTION_FIT_COMPLETE.json").write_text(
        json.dumps({"status": "COMPLETE", "truth_used": False, "parent": parent}),
        encoding="utf-8",
    )
    (benchmark / "TRUTH_FREE_ARCHITECTURE_WINNER.json").write_text(
        json.dumps(
            {
                "status": "WINNER_SELECTED",
                "truth_used": False,
                "winner": "realnvp_wide",
                "winner_passes_all_truth_free_distribution_gates": True,
                "winner_passes_nll_non_regression_gate": True,
                "artifacts": {"parent_checkpoint": str(files["parent.eqx"])},
            }
        ),
        encoding="utf-8",
    )
    (benchmark / "POPULATION_PROJECTION_COMPLETE.json").write_text(
        json.dumps({"status": "DIAGNOSTIC_COMPLETE"}), encoding="utf-8"
    )

    base_config = {
        "catalog_path": str(dataset),
        "bands": [{"name": "lsst_r", "column": "flux_lsst_r", "units": "nJy"}],
        "dataset": {"id_column": "object_id"},
        "truth": {"parameter_columns": {"z_obs": "z_truth"}},
        "fit": {"free_parameters": {"z_obs": {}}},
        "amortized": {"inference": {}},
    }
    monkeypatch.setattr(
        prepare,
        "load_config",
        lambda _path: json.loads(json.dumps(base_config)),
    )
    output = tmp_path / "diagnostic"
    frozen = prepare.prepare(
        benchmark_root=benchmark,
        out=output,
        repo=ROOT,
        objects=8,
        shards=2,
        panels=4,
        posterior_draws=1024,
        resample_draws=256,
    )

    inference = yaml.safe_load(
        (output / "inference_config.yaml").read_text(encoding="utf-8")
    )
    assert frozen["cohort"]["truth_used"] is False
    assert frozen["truth_boundary"]["final_closure_and_plot_overlay"] is True
    assert len(frozen["cohort"]["shards"]) == 2
    assert inference["truth"]["parameter_columns"] == {}


def test_support_gate_uses_ess_pareto_and_maximum_weight() -> None:
    finalize = _load_script("finalize_feniks_sc_drws_population_posterior.py")
    passing = pd.DataFrame(
        {
            "raw_ess": [70.0, 80.0, 90.0],
            "raw_ess_fraction": [0.068, 0.078, 0.088],
            "pareto_k": [0.2, 0.3, 0.4],
            "max_raw_weight": [0.2, 0.3, 0.4],
        }
    )
    failing = passing.copy()
    failing["max_raw_weight"] = [0.85, 0.90, 0.95]

    assert finalize._support_summary(passing)["status"] == "PASS"
    assert finalize._support_summary(failing)["status"] == "FAIL"


def test_corner_supports_q_iw_prior_truth_overlay(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(8)
    columns = ["z_obs", "log10_stellar_mass"]
    q = pd.DataFrame(rng.normal(size=(128, 2)), columns=columns)
    iw = pd.DataFrame(rng.normal(0.2, 0.8, size=(96, 2)), columns=columns)
    prior = pd.DataFrame(rng.normal(size=(256, 2)), columns=columns)
    truth = pd.DataFrame([[0.1, -0.2]], columns=columns)
    config = {"fit": {"free_parameters": {name: {} for name in columns}}}

    path = _write_multi_overlay_corner_plot(
        q,
        tmp_path,
        plt,
        truth=truth,
        prior=prior,
        filename="overlay.png",
        title="q / IW / prior / truth",
        posterior_label="q",
        config=config,
        additional_overlays=[
            {"key": "iw", "label": "PSIS-IW", "frame": iw, "color": "#009E73"}
        ],
    )

    assert path is not None and path.stat().st_size > 0
    metadata = pd.read_csv(tmp_path / "overlay_columns.csv")
    assert metadata["iw_finite_rows"].eq(len(iw)).all()


def test_population_posterior_slurm_contract_is_truth_free_and_parallel() -> None:
    worker = (
        ROOT / "scripts/feniks_sc_drws_population_posterior_h100.slurm"
    ).read_text(encoding="utf-8")
    submit = (ROOT / "scripts/submit_feniks_sc_drws_population_posterior.sh").read_text(
        encoding="utf-8"
    )
    finalizer = (
        ROOT / "scripts/finalize_feniks_sc_drws_population_posterior.py"
    ).read_text(encoding="utf-8")

    assert '--posterior-samples "$POSTERIOR_DRAWS"' in worker
    assert "projected_parent_iw" in worker and "source_prior_iw" in worker
    assert "worker inference config must be truth-free" in worker
    assert '--array="0-${LAST_TASK}%${POSTERIOR_MAX_PARALLEL}"' in submit
    assert '--dependency="afterok:$INFERENCE_JOB"' in submit
    assert '"truth_used_for_inference_or_support": False' in finalizer
    assert '"truth_used_for_final_closure": True' in finalizer
