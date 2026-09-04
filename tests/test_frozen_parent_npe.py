from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _posterior_manifest(*, cohort: str, draws: int, winner_sha: str | None = None):
    model = {"freeze_receipt": None}
    if winner_sha is not None:
        model["freeze_receipt"] = {"sha256": winner_sha}
    return {
        "cohort": {"sha256": cohort},
        "inference": {"posterior_draws_per_object": draws},
        "model": model,
    }


def _posterior_receipt(*, pit: float, ece: float, ess: float, bad_k: float):
    return {
        "status": "DIAGNOSTIC_COMPLETE",
        "redshift_calibration": {
            "q": {
                "pit_ks_uniform": pit,
                "coverage_ece": ece,
                "coverage_68": 0.55,
                "coverage_95": 0.82,
            }
        },
        "projected_parent_support": {
            "status": "PASS" if ess / 1024 >= 0.05 and bad_k <= 0.2 else "FAIL",
            "median_raw_ess": ess,
            "median_raw_ess_fraction": ess / 1024,
            "fraction_pareto_k_gt_0p7": bad_k,
            "p90_max_raw_weight": 0.5,
        },
        "population_distributions": {"q_aggregate_vs_selected_truth": {}},
        "artifacts": {},
    }


def _write_diagnostic(root: Path, manifest: dict, receipt: dict) -> None:
    root.mkdir(parents=True)
    (root / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def test_sleep_npe_config_freezes_prior_and_excludes_truth() -> None:
    config = load_config(
        ROOT / "configs/experiments/feniks_sc_drws_r29_frozen_parent_sleep_npe.yaml"
    )
    amortized = config["amortized"]

    assert config["truth"]["parameter_columns"] == {}
    assert amortized["prior"]["train_jointly"] is False
    assert amortized["objective"]["mode"] == "reweighted_wake_sleep"
    assert amortized["objective"]["sleep"]["selection"]["candidate_factor"] == 2
    assert amortized["objective"]["wake"]["train_encoder"] is False
    assert amortized["objective"]["wake"]["train_prior"] is False
    assert amortized["training"]["best_checkpoint_metric"] == "validation_sleep_nll"
    assert amortized["training"]["data_parallel"] == "pmap"


def test_prepare_sleep_npe_freezes_cohorts_and_parent(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script("prepare_feniks_sc_drws_frozen_parent_npe.py")
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    inputs = {}
    for name in (
        "parent.eqx",
        "parent.eqx.json",
        "parent.yaml",
        "features.json",
        "train.parquet",
    ):
        inputs[name] = tmp_path / name
        inputs[name].write_text(name, encoding="utf-8")
    fit_rows = tmp_path / "fit.npy"
    validation_rows = tmp_path / "validation.npy"
    np.save(fit_rows, np.arange(12, dtype=np.int64), allow_pickle=False)
    np.save(validation_rows, np.arange(12, 16, dtype=np.int64), allow_pickle=False)
    parent = {
        "checkpoint": str(inputs["parent.eqx"]),
        "checkpoint_sha256": module.sha256_file(inputs["parent.eqx"]),
        "checkpoint_sidecar": str(inputs["parent.eqx.json"]),
        "checkpoint_sidecar_sha256": module.sha256_file(inputs["parent.eqx.json"]),
        "config": str(inputs["parent.yaml"]),
        "config_sha256": module.sha256_file(inputs["parent.yaml"]),
    }
    manifest = {
        "source": {
            "feature_stats": str(inputs["features.json"]),
            "feature_stats_sha256": module.sha256_file(inputs["features.json"]),
        },
        "datasets": {
            "train": {
                "path": str(inputs["train.parquet"]),
                "sha256": module.sha256_file(inputs["train.parquet"]),
            }
        },
        "q_banks": {
            "fit": {
                "cohort_path": str(fit_rows),
                "cohort_sha256": module.sha256_file(fit_rows),
            },
            "validation": {
                "cohort_path": str(validation_rows),
                "cohort_sha256": module.sha256_file(validation_rows),
            },
        },
    }
    (benchmark / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (benchmark / "TRUTH_FREE_ARCHITECTURE_WINNER.json").write_text(
        json.dumps(
            {
                "status": "WINNER_SELECTED",
                "winner": "realnvp_wide",
                "truth_used": False,
                "winner_passes_all_truth_free_distribution_gates": True,
                "winner_passes_nll_non_regression_gate": True,
            }
        ),
        encoding="utf-8",
    )
    (benchmark / "PROJECTION_FIT_COMPLETE.json").write_text(
        json.dumps({"status": "COMPLETE", "truth_used": False, "parent": parent}),
        encoding="utf-8",
    )
    (benchmark / "POPULATION_PROJECTION_COMPLETE.json").write_text(
        json.dumps({"status": "DIAGNOSTIC_COMPLETE"}), encoding="utf-8"
    )
    config_path = tmp_path / "sleep.yaml"
    config_path.write_text("sleep", encoding="utf-8")
    config = {
        "truth": {"parameter_columns": {}},
        "fit": {"free_parameters": {"z_obs": {}}},
        "amortized": {
            "prior": {"train_jointly": False},
            "objective": {
                "mode": "reweighted_wake_sleep",
                "sleep": {"selection": {"candidate_factor": 2}},
            },
        },
    }
    monkeypatch.setattr(module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        module, "latent_spec_from_config", lambda _config: SimpleNamespace(names=("z_obs",))
    )

    out = tmp_path / "npe"
    prepared = module.prepare(
        benchmark_root=benchmark,
        config_path=config_path,
        out=out,
        repo=ROOT,
        epochs=8,
        seed=19,
    )

    assert prepared["status"] == "PREPARED"
    assert prepared["cohorts"]["train"]["objects"] == 12
    assert prepared["cohorts"]["validation"]["objects"] == 4
    assert prepared["training"]["gpus_per_arm"] == 4
    assert prepared["truth_boundary"]["checkpoint_selection"] is False
    assert (out / "STAGE1_PASS.json").is_file()


def test_five_stage_submission_contract_is_parallel_and_distributional() -> None:
    combined = (
        ROOT / "scripts/submit_feniks_sc_drws_frozen_parent_npe_experiments.sh"
    ).read_text(encoding="utf-8")
    baseline = (
        ROOT / "scripts/submit_feniks_sc_drws_full_test_posterior.sh"
    ).read_text(encoding="utf-8")
    npe = (ROOT / "scripts/submit_feniks_sc_drws_frozen_parent_npe.sh").read_text(
        encoding="utf-8"
    )
    worker = (
        ROOT / "scripts/feniks_sc_drws_frozen_parent_npe_train_h100.slurm"
    ).read_text(encoding="utf-8")
    submit_evaluation = (
        ROOT / "scripts/feniks_sc_drws_frozen_parent_npe_submit_evaluation.slurm"
    ).read_text(encoding="utf-8")

    assert combined.index("submit_feniks_sc_drws_full_test_posterior.sh") < combined.index(
        "submit_feniks_sc_drws_frozen_parent_npe.sh"
    )
    assert "POSTERIOR_OBJECTS=4706" in baseline
    assert "POSTERIOR_DRAWS=256" in baseline
    assert "POSTERIOR_OBJECTS=512" in baseline
    assert "POSTERIOR_DRAWS=1024" in baseline
    assert "--array=0-1%2" in npe
    assert "#SBATCH --gres=gpu:4" in worker
    assert "--freeze-prior" in worker
    assert "--n-samples 1" in worker
    assert "--data-parallel pmap" in worker
    assert "afterok:$CLOSURE_DEPENDENCY" in submit_evaluation
    assert "point_estimates_used" in (
        ROOT / "scripts/finalize_feniks_sc_drws_frozen_parent_npe_closure.py"
    ).read_text(encoding="utf-8")


def test_matched_closure_compares_full_and_support_cohorts(tmp_path: Path) -> None:
    module = _load_script("finalize_feniks_sc_drws_frozen_parent_npe_closure.py")
    root = tmp_path / "npe"
    baseline = tmp_path / "baseline"
    full = root / "evaluation/full_test_k256"
    support = root / "evaluation/support_k1024"
    root.mkdir(parents=True)
    winner = {
        "status": "FROZEN",
        "selected_arm": "warm_start",
        "checkpoint_sha256": "checkpoint",
        "validation_sleep_nll": 12.5,
    }
    winner_path = root / "NPE_WINNER_FROZEN.json"
    winner_path.write_text(json.dumps(winner), encoding="utf-8")
    winner_sha = module.sha256_file(winner_path)
    _write_diagnostic(
        baseline / "full_test_k256",
        _posterior_manifest(cohort="full", draws=256),
        _posterior_receipt(pit=0.24, ece=0.16, ess=3.0, bad_k=0.9),
    )
    _write_diagnostic(
        baseline / "support_k1024",
        _posterior_manifest(cohort="support", draws=1024),
        _posterior_receipt(pit=0.24, ece=0.16, ess=8.0, bad_k=0.8),
    )
    _write_diagnostic(
        full,
        _posterior_manifest(cohort="full", draws=256, winner_sha=winner_sha),
        _posterior_receipt(pit=0.04, ece=0.03, ess=20.0, bad_k=0.3),
    )
    _write_diagnostic(
        support,
        _posterior_manifest(cohort="support", draws=1024, winner_sha=winner_sha),
        _posterior_receipt(pit=0.04, ece=0.03, ess=60.0, bad_k=0.1),
    )

    receipt = module.finalize(
        root=root,
        baseline_root=baseline,
        full_root=full,
        support_root=support,
    )

    assert receipt["status"] == "POSTERIOR_TARGET_PASS"
    assert receipt["point_estimates_used"] is False
    assert receipt["truth_used_for_training_or_checkpoint_selection"] is False
    assert receipt["delta_sleep_npe_minus_baseline"]["pit_ks_uniform"] < 0
    assert receipt["delta_sleep_npe_minus_baseline"]["median_raw_ess"] > 0
    assert Path(receipt["artifacts"]["comparison_plot"]).stat().st_size > 0
