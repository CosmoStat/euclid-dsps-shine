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
    FULL_SEEDS,
    SEEDS,
    finalize_confirmation,
    finalize_full,
    predictive_metrics,
    select_pilot,
    summarize_candidate,
    support_tail_metrics,
    truth_free_inference_contract,
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
        final_validation_objects=4,
        seed=9,
    )
    training = np.load(out / "train_indices.npy")
    pilot_training = np.load(out / "pilot_train_indices.npy")
    confirmation_training = np.load(out / "confirmation_train_indices.npy")
    validation = np.load(out / "validation_indices.npy")
    pilot = np.load(out / "pilot_indices.npy")
    confirmation = np.load(out / "confirmation_indices.npy")
    final_validation = np.load(out / "final_validation_indices.npy")
    threshold = float(np.asarray(abmag_to_fnu_cgs(29.0)))
    train_frame = pd.read_parquet(train, columns=["flux_lsst_r"])
    test_frame = pd.read_parquet(test, columns=["flux_lsst_r"])

    assert set(training).isdisjoint(set(validation))
    assert set(pilot_training).isdisjoint(set(confirmation_training))
    assert len(pilot_training) == 6
    assert len(confirmation_training) == 10
    assert set(pilot).isdisjoint(set(confirmation))
    assert set(pilot).isdisjoint(set(final_validation))
    assert set(confirmation).isdisjoint(set(final_validation))
    assert np.all(train_frame.iloc[training].flux_lsst_r.to_numpy() > threshold)
    assert np.all(train_frame.iloc[validation].flux_lsst_r.to_numpy() > threshold)
    assert np.all(test_frame.iloc[pilot].flux_lsst_r.to_numpy() > threshold)
    assert np.all(test_frame.iloc[confirmation].flux_lsst_r.to_numpy() > threshold)
    assert receipt["truth_columns_requested"] == []
    assert receipt["truth_used_for_training_or_checkpoint_selection"] is False
    assert receipt["selection"]["max_mag_ab"] == 29.0
    assert set(receipt["selection"]["retention_grid"]["train"]) == {
        "25", "26", "27", "27.5", "28", "28.5", "29"
    }
    assert receipt["c0_scope_statement"].endswith(
        "additional observed r<29.0 selection."
    )


def test_manifest_fails_closed_and_recommends_next_fainter_grid_cut(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    faint = float(np.asarray(abmag_to_fnu_cgs(27.75)))
    bright = float(np.asarray(abmag_to_fnu_cgs(26.0)))
    values = np.asarray([bright] * 16 + [faint] * 4)
    pd.DataFrame({"flux_lsst_r": values}).to_parquet(train)
    pd.DataFrame({"flux_lsst_r": np.tile(values, 2)}).to_parquet(test)
    with pytest.raises(ValueError, match="recommend max_mag_ab=28"):
        build(
            train_catalog=train,
            test_catalog=test,
            out=tmp_path / "manifests",
            validation_objects=2,
            pilot_objects=2,
            confirmation_objects=2,
            final_validation_objects=2,
            seed=1,
            max_mag_ab=27.5,
        )


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


def _write_sc_variant(
    root: Path,
    *,
    score: float,
    support_status: str = "PASS",
    ess_fraction: float = 0.10,
    max_weight: float = 0.10,
    pareto_fraction: float = 0.0,
    predictive_scale: float = 1.0,
) -> tuple[Path, Path]:
    support = root / "support"
    importance = root / "importance"
    predictive = root / "predictive"
    support.mkdir(parents=True)
    importance.mkdir(parents=True)
    predictive.mkdir(parents=True)
    truth_free_summary = {
        "complete": True,
        "truth_snapshot_enabled": False,
        "truth_diagnostics_enabled": False,
        "truth_snapshot_rows": 0,
        "truth_used_for_inference_or_checkpoint_selection": False,
    }
    (support / "inference_summary.json").write_text(json.dumps(truth_free_summary))
    (predictive / "inference_summary.json").write_text(json.dumps(truth_free_summary))
    pd.DataFrame(
        {
            "n_proposal_samples": [2048] * 4,
            "n_finite_logweights": [2048] * 4,
            "raw_ess_fraction": [ess_fraction] * 4,
            "max_raw_weight": [max_weight] * 4,
        }
    ).to_parquet(importance / "importance_diagnostics.parquet")
    (importance / "importance_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "support_gate": {
                    "status": support_status,
                    "mean_log_evidence_is": score,
                },
                "n_objects": 4,
                "n_joint_draws": 8192,
                "median_raw_ess_fraction": ess_fraction,
                "fraction_pareto_k_gt_0p7": pareto_fraction,
                "fraction_pareto_k_gt_1": pareto_fraction,
                "inputs": {
                    "posterior_inference_summary": {
                        "path": str(support / "inference_summary.json")
                    }
                },
            }
        )
    )
    pd.DataFrame(
        {
            "row_index": [0],
            "band": ["r"],
            "obs_flux_fnu_cgs": [0.0],
            "obs_err_fnu_cgs": [1.0],
            "valid": [True],
        }
    ).to_parquet(predictive / "posterior_predictive_residual_summary.parquet")
    pd.DataFrame(
        {
            "row_index": [0, 0],
            "sample_id": [0, 1],
            "band": ["r", "r"],
            "model_flux_fnu_cgs": [-predictive_scale, predictive_scale],
        }
    ).to_parquet(predictive / "posterior_predictive_flux.parquet")
    return importance, predictive


def test_truth_free_inference_contract_rejects_snapshot_or_missing_receipt(
    tmp_path: Path,
) -> None:
    assert truth_free_inference_contract(tmp_path)["status"] == "FAIL"
    (tmp_path / "inference_summary.json").write_text(
        json.dumps(
            {
                "truth_snapshot_enabled": False,
                "truth_diagnostics_enabled": False,
                "truth_snapshot_rows": 0,
                "truth_used_for_inference_or_checkpoint_selection": False,
            }
        )
    )
    assert truth_free_inference_contract(tmp_path)["status"] == "PASS"
    (tmp_path / "inference_truth.parquet").write_bytes(b"forbidden")
    assert truth_free_inference_contract(tmp_path)["status"] == "FAIL"


def test_sc_drws_selects_raw_or_ema_only_after_independent_support_gates(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "feature_stats.json").write_text("{}")
    receipt = {
        "status": "TRAINING_COMPLETE_PENDING_SUPPORT_SELECTION",
        "wake_updates": 45,
        "prior_updates": 2,
        "selected_training_rows": 512,
        "phase_schedule": {"warmup_epochs": 60, "joint_epochs": 120},
        "selection_log_alpha": -0.2,
        "selection_alpha": float(np.exp(-0.2)),
        "reference_entropy": 20.0,
        "truth_used_for_training_validation_or_checkpoint_selection": False,
        "raw_model_checkpoint": {"path": "/raw.eqx"},
        "ema_model_checkpoint": {"path": "/ema.eqx"},
    }
    (train / "training_receipt.json").write_text(json.dumps(receipt))
    pd.DataFrame(
        {
            "epoch": [180, 180],
            "update_kind": ["sleep", "wake"],
            "full_entropy": [19.0, 19.2],
            "expanded_ess_fraction": [np.nan, 0.1],
            "expansion_fraction": [0.0, 0.5],
            "unresolved_fraction": [0.0, 0.0],
            "q_grads_finite": [True, True],
            "q_grad_clipped": [False, False],
        }
    ).to_csv(train / "sc_drws_training_log.csv", index=False)
    pd.DataFrame(
        {
            "gate_accepted": [True],
            "update_applied": [True],
            "selection_gradient_finite": [True],
            "data_gradient_finite": [True],
            "trust_gradient_finite": [True],
        }
    ).to_csv(train / "sc_drws_prior_log.csv", index=False)
    raw_iw, raw_ppc = _write_sc_variant(tmp_path / "raw", score=1.0)
    ema_iw, ema_ppc = _write_sc_variant(tmp_path / "ema", score=2.0)
    result = summarize_candidate(
        candidate="historical_4x128",
        seed=260826,
        phase="pilot",
        config=CONFIG_DIR / "feniks_sc_drws_r29_historical.yaml",
        train=train,
        importance=raw_iw,
        predictive=raw_ppc,
        ema_importance=ema_iw,
        ema_predictive=ema_ppc,
        out=tmp_path / "summary",
    )
    assert result["status"] == "PASS"
    assert result["checkpoint_variant"] == "ema"
    assert result["checkpoint"] == "/ema.eqx"


def test_undertrained_smoke_is_technical_only_but_pilot_stays_fail_closed(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "feature_stats.json").write_text("{}")
    receipt = {
        "status": "TRAINING_COMPLETE_PENDING_SUPPORT_SELECTION",
        "wake_updates": 2,
        "prior_updates": 0,
        "selected_training_rows": 128,
        "phase_schedule": {"warmup_epochs": 4, "joint_epochs": 4},
        "selection_log_alpha": -0.2,
        "selection_alpha": float(np.exp(-0.2)),
        "reference_entropy": 20.0,
        "truth_used_for_training_validation_or_checkpoint_selection": False,
        "raw_model_checkpoint": {"path": "/raw.eqx"},
        "ema_model_checkpoint": {"path": "/ema.eqx"},
    }
    (train / "training_receipt.json").write_text(json.dumps(receipt))
    pd.DataFrame(
        {
            "epoch": [7, 8],
            "update_kind": ["sleep", "wake"],
            "full_entropy": [19.0, 19.0],
            "expanded_ess_fraction": [np.nan, 1.0 / 512.0],
            "expansion_fraction": [0.0, 1.0],
            "unresolved_fraction": [0.0, 1.0],
            "q_grads_finite": [True, True],
            "q_grad_clipped": [False, False],
        }
    ).to_csv(train / "sc_drws_training_log.csv", index=False)
    pd.DataFrame(
        {
            "gate_accepted": [False],
            "update_applied": [False],
            "selection_gradient_finite": [False],
            "data_gradient_finite": [False],
            "trust_gradient_finite": [False],
        }
    ).to_csv(train / "sc_drws_prior_log.csv", index=False)
    raw_iw, raw_ppc = _write_sc_variant(
        tmp_path / "raw",
        score=1.0,
        support_status="FAIL",
        ess_fraction=1.0 / 32.0,
        max_weight=0.99,
        pareto_fraction=0.30,
        predictive_scale=8.0,
    )
    ema_iw, ema_ppc = _write_sc_variant(
        tmp_path / "ema",
        score=1.1,
        support_status="FAIL",
        ess_fraction=1.0 / 32.0,
        max_weight=0.99,
        pareto_fraction=0.30,
        predictive_scale=8.0,
    )
    common = {
        "candidate": "historical_4x128",
        "seed": 260826,
        "phase": "pilot",
        "config": CONFIG_DIR / "feniks_sc_drws_r29_historical.yaml",
        "train": train,
        "importance": raw_iw,
        "predictive": raw_ppc,
        "ema_importance": ema_iw,
        "ema_predictive": ema_ppc,
    }

    smoke = summarize_candidate(out=tmp_path / "smoke", smoke=True, **common)
    pilot = summarize_candidate(out=tmp_path / "pilot", smoke=False, **common)

    assert smoke["status"] == "SMOKE_PASS"
    assert smoke["smoke_contract"]["status"] == "PASS"
    assert smoke["training"]["status"] == "FAIL"
    assert smoke["training"]["technical_smoke_status"] == "PASS"
    assert all(variant["status"] == "FAIL" for variant in smoke["variants"])
    assert all(
        variant["technical_status"] == "PASS" for variant in smoke["variants"]
    )
    assert pilot["status"] == "FAIL"


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
    assert final["ready_for_smc_diversity_benchmark"] is False
    assert final["ready_for_population_prior_update"] is True
    assert final["ready_for_full_catalogue"] is True
    assert (tmp_path / "RWS_RECOVERY_PASS.json").is_file()


def test_full_finalizer_requires_only_configured_single_seed(tmp_path: Path) -> None:
    candidate = "historical_4x128"
    (tmp_path / "RWS_RECOVERY_PASS.json").write_text(
        json.dumps({"status": "PASS", "selected_candidate": candidate})
    )
    run_dir = tmp_path / "full" / candidate / f"seed_{FULL_SEEDS[0]}"
    run_dir.mkdir(parents=True)
    summary = {
        "status": "PASS",
        "seed": FULL_SEEDS[0],
        "checkpoint": "/full/raw.eqx",
        "feature_stats": "/full/feature_stats.json",
        "variants": [
            {
                "status": "PASS",
                "selection_corrected_exact_gaussian_iw_score": 1.0,
            }
        ],
    }
    (run_dir / "full_summary.json").write_text(json.dumps(summary))

    receipt = finalize_full(tmp_path)

    assert FULL_SEEDS == (260826,)
    assert receipt["status"] == "PASS"
    assert receipt["selected_seed"] == 260826
    assert receipt["configured_final_seeds"] == [260826]
    assert receipt["all_configured_final_seeds_passed"] is True
    assert receipt["ready_for_four_shard_catalogue_inference"] is True


def test_full_finalizer_accepts_explicit_unconfirmed_authorization(
    tmp_path: Path,
) -> None:
    candidate = "current_residual_6x256"
    (tmp_path / "FULL_LAUNCH_AUTHORIZATION.json").write_text(
        json.dumps(
            {
                "status": "EXPLICIT_UNCONFIRMED_FULL_OVERRIDE",
                "selected_candidate": candidate,
                "truth_used_for_training_or_authorization": False,
            }
        )
    )
    run_dir = tmp_path / "full" / candidate / f"seed_{FULL_SEEDS[0]}"
    run_dir.mkdir(parents=True)
    (run_dir / "full_summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "seed": FULL_SEEDS[0],
                "checkpoint": "/full/ema.eqx",
                "feature_stats": "/full/feature_stats.json",
                "variants": [
                    {
                        "status": "PASS",
                        "selection_corrected_exact_gaussian_iw_score": 2.0,
                    }
                ],
            }
        )
    )

    receipt = finalize_full(tmp_path)

    assert receipt["status"] == "PASS"
    assert (
        receipt["launch_authorization_status"]
        == "EXPLICIT_UNCONFIRMED_FULL_OVERRIDE"
    )


def test_submitter_encodes_smoke_pilot_confirmation_and_safe_cache() -> None:
    submitter = (ROOT / "scripts" / "submit_feniks_rws_recovery.sh").read_text()
    worker = (ROOT / "scripts" / "feniks_rws_recovery_pilot_h100.slurm").read_text()
    monitor = (ROOT / "scripts" / "monitor_feniks_rws_recovery.sh").read_text()

    assert "--array=0-3%4" in submitter
    assert 'afterok:$SMOKE_JOB' in submitter
    assert 'afterok:$PILOT_JOB' in submitter
    assert 'afterok:$PILOT_GATE_JOB' in submitter
    assert 'afterok:$CONFIRM_JOB' in submitter
    assert "--confirmation-objects 2000" in submitter
    assert 'EUCLID_DSPS_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"' in worker
    assert 'JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT/jax"' in worker
    assert "export JAX_ENABLE_X64=true" in worker
    assert "train_feniks_sc_drws.py" in worker
    assert 'export SMOKE_ROOT=%q' in submitter
    assert "sc_drws_training_log.csv" in monitor
    assert "full catalogue autorisé mais non soumis" in monitor
    assert "pilot_train_indices.npy" in worker
    assert "for VARIANT in raw ema" in worker
    assert '${VARIANT}_model.eqx' in worker
    assert "--posterior-samples \"$SUPPORT_SAMPLES\"" in worker
    assert "--min-median-ess-fraction 0.05" in worker
    assert "--max-fraction-pareto-k-gt-0p7 0.20" in worker
    assert 'RESUME_EXISTING="${RESUME_EXISTING:-0}"' in submitter
    assert 'if [[ "$RESUME_EXISTING" == "1" ]]' in submitter
    assert "missing resume artifact" in submitter
    assert "TRAINING_COMPLETE=1" in worker
    assert "reusing completed truth-free training" in worker
    assert 'if [[ "$TRAINING_COMPLETE" == "0" ]]' in worker
    assert "inference_artifact_complete" in worker
    assert "importance_artifact_complete" in worker
    assert "reusing complete truth-free" in worker
    assert "reusing complete ${VARIANT} ordinary-IW artifact" in worker
