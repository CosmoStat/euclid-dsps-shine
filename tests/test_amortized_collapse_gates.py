from __future__ import annotations

import pandas as pd

import json

from euclid_dsps.amortized.collapse_gates import (
    write_inference_collapse_gate,
    write_training_collapse_gate,
)


def test_training_gate_uses_latest_applied_training_gradient(tmp_path) -> None:
    pd.DataFrame(
        {
            "split": ["train", "validation"],
            "update_applied": [1, 0],
            "loss": [1.0, 1.0],
            "encoder_grad_norm": [2500.0, 0.0],
        }
    ).to_csv(tmp_path / "training_log.csv", index=False)

    payload = write_training_collapse_gate(tmp_path)
    gradient = next(
        check
        for check in payload["checks"]
        if check["name"] == "latest_encoder_grad_norm"
    )

    assert gradient["value"] == 2500.0
    assert gradient["status"] == "FAIL"


def test_training_gate_rejects_collapsed_wake_importance_weights(tmp_path) -> None:
    pd.DataFrame(
        {
            "split": ["train", "train"],
            "update_applied": [1, 1],
            "loss": [1.0, 1.0],
            "wake_active": [1, 1],
            "wake_ess_fraction_mean": [0.13, 0.14],
            "wake_weight_max_mean": [0.97, 0.98],
            "wake_physical_valid_fraction": [1.0, 1.0],
        }
    ).to_csv(tmp_path / "training_log.csv", index=False)
    payload = write_training_collapse_gate(tmp_path)
    assert payload["status"] == "FAIL"
    failed = {
        row["name"] for row in payload["checks"] if row["status"] == "FAIL"
    }
    assert "wake_ess_fraction_mean" in failed
    assert "wake_weight_max_mean" in failed


def test_training_gate_rejects_trainable_calibration_without_gradient(
    tmp_path,
) -> None:
    pd.DataFrame(
        {
            "split": ["train"],
            "update_applied": [1],
            "loss": [1.0],
            "wake_active": [1],
            "band_alpha_grad_norm": [0.0],
        }
    ).to_csv(tmp_path / "training_log.csv", index=False)
    (tmp_path / "training_summary.json").write_text(
        json.dumps({"per_band_flux_calibration": {"trainable": True}}),
        encoding="utf-8",
    )
    payload = write_training_collapse_gate(tmp_path)
    check = next(
        row
        for row in payload["checks"]
        if row["name"] == "trainable_band_calibration_has_wake_gradient"
    )
    assert check["status"] == "FAIL"


def test_inference_gate_rejects_bad_real_photometry(tmp_path) -> None:
    (tmp_path / "posterior_diagnostics_summary.json").write_text(
        json.dumps(
            {
                "median_posterior_predictive_chi2": 2600.0,
                "median_valid_bands": 26.0,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"band": ["__all__"], "frac_abs_gt_5": [0.7]}
    ).to_csv(
        tmp_path / "posterior_predictive_normalized_residual_tails.csv",
        index=False,
    )
    pd.DataFrame(
        {"parameter": ["x"], "frac_within_5pct_boundary": [0.95]}
    ).to_csv(tmp_path / "parameter_bound_diagnostics.csv", index=False)
    payload = write_inference_collapse_gate(tmp_path)
    assert payload["status"] == "FAIL"
