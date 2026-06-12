from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pandas as pd

from euclid_dsps.amortized.latent import LatentSpec
from euclid_dsps.config import load_config
from euclid_dsps.diffsky_full_validation import (
    _prior_spec_incompatibility,
    write_full_validation_report,
)


def test_full_validation_h100_config_declares_alpha_report_section() -> None:
    config = load_config(Path("configs/experiments/diffsky_hltds_full_h100.yaml"))

    sections = config["full_validation"]["report_sections"]
    alpha_fields = config["full_validation"]["global_sed_scale_reporting"]["fields"]
    assert "Global SED scale calibration" in sections
    assert "alpha_sed" in alpha_fields
    assert "mass_bias_alpha_corrected" in alpha_fields
    assert config["calibration"]["per_band_zero_points"]["enabled"] is False


def test_full_validation_report_collects_alpha_and_mass_metrics(tmp_path) -> None:
    run = tmp_path / "amortized_joint"
    run.mkdir()
    (run / "inference_summary.json").write_text(
        json.dumps(
            {
                "global_sed_scale": {
                    "alpha_sed": 1.2,
                    "log_alpha_sed": 0.182,
                    "delta_mag_global": -0.198,
                    "alpha_prior_penalty": 1.66,
                    "large_scale_warning": False,
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"metric_name": "mass_bias_raw", "bias": -0.2, "rmse": 0.3},
            {"metric_name": "mass_bias_alpha_corrected", "bias": -0.1, "rmse": 0.2},
        ]
    ).to_csv(run / "posterior_vs_truth_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "label": "joint",
                "median_bias": 0.0,
                "likelihood_type": "student_t",
                "alpha_sed": 1.2,
            }
        ]
    ).to_csv(run / "photoz_metrics.csv", index=False)
    closure = tmp_path / "closure"
    closure.mkdir()
    (closure / "forward_closure_summary.json").write_text(
        json.dumps(
            {
                "global_sed_scale": {
                    "mode": "fit_global",
                    "alpha_sed": 1.1,
                    "delta_mag_global": -0.103,
                }
            }
        ),
        encoding="utf-8",
    )

    report = write_full_validation_report(
        tmp_path / "report",
        dataset_path="mock.parquet",
        inference_runs=[("joint", run)],
        closure_run=closure,
        stage_outputs={
            "skipped_stages": [
                {
                    "stage": "amortized_supervised_prior",
                    "status": "skipped",
                    "reason": "incompatible latent names",
                }
            ]
        },
    )
    summary = json.loads((tmp_path / "report" / "full_validation_summary.json").read_text())
    text = report.read_text(encoding="utf-8")

    assert "Global SED scale calibration" in text
    assert "Skipped stages" in text
    assert summary["skipped_stages"][0]["stage"] == "amortized_supervised_prior"
    assert summary["global_sed_scale"][0]["alpha_sed"] == 1.2
    assert {
        row["metric_name"] for row in summary["mass_recovery"]
    } == {"mass_bias_raw", "mass_bias_alpha_corrected"}


def test_full_validation_detects_supervised_prior_latent_mismatch() -> None:
    active = LatentSpec(
        names=("z_obs", "log10_stellar_mass", "dlog10_sfr_1"),
        lower=jnp.zeros(3, dtype=jnp.float32),
        upper=jnp.ones(3, dtype=jnp.float32),
    )
    loaded = LatentSpec(
        names=("z_obs", "log10_stellar_mass", "log10_ssfr_at_obs"),
        lower=jnp.zeros(3, dtype=jnp.float32),
        upper=jnp.ones(3, dtype=jnp.float32),
    )

    reason = _prior_spec_incompatibility(active, loaded)

    assert reason is not None
    assert "incompatible latent names" in reason
    assert _prior_spec_incompatibility(active, active) is None
