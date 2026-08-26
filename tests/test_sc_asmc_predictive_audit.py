from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.sc_asmc_predictive_audit import (
    _posterior_theta_block,
    finalize_existing_predictive_audit,
    object_predictive_rows,
    residual_summary_rows,
)


def test_residual_summary_preserves_all_draws_and_masks() -> None:
    residual = np.asarray([[[0.0, 1.0], [2.0, 99.0]], [[1.0, -1.0], [0.0, 99.0]]])
    mask = np.asarray([[True, True], [True, False]])

    rows = residual_summary_rows("q1", residual, mask, ("a", "b"))

    assert rows[0]["objects"] == 2
    assert rows[0]["draws_per_object"] == 2
    assert rows[0]["rms_normalized_residual"] == np.sqrt(1.25)
    assert rows[1]["objects"] == 1
    assert rows[1]["median_normalized_residual"] == 0.0


def test_object_predictive_rows_uses_full_draw_distribution() -> None:
    residual = np.asarray([[[1.0, 1.0]], [[3.0, 3.0]]])
    rows = object_predictive_rows(
        "smc_em2",
        np.asarray([7]),
        np.asarray(["g7"]),
        residual,
        np.asarray([[True, True]]),
    )

    assert rows.loc[0, "posterior_median_reduced_chi2"] == 5.0
    assert rows.loc[0, "posterior_mean_reduced_chi2"] == 5.0


def test_posterior_theta_block_selects_even_dense_draws() -> None:
    frame = pd.DataFrame(
        {
            "row_index": np.repeat([4, 9], 4),
            "object_id": np.repeat(["a", "b"], 4),
            "sample_id": np.tile(np.arange(4), 2),
            "x": np.arange(8),
            "y": np.arange(8) + 10,
        }
    )

    rows, object_ids, theta = _posterior_theta_block(
        frame, parameters=("x", "y"), requested_draws=2
    )

    np.testing.assert_array_equal(rows, [4, 9])
    np.testing.assert_array_equal(object_ids, ["a", "b"])
    assert theta.shape == (2, 2, 2)
    np.testing.assert_array_equal(theta[:, 0, 0], [0, 3])


def test_predictive_audit_launcher_uses_four_gpus_and_frozen_inputs() -> None:
    source = Path("scripts/feniks_sc_asmc_predictive_audit_4gpu.slurm").read_text()

    assert "#SBATCH --gres=gpu:4" in source
    assert 'test -s "$RUN_ROOT/FINAL_PASS"' in source
    assert 'test -s "$CLOSURE_ROOT/truth_closure_receipt.json"' in source
    assert "--posterior-draws 16" in source


def test_finalize_existing_predictive_audit_recovers_plot_failure(tmp_path) -> None:
    methods = ("truth_forward", "q0", "smc_em1", "q1", "smc_em2")
    audit = tmp_path / "audit"
    closure = tmp_path / "closure"
    audit.mkdir()
    closure.mkdir()
    (closure / "truth_closure_receipt.json").write_text(
        '{"status":"PASS","final_receipt_sha256":"frozen-hash"}\n'
    )
    summary_rows = []
    object_rows = []
    for method in methods:
        for band in ("a", "b"):
            summary_rows.append(
                {
                    "method": method,
                    "band": band,
                    "objects": 2,
                    "draws_per_object": 1 if method == "truth_forward" else 16,
                    "median_normalized_residual": 0.0,
                    "mean_normalized_residual": 0.0,
                    "rms_normalized_residual": 1.0,
                    "q05_normalized_residual": -1.0,
                    "q95_normalized_residual": 1.0,
                    "fraction_abs_lt_1": 0.68,
                    "fraction_abs_lt_3": 0.997,
                    "finite_fraction": 1.0,
                }
            )
        for row in (0, 1):
            object_rows.append(
                {
                    "method": method,
                    "row_index": row,
                    "object_id": str(row),
                    "posterior_median_reduced_chi2": 1.0,
                    "posterior_mean_reduced_chi2": 1.0,
                    "valid_bands": 2,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        audit / "predictive_residuals_by_band.csv", index=False
    )
    pd.DataFrame(object_rows).to_parquet(
        audit / "predictive_diagnostics_by_object.parquet", index=False
    )

    receipt = finalize_existing_predictive_audit(
        out_dir=audit,
        closure_root=closure,
    )

    assert receipt["recovered_after_postprocessing_failure"] is True
    assert receipt["scientific_gate_pass"] is True
    assert (audit / "predictive_audit_receipt.json").is_file()
    assert (audit / "full_catalogue_predictive_residuals.png").is_file()
    assert (audit / "truth_forward_residuals_by_band.pdf").is_file()
    assert (audit / "posterior_predictive_smc_zoom.png").is_file()
    assert (audit / "posterior_predictive_reduced_chi2_ecdf.pdf").is_file()
