from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.sc_asmc_predictive_audit import (
    _posterior_theta_block,
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
