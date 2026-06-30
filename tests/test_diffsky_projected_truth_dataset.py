from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_diffsky_lowz_projected_truth_dataset import sfr_consistency_frame


def test_sfr_consistency_frame_compares_projected_bins_to_catalog_truth() -> None:
    frame = pd.DataFrame(
        {
            "logsfr_true": [0.0, 1.0],
            "logsm_true": [10.0, 10.5],
            "logssfr_true": [-10.0, -9.5],
        }
    )
    log_sfr_bins = np.asarray(
        [
            [0.0, -0.2],
            [1.1, 0.7],
        ]
    )

    metrics = sfr_consistency_frame(frame, log_sfr_bins)

    row = metrics.loc[
        metrics["metric"] == "projected_log10_sfr_bin_1_vs_logsfr_true"
    ].iloc[0]
    assert row["n_objects"] == 2
    assert row["median_delta_projected_minus_reference"] == pytest.approx(0.05)
    ssfr_row = metrics.loc[
        metrics["metric"]
        == "projected_log10_sfr_bin_1_minus_logsm_true_vs_logssfr_true"
    ].iloc[0]
    assert ssfr_row["median_delta_projected_minus_reference"] == pytest.approx(0.05)
    identity = metrics.loc[
        metrics["metric"] == "logsfr_true_minus_logsm_true_vs_logssfr_true"
    ].iloc[0]
    assert identity["p95_abs_delta"] == 0.0
