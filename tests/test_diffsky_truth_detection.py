from __future__ import annotations

from euclid_dsps.diffsky_data.truth import detect_truth_columns


def test_truth_detector_finds_basic_and_generated_truths() -> None:
    report = detect_truth_columns(
        [
            "data/redshift_true",
            "data/logsm_obs",
            "data/logssfr_obs",
            "data/logmp_obs",
            "data/central",
            "data/lgmcrit",
            "data/early_index",
            "data/av",
            "data/lsst_g",
        ]
    )

    assert report.redshift == "data/redshift_true"
    assert report.stellar_mass == "data/logsm_obs"
    assert "data/lgmcrit" in report.diffstar_columns
    assert "data/early_index" in report.diffmah_columns
    assert "data/lsst_g" in report.photometry_columns
