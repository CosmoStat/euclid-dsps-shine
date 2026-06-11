from __future__ import annotations

import numpy as np

from euclid_dsps.diffsky_data.photometry import (
    detect_photometry_columns,
    standardize_magnitude_photometry,
)


def test_photometry_detector_standardizes_ab_magnitudes() -> None:
    report = detect_photometry_columns(["lsst_u", "lsst_g", "lsst_g_bulge"])
    frame = standardize_magnitude_photometry(
        {"lsst_u": np.asarray([25.0]), "lsst_g": np.asarray([24.0])},
        report,
        snr=50.0,
    )

    assert report.band_names == ("lsst_u", "lsst_g")
    assert frame["flux_lsst_u"].iloc[0] > 0.0
    assert frame["fluxerr_lsst_g"].iloc[0] > 0.0
    assert frame["mask_lsst_g"].iloc[0]


def test_photometry_standardization_can_skip_synthetic_errors() -> None:
    report = detect_photometry_columns(["lsst_u"])
    frame = standardize_magnitude_photometry(
        {"lsst_u": np.asarray([25.0])},
        report,
        snr=50.0,
        add_synthetic_errors=False,
    )

    assert "mag_lsst_u" in frame
    assert "flux_lsst_u" in frame
    assert "mask_lsst_u" in frame
    assert "fluxerr_lsst_u" not in frame
