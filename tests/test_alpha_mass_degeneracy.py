from __future__ import annotations

import numpy as np

from euclid_dsps.amortized.catalog import (
    posterior_predictive_flux_frame,
    posterior_summary_frame,
)
from euclid_dsps.calibration import log10_mass_alpha_corrected


def test_alpha_sed_mass_correction_formula() -> None:
    raw = np.asarray([9.0, 10.0])

    assert np.allclose(log10_mass_alpha_corrected(raw, 1.0), raw)
    assert np.allclose(log10_mass_alpha_corrected(raw, 10.0), raw + 1.0)


def test_posterior_summary_reports_raw_and_alpha_corrected_mass() -> None:
    theta = np.asarray([[[9.0]], [[10.0]], [[11.0]]])
    summary = posterior_summary_frame(
        object_id=np.asarray([101]),
        theta=theta,
        parameter_names=("log10_stellar_mass",),
        loglike=np.zeros((3, 1)),
        chi2=np.zeros((3, 1)),
        mask=np.ones((1, 2), dtype=bool),
        alpha_sed=10.0,
    )

    assert np.isclose(summary.loc[0, "log10_stellar_mass_raw"], 10.0)
    assert np.isclose(summary.loc[0, "log10_stellar_mass_alpha_corrected"], 11.0)


def test_predictive_flux_frame_keeps_raw_and_scaled_flux() -> None:
    frame = posterior_predictive_flux_frame(
        object_id=np.asarray(["prior"]),
        model_flux=np.asarray([[[2.0, 4.0]]]),
        model_flux_raw=np.asarray([[[1.0, 2.0]]]),
        band_names=("b1", "b2"),
        alpha_sed=2.0,
    )

    assert set(frame.columns) >= {
        "model_flux_raw_fnu_cgs",
        "model_flux_scaled_fnu_cgs",
        "alpha_sed",
    }
    assert np.allclose(
        frame["model_flux_scaled_fnu_cgs"],
        2.0 * frame["model_flux_raw_fnu_cgs"],
    )
