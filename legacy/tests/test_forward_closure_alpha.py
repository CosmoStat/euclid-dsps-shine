from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.calibration import GlobalSedScaleConfig, delta_mag_from_alpha
from euclid_dsps.diffsky_forward_closure import (
    _closure_alpha_fit,
    _write_forward_closure_report,
    forward_closure_residuals,
)


def test_closure_fit_global_alpha_removes_median_mag_residual() -> None:
    observed = np.asarray([[24.0, 25.0], [24.0, 25.0]])
    raw_model = observed + 0.5
    fit = _closure_alpha_fit(
        GlobalSedScaleConfig(enabled=True, mode="fit_global"),
        observed,
        raw_model,
    )

    assert np.isclose(fit["delta_mag_global"], -0.5)
    assert fit["alpha_sed"] > 1.0


def test_forward_closure_residuals_report_raw_and_scaled_flux() -> None:
    observed = np.asarray([[24.0]])
    raw_model = np.asarray([[24.0]])
    alpha = 2.0
    scaled_model = raw_model + delta_mag_from_alpha(alpha)
    photometry, _by_band, summary = forward_closure_residuals(
        object_id=np.asarray([1]),
        observed_mag=observed,
        model_mag=scaled_model,
        model_mag_raw=raw_model,
        band_names=("lsst_u",),
        log_alpha_sed=float(np.log(alpha)),
        alpha_sed=alpha,
    )

    row = photometry.iloc[0]
    assert np.isclose(row["model_flux_scaled_fnu_cgs"], 2.0 * row["model_flux_raw_fnu_cgs"])
    assert np.isclose(summary["alpha_sed"], 2.0)


def test_forward_closure_report_contains_alpha_section(tmp_path) -> None:
    path = tmp_path / "report.md"
    _write_forward_closure_report(
        path,
        {
            "dataset_path": "mock.parquet",
            "sfh_model": "diffsky_basic",
            "n_objects": 1,
            "n_bands": 1,
            "median_abs_residual_mag": 0.0,
            "rms_residual_mag": 0.0,
            "allow_partial_truth": False,
            "global_sed_scale": {
                "mode": "fit_global",
                "alpha_sed": 1.0,
                "log_alpha_sed": 0.0,
                "delta_mag_global": 0.0,
                "alpha_prior_penalty": 0.0,
                "warning": "",
            },
        },
        pd.DataFrame({"band": ["lsst_u"], "rms_residual_mag": [0.0]}),
        pd.DataFrame({"parameter": ["z_obs"], "source_kind": ["truth"]}),
    )

    text = path.read_text(encoding="utf-8")
    assert "Global SED Scale" in text
    assert "alpha_sed" in text
