from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.diffsky_forward_closure import (
    build_trueparam_theta,
    forward_closure_residuals,
)
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES


def _truth_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_id": [1, 2],
            "redshift_true": [0.2, 0.3],
            "logsm_true": [9.0, 10.0],
            "diffstar_lgmcrit": [11.0, 11.1],
            "diffstar_lgy_at_mcrit": [-10.0, -10.1],
            "diffstar_indx_lo": [1.0, 1.1],
            "diffstar_indx_hi": [-1.0, -1.1],
            "diffstar_lg_qt": [0.5, 0.6],
            "diffstar_qlglgdt": [-0.5, -0.6],
            "diffstar_lg_drop": [-1.0, -1.1],
            "diffstar_lg_rejuv": [-0.2, -0.3],
            "diffmah_logm0": [12.0, 12.1],
            "diffmah_logtc": [0.05, 0.06],
            "diffmah_early_index": [2.0, 2.1],
            "diffmah_late_index": [0.2, 0.3],
            "diffmah_t_peak": [10.0, 10.1],
            "dust_av": [0.2, 0.3],
            "dust_delta": [-0.2, -0.1],
        }
    )


def test_trueparam_theta_records_fixed_metallicity() -> None:
    theta, sources = build_trueparam_theta(
        _truth_frame(),
        {"model": {"fixed_parameters": {"log10_stellar_metallicity": -0.8}}},
    )

    assert theta.shape == (2, len(DIFFSKY_BASIC_PARAMETER_NAMES))
    metallicity_index = DIFFSKY_BASIC_PARAMETER_NAMES.index(
        "log10_stellar_metallicity"
    )
    assert np.all(theta[:, metallicity_index] == -0.8)
    row = sources.set_index("parameter").loc["log10_stellar_metallicity"]
    assert row["source_kind"] == "nuisance_fixed"


def test_trueparam_theta_missing_diffstar_fails_clearly() -> None:
    frame = _truth_frame().drop(columns=["diffstar_lgmcrit"])

    with pytest.raises(ValueError, match="Missing Diffsky true-parameter columns"):
        build_trueparam_theta(frame, {"model": {"fixed_parameters": {}}})


def test_forward_closure_mock_decoder_zero_residual() -> None:
    observed = np.asarray([[24.0, 25.0], [23.5, 24.5]])
    photometry, by_band, summary = forward_closure_residuals(
        object_id=np.asarray([1, 2]),
        observed_mag=observed,
        model_mag=observed.copy(),
        band_names=("lsst_u", "lsst_g"),
        truth_context=pd.DataFrame(
            {
                "object_id": [1, 2],
                "redshift_true": [0.2, 0.3],
                "logsm_true": [9.0, 10.0],
            }
        ),
    )

    assert np.allclose(photometry["residual_mag"], 0.0)
    assert np.allclose(by_band["rms_residual_mag"], 0.0)
    assert summary["rms_residual_mag"] == 0.0
