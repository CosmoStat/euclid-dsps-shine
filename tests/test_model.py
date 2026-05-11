from __future__ import annotations

import numpy as np
import pytest

from euclid_dsps.model import apply_cosmos_two_component_dust_jax, build_lognormal_sfh


def test_burst_and_quench_sfh_stays_positive_and_changes_shape() -> None:
    time = np.linspace(0.1, 10.0, 128)

    base = build_lognormal_sfh(time, 0.0, 4.0, 0.6)
    modified = build_lognormal_sfh(
        time,
        0.0,
        4.0,
        0.6,
        sfh_burst_fraction=1.0,
        sfh_burst_time=2.0,
        sfh_quench_time=7.0,
        sfh_quench_depth=0.8,
    )

    assert np.all(modified > 0.0)
    assert modified[np.argmin(np.abs(time - 2.0))] > base[np.argmin(np.abs(time - 2.0))]
    assert modified[-1] < base[-1]


def test_cosmos_two_component_dust_mixes_configured_curves() -> None:
    rest_sed = np.ones(3)
    k_by_code = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
        ]
    )

    attenuated = np.asarray(
        apply_cosmos_two_component_dust_jax(
            rest_sed,
            {
                "cosmos_ext_curve_1": 1.0,
                "cosmos_ext_curve_2": 2.0,
                "cosmos_ebv_1": 0.5,
                "cosmos_ebv_2": 0.25,
                "cosmos_frac_1": 2.0,
                "cosmos_frac_2": 1.0,
            },
            k_by_code,
        )
    )

    expected = (2.0 / 3.0) * 10 ** (-0.4 * 0.5 * 2.0) + (1.0 / 3.0) * 10 ** (
        -0.4 * 0.25 * 4.0
    )
    assert attenuated.tolist() == pytest.approx([expected, expected, expected])
