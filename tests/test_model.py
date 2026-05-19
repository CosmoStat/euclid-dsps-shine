from __future__ import annotations

import numpy as np
import pytest

from euclid_dsps.model import (
    apply_cosmos_two_component_dust_jax,
    build_binned_sfh_jax,
    build_lognormal_sfh,
    normalize_sfh_mass_jax,
)


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


def test_sfh_can_be_normalized_to_formed_mass() -> None:
    time = np.linspace(0.1, 10.0, 128)
    sfr = build_lognormal_sfh(time, 0.0, 4.0, 0.6)

    scaled, formed_mass = normalize_sfh_mass_jax(
        time, sfr, {"log10_formed_mass_msun": 10.0}
    )

    assert float(formed_mass) == pytest.approx(1.0e10, rel=1.0e-5)
    assert np.trapezoid(np.asarray(scaled), time) * 1.0e9 == pytest.approx(
        1.0e10, rel=1.0e-5
    )


def test_binned_sfh_is_positive_and_responds_to_bin_weights() -> None:
    time = np.linspace(0.1, 10.0, 128)

    flat = np.asarray(build_binned_sfh_jax(time, np.zeros(6)))
    rising = np.asarray(build_binned_sfh_jax(time, np.linspace(-1.0, 1.0, 6)))

    assert np.all(flat > 0.0)
    assert np.all(rising > 0.0)
    assert rising[-1] > rising[0]
