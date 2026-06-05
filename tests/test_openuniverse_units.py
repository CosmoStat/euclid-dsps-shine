from __future__ import annotations

import numpy as np
import pytest

from euclid_dsps.openuniverse.units import (
    internal_to_photon_flux,
    photon_flux_to_internal,
    validate_flux_unit,
)


def test_native_openuniverse_flux_unit_roundtrip() -> None:
    flux = np.asarray([1.0, 2.0], dtype=float)

    internal = photon_flux_to_internal(flux, "photon_per_sec_cm2")
    back = internal_to_photon_flux(internal)

    assert validate_flux_unit("photons/sec/cm^2") == "photon_per_sec_cm2"
    np.testing.assert_allclose(back, flux)


def test_units_do_not_silently_convert_to_fnu() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        photon_flux_to_internal([1.0], "fnu_cgs")
