"""OpenUniverse photon-rate unit contract.

TODO units:
- choose the definitive internal unit for OpenUniverse/DSPS training;
- implement photon-rate photometry in the DSPS decoder or a validated
  photon-rate <-> flux-density conversion;
- verify the conversion with LSST and Roman throughput curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import OU_NATIVE_FLUX_UNIT

SUPPORTED_OPENUNIVERSE_FLUX_UNITS = {
    OU_NATIVE_FLUX_UNIT,
    "photon/s/cm2",
    "photons/sec/cm^2",
}


def validate_flux_unit(unit: str) -> str:
    """Normalize and validate an OpenUniverse flux unit string."""
    normalized = str(unit).strip()
    if normalized in SUPPORTED_OPENUNIVERSE_FLUX_UNITS:
        return OU_NATIVE_FLUX_UNIT
    if normalized in {"fnu_cgs", "abmag", "microjy", "ujy"}:
        raise NotImplementedError(
            "OpenUniverse photon-rate conversion to/from "
            f"{normalized!r} is not implemented. Keep native "
            f"{OU_NATIVE_FLUX_UNIT!r} units or add a validated filter-aware "
            "conversion."
        )
    raise ValueError(
        f"Unsupported OpenUniverse flux unit {unit!r}; expected native "
        f"{OU_NATIVE_FLUX_UNIT!r}"
    )


def photon_flux_to_internal(
    values: Any,
    unit: str = OU_NATIVE_FLUX_UNIT,
) -> np.ndarray:
    """Convert photon-rate fluxes to the current OpenUniverse internal unit."""
    validate_flux_unit(unit)
    return np.asarray(values, dtype=np.float32)


def internal_to_photon_flux(
    values: Any,
    unit: str = OU_NATIVE_FLUX_UNIT,
) -> np.ndarray:
    """Convert current OpenUniverse internal fluxes back to photon-rate units."""
    validate_flux_unit(unit)
    return np.asarray(values, dtype=np.float32)
