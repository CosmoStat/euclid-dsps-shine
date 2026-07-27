"""Unit conventions for prepared Diffsky datasets."""

from __future__ import annotations


def describe_photometry_unit(native_kind: str, native_unit: str) -> dict:
    warnings = []
    if native_kind == "magnitude" and "AB" not in native_unit:
        warnings.append("Magnitude unit is not explicitly AB.")
    if native_kind == "flux" and native_unit == "unknown_flux_unit":
        warnings.append("Flux unit is unknown; no physical conversion applied.")
    return {
        "kind": native_kind,
        "unit": native_unit,
        "conversion_applied": native_kind == "magnitude" and "AB" in native_unit,
        "prepared_flux_unit": (
            "fnu_cgs"
            if native_kind == "magnitude" and "AB" in native_unit
            else native_unit
        ),
        "warnings": warnings,
    }
