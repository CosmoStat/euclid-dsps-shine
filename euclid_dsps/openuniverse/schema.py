"""Column and band contracts for OpenUniverse SkyCatalog inputs."""

from __future__ import annotations

OU_NATIVE_FLUX_UNIT = "photon_per_sec_cm2"

OU_LSST_BANDS = (
    "lsst_u",
    "lsst_g",
    "lsst_r",
    "lsst_i",
    "lsst_z",
    "lsst_y",
)

OU_ROMAN_BANDS = (
    "roman_W146",
    "roman_R062",
    "roman_Z087",
    "roman_Y106",
    "roman_J129",
    "roman_H158",
    "roman_F184",
    "roman_K213",
)

OU_LSST_ROMAN_14_BANDS = (*OU_LSST_BANDS, *OU_ROMAN_BANDS)

OU_FLUX_COLUMNS = {
    "lsst_u": "lsst_flux_u",
    "lsst_g": "lsst_flux_g",
    "lsst_r": "lsst_flux_r",
    "lsst_i": "lsst_flux_i",
    "lsst_z": "lsst_flux_z",
    "lsst_y": "lsst_flux_y",
    "roman_W146": "roman_flux_W146",
    "roman_R062": "roman_flux_R062",
    "roman_Z087": "roman_flux_Z087",
    "roman_Y106": "roman_flux_Y106",
    "roman_J129": "roman_flux_J129",
    "roman_H158": "roman_flux_H158",
    "roman_F184": "roman_flux_F184",
    "roman_K213": "roman_flux_K213",
}

OU_TRUTH_COLUMNS = {
    "redshift": "redshift",
    "redshift_hubble": "redshiftHubble",
    "stellar_mass": "um_source_galaxy_obs_sm",
}

OU_MAIN_COLUMNS = (
    "galaxy_id",
    "ra",
    "dec",
    "redshift",
    "redshiftHubble",
    "peculiarVelocity",
    "um_source_galaxy_obs_sm",
    "MW_av",
    "MW_rv",
    "shear_1",
    "shear_2",
    "convergence",
)

OU_REQUIRED_MAIN_COLUMNS = (
    "galaxy_id",
    "ra",
    "dec",
    "redshift",
    "redshiftHubble",
    "um_source_galaxy_obs_sm",
)

OU_REQUIRED_FLUX_COLUMNS = ("galaxy_id", *OU_FLUX_COLUMNS.values())


def normalized_flux_column(band_name: str) -> str:
    """Return the normalized observed-flux column for an OpenUniverse band."""
    _validate_band_name(band_name)
    return f"flux_{band_name}"


def normalized_fluxerr_column(band_name: str) -> str:
    """Return the normalized flux-error column for an OpenUniverse band."""
    _validate_band_name(band_name)
    return f"fluxerr_{band_name}"


def normalized_flux_truth_column(band_name: str) -> str:
    """Return the normalized truth-flux column for an OpenUniverse band."""
    _validate_band_name(band_name)
    return f"flux_truth_{band_name}"


def normalized_mask_column(band_name: str) -> str:
    """Return the normalized validity-mask column for an OpenUniverse band."""
    _validate_band_name(band_name)
    return f"mask_{band_name}"


def _validate_band_name(band_name: str) -> None:
    if band_name not in OU_LSST_ROMAN_14_BANDS:
        raise ValueError(
            f"Unknown OpenUniverse LSST+Roman band {band_name!r}; "
            f"expected one of {OU_LSST_ROMAN_14_BANDS}"
        )
