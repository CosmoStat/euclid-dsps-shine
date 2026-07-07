from __future__ import annotations

from euclid_dsps.openuniverse.schema import (
    OU_FLUX_COLUMNS,
    OU_LSST_BANDS,
    OU_LSST_ROMAN_14_BANDS,
    OU_ROMAN_BANDS,
    OU_TRUTH_COLUMNS,
    normalized_flux_column,
    normalized_flux_truth_column,
    normalized_fluxerr_column,
    normalized_mask_column,
)


def test_openuniverse_lsst_roman_schema_contract() -> None:
    assert OU_LSST_BANDS == (
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
    )
    assert len(OU_ROMAN_BANDS) == 8
    assert len(OU_LSST_ROMAN_14_BANDS) == 14
    assert OU_FLUX_COLUMNS["roman_K213"] == "roman_flux_K213"
    assert OU_TRUTH_COLUMNS["stellar_mass"] == "um_source_galaxy_obs_sm"


def test_openuniverse_normalized_column_helpers() -> None:
    assert normalized_flux_truth_column("lsst_u") == "flux_truth_lsst_u"
    assert normalized_flux_column("roman_W146") == "flux_roman_W146"
    assert normalized_fluxerr_column("roman_W146") == "fluxerr_roman_W146"
    assert normalized_mask_column("roman_W146") == "mask_roman_W146"
