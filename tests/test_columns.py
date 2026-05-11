from __future__ import annotations

from test_config import minimal_config

from euclid_dsps.columns import CATALOG_COLUMN_BY_NAME, CATALOG_COLUMNS
from euclid_dsps.config import normalize_config
from euclid_dsps.io import required_catalog_columns


def test_catalog_metadata_names_are_unique() -> None:
    names = [column.name for column in CATALOG_COLUMNS]

    assert len(names) == len(set(names))


def test_catalog_metadata_covers_default_contract_columns() -> None:
    config = normalize_config(minimal_config())

    assert set(required_catalog_columns(config)).issubset(CATALOG_COLUMN_BY_NAME)


def test_catalog_metadata_documents_key_units() -> None:
    assert CATALOG_COLUMN_BY_NAME["ra_gal"].unit == "deg"
    assert CATALOG_COLUMN_BY_NAME["z_phz"].unit == "dimensionless"
    assert CATALOG_COLUMN_BY_NAME["euclid_vis"].unit == "erg s^-1 cm^-2 Hz^-1"
    assert (
        CATALOG_COLUMN_BY_NAME["euclid_nisp_h_abs"].unit
        == "erg s^-1 cm^-2 Hz^-1 at 10 pc"
    )
    assert CATALOG_COLUMN_BY_NAME["sed_cosmos_1"].unit == "index"
    assert CATALOG_COLUMN_BY_NAME["frac_cosmos_1"].unit == "fraction"
    assert CATALOG_COLUMN_BY_NAME["log_sfr_true"].unit == "log10(Msun yr^-1)"
