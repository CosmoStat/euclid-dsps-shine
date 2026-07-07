from __future__ import annotations

import pandas as pd
import pytest

from euclid_dsps.openuniverse.access import openuniverse_paths_for_hpix
from euclid_dsps.openuniverse.io import (
    JOIN_ATTR_KEY,
    join_main_and_flux,
    read_openuniverse_hpix,
)


def test_join_main_and_flux_conserves_galaxy_id() -> None:
    main = pd.DataFrame({"galaxy_id": [1, 2], "redshift": [0.1, 0.2]})
    flux = pd.DataFrame({"galaxy_id": [2, 1], "lsst_flux_u": [20.0, 10.0]})

    joined = join_main_and_flux(main, flux)

    assert joined["galaxy_id"].tolist() == [1, 2]
    assert joined["lsst_flux_u"].tolist() == [10.0, 20.0]
    assert joined.attrs[JOIN_ATTR_KEY]["main_rows"] == 2
    assert joined.attrs[JOIN_ATTR_KEY]["flux_rows"] == 2
    assert joined.attrs[JOIN_ATTR_KEY]["joined_rows"] == 2


def test_join_main_and_flux_requires_galaxy_id() -> None:
    main = pd.DataFrame({"id": [1]})
    flux = pd.DataFrame({"galaxy_id": [1]})

    with pytest.raises(ValueError, match="galaxy_id"):
        join_main_and_flux(main, flux)


def test_join_main_and_flux_rejects_duplicate_galaxy_id() -> None:
    main = pd.DataFrame({"galaxy_id": [1, 1]})
    flux = pd.DataFrame({"galaxy_id": [1]})

    with pytest.raises(ValueError, match="duplicate galaxy_id"):
        join_main_and_flux(main, flux)


def test_read_openuniverse_hpix_selects_joined_columns(tmp_path) -> None:
    main_path = tmp_path / "galaxy_5.parquet"
    flux_path = tmp_path / "galaxy_flux_5.parquet"
    pd.DataFrame({"galaxy_id": [1], "redshift": [0.1]}).to_parquet(main_path)
    pd.DataFrame({"galaxy_id": [1], "lsst_flux_u": [10.0]}).to_parquet(flux_path)

    frame = read_openuniverse_hpix(
        main_path,
        flux_path,
        columns=["galaxy_id", "lsst_flux_u"],
    )

    assert frame.to_dict(orient="records") == [{"galaxy_id": 1, "lsst_flux_u": 10.0}]
    assert frame.attrs[JOIN_ATTR_KEY]["joined_rows"] == 1


def test_openuniverse_paths_for_hpix_local_and_template() -> None:
    paths = openuniverse_paths_for_hpix(9812, "/data/ou")

    assert paths["main"] == "/data/ou/galaxy_9812.parquet"
    assert paths["flux"] == "/data/ou/galaxy_flux_9812.parquet"
    assert paths["sed"] == "/data/ou/galaxy_sed_9812.hdf5"

    templated = openuniverse_paths_for_hpix(
        9812,
        "/data/ou/{kind}/chunk_{hpix}/{filename}",
    )
    assert templated["main"] == "/data/ou/main/chunk_9812/galaxy_9812.parquet"
