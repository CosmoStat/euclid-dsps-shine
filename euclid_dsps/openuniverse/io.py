"""Parquet readers and joins for OpenUniverse SkyCatalog HEALPix files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

JOIN_ATTR_KEY = "openuniverse_join"


def read_openuniverse_main(path: str | Path) -> pd.DataFrame:
    """Read one OpenUniverse ``galaxy_<hpix>.parquet`` main file."""
    return pd.read_parquet(path)


def read_openuniverse_flux(path: str | Path) -> pd.DataFrame:
    """Read one OpenUniverse ``galaxy_flux_<hpix>.parquet`` flux file."""
    return pd.read_parquet(path)


def join_main_and_flux(main: pd.DataFrame, flux: pd.DataFrame) -> pd.DataFrame:
    """Inner-join OpenUniverse main and flux tables on unique ``galaxy_id``."""
    _validate_galaxy_id(main, "main")
    _validate_galaxy_id(flux, "flux")
    main_rows = int(len(main))
    flux_rows = int(len(flux))
    joined = main.merge(
        flux,
        on="galaxy_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_flux"),
    )
    joined.attrs[JOIN_ATTR_KEY] = {
        "main_rows": main_rows,
        "flux_rows": flux_rows,
        "joined_rows": int(len(joined)),
    }
    return joined


def read_openuniverse_hpix(
    main_path: str | Path,
    flux_path: str | Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read and join one HEALPix main/flux pair.

    ``columns`` names the final joined columns to return. ``galaxy_id`` is kept
    whenever it is requested explicitly or needed for the join.
    """
    joined = join_main_and_flux(
        read_openuniverse_main(main_path),
        read_openuniverse_flux(flux_path),
    )
    if columns is None:
        return joined
    selected_columns = list(dict.fromkeys(columns))
    missing = [column for column in selected_columns if column not in joined]
    if missing:
        raise ValueError(
            "Requested OpenUniverse joined columns are missing: "
            + ", ".join(sorted(missing))
        )
    subset = joined.loc[:, selected_columns].copy()
    subset.attrs[JOIN_ATTR_KEY] = dict(joined.attrs.get(JOIN_ATTR_KEY, {}))
    return subset


def _validate_galaxy_id(frame: pd.DataFrame, label: str) -> None:
    if "galaxy_id" not in frame.columns:
        raise ValueError(f"OpenUniverse {label} table must contain 'galaxy_id'")
    duplicates = frame["galaxy_id"].duplicated()
    if bool(duplicates.any()):
        duplicate_count = int(duplicates.sum())
        raise ValueError(
            f"OpenUniverse {label} table has {duplicate_count} duplicate galaxy_id rows"
        )
