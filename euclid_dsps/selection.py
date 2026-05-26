"""Galaxy selection utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_galaxy_row(
    df: pd.DataFrame,
    band_columns: list[str],
    index: int | None = None,
    require_positive_flux: bool = True,
    nondetection_policy: str | None = None,
    sort_by_flux: str | None = None,
) -> tuple[int, pd.Series]:
    """Select one galaxy for a smoke-test run."""
    work = df.copy()
    policy = (
        str(nondetection_policy)
        if nondetection_policy is not None
        else ("drop" if require_positive_flux else "gaussian_flux")
    )
    if policy == "upper_limit":
        raise NotImplementedError("Upper-limit likelihood is not implemented")
    mask = np.ones(len(work), dtype=bool)
    if policy == "drop":
        for column in band_columns:
            mask &= np.isfinite(work[column].to_numpy()) & (work[column].to_numpy() > 0)
    elif policy == "gaussian_flux":
        for column in band_columns:
            mask &= np.isfinite(work[column].to_numpy())
    else:
        raise ValueError(f"Unsupported nondetection_policy: {policy}")
    work = work.loc[mask]
    if work.empty:
        raise ValueError("No galaxy satisfies the configured selection.")

    if index is not None:
        if index in work.index:
            return int(index), work.loc[index]
        if 0 <= index < len(work):
            row = work.iloc[index]
            return int(row.name), row
        raise IndexError(f"Configured index {index} is outside the selected catalog.")

    if sort_by_flux:
        row = work.sort_values(sort_by_flux, ascending=False).iloc[0]
    else:
        row = work.iloc[0]
    return int(row.name), row
