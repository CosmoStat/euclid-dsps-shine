"""Array extraction for prepared OpenUniverse photometry tables."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.observation_arrays import PhotometryArrays

from .schema import (
    OU_LSST_ROMAN_14_BANDS,
    normalized_flux_column,
    normalized_fluxerr_column,
    normalized_mask_column,
)


def load_openuniverse_photometry_arrays(
    path: str | Path,
    *,
    band_names: Sequence[str] = OU_LSST_ROMAN_14_BANDS,
    id_column: str = "galaxy_id",
    limit: int | None = None,
    truth_columns: Sequence[str] = (
        "redshift",
        "redshiftHubble",
        "redshift_truth",
        "redshift_hubble_truth",
        "stellar_mass",
    ),
) -> PhotometryArrays:
    """Load normalized OpenUniverse flux/error/mask columns into arrays."""
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(max(int(limit), 0))
    bands = tuple(str(band) for band in band_names)
    _validate_normalized_columns(frame, bands, id_column=id_column)
    truth = {
        str(column): frame[str(column)].to_numpy()
        for column in truth_columns
        if str(column) in frame
    }
    return PhotometryArrays(
        object_id=frame[id_column].to_numpy(),
        flux=np.stack(
            [frame[normalized_flux_column(band)].to_numpy(dtype=np.float32) for band in bands],
            axis=1,
        ),
        flux_err=np.stack(
            [
                frame[normalized_fluxerr_column(band)].to_numpy(dtype=np.float32)
                for band in bands
            ],
            axis=1,
        ),
        mask=np.stack(
            [frame[normalized_mask_column(band)].to_numpy(dtype=bool) for band in bands],
            axis=1,
        ),
        band_names=bands,
        truth=truth or None,
    )


def _validate_normalized_columns(
    frame: pd.DataFrame,
    band_names: tuple[str, ...],
    *,
    id_column: str,
) -> None:
    required = [id_column]
    for band in band_names:
        required.extend(
            [
                normalized_flux_column(band),
                normalized_fluxerr_column(band),
                normalized_mask_column(band),
            ]
        )
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(
            "Prepared OpenUniverse table is missing normalized photometry columns: "
            + ", ".join(sorted(missing))
        )
