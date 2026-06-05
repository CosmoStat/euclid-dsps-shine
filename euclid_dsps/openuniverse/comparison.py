"""FS2 versus OpenUniverse comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_magnitudes_from_flux(
    flux: np.ndarray,
    *,
    zero_point: float = 0.0,
) -> np.ndarray:
    """Convert positive arbitrary fluxes to relative magnitudes."""
    values = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = float(zero_point) - 2.5 * np.log10(values)
    return np.where(np.isfinite(values) & (values > 0.0), mag, np.nan)


def make_magnitude_histograms(
    magnitudes: dict[str, np.ndarray],
    *,
    bins: int | np.ndarray = 40,
) -> pd.DataFrame:
    """Return histogram rows for one or more magnitude arrays."""
    rows = []
    for label, values in magnitudes.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            continue
        counts, edges = np.histogram(finite, bins=bins)
        for index, count in enumerate(counts):
            rows.append(
                {
                    "series": str(label),
                    "bin_left": float(edges[index]),
                    "bin_right": float(edges[index + 1]),
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def make_color_color_diagrams(
    frame: pd.DataFrame,
    *,
    mag_a: str,
    mag_b: str,
    mag_c: str,
    mag_d: str,
) -> pd.DataFrame:
    """Return finite color-color points from four magnitude columns."""
    required = [mag_a, mag_b, mag_c, mag_d]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError("Missing magnitude columns: " + ", ".join(missing))
    x = frame[mag_a].to_numpy(dtype=float) - frame[mag_b].to_numpy(dtype=float)
    y = frame[mag_c].to_numpy(dtype=float) - frame[mag_d].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    return pd.DataFrame({"color_x": x[finite], "color_y": y[finite]})


def compare_redshift_distributions(
    fs2_redshift: np.ndarray,
    openuniverse_redshift: np.ndarray,
    *,
    bins: int | np.ndarray = 40,
) -> pd.DataFrame:
    """Return redshift histogram rows for FS2 and OpenUniverse."""
    return make_magnitude_histograms(
        {
            "fs2": np.asarray(fs2_redshift, dtype=float),
            "openuniverse": np.asarray(openuniverse_redshift, dtype=float),
        },
        bins=bins,
    ).rename(
        columns={
            "bin_left": "z_left",
            "bin_right": "z_right",
        }
    )
