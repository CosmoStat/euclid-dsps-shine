"""Filter-curve loading for OpenUniverse data-side photometry checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import OU_LSST_ROMAN_14_BANDS


@dataclass(frozen=True)
class OpenUniverseFilterCurve:
    """One OpenUniverse filter response curve."""

    band_name: str
    wave_angstrom: np.ndarray
    transmission: np.ndarray
    source: str
    approximate: bool = False


DEFAULT_LSST_FILTER_FILENAMES = {
    "lsst_u": "LSST_LSST.u.dat",
    "lsst_g": "LSST_LSST.g.dat",
    "lsst_r": "LSST_LSST.r.dat",
    "lsst_i": "LSST_LSST.i.dat",
    "lsst_z": "LSST_LSST.z.dat",
    "lsst_y": "LSST_LSST.y.dat",
}

DEFAULT_ROMAN_FILTER_FILENAMES = {
    "roman_R062": "Roman_WFI.F062.dat",
    "roman_Z087": "Roman_WFI.F087.dat",
    "roman_Y106": "Roman_WFI.F106.dat",
    "roman_J129": "Roman_WFI.F129.dat",
    "roman_H158": "Roman_WFI.F158.dat",
    "roman_F184": "Roman_WFI.F184.dat",
    "roman_K213": "Roman_WFI.F213.dat",
    "roman_W146": "Roman_WFI.F146.dat",
}

# Coarse top-hat smoke-test ranges only. These are not science-grade Roman WFI
# response curves and are disabled unless the caller opts in.
APPROX_ROMAN_FILTERS_ANGSTROM = {
    "roman_R062": (4800.0, 7600.0),
    "roman_Z087": (7600.0, 9770.0),
    "roman_Y106": (9270.0, 11920.0),
    "roman_J129": (11310.0, 14540.0),
    "roman_H158": (13800.0, 17740.0),
    "roman_F184": (16830.0, 20000.0),
    "roman_K213": (19500.0, 23000.0),
    "roman_W146": (9270.0, 20000.0),
}


def load_openuniverse_filter_curves(
    band_names: Sequence[str],
    *,
    filter_root: str | Path = "filters",
    filter_paths: Mapping[str, str | Path] | None = None,
    allow_approx_filters: bool = False,
) -> dict[str, OpenUniverseFilterCurve]:
    """Load OpenUniverse filter curves for the requested bands.

    LSST defaults use repository filter files. Roman bands require explicit
    paths unless ``allow_approx_filters`` is enabled for smoke tests.
    """
    filters = {}
    for band in tuple(str(name) for name in band_names):
        if band not in OU_LSST_ROMAN_14_BANDS:
            raise ValueError(f"Unsupported OpenUniverse band {band!r}")
        filters[band] = load_openuniverse_filter_curve(
            band,
            filter_root=filter_root,
            filter_paths=filter_paths or {},
            allow_approx_filters=allow_approx_filters,
        )
    return filters


def load_openuniverse_filter_curve(
    band_name: str,
    *,
    filter_root: str | Path = "filters",
    filter_paths: Mapping[str, str | Path] | None = None,
    allow_approx_filters: bool = False,
) -> OpenUniverseFilterCurve:
    """Load one OpenUniverse filter curve."""
    paths = filter_paths or {}
    if band_name in paths:
        return _load_ascii_filter(band_name, Path(paths[band_name]))

    if band_name in DEFAULT_LSST_FILTER_FILENAMES:
        path = Path(filter_root) / DEFAULT_LSST_FILTER_FILENAMES[band_name]
        if path.exists():
            return _load_ascii_filter(band_name, path)

    if band_name in DEFAULT_ROMAN_FILTER_FILENAMES:
        path = Path(filter_root) / DEFAULT_ROMAN_FILTER_FILENAMES[band_name]
        if path.exists():
            return _load_ascii_filter(band_name, path)

    if allow_approx_filters and band_name in APPROX_ROMAN_FILTERS_ANGSTROM:
        low, high = APPROX_ROMAN_FILTERS_ANGSTROM[band_name]
        wave = np.linspace(low, high, 512, dtype=float)
        return OpenUniverseFilterCurve(
            band_name=band_name,
            wave_angstrom=wave,
            transmission=np.ones_like(wave),
            source=f"approx_tophat_{low:.0f}_{high:.0f}A",
            approximate=True,
        )

    raise FileNotFoundError(
        f"No filter curve available for OpenUniverse band {band_name!r}. "
        "Pass --filter band=path for exact curves, or use "
        "--allow-approx-filters for non-science smoke tests."
    )


def parse_filter_path_overrides(overrides: Sequence[str] | None) -> dict[str, Path]:
    """Parse CLI overrides of the form ``band=/path/to/filter.dat``."""
    parsed: dict[str, Path] = {}
    for override in overrides or ():
        if "=" not in override:
            raise ValueError(
                f"Filter override {override!r} must have form band=/path/to/filter"
            )
        band, path = override.split("=", 1)
        band = band.strip()
        if band not in OU_LSST_ROMAN_14_BANDS:
            raise ValueError(f"Unsupported OpenUniverse band in override: {band!r}")
        parsed[band] = Path(path.strip())
    return parsed


def _load_ascii_filter(band_name: str, path: Path) -> OpenUniverseFilterCurve:
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found for {band_name}: {path}")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Filter file must have at least two columns: {path}")
    wave = np.asarray(data[:, 0], dtype=float)
    transmission = np.asarray(data[:, 1], dtype=float)
    order = np.argsort(wave)
    wave = wave[order]
    transmission = np.clip(transmission[order], 0.0, np.inf)
    finite = np.isfinite(wave) & np.isfinite(transmission)
    if finite.sum() < 2:
        raise ValueError(f"Filter file has fewer than two finite rows: {path}")
    return OpenUniverseFilterCurve(
        band_name=band_name,
        wave_angstrom=wave[finite],
        transmission=transmission[finite],
        source=str(path),
        approximate=False,
    )
