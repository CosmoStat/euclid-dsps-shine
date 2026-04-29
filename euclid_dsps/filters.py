"""Filter loading helpers.

Exact HDF5 transmission curves are preferred. Approximate top-hat filters are
available so the pipeline can run before all Euclid curves are staged locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dsps import load_transmission_curve


@dataclass(frozen=True)
class FilterCurve:
    name: str
    wave: np.ndarray
    transmission: np.ndarray
    source: str

    @property
    def effective_wavelength(self) -> float:
        """Transmission-weighted central wavelength in Angstrom."""
        norm = np.trapezoid(self.transmission, self.wave)
        if not np.isfinite(norm) or norm <= 0:
            return float(np.nanmean(self.wave))
        return float(np.trapezoid(self.wave * self.transmission, self.wave) / norm)


APPROX_FILTERS_ANGSTROM = {
    "euclid_vis": (5500.0, 9000.0),
    "euclid_y": (9200.0, 11400.0),
    "euclid_j": (11400.0, 13700.0),
    "euclid_h": (13700.0, 20000.0),
    "euclid_nisp_y": (9200.0, 11400.0),
    "euclid_nisp_j": (11400.0, 13700.0),
    "euclid_nisp_h": (13700.0, 20000.0),
    "lsst_u": (3200.0, 4000.0),
    "lsst_g": (4000.0, 5500.0),
    "lsst_r": (5500.0, 7000.0),
    "lsst_i": (6900.0, 8200.0),
    "lsst_z": (8200.0, 9300.0),
    "lsst_y": (9500.0, 10500.0),
}


def load_filters(band_configs: list[dict[str, Any]]) -> dict[str, FilterCurve]:
    """Load all filters declared in the config."""
    filters: dict[str, FilterCurve] = {}
    for band in band_configs:
        name = band["name"]
        filters[name] = load_filter(name, band.get("filter", {}))
    return filters


def load_filter(name: str, filter_config: dict[str, Any]) -> FilterCurve:
    """Load an exact HDF5 curve or build an approximate top-hat."""
    kind = filter_config.get("kind", "auto")
    if kind == "hdf5" or "path" in filter_config:
        path = Path(filter_config["path"])
        curve = load_transmission_curve(fn=str(path))
        return FilterCurve(
            name=name,
            wave=np.asarray(curve.wave, dtype=float),
            transmission=np.asarray(curve.transmission, dtype=float),
            source=str(path),
        )

    if kind in {"auto", "tophat"}:
        wave_min = filter_config.get("wave_min")
        wave_max = filter_config.get("wave_max")
        if wave_min is None or wave_max is None:
            if name not in APPROX_FILTERS_ANGSTROM:
                raise ValueError(f"No approximate filter range known for {name}")
            wave_min, wave_max = APPROX_FILTERS_ANGSTROM[name]
        n_wave = int(filter_config.get("n_wave", 512))
        wave = np.linspace(float(wave_min), float(wave_max), n_wave)
        transmission = np.ones_like(wave)
        source = f"approx_tophat_{float(wave_min):.0f}_{float(wave_max):.0f}A"
        return FilterCurve(name=name, wave=wave, transmission=transmission, source=source)

    raise ValueError(f"Unsupported filter kind for {name}: {kind}")
