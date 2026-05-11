"""Filter loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
    """Load an exact HDF5/FITS/DAT curve or build an approximate top-hat."""
    kind = filter_config.get("kind", "auto")
    if kind == "hdf5" or (
        "path" in filter_config
        and str(filter_config["path"]).endswith((".h5", ".hdf5"))
    ):
        from dsps import load_transmission_curve

        path = Path(filter_config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Filter file not found for {name}: {path}")
        curve = load_transmission_curve(fn=str(path))
        return FilterCurve(
            name=name,
            wave=np.asarray(curve.wave, dtype=float),
            transmission=np.asarray(curve.transmission, dtype=float),
            source=str(path),
        )

    if kind == "fits" or (
        "path" in filter_config
        and str(filter_config["path"]).endswith((".fits", ".fit", ".fts"))
    ):
        path = Path(filter_config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Filter file not found for {name}: {path}")
        return load_fits_filter(name, path, filter_config)

    if (
        kind == "ascii"
        or kind == "dat"
        or (
            "path" in filter_config
            and str(filter_config["path"]).endswith((".dat", ".txt", ".ascii"))
        )
    ):
        path = Path(filter_config["path"])
        if not path.exists():
            raise FileNotFoundError(f"Filter file not found for {name}: {path}")
        return load_ascii_filter(name, path, filter_config)

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
        return FilterCurve(
            name=name, wave=wave, transmission=transmission, source=source
        )

    raise ValueError(f"Unsupported filter kind for {name}: {kind}")


def load_ascii_filter(
    name: str, path: Path, filter_config: dict[str, Any]
) -> FilterCurve:
    """Load two-column ASCII throughput files."""
    data = np.loadtxt(path)
    wave = np.asarray(data[:, 0], dtype=float)
    transmission = np.asarray(data[:, 1], dtype=float)
    wave = wave * _wave_unit_to_angstrom_factor(
        str(filter_config.get("wave_unit", "angstrom"))
    )
    order = np.argsort(wave)
    wave = wave[order]
    transmission = np.clip(transmission[order], 0.0, 1.0)
    mask = np.isfinite(wave) & np.isfinite(transmission)
    return FilterCurve(
        name=name, wave=wave[mask], transmission=transmission[mask], source=str(path)
    )


def load_fits_filter(
    name: str, path: Path, filter_config: dict[str, Any]
) -> FilterCurve:
    """Load Euclid throughput FITS files and convert wavelength to Angstrom."""
    from astropy.io import fits

    data = fits.getdata(path, int(filter_config.get("hdu", 1)))
    wave_column = filter_config.get("wave_column", "WAVE")
    throughput_column = filter_config.get("throughput_column", "T_TOTAL")
    wave = np.asarray(data[wave_column], dtype=float)
    transmission = np.asarray(data[throughput_column], dtype=float)
    wave = wave * _wave_unit_to_angstrom_factor(
        str(filter_config.get("wave_unit", "nm"))
    )
    order = np.argsort(wave)
    wave = wave[order]
    transmission = np.clip(transmission[order], 0.0, 1.0)
    mask = np.isfinite(wave) & np.isfinite(transmission)
    return FilterCurve(
        name=name, wave=wave[mask], transmission=transmission[mask], source=str(path)
    )


def _wave_unit_to_angstrom_factor(unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"angstrom", "ang", "aa", "a"}:
        return 1.0
    if normalized in {"nm", "nanometer", "nanometers"}:
        return 10.0
    if normalized in {"micron", "microns", "um"}:
        return 10_000.0
    raise ValueError(f"Unsupported filter wavelength unit: {unit}")
