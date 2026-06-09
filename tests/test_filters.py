from __future__ import annotations

import math

import h5py
import numpy as np

from euclid_dsps.filters import FilterCurve, _wave_unit_to_angstrom_factor, load_filter


def test_wave_unit_to_angstrom_factor() -> None:
    assert _wave_unit_to_angstrom_factor("angstrom") == 1.0
    assert _wave_unit_to_angstrom_factor("nm") == 10.0
    assert _wave_unit_to_angstrom_factor("micron") == 10_000.0


def test_filter_effective_wavelength() -> None:
    curve = FilterCurve(
        name="test",
        wave=np.asarray([4000.0, 5000.0, 6000.0]),
        transmission=np.asarray([0.0, 1.0, 0.0]),
        source="test",
    )

    assert math.isclose(curve.effective_wavelength, 5000.0)


def test_load_ascii_filter_sorts_and_clips(tmp_path) -> None:
    path = tmp_path / "filter.dat"
    path.write_text("6000 2.0\n4000 -1.0\n5000 0.5\n", encoding="utf-8")

    curve = load_filter("test", {"path": str(path)})

    assert curve.wave.tolist() == [4000.0, 5000.0, 6000.0]
    assert curve.transmission.tolist() == [0.0, 0.5, 1.0]


def test_load_csv_filter_from_value_added_style_file(tmp_path) -> None:
    path = tmp_path / "filter.csv"
    path.write_text("6000,1.0\n4000,0.0\n5000,0.5\n", encoding="utf-8")

    curve = load_filter("test", {"path": str(path), "kind": "ascii"})

    assert curve.wave.tolist() == [4000.0, 5000.0, 6000.0]
    assert curve.transmission.tolist() == [0.0, 0.5, 1.0]


def test_load_hdf5_group_filter(tmp_path) -> None:
    path = tmp_path / "filters.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("lsst_g")
        group["wave"] = np.asarray([5000.0, 4000.0, 6000.0])
        group["transmission"] = np.asarray([0.5, -1.0, 2.0])

    curve = load_filter(
        "lsst_g",
        {
            "kind": "hdf5_group",
            "path": str(path),
            "group": "lsst_g",
            "wave_dataset": "wave",
            "transmission_dataset": "transmission",
        },
    )

    assert curve.wave.tolist() == [4000.0, 5000.0, 6000.0]
    assert curve.transmission.tolist() == [0.0, 0.5, 2.0]
    assert curve.source.endswith("filters.hdf5:lsst_g")
