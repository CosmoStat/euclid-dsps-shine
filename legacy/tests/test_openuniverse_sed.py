from __future__ import annotations

import h5py
import numpy as np
import pytest

from euclid_dsps.openuniverse.sed import (
    OU_SED_COMPONENT_NAMES,
    component_names_for_count,
    inventory_openuniverse_sed,
    read_sed_components,
)


def test_inventory_openuniverse_sed_reads_bounded_metadata(tmp_path) -> None:
    path = tmp_path / "galaxy_sed_12345.hdf5"
    _write_mock_sed(path)

    inventory = inventory_openuniverse_sed(path, sample_limit=1)

    assert inventory.exists
    assert inventory.top_level_groups == ("galaxy", "meta")
    assert inventory.wavelength_dataset == "meta/wave_list"
    assert inventory.wavelength_size == 4
    assert inventory.galaxy_prefix_count == 1
    assert inventory.sed_dataset_count == 2
    assert len(inventory.sample_datasets) == 1
    assert inventory.sample_datasets[0].shape == (3, 4)


def test_inventory_openuniverse_sed_missing_file_is_explicit(tmp_path) -> None:
    inventory = inventory_openuniverse_sed(tmp_path / "missing.hdf5")

    assert not inventory.exists
    assert inventory.sed_dataset_count is None
    assert inventory.sample_datasets == ()


def test_read_sed_components_uses_openuniverse_group_prefix(tmp_path) -> None:
    path = tmp_path / "galaxy_sed_12345.hdf5"
    _write_mock_sed(path)

    components = read_sed_components(path, [12345000123456])

    assert components.galaxy_id.tolist() == [12345000123456]
    assert components.wavelength.tolist() == [500.0, 700.0, 900.0, 1100.0]
    assert components.sed.shape == (1, 3, 4)
    assert components.component_names == OU_SED_COMPONENT_NAMES
    assert components.sed_paths == ("galaxy/123450001/12345000123456",)


def test_read_sed_components_missing_modes(tmp_path) -> None:
    path = tmp_path / "galaxy_sed_12345.hdf5"
    _write_mock_sed(path)

    skipped = read_sed_components(path, [999], missing="skip")
    assert skipped.galaxy_id.size == 0

    with pytest.raises(KeyError, match="galaxy_id=999"):
        read_sed_components(path, [999], missing="raise")


def test_component_names_fallback_for_nonstandard_component_count() -> None:
    assert component_names_for_count(2) == ("component_0", "component_1")


def _write_mock_sed(path) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "meta/wave_list",
            data=np.asarray([500.0, 700.0, 900.0, 1100.0], dtype=np.float32),
        )
        group = handle.create_group("galaxy/123450001")
        group.create_dataset(
            "12345000123456",
            data=np.asarray(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                dtype=np.float32,
            ),
        )
        group.create_dataset(
            "12345000123457",
            data=np.ones((3, 4), dtype=np.float32),
        )
