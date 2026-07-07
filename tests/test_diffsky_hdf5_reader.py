from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from euclid_dsps.diffsky_data.hdf5_reader import inspect_hdf5_file, load_hdf5_columns


def test_hdf5_reader_finds_interesting_columns(tmp_path: Path) -> None:
    path = tmp_path / "mini.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("data")
        group.create_dataset("redshift_true", data=np.asarray([0.1, 0.2]))
        group.create_dataset("logsm_obs", data=np.asarray([9.0, 10.0]))

    report = inspect_hdf5_file(path)
    loaded = load_hdf5_columns(path, ["data/redshift_true"], limit=1)

    names = {item["name"] for item in report["datasets"] if item["interesting"]}
    assert {"data/redshift_true", "data/logsm_obs"} <= names
    assert loaded["data/redshift_true"].tolist() == [0.1]
