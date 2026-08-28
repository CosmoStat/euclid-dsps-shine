from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from euclid_dsps.amortized.train import build_training_split


def test_explicit_split_does_not_read_configured_redshift_truth(
    tmp_path, monkeypatch
) -> None:
    catalogue = tmp_path / "catalog.parquet"
    pq.write_table(
        pa.table({"object_id": np.arange(8), "z_obs": np.arange(8, dtype=float)}),
        catalogue,
    )
    train = tmp_path / "train.npy"
    validation = tmp_path / "validation.npy"
    np.save(train, np.asarray([0, 1, 2, 3, 4, 5]))
    np.save(validation, np.asarray([6, 7]))

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit split attempted to read redshift truth")

    monkeypatch.setattr(
        "euclid_dsps.amortized.train._read_redshift_column",
        forbidden,
    )
    split = build_training_split(
        {
            "catalog_path": str(catalogue),
            "redshift": {"column": "z_obs"},
            "amortized": {"data": {}, "training": {}},
        },
        limit=None,
        seed=1,
        train_indices_file=train,
        validation_indices_file=validation,
    )

    assert split.redshift_column is None
    assert split.train_redshift.size == 0
    assert split.validation_redshift.size == 0
    assert split.selection_mode == "explicit_train_validation_files_no_truth"


def test_explicit_cross_catalog_split_uses_independent_index_namespaces(
    tmp_path, monkeypatch
) -> None:
    train_catalogue = tmp_path / "train.parquet"
    validation_catalogue = tmp_path / "test.parquet"
    for path in (train_catalogue, validation_catalogue):
        pq.write_table(pa.table({"object_id": np.arange(6)}), path)
    train = tmp_path / "full_train.npy"
    validation = tmp_path / "confirmation.npy"
    np.save(train, np.arange(6, dtype=np.int64))
    np.save(validation, np.asarray([0, 1, 2], dtype=np.int64))

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit cross-catalog split attempted to read truth")

    monkeypatch.setattr(
        "euclid_dsps.amortized.train._read_redshift_column",
        forbidden,
    )
    split = build_training_split(
        {
            "catalog_path": str(train_catalogue),
            "redshift": {"column": "z_obs"},
            "amortized": {"data": {}, "training": {}},
        },
        limit=None,
        seed=1,
        train_indices_file=train,
        validation_indices_file=validation,
        validation_catalog_path=validation_catalogue,
    )

    assert np.array_equal(split.train_indices, np.arange(6))
    assert np.array_equal(split.validation_indices, np.asarray([0, 1, 2]))
    assert split.selection_mode == "explicit_cross_catalog_train_validation_no_truth"
    assert split.validation_catalog_path == str(validation_catalogue)
