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
