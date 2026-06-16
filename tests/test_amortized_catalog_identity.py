from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.amortized.catalog_identity import (
    select_catalog_row_indices,
    write_truth_snapshot,
)


def _config(path):
    return {
        "catalog_path": str(path),
        "dataset": {"id_column": "object_id"},
        "truth": {"redshift_column": "redshift_true"},
        "amortized": {
            "data": {
                "stratify_column": "redshift_true",
                "redshift_bins": [0.0, 0.5, 1.0],
            },
            "inference": {
                "redshift_bins": [0.0, 0.5, 1.0],
            },
        },
    }


def test_balanced_inference_selection_and_truth_snapshot(tmp_path) -> None:
    catalog = tmp_path / "catalog.parquet"
    pd.DataFrame(
        {
            "object_id": np.arange(10) + 100,
            "redshift_true": [0.1, 0.2, 0.3, 0.4, 0.45, 0.6, 0.7, 0.8, 0.9, 0.95],
            "logsm_true": np.arange(10, dtype=float),
        }
    ).to_parquet(catalog, index=False)
    config = _config(catalog)

    selected, summary = select_catalog_row_indices(
        config,
        limit=4,
        selection_mode="stratified_redshift",
        stratified_strategy="balanced",
        seed=123,
    )
    truth = write_truth_snapshot(
        tmp_path,
        config,
        row_indices=selected,
        limit=4,
        batch_size=3,
    )

    assert selected is not None
    assert len(selected) == 4
    assert summary["selected_rows"] == 4
    assert set(truth["row_index"]) == set(selected.tolist())
    assert "object_id" in truth
    assert int(truth["row_index"].min()) >= 0
