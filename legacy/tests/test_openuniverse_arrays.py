from __future__ import annotations

import json

import numpy as np
import pandas as pd

from euclid_dsps.openuniverse.arrays import load_openuniverse_photometry_arrays
from euclid_dsps.openuniverse.cli import main as ou_cli_main
from euclid_dsps.openuniverse.schema import OU_LSST_ROMAN_14_BANDS


def test_load_openuniverse_photometry_arrays_b14(tmp_path) -> None:
    path = tmp_path / "ou.parquet"
    _write_normalized_openuniverse(path, n_rows=3)

    arrays = load_openuniverse_photometry_arrays(path, limit=2)

    assert arrays.object_id.tolist() == [100, 101]
    assert arrays.flux.shape == (2, 14)
    assert arrays.flux_err.shape == (2, 14)
    assert arrays.mask.shape == (2, 14)
    assert arrays.band_names == OU_LSST_ROMAN_14_BANDS
    assert arrays.truth is not None
    assert arrays.truth["redshift"].tolist() == [0.1, 0.2]


def test_openuniverse_feature_stats_cli_writes_feature_dim_28(tmp_path) -> None:
    path = tmp_path / "ou.parquet"
    out = tmp_path / "feature_stats.json"
    _write_normalized_openuniverse(path, n_rows=4)

    ou_cli_main(
        [
            "feature-stats",
            "--input",
            str(path),
            "--out",
            str(out),
            "--limit",
            "4",
        ]
    )

    summary = json.loads(out.with_suffix(".summary.json").read_text())
    assert out.exists()
    assert summary["n_bands"] == 14
    assert summary["feature_dim"] == 28
    assert summary["truth_columns"] == [
        "redshift",
        "redshiftHubble",
        "stellar_mass",
    ]


def _write_normalized_openuniverse(path, *, n_rows: int) -> None:
    frame = pd.DataFrame(
        {
            "galaxy_id": np.arange(100, 100 + n_rows),
            "redshift": np.linspace(0.1, 0.3, n_rows),
            "redshiftHubble": np.linspace(0.11, 0.31, n_rows),
            "stellar_mass": np.linspace(1.0e9, 2.0e9, n_rows),
        }
    )
    for band_index, band in enumerate(OU_LSST_ROMAN_14_BANDS):
        frame[f"flux_{band}"] = 100.0 + 10.0 * band_index + np.arange(n_rows)
        frame[f"fluxerr_{band}"] = 2.0 + 0.1 * band_index
        frame[f"mask_{band}"] = True
    frame.to_parquet(path, index=False)
