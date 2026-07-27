from __future__ import annotations

import numpy as np
import pandas as pd

from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES
from scripts.build_feniks_spline15d_amortized_catalog import build_catalogs


def test_build_catalogs_joins_exact_truth_by_object_id(tmp_path) -> None:
    source_dir = tmp_path / "source"
    spline_dir = tmp_path / "spline"
    out_dir = tmp_path / "out"
    source_dir.mkdir()
    spline_dir.mkdir()
    ids = np.asarray([20, 10, 30])
    source = pd.DataFrame({"object_id": ids, "flux_lsst_g": [1.0, 2.0, 3.0]})
    truth = pd.DataFrame({"object_id": ids[::-1]})
    for index, name in enumerate(SPLINE15D_PARAMETER_NAMES):
        truth[name] = np.arange(3, dtype=float) + index
    for split in ("train", "validation", "test"):
        source.to_parquet(source_dir / f"{split}.parquet", index=False)
        truth.to_parquet(spline_dir / f"{split}_exact.parquet", index=False)

    contract = build_catalogs(
        source_dir,
        spline_dir,
        out_dir,
        splits=("train", "validation", "test"),
    )
    joined = pd.read_parquet(out_dir / "train.parquet")

    assert contract["regenerates_diffsky"] is False
    assert joined["object_id"].tolist() == ids.tolist()
    assert set(SPLINE15D_PARAMETER_NAMES).issubset(joined.columns)
    expected = truth.set_index("object_id").loc[ids, "z_obs"].to_numpy()
    np.testing.assert_allclose(joined["z_obs"], expected)
