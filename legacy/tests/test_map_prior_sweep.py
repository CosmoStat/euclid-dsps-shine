from __future__ import annotations

import json

import pandas as pd

from euclid_dsps.amortized.map_prior_sweep import finalize_map_prior_weight_sweep


def test_finalize_map_prior_weight_sweep_combines_shards(tmp_path) -> None:
    root = tmp_path / "sweep"
    weight_dir = root / "prior_weight_0"
    for shard_index in range(2):
        shard = weight_dir / f"shard_{shard_index:03d}"
        shard.mkdir(parents=True)
        pd.DataFrame(
            {
                "row_index": [shard_index],
                "object_id": [100 + shard_index],
                "map_photometric_nll": [1.0 + shard_index],
                "map_prior_logprob": [-2.0],
                "map_chi2": [3.0],
                "z_obs": [0.1 + 0.01 * shard_index],
            }
        ).to_parquet(shard / "map_estimates.parquet", index=False)
        pd.DataFrame(
            {
                "row_index": [shard_index],
                "redshift_true": [0.1],
            }
        ).to_parquet(shard / "inference_truth.parquet", index=False)
        (shard / "map_summary.json").write_text(
            json.dumps({"prior_weight": 0.0}),
            encoding="utf-8",
        )

    payload = finalize_map_prior_weight_sweep(root, verbose=False)

    combined = pd.read_parquet(weight_dir / "map_estimates.parquet")
    summary = pd.read_csv(root / "map_prior_weight_sweep_summary.csv")
    assert len(combined) == 2
    assert payload["n_weight_runs"] == 1
    assert summary.loc[0, "prior_weight"] == 0.0
    assert summary.loc[0, "n_objects"] == 2
