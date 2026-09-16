from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from euclid_dsps.amortized.sc_asmc_manifest import (
    prepare_sc_asmc_manifest,
    validate_sc_asmc_manifest,
)
from euclid_dsps.config import load_config
from euclid_dsps.photometry import abmag_to_fnu_cgs


def _write_catalog(path: Path, config: dict) -> None:
    rng = np.random.default_rng(8)
    rows = 640
    limit = float(np.asarray(abmag_to_fnu_cgs(25.0)))
    payload = {}
    for band in config["bands"]:
        payload[band["column"]] = rng.normal(4.0 * limit, limit, rows)
        payload[band["error_column"]] = rng.uniform(0.1 * limit, limit, rows)
        payload[f"mask_{band['name']}"] = np.ones(rows, dtype=bool)
    payload["flux_lsst_r"] = np.linspace(1.1 * limit, 8.0 * limit, rows)
    payload["object_id"] = np.arange(10_000, 10_000 + rows)
    # This deliberately exists but must never be included in columns_read.
    payload["z_obs"] = np.linspace(0.0, 6.0, rows)
    pq.write_table(pa.table(payload), path)


def test_manifest_is_observed_only_complete_and_resumable(tmp_path) -> None:
    config = load_config("configs/experiments/feniks_sc_asmc_em_r25.yaml")
    catalogue = tmp_path / "catalog.parquet"
    _write_catalog(catalogue, config)
    out = tmp_path / "manifest"

    manifest = prepare_sc_asmc_manifest(
        config,
        out,
        catalogue_path=catalogue,
        n_estep_shards=4,
    )
    validated = validate_sc_asmc_manifest(out / "run_manifest.json")
    resumed = prepare_sc_asmc_manifest(
        config,
        out,
        catalogue_path=catalogue,
        n_estep_shards=4,
    )

    assert manifest == validated == resumed
    assert manifest["truth_columns_requested"] == []
    assert "z_obs" not in manifest["observed_columns_read"]
    assert manifest["preflight"]["objects"] == 512
    assert sum(manifest["e_step_shards"]["object_counts"]) == 640
    assert manifest["features"]["input_dim"] == 54
    assert len(manifest["config_sha256"]) == 64
    assert manifest["code"]["commit"]
    assert isinstance(manifest["code"]["working_tree_dirty"], bool)

    changed = deepcopy(config)
    changed["amortized"]["sc_asmc_em"]["preflight"][
        "projected_non_estep_overhead_fraction"
    ] = 0.21
    with pytest.raises(ValueError, match="another configuration"):
        prepare_sc_asmc_manifest(
            changed,
            out,
            catalogue_path=catalogue,
            n_estep_shards=4,
        )
