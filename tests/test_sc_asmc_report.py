from __future__ import annotations

import numpy as np

from euclid_dsps.amortized.posterior_bank import (
    merge_posterior_bank_shards,
    write_posterior_bank_shard,
)
from euclid_dsps.amortized.sc_asmc_report import (
    _weighted_correlation,
    summarize_posterior_bank,
)
from tests.test_posterior_bank import _provenance, _shard


def test_report_streams_full_bank_method_and_cost_diagnostics(tmp_path) -> None:
    write_posterior_bank_shard(tmp_path, 0, _shard(0, 3), _provenance())
    write_posterior_bank_shard(tmp_path, 1, _shard(3, 3), _provenance())
    paths = sorted((tmp_path / "shards").iterdir())
    merge_posterior_bank_shards(
        tmp_path,
        paths,
        expected_row_indices=np.arange(6),
    )

    summary = summarize_posterior_bank(tmp_path / "posterior_bank_manifest.json")

    assert summary["objects"] == 6
    assert summary["resolved_fraction"] == 1.0
    assert summary["method_counts"] == {
        "IS": 2,
        "primary SMC": 2,
        "fallback SMC": 2,
        "extended SMC": 0,
        "unresolved": 0,
    }
    assert summary["total_dsps_evaluations"] == 600


def test_beta_weighted_correlation_differs_from_parent_when_weights_vary() -> None:
    values = np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0], [4.0, 5.0]])
    parent = _weighted_correlation(values, None)
    selected = _weighted_correlation(values, np.asarray([0.01, 0.01, 0.01, 0.97]))

    assert parent.shape == (2, 2)
    assert selected.shape == (2, 2)
    assert not np.allclose(parent, selected)
