from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from euclid_dsps.photometry import abmag_to_fnu_cgs
from scripts.feniks_avi_next_validation import (
    _corner_compare,
    _weighted_quantile,
    build_selection_closure,
    observed_selection_mask,
)


def _frame() -> pd.DataFrame:
    threshold = float(abmag_to_fnu_cgs(29.0))
    data = {
        "object_id": [10, 11, 12, 13],
        "flux_lsst_r": [threshold * 0.5, threshold * 2, threshold * 3, threshold * 4],
        "fluxerr_lsst_r": [threshold / 5] * 4,
        "mask_lsst_r": [True, True, False, True],
    }
    data.update(
        {
            name: np.arange(4, dtype=float) + i
            for i, name in enumerate(FENIKS_SPLINE15D_PARAMETERS)
        }
    )
    return pd.DataFrame(data)


def test_observed_selection_uses_noisy_flux_and_mask() -> None:
    selected, summary = observed_selection_mask(_frame())
    np.testing.assert_array_equal(selected, [False, True, False, True])
    assert summary["selected_objects"] == 2
    assert summary["empirical_alpha"] == 0.5
    assert "saved noisy observed flux" in summary["noise_contract"]


def test_selection_closure_preserves_parent_identity(tmp_path: Path) -> None:
    catalog = tmp_path / "parent.parquet"
    _frame().to_parquet(catalog, index=False)
    out = tmp_path / "selection"
    receipt = build_selection_closure(catalog, out)
    assert receipt["truth_used_to_select"] is False
    parent = pd.read_parquet(out / "true_parent.parquet")
    selected = pd.read_parquet(out / "true_selected.parquet")
    assert parent.object_id.tolist() == [10, 11, 12, 13]
    assert selected.object_id.tolist() == [11, 13]
    assert (
        json.loads((out / "FINAL.json").read_text())["status"]
        == "EXACT_OBSERVED_SELECTION_COMPLETE"
    )


def test_selection_closure_joins_separate_exact_truth(tmp_path: Path) -> None:
    frame = _frame()
    truth = frame[["object_id", *FENIKS_SPLINE15D_PARAMETERS]]
    photometry = frame.drop(columns=list(FENIKS_SPLINE15D_PARAMETERS))
    photometry["galaxy_weight"] = [1.0, 2.0, 3.0, 4.0]
    parent_path = tmp_path / "photometry.parquet"
    truth_path = tmp_path / "truth.parquet"
    photometry.to_parquet(parent_path, index=False)
    truth.to_parquet(truth_path, index=False)
    receipt = build_selection_closure(
        parent_path, tmp_path / "selection", truth_catalogs=(truth_path,)
    )
    assert receipt["weighted_alpha"] == pytest.approx(0.6)
    selected = pd.read_parquet(tmp_path / "selection/true_selected.parquet")
    assert selected.object_id.tolist() == [11, 13]
    assert selected.population_weight.tolist() == [2.0, 4.0]


def test_selection_closure_rejects_non_distinct_cohort() -> None:
    frame = _frame()
    frame["flux_lsst_r"] = float(abmag_to_fnu_cgs(29.0)) * 2
    frame["mask_lsst_r"] = True
    with pytest.raises(ValueError, match="distinct nonempty"):
        observed_selection_mask(frame)


def test_weighted_quantile_tracks_weighted_empirical_distribution() -> None:
    result = _weighted_quantile(
        np.array([0.0, 10.0, 20.0]),
        np.array([1.0, 8.0, 1.0]),
        (0.25, 0.5, 0.75),
    )
    np.testing.assert_allclose(result, [40.0 / 9.0, 10.0, 140.0 / 9.0])


def test_two_distribution_corner_keeps_truth_visible(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    first = rng.normal(size=(200, 5))
    second = rng.normal(0.2, 1.1, size=(200, 5))
    path = tmp_path / "corner.png"
    _corner_compare(
        path,
        first,
        second,
        np.full(5, 8.0),
        first_label="first",
        second_label="second",
        title="fixture",
    )
    assert path.stat().st_size > 10_000
