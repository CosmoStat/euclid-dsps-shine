from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from euclid_dsps import cli
from euclid_dsps.amortized.map_adam import (
    _map_fit_summary,
    _map_photometry_frame,
    _validate_checkpoint_free_map,
)

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_free_map_is_strictly_likelihood_only() -> None:
    _validate_checkpoint_free_map(
        checkpoint=None,
        prior_weight=0.0,
        start_mode="latin_hypercube",
    )
    with pytest.raises(ValueError, match="prior_weight=0"):
        _validate_checkpoint_free_map(
            checkpoint=None,
            prior_weight=0.1,
            start_mode="latin_hypercube",
        )
    with pytest.raises(ValueError, match="encoder-independent"):
        _validate_checkpoint_free_map(
            checkpoint=None,
            prior_weight=0.0,
            start_mode="encoder",
        )


def test_map_cli_accepts_likelihood_only_mode_without_checkpoint() -> None:
    args = cli.build_parser().parse_args(
        [
            "--config",
            str(ROOT / "configs/experiments/popcosmos_native15d_rws.yaml"),
            "diffsky-map-adam-prior",
            "--prior-weight",
            "0",
            "--start-mode",
            "latin_hypercube",
        ]
    )
    assert args.checkpoint is None


def test_map_cli_accepts_progress_interval() -> None:
    args = cli.build_parser().parse_args(
        [
            "diffsky-map-adam-prior",
            "--progress-interval",
            "25",
        ]
    )
    assert args.progress_interval == 25


def test_map_photometry_and_fit_summaries_expose_forward_fit_quality() -> None:
    batch = SimpleNamespace(
        object_id=np.asarray([101, 202]),
        row_index=np.asarray([3, 7]),
        flux=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        flux_err=jnp.asarray([[0.5, 1.0], [1.0, 2.0]]),
        mask=jnp.asarray([[True, True], [True, False]]),
    )
    frame = _map_photometry_frame(
        batch,
        np.asarray([[1.5, 1.0], [2.0, 100.0]]),
        ("g", "r"),
    )
    assert len(frame) == 4
    assert frame["normalized_residual"].iloc[:3].tolist() == [1.0, -1.0, -1.0]
    assert np.isnan(frame["normalized_residual"].iloc[3])

    estimates = pd.DataFrame(
        {
            "map_n_parameters": [15, 15],
            "map_chi2_per_valid_band": [1.0, 3.0],
            "map_reduced_chi2": [2.0, 8.0],
        }
    )
    summary = _map_fit_summary(estimates)
    assert summary["n_parameters"] == 15
    assert summary["median_chi2_per_valid_band"] == 2.0
    assert summary["median_reduced_chi2"] == 5.0
    assert summary["fraction_reduced_chi2_le_2"] == 0.5
