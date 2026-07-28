from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/compare_popcosmos_a24.py"
    spec = importlib.util.spec_from_file_location("compare_popcosmos_a24", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sky_match_is_unique_and_respects_radius() -> None:
    module = _module()
    ours = pd.DataFrame(
        {
            "object_id": [1, 2, 3],
            "ra_deg": [150.0, 150.00001, 151.0],
            "dec_deg": [2.0, 2.0, 2.0],
        }
    )
    reference = pd.DataFrame(
        {
            "INDEX": [10, 11],
            "RA": [150.0, 152.0],
            "DEC": [2.0, 2.0],
        }
    )
    matched = module.match_catalogs(ours, reference, radius_arcsec=0.3)
    assert len(matched) == 1
    assert matched.loc[0, "rws_object_id"] == 1
    assert matched.loc[0, "match_arcsec"] == pytest.approx(0.0)


def test_photoz_metrics_use_normalized_residuals_and_coverage() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "a24_z_SPEC": [1.0, 2.0, np.nan],
            "rws_z_median": [1.0, 2.3, 0.4],
            "rws_z_q16": [0.8, 2.0, 0.2],
            "rws_z_q84": [1.2, 2.6, 0.6],
        }
    )
    metrics = module._photoz_metrics(frame, "rws_z")
    assert metrics["n_spec"] == 2
    assert metrics["coverage_68"] == 1.0
    assert metrics["outlier_fraction_0p15"] == 0.0
