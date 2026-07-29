from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _module():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "audit_popcosmos_a24_dsps_forward.py"
    )
    spec = importlib.util.spec_from_file_location("a24_dsps_forward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a24_parameter_matrix_uses_public_median_columns() -> None:
    module = _module()
    frame = pd.DataFrame({"a24_z_pc_500": [0.5], "a24_log10M_pc_500": [10.0]})
    matrix = module.a24_parameter_matrix(
        frame, ("z_obs", "log10_stellar_mass")
    )
    np.testing.assert_allclose(matrix, [[0.5, 10.0]])


def test_forward_audit_reports_reduced_chi2_and_residual_tail() -> None:
    module = _module()
    objects, bands, summary = module.forward_audit_tables(
        np.asarray([1, 2]),
        observed_flux=np.asarray([[1.0, 2.0], [1.0, 2.0]]),
        observed_error=np.ones((2, 2)),
        mask=np.ones((2, 2), dtype=bool),
        model_flux=np.asarray([[1.0, 2.0], [7.0, 2.0]]),
        band_names=("a", "b"),
        label="test",
    )
    assert objects.loc[0, "reduced_chi2"] == pytest.approx(0.0)
    assert objects.loc[1, "reduced_chi2"] == pytest.approx(18.0)
    assert summary["median_reduced_chi2"] == pytest.approx(9.0)
    assert summary["frac_abs_gt_5"] == pytest.approx(0.25)
    assert bands.loc[bands["band"] == "a", "frac_abs_gt_5"].iloc[0] == 0.5
