from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from euclid_dsps.config import load_config
from euclid_dsps.prior_learning.marginal_normalization import (
    ATOM_PARAMETER_NAMES,
    forward_marginal,
    inverse_marginal,
    load_marginal_transforms,
    shared_atom_mask,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "config_name,version,flow_type",
    [
        ("feniks_normalization_hybrid_realnvp_18d_h100.yaml", "hybrid", "realnvp"),
        (
            "feniks_normalization_hybrid_rqspline_18d_h100.yaml",
            "hybrid",
            "rq_spline_coupling",
        ),
        (
            "feniks_normalization_dirac_realnvp_18d_h100.yaml",
            "dirac_preserved",
            "realnvp",
        ),
        (
            "feniks_normalization_dirac_rqspline_18d_h100.yaml",
            "dirac_preserved",
            "rq_spline_coupling",
        ),
    ],
)
def test_normalization_benchmark_configs_load(
    config_name: str,
    version: str,
    flow_type: str,
) -> None:
    config = load_config(ROOT / "configs" / "experiments" / config_name)
    assert config["normalization_benchmark"]["version"] == version
    assert config["prior_learning"]["flow"]["type"] == flow_type
    assert config["prior_learning"]["train_dataset"].endswith("_18band/train.parquet")


@pytest.mark.parametrize(
    "spec_path",
    [
        ROOT / "configs" / "normalizations" / "feniks_18band_hybrid_v1.json",
        ROOT / "configs" / "normalizations" / "feniks_18band_dirac_preserved_v1.json",
    ],
)
def test_fitted_marginal_transforms_roundtrip(spec_path: Path) -> None:
    transforms = load_marginal_transforms(spec_path)
    for spec in transforms.values():
        target = spec.get("continuous_transform", spec)
        family = target["family"]
        if family == "quantile_spline":
            values = np.quantile(
                np.asarray(target["theta_knots"]), [0.0, 0.2, 0.5, 0.8, 1.0]
            )
        elif family == "wide_bound_logit":
            span = target["upper"] - target["lower"]
            values = np.asarray(
                [
                    target["lower"] + 0.01 * span,
                    target["lower"] + 0.5 * span,
                    target["upper"] - 0.01 * span,
                ]
            )
        elif family == "atom_centered_asinh":
            values = target["atom_value"] + target["lambda"] * np.asarray(
                [-2.0, 0.0, 2.0]
            )
        else:
            center = float(target.get("shift", target.get("center", 0.0)))
            scale = max(
                abs(float(target.get("scale", target.get("lambda", 1.0)))), 1.0e-3
            )
            values = center + scale * np.asarray([-2.0, 0.0, 2.0])
        reconstructed = inverse_marginal(forward_marginal(values, target), target)
        assert np.allclose(reconstructed, values, rtol=1.0e-10, atol=1.0e-10)


def test_hybrid_atom_mask_requires_one_shared_exact_state() -> None:
    transforms = load_marginal_transforms(
        ROOT / "configs" / "normalizations" / "feniks_18band_hybrid_v1.json"
    )
    names = ATOM_PARAMETER_NAMES
    atom = np.asarray([transforms[name]["atom_value"] for name in names])
    theta = np.vstack([atom, atom + np.asarray([0.1, 0.2, 0.3, 0.4])])
    assert shared_atom_mask(theta, names, transforms).tolist() == [True, False]

    broken = theta.copy()
    broken[0, 1] += 1.0e-12
    with pytest.raises(ValueError, match="not identical"):
        shared_atom_mask(broken, names, transforms)
