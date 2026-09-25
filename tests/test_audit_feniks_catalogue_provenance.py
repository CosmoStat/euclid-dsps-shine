import numpy as np
import pandas as pd
import pytest

from scripts.audit_feniks_catalogue_provenance import replay_configs, weighting_summary


def test_replay_changes_only_named_numerical_contracts():
    native = {"bands": [], "model": {"sfh_model": "diffsky_basic"}}
    current = {
        "bands": [],
        "model": {
            "sfh_model": "spline15d",
            "photometry_integrator": "merged_gauss4_v1",
            "mdf_weight_precision": "float64_v1",
            "spline_precision": "float64_v1",
        },
    }
    variants = replay_configs(native, current)
    assert variants["native_legacy"]["model"]["sfh_model"] == "diffsky_basic"
    assert variants["spline_legacy"]["model"]["spline_precision"] == "float32_legacy"
    assert variants["spline_current"] == current
    assert "photometry_integrator" not in native["model"]
    with pytest.raises(ValueError, match="Unmatched bands"):
        replay_configs(native, dict(current, bands=["different"]))


def test_weighting_aligns_identities_and_preserves_input():
    catalogue = pd.DataFrame(
        {"object_id": [1, 2], "redshift_true": [1.0, 2.0], "galaxy_weight": [1.0, 3.0]}
    )
    truth = pd.DataFrame(
        {"object_id": [2, 1], "z_obs": [2.0, 1.0], "population_weight": [3.0, 1.0]}
    )
    result = weighting_summary(catalogue, truth)
    assert result["truth_weights_equal_retained_proposal_weights"]
    assert result["truth_cohort"]["mean_z_unweighted"] == 1.5
    assert result["truth_cohort"]["mean_z_proposal_weighted_again"] == 1.75
    np.testing.assert_array_equal(truth.population_weight, [3.0, 1.0])
    with pytest.raises(ValueError, match="Nonunique"):
        weighting_summary(pd.concat([catalogue, catalogue]), truth)
    with pytest.raises(ValueError, match="missing"):
        weighting_summary(catalogue.iloc[:1], truth)
