from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.feniks_population_basis_audit import _select_candidate


def test_candidate_selection_uses_heldout_then_bootstrap_not_truth():
    path = pd.DataFrame(
        {
            "resolution": [128, 64, 32],
            "strength": [0.0, 0.01, 0.03],
            "heldout_log_likelihood": [1.0, 0.9998, 0.8],
            "bootstrap_parent_sw_median": [0.03, 0.01, 0.001],
            "parent_physical_sliced_wasserstein": [0.001, 10.0, 0.0],
        }
    )
    rng = np.random.default_rng(5)
    base = rng.normal(size=1000)
    heldout = {
        (128, 0.0): base,
        (64, 0.01): base - 0.0002 + rng.normal(scale=0.02, size=1000),
        (32, 0.03): base - 0.2,
    }

    evaluated, selected = _select_candidate(path, heldout)

    assert selected == (64, 0.01)
    assert evaluated.set_index("resolution").loc[64, "heldout_one_se"]
    assert not evaluated.set_index("resolution").loc[32, "heldout_one_se"]


def test_hierarchy_config_is_one_controlled_audit():
    repository = Path(__file__).parents[1]
    config = yaml.safe_load(
        (
            repository / "configs/experiments/feniks_population_basis_audit.yaml"
        ).read_text()
    )

    assert config["resolutions"] == [32, 48, 64, 96, 128]
    assert config["bootstraps"] == 32
    assert config["metric_seed"] == 2026092404
    assert config["hierarchy"]["physical_weight"] > 0
    assert config["hierarchy"]["photometric_weight"] > 0
