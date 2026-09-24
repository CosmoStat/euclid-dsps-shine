from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_population_inversion_audit import _json_scalar, _select_strength


def test_selection_uses_one_se_then_bootstrap_stability():
    path = pd.DataFrame(
        {
            "strength": [0.0, 0.01, 0.1],
            "heldout_log_likelihood": [1.0, 0.9995, 0.8],
            "bootstrap_parent_sw_median": [0.03, 0.01, 0.001],
        }
    )
    rng = np.random.default_rng(4)
    base = rng.normal(size=1000)
    heldout = {
        0.0: base,
        0.01: base - 0.0005 + rng.normal(scale=0.02, size=1000),
        0.1: base - 0.2,
    }

    evaluated, selected = _select_strength(path, heldout)

    assert selected == 0.01
    assert evaluated.set_index("strength").loc[0.01, "heldout_one_se"]
    assert not evaluated.set_index("strength").loc[0.1, "heldout_one_se"]


def test_workflow_freezes_classifier_and_does_not_select_on_truth():
    repository = Path(__file__).parents[1]
    workflow = (repository / "scripts/feniks_population_inversion_audit.py").read_text()
    launcher = (
        repository / "scripts/submit_feniks_population_inversion_audit.sh"
    ).read_text()
    recovery = (
        repository / "scripts/resume_feniks_population_inversion_audit.sh"
    ).read_text()

    assert "tree_deserialise_leaves" in workflow
    assert "supervised_fit" not in workflow
    assert "bootstrap_parent_sw_median" in workflow
    assert "truth_used_for_selection=False" in workflow
    assert '--array="0-1%$ARM_CONCURRENCY"' in launcher
    assert '--dependency="afterok:$ARM_JOB"' in launcher
    assert "checkpoint_stability.csv" in recovery
    assert '--dependency="afterok:$ARM_JOB"' in recovery


def test_regularization_row_removes_duplicate_strength_keyword():
    repository = Path(__file__).parents[1]
    workflow = (repository / "scripts/feniks_population_inversion_audit.py").read_text()

    assert 'if key != "strength"' in workflow


def test_optional_solver_columns_serialize_as_json_null():
    assert _json_scalar(np.nan) is None
    assert _json_scalar(np.inf) is None
