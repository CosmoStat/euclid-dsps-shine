from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_ratio_convergence import (
    _parent_stability,
    _plateau_diagnostics,
)


def _settings():
    return {
        "epoch_block": 20,
        "required_plateau_windows": 2,
        "minimum_recent_nll_improvement": 0.002,
    }


def test_plateau_requires_two_consecutive_small_improvements():
    first = np.linspace(3.0, 2.95, 20)
    second = np.linspace(2.949, 2.9485, 20)
    third = np.linspace(2.9480, 2.9475, 20)
    history = pd.DataFrame(
        {"epoch": np.arange(1, 61), "validation_nll": np.r_[first, second, third]}
    )

    improvements, plateau = _plateau_diagnostics(history, _settings())

    assert len(improvements) == 2
    assert plateau


def test_plateau_rejects_recent_material_improvement():
    first = np.linspace(3.0, 2.99, 20)
    second = np.linspace(2.98, 2.97, 20)
    third = np.linspace(2.96, 2.94, 20)
    history = pd.DataFrame(
        {"epoch": np.arange(1, 61), "validation_nll": np.r_[first, second, third]}
    )

    improvements, plateau = _plateau_diagnostics(history, _settings())

    assert max(improvements) > _settings()["minimum_recent_nll_improvement"]
    assert not plateau


def test_first_parent_checkpoint_has_json_safe_missing_delta():
    change, stable = _parent_stability([], 0.031, 0.0025)

    assert change is None
    assert stable is False


def test_parent_stability_compares_successive_checkpoints():
    trajectory = [{"parent_physical_sliced_wasserstein": 0.031}]

    change, stable = _parent_stability(trajectory, 0.030, 0.0025)

    assert np.isclose(change, 0.001)
    assert stable is True


def test_convergence_submission_has_two_cells_and_no_bank_stage():
    repository = Path(__file__).parents[1]
    launcher = (repository / "scripts/submit_feniks_ratio_convergence.sh").read_text()
    recovery = (repository / "scripts/resume_feniks_ratio_convergence.sh").read_text()
    watcher = (repository / "scripts/watch_feniks_ratio_convergence.sh").read_text()
    workflow = (repository / "scripts/feniks_ratio_convergence.py").read_text()

    assert '--array="0-1%$CLASSIFIER_CONCURRENCY"' in launcher
    assert '--dependency="afterok:$CLASSIFIER_JOB"' in launcher
    assert " bank " not in launcher
    assert "maximum_epochs" in workflow
    assert "parent_stable" in workflow
    assert "MAX_EPOCH" in watcher
    assert "trajectory.csv" in recovery
    assert '--dependency="afterok:$CLASSIFIER_JOB"' in recovery
