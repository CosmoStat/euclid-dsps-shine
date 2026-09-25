import numpy as np
import pandas as pd
import pytest

from scripts.analyze_feniks_coherent_representation import convergence, verify
from scripts.feniks_avi_experiments import sha, write


def test_partial_bundle_verification_does_not_certify_missing_artifacts(tmp_path):
    write(tmp_path / "coordinates.json", dict(role="test"))
    write(
        tmp_path / "MANIFEST.json",
        dict(
            frozen_files={
                "coordinates.json": sha(tmp_path / "coordinates.json"),
                "excluded.npz": "absent",
            }
        ),
    )
    digest = sha(tmp_path / "MANIFEST.json")
    for stage in ("physical", "sfh_conditional", "sfh_zeros", "report"):
        (tmp_path / stage).mkdir()
        write(
            tmp_path / stage / "FINAL.json",
            dict(status="COMPLETE", contract=digest, artifacts={}),
        )
    result = verify(tmp_path)
    assert result == dict(
        verified=["coordinates.json"], excluded_or_missing=["excluded.npz"]
    )
    write(tmp_path / "coordinates.json", dict(role="modified"))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(tmp_path)


def test_convergence_uses_real_best_epoch_not_budget_endpoint():
    loss = np.linspace(0, -2, 40)
    loss[-1] = -1
    history = pd.DataFrame(
        dict(
            epoch=np.arange(1, 41),
            validation_nll=loss,
            train_nll=loss - 0.1,
            best_nll=np.minimum.accumulate(loss),
            learning_rate=np.full(40, 1e-5),
        )
    )
    result = convergence(history)
    assert result["epochs"] == 40
    assert result["best_epoch"] == 39
    assert result["best_gain_last_20"] > 0
    assert result["recent_train_validation_gap_median"] == pytest.approx(0.1)
