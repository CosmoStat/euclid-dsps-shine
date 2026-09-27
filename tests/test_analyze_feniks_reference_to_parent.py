import json

import pytest

from scripts.analyze_feniks_reference_to_parent import verify_available, weight_checks
from scripts.feniks_avi_experiments import sha


def test_parent_normalization_and_inverse_selection():
    result = weight_checks(
        dict(u=[0.8, 0.2], v=[0.5, 0.5], alpha=[0.25, 1.0], alpha_parent=0.4)
    )
    assert result["reconstruction_max_error"] == 0
    assert result["alpha_from_components"] == 0.4


@pytest.mark.parametrize(
    "change",
    [
        {"u": [0.5, 0.5]},
        {"v": [0.25, 0.25]},
        {"alpha": [0.0, 1.0]},
        {"alpha_parent": 0.7},
    ],
)
def test_incorrect_parent_contract_fails(change):
    parent = dict(u=[0.8, 0.2], v=[0.5, 0.5], alpha=[0.25, 1.0], alpha_parent=0.4)
    parent.update(change)
    with pytest.raises(ValueError):
        weight_checks(parent)


def test_missing_large_artifacts_are_reported_but_changed_metrics_rejected(tmp_path):
    (tmp_path / "MANIFEST.json").write_text("{}")
    contract = sha(tmp_path / "MANIFEST.json")
    for stage in ("reference", "qualification", "population", "report"):
        folder = tmp_path / stage
        folder.mkdir()
        result = folder / "result.json"
        result.write_text("{}")
        receipt = dict(
            status="COMPLETE",
            contract=contract,
            artifacts={"result.json": sha(result), "large.npz": "unavailable"},
        )
        (folder / "FINAL.json").write_text(json.dumps(receipt))
    checks = verify_available(tmp_path)
    assert len(checks["verified"]) == 4
    assert len(checks["unavailable"]) == 4
    (tmp_path / "report/result.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="Artifact changed"):
        verify_available(tmp_path)
