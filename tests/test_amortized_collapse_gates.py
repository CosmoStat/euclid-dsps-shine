from __future__ import annotations

import pandas as pd

from euclid_dsps.amortized.collapse_gates import write_training_collapse_gate


def test_training_gate_uses_latest_applied_training_gradient(tmp_path) -> None:
    pd.DataFrame(
        {
            "split": ["train", "validation"],
            "update_applied": [1, 0],
            "loss": [1.0, 1.0],
            "encoder_grad_norm": [2500.0, 0.0],
        }
    ).to_csv(tmp_path / "training_log.csv", index=False)

    payload = write_training_collapse_gate(tmp_path)
    gradient = next(
        check
        for check in payload["checks"]
        if check["name"] == "latest_encoder_grad_norm"
    )

    assert gradient["value"] == 2500.0
    assert gradient["status"] == "FAIL"
