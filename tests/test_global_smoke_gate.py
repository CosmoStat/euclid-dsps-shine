import json

import pandas as pd
import pytest

from euclid_dsps.amortized.sc_drws import SCDrwsSchedule, update_kind_for_epoch
from euclid_dsps.amortized.sc_drws_trainer import smoke_schedule
from scripts.check_feniks_global_smoke import check


def test_schedule_reaches_both_wake_phases():
    s = smoke_schedule(SCDrwsSchedule())
    assert [i for i in range(1, 9) if update_kind_for_epoch(i, s) == "wake"] == [4, 7]
    assert s.flow_freeze_epochs == 0
    assert s.flow_thaw_end_epoch == 1


@pytest.mark.parametrize("bad", [None, "zero", "ascent", "frozen"])
def test_gate(tmp_path, bad):
    receipt = {"wake_updates": 2, "prior_updates": 1}
    if bad == "zero":
        receipt["prior_updates"] = 0
    (tmp_path / "training_receipt.json").write_text(json.dumps(receipt))
    rows = [
        dict(
            update_kind="wake",
            phase=phase,
            q_update_applied=True,
            flow_gradient_multiplier=0 if bad == "frozen" else 0.5,
            loss=10.0,
            q_objective_after=11.0 if bad == "ascent" else 9.0,
        )
        for phase in ("robust_warmup", "joint_gaussian")
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "sc_drws_training_log.csv", index=False)
    if bad:
        with pytest.raises(ValueError):
            check(tmp_path)
    else:
        assert check(tmp_path)["status"] == "PASS"
