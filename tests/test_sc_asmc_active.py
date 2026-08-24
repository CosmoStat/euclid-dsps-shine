from __future__ import annotations

import pytest


def test_active_bootstrap_contract_is_bounded_in_config() -> None:
    from euclid_dsps.amortized.sc_asmc_config import sc_asmc_em_schedule
    from euclid_dsps.config import load_config

    config = load_config("configs/experiments/feniks_sc_asmc_em_r25.yaml")
    schedule = sc_asmc_em_schedule(config)
    assert 128 <= schedule.active_bootstrap_count <= 256
    assert schedule.active_bootstrap_count == 192


def test_active_bootstrap_resume_rejects_changed_inputs() -> None:
    from euclid_dsps.amortized.sc_asmc_active import (
        _validate_active_bootstrap_resume_inputs,
    )

    contract = {
        "input_q_checkpoint_sha256": "1" * 64,
        "input_hard_rows_sha256": "2" * 64,
        "input_seed": 17,
    }
    _validate_active_bootstrap_resume_inputs(dict(contract), contract)
    changed = dict(contract, input_seed=18)
    with pytest.raises(ValueError, match="resume inputs changed"):
        _validate_active_bootstrap_resume_inputs(changed, contract)
