from scripts.summarize_feniks_conditional_posterior_task import (
    _completed_epoch_count,
)


def test_completed_epoch_count_for_fresh_training() -> None:
    assert _completed_epoch_count({"epochs": 120}) == 120


def test_completed_epoch_count_for_warm_restart() -> None:
    assert _completed_epoch_count({"epochs": 120, "start_epoch": 92}) == 29
