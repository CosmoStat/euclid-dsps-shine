from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

HAS_DEPS = (
    importlib.util.find_spec("equinox") is not None
    and importlib.util.find_spec("optax") is not None
)
pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason="Equinox/Optax optional dependencies are not installed",
)

if HAS_DEPS:
    from euclid_dsps.amortized.synthetic import run_synthetic_smoke


def test_synthetic_smoke_writes_outputs(tmp_path) -> None:
    config = {
        "amortized": {
            "encoder": {"hidden_sizes": [8]},
            "prior": {"n_layers": 2, "hidden_size": 8},
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "n_samples": 1,
                "kl_annealing_epochs": 1,
            },
            "output": {"save_training_curves": False},
        }
    }

    run_synthetic_smoke(
        config,
        tmp_path,
        n_objects=8,
        epochs=1,
        batch_size=4,
        seed=1,
        mock_decoder=True,
    )

    assert (tmp_path / "training_log.csv").exists()
    assert (tmp_path / "training_summary.json").exists()
    assert (tmp_path / "training_progress.json").exists()
    assert (tmp_path / "checkpoints" / "best.eqx").exists()
    assert (tmp_path / "checkpoints" / "last.eqx").exists()
    assert (tmp_path / "checkpoints" / "epoch_0001.eqx").exists()
    log = pd.read_csv(tmp_path / "training_log.csv")
    assert {"encoder_grad_norm", "prior_grad_norm", "joint_grad_norm"} <= set(log)
    assert log["encoder_grad_norm"].max() > 0.0
    assert log["prior_grad_norm"].max() > 0.0
