from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.prior_learning.data import load_truth_dataset
from euclid_dsps.prior_learning.schema import build_truth_schema

HAS_EQUINOX = importlib.util.find_spec("equinox") is not None


def test_truth_schema_builds_basic_theta_columns() -> None:
    schema = build_truth_schema(
        [
            "object_id",
            "redshift_true",
            "logsm_true",
            "logssfr_true",
            "logsfr_true",
            "dust_av",
            "diffstar_lgmcrit",
        ],
        schema_name="diffsky_truth_basic",
    )

    assert [param.name for param in schema.parameters] == [
        "z_obs",
        "log10_stellar_mass",
        "log10_ssfr_at_obs",
        "dust_av",
    ]
    assert [param.column for param in schema.parameters] == [
        "redshift_true",
        "logsm_true",
        "logssfr_true",
        "dust_av",
    ]


def test_extended_truth_schema_reduces_missing_generated_columns() -> None:
    schema = build_truth_schema(
        ["redshift_true", "logsm_true", "logsfr_true", "diffstar_lgmcrit"],
        schema_name="diffsky_truth_extended",
        missing_policy="reduce",
    )

    assert "diffstar_lgmcrit" in [param.name for param in schema.parameters]
    assert schema.reduced is True
    assert "diffstar_lgy_at_mcrit" in schema.missing_columns


def test_load_truth_dataset_uses_bounded_transform(tmp_path: Path) -> None:
    path = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [10, 11],
            "redshift_true": [0.1, 0.2],
            "logsm_true": [9.0, 10.0],
            "logsfr_true": [-1.0, 0.5],
        }
    ).to_parquet(path)

    truth = load_truth_dataset(
        path,
        schema_name="diffsky_truth_basic",
        bounds={
            "z_obs": [0.0, 1.0],
            "log10_stellar_mass": [8.0, 11.0],
            "log10_sfr_at_obs": [-2.0, 1.0],
        },
    )

    assert truth.parameter_names == (
        "z_obs",
        "log10_stellar_mass",
        "log10_sfr_at_obs",
    )
    assert truth.theta.shape == (2, 3)
    assert np.all(np.isfinite(truth.x))


@pytest.mark.skipif(
    not HAS_EQUINOX,
    reason="Equinox optional dependency is not installed",
)
def test_realnvp_supervised_prior_learns_toy_distribution() -> None:
    from euclid_dsps.prior_learning.train import fit_realnvp_to_x

    rng = np.random.default_rng(0)
    x = rng.normal(loc=np.asarray([1.4, -0.8]), scale=0.25, size=(128, 2)).astype(
        np.float32
    )

    result = fit_realnvp_to_x(
        x,
        None,
        latent_dim=2,
        flow_config={"n_layers": 4, "hidden_size": 16, "scale_clamp": 0.2},
        training_config={
            "epochs": 30,
            "batch_size": 64,
            "learning_rate": 5.0e-3,
            "weight_decay": 0.0,
            "gradient_clip_norm": 5.0,
        },
        seed=0,
    )

    final_loss = float(result.training_log["loss"].tail(2).mean())
    assert final_loss < result.initial_train_nll


@pytest.mark.skipif(
    not HAS_EQUINOX,
    reason="Equinox optional dependency is not installed",
)
def test_train_supervised_prior_writes_expected_outputs(tmp_path: Path) -> None:
    from euclid_dsps.prior_learning.train import (
        load_prior_checkpoint,
        train_supervised_prior,
    )

    dataset = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": np.arange(32),
            "redshift_true": np.linspace(0.1, 0.5, 32),
            "logsm_true": np.linspace(9.0, 10.0, 32),
            "logsfr_true": np.linspace(-1.0, 0.5, 32),
            "dust_av": np.linspace(0.1, 0.4, 32),
        }
    ).to_parquet(dataset)

    config = {
        "catalog_path": str(dataset),
        "prior_learning": {
            "dataset": str(dataset),
            "schema": "diffsky_truth_basic",
            "bounds": {
                "z_obs": [0.0, 1.0],
                "log10_stellar_mass": [8.0, 11.0],
                "log10_sfr_at_obs": [-2.0, 1.0],
                "dust_av": [0.0, 1.0],
            },
            "flow": {"n_layers": 2, "hidden_size": 8, "scale_clamp": 0.1},
            "training": {
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 1.0e-3,
                "validation_fraction": 0.25,
                "seed": 1,
            },
            "output": {"prior_samples": 16},
        },
    }
    out = tmp_path / "run"

    train_supervised_prior(config, out, verbose=False, progress=False)

    assert (out / "prior_training_log.csv").exists()
    assert (out / "prior_validation_loglike.csv").exists()
    assert (out / "learned_prior_samples.parquet").exists()
    assert (out / "truth_theta_samples.parquet").exists()
    assert (out / "supervised_prior_summary.json").exists()
    assert (out / "supervised_prior_vs_truth_report.md").exists()
    assert (out / "checkpoints" / "best.eqx").exists()

    sidecar_path = out / "checkpoints" / "best.eqx.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["architecture"]["parameter_dtype"] == "float32"
    load_prior_checkpoint(out / "checkpoints" / "best.eqx")

    sidecar["architecture"].pop("parameter_dtype")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    load_prior_checkpoint(out / "checkpoints" / "best.eqx")
