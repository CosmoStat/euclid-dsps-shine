from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from euclid_dsps.prior_learning.data import (
    load_truth_dataset,
    load_truth_dataset_with_schema,
    truth_dataset_with_latent_spec,
    truth_standardized_latent_spec,
)
from euclid_dsps.prior_learning.diagnostics import (
    prior_quality_gate,
    write_supervised_prior_diagnostics,
)
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


def test_load_truth_dataset_accepts_row_indices_file(tmp_path: Path) -> None:
    path = tmp_path / "truth.parquet"
    pd.DataFrame(
        {
            "object_id": [10, 11, 12],
            "redshift_true": [0.1, 0.2, 0.3],
            "logsm_true": [9.0, 10.0, 11.0],
            "logsfr_true": [-1.0, 0.5, 0.7],
        }
    ).to_parquet(path)
    rows = tmp_path / "rows.txt"
    rows.write_text("2\n0\n", encoding="utf-8")

    truth = load_truth_dataset(
        path,
        schema_name="diffsky_truth_basic",
        bounds={
            "z_obs": [0.0, 1.0],
            "log10_stellar_mass": [8.0, 12.0],
            "log10_sfr_at_obs": [-2.0, 1.0],
        },
        row_indices_file=rows,
    )

    assert truth.object_id.tolist() == [10, 12]
    assert truth.source_rows.tolist() == [0, 2]


def test_truth_standardized_latent_spec_reuses_train_coordinates(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    train = pd.DataFrame(
        {
            "object_id": np.arange(6),
            "redshift_true": [0.001, 0.1, 0.2, 0.4, 0.8, 0.999],
            "logsm_true": [8.0, 8.2, 8.8, 9.4, 10.5, 11.0],
            "logsfr_true": [-2.0, -1.8, -1.0, -0.3, 0.2, 1.0],
        }
    )
    validation = pd.DataFrame(
        {
            "object_id": [20, 21],
            "redshift_true": [0.15, 0.7],
            "logsm_true": [8.6, 10.2],
            "logsfr_true": [-1.5, 0.4],
        }
    )
    train.to_parquet(train_path)
    validation.to_parquet(validation_path)
    truth = load_truth_dataset(
        train_path,
        schema_name="diffsky_truth_basic",
        bounds={
            "z_obs": [0.0, 1.0],
            "log10_stellar_mass": [8.0, 11.0],
            "log10_sfr_at_obs": [-2.0, 1.0],
        },
    )

    latent_spec, payload = truth_standardized_latent_spec(truth, min_raw_scale=0.1)
    normalized = truth_dataset_with_latent_spec(truth, latent_spec)
    validation_truth = load_truth_dataset_with_schema(
        validation_path,
        schema=truth.schema,
        latent_spec=latent_spec,
    )

    assert latent_spec.normalization == "truth_standardized_logit"
    assert payload["n_raw_scale_clipped_low"] >= 0
    assert np.all(np.isfinite(normalized.x))
    assert np.all(np.isfinite(validation_truth.x))
    assert np.allclose(np.mean(normalized.x, axis=0), 0.0, atol=1.0e-5)


def test_supervised_prior_diagnostics_skip_constant_corner_columns(
    tmp_path: Path,
) -> None:
    truth = pd.DataFrame(
        {
            "z_obs": np.linspace(0.1, 0.5, 32),
            "log10_stellar_mass": np.linspace(9.0, 10.0, 32),
            "constant_truth": np.ones(32),
        }
    )
    prior = pd.DataFrame(
        {
            "z_obs": np.linspace(0.12, 0.52, 32),
            "log10_stellar_mass": np.linspace(8.9, 10.1, 32),
            "constant_truth": np.ones(32),
        }
    )

    outputs = write_supervised_prior_diagnostics(
        truth=truth,
        prior=prior,
        parameter_names=("z_obs", "log10_stellar_mass", "constant_truth"),
        out_dir=tmp_path,
        summary={"schema": "toy"},
    )

    assert (tmp_path / "prior_vs_truth_metrics.csv").exists()
    assert (tmp_path / "supervised_prior_vs_truth_report.md").exists()
    assert "plot_skipped_parameters" in outputs
    skipped = json.loads(
        Path(outputs["plot_skipped_parameters"]).read_text(encoding="utf-8")
    )
    assert skipped["parameters"] == ["constant_truth"]


def test_supervised_prior_diagnostics_handles_one_sided_constant_corner_column(
    tmp_path: Path,
) -> None:
    truth = pd.DataFrame(
        {
            "z_obs": np.linspace(0.1, 0.5, 64),
            "log10_stellar_mass": np.linspace(9.0, 10.0, 64),
            "dust_av": np.linspace(0.0, 1.0, 64),
        }
    )
    prior = pd.DataFrame(
        {
            "z_obs": np.linspace(0.12, 0.52, 64),
            "log10_stellar_mass": np.full(64, 9.5),
            "dust_av": np.linspace(0.1, 0.8, 64),
        }
    )

    outputs = write_supervised_prior_diagnostics(
        truth=truth,
        prior=prior,
        parameter_names=("z_obs", "log10_stellar_mass", "dust_av"),
        out_dir=tmp_path,
        summary={"schema": "toy"},
        max_corner_rows=32,
    )

    assert Path(outputs["corner"]).exists()
    assert Path(outputs["corner_plot_metadata"]).exists()
    metadata = pd.read_csv(outputs["corner_plot_metadata"])
    legacy = metadata.loc[metadata["kind"] == "legacy_first8"].iloc[0]
    assert bool(legacy["written"]) is True
    assert "log10_stellar_mass" in legacy["plotted_columns"]


def test_supervised_prior_diagnostics_quality_gate_flags_bad_prior(
    tmp_path: Path,
) -> None:
    truth = pd.DataFrame(
        {
            "z_obs": np.linspace(0.1, 1.0, 128),
            "log10_stellar_mass": np.linspace(8.5, 10.5, 128),
        }
    )
    prior = pd.DataFrame(
        {
            "z_obs": np.full(128, 5.5),
            "log10_stellar_mass": np.full(128, 6.0),
        }
    )

    outputs = write_supervised_prior_diagnostics(
        truth=truth,
        prior=prior,
        parameter_names=("z_obs", "log10_stellar_mass"),
        out_dir=tmp_path,
        summary={"schema": "toy"},
        max_corner_rows=32,
    )

    summary = json.loads(Path(outputs["summary"]).read_text(encoding="utf-8"))
    assert summary["prior_quality_gate_status"] == "FAIL"
    assert summary["prior_quality_gate"]["worst_ks_parameters"]
    report = Path(outputs["report"]).read_text(encoding="utf-8")
    assert "Prior Quality Gate" in report


def test_prior_quality_gate_passes_close_population() -> None:
    truth = pd.DataFrame(
        {
            "parameter": ["z_obs", "log10_stellar_mass"],
            "ks_distance": [0.02, 0.03],
            "wasserstein_distance": [0.01, 0.02],
            "median_residual": [0.01, -0.02],
        }
    )

    gate = prior_quality_gate(
        metrics=truth,
        correlation_payload={"frobenius_error": 0.1},
        multivariate={"sliced_wasserstein_distance": 0.02, "energy_distance": 0.03},
    )

    assert gate["status"] == "PASS"


@pytest.mark.skipif(
    not HAS_EQUINOX,
    reason="Equinox optional dependency is not installed",
)
def test_realnvp_supervised_prior_learns_toy_distribution() -> None:
    import jax
    import jax.numpy as jnp

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
    for layer in result.prior.layers:
        assert layer.mask.dtype == jnp.bool_
        assert np.all(np.asarray((layer.mask == 0) | (layer.mask == 1)))
    samples = result.prior.sample(jax.random.PRNGKey(11), 128)
    recovered, _ = result.prior.inverse(samples)
    roundtrip, _ = result.prior.forward(recovered)
    assert np.max(np.abs(np.asarray(roundtrip - samples))) < 1.0e-3


@pytest.mark.skipif(
    not HAS_EQUINOX,
    reason="Equinox optional dependency is not installed",
)
def test_realnvp_training_can_resume_from_prior_and_epoch() -> None:
    from euclid_dsps.prior_learning.train import fit_realnvp_to_x

    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 2)).astype(np.float32)
    flow_config = {"n_layers": 2, "hidden_size": 8, "scale_clamp": 0.2}
    training_config = {
        "epochs": 1,
        "batch_size": 16,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 5.0,
    }
    first = fit_realnvp_to_x(
        x,
        x,
        latent_dim=2,
        flow_config=flow_config,
        training_config=training_config,
        seed=3,
    )
    resumed = fit_realnvp_to_x(
        x,
        x,
        latent_dim=2,
        flow_config=flow_config,
        training_config={**training_config, "epochs": 2},
        seed=3,
        initial_prior=first.last_prior,
        initial_epoch=1,
    )

    assert resumed.training_log["epoch"].unique().tolist() == [2]
    assert resumed.validation_log["epoch"].tolist()[0] == 1
    assert bool(resumed.validation_log.iloc[0]["resumed_checkpoint"])


@pytest.mark.skipif(
    not HAS_EQUINOX,
    reason="Equinox optional dependency is not installed",
)
def test_rq_spline_train_supervised_prior_checkpoint_roundtrip(tmp_path: Path) -> None:
    import jax

    from euclid_dsps.prior_learning.flows import RQSplineCouplingPrior
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
            "flow": {
                "type": "rq_spline_coupling",
                "n_layers": 2,
                "hidden_size": 8,
                "n_bins": 4,
                "tail_bound": 4.0,
                "init": "identity",
                "init_scale": 0.0,
            },
            "training": {
                "epochs": 1,
                "batch_size": 16,
                "learning_rate": 1.0e-3,
                "validation_fraction": 0.25,
                "seed": 1,
            },
            "snapshots": {"enabled": False, "checkpoint_every": 0},
            "output": {"prior_samples": 8, "truth_sample_limit": 16},
        },
    }
    out = tmp_path / "spline_run"

    train_supervised_prior(config, out, verbose=False, progress=False)

    sidecar = json.loads(
        (out / "checkpoints" / "best.eqx.json").read_text(encoding="utf-8")
    )
    assert sidecar["architecture"]["type"] == "rq_spline_coupling"
    assert sidecar["flow_integrity"]["status"] == "PASS"
    prior, _sidecar, latent_spec, _schema = load_prior_checkpoint(
        out / "checkpoints" / "best.eqx"
    )
    assert isinstance(prior, RQSplineCouplingPrior)
    assert latent_spec.normalization == "identity"
    samples = prior.sample(jax.random.PRNGKey(2), 5)
    assert samples.shape == (5, 4)
    assert np.all(np.isfinite(np.asarray(prior.log_prob(samples))))


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
            "snapshots": {
                "enabled": True,
                "every_epochs": 1,
                "include_epoch_zero": True,
                "prior_samples": 8,
                "truth_sample_limit": 8,
                "max_corner_rows": 8,
                "checkpoint_every": 1,
            },
            "output": {"prior_samples": 16},
        },
    }
    out = tmp_path / "run"

    train_supervised_prior(config, out, verbose=False, progress=False)

    assert (out / "prior_training_log.csv").exists()
    assert (out / "prior_validation_loglike.csv").exists()
    assert (out / "prior_training_progress.json").exists()
    assert (out / "learned_prior_samples.parquet").exists()
    assert (out / "truth_theta_samples.parquet").exists()
    assert (out / "supervised_prior_summary.json").exists()
    assert (out / "supervised_prior_vs_truth_report.md").exists()
    assert (out / "checkpoints" / "best.eqx").exists()
    assert (out / "checkpoints" / "epoch_0001.eqx").exists()
    assert (out / "snapshots" / "epoch_0000" / "prior_samples.parquet").exists()
    assert (out / "snapshots" / "epoch_0001" / "truth_samples.parquet").exists()
    assert (out / "snapshots" / "epoch_0002" / "snapshot_summary.json").exists()

    sidecar_path = out / "checkpoints" / "best.eqx.json"
    progress = json.loads(
        (out / "prior_training_progress.json").read_text(encoding="utf-8")
    )
    assert progress["completed_epochs"] == 2
    assert progress["last"]["best_epoch"] >= 1
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["architecture"]["parameter_dtype"] == "float32"
    assert sidecar["architecture"]["shift_clamp"] == 5.0
    assert sidecar["realnvp_integrity"]["status"] == "PASS"
    _prior, _sidecar, latent_spec, _schema = load_prior_checkpoint(
        out / "checkpoints" / "best.eqx"
    )
    for layer in _prior.layers:
        assert layer.mask.dtype.name == "bool"
    assert latent_spec.normalization == "identity"
    assert np.allclose(np.asarray(latent_spec.raw_center), 0.0)
    assert np.allclose(np.asarray(latent_spec.raw_scale), 1.0)

    sidecar["architecture"].pop("parameter_dtype")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    load_prior_checkpoint(out / "checkpoints" / "best.eqx")
