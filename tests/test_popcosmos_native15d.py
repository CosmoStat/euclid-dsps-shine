from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.amortized.catalog_identity import configured_redshift_column
from euclid_dsps.amortized.config import amortized_config
from euclid_dsps.amortized.features import FeatureStats
from euclid_dsps.amortized.train import (
    _configured_redshift_column,
    _sleep_flux_error,
    _sleep_runtime_config,
)
from euclid_dsps.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/popcosmos_native15d_rws.yaml"

NATIVE_PARAMETERS = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "sfh_dlog_sfr_01",
    "sfh_dlog_sfr_02",
    "sfh_dlog_sfr_03",
    "sfh_dlog_sfr_04",
    "sfh_dlog_sfr_05",
    "sfh_dlog_sfr_06",
    "sfh_dlog_sfr_07",
    "sfh_dlog_sfr_08",
    "sfh_dlog_sfr_09",
    "sfh_dlog_sfr_10",
)


def _load_redshift_evaluator():
    path = ROOT / "scripts/evaluate_popcosmos_native15d_redshift.py"
    spec = importlib.util.spec_from_file_location("native15d_redshift", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_builder():
    path = ROOT / "scripts/build_popcosmos_native15d_eval_indices.py"
    spec = importlib.util.spec_from_file_location("native15d_eval_indices", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_scaling_summarizer():
    path = ROOT / "scripts/summarize_popcosmos_native15d_scaling.py"
    spec = importlib.util.spec_from_file_location("native15d_scaling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_popcosmos_native_config_changes_observations_not_latent_physics() -> None:
    config = load_config(CONFIG)
    amortized = amortized_config(config)
    a24 = load_config(ROOT / "configs/experiments/popcosmos_a24_rws_joint.yaml")

    assert tuple(config["fit"]["free_parameters"]) == NATIVE_PARAMETERS
    assert config["model"]["sfh_model"] == "spline15d"
    assert config["model"]["nebular_model"] == "fixed_ssp"
    assert config["model"]["agn_model"] == "none"
    assert config["truth"]["parameter_columns"] == {}
    assert config["truth"]["redshift_column"] == "redshift_true"
    assert config["science_target"]["reported_parameters"] == ["z_obs"]
    assert config["science_target"]["truth_evaluated_parameters"] == ["z_obs"]
    assert len(config["science_target"]["nuisance_parameters"]) == 14
    assert "lp_zbest" not in config["extra_columns"]
    assert amortized["data"]["selection_mode"] == "random"
    assert amortized["data"]["use_redshift_for_split"] is False
    assert configured_redshift_column(config) is None
    assert _configured_redshift_column(config, amortized["data"]) is None
    assert amortized["objective"]["sleep"]["error_model"] == "observed_catalog"
    assert amortized["features"]["stats_catalog_path"].endswith(
        "farmer_a24_n40000.parquet"
    )
    assert [band["name"] for band in config["bands"]] == [
        band["name"] for band in a24["bands"]
    ]

    assert amortized["latent"]["schema"] == "feniks_spline15d"
    assert amortized["latent"]["normalization"] == "spline15d_checkpoint"
    assert amortized["encoder"]["input_dim"] == 52
    assert amortized["encoder"]["latent_dim"] == 15
    assert amortized["likelihood"] == {
        "type": "student_t",
        "student_t_dof": 2.0,
        "error_floor_frac": 0.0,
        "error_jitter": 0.0,
    }
    assert amortized["objective"]["mode"] == "reweighted_wake_sleep"
    assert amortized["objective"]["wake"]["n_particles"] == 8
    assert amortized["objective"]["wake"]["start_encoder_epoch"] == 4
    assert amortized["objective"]["wake"]["every_encoder_epochs"] == 4

    stats = FeatureStats(
        flux_scale=np.ones(26),
        err_scale=np.ones(26),
        band_names=tuple(band["name"] for band in config["bands"]),
    )
    sleep = _sleep_runtime_config(config, stats)
    assert sleep["error_model"] == "observed_catalog"
    assert "m5" not in sleep


def test_native_evaluation_indices_exclude_largest_training_pool() -> None:
    builder = _load_eval_builder()
    full = np.arange(12)
    excluded = np.arange(5)
    first = builder.build_indices(full, excluded, size=4, seed=7)
    second = builder.build_indices(full, excluded, size=4, seed=7)
    assert np.array_equal(first, second)
    assert not np.isin(full[first], excluded).any()


def test_native_sleep_uses_observed_farmer_uncertainties() -> None:
    batch = SimpleNamespace(
        flux_err=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        mask=jnp.asarray([[True, False], [True, True]]),
    )
    errors = _sleep_flux_error(
        jnp.zeros((2, 2)),
        batch,
        {
            "error_model": "observed_catalog",
            "feature_err_scale": (10.0, 20.0),
        },
    )
    assert np.allclose(np.asarray(errors), [[1.0, 20.0], [3.0, 4.0]])


def test_native_map_wrapper_omits_amortized_checkpoint() -> None:
    script = (ROOT / "scripts/popcosmos_native15d_map_h100.slurm").read_text()
    assert "--prior-weight 0" in script
    assert "--start-mode latin_hypercube" in script
    assert "--checkpoint" not in script
    assert "map_normalized_residuals_by_band.csv" in script


def test_native_redshift_metrics_ignore_nuisance_latents() -> None:
    evaluator = _load_redshift_evaluator()
    frame = pd.DataFrame(
        {
            "redshift_true": [0.5, 1.0, np.nan],
            "z_obs_median": [0.5, 1.2, 4.0],
            "z_obs_q16": [0.4, 1.1, 3.0],
            "z_obs_q84": [0.6, 1.3, 5.0],
            "log10_stellar_mass_median": [8.0, 12.0, 9.0],
        }
    )
    metrics = evaluator.redshift_metrics(frame)
    intervals = evaluator.bootstrap_redshift_metrics(
        frame, n_bootstrap=20, seed=7
    )
    assert metrics["n_spec"] == 2
    assert metrics["coverage_68"] == 0.5
    assert metrics["outlier_fraction_0p15"] == 0.0
    assert "nmad" in intervals


def test_native_scaling_summary_collects_only_redshift_metrics(tmp_path) -> None:
    summarizer = _load_scaling_summarizer()
    for stage, size in summarizer.STAGES:
        out = tmp_path / stage / "inference"
        out.mkdir(parents=True)
        payload = {
            "n_inference": size // 10,
            "metrics": {
                "n_spec": size // 100,
                "median_bias": 0.0,
                "nmad": 0.1,
                "rmse": 0.2,
                "outlier_fraction_0p15": 0.05,
                "coverage_68": 0.68,
                "median_interval_width_68": 0.3,
            },
        }
        (out / "redshift_metrics.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )
    rows = summarizer.collect_scaling_rows(tmp_path)
    assert [row["train_catalog_size"] for row in rows] == [5000, 20000, 40000]
    assert all("log10_stellar_mass" not in row for row in rows)


def test_native_rws_stages_have_no_a24_parameter_comparison() -> None:
    wrapper = (ROOT / "scripts/popcosmos_native15d_rws_h100.slurm").read_text()
    submit = (ROOT / "scripts/submit_popcosmos_native15d_rws.sh").read_text()

    assert "evaluate_popcosmos_native15d_redshift.py" in wrapper
    assert "compare_popcosmos_a24.py" not in wrapper
    assert "science_target=z_obs" in wrapper
    assert "nuisance_latents=14" in wrapper
    assert "build_popcosmos_native15d_eval_indices.py" in wrapper
    assert "summarize_popcosmos_native15d_scaling.py" in wrapper
    assert '--row-indices-file "$EVAL_INDICES"' in wrapper
    assert '--dataset "$FULL_DATASET"' in wrapper
    assert 'STAGE must be n5k,n20k,n40k,full' in wrapper
    assert 'n5k) WALLTIME=04:00:00' in submit
    assert 'n20k) WALLTIME=08:00:00' in submit
    assert 'n40k|full) WALLTIME=15:00:00' in submit
    assert 'MAP_DIR' not in wrapper
    assert 'MAP_DIR' not in submit
    assert 'SKIP_TRAINING' in wrapper
