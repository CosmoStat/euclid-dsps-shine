from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.features import compute_feature_stats, make_encoder_features
from euclid_dsps.config import load_config


def test_encoder_features_are_two_times_band_count_for_b14() -> None:
    flux = np.ones((5, 14), dtype=float)
    flux_err = np.ones((5, 14), dtype=float) * 0.2
    mask = np.ones((5, 14), dtype=bool)

    stats = compute_feature_stats(
        flux,
        flux_err,
        mask,
        band_names=tuple(f"band_{index}" for index in range(14)),
    )
    features = make_encoder_features(jnp.asarray(flux), jnp.asarray(flux_err), stats)

    assert features.shape == (5, 28)


HAS_DEPS = (
    importlib.util.find_spec("equinox") is not None
    and importlib.util.find_spec("optax") is not None
)


@pytest.mark.skipif(
    not HAS_DEPS,
    reason="Equinox/Optax optional dependencies are not installed",
)
def test_amortized_model_reads_b14_encoder_input_dim_from_config() -> None:
    from euclid_dsps.amortized.train import build_amortized_model

    config = {
        "amortized": {
            "data": {"expected_n_bands": 14},
            "features": {"n_flux_bands": 14, "n_error_bands": 14},
            "encoder": {"input_dim": 28, "hidden_sizes": [8]},
            "prior": {"n_layers": 2, "hidden_size": 8},
        }
    }

    model = build_amortized_model(config, jax.random.PRNGKey(0))
    mean, log_std = model.encoder(jnp.ones((2, 28), dtype=jnp.float32))

    assert mean.shape == (2, 16)
    assert log_std.shape == (2, 16)


def test_openuniverse_amortized_config_declares_b14_and_input_dim_28() -> None:
    config = load_config("configs/amortized_openuniverse_lsst_roman_realnvp.yaml")

    assert len(config["bands"]) == 14
    assert config["amortized"]["data"]["expected_n_bands"] == 14
    assert config["amortized"]["features"]["n_flux_bands"] == 14
    assert config["amortized"]["features"]["n_error_bands"] == 14
    assert config["amortized"]["encoder"]["input_dim"] == 28


def test_openuniverse_fit_ready_config_uses_fnu_cgs_and_exact_filters() -> None:
    config = load_config(
        "configs/amortized_openuniverse_lsst_roman_fit_ready_realnvp.yaml"
    )

    assert len(config["bands"]) == 14
    assert Path(config["catalog_path"]).name == "ou_lsst_roman_14_subset_fit_ready.parquet"
    assert {band["units"] for band in config["bands"]} == {"fnu_cgs"}
    assert {band["error_units"] for band in config["bands"]} == {"fnu_cgs"}
    assert {band["filter"]["kind"] for band in config["bands"]} == {"ascii"}
    assert config["openuniverse"]["lensing_mode"] == "unlensed"
    assert config["amortized"]["encoder"]["input_dim"] == 28
    assert config["truth"]["parameter_columns"]["dust_av"]["column"] is None
    assert config["truth"]["parameter_columns"]["log10_sfr_at_obs"]["column"] is None


def test_openuniverse_fit_ready_uses_blind_redshift_initialization() -> None:
    config = load_config("configs/openuniverse_lsst_roman_14_fit_ready.yaml")

    assert config["redshift"]["initial"] == "random_uniform"
    assert config["redshift"]["column"] == "redshift"
    assert config["truth"]["redshift_column"] == "redshift"
    assert config["fit"]["free_parameters"]["z_obs"]["initial"] == "from_base"
    assert config["fit"]["free_parameters"]["z_obs"]["bounds"] == [0.001, 2.5]
    assert config["fit"]["free_parameters"]["log10_stellar_mass"]["initial"] == 8.0


def test_openuniverse_lsst_only_fit_ready_config_uses_six_lsst_bands() -> None:
    config = load_config("configs/openuniverse_lsst_6_fit_ready.yaml")

    band_names = tuple(band["name"] for band in config["bands"])
    assert band_names == ("lsst_u", "lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y")
    assert {band["units"] for band in config["bands"]} == {"fnu_cgs"}
    assert config["amortized"]["data"]["expected_n_bands"] == 6
    assert config["amortized"]["features"]["n_flux_bands"] == 6
    assert config["amortized"]["features"]["n_error_bands"] == 6
    assert config["amortized"]["encoder"]["input_dim"] == 12
