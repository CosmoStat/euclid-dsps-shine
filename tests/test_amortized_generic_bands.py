from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.amortized.data import iter_photometry_batches_from_arrays
from euclid_dsps.amortized.features import compute_feature_stats, make_encoder_features
from euclid_dsps.config import load_config
from euclid_dsps.observation_arrays import PhotometryArrays


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


def test_diffsky_simple_gpu_config_uses_fourteen_native_flux_bands() -> None:
    config = load_config("configs/diffsky_hltds_04_14_simple_gpu.yaml")

    band_names = tuple(band["name"] for band in config["bands"])
    assert len(band_names) == 14
    assert band_names[:6] == (
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
    )
    assert band_names[6:] == (
        "roman_F062",
        "roman_F087",
        "roman_F106",
        "roman_F129",
        "roman_F146",
        "roman_F158",
        "roman_F184",
        "roman_F213",
    )
    assert {band["units"] for band in config["bands"]} == {"fnu_cgs"}
    assert {band["error_units"] for band in config["bands"]} == {"fnu_cgs"}
    assert all(band["column"].startswith("flux_") for band in config["bands"])
    assert all(band["error_column"].startswith("fluxerr_") for band in config["bands"])
    assert config["runtime"]["require_gpu"] is True


def test_diffsky_amortized_gpu_config_is_b14_latent12_realnvp() -> None:
    config = load_config("configs/amortized_diffsky_hltds_04_14_realnvp_gpu.yaml")

    assert len(config["bands"]) == 14
    assert config["amortized"]["encoder"]["input_dim"] == 28
    assert config["amortized"]["encoder"]["latent_dim"] == 12
    assert config["amortized"]["latent"]["schema"] == "diffsky_hltds_prior_v1"
    assert config["amortized"]["latent"]["normalization"] == "standardized_logit"
    assert config["amortized"]["prior"]["type"] == "realnvp"
    assert config["runtime"]["require_gpu"] is True
    assert config["model"]["ssp_model"] == "compressed_basis"
    assert config["model"]["compressed_ssp_runtime_dtype"] == "float32"
    assert "04_14_2026_ssp_basis_k64_coeff16" in config["model"]["compressed_ssp_path"]
    assert config["amortized"]["training"]["jax_batch_size"] == 4
    assert config["amortized"]["inference"]["jax_batch_size"] == 4


def test_diffsky_supervised_prior_config_is_b14_latent5_checkpoint() -> None:
    config = load_config("configs/amortized_diffsky_hltds_supervised_prior_gpu.yaml")

    assert len(config["bands"]) == 14
    assert config["amortized"]["encoder"]["input_dim"] == 28
    assert config["amortized"]["encoder"]["latent_dim"] == 5
    assert config["amortized"]["latent"]["schema"] == "diffsky_truth_basic"
    assert config["amortized"]["prior"]["source"] == "supervised_checkpoint"
    assert config["amortized"]["prior"]["train_jointly"] is False
    assert tuple(config["fit"]["free_parameters"]) == (
        "z_obs",
        "log10_stellar_mass",
        "log10_ssfr_at_obs",
        "dust_av",
        "dust_delta",
    )


def test_generic_batches_preserve_large_object_ids_as_int64() -> None:
    flux = np.ones((2, 3), dtype=np.float32)
    flux_err = np.full((2, 3), 0.1, dtype=np.float32)
    mask = np.ones((2, 3), dtype=bool)
    object_id = np.asarray([734086782211090368, 1319559050211399780], dtype=np.int64)
    stats = compute_feature_stats(
        flux,
        flux_err,
        mask,
        band_names=("a", "b", "c"),
    )
    arrays = PhotometryArrays(
        object_id=object_id,
        flux=flux,
        flux_err=flux_err,
        mask=mask,
        band_names=("a", "b", "c"),
        row_index=np.asarray([42, 99], dtype=np.int64),
    )

    batch = next(
        iter_photometry_batches_from_arrays(arrays, batch_size=2, feature_stats=stats)
    )

    assert isinstance(batch.object_id, np.ndarray)
    assert batch.object_id.dtype == np.int64
    np.testing.assert_array_equal(batch.object_id, object_id)
    np.testing.assert_array_equal(batch.row_index, np.asarray([42, 99], dtype=np.int64))
