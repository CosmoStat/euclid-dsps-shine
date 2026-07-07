from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.filters import FilterCurve
from euclid_dsps.model import DspsContext, dynamic_model_args
from euclid_dsps.parameters import POPCOSMOS_PARAMETER_NAMES

HAS_DEPS = (
    importlib.util.find_spec("equinox") is not None
    and importlib.util.find_spec("optax") is not None
)
pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason="Equinox/Optax optional dependencies are not installed",
)

if HAS_DEPS:
    from euclid_dsps.amortized.data import PhotometryBatch
    from euclid_dsps.amortized.decoder import model_flux_from_x
    from euclid_dsps.amortized.elbo import negative_elbo
    from euclid_dsps.amortized.features import (
        compute_feature_stats,
        make_encoder_features,
    )
    from euclid_dsps.amortized.latent import LatentSpec
    from euclid_dsps.amortized.train import (
        build_amortized_model,
        component_grad_norms,
    )


def test_amortized_elbo_uses_true_dsps_decoder_and_joint_gradients() -> None:
    context = _tiny_popcosmos_dsps_context()
    model_args = dynamic_model_args(context)
    latent_spec = _latent_spec()
    x_truth = jnp.asarray(
        [
            np.linspace(-0.20, 0.20, 16),
            np.linspace(0.15, -0.15, 16),
        ],
        dtype=jnp.float32,
    )
    true_flux = model_flux_from_x(
        x_truth,
        latent_spec,
        context,
        model_args,
        POPCOSMOS_PARAMETER_NAMES,
    )
    flux_err = 0.08 * jnp.maximum(jnp.abs(true_flux), jnp.median(jnp.abs(true_flux)))
    flux_err = flux_err + 1.0e-14
    stats = compute_feature_stats(
        np.asarray(true_flux),
        np.asarray(flux_err),
        np.ones(true_flux.shape, dtype=bool),
        band_names=_fs2_band_names(),
    )
    batch = PhotometryBatch(
        object_id=jnp.arange(true_flux.shape[0], dtype=jnp.int32),
        flux=true_flux,
        flux_err=flux_err,
        mask=jnp.ones_like(true_flux, dtype=bool),
        features=make_encoder_features(true_flux, flux_err, stats),
    )
    config = {
        "amortized": {
            "encoder": {"hidden_sizes": [16]},
            "prior": {"n_layers": 2, "hidden_size": 16},
            "likelihood": {"type": "student_t", "student_t_dof": 2.0},
        }
    }
    model = build_amortized_model(config, jax.random.PRNGKey(0))

    def loss_fn(candidate):
        return negative_elbo(
            candidate,
            batch,
            latent_spec,
            context,
            model_args,
            POPCOSMOS_PARAMETER_NAMES,
            jax.random.PRNGKey(1),
            1,
            1.0,
            {"type": "student_t", "student_t_dof": 2.0},
            use_mock_decoder=False,
        )

    eqx = importlib.import_module("equinox")
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    norms = component_grad_norms(grads)

    assert true_flux.shape == (2, 10)
    assert jnp.all(jnp.isfinite(true_flux))
    assert jnp.isfinite(loss)
    assert jnp.isfinite(metrics["logprior_mean"])
    assert jnp.isfinite(metrics["logq_mean"])
    assert norms["encoder_grad_norm"] > 0.0
    assert norms["prior_grad_norm"] > 0.0


def _tiny_popcosmos_dsps_context() -> DspsContext:
    wave = np.linspace(900.0, 24000.0, 128)
    lgmet = np.log10(np.asarray([0.004, 0.0142, 0.03]))
    lg_age = np.linspace(-3.0, 1.05, 32)
    met_factor = np.linspace(0.75, 1.25, len(lgmet))[:, None, None]
    age_factor = np.linspace(1.4, 0.45, len(lg_age))[None, :, None]
    wave_factor = (1.0 + 0.15 * (wave / wave.max()))[None, None, :]
    ssp_flux = 1.0e-3 * met_factor * age_factor * wave_factor
    return DspsContext(
        ssp=None,
        filters=_ten_filters(),
        n_sfh_bins=96,
        z_sun=0.0142,
        model_config={
            "sfh_model": "popcosmos_bins",
            "stellar_metallicity_model": "single",
            "dust_model": "charlot_fall",
            "igm_model": "none",
            "nebular_model": "fixed_ssp",
            "agn_model": "none",
            "z_sun": 0.0142,
        },
        ssp_wave_jax=jnp.asarray(wave, dtype=jnp.float32),
        ssp_lgmet_jax=jnp.asarray(lgmet, dtype=jnp.float32),
        ssp_lg_age_gyr_jax=jnp.asarray(lg_age, dtype=jnp.float32),
        ssp_flux_jax=jnp.asarray(ssp_flux, dtype=jnp.float32),
        jax_filters=tuple(
            (
                jnp.asarray(curve.wave, dtype=jnp.float32),
                jnp.asarray(curve.transmission, dtype=jnp.float32),
            )
            for curve in _ten_filters().values()
        ),
    )


def _ten_filters() -> dict[str, FilterCurve]:
    filters = {}
    starts = np.linspace(3200.0, 15500.0, 10)
    for name, start in zip(_fs2_band_names(), starts, strict=True):
        wave = np.linspace(start, start + 1800.0, 64)
        filters[name] = FilterCurve(
            name=name,
            wave=wave,
            transmission=np.ones_like(wave),
            source="synthetic_dsps_e2e",
        )
    return filters


def _latent_spec() -> LatentSpec:
    lower = np.asarray(
        [
            0.2,
            9.0,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.4,
            -0.7,
            0.05,
            -1.2,
            0.1,
            -0.7,
            -3.5,
            -12.0,
            np.log(5.0),
        ],
        dtype=np.float32,
    )
    upper = np.asarray(
        [
            0.9,
            10.8,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.2,
            0.8,
            -0.3,
            1.8,
            0.3,
            -1.5,
            -7.0,
            np.log(80.0),
        ],
        dtype=np.float32,
    )
    return LatentSpec(
        names=POPCOSMOS_PARAMETER_NAMES,
        lower=jnp.asarray(lower),
        upper=jnp.asarray(upper),
    )


def _fs2_band_names() -> tuple[str, ...]:
    return (
        "lsst_u",
        "lsst_g",
        "lsst_r",
        "lsst_i",
        "lsst_z",
        "lsst_y",
        "euclid_vis",
        "euclid_nisp_y",
        "euclid_nisp_j",
        "euclid_nisp_h",
    )
