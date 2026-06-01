from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.io import BandObservation, GalaxyObservation
from euclid_dsps.model import DspsContext

mcmc = pytest.importorskip("euclid_dsps.mcmc", exc_type=ImportError)
_prior_distribution = mcmc._prior_distribution
_prior_location = mcmc._prior_location
sample_one_galaxy = mcmc.sample_one_galaxy


def test_prior_location_can_use_row_resolved_base_value() -> None:
    value = _prior_location(
        "z_obs",
        {"initial": "from_base", "bounds": [0.0, 6.0]},
        {"loc": "from_base"},
        {"z_obs": 0.72},
    )

    assert value == 0.72


def test_scaled_beta_prior_uses_fit_bounds() -> None:
    pytest.importorskip("numpyro", exc_type=ImportError)
    prior = _prior_distribution(
        "dust_av",
        {"initial": 0.2, "bounds": [0.0, 3.0]},
        {"type": "scaled_beta", "alpha": 1.2, "beta": 3.0},
        {"dust_av": 0.2},
    )

    assert np.isfinite(float(prior.log_prob(jnp.asarray(1.5))))
    assert not np.isfinite(float(prior.log_prob(jnp.asarray(0.0))))
    assert not np.isfinite(float(prior.log_prob(jnp.asarray(3.0))))


def test_uniform_redshift_prior_samples_within_fit_bounds() -> None:
    pytest.importorskip("numpyro", exc_type=ImportError)
    prior = _prior_distribution(
        "z_obs",
        {"initial": "from_base", "bounds": [0.001, 6.0]},
        {"type": "uniform"},
        {"z_obs": 0.5},
    )

    assert np.isfinite(float(prior.log_prob(jnp.asarray(0.5))))
    assert bool(prior.support(jnp.asarray(0.5)))
    assert not bool(prior.support(jnp.asarray(7.0)))


def test_sample_one_galaxy_passes_dynamic_model_args(monkeypatch) -> None:
    pytest.importorskip("numpyro", exc_type=ImportError)

    def fake_dynamic_model_args(context):
        del context
        return (jnp.asarray([3.0], dtype=jnp.float32),)

    def fake_model_mags_dynamic(context, args, params):
        del context
        return jnp.asarray(
            [20.0 + 0.1 * params["x"] + 0.0 * args[0][0]],
            dtype=jnp.float32,
        )

    monkeypatch.setattr(mcmc, "dynamic_model_args", fake_dynamic_model_args)
    monkeypatch.setattr(mcmc, "model_mags_jax_dynamic", fake_model_mags_dynamic)
    monkeypatch.setattr(
        mcmc,
        "predict_batch_mags",
        lambda context, names, matrix: np.full((matrix.shape[0], 1), 20.0),
    )
    monkeypatch.setattr(mcmc, "predict_batch_derived", lambda context, names, matrix: {})

    result = sample_one_galaxy(
        DspsContext(ssp=None, filters={}, model_config={}),
        GalaxyObservation(
            row_index=0,
            row={},
            bands=[
                BandObservation(
                    name="wide",
                    column="wide",
                    flux_fnu_cgs=1.0,
                    mag_ab=20.0,
                    sigma_mag=0.1,
                )
            ],
        ),
        {"x": 0.0},
        {
            "likelihood_space": "mag",
            "photometric_likelihood": "gaussian",
            "free_parameters": {"x": {"initial": 0.0, "bounds": [-1.0, 1.0]}},
        },
        {
            "sampler": "hmc",
            "num_warmup": 1,
            "num_samples": 1,
            "num_chains": 1,
            "chain_method": "sequential",
            "num_steps": 1,
            "progress_bar": False,
            "seed": 0,
            "priors": {"x": {"type": "uniform"}},
        },
        initial_params={"x": 0.0},
    )

    assert "x" in result.samples
    assert result.posterior_model_mags.shape == (1, 1)
    assert result.diagnostics["backend"] == "numpyro_hmc"


def test_sample_one_galaxy_mclmc_smoke(monkeypatch) -> None:
    pytest.importorskip("blackjax")
    from euclid_dsps import posterior_target as target_module

    def fake_dynamic_model_args(context):
        del context
        return ()

    def fake_model_mags_dynamic(context, args, params):
        del context, args
        return jnp.asarray([20.0 + 0.1 * params["x"] + 0.2 * params["y"]])

    monkeypatch.setattr(mcmc, "dynamic_model_args", fake_dynamic_model_args)
    monkeypatch.setattr(
        target_module, "model_mags_jax_dynamic", fake_model_mags_dynamic
    )
    monkeypatch.setattr(
        mcmc,
        "predict_batch_mags",
        lambda context, names, matrix: np.full((matrix.shape[0], 1), 20.0),
    )
    monkeypatch.setattr(mcmc, "predict_batch_derived", lambda context, names, matrix: {})

    result = sample_one_galaxy(
        DspsContext(ssp=None, filters={}, model_config={}),
        GalaxyObservation(
            row_index=0,
            row={},
            bands=[
                BandObservation(
                    name="wide",
                    column="wide",
                    flux_fnu_cgs=1.0,
                    mag_ab=20.0,
                    sigma_mag=0.1,
                )
            ],
        ),
        {"x": 0.0, "y": 0.0},
        {
            "likelihood_space": "mag",
            "photometric_likelihood": "gaussian",
            "free_parameters": {
                "x": {"initial": 0.0, "bounds": [-1.0, 1.0]},
                "y": {"initial": 0.0, "bounds": [-1.0, 1.0]},
            },
        },
        {
            "sampler": "mclmc",
            "num_warmup": 2,
            "num_samples": 3,
            "progress_bar": False,
            "seed": 0,
            "priors": {"x": {"type": "uniform"}, "y": {"type": "uniform"}},
            "mclmc_l": 1.0,
            "mclmc_step_size": 0.01,
        },
        initial_params={"x": 0.0, "y": 0.0},
    )

    assert result.diagnostics["backend"] == "blackjax_mclmc"
    assert result.diagnostics["sampler"] == "mclmc"
    assert result.posterior_model_mags.shape == (3, 1)
    assert set(result.samples) == {"x", "y"}
