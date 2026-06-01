from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from euclid_dsps.model import DspsContext
from euclid_dsps.posterior_target import (
    BoundedParameterTransform,
    build_posterior_target,
    initial_unconstrained_position,
)


def test_bounded_parameter_transform_roundtrip_and_logjacobian() -> None:
    transform = BoundedParameterTransform(
        names=("x", "y"),
        lower=jnp.asarray([-2.0, 0.0]),
        upper=jnp.asarray([2.0, 5.0]),
    )
    theta = jnp.asarray([-0.4, 2.0])
    y = transform.to_unconstrained(theta)

    roundtrip = transform.to_bounded(y)

    np.testing.assert_allclose(np.asarray(roundtrip), np.asarray(theta), rtol=1e-5)
    assert np.isfinite(float(transform.log_abs_det_jacobian(y)))


def test_posterior_target_logdensity_has_finite_gradient(monkeypatch) -> None:
    from euclid_dsps import posterior_target as target_module

    def fake_model_mags_dynamic(context, model_args, params):
        del context, model_args
        return jnp.asarray([20.0 + 0.1 * params["x"] + 0.2 * params["y"]])

    monkeypatch.setattr(
        target_module, "model_mags_jax_dynamic", fake_model_mags_dynamic
    )
    target = build_posterior_target(
        context=DspsContext(ssp=None, filters={}, model_config={}),
        model_args=(),
        base_params={"x": 0.0, "y": 0.0},
        fit_config={
            "likelihood_space": "mag",
            "photometric_likelihood": "gaussian",
            "free_parameters": {
                "x": {"initial": 0.0, "bounds": [-1.0, 1.0]},
                "y": {"initial": 0.0, "bounds": [-1.0, 1.0]},
            },
        },
        sample_config={"priors": {"x": {"type": "uniform"}, "y": {"type": "uniform"}}},
        observed_mag=np.asarray([20.0]),
        sigma_mag=np.asarray([0.1]),
        observed_flux=np.asarray([1.0]),
        flux_error=np.asarray([0.1]),
    )
    y0 = initial_unconstrained_position(target, {"x": 0.0, "y": 0.0}, target.fit_config)

    value, grad = jax.value_and_grad(target.logdensity)(y0)

    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_posterior_target_gas_constraint_returns_negative_infinity(monkeypatch) -> None:
    from euclid_dsps import posterior_target as target_module

    monkeypatch.setattr(
        target_module,
        "model_mags_jax_dynamic",
        lambda context, model_args, params: jnp.asarray([20.0]),
    )
    target = build_posterior_target(
        context=DspsContext(
            ssp=None,
            filters={},
            model_config={"sfh_model": "popcosmos_bins", "nebular_model": "gas_grid"},
        ),
        model_args=(),
        base_params={
            "log10_stellar_metallicity": 0.0,
            "log10_gas_metallicity": 0.0,
        },
        fit_config={
            "likelihood_space": "mag",
            "photometric_likelihood": "gaussian",
            "free_parameters": {
                "log10_stellar_metallicity": {
                    "initial": 0.0,
                    "bounds": [-1.0, 1.0],
                },
                "log10_gas_metallicity": {
                    "initial": 0.0,
                    "bounds": [-1.0, 1.0],
                },
            },
        },
        sample_config={"priors": {}},
        observed_mag=np.asarray([20.0]),
        sigma_mag=np.asarray([0.1]),
        observed_flux=np.asarray([1.0]),
        flux_error=np.asarray([0.1]),
    )
    invalid_theta = jnp.asarray([0.5, 0.0])
    invalid_y = target.unconstrained_from_theta(invalid_theta)

    assert float(target.logdensity(invalid_y)) == -np.inf
