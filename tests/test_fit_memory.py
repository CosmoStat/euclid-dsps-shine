from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from euclid_dsps.fit import (
    _context_gas_grid_nbytes,
    _should_jit_single_fit,
    fit_galaxy_batch_adam,
    fit_one_galaxy,
    fit_population_batch_adam,
)
from euclid_dsps.io import BandObservation, GalaxyObservation
from euclid_dsps.model import (
    DspsContext,
    ModelResult,
    bind_dynamic_model_args,
    dynamic_model_args,
)


class FakeGrid:
    shape = (7, 7, 12, 107, 11149)
    dtype = "float32"


def test_single_fit_keeps_jit_for_dynamic_large_gas_grid() -> None:
    context = DspsContext(ssp=None, filters={}, ssp_flux_gas_grid_jax=FakeGrid())

    assert _context_gas_grid_nbytes(context) / 1024**3 > 2.5
    assert _should_jit_single_fit(context, {}) is True
    assert _should_jit_single_fit(context, {"jit": False}) is False


def test_dynamic_model_args_rebind_large_arrays() -> None:
    context = DspsContext(ssp=None, filters={}, ssp_flux_gas_grid_jax=FakeGrid())
    args = dynamic_model_args(context)
    rebound = bind_dynamic_model_args(context, args)

    assert rebound is not context
    assert rebound.ssp_flux_gas_grid_jax is context.ssp_flux_gas_grid_jax


def test_single_fit_applies_gas_constraint_with_unpacked_params(monkeypatch) -> None:
    import euclid_dsps.fit as fit_module

    def fake_model_mags(context, args, params):
        del context, args
        return jnp.asarray(
            [20.0 + 0.01 * params["log10_gas_metallicity"]], dtype=jnp.float32
        )

    def fake_run_dsps_model(context, params):
        del context
        return ModelResult(
            parameters={key: float(value) for key, value in params.items()},
            derived={},
            wave=np.asarray([1000.0, 2000.0]),
            rest_sed=np.ones(2),
            dusted_rest_sed=np.ones(2),
            photometry={},
        )

    monkeypatch.setattr(fit_module, "model_mags_jax_dynamic", fake_model_mags)
    monkeypatch.setattr(fit_module, "run_dsps_model", fake_run_dsps_model)
    context = DspsContext(
        ssp=None,
        filters={},
        model_config={"sfh_model": "popcosmos_bins", "nebular_model": "gas_grid"},
    )
    observation = GalaxyObservation(
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
    )
    result = fit_one_galaxy(
        context,
        observation,
        {
            "log10_stellar_metallicity": -0.5,
            "log10_gas_metallicity": -0.4,
        },
        {
            "method": "jax_adam",
            "jit": False,
            "maxiter": 1,
            "free_parameters": {
                "log10_stellar_metallicity": {
                    "initial": -0.5,
                    "bounds": [-1.0, 0.5],
                },
                "log10_gas_metallicity": {
                    "initial": -0.4,
                    "bounds": [-1.0, 0.5],
                },
            },
        },
    )

    assert result.success
    assert result.best_parameters["log10_gas_metallicity"] >= (
        result.best_parameters["log10_stellar_metallicity"]
    )


def test_population_fit_optimizer_trace_is_sparse(monkeypatch) -> None:
    import euclid_dsps.fit as fit_module

    def fake_model_mags(context, args, params):
        del context, args
        return jnp.asarray([20.0 + params["x"]], dtype=jnp.float32)

    monkeypatch.setattr(fit_module, "model_mags_jax_dynamic", fake_model_mags)
    context = DspsContext(ssp=None, filters={}, model_config={})
    result = fit_population_batch_adam(
        context,
        [{"x": 0.0}, {"x": 0.2}, {"x": -0.1}],
        observed_mag=np.asarray([[20.0], [20.1], [19.9]], dtype=float),
        sigma_mag=np.asarray([[0.1], [0.1], [0.1]], dtype=float),
        fit_config={
            "maxiter": 4,
            "learning_rate": 0.01,
            "trace_mode": "optimizer",
            "trace_interval": 2,
            "free_parameters": {"x": {"initial": 0.0, "bounds": [-1.0, 1.0]}},
        },
    )

    assert np.all(result.batch.success)
    assert [row["iteration"] for row in result.batch.trace] == [1.0, 3.0]
    assert all(np.isnan(row["median_chi2"]) for row in result.batch.trace)


def test_batch_fit_jax_optimizer_options(monkeypatch) -> None:
    import euclid_dsps.fit as fit_module

    def fake_model_mags(context, args, params):
        del context, args
        return jnp.asarray([20.0 + 0.2 * params["x"]], dtype=jnp.float32)

    monkeypatch.setattr(fit_module, "model_mags_jax_dynamic", fake_model_mags)
    context = DspsContext(ssp=None, filters={}, model_config={})
    result = fit_galaxy_batch_adam(
        context,
        [{"x": 0.0}, {"x": 0.1}],
        observed_mag=np.asarray([[20.0], [20.1]], dtype=float),
        sigma_mag=np.asarray([[0.1], [0.1]], dtype=float),
        fit_config={
            "maxiter": 3,
            "learning_rate": 0.01,
            "trace_mode": "optimizer",
            "trace_interval": 2,
            "scan_unroll": 2,
            "donate_optimizer_inputs": True,
            "remat_model_mags": True,
            "batch_grad_mode": "sum",
            "free_parameters": {"x": {"initial": 0.0, "bounds": [-1.0, 1.0]}},
        },
    )

    assert np.all(result.success)
    assert result.model_mags.shape == (2, 1)
    assert [row["iteration"] for row in result.trace] == [1.0, 3.0]


def test_batch_fit_sum_grad_mode_matches_per_galaxy_grad_mode(monkeypatch) -> None:
    import euclid_dsps.fit as fit_module

    def fake_model_mags(context, args, params):
        del context, args
        return jnp.asarray(
            [20.0 + 0.2 * params["x"] - 0.05 * params["y"]],
            dtype=jnp.float32,
        )

    monkeypatch.setattr(fit_module, "model_mags_jax_dynamic", fake_model_mags)
    context = DspsContext(ssp=None, filters={}, model_config={})
    common_config = {
        "maxiter": 5,
        "learning_rate": 0.01,
        "trace_mode": "none",
        "free_parameters": {
            "x": {"initial": 0.1, "bounds": [-1.0, 1.0]},
            "y": {"initial": -0.2, "bounds": [-1.0, 1.0]},
        },
    }
    kwargs = dict(
        context=context,
        base_params_rows=[{"x": 0.0, "y": 0.0}, {"x": 0.1, "y": -0.1}],
        observed_mag=np.asarray([[20.0], [20.1]], dtype=float),
        sigma_mag=np.asarray([[0.1], [0.1]], dtype=float),
    )

    per_galaxy = fit_galaxy_batch_adam(
        **kwargs,
        fit_config={**common_config, "batch_grad_mode": "per_galaxy"},
    )
    summed = fit_galaxy_batch_adam(
        **kwargs,
        fit_config={**common_config, "batch_grad_mode": "sum"},
    )

    np.testing.assert_allclose(
        summed.best_parameter_matrix,
        per_galaxy.best_parameter_matrix,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        summed.model_mags,
        per_galaxy.model_mags,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
