from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from euclid_dsps.fit import (
    _context_gas_grid_nbytes,
    _should_jit_single_fit,
    fit_one_galaxy,
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
