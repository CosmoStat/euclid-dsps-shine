from __future__ import annotations

from euclid_dsps.fit import _context_gas_grid_nbytes, _should_jit_single_fit
from euclid_dsps.model import DspsContext, bind_dynamic_model_args, dynamic_model_args


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
