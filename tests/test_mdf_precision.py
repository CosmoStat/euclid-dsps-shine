import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps import model as sed
from euclid_dsps.amortized.local_vi_diagnostic import Budget
from euclid_dsps.amortized.mdf_precision import probe, triweight_reference


def test_reference_polynomial_and_derivatives():
    from dsps.constants import LGMET_HI, LGMET_LO
    from dsps.utils import _get_bin_edges

    grid = jnp.linspace(-4, -1, 12, dtype=jnp.float64)
    edges = np.asarray(_get_bin_edges(grid, LGMET_LO, LGMET_HI))

    def fn(m):
        return sed.lognormal_mdf_lgmet_weights_jax(
            grid, m, 0.2, numerical_dtype=jnp.float64
        )

    for center in (-20.0, -3.91, -2.33, -1.02, 10.0):
        value, derivative = triweight_reference(center, 0.2, edges)
        np.testing.assert_allclose(fn(center), value, atol=2e-14, rtol=1e-12)
        np.testing.assert_allclose(
            jax.jacfwd(fn)(center), derivative, atol=2e-13, rtol=1e-11
        )
        h = 1e-5
        fd = (
            triweight_reference(center + h, 0.2, edges)[0]
            - triweight_reference(center - h, 0.2, edges)[0]
        ) / (2 * h)
        np.testing.assert_allclose(derivative, fd, atol=1e-8, rtol=1e-7)
        assert value.sum() == pytest.approx(1)


def test_probe_exports_both_precisions_without_promotion():
    report, rows = probe(np.linspace(-4, -1, 12), [-2.33], 0.2, Budget(60, 1))
    assert len(rows) == 2 * 20 * 12
    assert report["scientific_promotion"] is False
    assert report["cases"][1]["max_abs_derivative_error"] < 1e-12
    assert report["cases"][0]["output_dtype"] == "float32"
    assert report["cases"][1]["output_dtype"] == "float64"


def test_legacy_weights_exact_and_precision_guard():
    from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf

    grid = jnp.linspace(-4, -1, 8)
    raw = calc_lgmet_weights_from_lognormal_mdf(
        jnp.float32(-2.33), jnp.float32(0.2), grid.astype(jnp.float32)
    )
    expected = jnp.clip(raw.astype(jnp.float32), 0, jnp.inf)
    expected /= jnp.maximum(expected.sum(), 1e-30)
    np.testing.assert_array_equal(
        sed.lognormal_mdf_lgmet_weights_jax(grid, -2.33, 0.2), expected
    )
    old = sed.photometry_numerics({})
    assert sed.photometry_numerics(dict(mdf_weight_precision="float32_legacy")) == old
    assert sed.photometry_numerics(dict(mdf_weight_precision="float64_v1")) != old
    with pytest.raises(ValueError, match="mdf_weight_precision"):
        sed.photometry_numerics(dict(mdf_weight_precision="typo"))
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
            sed.lognormal_mdf_lgmet_weights_jax(
                grid, -2.33, 0.2, numerical_dtype=jnp.float64
            )
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_mdf_full_sed_real_dsps_synthetic_smoke():
    from test_model import _synthetic_context

    from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES

    old = _synthetic_context(
        dict(
            sfh_model="spline15d",
            agn_model="none",
            dust_model="prospector_fsps",
            igm_model="fsps_madau95",
            stellar_metallicity_model="lognormal_mdf_fixed_scatter",
            stellar_metallicity_scatter_dex=0.2,
            photometry_integrator="merged_gauss4_v1",
        )
    )
    new = copy.copy(old)
    new.model_config = dict(old.model_config, mdf_weight_precision="float64_v1")
    theta = jnp.array([0.5, 9.3, -0.2, 0.3, -0.2] + [0.02] * 10, dtype=jnp.float32)

    def forward(ctx, t):
        return sed.run_spline15d_model_jax(
            ctx, dict(zip(SPLINE15D_PARAMETER_NAMES, t, strict=True))
        ).model_mags

    before = forward(old, theta)
    f = jax.jit(lambda t: forward(new, t))
    assert np.isfinite(f(theta)).all()
    assert np.isfinite(jax.jacfwd(f)(theta)).all()
    np.testing.assert_array_equal(forward(old, theta), before)
    grid_flux = sed.context_ssp_lognormal_mdf_jax(new, -2.33, 0.2)
    assert grid_flux.dtype == jnp.float64


def test_compressed_contraction_and_survival_use_same_weights():
    from test_model import _synthetic_context

    context = _synthetic_context(
        dict(
            stellar_metallicity_model="lognormal_mdf_fixed_scatter",
            mdf_weight_precision="float64_v1",
        )
    )
    context.compressed_ssp_coeff_jax = (
        jnp.arange(3 * 24 * 2, dtype=jnp.float32).reshape(3, 24, 2) / 100
    )
    context.compressed_ssp_scale_jax = jnp.full((3, 24), 0.01, dtype=jnp.float32)
    context.compressed_ssp_basis_jax = jnp.ones((2, 96), dtype=jnp.float32)
    context.ssp_surviving_mstar_jax = jnp.linspace(
        0.3, 0.9, 3 * 24, dtype=jnp.float32
    ).reshape(3, 24)

    def expected(m):
        w = sed.lognormal_mdf_lgmet_weights_jax(
            context.ssp_lgmet_jax, m, 0.2, numerical_dtype=jnp.float64
        )
        coeff = (
            np.asarray(context.compressed_ssp_coeff_jax)
            * np.asarray(context.compressed_ssp_scale_jax)[..., None]
        )
        return jnp.einsum(
            "m,mak,kw->aw",
            w,
            jnp.asarray(coeff, dtype=jnp.float64),
            context.compressed_ssp_basis_jax.astype(jnp.float64),
        )

    def actual(m):
        return sed.context_ssp_lognormal_mdf_jax(context, m, 0.2)

    np.testing.assert_allclose(actual(-2.0), expected(-2.0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        jax.jacfwd(actual)(-2.0), jax.jacfwd(expected)(-2.0), rtol=1e-12, atol=1e-12
    )
    survival = sed._diffsky_basic_surviving_mstar_by_age_jax(
        context, context.model_config, -2.0
    )
    w = sed.lognormal_mdf_lgmet_weights_jax(
        context.ssp_lgmet_jax, -2.0, 0.2, numerical_dtype=jnp.float64
    )
    np.testing.assert_allclose(
        survival, w @ context.ssp_surviving_mstar_jax, atol=1e-12
    )
    assert survival.dtype == jnp.float64
