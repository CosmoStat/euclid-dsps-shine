import jax
import jax.numpy as jnp
import numpy as np
import pytest

from euclid_dsps.photometry_quadrature import merged_integral_jax
from scripts.photometry_merge_candidate import (
    grouped_dust,
    ordered_integral,
    specialized_integral,
    trim_zero_filter_tails,
)


@pytest.fixture(autouse=True)
def enable_float64():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("candidate", [ordered_integral, specialized_integral])
@pytest.mark.parametrize("order", [4, 8])
def test_candidate_values_and_gradients(candidate, order):
    rng = np.random.default_rng(261002)
    wave = jnp.asarray(np.sort(rng.uniform(1000, 10000, 103)))
    spectrum = jnp.asarray(rng.uniform(0.01, 2, 103))
    fw = jnp.asarray(np.linspace(3000, 11000, 35))
    ft = jnp.asarray(rng.uniform(0, 1, 35))

    def compiled(implementation):
        return jax.jit(
            jax.value_and_grad(
                lambda s, z: implementation(wave, s, fw, ft, z, order=order),
                argnums=(0, 1),
            )
        )

    old, new = compiled(merged_integral_jax), compiled(candidate)
    for z in (0.0, 0.013, 0.31, 0.9, 12.0, float(fw[10] / wave[50] - 1)):
        for a, b in zip(
            jax.tree.leaves(old(spectrum, z)),
            jax.tree.leaves(new(spectrum, z)),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("candidate", [ordered_integral, specialized_integral])
def test_duplicate_knots_and_moving_overlap(candidate):
    wave = jnp.array([2.0, 3.0, 4.0])
    fw = jnp.array([3.0, 4.0, 5.0])

    def run(implementation, z):
        return jax.value_and_grad(
            lambda zz: implementation(
                wave, jnp.ones_like(wave), fw, jnp.ones_like(fw), zz
            )
        )(z)

    for z in (0.0, 0.1, 3.0):
        np.testing.assert_allclose(
            run(candidate, z), run(merged_integral_jax, z), rtol=1e-8, atol=1e-10
        )


def test_grouped_dust_preserves_sed_and_parameter_gradients():
    from euclid_dsps.model import apply_popcosmos_dust_by_age_jax

    wave = jnp.linspace(1000.0, 15000.0, 43)
    ages = jnp.linspace(-4.0, 1.3, 17)
    sed = jnp.asarray(np.random.default_rng(12).uniform(0.01, 2, (17, 43)))
    config = {"dust_model": "prospector_fsps", "dust_tesc_logyr": 7.0}

    def compile_fn(fn):
        def total(sed, dust):
            return fn(wave, ages, sed, *dust, config, numerical_dtype=jnp.float64).sum()

        return jax.jit(jax.value_and_grad(total, argnums=(0, 1)))

    old, new = (
        compile_fn(apply_popcosmos_dust_by_age_jax),
        compile_fn(grouped_dust(apply_popcosmos_dust_by_age_jax)),
    )
    for dust in ([0.3, -0.7, 1.0], [0.0, 0.0, 0.0], [1.0, 0.2, 2.0]):
        for a, b in zip(
            jax.tree.leaves(old(sed, jnp.array(dust))),
            jax.tree.leaves(new(sed, jnp.array(dust))),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


def test_zero_tail_trim_preserves_integrals_and_gradients():
    wave = jnp.linspace(1000.0, 15000.0, 103)
    spectrum = jnp.exp(-wave / 12000.0)
    fw = jnp.linspace(2000.0, 14000.0, 31)
    ft = jnp.zeros_like(fw).at[10:16].set(jnp.array([0.1, 0.2, 0.5, 1.0, 0.4, 0.1]))
    (cropped,) = trim_zero_filter_tails(((fw, ft),))
    assert len(cropped[0]) == 8

    def kernel(curve):
        return jax.jit(
            jax.value_and_grad(
                lambda s, z: merged_integral_jax(wave, s, *curve, z, order=4),
                argnums=(0, 1),
            )
        )

    old, new = kernel((fw, ft)), kernel(cropped)
    for z in (0.0, 0.31, 3.0, 20.0):
        for a, b in zip(
            jax.tree.leaves(old(spectrum, z)),
            jax.tree.leaves(new(spectrum, z)),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


def test_specialized_grid_derivatives_fall_back_to_reference():
    args = (
        jnp.array([1000.0, 2500.0, 5000.0, 8000.0]),
        jnp.array([0.2, 1.0, 0.4, 0.1]),
        jnp.array([2000.0, 3500.0, 6000.0]),
        jnp.array([0.1, 1.0, 0.2]),
        jnp.array(0.23),
    )
    for argnum in range(5):
        old = jax.value_and_grad(merged_integral_jax, argnums=argnum)(*args)
        new = jax.value_and_grad(specialized_integral, argnums=argnum)(*args)
        for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(new), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


def test_production_autodiff_option_preserves_receipt_and_projection():
    from euclid_dsps.model import fixed_spectrum_projection_jax, photometry_numerics

    base = {"photometry_integrator": "merged_gauss4_v1"}
    optimized = {**base, "photometry_autodiff": "scalar_redshift_jvp_v1"}
    assert photometry_numerics(base) == photometry_numerics(optimized)
    with pytest.raises(ValueError, match="requires merged"):
        photometry_numerics({"photometry_autodiff": "scalar_redshift_jvp_v1"})
    with pytest.raises(ValueError, match="unsupported photometry_autodiff"):
        photometry_numerics({**base, "photometry_autodiff": "typo"})
    wave = jnp.linspace(1000.0, 10000.0, 43)
    fw = jnp.linspace(3000.0, 9000.0, 17)
    ft = jnp.exp(-(((fw - 6000.0) / 1000.0) ** 2))

    def compile_fn(option):
        return jax.jit(
            jax.value_and_grad(
                lambda s, z: fixed_spectrum_projection_jax(
                    wave, s, fw, ft, z, method="merged", order=4, autodiff=option
                ),
                argnums=(0, 1),
            )
        )

    old, new = compile_fn("reverse_v1"), compile_fn("scalar_redshift_jvp_v1")
    for a, b in zip(
        jax.tree.leaves(old(jnp.ones_like(wave), 0.3)),
        jax.tree.leaves(new(jnp.ones_like(wave), 0.3)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-30)


def test_predict_mags_routes_configured_autodiff():
    from types import SimpleNamespace

    from euclid_dsps.model import predict_mags_jax

    wave = jnp.linspace(1000.0, 10000.0, 43)
    fw = jnp.linspace(3000.0, 9000.0, 17)
    ft = jnp.exp(-(((fw - 6000.0) / 1000.0) ** 2))

    def compile_fn(option):
        context = SimpleNamespace(
            jax_filters=((fw, ft), (fw * 1.1, ft)),
            model_config={
                "photometry_integrator": "merged_gauss4_v1",
                "photometry_autodiff": option,
            },
        )
        return jax.jit(
            jax.value_and_grad(
                lambda s, z: predict_mags_jax(context, wave, s, z).sum(),
                argnums=(0, 1),
            )
        )

    old, new = compile_fn("reverse_v1"), compile_fn("scalar_redshift_jvp_v1")
    for a, b in zip(
        jax.tree.leaves(old(jnp.ones_like(wave), 0.3)),
        jax.tree.leaves(new(jnp.ones_like(wave), 0.3)),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
