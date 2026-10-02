"""Integration of existing linear SED/filter interpolants.

The default decoder remains legacy. Versioned merged integration resolves BOTH
wavelength grids; the NumPy reference integrates each affine-product segment.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero


def merged_integral_jax(wave, spectrum, filter_wave, transmission, z, *, order=8):
    """Integral T(lambda) L(lambda/(1+z)) / lambda on overlapping support.

    All inputs must be float64 for the diagnostic. Repeated clipped knots give
    zero-width intervals; retaining them keeps shapes static under JAX tracing.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    shifted = wave * (1 + z)
    lo = jnp.maximum(shifted[0], filter_wave[0])
    hi = jnp.maximum(lo, jnp.minimum(shifted[-1], filter_wave[-1]))
    knots = jnp.sort(jnp.clip(jnp.concatenate((shifted, filter_wave)), lo, hi))
    widths = jnp.diff(knots)
    t = (jnp.asarray(nodes) + 1) / 2
    samples = knots[:-1, None] + widths[:, None] * t
    # Both interpolants are affine between merged knots. Search once per knot,
    # then evaluate every quadrature node by blending the endpoint values.
    lum_knots = jnp.interp(knots, shifted, spectrum, left=0, right=0)
    trans_knots = jnp.interp(knots, filter_wave, transmission, left=0, right=0)
    lum = lum_knots[:-1, None] + jnp.diff(lum_knots)[:, None] * t
    trans = trans_knots[:-1, None] + jnp.diff(trans_knots)[:, None] * t
    return jnp.sum(
        widths * jnp.sum(lum * trans / samples * jnp.asarray(weights), axis=-1) / 2
    )


_reference_merged_integral = merged_integral_jax


@partial(jax.custom_jvp, nondiff_argnums=(5,))
def _scalar_redshift_integral(wave, spectrum, filter_wave, transmission, z, order):
    return _reference_merged_integral(
        wave, spectrum, filter_wave, transmission, z, order=order
    )


def _scalar_redshift_integral_jvp(order, primals, tangents):
    wave, spectrum, fw, ft, z = primals
    dw, ds, dfw, dft, dz = tangents

    def evaluate(*args):
        return _reference_merged_integral(*args, order=order)

    # Trainable wavelength/filter grids use the general derivative rule.
    if any(not isinstance(t, SymbolicZero) for t in (dw, dfw, dft)):
        actual = tuple(
            jnp.zeros_like(p) if isinstance(t, SymbolicZero) else t
            for p, t in zip(primals, tangents, strict=True)
        )
        return jax.jvp(evaluate, primals, actual)
    if isinstance(dz, SymbolicZero):
        value = evaluate(*primals)
        redshift_term = jnp.zeros_like(value)
    else:
        value, derivative_z = jax.jvp(
            lambda zz: evaluate(wave, spectrum, fw, ft, zz), (z,), (jnp.ones_like(z),)
        )
        redshift_term = derivative_z * dz
    # Linearity in the SED and scalar forward-mode in z reduce AD temporaries.
    spectrum_term = (
        jnp.zeros_like(value)
        if isinstance(ds, SymbolicZero)
        else evaluate(wave, ds, fw, ft, z)
    )
    return value, spectrum_term + redshift_term


_scalar_redshift_integral.defjvp(_scalar_redshift_integral_jvp, symbolic_zeros=True)


def merged_integral_scalar_redshift_jvp_jax(
    wave, spectrum, filter_wave, transmission, z, *, order=8
):
    """Same quadrature with a memory-saving first-derivative implementation.

    This is an opt-in for fixed grids, not a different integrator or precision.
    Active grid derivatives fall back to the reference JVP.
    """
    return _scalar_redshift_integral(
        wave, spectrum, filter_wave, transmission, z, order
    )


def filter_integral_jax(filter_wave, transmission, *, order=8):
    """Integral T/lambda using the same candidate quadrature, without an SED."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    width = jnp.diff(filter_wave)
    t = (jnp.asarray(nodes) + 1) / 2
    samples = filter_wave[:-1, None] + width[:, None] * t
    trans = transmission[:-1, None] + jnp.diff(transmission)[:, None] * t
    return jnp.sum(width * jnp.sum(trans / samples * jnp.asarray(weights), axis=-1) / 2)


def _validate_grid(wave, values):
    wave, values = (
        np.asarray(wave, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )
    if wave.ndim != 1 or values.shape != wave.shape or len(wave) < 2:
        raise ValueError("one-dimensional matching grids required")
    if (
        not np.isfinite(wave).all()
        or not np.isfinite(values).all()
        or np.any(wave <= 0)
        or np.any(np.diff(wave) <= 0)
    ):
        raise ValueError("finite positive strictly increasing wavelength grid required")
    return wave, values


def linear_product_integral_numpy(wave, spectrum, filter_wave, transmission, z):
    """Analytic segment reference independent of JAX, AD and Gauss quadrature.

    On each merged segment let t=(lambda-lo)/width. Integrate the quadratic
    product divided by 1+r*t, r=width/lo. Small-r series avoid cancellation in
    recurrence formulas. This is exact for the supplied linear interpolants up
    to floating arithmetic, not a guarantee that the original assets resolve
    all physical spectral/filter structure.
    """
    wave, spectrum = _validate_grid(wave, spectrum)
    fw, trans = _validate_grid(filter_wave, transmission)
    if not np.isfinite(z) or z <= -1:
        raise ValueError("invalid redshift")
    shifted = wave * (1 + z)
    lo, hi = max(shifted[0], fw[0]), min(shifted[-1], fw[-1])
    if hi <= lo:
        return 0.0
    knots = np.unique(np.clip(np.concatenate((shifted, fw)), lo, hi))
    r = np.diff(knots) / knots[:-1]
    lum = np.interp(knots, shifted, spectrum)
    filt = np.interp(knots, fw, trans)
    a = lum[:-1] * filt[:-1]
    b = lum[:-1] * np.diff(filt) + filt[:-1] * np.diff(lum)
    c = np.diff(lum) * np.diff(filt)
    j0 = np.log1p(r) / r
    j1 = (1 - j0) / r
    j2 = (0.5 - j1) / r
    small = r < 0.05
    moments = [j0, j1, j2]
    for n in range(3):
        moments[n][small] = sum((-r[small]) ** k / (n + k + 1) for k in range(16))
    return float(np.sum(r * (a * moments[0] + b * moments[1] + c * moments[2])))


def filter_integral_numpy(wave, transmission):
    wave, transmission = _validate_grid(wave, transmission)
    return linear_product_integral_numpy(
        wave, np.ones_like(wave), wave, transmission, 0.0
    )


def interpolation_knot_audit(wave, filter_wave, z, steps):
    """Nearest z where a filter sample crosses an SED knot; no dense NxM table."""
    wave = np.asarray(wave, dtype=np.float64)
    fw = np.asarray(filter_wave, dtype=np.float64)
    indices = np.searchsorted(wave, fw / (1 + z))
    candidates = np.concatenate(
        [fw / wave[np.clip(indices + d, 0, len(wave) - 1)] - 1 for d in (-1, 0)]
    )
    nearest = float(np.min(np.abs(candidates - z)))
    crossings = []
    for h in steps:
        plus = np.searchsorted(wave, fw / (1 + z + h))
        minus = np.searchsorted(wave, fw / (1 + z - h))
        crossings.append(
            dict(
                physical_z_step=float(h),
                switched_filter_samples=int(np.count_nonzero(plus != minus)),
            )
        )
    return dict(nearest_knot_distance_z=nearest, crossings=crossings)
