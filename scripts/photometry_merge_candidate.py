"""Experimental ordered-grid merge; not a production numerical default."""

import jax.numpy as jnp
import numpy as np

from euclid_dsps.photometry_quadrature import (
    merged_integral_jax as reference_integral,
)
from euclid_dsps.photometry_quadrature import (
    merged_integral_scalar_redshift_jvp_jax as specialized_integral,
)

__all__ = [
    "ordered_integral",
    "grouped_dust",
    "trim_zero_filter_tails",
    "specialized_integral",
    "reference_integral",
]


def ordered_knots(wave, filter_wave, z):
    shifted = wave * (1 + z)
    lo = jnp.maximum(shifted[0], filter_wave[0])
    hi = jnp.maximum(lo, jnp.minimum(shifted[-1], filter_wave[-1]))
    # Stable merge: spectral knots precede equal filter knots. Ranks are discrete;
    # scatter retains derivatives of wavelength values, as the sorted merge does.
    spectral_rank = jnp.arange(len(wave)) + jnp.searchsorted(
        filter_wave, shifted, side="left"
    )
    filter_rank = jnp.arange(len(filter_wave)) + jnp.searchsorted(
        shifted, filter_wave, side="right"
    )
    knots = jnp.zeros(len(wave) + len(filter_wave), dtype=shifted.dtype)
    knots = knots.at[spectral_rank].set(shifted)
    knots = knots.at[filter_rank].set(filter_wave)
    return jnp.clip(knots, lo, hi)


def ordered_integral(wave, spectrum, filter_wave, transmission, z, *, order=8):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    shifted = wave * (1 + z)
    knots = ordered_knots(wave, filter_wave, z)
    widths = jnp.diff(knots)
    t = (jnp.asarray(nodes) + 1) / 2
    samples = knots[:-1, None] + widths[:, None] * t
    lum_knots = jnp.interp(knots, shifted, spectrum, left=0, right=0)
    trans_knots = jnp.interp(knots, filter_wave, transmission, left=0, right=0)
    lum = lum_knots[:-1, None] + jnp.diff(lum_knots)[:, None] * t
    trans = trans_knots[:-1, None] + jnp.diff(trans_knots)[:, None] * t
    return jnp.sum(
        widths * jnp.sum(lum * trans / samples * jnp.asarray(weights), axis=-1) / 2
    )


def grouped_dust(original):
    def apply(
        wave,
        ages,
        sed_by_age,
        tau2,
        dust_index_n,
        tau1_over_tau2,
        config,
        *,
        numerical_dtype,
    ):
        if config.get("dust_model") != "prospector_fsps":
            return original(
                wave,
                ages,
                sed_by_age,
                tau2,
                dust_index_n,
                tau1_over_tau2,
                config,
                numerical_dtype=numerical_dtype,
            )
        young = jnp.asarray(ages, dtype=numerical_dtype) + 9 <= config.get(
            "dust_tesc_logyr", 7.0
        )
        grouped = jnp.stack(
            (
                jnp.sum(jnp.where(young[:, None], sed_by_age, 0), axis=0),
                jnp.sum(jnp.where(young[:, None], 0, sed_by_age), axis=0),
            )
        )
        return original(
            wave,
            ages[jnp.array([0, len(ages) - 1])],
            grouped,
            tau2,
            dust_index_n,
            tau1_over_tau2,
            config,
            numerical_dtype=numerical_dtype,
        )

    return apply


def trim_zero_filter_tails(curves):
    """Drop only zero tails, retaining a zero boundary knot on each side."""
    result = []
    for wave, transmission in curves:
        nonzero = np.flatnonzero(np.asarray(transmission) != 0)
        if len(nonzero):
            start, end = (
                max(0, int(nonzero[0]) - 1),
                min(len(wave), int(nonzero[-1]) + 2),
            )
            result.append((wave[start:end], transmission[start:end]))
        else:
            result.append((wave, transmission))
    return tuple(result)
