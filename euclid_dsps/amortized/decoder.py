"""DSPS decoder wrappers for amortized inference."""

from __future__ import annotations

import jax.numpy as jnp

from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
from euclid_dsps.photometry import abmag_to_fnu_cgs_jax

from .latent import LatentSpec, x_to_theta


def model_flux_from_x(
    x: jnp.ndarray,
    latent_spec: LatentSpec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
) -> jnp.ndarray:
    """Decode latent ``x`` to model fluxes through the fixed DSPS model."""
    theta = x_to_theta(x, latent_spec)
    model_mags = model_mags_from_theta_matrix_jax(
        context,
        model_args,
        theta,
        parameter_names,
    )
    return abmag_to_fnu_cgs_jax(model_mags)


def mock_model_flux_from_x(x, weights, bias):
    """Fast linear mock decoder for asset-free tests and synthetic smoke."""
    x = jnp.asarray(x, dtype=jnp.float32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    bias = jnp.asarray(bias, dtype=jnp.float32)
    raw = jnp.einsum("...d,db->...b", x, weights) + bias
    return jnp.exp(jnp.clip(raw, -30.0, 30.0))
