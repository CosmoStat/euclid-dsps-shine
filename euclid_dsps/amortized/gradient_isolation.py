"""Forensic flux/likelihood derivatives; no optimizer or promotion decision."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .likelihood import photometric_loglike, photometric_sigma_eff


def isolate_gradient(
    target,
    observation,
    point,
    direction,
    budget,
    *,
    band_names,
    coordinate_names,
    progress=None,
):
    """Assumes the caller verified Gaussian, complete data and zero floor/jitter.

    Gaussian differences below use float64 host arithmetic on existing flux
    outputs. They do not turn the DSPS decoder or training into float64.
    """
    budget.charge(1)
    center = target(point, observation)
    flux = jnp.asarray(center.model_flux, dtype=jnp.float32)
    sigma = jax.lax.stop_gradient(
        photometric_sigma_eff(
            observation.flux,
            flux,
            observation.flux_err,
            observation.mask,
            error_floor_frac=0,
            error_jitter=0,
        )
    )
    sigma_np = np.asarray(sigma, dtype=np.float64).reshape(-1)
    obs_np = np.asarray(observation.flux, dtype=np.float64).reshape(-1)
    center_np = np.asarray(flux, dtype=np.float64).reshape(-1)
    if len(sigma_np) != len(band_names) or not np.all(
        np.isfinite(sigma_np) & (sigma_np > 0)
    ):
        raise ValueError("expected one fully observed object")

    def flux_map(x):
        return (
            jnp.asarray(target(x, observation).model_flux, dtype=jnp.float32) / sigma
        ).reshape(-1)

    budget.charge(len(band_names), gradient=True)
    jacobian = np.asarray(jax.jacrev(flux_map)(point), dtype=np.float64).reshape(
        len(band_names), len(coordinate_names)
    )
    budget.charge(1, gradient=True)
    canonical_grad = np.asarray(
        jax.grad(lambda x: jnp.sum(target(x, observation).loglike))(point),
        dtype=np.float64,
    ).reshape(-1)
    residual = (center_np - obs_np) / sigma_np
    reconstructed_grad = -residual @ jacobian

    # Test likelihood algebra alone, perturbing fluxes in sigma units, not x.
    flux_direction = jnp.asarray(
        direction.reshape(-1) @ jnp.asarray(jacobian).T, dtype=jnp.float32
    )
    norm = jnp.linalg.norm(flux_direction)
    flux_direction = jnp.where(
        norm > 0,
        flux_direction / jnp.maximum(norm, 1e-30),
        jnp.ones_like(flux_direction) / np.sqrt(len(band_names)),
    )

    def flux_only(t):
        proposed = flux + t * flux_direction.reshape(flux.shape) * sigma
        return jnp.sum(
            photometric_loglike(
                observation.flux,
                proposed,
                observation.flux_err,
                observation.mask,
                likelihood_type="gaussian",
                error_floor_frac=0,
                error_jitter=0,
            )
        )

    flux_only_ad = float(jax.grad(flux_only)(jnp.array(0.0, dtype=jnp.float32)))
    flux_only_analytic = float(-residual @ np.asarray(flux_direction, dtype=np.float64))
    flux_only_steps = []
    for h in (0.1, 0.03, 0.01, 0.003):
        plus, minus = float(flux_only(h)), float(flux_only(-h))
        flux_only_steps.append(
            dict(
                step=h,
                plus=plus,
                minus=minus,
                finite_difference=(plus - minus) / (2 * h),
            )
        )

    directions = [("original", np.asarray(direction).reshape(-1))]
    directions += [
        (name, np.eye(len(coordinate_names))[i])
        for i, name in enumerate(coordinate_names)
    ]
    rows = []
    for label, vector in directions:
        tangent = jnp.asarray(vector, dtype=point.dtype).reshape(point.shape)
        ad_flux = jacobian @ vector
        for h in (0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625):
            budget.charge(2)
            plus = target(point + h * tangent, observation)
            minus = target(point - h * tangent, observation)
            fp = (
                np.asarray(plus.model_flux, dtype=np.float32)
                .astype(np.float64)
                .reshape(-1)
            )
            fm = (
                np.asarray(minus.model_flux, dtype=np.float32)
                .astype(np.float64)
                .reshape(-1)
            )
            rp, rm = (fp - obs_np) / sigma_np, (fm - obs_np) / sigma_np
            fd_flux = (fp - fm) / (2 * h * sigma_np)
            # Factoring the square difference avoids subtracting large log normalizers.
            fd_like = -0.5 * (rp - rm) * (rp + rm) / (2 * h)
            scalar_fd = (
                float(np.asarray(plus.loglike).item())
                - float(np.asarray(minus.loglike).item())
            ) / (2 * h)
            for i, band in enumerate(band_names):
                rows.append(
                    dict(
                        direction=label,
                        step=h,
                        band=band,
                        flux_plus=fp[i],
                        flux_minus=fm[i],
                        sigma=sigma_np[i],
                        residual_at_center=residual[i],
                        flux_ad_sigma=ad_flux[i],
                        flux_fd_sigma=fd_flux[i],
                        loglike_ad_contribution=-residual[i] * ad_flux[i],
                        loglike_centered_fd_contribution=fd_like[i],
                        canonical_loglike_ad=float(canonical_grad @ vector),
                        canonical_loglike_fd=scalar_fd,
                    )
                )
        if progress is not None:
            progress(label, rows)
    summary = dict(
        status="DIAGNOSTIC_COMPLETE",
        scientific_promotion=False,
        local_optimization_started=False,
        population_training_started=False,
        truth_used=False,
        point=np.asarray(point).tolist(),
        direction=np.asarray(direction).tolist(),
        band_names=list(band_names),
        coordinate_names=list(coordinate_names),
        flux_sigma_jacobian=jacobian.tolist(),
        canonical_loglike_gradient=canonical_grad.tolist(),
        gaussian_chain_gradient=reconstructed_grad.tolist(),
        maximum_abs_chain_gradient_delta=float(
            np.max(np.abs(canonical_grad - reconstructed_grad))
        ),
        likelihood_only=dict(
            autodiff=flux_only_ad, analytic=flux_only_analytic, steps=flux_only_steps
        ),
        decoder_output_dtype=str(center.model_flux.dtype),
        likelihood_dtype=str(center.loglike.dtype),
        interpretation="forensic measurements, not an acceptance gate; centered FD removes final scalar cancellation only",
    )
    return summary, rows
