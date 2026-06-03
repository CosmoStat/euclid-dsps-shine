"""Negative ELBO objective for joint amortized inference training."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .config import require_equinox
from .decoder import mock_model_flux_from_x, model_flux_from_x
from .latent import LatentSpec
from .likelihood import photometric_loglike

eqx = require_equinox()


@dataclass
class AmortizedModel(eqx.Module):
    encoder: object
    prior: object


def negative_elbo(
    model: AmortizedModel,
    batch,
    latent_spec: LatentSpec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    key,
    n_samples: int,
    kl_weight: float,
    likelihood_config: dict,
    use_mock_decoder: bool = False,
    mock_decoder_params=None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return the Monte Carlo negative ELBO and diagnostic metrics.

    ``logq`` and ``logp`` are exact pointwise. The KL is estimated by Monte
    Carlo as ``E_q[logq - logp]`` because the RealNVP prior has nonlinear
    coupling networks, so the expectation has no Gaussian closed form.
    """
    x_samples, logq = model.encoder.sample_and_log_prob(
        key,
        batch.features,
        int(n_samples),
    )
    if use_mock_decoder:
        if mock_decoder_params is None:
            raise ValueError(
                "mock_decoder_params is required with use_mock_decoder=True"
            )
        model_flux = mock_model_flux_from_x(
            x_samples,
            mock_decoder_params["weights"],
            mock_decoder_params["bias"],
        )
    else:
        model_flux = model_flux_from_x(
            x_samples,
            latent_spec,
            context,
            model_args,
            parameter_names,
        )
    loglike = photometric_loglike(
        obs_flux=batch.flux,
        model_flux=model_flux,
        obs_err=batch.flux_err,
        mask=batch.mask,
        likelihood_type=str(likelihood_config.get("type", "student_t")),
        student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
    )
    logp = model.prior.log_prob(x_samples)
    kl_mc = logq - logp
    loss_terms = -loglike + float(kl_weight) * kl_mc
    loss = jnp.mean(loss_terms)
    obs = batch.flux[None, :, :]
    mask = batch.mask[None, :, :]
    residual = jnp.where(mask, model_flux - obs, 0.0)
    n_valid = jnp.maximum(jnp.sum(mask), 1)
    metrics = {
        "loss": loss,
        "negative_loglike": jnp.mean(-loglike),
        "loglike_mean": jnp.mean(loglike),
        "logprior_mean": jnp.mean(logp),
        "logq_mean": jnp.mean(logq),
        "kl_mc_mean": jnp.mean(kl_mc),
        "model_flux_mean": jnp.mean(model_flux),
        "residual_rms": jnp.sqrt(jnp.sum(residual**2) / n_valid),
        "finite_fraction": jnp.mean(jnp.isfinite(loss_terms).astype(jnp.float32)),
    }
    return loss, metrics
