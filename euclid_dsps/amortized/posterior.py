"""Posterior families for amortized photometric inference."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import require_equinox
from .encoder import GaussianEncoder, _diag_normal_log_prob
from .flows import (
    _apply_net,
    _rational_quadratic_spline,
    _scale_last_linear,
)

eqx = require_equinox()


class PosteriorSample(NamedTuple):
    x: jnp.ndarray
    logq: jnp.ndarray
    logprior: jnp.ndarray
    base_mean: jnp.ndarray
    base_log_std: jnp.ndarray
    residual_logdet: jnp.ndarray


class PriorTransportGaussianEncoder(eqx.Module):
    """Diagonal Gaussian in the frozen prior's standard-normal coordinates."""

    base: GaussianEncoder

    def __init__(self, key, **kwargs) -> None:
        self.base = GaussianEncoder(key, **kwargs)

    def __call__(self, features):
        return self.base(features)


class _ConditionalCoupling(eqx.Module):
    mask: jnp.ndarray
    net: object
    family: str = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    scale_clamp: float = eqx.field(static=True)
    shift_clamp: float = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)
    min_bin_width: float = eqx.field(static=True)
    min_bin_height: float = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        latent_dim: int,
        context_dim: int,
        hidden_size: int,
        mask,
        family: str,
        n_bins: int,
        scale_clamp: float,
        shift_clamp: float,
        tail_bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
        init_scale: float,
    ) -> None:
        family = _normalize_family(family)
        multiplier = 2 if family == "realnvp" else 3 * int(n_bins) + 1
        self.mask = jnp.asarray(mask, dtype=jnp.bool_)
        self.net = eqx.nn.MLP(
            in_size=int(latent_dim) + int(context_dim),
            out_size=int(latent_dim) * multiplier,
            width_size=int(hidden_size),
            depth=2,
            activation=jax.nn.gelu,
            key=key,
        )
        self.net = _scale_last_linear(self.net, float(init_scale))
        self.family = family
        self.latent_dim = int(latent_dim)
        self.n_bins = int(n_bins)
        self.scale_clamp = float(scale_clamp)
        self.shift_clamp = float(shift_clamp)
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

    def forward(self, value, context):
        return self._transform(value, context, inverse=False)

    def inverse(self, value, context):
        return self._transform(value, context, inverse=True)

    def _transform(self, value, context, *, inverse: bool):
        value = jnp.asarray(value, dtype=jnp.float32)
        context = _broadcast_context(context, value.shape[:-1])
        mask = jnp.asarray(self.mask, dtype=value.dtype)
        active = 1.0 - mask
        masked = value * mask
        raw = _apply_net(self.net, jnp.concatenate([masked, context], axis=-1))
        if self.family == "realnvp":
            raw = raw.reshape(value.shape[:-1] + (self.latent_dim, 2))
            log_scale = self.scale_clamp * jnp.tanh(raw[..., 0]) * active
            shift = self.shift_clamp * jnp.tanh(raw[..., 1]) * active
            if inverse:
                transformed = (value - shift) * jnp.exp(-log_scale)
                logdet = -jnp.sum(log_scale, axis=-1)
            else:
                transformed = value * jnp.exp(log_scale) + shift
                logdet = jnp.sum(log_scale, axis=-1)
        else:
            params = raw.reshape(
                value.shape[:-1] + (self.latent_dim, 3 * self.n_bins + 1)
            )
            transformed, element_logdet = _rational_quadratic_spline(
                value,
                params,
                inverse=inverse,
                n_bins=self.n_bins,
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            logdet = jnp.sum(active * element_logdet, axis=-1)
        return masked + active * transformed, logdet


class ConditionalFlowEncoder(eqx.Module):
    """MLP Gaussian base followed by a conditional residual coupling flow."""

    base: GaussianEncoder
    layers: tuple
    permutations: tuple
    inverse_permutations: tuple
    family: str = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    output_space: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        input_dim: int,
        latent_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: str,
        log_std_min: float,
        log_std_max: float,
        initial_log_std: float,
        family: str = "realnvp",
        n_layers: int = 4,
        hidden_size: int = 128,
        n_bins: int = 8,
        scale_clamp: float = 0.5,
        shift_clamp: float = 3.0,
        tail_bound: float = 8.0,
        min_bin_width: float = 1.0e-3,
        min_bin_height: float = 1.0e-3,
        min_derivative: float = 1.0e-3,
        init_scale: float = 0.0,
        output_space: str = "prior_base",
    ) -> None:
        keys = jax.random.split(key, int(n_layers) + 1)
        self.base = GaussianEncoder(
            keys[0],
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            initial_log_std=initial_log_std,
        )
        masks = tuple(
            (jnp.arange(int(latent_dim)) % 2 == index % 2)
            for index in range(int(n_layers))
        )
        self.layers = tuple(
            _ConditionalCoupling(
                keys[index + 1],
                latent_dim=latent_dim,
                context_dim=2 * int(latent_dim),
                hidden_size=hidden_size,
                mask=masks[index],
                family=family,
                n_bins=n_bins,
                scale_clamp=scale_clamp,
                shift_clamp=shift_clamp,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
                init_scale=init_scale,
            )
            for index in range(int(n_layers))
        )
        permutations = tuple(
            jnp.roll(jnp.arange(int(latent_dim)), index + 1)
            for index in range(int(n_layers))
        )
        self.permutations = permutations
        self.inverse_permutations = tuple(jnp.argsort(value) for value in permutations)
        self.family = _normalize_family(family)
        self.latent_dim = int(latent_dim)
        self.output_space = _normalize_output_space(output_space)

    def __call__(self, features):
        return self.base(features)

    def forward(self, value, context):
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for layer, permutation in zip(self.layers, self.permutations, strict=True):
            value, delta = layer.forward(value, context)
            value = jnp.take(value, permutation, axis=-1)
            logdet = logdet + delta
        return value, logdet

    def inverse(self, value, context):
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        items = zip(
            reversed(self.layers),
            reversed(self.inverse_permutations),
            strict=True,
        )
        for layer, inverse_permutation in items:
            value = jnp.take(value, inverse_permutation, axis=-1)
            value, delta = layer.inverse(value, context)
            logdet = logdet + delta
        return value, logdet


def sample_posterior(model, key, features, n_samples: int) -> PosteriorSample:
    """Sample any configured posterior and return exact densities in x-space."""
    mean, log_std = model.encoder(features)
    eps = jax.random.normal(
        key,
        (int(n_samples),) + mean.shape,
        dtype=mean.dtype,
    )
    base = mean[None, ...] + jnp.exp(log_std)[None, ...] * eps
    logq_base = _diag_normal_log_prob(base, mean[None, ...], log_std[None, ...])
    residual_logdet = jnp.zeros_like(logq_base)

    if isinstance(model.encoder, GaussianEncoder):
        x = base
        logq = logq_base
        logprior = model.prior.log_prob(x)
    elif isinstance(model.encoder, ConditionalFlowEncoder) and (
        model.encoder.output_space == "latent_x"
    ):
        context = jnp.concatenate([mean, log_std], axis=-1)
        x, residual_logdet = model.encoder.forward(base, context)
        logq = logq_base - residual_logdet
        logprior = model.prior.log_prob(x)
    else:
        context = jnp.concatenate([mean, log_std], axis=-1)
        u = base
        if isinstance(model.encoder, ConditionalFlowEncoder):
            u, residual_logdet = model.encoder.forward(u, context)
        x, prior_logdet = model.prior.forward(u)
        logq = logq_base - residual_logdet - prior_logdet
        base_logprior = _standard_normal_log_prob(u)
        logprior = base_logprior - prior_logdet
    return PosteriorSample(x, logq, logprior, mean, log_std, residual_logdet)


def posterior_log_prob(model, features, x) -> jnp.ndarray:
    """Evaluate exact conditional posterior density for supervised NPE."""
    mean, log_std = model.encoder(features)
    if isinstance(model.encoder, GaussianEncoder):
        return _diag_normal_log_prob(x, mean, log_std)
    context = jnp.concatenate([mean, log_std], axis=-1)
    if isinstance(model.encoder, ConditionalFlowEncoder) and (
        model.encoder.output_space == "latent_x"
    ):
        base, inverse_logdet = model.encoder.inverse(x, context)
        return _diag_normal_log_prob(base, mean, log_std) + inverse_logdet
    u, prior_inverse_logdet = model.prior.inverse(x)
    residual_inverse_logdet = jnp.zeros(u.shape[:-1], dtype=u.dtype)
    base = u
    if isinstance(model.encoder, ConditionalFlowEncoder):
        base, residual_inverse_logdet = model.encoder.inverse(u, context)
    return (
        _diag_normal_log_prob(base, mean, log_std)
        + residual_inverse_logdet
        + prior_inverse_logdet
    )


def _broadcast_context(context, leading_shape):
    context = jnp.asarray(context, dtype=jnp.float32)
    while context.ndim < len(leading_shape) + 1:
        context = context[None, ...]
    return jnp.broadcast_to(context, tuple(leading_shape) + (context.shape[-1],))


def _standard_normal_log_prob(value):
    return -0.5 * jnp.sum(value**2 + jnp.log(2.0 * jnp.pi), axis=-1)


def _normalize_family(value: str) -> str:
    family = str(value).strip().lower().replace("-", "_")
    aliases = {
        "realnvp": "realnvp",
        "affine": "realnvp",
        "rq_spline": "rq_spline",
        "rqspline": "rq_spline",
        "neural_spline": "rq_spline",
    }
    if family not in aliases:
        raise ValueError("Conditional flow family must be realnvp or rq_spline")
    return aliases[family]


def _normalize_output_space(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "prior_base": "prior_base",
        "shared_prior_transport": "prior_base",
        "latent_x": "latent_x",
        "independent_x": "latent_x",
    }
    if normalized not in aliases:
        raise ValueError(
            "Conditional flow output_space must be prior_base or latent_x"
        )
    return aliases[normalized]
