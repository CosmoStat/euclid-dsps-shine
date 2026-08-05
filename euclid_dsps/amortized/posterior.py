"""Posterior families for amortized photometric inference."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import require_equinox
from .encoder import GaussianEncoder, MixtureGaussianEncoder, _diag_normal_log_prob
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


class PosteriorEncoderState(NamedTuple):
    """Encoder outputs sufficient for posterior generation without reevaluation."""

    mean: jnp.ndarray
    log_std: jnp.ndarray
    mixture_logits: jnp.ndarray | None
    mixture_means: jnp.ndarray | None
    mixture_log_stds: jnp.ndarray | None


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

    base: object
    layers: tuple
    permutations: tuple
    inverse_permutations: tuple
    family: str = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    output_space: str = eqx.field(static=True)
    base_components: int = eqx.field(static=True)

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
        base_components: int = 1,
    ) -> None:
        keys = jax.random.split(key, int(n_layers) + 1)
        self.base_components = int(base_components)
        base_class = (
            GaussianEncoder if self.base_components == 1 else MixtureGaussianEncoder
        )
        base_kwargs = dict(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            initial_log_std=initial_log_std,
        )
        if self.base_components > 1:
            base_kwargs["n_components"] = self.base_components
        self.base = base_class(keys[0], **base_kwargs)
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


def sample_posterior(
    model,
    key,
    features,
    n_samples: int,
    *,
    sample_strategy: str = "random",
    base_temperature: float = 1.0,
) -> PosteriorSample:
    """Sample any configured posterior and return exact densities in x-space."""
    state = posterior_encoder_state(model, features)
    return sample_posterior_from_state(
        model,
        key,
        state,
        n_samples,
        sample_strategy=sample_strategy,
        base_temperature=base_temperature,
    )


def posterior_encoder_state(model, features) -> PosteriorEncoderState:
    """Evaluate the encoder once and retain everything required for sampling."""
    if _is_mixture_encoder(model.encoder):
        logits, component_means, component_log_stds = (
            model.encoder.base.mixture_parameters(features)
        )
        weights = jax.nn.softmax(logits, axis=-1)
        mean = jnp.sum(weights[..., :, None] * component_means, axis=-2)
        second = jnp.sum(
            weights[..., :, None]
            * (jnp.exp(2.0 * component_log_stds) + component_means**2),
            axis=-2,
        )
        variance = jnp.maximum(second - mean**2, jnp.asarray(1.0e-12))
        log_std = 0.5 * jnp.log(variance)
        return PosteriorEncoderState(
            mean,
            log_std,
            logits,
            component_means,
            component_log_stds,
        )
    mean, log_std = model.encoder(features)
    return PosteriorEncoderState(mean, log_std, None, None, None)


def sample_posterior_from_state(
    model,
    key,
    state: PosteriorEncoderState,
    n_samples: int,
    *,
    sample_strategy: str = "random",
    base_temperature: float = 1.0,
) -> PosteriorSample:
    """Generate posterior samples from a precomputed encoder state."""
    mean, log_std = state.mean, state.log_std
    n_samples = int(n_samples)
    temperature = float(base_temperature)
    if temperature <= 0.0:
        raise ValueError("base_temperature must be positive")
    if _is_mixture_encoder(model.encoder):
        if str(sample_strategy).strip().lower() not in {"random", "iid", "independent"}:
            raise ValueError(
                "Mixture posterior sampling currently requires random sampling"
            )
        component_key, normal_key = jax.random.split(key)
        logits = state.mixture_logits
        component_means = state.mixture_means
        component_log_stds = state.mixture_log_stds
        if logits is None or component_means is None or component_log_stds is None:
            raise ValueError("Mixture encoder state is missing component parameters")
        component = jax.random.categorical(
            component_key,
            logits,
            axis=-1,
            shape=(n_samples,) + logits.shape[:-1],
        )
        selected_mean = jnp.take_along_axis(
            component_means[None, ...], component[..., None, None], axis=-2
        )[..., 0, :]
        selected_log_std = jnp.take_along_axis(
            component_log_stds[None, ...], component[..., None, None], axis=-2
        )[..., 0, :]
        selected_log_std = selected_log_std + jnp.log(
            jnp.asarray(temperature, dtype=selected_log_std.dtype)
        )
        eps = jax.random.normal(normal_key, selected_mean.shape, dtype=mean.dtype)
        base = selected_mean + jnp.exp(selected_log_std) * eps
        logq_base = _mixture_base_log_prob(
            base, logits, component_means, component_log_stds, temperature
        )
    else:
        eps = _sample_standard_normal(
            key,
            (n_samples,) + mean.shape,
            dtype=mean.dtype,
            strategy=sample_strategy,
        )
        proposal_log_std = log_std + jnp.log(
            jnp.asarray(temperature, dtype=log_std.dtype)
        )
        base = mean[None, ...] + jnp.exp(proposal_log_std)[None, ...] * eps
        logq_base = _diag_normal_log_prob(
            base,
            mean[None, ...],
            proposal_log_std[None, ...],
        )
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


def posterior_reference_from_base_mean(model, features) -> jnp.ndarray:
    """Push the encoder base mean through every configured posterior transform.

    This is a deterministic, differentiable reference used by local
    autoencoder diagnostics. It is not claimed to be the mean of a nonlinear
    transformed posterior; empirical posterior means should be computed from
    ``sample_posterior`` instead.
    """
    mean, log_std = model.encoder(features)
    if isinstance(model.encoder, GaussianEncoder):
        return mean
    context = jnp.concatenate([mean, log_std], axis=-1)
    transformed = mean
    if isinstance(model.encoder, ConditionalFlowEncoder):
        transformed, _ = model.encoder.forward(transformed, context)
        if model.encoder.output_space == "latent_x":
            return transformed
    x, _ = model.prior.forward(transformed)
    return x


def posterior_log_prob(
    model,
    features,
    x,
    *,
    base_temperature: float = 1.0,
) -> jnp.ndarray:
    """Evaluate exact conditional posterior density for supervised NPE."""
    mean, log_std = model.encoder(features)
    temperature = float(base_temperature)
    if temperature <= 0.0:
        raise ValueError("base_temperature must be positive")
    log_std = log_std + jnp.log(jnp.asarray(temperature, dtype=log_std.dtype))
    if isinstance(model.encoder, GaussianEncoder):
        return _diag_normal_log_prob(x, mean, log_std)
    context = jnp.concatenate([mean, log_std], axis=-1)
    if isinstance(model.encoder, ConditionalFlowEncoder) and (
        model.encoder.output_space == "latent_x"
    ):
        base, inverse_logdet = model.encoder.inverse(x, context)
        base_log_prob = (
            _mixture_base_log_prob_from_encoder(
                model.encoder, features, base, temperature
            )
            if _is_mixture_encoder(model.encoder)
            else _diag_normal_log_prob(base, mean, log_std)
        )
        return base_log_prob + inverse_logdet
    u, prior_inverse_logdet = model.prior.inverse(x)
    residual_inverse_logdet = jnp.zeros(u.shape[:-1], dtype=u.dtype)
    base = u
    if isinstance(model.encoder, ConditionalFlowEncoder):
        base, residual_inverse_logdet = model.encoder.inverse(u, context)
    base_log_prob = (
        _mixture_base_log_prob_from_encoder(model.encoder, features, base, temperature)
        if _is_mixture_encoder(model.encoder)
        else _diag_normal_log_prob(base, mean, log_std)
    )
    return base_log_prob + residual_inverse_logdet + prior_inverse_logdet


def posterior_mixture_diagnostics(model, features) -> dict[str, jnp.ndarray]:
    """Return exact component occupancy diagnostics, or neutral values for M=1."""
    if not _is_mixture_encoder(model.encoder):
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {
            "components": jnp.asarray(1.0),
            "entropy": zero,
            "max_weight": jnp.asarray(1.0),
        }
    logits, _means, _log_stds = model.encoder.base.mixture_parameters(features)
    weights = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(jnp.where(weights > 0, weights * jnp.log(weights), 0.0), axis=-1)
    return {
        "components": jnp.asarray(float(model.encoder.base_components)),
        "entropy": jnp.mean(entropy),
        "max_weight": jnp.mean(jnp.max(weights, axis=-1)),
    }


def _is_mixture_encoder(encoder) -> bool:
    return isinstance(encoder, ConditionalFlowEncoder) and isinstance(
        encoder.base, MixtureGaussianEncoder
    )


def _mixture_base_log_prob_from_encoder(encoder, features, value, temperature):
    logits, means, log_stds = encoder.base.mixture_parameters(features)
    return _mixture_base_log_prob(value, logits, means, log_stds, temperature)


def _mixture_base_log_prob(value, logits, means, log_stds, temperature):
    value = jnp.asarray(value)
    leading_samples = value.ndim - means.ndim + 1
    for _ in range(max(0, leading_samples)):
        logits = logits[None, ...]
        means = means[None, ...]
        log_stds = log_stds[None, ...]
    scaled_log_stds = log_stds + jnp.log(jnp.asarray(temperature, dtype=log_stds.dtype))
    component_log_prob = _diag_normal_log_prob(
        value[..., None, :], means, scaled_log_stds
    )
    return jax.scipy.special.logsumexp(
        jax.nn.log_softmax(logits, axis=-1) + component_log_prob,
        axis=-1,
    )


def _sample_standard_normal(key, shape, *, dtype, strategy: str) -> jnp.ndarray:
    normalized = str(strategy).strip().lower().replace("-", "_")
    if normalized in {"random", "iid", "independent"}:
        return jax.random.normal(key, shape, dtype=dtype)
    if normalized in {"antithetic", "paired_antithetic"}:
        n_samples = int(shape[0])
        if n_samples < 2 or n_samples % 2:
            raise ValueError(
                "Antithetic posterior sampling requires a positive even n_samples"
            )
        half = jax.random.normal(key, (n_samples // 2,) + tuple(shape[1:]), dtype=dtype)
        return jnp.concatenate((half, -half), axis=0)
    raise ValueError("sample_strategy must be random or antithetic")


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
        raise ValueError("Conditional flow output_space must be prior_base or latent_x")
    return aliases[normalized]
