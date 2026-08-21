"""Posterior families for amortized photometric inference."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import require_equinox
from .encoder import (
    GaussianEncoder,
    MixtureGaussianEncoder,
    PassbandSetEncoder,
    _diag_normal_log_prob,
)
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
    flow_context: jnp.ndarray


class DefensiveProposalSample(NamedTuple):
    """Draws and exact density from a configurable defensive mixture."""

    x: jnp.ndarray
    logproposal: jnp.ndarray
    component_index: jnp.ndarray
    component_fractions: jnp.ndarray
    posterior_tempered_fraction: jnp.ndarray
    prior_fraction: jnp.ndarray
    maximum_posterior_temperature: jnp.ndarray


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


class _ConditionalAutoregressiveSpline(eqx.Module):
    """Exact scalar-autoregressive RQ-spline transform.

    Sampling is sequential in the 15 latent coordinates. Density evaluation is
    exact and uses the same conditioners, preserving the ordinary-IW contract.
    """

    conditioners: tuple
    latent_dim: int = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
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
        n_bins: int,
        tail_bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
        init_scale: float,
    ) -> None:
        keys = jax.random.split(key, int(latent_dim))
        output_dim = 3 * int(n_bins) + 1
        self.conditioners = tuple(
            _scale_last_linear(
                eqx.nn.MLP(
                    in_size=int(context_dim) + coordinate,
                    out_size=output_dim,
                    width_size=int(hidden_size),
                    depth=2,
                    activation=jax.nn.gelu,
                    key=keys[coordinate],
                ),
                float(init_scale),
            )
            for coordinate in range(int(latent_dim))
        )
        self.latent_dim = int(latent_dim)
        self.n_bins = int(n_bins)
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

    def forward(self, value, context):
        value = jnp.asarray(value, dtype=jnp.float32)
        context = _broadcast_context(context, value.shape[:-1])
        transformed = jnp.zeros_like(value)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for coordinate, conditioner in enumerate(self.conditioners):
            inputs = jnp.concatenate((context, transformed[..., :coordinate]), axis=-1)
            params = _apply_net(conditioner, inputs)
            output, delta = _rational_quadratic_spline(
                value[..., coordinate],
                params,
                inverse=False,
                n_bins=self.n_bins,
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            transformed = transformed.at[..., coordinate].set(output)
            logdet = logdet + delta
        return transformed, logdet

    def inverse(self, value, context):
        value = jnp.asarray(value, dtype=jnp.float32)
        context = _broadcast_context(context, value.shape[:-1])
        base = jnp.zeros_like(value)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for coordinate, conditioner in enumerate(self.conditioners):
            inputs = jnp.concatenate((context, value[..., :coordinate]), axis=-1)
            params = _apply_net(conditioner, inputs)
            output, delta = _rational_quadratic_spline(
                value[..., coordinate],
                params,
                inverse=True,
                n_bins=self.n_bins,
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
            )
            base = base.at[..., coordinate].set(output)
            logdet = logdet + delta
        return base, logdet


class ConditionalFlowEncoder(eqx.Module):
    """MLP Gaussian base followed by a conditional residual coupling flow."""

    base: object
    layers: tuple
    permutations: tuple
    inverse_permutations: tuple
    family: str = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    context_encoder_type: str = eqx.field(static=True)
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
        context_encoder_type: str = "base_moments",
        set_n_bands: int | None = None,
        set_token_dim: int = 64,
        set_context_dim: int = 128,
        set_num_heads: int = 4,
        set_num_layers: int = 2,
    ) -> None:
        keys = jax.random.split(key, int(n_layers) + 1)
        self.base_components = int(base_components)
        context_encoder_type = _normalize_context_encoder(context_encoder_type)
        base_kwargs = dict(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            initial_log_std=initial_log_std,
        )
        if context_encoder_type == "passband_set_transformer":
            if self.base_components != 1:
                raise ValueError(
                    "passband-set conditional flows currently require base_components=1"
                )
            if set_n_bands is None:
                raise ValueError("set_n_bands is required for passband-set context")
            self.base = PassbandSetEncoder(
                keys[0],
                input_dim=input_dim,
                latent_dim=latent_dim,
                n_bands=int(set_n_bands),
                token_dim=int(set_token_dim),
                context_dim=int(set_context_dim),
                n_heads=int(set_num_heads),
                n_layers=int(set_num_layers),
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                initial_log_std=initial_log_std,
            )
            context_dim = int(set_context_dim)
        else:
            base_class = (
                GaussianEncoder if self.base_components == 1 else MixtureGaussianEncoder
            )
            if self.base_components > 1:
                base_kwargs["n_components"] = self.base_components
            self.base = base_class(keys[0], **base_kwargs)
            context_dim = 2 * int(latent_dim)
        masks = tuple(
            (jnp.arange(int(latent_dim)) % 2 == index % 2)
            for index in range(int(n_layers))
        )
        normalized_family = _normalize_family(family)
        if normalized_family == "autoregressive_rq_spline":
            self.layers = tuple(
                _ConditionalAutoregressiveSpline(
                    keys[index + 1],
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    hidden_size=hidden_size,
                    n_bins=n_bins,
                    tail_bound=tail_bound,
                    min_bin_width=min_bin_width,
                    min_bin_height=min_bin_height,
                    min_derivative=min_derivative,
                    init_scale=init_scale,
                )
                for index in range(int(n_layers))
            )
        else:
            self.layers = tuple(
                _ConditionalCoupling(
                    keys[index + 1],
                    latent_dim=latent_dim,
                    context_dim=context_dim,
                    hidden_size=hidden_size,
                    mask=masks[index],
                    family=normalized_family,
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
        self.family = normalized_family
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.context_encoder_type = context_encoder_type
        self.output_space = _normalize_output_space(output_space)

    def __call__(self, features):
        return self.base(features)

    def flow_context(self, features, mean=None, log_std=None):
        if self.context_encoder_type == "passband_set_transformer":
            return self.base.context(features)
        if mean is None or log_std is None:
            mean, log_std = self.base(features)
        return jnp.concatenate((mean, log_std), axis=-1)

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
            model.encoder.flow_context(features, mean, log_std),
        )
    mean, log_std = model.encoder(features)
    context = (
        model.encoder.flow_context(features, mean, log_std)
        if isinstance(model.encoder, ConditionalFlowEncoder)
        else jnp.concatenate((mean, log_std), axis=-1)
    )
    return PosteriorEncoderState(mean, log_std, None, None, None, context)


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
        context = state.flow_context
        x, residual_logdet = model.encoder.forward(base, context)
        logq = logq_base - residual_logdet
        logprior = model.prior.log_prob(x)
    else:
        context = state.flow_context
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
    context = (
        model.encoder.flow_context(features, mean, log_std)
        if isinstance(model.encoder, ConditionalFlowEncoder)
        else jnp.concatenate([mean, log_std], axis=-1)
    )
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
    proposal_log_std = log_std + jnp.log(jnp.asarray(temperature, dtype=log_std.dtype))
    if isinstance(model.encoder, GaussianEncoder):
        return _diag_normal_log_prob(x, mean, proposal_log_std)
    context = (
        model.encoder.flow_context(features, mean, log_std)
        if isinstance(model.encoder, ConditionalFlowEncoder)
        else jnp.concatenate([mean, log_std], axis=-1)
    )
    if isinstance(model.encoder, ConditionalFlowEncoder) and (
        model.encoder.output_space == "latent_x"
    ):
        base, inverse_logdet = model.encoder.inverse(x, context)
        base_log_prob = (
            _mixture_base_log_prob_from_encoder(
                model.encoder, features, base, temperature
            )
            if _is_mixture_encoder(model.encoder)
            else _diag_normal_log_prob(base, mean, proposal_log_std)
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
        else _diag_normal_log_prob(base, mean, proposal_log_std)
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


def defensive_posterior_proposal(
    model: Any,
    key: jax.Array,
    features: jnp.ndarray,
    n_samples: int,
    components: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> DefensiveProposalSample:
    """Sample a stratified defensive mixture and evaluate its complete density.

    Every selected draw is scored under every component. In particular, a
    draw originating from a tempered posterior is never divided only by that
    component density.
    """
    count = int(n_samples)
    if count <= 0:
        raise ValueError("n_samples must be positive")
    normalized = _normalize_defensive_components(components)
    requested_fractions = np.asarray([item[2] for item in normalized], dtype=float)
    component_counts = _allocate_component_counts(count, requested_fractions)
    realized_fractions = component_counts / float(count)
    fractions = jnp.asarray(realized_fractions, dtype=jnp.float32)
    keys = jax.random.split(key, len(normalized) + 1)
    n_objects = int(features.shape[0])
    candidates = []
    component_labels = []
    for component_id, (component_key, component_count, component) in enumerate(
        zip(keys[1:], component_counts, normalized, strict=True)
    ):
        source, temperature, _fraction = component
        if source == "posterior":
            draw = sample_posterior(
                model,
                component_key,
                features,
                int(component_count),
                base_temperature=temperature,
            ).x
        else:
            draw = model.prior.sample(
                component_key, int(component_count) * n_objects
            ).reshape(int(component_count), n_objects, -1)
        candidates.append(draw)
        component_labels.append(
            jnp.full(
                (int(component_count), n_objects),
                component_id,
                dtype=jnp.int32,
            )
        )
    x = jnp.concatenate(candidates, axis=0)
    component_index = jnp.concatenate(component_labels, axis=0)
    permutation = jax.random.permutation(keys[0], count)
    x = jax.lax.stop_gradient(jnp.take(x, permutation, axis=0))
    component_index = jnp.take(component_index, permutation, axis=0)
    component_log_prob = []
    for source, temperature, _fraction in normalized:
        if source == "posterior":
            value = posterior_log_prob(
                model,
                features,
                x,
                base_temperature=temperature,
            )
        else:
            value = model.prior.log_prob(x)
        component_log_prob.append(value)
    logproposal = defensive_mixture_log_prob(
        jnp.stack(component_log_prob, axis=0), fractions
    )
    tempered_fraction = sum(
        float(realized_fractions[index])
        for index, (source, temperature, _fraction) in enumerate(normalized)
        if source == "posterior" and temperature > 1.0
    )
    prior_fraction = sum(
        float(realized_fractions[index])
        for index, (source, _temperature, _fraction) in enumerate(normalized)
        if source == "prior"
    )
    posterior_temperatures = [
        temperature
        for source, temperature, _fraction in normalized
        if source == "posterior"
    ]
    return DefensiveProposalSample(
        x=x,
        logproposal=logproposal,
        component_index=component_index,
        component_fractions=fractions,
        posterior_tempered_fraction=jnp.asarray(tempered_fraction, dtype=x.dtype),
        prior_fraction=jnp.asarray(prior_fraction, dtype=x.dtype),
        maximum_posterior_temperature=jnp.asarray(
            max(posterior_temperatures, default=1.0), dtype=x.dtype
        ),
    )


def defensive_mixture_log_prob(
    component_log_prob: jnp.ndarray,
    fractions: jnp.ndarray,
) -> jnp.ndarray:
    """Return ``logsumexp(log fraction_i + log density_i)`` exactly."""
    values = jnp.asarray(component_log_prob)
    weights = jnp.asarray(fractions, dtype=values.dtype)
    if values.ndim < 1 or values.shape[0] != weights.shape[0]:
        raise ValueError("component_log_prob and fractions must share component axis")
    weights = weights / jnp.sum(weights)
    shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return jax.scipy.special.logsumexp(
        jnp.log(weights).reshape(shape) + values,
        axis=0,
    )


def posterior_entropy_diagnostics(
    model: Any,
    features: jnp.ndarray,
    key: jax.Array,
    *,
    n_samples: int = 1,
) -> dict[str, jnp.ndarray]:
    """Estimate full conditional-flow entropy and its contraction terms."""
    draw = sample_posterior(model, key, features, max(1, int(n_samples)))
    residual = draw.residual_logdet
    full_entropy = -jnp.mean(draw.logq)
    base_entropy_mc = jnp.mean(-draw.logq - residual)
    return {
        "posterior_full_entropy_mc": full_entropy,
        "posterior_base_entropy": base_entropy_mc,
        "posterior_residual_logdet_mean": jnp.mean(residual),
        "posterior_residual_logdet_q05": jnp.quantile(residual, 0.05),
        "posterior_residual_logdet_q95": jnp.quantile(residual, 0.95),
    }


def _normalize_defensive_components(
    components: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[tuple[str, float, float], ...]:
    if not components:
        raise ValueError("defensive proposal requires at least one component")
    normalized = []
    for item in components:
        source = str(item.get("source", "posterior")).strip().lower()
        if source not in {"posterior", "prior"}:
            raise ValueError("defensive proposal source must be posterior or prior")
        temperature = float(item.get("temperature", 1.0))
        fraction = float(item.get("fraction", 0.0))
        if source == "prior":
            temperature = 1.0
        if temperature <= 0.0 or fraction <= 0.0:
            raise ValueError("proposal temperatures and fractions must be positive")
        normalized.append((source, temperature, fraction))
    total = sum(item[2] for item in normalized)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("defensive proposal fractions must have a finite sum")
    return tuple(
        (source, temperature, fraction / total)
        for source, temperature, fraction in normalized
    )


def _allocate_component_counts(count: int, fractions: np.ndarray) -> np.ndarray:
    """Allocate a fixed particle budget by largest remainder."""
    requested = np.asarray(fractions, dtype=float)
    raw = requested * int(count)
    allocated = np.floor(raw).astype(np.int64)
    missing = int(count) - int(np.sum(allocated))
    if missing:
        order = np.argsort(-(raw - allocated), kind="stable")
        allocated[order[:missing]] += 1
    if np.any(allocated <= 0):
        raise ValueError(
            "n_samples is too small to allocate at least one defensive draw "
            "to every configured component"
        )
    return allocated


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
        "autoregressive_rq_spline": "autoregressive_rq_spline",
        "autoregressive_rqspline": "autoregressive_rq_spline",
        "masked_autoregressive_spline": "autoregressive_rq_spline",
        "maf_rq_spline": "autoregressive_rq_spline",
    }
    if family not in aliases:
        raise ValueError(
            "Conditional flow family must be realnvp, rq_spline, or "
            "autoregressive_rq_spline"
        )
    return aliases[family]


def _normalize_context_encoder(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "base_moments": "base_moments",
        "compressed_moments": "base_moments",
        "current": "base_moments",
        "passband_set_transformer": "passband_set_transformer",
        "set_transformer": "passband_set_transformer",
        "band_set": "passband_set_transformer",
    }
    if normalized not in aliases:
        raise ValueError(
            "Conditional context encoder must be base_moments or "
            "passband_set_transformer"
        )
    return aliases[normalized]


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
