"""Posterior families for amortized photometric inference."""

from __future__ import annotations

import hashlib
import json
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import require_equinox
from .encoder import (
    GaussianEncoder,
    MixtureGaussianEncoder,
    PassbandSetEncoder,
    ResidualPhotometryEncoder,
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


class PosteriorTransportValue(NamedTuple):
    """A value mapped through q together with ``log|dx / d epsilon|``."""

    value: jnp.ndarray
    logabsdet_dx_depsilon: jnp.ndarray


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

    def forward(self, value, context, *, scale_clamp=None):
        return self._transform(value, context, inverse=False, scale_clamp=scale_clamp)

    def inverse(self, value, context, *, scale_clamp=None):
        return self._transform(value, context, inverse=True, scale_clamp=scale_clamp)

    def _transform(self, value, context, *, inverse: bool, scale_clamp=None):
        value = jnp.asarray(value, dtype=jnp.float32)
        context = _broadcast_context(context, value.shape[:-1])
        mask = jnp.asarray(self.mask, dtype=value.dtype)
        active = 1.0 - mask
        masked = value * mask
        raw = _apply_net(self.net, jnp.concatenate([masked, context], axis=-1))
        if self.family == "realnvp":
            raw = raw.reshape(value.shape[:-1] + (self.latent_dim, 2))
            effective_clamp = jnp.asarray(
                self.scale_clamp if scale_clamp is None else scale_clamp,
                dtype=value.dtype,
            )
            log_scale = effective_clamp * jnp.tanh(raw[..., 0]) * active
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
        residual_trunk_width: int = 512,
        residual_blocks: int = 3,
        residual_representation_width: int = 256,
        residual_context_dim: int = 128,
        mean_init_scale: float = 1.0e-3,
        permutation: str = "indexed_roll",
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
        if context_encoder_type == "residual_photometry":
            if self.base_components != 1:
                raise ValueError(
                    "residual photometry conditional flows require base_components=1"
                )
            self.base = ResidualPhotometryEncoder(
                keys[0],
                input_dim=input_dim,
                latent_dim=latent_dim,
                trunk_width=int(residual_trunk_width),
                residual_blocks=int(residual_blocks),
                representation_width=int(residual_representation_width),
                context_dim=int(residual_context_dim),
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                initial_log_std=initial_log_std,
                mean_init_scale=float(mean_init_scale),
            )
            context_dim = int(residual_context_dim)
        elif context_encoder_type == "passband_set_transformer":
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
        permutation_mode = str(permutation).strip().lower()
        if permutation_mode in {"roll", "indexed_roll"}:
            shifts = tuple(index + 1 for index in range(int(n_layers)))
        elif permutation_mode in {"alternating_roll", "balanced_roll"}:
            shifts = tuple(
                1 if index % 2 == 0 else -1 for index in range(int(n_layers))
            )
        elif permutation_mode in {"none", "identity"}:
            shifts = (0,) * int(n_layers)
        else:
            raise ValueError(
                "conditional flow permutation must be indexed_roll, "
                "alternating_roll, or none"
            )
        permutations = tuple(
            jnp.roll(jnp.arange(int(latent_dim)), shift) for shift in shifts
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
        if self.context_encoder_type in {
            "passband_set_transformer",
            "residual_photometry",
        }:
            return self.base.context(features)
        if mean is None or log_std is None:
            mean, log_std = self.base(features)
        return jnp.concatenate((mean, log_std), axis=-1)

    def forward(self, value, context, *, scale_clamp=None):
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for layer, permutation in zip(self.layers, self.permutations, strict=True):
            value, delta = (
                layer.forward(value, context, scale_clamp=scale_clamp)
                if isinstance(layer, _ConditionalCoupling)
                else layer.forward(value, context)
            )
            value = jnp.take(value, permutation, axis=-1)
            logdet = logdet + delta
        return value, logdet

    def inverse(self, value, context, *, scale_clamp=None):
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        items = zip(
            reversed(self.layers),
            reversed(self.inverse_permutations),
            strict=True,
        )
        for layer, inverse_permutation in items:
            value = jnp.take(value, inverse_permutation, axis=-1)
            value, delta = (
                layer.inverse(value, context, scale_clamp=scale_clamp)
                if isinstance(layer, _ConditionalCoupling)
                else layer.inverse(value, context)
            )
            logdet = logdet + delta
        return value, logdet


def conditional_flow_topology(
    encoder: ConditionalFlowEncoder,
    *,
    coordinate_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Describe coupling coverage from the arrays actually held by a flow.

    Masks and permutations are serialized Equinox leaves.  This diagnostic
    deliberately inspects those leaves instead of inferring topology from the
    active YAML, which may differ when an historical checkpoint is restored.
    """
    if not isinstance(encoder, ConditionalFlowEncoder):
        raise TypeError("conditional-flow topology requires ConditionalFlowEncoder")
    latent_dim = int(encoder.latent_dim)
    if coordinate_names is None:
        names = tuple(f"x_{index:02d}" for index in range(latent_dim))
    else:
        names = tuple(str(name) for name in coordinate_names)
        if len(names) != latent_dim:
            raise ValueError(
                "coordinate_names length does not match conditional-flow latent_dim"
            )
    if len(encoder.layers) != len(encoder.permutations):
        raise ValueError("conditional-flow layers and permutations differ in length")

    origins = np.arange(latent_dim, dtype=np.int64)
    counts = np.zeros(latent_dim, dtype=np.int64)
    masks: list[list[bool] | None] = []
    permutations: list[list[int]] = []
    for layer_index, (layer, raw_permutation) in enumerate(
        zip(encoder.layers, encoder.permutations, strict=True)
    ):
        permutation = np.asarray(jax.device_get(raw_permutation), dtype=np.int64)
        if permutation.shape != (latent_dim,) or not np.array_equal(
            np.sort(permutation), np.arange(latent_dim)
        ):
            raise ValueError(
                f"invalid conditional-flow permutation at layer {layer_index}"
            )
        if isinstance(layer, _ConditionalCoupling):
            mask = np.asarray(jax.device_get(layer.mask), dtype=bool)
            if mask.shape != (latent_dim,):
                raise ValueError(
                    f"invalid conditional-flow mask at layer {layer_index}"
                )
            active = ~mask
            masks.append(mask.tolist())
        else:
            active = np.ones(latent_dim, dtype=bool)
            masks.append(None)
        counts[origins[active]] += 1
        origins = origins[permutation]
        permutations.append(permutation.tolist())

    core = {
        "schema_version": 1,
        "family": str(encoder.family),
        "output_space": str(encoder.output_space),
        "latent_dim": latent_dim,
        "layers": len(encoder.layers),
        "coordinate_names": list(names),
        "transform_counts": counts.tolist(),
        "untransformed_coordinates": [
            names[index] for index, count in enumerate(counts) if count == 0
        ],
        "masks": masks,
        "permutations": permutations,
    }
    fingerprint = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        **core,
        "minimum_transform_count": int(np.min(counts)) if counts.size else 0,
        "maximum_transform_count": int(np.max(counts)) if counts.size else 0,
        "all_coordinates_transformed": bool(np.all(counts > 0)),
        "fingerprint_sha256": fingerprint,
    }


def transfer_residual_photometry_trunk(
    source: ConditionalFlowEncoder,
    target: ConditionalFlowEncoder,
) -> ConditionalFlowEncoder:
    """Copy only the compatible photometry representation into a new flow.

    The posterior base heads, context head, coupling nets, masks and
    permutations intentionally remain those of ``target``.  This prevents an
    historical flow topology from leaking into a rebuilt checkpoint.
    """
    if not isinstance(source, ConditionalFlowEncoder) or not isinstance(
        target, ConditionalFlowEncoder
    ):
        raise TypeError("trunk transfer requires two conditional-flow encoders")
    if not isinstance(source.base, ResidualPhotometryEncoder) or not isinstance(
        target.base, ResidualPhotometryEncoder
    ):
        raise TypeError("trunk transfer requires residual photometry encoders")
    source_shape = (
        source.base.input_dim,
        source.base.trunk_width,
        len(source.base.blocks),
        source.base.representation_width,
    )
    target_shape = (
        target.base.input_dim,
        target.base.trunk_width,
        len(target.base.blocks),
        target.base.representation_width,
    )
    if source_shape != target_shape:
        raise ValueError(
            "residual photometry trunk shapes differ: "
            f"source={source_shape}, target={target_shape}"
        )
    return eqx.tree_at(
        lambda encoder: (
            encoder.base.input_projection,
            encoder.base.blocks,
            encoder.base.representation_projection,
        ),
        target,
        (
            source.base.input_projection,
            source.base.blocks,
            source.base.representation_projection,
        ),
    )


def sample_posterior(
    model,
    key,
    features,
    n_samples: int,
    *,
    sample_strategy: str = "random",
    base_temperature: float = 1.0,
    log_std_floor=None,
    flow_scale_clamp=None,
) -> PosteriorSample:
    """Sample any configured posterior and return exact densities in x-space."""
    state = posterior_encoder_state(
        model,
        features,
        log_std_floor=log_std_floor,
    )
    return sample_posterior_from_state(
        model,
        key,
        state,
        n_samples,
        sample_strategy=sample_strategy,
        base_temperature=base_temperature,
        flow_scale_clamp=flow_scale_clamp,
    )


def posterior_encoder_state(
    model,
    features,
    *,
    log_std_floor=None,
) -> PosteriorEncoderState:
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
        if log_std_floor is not None:
            log_std = jnp.maximum(
                log_std,
                jnp.asarray(log_std_floor, dtype=log_std.dtype),
            )
        return PosteriorEncoderState(
            mean,
            log_std,
            logits,
            component_means,
            component_log_stds,
            model.encoder.flow_context(features, mean, log_std),
        )
    if isinstance(model.encoder, ConditionalFlowEncoder) and isinstance(
        model.encoder.base, ResidualPhotometryEncoder
    ):
        mean, log_std, context = model.encoder.base.encode(features)
        if log_std_floor is not None:
            log_std = jnp.maximum(
                log_std,
                jnp.asarray(log_std_floor, dtype=log_std.dtype),
            )
        return PosteriorEncoderState(mean, log_std, None, None, None, context)
    mean, log_std = model.encoder(features)
    if log_std_floor is not None:
        log_std = jnp.maximum(
            log_std,
            jnp.asarray(log_std_floor, dtype=log_std.dtype),
        )
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
    flow_scale_clamp=None,
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
        x, residual_logdet = model.encoder.forward(
            base,
            context,
            scale_clamp=flow_scale_clamp,
        )
        logq = logq_base - residual_logdet
        logprior = model.prior.log_prob(x)
    else:
        context = state.flow_context
        u = base
        if isinstance(model.encoder, ConditionalFlowEncoder):
            u, residual_logdet = model.encoder.forward(
                u,
                context,
                scale_clamp=flow_scale_clamp,
            )
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
    log_std_floor=None,
    flow_scale_clamp=None,
) -> jnp.ndarray:
    """Evaluate exact conditional posterior density for supervised NPE."""
    mean, log_std = model.encoder(features)
    if log_std_floor is not None:
        log_std = jnp.maximum(
            log_std,
            jnp.asarray(log_std_floor, dtype=log_std.dtype),
        )
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
        base, inverse_logdet = model.encoder.inverse(
            x,
            context,
            scale_clamp=flow_scale_clamp,
        )
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
        base, residual_inverse_logdet = model.encoder.inverse(
            u,
            context,
            scale_clamp=flow_scale_clamp,
        )
    base_log_prob = (
        _mixture_base_log_prob_from_encoder(model.encoder, features, base, temperature)
        if _is_mixture_encoder(model.encoder)
        else _diag_normal_log_prob(base, mean, proposal_log_std)
    )
    return base_log_prob + residual_inverse_logdet + prior_inverse_logdet


def posterior_standard_base_to_x(
    model,
    features: jnp.ndarray,
    epsilon: jnp.ndarray,
) -> PosteriorTransportValue:
    """Map standard-normal coordinates through a single-base conditional q.

    This transform is the preconditioner used by adaptive bridge SMC. It is
    intentionally restricted to the production contract: a single diagonal
    Gaussian base followed by a conditional flow in ``latent_x`` space.
    """
    if _is_mixture_encoder(model.encoder):
        raise ValueError("posterior transport requires base_components=1")
    if not isinstance(model.encoder, ConditionalFlowEncoder):
        raise ValueError("posterior transport requires ConditionalFlowEncoder")
    if model.encoder.output_space != "latent_x":
        raise ValueError("posterior transport requires flow_output_space=latent_x")
    state = posterior_encoder_state(model, features)
    epsilon = jnp.asarray(epsilon)
    mean = _broadcast_context(state.mean, epsilon.shape[:-1])
    log_std = _broadcast_context(state.log_std, epsilon.shape[:-1])
    base = mean + jnp.exp(log_std) * epsilon
    x, flow_logdet = model.encoder.forward(base, state.flow_context)
    return PosteriorTransportValue(
        value=x,
        logabsdet_dx_depsilon=jnp.sum(log_std, axis=-1) + flow_logdet,
    )


def posterior_x_to_standard_base(
    model,
    features: jnp.ndarray,
    x: jnp.ndarray,
) -> PosteriorTransportValue:
    """Invert the production conditional q transport exactly."""
    if _is_mixture_encoder(model.encoder):
        raise ValueError("posterior transport requires base_components=1")
    if not isinstance(model.encoder, ConditionalFlowEncoder):
        raise ValueError("posterior transport requires ConditionalFlowEncoder")
    if model.encoder.output_space != "latent_x":
        raise ValueError("posterior transport requires flow_output_space=latent_x")
    state = posterior_encoder_state(model, features)
    x = jnp.asarray(x)
    mean = _broadcast_context(state.mean, x.shape[:-1])
    log_std = _broadcast_context(state.log_std, x.shape[:-1])
    base, inverse_flow_logdet = model.encoder.inverse(x, state.flow_context)
    epsilon = (base - mean) * jnp.exp(-log_std)
    return PosteriorTransportValue(
        value=epsilon,
        logabsdet_dx_depsilon=jnp.sum(log_std, axis=-1) - inverse_flow_logdet,
    )


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
    *,
    antithetic: bool = False,
    log_std_floor=None,
    flow_scale_clamp=None,
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
            strategy = (
                "antithetic"
                if antithetic and int(component_count) % 2 == 0
                else "random"
            )
            draw = sample_posterior(
                model,
                component_key,
                features,
                int(component_count),
                sample_strategy=strategy,
                base_temperature=temperature,
                log_std_floor=log_std_floor,
                flow_scale_clamp=flow_scale_clamp,
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
                log_std_floor=log_std_floor,
                flow_scale_clamp=flow_scale_clamp,
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
    log_std_floor=None,
    flow_scale_clamp=None,
) -> dict[str, jnp.ndarray]:
    """Estimate full conditional-flow entropy and its contraction terms."""
    draw = sample_posterior(
        model,
        key,
        features,
        max(1, int(n_samples)),
        log_std_floor=log_std_floor,
        flow_scale_clamp=flow_scale_clamp,
    )
    residual = draw.residual_logdet
    full_entropy = -jnp.mean(draw.logq)
    base_entropy_mc = jnp.mean(-draw.logq - residual)
    log_std = jnp.asarray(draw.base_log_std)
    return {
        "posterior_full_entropy_mc": full_entropy,
        "posterior_base_entropy": base_entropy_mc,
        "posterior_residual_logdet_mean": jnp.mean(residual),
        "posterior_residual_logdet_q05": jnp.quantile(residual, 0.05),
        "posterior_residual_logdet_q95": jnp.quantile(residual, 0.95),
        "posterior_log_std_mean": jnp.mean(log_std),
        "posterior_log_std_min": jnp.min(log_std),
        "posterior_log_std_max": jnp.max(log_std),
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
        "residual_photometry": "residual_photometry",
        "residual_mlp": "residual_photometry",
        "sc_asmc_em": "residual_photometry",
    }
    if normalized not in aliases:
        raise ValueError(
            "Conditional context encoder must be base_moments, "
            "passband_set_transformer, or residual_photometry"
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
