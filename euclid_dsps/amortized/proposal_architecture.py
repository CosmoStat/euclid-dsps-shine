"""Experimental exact-density proposals for posterior architecture sweeps.

These proposals are deliberately separate from the production encoder. They
keep the population prior fixed and expose the conditioning representation as
the only architectural variable in controlled SMC distillation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .config import require_amortized_dependencies
from .encoder import _diag_normal_log_prob
from .posterior import _ConditionalCoupling

eqx, optax = require_amortized_dependencies()


class ResidualContextEncoder(eqx.Module):
    """Dedicated residual MLP context, independent of Gaussian base moments."""

    input_layer: object
    blocks: tuple
    output_layer: object
    input_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        input_dim: int,
        context_dim: int,
        hidden_size: int,
        depth: int = 3,
    ) -> None:
        if min(input_dim, context_dim, hidden_size, depth) <= 0:
            raise ValueError("context dimensions and depth must be positive")
        keys = jax.random.split(key, int(depth) + 2)
        self.input_layer = eqx.nn.Linear(input_dim, hidden_size, key=keys[0])
        self.blocks = tuple(
            eqx.nn.Linear(hidden_size, hidden_size, key=keys[index + 1])
            for index in range(int(depth))
        )
        self.output_layer = eqx.nn.Linear(hidden_size, context_dim, key=keys[-1])
        self.input_dim = int(input_dim)
        self.context_dim = int(context_dim)

    def __call__(self, observations):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        if observations.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected context input {self.input_dim}, got {observations.shape[-1]}"
            )
        if observations.ndim == 1:
            return self._single(observations)
        return jax.vmap(self._single)(observations)

    def _single(self, observations):
        value = jax.nn.gelu(self.input_layer(observations))
        for layer in self.blocks:
            value = value + jax.nn.gelu(layer(value)) / jnp.sqrt(
                jnp.asarray(2.0, dtype=value.dtype)
            )
        return self.output_layer(value)


class BandTokenContextEncoder(eqx.Module):
    """Permutation-aware per-band encoder with masked attention pooling."""

    token_layer: object
    token_block: object
    attention: object
    output: object
    n_bands: int = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        n_bands: int,
        token_dim: int,
        context_dim: int,
    ) -> None:
        if min(n_bands, token_dim, context_dim) <= 0:
            raise ValueError("token context dimensions must be positive")
        keys = jax.random.split(key, 4)
        # Each token contains transformed flux, log error, validity and a
        # deterministic normalized band coordinate.
        self.token_layer = eqx.nn.Linear(4, token_dim, key=keys[0])
        self.token_block = eqx.nn.Linear(token_dim, token_dim, key=keys[1])
        self.attention = eqx.nn.Linear(token_dim, 1, key=keys[2])
        self.output = eqx.nn.MLP(
            in_size=2 * token_dim,
            out_size=context_dim,
            width_size=max(token_dim, context_dim),
            depth=2,
            activation=jax.nn.gelu,
            key=keys[3],
        )
        self.n_bands = int(n_bands)
        self.token_dim = int(token_dim)
        self.context_dim = int(context_dim)

    def __call__(self, observations):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        expected = 3 * self.n_bands
        if observations.shape[-1] != expected:
            raise ValueError(
                f"expected {expected} band-token inputs, got {observations.shape[-1]}"
            )
        if observations.ndim == 1:
            return self._single(observations)
        return jax.vmap(self._single)(observations)

    def _single(self, observations):
        flux = observations[: self.n_bands]
        error = observations[self.n_bands : 2 * self.n_bands]
        valid = observations[2 * self.n_bands :] > 0.5
        coordinate = jnp.linspace(-1.0, 1.0, self.n_bands, dtype=flux.dtype)
        token_input = jnp.stack(
            (flux, error, valid.astype(flux.dtype), coordinate), axis=-1
        )
        token = jax.vmap(self.token_layer)(token_input)
        token = jax.nn.gelu(token)
        token = token + jax.nn.gelu(jax.vmap(self.token_block)(token))
        score = jax.vmap(self.attention)(token)[..., 0]
        score = jnp.where(valid, score, jnp.asarray(-1.0e9, dtype=score.dtype))
        weight = jax.nn.softmax(score)
        pooled = jnp.sum(weight[:, None] * token, axis=0)
        valid_token = jnp.where(valid[:, None], token, -jnp.inf)
        maximum = jnp.max(valid_token, axis=0)
        maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
        return self.output(jnp.concatenate((pooled, maximum)))


class FreeResidualContextAdapter(eqx.Module):
    """Zero-initialized per-object context corrections for an oracle test."""

    delta: jnp.ndarray
    one_hot_offset: int = eqx.field(static=True)
    n_objects: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    def __init__(self, *, one_hot_offset: int, n_objects: int, context_dim: int):
        if min(n_objects, context_dim) <= 0 or int(one_hot_offset) < 0:
            raise ValueError("invalid free-adapter dimensions")
        self.delta = jnp.zeros((int(n_objects), int(context_dim)), dtype=jnp.float32)
        self.one_hot_offset = int(one_hot_offset)
        self.n_objects = int(n_objects)
        self.context_dim = int(context_dim)

    def __call__(self, observations):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        one_hot = observations[
            ..., self.one_hot_offset : self.one_hot_offset + self.n_objects
        ]
        return one_hot @ self.delta


class WarmStartResidualProposal(eqx.Module):
    """Current conditional flow plus a zero-initialized context correction."""

    source_encoder: object
    adapter: object
    feature_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(self, source_encoder, adapter, *, feature_dim: int) -> None:
        if source_encoder.output_space != "latent_x":
            raise ValueError("warm-start adapters require latent_x flow output")
        if int(source_encoder.base_components) != 1:
            raise ValueError("warm-start adapters currently require a Gaussian base")
        self.source_encoder = source_encoder
        self.adapter = adapter
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(source_encoder.latent_dim)
        self.context_dim = 2 * self.latent_dim

    def base_parameters(self, observations):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        features = observations[..., : self.feature_dim]
        mean, log_std = self.source_encoder.base(features)
        base_context = jnp.concatenate((mean, log_std), axis=-1)
        delta = self.adapter(observations)
        if delta.shape != base_context.shape:
            raise ValueError(
                f"adapter context shape {delta.shape} != {base_context.shape}"
            )
        return base_context + delta, mean, log_std

    def forward(self, value, context):
        return self.source_encoder.forward(value, context)

    def inverse(self, value, context):
        return self.source_encoder.inverse(value, context)


def zero_residual_mlp_adapter(
    key,
    *,
    input_dim: int,
    context_dim: int,
    hidden_size: int = 128,
    depth: int = 3,
):
    """Build an MLP adapter whose initial output is exactly zero."""
    adapter = ResidualContextEncoder(
        key,
        input_dim=input_dim,
        context_dim=context_dim,
        hidden_size=hidden_size,
        depth=depth,
    )
    return eqx.tree_at(
        lambda item: (item.output_layer.weight, item.output_layer.bias),
        adapter,
        (
            jnp.zeros_like(adapter.output_layer.weight),
            jnp.zeros_like(adapter.output_layer.bias),
        ),
    )


def zero_band_token_adapter(
    key,
    *,
    n_bands: int,
    token_dim: int,
    context_dim: int,
):
    """Build a band-token adapter whose initial output is exactly zero."""
    adapter = BandTokenContextEncoder(
        key,
        n_bands=n_bands,
        token_dim=token_dim,
        context_dim=context_dim,
    )
    last = adapter.output.layers[-1]
    return eqx.tree_at(
        lambda item: (
            item.output.layers[-1].weight,
            item.output.layers[-1].bias,
        ),
        adapter,
        (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)),
    )


class ContextualFlowProposal(eqx.Module):
    """Diagonal Gaussian base plus exact conditional coupling transforms."""

    context_encoder: object
    mean_head: object
    log_std_head: object
    layers: tuple
    permutations: tuple
    inverse_permutations: tuple
    latent_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)
    initial_log_std: float = eqx.field(static=True)
    family: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        context_encoder,
        context_dim: int,
        latent_dim: int,
        family: str,
        n_layers: int,
        hidden_size: int,
        n_bins: int = 8,
        log_std_min: float = -6.0,
        log_std_max: float = 2.0,
        initial_log_std: float = -1.0,
        init_scale: float = 0.0,
    ) -> None:
        if int(n_layers) <= 0:
            raise ValueError("n_layers must be positive")
        keys = jax.random.split(key, int(n_layers) + 2)
        self.context_encoder = context_encoder
        self.mean_head = eqx.nn.Linear(context_dim, latent_dim, key=keys[0])
        self.log_std_head = eqx.nn.Linear(context_dim, latent_dim, key=keys[1])
        masks = tuple(
            jnp.arange(int(latent_dim)) % 2 == index % 2
            for index in range(int(n_layers))
        )
        self.layers = tuple(
            _ConditionalCoupling(
                keys[index + 2],
                latent_dim=latent_dim,
                context_dim=context_dim,
                hidden_size=hidden_size,
                mask=masks[index],
                family=family,
                n_bins=n_bins,
                scale_clamp=0.5,
                shift_clamp=3.0,
                tail_bound=8.0,
                min_bin_width=1.0e-3,
                min_bin_height=1.0e-3,
                min_derivative=1.0e-3,
                init_scale=init_scale,
            )
            for index in range(int(n_layers))
        )
        self.permutations = tuple(
            jnp.roll(jnp.arange(int(latent_dim)), index + 1)
            for index in range(int(n_layers))
        )
        self.inverse_permutations = tuple(
            jnp.argsort(value) for value in self.permutations
        )
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.initial_log_std = float(initial_log_std)
        self.family = str(family)

    def context(self, observations):
        return self.context_encoder(observations)

    def base_parameters(self, observations):
        context = self.context(observations)
        if context.ndim == 1:
            mean = self.mean_head(context)
            raw = self.log_std_head(context)
        else:
            mean = jax.vmap(self.mean_head)(context)
            raw = jax.vmap(self.log_std_head)(context)
        log_std = jnp.clip(
            raw + self.initial_log_std, self.log_std_min, self.log_std_max
        )
        return context, mean, log_std

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
            reversed(self.layers), reversed(self.inverse_permutations), strict=True
        )
        for layer, inverse_permutation in items:
            value = jnp.take(value, inverse_permutation, axis=-1)
            value, delta = layer.inverse(value, context)
            logdet = logdet + delta
        return value, logdet


@dataclass(frozen=True)
class ProposalFitResult:
    proposal: object
    history: tuple[dict[str, float | int], ...]
    initial_train_nll: float
    initial_validation_nll: float
    best_train_nll: float
    best_validation_nll: float
    best_epoch: int


def contextual_log_prob(proposal, observations, values):
    """Evaluate exact normalized ``log q(x | observation)``."""
    context, mean, log_std = proposal.base_parameters(observations)
    base, inverse_logdet = proposal.inverse(values, context)
    return _diag_normal_log_prob(base, mean, log_std) + inverse_logdet


def sample_contextual_proposal(proposal, key, observations, n_samples: int):
    """Sample a contextual proposal and return samples and exact log density."""
    context, mean, log_std = proposal.base_parameters(observations)
    epsilon = jax.random.normal(key, (int(n_samples),) + mean.shape, dtype=mean.dtype)
    base = mean[None, ...] + jnp.exp(log_std)[None, ...] * epsilon
    base_logq = _diag_normal_log_prob(base, mean, log_std)
    values, logdet = proposal.forward(base, context)
    return values, base_logq - logdet


def fit_contextual_proposal(
    proposal,
    *,
    observations,
    train_particles,
    train_weights,
    validation_particles,
    validation_weights,
    train_indices,
    validation_indices,
    epochs: int,
    object_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    progress_label: str,
    freeze_source_encoder: bool = False,
) -> ProposalFitResult:
    """Fit inclusive KL on bank A and select on independent bank B."""
    observations = jnp.asarray(observations, dtype=jnp.float32)
    train_particles = jnp.asarray(train_particles, dtype=jnp.float32)
    validation_particles = jnp.asarray(validation_particles, dtype=jnp.float32)
    train_weights = _normalized_weights(train_weights)
    validation_weights = _normalized_weights(validation_weights)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    _validate_arrays(
        observations,
        train_particles,
        train_weights,
        validation_particles,
        validation_weights,
        train_indices,
        validation_indices,
    )

    @eqx.filter_jit
    def loss_and_grad(value, batch_observations, particles, weights):
        def loss_fn(item):
            logq = contextual_log_prob(item, batch_observations, particles)
            return -jnp.mean(jnp.sum(weights * logq, axis=0))

        return eqx.filter_value_and_grad(loss_fn)(value)

    @eqx.filter_jit
    def evaluate(value, particles, weights, indices):
        logq = contextual_log_prob(value, observations[indices], particles[:, indices])
        return -jnp.mean(jnp.sum(weights[:, indices] * logq, axis=0))

    initial_train = float(
        evaluate(proposal, train_particles, train_weights, train_indices)
    )
    initial_validation = float(
        evaluate(proposal, validation_particles, validation_weights, validation_indices)
    )
    optimizer = optax.adamw(
        learning_rate=float(learning_rate), weight_decay=float(weight_decay)
    )
    opt_state = optimizer.init(
        _trainable_tree(proposal, freeze_source_encoder=bool(freeze_source_encoder))
    )
    best = proposal
    best_validation = initial_validation
    best_train = initial_train
    best_epoch = 0
    history = []
    rng = np.random.default_rng(int(seed))
    for epoch in range(1, int(epochs) + 1):
        losses = []
        order = rng.permutation(train_indices)
        for start in range(0, len(order), int(object_batch_size)):
            index = order[start : start + int(object_batch_size)]
            loss, grads = loss_and_grad(
                proposal,
                observations[index],
                train_particles[:, index],
                train_weights[:, index],
            )
            grads = _trainable_tree(
                grads, freeze_source_encoder=bool(freeze_source_encoder)
            )
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                _trainable_tree(
                    proposal, freeze_source_encoder=bool(freeze_source_encoder)
                ),
            )
            proposal = eqx.apply_updates(proposal, updates)
            losses.append(float(loss))
        train_nll = float(
            evaluate(proposal, train_particles, train_weights, train_indices)
        )
        validation_nll = float(
            evaluate(
                proposal,
                validation_particles,
                validation_weights,
                validation_indices,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "minibatch_train_weighted_nll": float(np.mean(losses)),
                "train_weighted_nll": train_nll,
                "validation_weighted_nll": validation_nll,
            }
        )
        print(
            "[proposal-architecture] "
            f"candidate={progress_label} epoch={epoch}/{epochs} "
            f"train_nll={train_nll:.6g} validation_nll={validation_nll:.6g}",
            flush=True,
        )
        if np.isfinite(validation_nll) and validation_nll < best_validation:
            best = proposal
            best_validation = validation_nll
            best_train = train_nll
            best_epoch = epoch
    return ProposalFitResult(
        proposal=best,
        history=tuple(history),
        initial_train_nll=initial_train,
        initial_validation_nll=initial_validation,
        best_train_nll=best_train,
        best_validation_nll=best_validation,
        best_epoch=best_epoch,
    )


def make_direct_observations(features, mask):
    """Concatenate frozen photometry features and the explicit validity mask."""
    features = jnp.asarray(features, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.concatenate((features, mask), axis=-1)


def make_band_token_observations(features, mask):
    """Return ``[flux features, error features, mask]`` for token contexts."""
    features = jnp.asarray(features, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    if features.shape[-1] != 2 * mask.shape[-1]:
        raise ValueError("band token features must contain flux and error per band")
    return jnp.concatenate((features, mask), axis=-1)


def _normalized_weights(values):
    values = jnp.asarray(values, dtype=jnp.float32)
    return values / jnp.sum(values, axis=0, keepdims=True)


def _trainable_tree(value, *, freeze_source_encoder):
    filtered = eqx.filter(value, eqx.is_inexact_array)
    if freeze_source_encoder:
        if not isinstance(value, WarmStartResidualProposal):
            raise TypeError("freeze_source_encoder requires WarmStartResidualProposal")
        filtered = eqx.tree_at(
            lambda item: item.source_encoder,
            filtered,
            None,
            is_leaf=lambda item: item is None,
        )
    return filtered


def _validate_arrays(
    observations,
    train_particles,
    train_weights,
    validation_particles,
    validation_weights,
    train_indices,
    validation_indices,
):
    n_objects = observations.shape[0]
    if train_particles.ndim != 3 or validation_particles.ndim != 3:
        raise ValueError("particle arrays must have shape [K,N,D]")
    if train_particles.shape[1:] != validation_particles.shape[1:]:
        raise ValueError("replicate particle banks have incompatible shapes")
    if train_particles.shape[1] != n_objects:
        raise ValueError("observation and particle object axes differ")
    if train_weights.shape != train_particles.shape[:2]:
        raise ValueError("training weight shape does not match particles")
    if validation_weights.shape != validation_particles.shape[:2]:
        raise ValueError("validation weight shape does not match particles")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("training and validation object sets must be non-empty")
    all_indices = np.concatenate((train_indices, validation_indices))
    if np.min(all_indices) < 0 or np.max(all_indices) >= n_objects:
        raise ValueError("object split index outside cohort")
