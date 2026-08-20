"""Gaussian MLP encoder ``q_psi(x | f_obs, err)``."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from .config import require_equinox

eqx = require_equinox()


class GaussianEncoder(eqx.Module):
    """Diagonal-Gaussian MLP encoder over unconstrained latent ``x``."""

    trunk: tuple
    mean_head: object
    log_std_head: object
    activation_name: str = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)
    initial_log_std: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        input_dim: int = 20,
        latent_dim: int = 16,
        hidden_sizes: tuple[int, ...] = (256, 256, 256),
        activation: str = "gelu",
        log_std_min: float = -6.0,
        log_std_max: float = 2.0,
        initial_log_std: float = -1.0,
    ) -> None:
        keys = jax.random.split(key, len(hidden_sizes) + 2)
        dims = (int(input_dim), *[int(size) for size in hidden_sizes])
        self.trunk = tuple(
            eqx.nn.Linear(dims[index], dims[index + 1], key=keys[index])
            for index in range(len(hidden_sizes))
        )
        last_dim = dims[-1]
        self.mean_head = eqx.nn.Linear(last_dim, int(latent_dim), key=keys[-2])
        self.log_std_head = eqx.nn.Linear(last_dim, int(latent_dim), key=keys[-1])
        self.activation_name = str(activation)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.initial_log_std = float(initial_log_std)

    def __call__(self, features):
        """Return ``mean`` and ``log_std`` for features shaped ``[N,input_dim]``."""
        features = jnp.asarray(features, dtype=jnp.float32)
        if features.ndim == 1:
            return self._single(features)
        mean, log_std = jax.vmap(self._single)(features)
        return mean, log_std

    def sample_and_log_prob(self, key, features, n_samples: int):
        """Sample ``x`` with the reparameterization trick and exact ``logq``."""
        mean, log_std = self(features)
        eps = jax.random.normal(
            key,
            (int(n_samples),) + mean.shape,
            dtype=mean.dtype,
        )
        std = jnp.exp(log_std)
        x = mean[None, ...] + std[None, ...] * eps
        logq = _diag_normal_log_prob(x, mean[None, ...], log_std[None, ...])
        return x, logq

    def _single(self, features):
        h = features
        activation = _activation(self.activation_name)
        for layer in self.trunk:
            h = activation(layer(h))
        mean = self.mean_head(h)
        raw_log_std = self.log_std_head(h) + self.initial_log_std
        log_std = jnp.clip(raw_log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class MixtureGaussianEncoder(eqx.Module):
    """MLP encoder for an exact mixture of diagonal Gaussian base densities."""

    trunk: tuple
    logits_head: object
    mean_head: object
    log_std_head: object
    activation_name: str = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    n_components: int = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)
    initial_log_std: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        input_dim: int = 20,
        latent_dim: int = 16,
        n_components: int = 2,
        hidden_sizes: tuple[int, ...] = (256, 256, 256),
        activation: str = "gelu",
        log_std_min: float = -6.0,
        log_std_max: float = 2.0,
        initial_log_std: float = -1.0,
    ) -> None:
        if int(n_components) < 2:
            raise ValueError("MixtureGaussianEncoder requires at least two components")
        keys = jax.random.split(key, len(hidden_sizes) + 3)
        dims = (int(input_dim), *[int(size) for size in hidden_sizes])
        self.trunk = tuple(
            eqx.nn.Linear(dims[index], dims[index + 1], key=keys[index])
            for index in range(len(hidden_sizes))
        )
        last_dim = dims[-1]
        self.logits_head = eqx.nn.Linear(last_dim, int(n_components), key=keys[-3])
        self.mean_head = eqx.nn.Linear(
            last_dim, int(n_components) * int(latent_dim), key=keys[-2]
        )
        self.log_std_head = eqx.nn.Linear(
            last_dim, int(n_components) * int(latent_dim), key=keys[-1]
        )
        self.activation_name = str(activation)
        self.latent_dim = int(latent_dim)
        self.n_components = int(n_components)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.initial_log_std = float(initial_log_std)

    def mixture_parameters(self, features):
        """Return logits, means and log-scales with shapes ``[..., M]``/``[..., M,D]``."""
        features = jnp.asarray(features, dtype=jnp.float32)
        if features.ndim == 1:
            return self._single_mixture(features)
        return jax.vmap(self._single_mixture)(features)

    def __call__(self, features):
        """Return moment-matched diagnostics compatible with Gaussian encoders."""
        logits, means, log_stds = self.mixture_parameters(features)
        weights = jax.nn.softmax(logits, axis=-1)
        mean = jnp.sum(weights[..., :, None] * means, axis=-2)
        second = jnp.sum(
            weights[..., :, None] * (jnp.exp(2.0 * log_stds) + means**2),
            axis=-2,
        )
        variance = jnp.maximum(second - mean**2, jnp.asarray(1.0e-12))
        return mean, 0.5 * jnp.log(variance)

    def _single_mixture(self, features):
        h = features
        activation = _activation(self.activation_name)
        for layer in self.trunk:
            h = activation(layer(h))
        logits = self.logits_head(h)
        means = self.mean_head(h).reshape(self.n_components, self.latent_dim)
        raw = self.log_std_head(h).reshape(self.n_components, self.latent_dim)
        log_stds = jnp.clip(
            raw + self.initial_log_std,
            self.log_std_min,
            self.log_std_max,
        )
        return logits, means, log_stds


class _SetAttentionBlock(eqx.Module):
    """Small pre-normalized multi-head self-attention block."""

    query: object
    key: object
    value: object
    attention_output: object
    feed_forward_in: object
    feed_forward_out: object
    norm_attention: object
    norm_feed_forward: object
    token_dim: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)

    def __init__(self, key, *, token_dim: int, n_heads: int) -> None:
        if int(token_dim) % int(n_heads):
            raise ValueError("set token_dim must be divisible by set_num_heads")
        keys = jax.random.split(key, 6)
        self.query = eqx.nn.Linear(token_dim, token_dim, key=keys[0])
        self.key = eqx.nn.Linear(token_dim, token_dim, key=keys[1])
        self.value = eqx.nn.Linear(token_dim, token_dim, key=keys[2])
        self.attention_output = eqx.nn.Linear(token_dim, token_dim, key=keys[3])
        self.feed_forward_in = eqx.nn.Linear(token_dim, 2 * token_dim, key=keys[4])
        self.feed_forward_out = eqx.nn.Linear(2 * token_dim, token_dim, key=keys[5])
        self.norm_attention = eqx.nn.LayerNorm((int(token_dim),))
        self.norm_feed_forward = eqx.nn.LayerNorm((int(token_dim),))
        self.token_dim = int(token_dim)
        self.n_heads = int(n_heads)

    def __call__(self, tokens):
        normalized = jax.vmap(self.norm_attention)(tokens)
        query = jax.vmap(self.query)(normalized)
        key = jax.vmap(self.key)(normalized)
        value = jax.vmap(self.value)(normalized)
        head_dim = self.token_dim // self.n_heads
        query = query.reshape(query.shape[0], self.n_heads, head_dim)
        key = key.reshape(key.shape[0], self.n_heads, head_dim)
        value = value.reshape(value.shape[0], self.n_heads, head_dim)
        scores = jnp.einsum("ihd,jhd->hij", query, key) / jnp.sqrt(
            jnp.asarray(head_dim, dtype=tokens.dtype)
        )
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("hij,jhd->ihd", weights, value).reshape(
            tokens.shape[0], self.token_dim
        )
        tokens = tokens + jax.vmap(self.attention_output)(attended)
        normalized = jax.vmap(self.norm_feed_forward)(tokens)
        hidden = jax.nn.gelu(jax.vmap(self.feed_forward_in)(normalized))
        return tokens + jax.vmap(self.feed_forward_out)(hidden)


class PassbandSetEncoder(eqx.Module):
    """Set-attention photometry encoder with a direct flow context.

    Each FENIKS band is represented by its normalized flux/error pair plus a
    learned passband identity. The context is consumed directly by conditional
    flow transforms; it is not compressed through posterior base moments.
    """

    input_projection: object
    band_embedding: jnp.ndarray
    blocks: tuple
    pool_query: jnp.ndarray
    context_projection: object
    mean_head: object
    log_std_head: object
    n_bands: int = eqx.field(static=True)
    token_dim: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    log_std_min: float = eqx.field(static=True)
    log_std_max: float = eqx.field(static=True)
    initial_log_std: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        input_dim: int,
        latent_dim: int,
        n_bands: int,
        token_dim: int,
        context_dim: int,
        n_heads: int,
        n_layers: int,
        log_std_min: float,
        log_std_max: float,
        initial_log_std: float,
    ) -> None:
        if int(input_dim) != 2 * int(n_bands):
            raise ValueError(
                "passband-set context requires flux then error features for every band"
            )
        if min(n_bands, token_dim, context_dim, n_heads, n_layers) <= 0:
            raise ValueError("passband-set dimensions must be positive")
        keys = jax.random.split(key, int(n_layers) + 6)
        self.input_projection = eqx.nn.Linear(2, token_dim, key=keys[0])
        scale = 1.0 / jnp.sqrt(jnp.asarray(token_dim, dtype=jnp.float32))
        self.band_embedding = scale * jax.random.normal(
            keys[1], (int(n_bands), int(token_dim)), dtype=jnp.float32
        )
        self.blocks = tuple(
            _SetAttentionBlock(keys[index + 2], token_dim=token_dim, n_heads=n_heads)
            for index in range(int(n_layers))
        )
        self.pool_query = scale * jax.random.normal(
            keys[-4], (int(token_dim),), dtype=jnp.float32
        )
        self.context_projection = eqx.nn.MLP(
            in_size=2 * int(token_dim),
            out_size=int(context_dim),
            width_size=max(int(token_dim), int(context_dim)),
            depth=2,
            activation=jax.nn.gelu,
            key=keys[-3],
        )
        self.mean_head = eqx.nn.Linear(context_dim, latent_dim, key=keys[-2])
        self.log_std_head = eqx.nn.Linear(context_dim, latent_dim, key=keys[-1])
        self.n_bands = int(n_bands)
        self.token_dim = int(token_dim)
        self.context_dim = int(context_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.initial_log_std = float(initial_log_std)

    def __call__(self, features):
        context = self.context(features)
        if context.ndim == 1:
            return self._moments(context)
        return jax.vmap(self._moments)(context)

    def context(self, features):
        features = jnp.asarray(features, dtype=jnp.float32)
        if features.shape[-1] != 2 * self.n_bands:
            raise ValueError(
                f"expected {2 * self.n_bands} passband features, "
                f"got {features.shape[-1]}"
            )
        if features.ndim == 1:
            return self._single_context(features)
        return jax.vmap(self._single_context)(features)

    def _single_context(self, features):
        token_input = jnp.stack(
            (features[: self.n_bands], features[self.n_bands :]), axis=-1
        )
        tokens = jax.vmap(self.input_projection)(token_input) + self.band_embedding
        for block in self.blocks:
            tokens = block(tokens)
        score = (
            tokens
            @ self.pool_query
            / jnp.sqrt(jnp.asarray(self.token_dim, dtype=tokens.dtype))
        )
        pooled = jnp.sum(jax.nn.softmax(score)[:, None] * tokens, axis=0)
        maximum = jnp.max(tokens, axis=0)
        return self.context_projection(jnp.concatenate((pooled, maximum)))

    def _moments(self, context):
        mean = self.mean_head(context)
        raw = self.log_std_head(context) + self.initial_log_std
        return mean, jnp.clip(raw, self.log_std_min, self.log_std_max)


def _diag_normal_log_prob(x, mean, log_std):
    var_term = ((x - mean) / jnp.exp(log_std)) ** 2
    return -0.5 * jnp.sum(
        var_term + 2.0 * log_std + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def _activation(name: str) -> Callable:
    normalized = name.strip().lower()
    if normalized == "gelu":
        return jax.nn.gelu
    if normalized == "tanh":
        return jnp.tanh
    if normalized == "relu":
        return jax.nn.relu
    raise ValueError(f"Unsupported encoder activation: {name}")
