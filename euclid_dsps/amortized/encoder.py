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
