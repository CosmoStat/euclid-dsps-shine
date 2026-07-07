"""RealNVP prior ``p_beta(x)`` for unconstrained latent space."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .config import require_equinox

eqx = require_equinox()


class _CouplingLayer(eqx.Module):
    mask: jnp.ndarray
    scale_net: object
    shift_net: object
    scale_clamp: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        latent_dim: int,
        hidden_size: int,
        mask,
        scale_clamp: float,
        init: str = "default",
        init_scale: float = 1.0,
    ):
        k_scale, k_shift = jax.random.split(key)
        self.mask = jnp.asarray(mask, dtype=jnp.float32)
        self.scale_net = eqx.nn.MLP(
            in_size=int(latent_dim),
            out_size=int(latent_dim),
            width_size=int(hidden_size),
            depth=2,
            activation=jax.nn.gelu,
            key=k_scale,
        )
        self.shift_net = eqx.nn.MLP(
            in_size=int(latent_dim),
            out_size=int(latent_dim),
            width_size=int(hidden_size),
            depth=2,
            activation=jax.nn.gelu,
            key=k_shift,
        )
        if str(init).lower() == "identity":
            self.scale_net = _scale_last_linear(self.scale_net, float(init_scale))
            self.shift_net = _scale_last_linear(self.shift_net, float(init_scale))
        self.scale_clamp = float(scale_clamp)

    def forward(self, x):
        x_masked = x * self.mask
        log_scale, shift = self._scale_shift(x_masked)
        y = x_masked + (1.0 - self.mask) * (x * jnp.exp(log_scale) + shift)
        logdet = jnp.sum(log_scale, axis=-1)
        return y, logdet

    def inverse(self, y):
        y_masked = y * self.mask
        log_scale, shift = self._scale_shift(y_masked)
        x = y_masked + (1.0 - self.mask) * ((y - shift) * jnp.exp(-log_scale))
        logdet = -jnp.sum(log_scale, axis=-1)
        return x, logdet

    def _scale_shift(self, masked):
        scale = _apply_net(self.scale_net, masked)
        shift = _apply_net(self.shift_net, masked)
        log_scale = self.scale_clamp * jnp.tanh(scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        return log_scale, shift


class RealNVPPrior(eqx.Module):
    """Fast exact-density RealNVP prior over unconstrained latent ``x``."""

    layers: tuple
    latent_dim: int = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        latent_dim: int = 16,
        n_layers: int = 8,
        hidden_size: int = 128,
        scale_clamp: float = 0.05,
        init: str = "default",
        init_scale: float = 1.0,
    ) -> None:
        keys = jax.random.split(key, int(n_layers))
        masks = _alternating_masks(int(latent_dim), int(n_layers))
        init = str(init).lower()
        if init not in {"default", "identity"}:
            raise ValueError("RealNVPPrior init must be 'default' or 'identity'")
        self.layers = tuple(
            _CouplingLayer(
                keys[index],
                latent_dim=int(latent_dim),
                hidden_size=int(hidden_size),
                mask=masks[index],
                scale_clamp=float(scale_clamp),
                init=init,
                init_scale=float(init_scale),
            )
            for index in range(int(n_layers))
        )
        self.latent_dim = int(latent_dim)

    def forward(self, u):
        """Map base ``u`` to latent ``x`` and return forward log-det."""
        value = jnp.asarray(u, dtype=jnp.float32)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for layer in self.layers:
            value, layer_logdet = layer.forward(value)
            logdet = logdet + layer_logdet
        return value, logdet

    def inverse(self, x):
        """Map latent ``x`` to base ``u`` and return inverse log-det."""
        value = jnp.asarray(x, dtype=jnp.float32)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for layer in reversed(self.layers):
            value, layer_logdet = layer.inverse(value)
            logdet = logdet + layer_logdet
        return value, logdet

    def log_prob(self, x):
        """Exact pointwise ``log p_beta(x)``."""
        u, logdet = self.inverse(x)
        base = -0.5 * jnp.sum(u**2 + jnp.log(2.0 * jnp.pi), axis=-1)
        return base + logdet

    def sample(self, key, shape=()):
        """Sample latent ``x`` from the RealNVP prior."""
        if isinstance(shape, int):
            shape = (shape,)
        u = jax.random.normal(key, tuple(shape) + (self.latent_dim,), dtype=jnp.float32)
        x, _logdet = self.forward(u)
        return x


class StandardNormalPrior(eqx.Module):
    """Non-trainable standard normal prior over unconstrained latent ``x``."""

    latent_dim: int = eqx.field(static=True)

    def __init__(self, *, latent_dim: int) -> None:
        self.latent_dim = int(latent_dim)

    def log_prob(self, x):
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dim {self.latent_dim}, got {x.shape[-1]}"
            )
        return -0.5 * jnp.sum(x**2 + jnp.log(2.0 * jnp.pi), axis=-1)

    def sample(self, key, shape=()):
        if isinstance(shape, int):
            shape = (shape,)
        return jax.random.normal(
            key,
            tuple(shape) + (self.latent_dim,),
            dtype=jnp.float32,
        )


def _apply_net(net, value):
    if value.ndim == 1:
        return net(value)
    leading = value.shape[:-1]
    flat = value.reshape((-1, value.shape[-1]))
    out = jax.vmap(net)(flat)
    return out.reshape(leading + (out.shape[-1],))


def _scale_last_linear(net, scale: float):
    """Scale the final MLP affine layer for near-identity flow initialization."""
    final = net.layers[-1]
    scaled_weight = jnp.asarray(final.weight) * jnp.asarray(scale, dtype=final.weight.dtype)
    if final.bias is None:
        return eqx.tree_at(lambda item: item.layers[-1].weight, net, scaled_weight)
    scaled_bias = jnp.asarray(final.bias) * jnp.asarray(scale, dtype=final.bias.dtype)
    return eqx.tree_at(
        lambda item: (item.layers[-1].weight, item.layers[-1].bias),
        net,
        (scaled_weight, scaled_bias),
    )


def _alternating_masks(latent_dim: int, n_layers: int) -> tuple[jnp.ndarray, ...]:
    base = (jnp.arange(latent_dim) % 2).astype(jnp.float32)
    return tuple(base if index % 2 == 0 else 1.0 - base for index in range(n_layers))
