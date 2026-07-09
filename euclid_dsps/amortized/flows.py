"""RealNVP prior ``p_beta(x)`` for unconstrained latent space."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .config import require_equinox

eqx = require_equinox()


class _CouplingLayer(eqx.Module):
    mask: jnp.ndarray
    scale_net: object
    shift_net: object
    scale_clamp: float = eqx.field(static=True)
    shift_clamp: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        latent_dim: int,
        hidden_size: int,
        mask,
        scale_clamp: float,
        shift_clamp: float,
        init: str = "default",
        init_scale: float = 1.0,
    ):
        k_scale, k_shift = jax.random.split(key)
        # Keep masks boolean so Equinox/Optax filters do not train them as
        # inexact arrays. Coupling masks are part of the flow topology.
        self.mask = jnp.asarray(mask, dtype=jnp.bool_)
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
        self.shift_clamp = float(shift_clamp)

    def forward(self, x):
        mask = _mask_as_float(self.mask, x)
        active = 1.0 - mask
        x_masked = x * mask
        log_scale, shift = self._scale_shift(x_masked)
        y = x_masked + active * (x * jnp.exp(log_scale) + shift)
        logdet = jnp.sum(log_scale, axis=-1)
        return y, logdet

    def inverse(self, y):
        mask = _mask_as_float(self.mask, y)
        active = 1.0 - mask
        y_masked = y * mask
        log_scale, shift = self._scale_shift(y_masked)
        x = y_masked + active * ((y - shift) * jnp.exp(-log_scale))
        logdet = -jnp.sum(log_scale, axis=-1)
        return x, logdet

    def _scale_shift(self, masked):
        mask = _mask_as_float(self.mask, masked)
        active = 1.0 - mask
        scale = _apply_net(self.scale_net, masked)
        shift = _apply_net(self.shift_net, masked)
        log_scale = self.scale_clamp * jnp.tanh(scale) * active
        shift = self.shift_clamp * jnp.tanh(shift) * active
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
        shift_clamp: float = 5.0,
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
                shift_clamp=float(shift_clamp),
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


def realnvp_integrity_diagnostics(
    prior: RealNVPPrior,
    *,
    key=None,
    sample_count: int = 128,
    roundtrip_atol: float = 1.0e-3,
    sample_abs_warn: float = 1.0e4,
    sample_abs_fail: float = 1.0e6,
) -> dict[str, Any]:
    """Return structural and numerical sanity checks for a RealNVP prior.

    These checks are intentionally independent of scientific truth coverage:
    they detect broken flow topology, mismatched forward/inverse conventions,
    and pathological samples from the model itself.
    """
    if not isinstance(prior, RealNVPPrior):
        raise TypeError("realnvp_integrity_diagnostics requires a RealNVPPrior")
    key = jax.random.PRNGKey(0) if key is None else key
    sample_count = max(int(sample_count), 4)
    masks = tuple(jnp.asarray(layer.mask) for layer in prior.layers)
    mask_dtypes = [str(mask.dtype) for mask in masks]
    masks_bool = all(mask.dtype == jnp.bool_ for mask in masks)
    masks_binary = all(bool(jnp.all((mask == 0) | (mask == 1))) for mask in masks)
    masks_static = all(not eqx.is_inexact_array(layer.mask) for layer in prior.layers)

    k_u, k_sample = jax.random.split(key)
    u = jax.random.normal(
        k_u,
        (sample_count, int(prior.latent_dim)),
        dtype=jnp.float32,
    )
    x, _forward_logdet = prior.forward(u)
    recovered_u, _inverse_logdet = prior.inverse(x)
    roundtrip_max_abs = float(jnp.max(jnp.abs(recovered_u - u)))
    roundtrip_median_abs = float(jnp.median(jnp.abs(recovered_u - u)))

    samples = prior.sample(k_sample, sample_count)
    log_prob = prior.log_prob(samples)
    sample_abs = jnp.abs(samples)
    sample_max_abs = float(jnp.max(sample_abs))
    sample_abs_q99 = float(jnp.quantile(sample_abs.reshape(-1), 0.99))
    finite_log_prob_fraction = float(jnp.mean(jnp.isfinite(log_prob)))

    checks = [
        _integrity_check("masks_bool", masks_bool),
        _integrity_check("masks_binary", masks_binary),
        _integrity_check("masks_static_not_trainable", masks_static),
        _integrity_check(
            "forward_inverse_roundtrip_max_abs",
            roundtrip_max_abs <= float(roundtrip_atol),
            value=roundtrip_max_abs,
            fail=float(roundtrip_atol),
        ),
        _integrity_check(
            "sample_log_prob_finite_fraction",
            finite_log_prob_fraction >= 1.0,
            value=finite_log_prob_fraction,
            fail=1.0,
        ),
        _integrity_check(
            "sample_max_abs",
            sample_max_abs < float(sample_abs_fail),
            value=sample_max_abs,
            warn=float(sample_abs_warn),
            fail=float(sample_abs_fail),
            warn_when=sample_max_abs >= float(sample_abs_warn),
        ),
    ]
    status = _status_from_checks(checks)
    return {
        "status": status,
        "checks": checks,
        "latent_dim": int(prior.latent_dim),
        "n_layers": int(len(prior.layers)),
        "mask_dtypes": mask_dtypes,
        "roundtrip_max_abs": roundtrip_max_abs,
        "roundtrip_median_abs": roundtrip_median_abs,
        "sample_abs_q99": sample_abs_q99,
        "sample_max_abs": sample_max_abs,
        "sample_log_prob_finite_fraction": finite_log_prob_fraction,
    }


def assert_realnvp_integrity(
    prior: RealNVPPrior,
    *,
    context: str,
    key=None,
    sample_count: int = 128,
) -> dict[str, Any]:
    """Raise if a RealNVP prior fails structural/numerical integrity checks."""
    diagnostics = realnvp_integrity_diagnostics(
        prior,
        key=key,
        sample_count=sample_count,
    )
    if diagnostics["status"] == "FAIL":
        failed = [
            check["name"]
            for check in diagnostics["checks"]
            if check["status"] == "FAIL"
        ]
        joined = ", ".join(failed)
        raise RuntimeError(
            f"RealNVP integrity check failed for {context}: {joined}. "
            "Do not use this checkpoint as a scientific prior."
        )
    return diagnostics


def _apply_net(net, value):
    if value.ndim == 1:
        return net(value)
    leading = value.shape[:-1]
    flat = value.reshape((-1, value.shape[-1]))
    out = jax.vmap(net)(flat)
    return out.reshape(leading + (out.shape[-1],))


def _mask_as_float(mask, reference):
    return jnp.asarray(mask, dtype=jnp.asarray(reference).dtype)


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
    base = (jnp.arange(latent_dim) % 2).astype(jnp.bool_)
    return tuple(base if index % 2 == 0 else jnp.logical_not(base) for index in range(n_layers))


def _integrity_check(
    name: str,
    ok: bool,
    *,
    value: float | bool | None = None,
    warn: float | None = None,
    fail: float | None = None,
    warn_when: bool = False,
) -> dict[str, Any]:
    if not ok:
        status = "FAIL"
    elif warn_when:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "name": name,
        "status": status,
        "value": value,
        "warn": warn,
        "fail": fail,
    }


def _status_from_checks(checks: list[dict[str, Any]]) -> str:
    if any(check["status"] == "FAIL" for check in checks):
        return "FAIL"
    if any(check["status"] == "WARN" for check in checks):
        return "WARN"
    return "PASS"
