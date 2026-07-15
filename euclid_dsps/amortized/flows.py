"""Exact-density flow priors ``p_beta(x)`` for unconstrained latent space."""

from __future__ import annotations

import math
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
    permutation: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        latent_dim: int = 16,
        n_layers: int = 8,
        hidden_size: int = 128,
        scale_clamp: float = 0.05,
        shift_clamp: float = 5.0,
        permutation: str = "none",
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
        self.permutation = _validate_flow_permutation(permutation)

    def forward(self, u):
        """Map base ``u`` to latent ``x`` and return forward log-det."""
        value = jnp.asarray(u, dtype=jnp.float32)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for index, layer in enumerate(self.layers):
            value, layer_logdet = layer.forward(value)
            value = jnp.take(
                value,
                _flow_permutation(self.latent_dim, index, self.permutation),
                axis=-1,
            )
            logdet = logdet + layer_logdet
        return value, logdet

    def inverse(self, x):
        """Map latent ``x`` to base ``u`` and return inverse log-det."""
        value = jnp.asarray(x, dtype=jnp.float32)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for index, layer in reversed(tuple(enumerate(self.layers))):
            permutation = _flow_permutation(
                self.latent_dim,
                index,
                self.permutation,
            )
            value = jnp.take(value, jnp.argsort(permutation), axis=-1)
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

    def sample_with_temperature(self, key, shape=(), *, temperature: float = 1.0):
        """Sample using a calibrated isotropic Gaussian base temperature."""
        if float(temperature) <= 0.0:
            raise ValueError("Base temperature must be positive")
        if isinstance(shape, int):
            shape = (shape,)
        u = float(temperature) * jax.random.normal(
            key,
            tuple(shape) + (self.latent_dim,),
            dtype=jnp.float32,
        )
        x, _logdet = self.forward(u)
        return x

    def log_prob_with_temperature(self, x, *, temperature: float = 1.0):
        """Evaluate exact density under an isotropic base temperature."""
        if float(temperature) <= 0.0:
            raise ValueError("Base temperature must be positive")
        u, logdet = self.inverse(x)
        temperature_array = jnp.asarray(temperature, dtype=u.dtype)
        base = -0.5 * jnp.sum(
            (u / temperature_array) ** 2
            + jnp.log(2.0 * jnp.pi)
            + 2.0 * jnp.log(temperature_array),
            axis=-1,
        )
        return base + logdet


class _RQSplineCouplingLayer(eqx.Module):
    mask: jnp.ndarray
    net: object
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
        hidden_size: int,
        mask,
        n_bins: int,
        tail_bound: float,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
        init: str = "default",
        init_scale: float = 1.0,
    ) -> None:
        if int(n_bins) < 2:
            raise ValueError("RQSplineCouplingPrior requires n_bins >= 2")
        if float(tail_bound) <= 0.0:
            raise ValueError("RQSplineCouplingPrior tail_bound must be positive")
        if float(min_bin_width) * int(n_bins) >= 1.0:
            raise ValueError("min_bin_width * n_bins must be < 1")
        if float(min_bin_height) * int(n_bins) >= 1.0:
            raise ValueError("min_bin_height * n_bins must be < 1")
        if not 0.0 < float(min_derivative) < 1.0:
            raise ValueError("min_derivative must be in (0, 1)")
        self.mask = jnp.asarray(mask, dtype=jnp.bool_)
        self.net = eqx.nn.MLP(
            in_size=int(latent_dim),
            out_size=int(latent_dim) * (3 * int(n_bins) + 1),
            width_size=int(hidden_size),
            depth=2,
            activation=jax.nn.gelu,
            key=key,
        )
        init = str(init).lower()
        if init not in {"default", "identity"}:
            raise ValueError("RQSplineCouplingPrior init must be 'default' or 'identity'")
        if init == "identity":
            self.net = _scale_last_linear(self.net, float(init_scale))
        self.n_bins = int(n_bins)
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

    def forward(self, x):
        mask = _mask_as_float(self.mask, x)
        active = 1.0 - mask
        x_masked = x * mask
        params = self._params(x_masked)
        transformed, logdet = _rational_quadratic_spline(
            x,
            params,
            inverse=False,
            n_bins=self.n_bins,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        y = x * mask + active * transformed
        return y, jnp.sum(active * logdet, axis=-1)

    def inverse(self, y):
        mask = _mask_as_float(self.mask, y)
        active = 1.0 - mask
        y_masked = y * mask
        params = self._params(y_masked)
        transformed, logdet = _rational_quadratic_spline(
            y,
            params,
            inverse=True,
            n_bins=self.n_bins,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        )
        x = y * mask + active * transformed
        return x, jnp.sum(active * logdet, axis=-1)

    def _params(self, masked):
        raw = _apply_net(self.net, masked)
        return raw.reshape(raw.shape[:-1] + (masked.shape[-1], 3 * self.n_bins + 1))


class RQSplineCouplingPrior(eqx.Module):
    """Fast coupling Neural Spline Flow prior over unconstrained latent ``x``."""

    layers: tuple
    permutations: tuple
    inverse_permutations: tuple
    latent_dim: int = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)
    min_bin_width: float = eqx.field(static=True)
    min_bin_height: float = eqx.field(static=True)
    min_derivative: float = eqx.field(static=True)
    permutation: str = eqx.field(static=True)

    def __init__(
        self,
        key,
        *,
        latent_dim: int = 16,
        n_layers: int = 12,
        hidden_size: int = 256,
        n_bins: int = 16,
        tail_bound: float = 12.0,
        min_bin_width: float = 1.0e-3,
        min_bin_height: float = 1.0e-3,
        min_derivative: float = 1.0e-3,
        permutation: str = "roll",
        init: str = "default",
        init_scale: float = 1.0,
    ) -> None:
        keys = jax.random.split(key, int(n_layers))
        masks = _alternating_masks(int(latent_dim), int(n_layers))
        self.layers = tuple(
            _RQSplineCouplingLayer(
                keys[index],
                latent_dim=int(latent_dim),
                hidden_size=int(hidden_size),
                mask=masks[index],
                n_bins=int(n_bins),
                tail_bound=float(tail_bound),
                min_bin_width=float(min_bin_width),
                min_bin_height=float(min_bin_height),
                min_derivative=float(min_derivative),
                init=str(init),
                init_scale=float(init_scale),
            )
            for index in range(int(n_layers))
        )
        permutations = tuple(
            _flow_permutation(int(latent_dim), index, str(permutation))
            for index in range(int(n_layers))
        )
        self.permutations = permutations
        self.inverse_permutations = tuple(jnp.argsort(perm) for perm in permutations)
        self.latent_dim = int(latent_dim)
        self.n_bins = int(n_bins)
        self.tail_bound = float(tail_bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)
        self.permutation = str(permutation).lower()

    def forward(self, u):
        """Map base ``u`` to latent ``x`` and return forward log-det."""
        value = jnp.asarray(u, dtype=jnp.float32)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for layer, permutation in zip(self.layers, self.permutations, strict=True):
            value, layer_logdet = layer.forward(value)
            value = jnp.take(value, permutation, axis=-1)
            logdet = logdet + layer_logdet
        return value, logdet

    def inverse(self, x):
        """Map latent ``x`` to base ``u`` and return inverse log-det."""
        value = jnp.asarray(x, dtype=jnp.float32)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for layer, inverse_permutation in zip(
            reversed(self.layers),
            reversed(self.inverse_permutations),
            strict=True,
        ):
            value = jnp.take(value, inverse_permutation, axis=-1)
            value, layer_logdet = layer.inverse(value)
            logdet = logdet + layer_logdet
        return value, logdet

    def log_prob(self, x):
        """Exact pointwise ``log p_beta(x)``."""
        u, logdet = self.inverse(x)
        base = -0.5 * jnp.sum(u**2 + jnp.log(2.0 * jnp.pi), axis=-1)
        return base + logdet

    def sample(self, key, shape=()):
        """Sample latent ``x`` from the spline coupling prior."""
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


def flow_integrity_diagnostics(
    prior,
    *,
    key=None,
    sample_count: int = 128,
    roundtrip_atol: float = 1.0e-3,
    roundtrip_fail_atol: float = 1.0e-2,
    sample_abs_warn: float = 1.0e4,
    sample_abs_fail: float = 1.0e6,
) -> dict[str, Any]:
    """Return structural and numerical sanity checks for an exact flow prior.

    These checks are intentionally independent of scientific truth coverage:
    they detect broken flow topology, mismatched forward/inverse conventions,
    and pathological samples from the model itself.
    """
    if not _is_exact_flow_prior(prior):
        raise TypeError(
            "flow_integrity_diagnostics requires a prior with forward, inverse, "
            "log_prob, sample, and latent_dim"
        )
    key = jax.random.PRNGKey(0) if key is None else key
    sample_count = max(int(sample_count), 4)
    layers = tuple(getattr(prior, "layers", ()))
    masks = tuple(jnp.asarray(layer.mask) for layer in layers if hasattr(layer, "mask"))
    mask_dtypes = [str(mask.dtype) for mask in masks]
    masks_bool = all(mask.dtype == jnp.bool_ for mask in masks)
    masks_binary = all(bool(jnp.all((mask == 0) | (mask == 1))) for mask in masks)
    masks_static = all(not eqx.is_inexact_array(layer.mask) for layer in layers)

    permutations = tuple(jnp.asarray(perm) for perm in getattr(prior, "permutations", ()))
    permutation_dtypes = [str(perm.dtype) for perm in permutations]
    expected_perm = jnp.arange(int(prior.latent_dim))
    permutations_valid = all(
        bool(jnp.array_equal(jnp.sort(perm), expected_perm)) for perm in permutations
    )
    permutations_static = all(not eqx.is_inexact_array(perm) for perm in permutations)

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
        _integrity_check("permutations_valid", permutations_valid),
        _integrity_check("permutations_static_not_trainable", permutations_static),
        _integrity_check(
            "forward_inverse_roundtrip_max_abs",
            roundtrip_max_abs <= float(roundtrip_fail_atol),
            value=roundtrip_max_abs,
            warn=float(roundtrip_atol),
            fail=float(roundtrip_fail_atol),
            warn_when=roundtrip_max_abs > float(roundtrip_atol),
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
        "prior_type": _prior_type_name(prior),
        "checks": checks,
        "latent_dim": int(prior.latent_dim),
        "n_layers": int(len(layers)),
        "mask_dtypes": mask_dtypes,
        "permutation_dtypes": permutation_dtypes,
        "roundtrip_max_abs": roundtrip_max_abs,
        "roundtrip_median_abs": roundtrip_median_abs,
        "sample_abs_q99": sample_abs_q99,
        "sample_max_abs": sample_max_abs,
        "sample_log_prob_finite_fraction": finite_log_prob_fraction,
    }


def assert_flow_integrity(
    prior,
    *,
    context: str,
    key=None,
    sample_count: int = 128,
) -> dict[str, Any]:
    """Raise if an exact flow prior fails structural/numerical integrity checks."""
    diagnostics = flow_integrity_diagnostics(
        prior,
        key=key,
        sample_count=sample_count,
    )
    if diagnostics["status"] == "FAIL":
        failed = [
            _format_failed_integrity_check(check)
            for check in diagnostics["checks"]
            if check["status"] == "FAIL"
        ]
        joined = ", ".join(failed)
        raise RuntimeError(
            f"{_prior_type_name(prior)} integrity check failed for {context}: {joined}. "
            "Do not use this checkpoint as a scientific prior."
        )
    return diagnostics


def realnvp_integrity_diagnostics(
    prior: RealNVPPrior,
    *,
    key=None,
    sample_count: int = 128,
    roundtrip_atol: float = 1.0e-3,
    roundtrip_fail_atol: float = 1.0e-2,
    sample_abs_warn: float = 1.0e4,
    sample_abs_fail: float = 1.0e6,
) -> dict[str, Any]:
    """Return structural and numerical sanity checks for a RealNVP prior."""
    if not isinstance(prior, RealNVPPrior):
        raise TypeError("realnvp_integrity_diagnostics requires a RealNVPPrior")
    return flow_integrity_diagnostics(
        prior,
        key=key,
        sample_count=sample_count,
        roundtrip_atol=roundtrip_atol,
        roundtrip_fail_atol=roundtrip_fail_atol,
        sample_abs_warn=sample_abs_warn,
        sample_abs_fail=sample_abs_fail,
    )


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
            _format_failed_integrity_check(check)
            for check in diagnostics["checks"]
            if check["status"] == "FAIL"
        ]
        joined = ", ".join(failed)
        raise RuntimeError(
            f"RealNVP integrity check failed for {context}: {joined}. "
            "Do not use this checkpoint as a scientific prior."
        )
    return diagnostics


def _is_exact_flow_prior(prior) -> bool:
    return all(
        hasattr(prior, name)
        for name in ("forward", "inverse", "log_prob", "sample", "latent_dim")
    )


def _prior_type_name(prior) -> str:
    if isinstance(prior, RealNVPPrior):
        return "RealNVP"
    if isinstance(prior, RQSplineCouplingPrior):
        return "RQSplineCoupling"
    return type(prior).__name__


def _format_failed_integrity_check(check: dict[str, Any]) -> str:
    name = str(check.get("name", "unknown"))
    if "value" not in check:
        return name
    value = check.get("value")
    fail = check.get("fail")
    if fail is None:
        return f"{name}={value}"
    return f"{name}={value} fail_threshold={fail}"


def _apply_net(net, value):
    if value.ndim == 1:
        return net(value)
    leading = value.shape[:-1]
    flat = value.reshape((-1, value.shape[-1]))
    out = jax.vmap(net)(flat)
    return out.reshape(leading + (out.shape[-1],))


def _mask_as_float(mask, reference):
    return jnp.asarray(mask, dtype=jnp.asarray(reference).dtype)


def _flow_permutation(latent_dim: int, index: int, mode: str) -> jnp.ndarray:
    indices = jnp.arange(int(latent_dim), dtype=jnp.int32)
    normalized = _validate_flow_permutation(mode)
    if normalized in {"none", "identity", "false"}:
        return indices
    if normalized == "reverse":
        return indices[::-1]
    if normalized == "roll":
        return jnp.roll(indices, shift=(int(index) + 1) % int(latent_dim))
    raise AssertionError(f"Unhandled flow permutation: {normalized}")


def _validate_flow_permutation(mode: str) -> str:
    normalized = str(mode).strip().lower()
    aliases = {"identity": "none", "false": "none"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"roll", "reverse", "none"}:
        raise ValueError("Flow permutation must be 'roll', 'reverse', or 'none'")
    return normalized


def _rational_quadratic_spline(
    inputs,
    raw_params,
    *,
    inverse: bool,
    n_bins: int,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
):
    inputs = jnp.asarray(inputs, dtype=jnp.float32)
    raw_params = jnp.asarray(raw_params, dtype=inputs.dtype)
    widths, heights, derivatives = _spline_parameters(
        raw_params,
        n_bins=int(n_bins),
        tail_bound=float(tail_bound),
        min_bin_width=float(min_bin_width),
        min_bin_height=float(min_bin_height),
        min_derivative=float(min_derivative),
    )
    cumwidths = _cumulative_knots(widths, float(tail_bound))
    cumheights = _cumulative_knots(heights, float(tail_bound))
    inside = (inputs >= -float(tail_bound)) & (inputs <= float(tail_bound))
    outputs, logabsdet = _rational_quadratic_spline_inside(
        inputs,
        widths,
        heights,
        derivatives,
        cumwidths,
        cumheights,
        inverse=bool(inverse),
    )
    outputs = jnp.where(inside, outputs, inputs)
    logabsdet = jnp.where(inside, logabsdet, jnp.zeros_like(inputs))
    return outputs, logabsdet


def _spline_parameters(
    raw_params,
    *,
    n_bins: int,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
):
    raw_widths = raw_params[..., :n_bins]
    raw_heights = raw_params[..., n_bins : 2 * n_bins]
    raw_derivatives = raw_params[..., 2 * n_bins :]
    widths = min_bin_width + (1.0 - min_bin_width * n_bins) * jax.nn.softmax(
        raw_widths,
        axis=-1,
    )
    heights = min_bin_height + (1.0 - min_bin_height * n_bins) * jax.nn.softmax(
        raw_heights,
        axis=-1,
    )
    widths = widths * (2.0 * float(tail_bound))
    heights = heights * (2.0 * float(tail_bound))
    derivative_offset = _inverse_softplus(1.0 - float(min_derivative))
    derivatives = float(min_derivative) + jax.nn.softplus(
        raw_derivatives + derivative_offset
    )
    return widths, heights, derivatives


def _cumulative_knots(bin_sizes, tail_bound: float):
    zeros = jnp.zeros(bin_sizes.shape[:-1] + (1,), dtype=bin_sizes.dtype)
    knots = jnp.concatenate([zeros, jnp.cumsum(bin_sizes, axis=-1)], axis=-1)
    knots = knots - float(tail_bound)
    knots = knots.at[..., 0].set(-float(tail_bound))
    knots = knots.at[..., -1].set(float(tail_bound))
    return knots


def _rational_quadratic_spline_inside(
    inputs,
    widths,
    heights,
    derivatives,
    cumwidths,
    cumheights,
    *,
    inverse: bool,
):
    bin_locations = cumheights if inverse else cumwidths
    bin_idx = jnp.sum(inputs[..., None] >= bin_locations[..., 1:-1], axis=-1)
    bin_idx = jnp.clip(bin_idx, 0, widths.shape[-1] - 1)

    input_cumwidths = _gather_spline_bin(cumwidths, bin_idx)
    input_bin_widths = _gather_spline_bin(widths, bin_idx)
    input_cumheights = _gather_spline_bin(cumheights, bin_idx)
    input_bin_heights = _gather_spline_bin(heights, bin_idx)
    input_delta = input_bin_heights / input_bin_widths
    input_derivatives = _gather_spline_bin(derivatives, bin_idx)
    input_derivatives_plus_one = _gather_spline_bin(derivatives, bin_idx + 1)

    if inverse:
        y_minus_yk = inputs - input_cumheights
        theta = _inverse_rational_quadratic_theta(
            y_minus_yk,
            input_bin_heights,
            input_delta,
            input_derivatives,
            input_derivatives_plus_one,
        )
        outputs = input_cumwidths + theta * input_bin_widths
        logabsdet = -_rational_quadratic_logabsdet(
            theta,
            input_delta,
            input_derivatives,
            input_derivatives_plus_one,
        )
        return outputs, logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1.0 - theta)
    numerator = input_bin_heights * (
        input_delta * theta**2 + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
    ) * theta_one_minus_theta
    outputs = input_cumheights + numerator / denominator
    logabsdet = _rational_quadratic_logabsdet(
        theta,
        input_delta,
        input_derivatives,
        input_derivatives_plus_one,
    )
    return outputs, logabsdet


def _inverse_rational_quadratic_theta(
    y_minus_yk,
    input_bin_heights,
    input_delta,
    input_derivatives,
    input_derivatives_plus_one,
):
    common = input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
    a = y_minus_yk * common + input_bin_heights * (input_delta - input_derivatives)
    b = input_bin_heights * input_derivatives - y_minus_yk * common
    c = -input_delta * y_minus_yk
    discriminant = jnp.maximum(b**2 - 4.0 * a * c, 0.0)
    sqrt_discriminant = jnp.sqrt(discriminant)
    theta = (2.0 * c) / (-b - sqrt_discriminant)
    return jnp.clip(theta, 0.0, 1.0)


def _rational_quadratic_logabsdet(
    theta,
    input_delta,
    input_derivatives,
    input_derivatives_plus_one,
):
    theta_one_minus_theta = theta * (1.0 - theta)
    denominator = input_delta + (
        input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
    ) * theta_one_minus_theta
    derivative_numerator = input_delta**2 * (
        input_derivatives_plus_one * theta**2
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * (1.0 - theta) ** 2
    )
    return jnp.log(derivative_numerator) - 2.0 * jnp.log(denominator)


def _gather_spline_bin(values, bin_idx):
    return jnp.take_along_axis(values, bin_idx[..., None], axis=-1)[..., 0]


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


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
