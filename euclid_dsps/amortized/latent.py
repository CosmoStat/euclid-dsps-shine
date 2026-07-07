"""Latent-space transforms for amortized inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.parameter_vectors import free_parameter_bounds_from_config
from euclid_dsps.parameters import POPCOSMOS_PARAMETER_NAMES


@dataclass(frozen=True)
class LatentSpec:
    names: tuple[str, ...]
    lower: jnp.ndarray
    upper: jnp.ndarray
    raw_center: jnp.ndarray | None = None
    raw_scale: jnp.ndarray | None = None
    normalization: str = "identity"


def latent_spec_from_config(config: dict[str, Any]) -> LatentSpec:
    """Build the latent transform spec from configured fit bounds."""
    amortized = config.get("amortized", {}) or {}
    latent = amortized.get("latent", {}) or {}
    schema = str(latent.get("schema", "popcosmos_16"))
    if not bool(latent.get("include_redshift", True)):
        raise ValueError("Amortized inference requires redshift in the latent")
    configured_names = tuple(config.get("fit", {}).get("free_parameters", {}))
    if schema == "popcosmos_16" and configured_names == POPCOSMOS_PARAMETER_NAMES:
        names = configured_names
    elif schema == "popcosmos_16" and set(configured_names) == set(
        POPCOSMOS_PARAMETER_NAMES
    ):
        names = POPCOSMOS_PARAMETER_NAMES
    elif schema in {
        "config_free_parameters",
        "diffsky_dsps_closure_full",
        "diffsky_hltds_prior_v1",
        "diffsky_truth_basic",
    }:
        names = configured_names
        if not names:
            raise ValueError("config.fit.free_parameters must be non-empty")
        if "z_obs" not in names:
            raise ValueError(
                f"Amortized schema {schema!r} requires z_obs in free_parameters"
            )
    else:
        raise ValueError(
            "Unsupported amortized latent schema. Use 'popcosmos_16', "
            "'config_free_parameters', 'diffsky_dsps_closure_full', "
            "'diffsky_hltds_prior_v1', or 'diffsky_truth_basic'."
        )
    lower, upper = free_parameter_bounds_from_config(config, names)
    raw_center, raw_scale, normalization = _latent_normalization_from_config(
        config,
        names,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    return LatentSpec(
        names=names,
        lower=jnp.asarray(lower, dtype=jnp.float32),
        upper=jnp.asarray(upper, dtype=jnp.float32),
        raw_center=jnp.asarray(raw_center, dtype=jnp.float32),
        raw_scale=jnp.asarray(raw_scale, dtype=jnp.float32),
        normalization=normalization,
    )


def x_to_theta(x: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map network latent ``x`` to bounded physical ``theta``."""
    x = _validate_last_dim(jnp.asarray(x, dtype=jnp.float32), spec)
    raw_x = network_x_to_raw_x(x, spec)
    theta = spec.lower + (spec.upper - spec.lower) * jax.nn.sigmoid(raw_x)
    constraint = _gas_metallicity_indices(spec.names)
    if constraint is None:
        return theta
    stellar_index, gas_index = constraint
    stellar_low = spec.lower[stellar_index]
    stellar_high = jnp.minimum(spec.upper[stellar_index], spec.upper[gas_index])
    stellar_span = jnp.maximum(stellar_high - stellar_low, 1.0e-12)
    stellar = stellar_low + stellar_span * jax.nn.sigmoid(raw_x[..., stellar_index])
    gas_low = jnp.maximum(spec.lower[gas_index], stellar)
    gas_span = jnp.maximum(spec.upper[gas_index] - gas_low, 1.0e-12)
    gas = gas_low + gas_span * jax.nn.sigmoid(raw_x[..., gas_index])
    theta = theta.at[..., stellar_index].set(stellar)
    return theta.at[..., gas_index].set(gas)


def theta_to_x(theta: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map bounded physical ``theta`` to network latent ``x``."""
    theta = _validate_last_dim(jnp.asarray(theta, dtype=jnp.float32), spec)
    eps = jnp.asarray(1.0e-6, dtype=theta.dtype)
    scaled = _safe_unit_interval((theta - spec.lower) / (spec.upper - spec.lower), eps)
    raw_x = _logit(scaled)
    constraint = _gas_metallicity_indices(spec.names)
    if constraint is not None:
        stellar_index, gas_index = constraint
        stellar_low = spec.lower[stellar_index]
        stellar_high = jnp.minimum(spec.upper[stellar_index], spec.upper[gas_index])
        stellar_span = jnp.maximum(stellar_high - stellar_low, 1.0e-12)
        stellar_scaled = _safe_unit_interval(
            (theta[..., stellar_index] - stellar_low) / stellar_span,
            eps,
        )
        gas_low = jnp.maximum(spec.lower[gas_index], theta[..., stellar_index])
        gas_span = jnp.maximum(spec.upper[gas_index] - gas_low, 1.0e-12)
        gas_scaled = _safe_unit_interval(
            (theta[..., gas_index] - gas_low) / gas_span,
            eps,
        )
        raw_x = raw_x.at[..., stellar_index].set(_logit(stellar_scaled))
        raw_x = raw_x.at[..., gas_index].set(_logit(gas_scaled))
    return raw_x_to_network_x(raw_x, spec)


def network_x_to_raw_x(x: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map standardized network latent coordinates to raw bounded logits."""
    return _latent_center(spec) + _latent_scale(spec) * x


def raw_x_to_network_x(raw_x: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    """Map raw bounded logits to standardized network latent coordinates."""
    return (raw_x - _latent_center(spec)) / _latent_scale(spec)


def theta_matrix_to_param_dict(
    theta: jnp.ndarray,
    names: tuple[str, ...],
) -> dict[str, jnp.ndarray]:
    """Convert a theta array to a JAX pytree keyed by parameter name."""
    theta = jnp.asarray(theta)
    if theta.shape[-1] != len(names):
        raise ValueError(
            f"theta last dimension must be {len(names)}, got {theta.shape[-1]}"
        )
    return {name: theta[..., index] for index, name in enumerate(names)}


def _validate_last_dim(value: jnp.ndarray, spec: LatentSpec) -> jnp.ndarray:
    if value.shape[-1] != len(spec.names):
        raise ValueError(
            f"Expected last dimension {len(spec.names)}, got {value.shape[-1]}"
        )
    return value


def _safe_unit_interval(value: jnp.ndarray, eps: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(value, eps, 1.0 - eps)


def _logit(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.log(value) - jnp.log1p(-value)


def _gas_metallicity_indices(names: tuple[str, ...]) -> tuple[int, int] | None:
    try:
        return (
            names.index("log10_stellar_metallicity"),
            names.index("log10_gas_metallicity"),
        )
    except ValueError:
        return None


def latent_spec_to_jsonable(spec: LatentSpec) -> dict[str, Any]:
    """Return a JSON-serializable latent spec payload."""
    return {
        "names": list(spec.names),
        "lower": np.asarray(spec.lower).astype(float).tolist(),
        "upper": np.asarray(spec.upper).astype(float).tolist(),
        "raw_center": np.asarray(_latent_center(spec)).astype(float).tolist(),
        "raw_scale": np.asarray(_latent_scale(spec)).astype(float).tolist(),
        "normalization": str(spec.normalization),
    }


def _latent_center(spec: LatentSpec) -> jnp.ndarray:
    if spec.raw_center is None:
        return jnp.zeros_like(spec.lower)
    return jnp.asarray(spec.raw_center, dtype=jnp.float32)


def _latent_scale(spec: LatentSpec) -> jnp.ndarray:
    if spec.raw_scale is None:
        return jnp.ones_like(spec.lower)
    return jnp.maximum(jnp.asarray(spec.raw_scale, dtype=jnp.float32), 1.0e-6)


def _latent_normalization_from_config(
    config: dict[str, Any],
    names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    latent = (config.get("amortized", {}) or {}).get("latent", {}) or {}
    mode = str(latent.get("normalization", latent.get("transform", "identity"))).lower()
    if mode in {"identity", "none", "raw_logit"}:
        return np.zeros(len(names), dtype=float), np.ones(len(names), dtype=float), "identity"
    if mode not in {"standardized_logit", "standardized", "centered_logit"}:
        raise ValueError(
            "amortized.latent.normalization must be 'identity' or 'standardized_logit'"
        )
    theta0 = latent_center_theta_from_config(config, names, lower, upper)
    span = np.maximum(upper - lower, 1.0e-12)
    eps = 1.0e-6
    scaled = np.clip((theta0 - lower) / span, eps, 1.0 - eps)
    raw_center = np.log(scaled) - np.log1p(-scaled)
    derivative = np.maximum(span * scaled * (1.0 - scaled), 1.0e-6)
    physical_scales = dict(latent.get("physical_scales", {}) or {})
    default_fraction = float(latent.get("default_physical_scale_fraction", 0.15))
    default_physical = np.maximum(default_fraction * span, 1.0e-6)
    raw_scale = np.empty(len(names), dtype=float)
    for index, name in enumerate(names):
        physical = float(physical_scales.get(name, default_physical[index]))
        physical = max(abs(physical), 1.0e-6)
        raw_scale[index] = physical / derivative[index]
    raw_scale = np.clip(
        raw_scale,
        float(latent.get("min_raw_scale", 0.05)),
        float(latent.get("max_raw_scale", 5.0)),
    )
    return raw_center, raw_scale, "standardized_logit"


def latent_center_theta_from_config(
    config: dict[str, Any],
    names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return the physical center used by the standardized latent transform.

    This is intentionally separate from ``initial_theta_from_config``.  The
    encoder may still be initialized from ``fit.free_parameters.<name>.initial``,
    while the coordinate system used by a standard-normal latent prior can be
    centered at the bounds midpoint or at explicit physical centers.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    latent = (config.get("amortized", {}) or {}).get("latent", {}) or {}
    source = str(latent.get("center_source", "fit_initial")).strip().lower()
    aliases = {
        "fit": "fit_initial",
        "fit_initials": "fit_initial",
        "initial": "fit_initial",
        "initial_theta": "fit_initial",
        "bounds_midpoint": "midpoint",
        "middle": "midpoint",
        "manual": "explicit",
        "centers": "explicit",
    }
    source = aliases.get(source, source)
    if source == "fit_initial":
        center = initial_theta_from_config(config, names, lower, upper)
    elif source == "midpoint":
        center = 0.5 * (lower + upper)
    elif source == "explicit":
        center = 0.5 * (lower + upper)
    else:
        raise ValueError(
            "amortized.latent.center_source must be one of "
            "'fit_initial', 'midpoint', or 'explicit'"
        )

    explicit = latent.get("centers", latent.get("center_values", {})) or {}
    if not isinstance(explicit, dict):
        raise ValueError("amortized.latent.centers must be a mapping when provided")
    values = np.asarray(center, dtype=float).copy()
    for index, name in enumerate(names):
        value = _finite_float_or_none(explicit.get(name))
        if value is not None:
            values[index] = value
    return _clip_theta_inside_bounds(values, lower, upper)


def initial_theta_from_config(
    config: dict[str, Any],
    names: tuple[str, ...],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Return physical latent initialization from config, with midpoint fallback."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.shape != (len(names),):
        raise ValueError("initial_theta_from_config bounds must match names")
    span = upper - lower
    if np.any(~np.isfinite(span)) or np.any(span <= 0.0):
        raise ValueError("initial_theta_from_config bounds must be finite increasing")
    free = (config.get("fit", {}) or {}).get("free_parameters", {}) or {}
    midpoint = 0.5 * (lower + upper)
    values = []
    for index, name in enumerate(names):
        configured = (free.get(name, {}) or {}).get("initial")
        value = _finite_float_or_none(configured)
        if value is None:
            value = float(midpoint[index])
        values.append(value)
    return _clip_theta_inside_bounds(np.asarray(values, dtype=float), lower, upper)


def latent_prior_geometry_frame(
    config: dict[str, Any],
    *,
    n_samples: int = 200_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize the physical distribution induced by ``x ~ N(0,1)``.

    The table makes the implicit reference prior visible.  It is especially
    useful for bounded sigmoid transforms where a harmless-looking unit normal
    in network coordinates can put substantial mass close to physical bounds.
    """
    spec = latent_spec_from_config(config)
    n_samples = max(int(n_samples), 128)
    rng = np.random.default_rng(int(seed))
    x = rng.normal(size=(n_samples, len(spec.names))).astype(np.float32)
    theta = np.asarray(jax.device_get(x_to_theta(jnp.asarray(x), spec)), dtype=float)
    lower = np.asarray(spec.lower, dtype=float)
    upper = np.asarray(spec.upper, dtype=float)
    span = np.maximum(upper - lower, 1.0e-12)
    raw_center = np.asarray(_latent_center(spec), dtype=float)
    raw_scale = np.asarray(_latent_scale(spec), dtype=float)
    latent_cfg = (config.get("amortized", {}) or {}).get("latent", {}) or {}
    center_source = str(latent_cfg.get("center_source", "fit_initial"))
    rows = []
    for index, name in enumerate(spec.names):
        values = theta[:, index]
        unit = (values - lower[index]) / span[index]
        quantiles = np.nanquantile(
            values,
            [0.001, 0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99, 0.999],
        )
        rows.append(
            {
                "parameter": str(name),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "latent_raw_center": float(raw_center[index]),
                "latent_raw_scale": float(raw_scale[index]),
                "q001": float(quantiles[0]),
                "q01": float(quantiles[1]),
                "q05": float(quantiles[2]),
                "q16": float(quantiles[3]),
                "q50": float(quantiles[4]),
                "q84": float(quantiles[5]),
                "q95": float(quantiles[6]),
                "q99": float(quantiles[7]),
                "q999": float(quantiles[8]),
                "frac_within_lower_1pct": float(np.nanmean(unit < 0.01)),
                "frac_within_lower_5pct": float(np.nanmean(unit < 0.05)),
                "frac_within_upper_5pct": float(np.nanmean(unit > 0.95)),
                "frac_within_upper_1pct": float(np.nanmean(unit > 0.99)),
                "frac_within_either_5pct": float(
                    np.nanmean((unit < 0.05) | (unit > 0.95))
                ),
            }
        )
    frame = pd.DataFrame(rows)
    payload = {
        "n_samples": int(n_samples),
        "seed": int(seed),
        "normalization": str(spec.normalization),
        "center_source": center_source,
        "parameter_names": list(spec.names),
        "max_frac_within_either_5pct": float(
            frame["frac_within_either_5pct"].max()
            if not frame.empty
            else float("nan")
        ),
        "parameters_near_bounds_5pct": frame.loc[
            frame["frac_within_either_5pct"] > 0.05,
            "parameter",
        ].astype(str).tolist(),
    }
    return frame, payload


def write_latent_prior_geometry(
    config: dict[str, Any],
    out_dir: str | Path,
    *,
    n_samples: int = 200_000,
    seed: int = 42,
    prefix: str = "latent_prior_geometry",
) -> dict[str, Any]:
    """Write CSV/JSON/PNG diagnostics for the implicit latent reference prior."""
    out = ensure_dir(out_dir)
    frame, payload = latent_prior_geometry_frame(
        config,
        n_samples=int(n_samples),
        seed=int(seed),
    )
    csv_path = out / f"{prefix}.csv"
    json_path = out / f"{prefix}.json"
    frame.to_csv(csv_path, index=False)
    payload = {
        **payload,
        "csv": csv_path.name,
    }
    png_path = _write_latent_prior_geometry_plot(frame, out / f"{prefix}.png")
    if png_path is not None:
        payload["plot"] = png_path.name
    write_json(json_path, payload)
    return payload


def _clip_theta_inside_bounds(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    span = upper - lower
    edge = np.minimum(1.0e-5, 0.01 * span)
    return np.asarray(
        np.clip(values, lower + edge, upper - edge),
        dtype=float,
    )


def _write_latent_prior_geometry_plot(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plot = frame.sort_values("frac_within_either_5pct", ascending=True)
    labels = plot["parameter"].astype(str).to_numpy()
    lower = plot["frac_within_lower_5pct"].to_numpy(dtype=float)
    upper = plot["frac_within_upper_5pct"].to_numpy(dtype=float)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot) + 1.5)))
    ax.barh(y, lower, color="#4C78A8", label="within lower 5%")
    ax.barh(y, upper, left=lower, color="#F58518", label="within upper 5%")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("fraction of x~N(0,1) mapped near bounds")
    ax.set_xlim(0.0, min(1.0, max(0.1, float(np.nanmax(lower + upper)) * 1.15)))
    ax.legend(loc="lower right")
    ax.set_title("Implicit physical mass near bounds")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
