"""Production helpers for the FENIKS spline-SFH 15D prior contract."""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z

from euclid_dsps import model as dsps_model
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
from euclid_dsps.synthetic_diffsky.photometry import (
    GROUND_TRUTH_COLUMNS,
    theta_from_truth_frame,
)

N_SPLINE_NODES = 11
PHYSICAL_PARAMETER_NAMES = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)
SFH_CONTRAST_NAMES = tuple(
    f"sfh_dlog_sfr_{index:02d}" for index in range(1, N_SPLINE_NODES)
)
SPLINE15D_PARAMETER_NAMES = PHYSICAL_PARAMETER_NAMES + SFH_CONTRAST_NAMES
DEFAULT_NORMALIZED_LOG_TIME_NODES = np.asarray(
    [
        0.0,
        0.21079183733299048,
        0.3184771392180239,
        0.4724391306958804,
        0.5856181563118241,
        0.7121889547787805,
        0.7983831964923024,
        0.8600970174232022,
        0.9124328691648703,
        0.9588273275857481,
        1.0,
    ],
    dtype=np.float64,
)
DEFAULT_NORMALIZED_LOG_TIME_NODES_JAX = jnp.asarray(DEFAULT_NORMALIZED_LOG_TIME_NODES)
GAUSSIAN_SCORE_PROBABILITIES = np.linspace(0.005, 0.995, 199)
GAUSSIAN_SCORE_TARGET = np.asarray(
    [NormalDist().inv_cdf(float(value)) for value in GAUSSIAN_SCORE_PROBABILITIES]
)


def validate_normalized_log_time_nodes(values: np.ndarray) -> np.ndarray:
    """Return a validated length-11 normalized log-time node vector."""
    nodes = np.asarray(values, dtype=np.float64)
    if nodes.shape != (N_SPLINE_NODES,):
        raise ValueError(f"Expected {N_SPLINE_NODES} normalized log-time nodes")
    if not np.isfinite(nodes).all() or np.any(np.diff(nodes) <= 0.0):
        raise ValueError("Normalized log-time nodes must be finite and increasing")
    if not np.isclose(nodes[0], 0.0) or not np.isclose(nodes[-1], 1.0):
        raise ValueError("Normalized log-time nodes must start at 0 and end at 1")
    return nodes


def project_diffsky_frame_to_spline15d(
    frame: pd.DataFrame,
    *,
    normalized_log_time_nodes: np.ndarray = DEFAULT_NORMALIZED_LOG_TIME_NODES,
    n_sfh_bins: int = 80,
    batch_size: int = 2048,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project Diffsky truth rows to five physical plus ten spline contrasts."""
    if frame.empty:
        raise ValueError("Cannot project an empty Diffsky frame")
    nodes = validate_normalized_log_time_nodes(normalized_log_time_nodes)
    if int(n_sfh_bins) < N_SPLINE_NODES:
        raise ValueError("n_sfh_bins must be at least the spline node count")
    theta = theta_from_truth_frame(frame)
    kernel = _projection_kernel(int(n_sfh_bins), tuple(float(value) for value in nodes))
    knot_time_parts = []
    knot_log_sfr_parts = []
    for start in range(0, len(frame), max(int(batch_size), 1)):
        knot_time, knot_log_sfr = kernel(
            jnp.asarray(theta[start : start + int(batch_size)], dtype=jnp.float32)
        )
        knot_time_parts.append(np.asarray(jax.device_get(knot_time), dtype=np.float64))
        knot_log_sfr_parts.append(
            np.asarray(jax.device_get(knot_log_sfr), dtype=np.float64)
        )
    knot_time = np.concatenate(knot_time_parts, axis=0)
    knot_log_sfr = np.concatenate(knot_log_sfr_parts, axis=0)
    contrasts = np.diff(knot_log_sfr, axis=1)
    latent_payload = {
        name: frame[GROUND_TRUTH_COLUMNS[name]].to_numpy(dtype=float)
        for name in PHYSICAL_PARAMETER_NAMES
    }
    latent_payload.update(
        {name: contrasts[:, index] for index, name in enumerate(SFH_CONTRAST_NAMES)}
    )
    latent = pd.DataFrame(latent_payload)
    audit = pd.DataFrame(
        {
            "object_id": _object_ids(frame),
            **{
                f"spline_time_gyr_{index:02d}": knot_time[:, index]
                for index in range(N_SPLINE_NODES)
            },
            **{
                f"spline_log_sfr_{index:02d}": knot_log_sfr[:, index]
                for index in range(N_SPLINE_NODES)
            },
        }
    )
    latent.insert(0, "object_id", _object_ids(frame))
    _validate_projected_frame(latent)
    return latent, audit


def dequantize_spline_contrast_atoms(
    frame: pd.DataFrame,
    *,
    half_width_dex: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Uniformly dequantize exact zero atoms in the ten SFH contrasts."""
    if float(half_width_dex) <= 0.0:
        raise ValueError("half_width_dex must be positive")
    result = frame.copy()
    rng = np.random.default_rng(int(seed))
    counts: dict[str, int] = {}
    for name in SFH_CONTRAST_NAMES:
        values = result[name].to_numpy(dtype=float, copy=True)
        mask = values == 0.0
        count = int(np.sum(mask))
        if count:
            values[mask] = rng.uniform(-half_width_dex, half_width_dex, count)
        result[name] = values
        counts[name] = count
    _validate_projected_frame(result)
    return result, counts


def fit_asinh_transforms(
    values: np.ndarray,
    names: tuple[str, ...] = SPLINE15D_PARAMETER_NAMES,
    *,
    log10_lambda_min: float = -8.0,
    log10_lambda_max: float = 8.0,
    grid_size: int = 257,
) -> dict[str, dict[str, Any]]:
    """Fit one low-capacity asinh transform per coordinate on train only."""
    matrix = _validate_matrix(values, names)
    if int(grid_size) < 3:
        raise ValueError("grid_size must be at least 3")
    exponents = np.linspace(log10_lambda_min, log10_lambda_max, int(grid_size))
    transforms: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        column = matrix[:, index]
        data_scale = max(float(np.std(column)), 1.0e-12)
        physical_quantiles = np.quantile(column, GAUSSIAN_SCORE_PROBABILITIES)
        best: tuple[float, float, float, float] | None = None
        for exponent in exponents:
            lam = data_scale * 10.0 ** float(exponent)
            raw = lam * np.arcsinh(column / lam)
            center = float(np.mean(raw))
            scale = float(np.std(raw))
            normalized_quantiles = (
                lam * np.arcsinh(physical_quantiles / lam) - center
            ) / scale
            score = float(
                np.sqrt(np.mean((normalized_quantiles - GAUSSIAN_SCORE_TARGET) ** 2))
            )
            candidate = (score, lam, center, scale)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        score, lam, center, scale = best
        transforms[name] = {
            "family": "asinh",
            "lambda": float(lam),
            "center": float(center),
            "scale": float(scale),
            "log10_lambda_over_raw_std": float(np.log10(lam / data_scale)),
            "train_gaussian_qrmse": float(score),
            "grid_log10_lambda_min": float(log10_lambda_min),
            "grid_log10_lambda_max": float(log10_lambda_max),
            "grid_size": int(grid_size),
        }
    return transforms


def fit_shifted_asinh_transforms(
    values: np.ndarray,
    names: tuple[str, ...] = SPLINE15D_PARAMETER_NAMES,
    *,
    lower_quantile: float = 0.15865525393145707,
    upper_quantile: float = 0.8413447460685429,
    minimum_lambda: float = 1.0e-6,
) -> dict[str, dict[str, Any]]:
    """Fit a robust shifted-asinh transform without marginal Gaussianization."""
    matrix = _validate_matrix(values, names)
    if not 0.0 < float(lower_quantile) < float(upper_quantile) < 1.0:
        raise ValueError("Shifted-asinh quantiles must satisfy 0 < lower < upper < 1")
    if float(minimum_lambda) <= 0.0:
        raise ValueError("minimum_lambda must be positive")
    transforms: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        column = matrix[:, index]
        location = float(np.median(column))
        q_low, q_high = np.quantile(
            column, [float(lower_quantile), float(upper_quantile)]
        )
        robust_scale = 0.5 * float(q_high - q_low)
        fallback_scale = max(float(np.std(column)), float(minimum_lambda))
        lam = max(robust_scale, fallback_scale * 1.0e-6, float(minimum_lambda))
        raw = np.arcsinh((column - location) / lam)
        center = float(np.mean(raw))
        scale = max(float(np.std(raw)), 1.0e-12)
        normalized = (raw - center) / scale
        transforms[name] = {
            "family": "shifted_asinh",
            "location": location,
            "lambda": float(lam),
            "center": center,
            "scale": scale,
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
            "train_q_low": float(q_low),
            "train_q_high": float(q_high),
            "train_gaussian_qrmse": gaussian_quantile_rmse(normalized),
        }
    return transforms


def forward_asinh_matrix(
    values: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    names: tuple[str, ...] = SPLINE15D_PARAMETER_NAMES,
) -> np.ndarray:
    """Apply fitted asinh transforms to a matrix."""
    matrix = _validate_matrix(values, names)
    columns = []
    for index, name in enumerate(names):
        transform = transforms[name]
        family = str(transform.get("family", "asinh"))
        if family == "shifted_asinh":
            raw = np.arcsinh(
                (matrix[:, index] - float(transform["location"]))
                / float(transform["lambda"])
            )
        elif family == "asinh":
            lam = float(transform["lambda"])
            raw = lam * np.arcsinh(matrix[:, index] / lam)
        else:
            raise ValueError(f"Unsupported spline15d marginal transform: {family}")
        columns.append((raw - float(transform["center"])) / float(transform["scale"]))
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("asinh normalization produced non-finite values")
    return result


def inverse_asinh_matrix(
    values: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    names: tuple[str, ...] = SPLINE15D_PARAMETER_NAMES,
) -> np.ndarray:
    """Invert fitted asinh transforms."""
    matrix = _validate_matrix(values, names)
    arguments = inverse_asinh_arguments_matrix(matrix, transforms, names)
    columns = []
    for index, name in enumerate(names):
        transform = transforms[name]
        family = str(transform.get("family", "asinh"))
        if family == "shifted_asinh":
            physical = float(transform["location"]) + float(
                transform["lambda"]
            ) * np.sinh(arguments[:, index])
        elif family == "asinh":
            physical = float(transform["lambda"]) * np.sinh(arguments[:, index])
        else:
            raise ValueError(f"Unsupported spline15d marginal transform: {family}")
        columns.append(physical)
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("inverse asinh normalization produced non-finite values")
    return result


def inverse_asinh_arguments_matrix(
    values: np.ndarray,
    transforms: dict[str, dict[str, Any]],
    names: tuple[str, ...] = SPLINE15D_PARAMETER_NAMES,
) -> np.ndarray:
    """Return the dimensionless arguments passed to sinh by the inverse."""
    matrix = _validate_matrix(values, names)
    columns = []
    for index, name in enumerate(names):
        transform = transforms[name]
        raw = float(transform["center"]) + float(transform["scale"]) * matrix[:, index]
        family = str(transform.get("family", "asinh"))
        if family == "asinh":
            raw = raw / float(transform["lambda"])
        elif family != "shifted_asinh":
            raise ValueError(f"Unsupported spline15d marginal transform: {family}")
        columns.append(raw)
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("inverse asinh arguments contain non-finite values")
    return result


def normalized_physical_zero(
    transform: dict[str, Any],
) -> float:
    """Return the marginal-normalized coordinate corresponding to physical zero."""
    family = str(transform.get("family", "asinh"))
    if family == "shifted_asinh":
        raw_zero = np.arcsinh(
            -float(transform["location"]) / float(transform["lambda"])
        )
    elif family == "asinh":
        raw_zero = 0.0
    else:
        raise ValueError(f"Unsupported spline15d marginal transform: {family}")
    return (float(raw_zero) - float(transform["center"])) / float(transform["scale"])


def dequantize_normalized_zero_atoms(
    normalized: np.ndarray,
    exact_physical: np.ndarray,
    *,
    half_width: float,
    seed: int,
    names: tuple[str, ...] = SPLINE15D_PARAMETER_NAMES,
) -> tuple[np.ndarray, dict[str, int]]:
    """Dequantize exact-zero SFH contrasts in normalized coordinates."""
    values = _validate_matrix(normalized, names).copy()
    physical = _validate_matrix(exact_physical, names)
    if float(half_width) <= 0.0:
        raise ValueError("Normalized atom half-width must be positive")
    rng = np.random.default_rng(int(seed))
    counts: dict[str, int] = {}
    for name in SFH_CONTRAST_NAMES:
        index = names.index(name)
        mask = physical[:, index] == 0.0
        count = int(np.sum(mask))
        if count:
            values[mask, index] += rng.uniform(-half_width, half_width, count)
        counts[name] = count
    return values, counts


def fit_affine_whitening(
    values: np.ndarray,
    *,
    covariance_jitter: float = 1.0e-5,
) -> dict[str, Any]:
    """Fit a full-covariance affine whitening transform."""
    matrix = _validate_matrix(values, SPLINE15D_PARAMETER_NAMES)
    if float(covariance_jitter) <= 0.0:
        raise ValueError("covariance_jitter must be positive")
    mean = np.mean(matrix, axis=0)
    covariance = np.cov(matrix, rowvar=False)
    regularized = covariance + float(covariance_jitter) * np.eye(matrix.shape[1])
    cholesky = np.linalg.cholesky(regularized)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {
        "family": "cholesky_whitening",
        "fit_split": "train",
        "mean": mean.tolist(),
        "cholesky": cholesky.tolist(),
        "covariance_jitter": float(covariance_jitter),
        "covariance_eigenvalue_min": float(np.min(eigenvalues)),
        "covariance_eigenvalue_max": float(np.max(eigenvalues)),
        "covariance_condition": float(np.max(eigenvalues) / np.min(eigenvalues)),
    }


def forward_affine_whitening(
    values: np.ndarray,
    whitening: dict[str, Any],
) -> np.ndarray:
    """Apply a fitted Cholesky whitening transform."""
    matrix = _validate_matrix(values, SPLINE15D_PARAMETER_NAMES)
    mean = np.asarray(whitening["mean"], dtype=np.float64)
    cholesky = np.asarray(whitening["cholesky"], dtype=np.float64)
    result = np.linalg.solve(cholesky, (matrix - mean).T).T
    if not np.isfinite(result).all():
        raise ValueError("Whitening produced non-finite values")
    return result


def inverse_affine_whitening(
    values: np.ndarray,
    whitening: dict[str, Any],
) -> np.ndarray:
    """Invert a fitted Cholesky whitening transform."""
    matrix = _validate_matrix(values, SPLINE15D_PARAMETER_NAMES)
    mean = np.asarray(whitening["mean"], dtype=np.float64)
    cholesky = np.asarray(whitening["cholesky"], dtype=np.float64)
    result = matrix @ cholesky.T + mean
    if not np.isfinite(result).all():
        raise ValueError("Inverse whitening produced non-finite values")
    return result


def inverse_spline15d_flow_coordinates(
    values: np.ndarray,
    *,
    transforms: dict[str, dict[str, Any]],
    whitening: dict[str, Any] | None = None,
    atom_half_width: float | None = None,
) -> np.ndarray:
    """Map internal flow coordinates back to exact scientific coordinates."""
    asinh_values = (
        np.asarray(values, dtype=np.float64)
        if whitening is None
        else inverse_affine_whitening(values, whitening)
    )
    physical = inverse_asinh_matrix(asinh_values, transforms)
    if atom_half_width is not None and float(atom_half_width) > 0.0:
        for name in SFH_CONTRAST_NAMES:
            index = SPLINE15D_PARAMETER_NAMES.index(name)
            transform = transforms[name]
            zero_normalized = normalized_physical_zero(transform)
            mask = np.abs(asinh_values[:, index] - zero_normalized) <= float(
                atom_half_width
            )
            physical[mask, index] = 0.0
    return physical


def gaussian_quantile_rmse(values: np.ndarray) -> float:
    """Return marginal quantile RMSE to a standard normal."""
    quantiles = np.quantile(
        np.asarray(values, dtype=float), GAUSSIAN_SCORE_PROBABILITIES
    )
    return float(np.sqrt(np.mean((quantiles - GAUSSIAN_SCORE_TARGET) ** 2)))


def spline_knot_times_jax(
    time_gyr: jnp.ndarray,
    normalized_log_time_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """Return per-galaxy spline times from normalized log-cosmic-time nodes."""
    log_min = jnp.log10(jnp.maximum(time_gyr[0], 1.0e-6))
    log_max = jnp.log10(jnp.maximum(time_gyr[-1], 1.0e-6))
    return 10 ** (log_min + normalized_log_time_nodes * (log_max - log_min))


def reconstruct_relative_sfh_jax(
    time_gyr: jnp.ndarray,
    contrasts: jnp.ndarray,
    normalized_log_time_nodes: jnp.ndarray = DEFAULT_NORMALIZED_LOG_TIME_NODES_JAX,
) -> jnp.ndarray:
    """Reconstruct relative SFH shape; stellar mass supplies its amplitude."""
    knot_time = spline_knot_times_jax(time_gyr, normalized_log_time_nodes)
    knot_log_sfr = jnp.concatenate((jnp.zeros(1), jnp.cumsum(contrasts)))
    log_sfr = pchip_interpolate_jax(
        jnp.log10(knot_time), knot_log_sfr, jnp.log10(time_gyr)
    )
    return 10**log_sfr


def pchip_interpolate_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_new: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate a shape-preserving cubic Hermite spline in JAX."""
    slopes = _pchip_slopes_jax(x, y)
    segment = jnp.clip(jnp.searchsorted(x, x_new, side="right") - 1, 0, x.size - 2)
    x0, x1 = x[segment], x[segment + 1]
    y0, y1 = y[segment], y[segment + 1]
    m0, m1 = slopes[segment], slopes[segment + 1]
    width = x1 - x0
    unit = (x_new - x0) / width
    h00 = 2.0 * unit**3 - 3.0 * unit**2 + 1.0
    h10 = unit**3 - 2.0 * unit**2 + unit
    h01 = -2.0 * unit**3 + 3.0 * unit**2
    h11 = unit**3 - unit**2
    return h00 * y0 + h10 * width * m0 + h01 * y1 + h11 * width * m1


def _projection_kernel(n_sfh_bins: int, nodes: tuple[float, ...]):
    names = tuple(DIFFSKY_BASIC_PARAMETER_NAMES)
    nodes_jax = jnp.asarray(nodes, dtype=jnp.float32)

    def single(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        params = {name: theta[index] for index, name in enumerate(names)}
        z_obs = params["z_obs"]
        t_obs = jnp.ravel(age_at_z(z_obs, *DEFAULT_COSMOLOGY))[0]
        time = jnp.linspace(0.05, jnp.maximum(t_obs, 0.06), int(n_sfh_bins))
        sfh = dsps_model.build_diffsky_basic_sfh_table_jax(time, t_obs, params)
        knot_time = spline_knot_times_jax(time, nodes_jax)
        knot_log_sfr = jnp.interp(
            jnp.log10(knot_time),
            jnp.log10(time),
            jnp.log10(jnp.maximum(sfh, 1.0e-30)),
        )
        return knot_time, knot_log_sfr

    return jax.jit(jax.vmap(single, in_axes=0))


def _pchip_endpoint_slope_jax(
    h0: jnp.ndarray,
    h1: jnp.ndarray,
    delta0: jnp.ndarray,
    delta1: jnp.ndarray,
) -> jnp.ndarray:
    slope = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1)
    slope = jnp.where(jnp.sign(slope) != jnp.sign(delta0), 0.0, slope)
    limited = (jnp.sign(delta0) != jnp.sign(delta1)) & (
        jnp.abs(slope) > 3.0 * jnp.abs(delta0)
    )
    return jnp.where(limited, 3.0 * delta0, slope)


def _pchip_slopes_jax(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    widths = jnp.diff(x)
    deltas = jnp.diff(y) / widths
    same_sign = deltas[:-1] * deltas[1:] > 0.0
    weight_left = 2.0 * widths[1:] + widths[:-1]
    weight_right = widths[1:] + 2.0 * widths[:-1]
    safe_left = jnp.where(same_sign, deltas[:-1], 1.0)
    safe_right = jnp.where(same_sign, deltas[1:], 1.0)
    interior = (weight_left + weight_right) / (
        weight_left / safe_left + weight_right / safe_right
    )
    interior = jnp.where(same_sign, interior, 0.0)
    first = _pchip_endpoint_slope_jax(widths[0], widths[1], deltas[0], deltas[1])
    last = _pchip_endpoint_slope_jax(widths[-1], widths[-2], deltas[-1], deltas[-2])
    return jnp.concatenate((first[None], interior, last[None]))


def _validate_matrix(values: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError(f"Expected matrix with shape [n, {len(names)}]")
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains non-finite values")
    return matrix


def _validate_projected_frame(frame: pd.DataFrame) -> None:
    missing = [name for name in SPLINE15D_PARAMETER_NAMES if name not in frame]
    if missing:
        raise ValueError(f"Projected frame is missing columns: {missing}")
    _validate_matrix(
        frame.loc[:, SPLINE15D_PARAMETER_NAMES].to_numpy(),
        SPLINE15D_PARAMETER_NAMES,
    )


def _object_ids(frame: pd.DataFrame) -> np.ndarray:
    if "object_id" in frame:
        return frame["object_id"].to_numpy()
    return frame.index.to_numpy(dtype=np.int64)
