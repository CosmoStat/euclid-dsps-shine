"""Invertible per-parameter transforms for supervised prior benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ATOM_PARAMETER_NAMES = (
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
)


def load_marginal_transforms(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and minimally validate a marginal-transform specification."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    transforms = payload.get("transforms")
    if not isinstance(transforms, dict) or not transforms:
        raise ValueError(f"No transforms mapping found in {path}")
    return transforms


def forward_marginal(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Map one physical coordinate to its normalized coordinate."""
    values = np.asarray(values, dtype=np.float64)
    family = str(spec["family"])
    if family == "mixed_atom_continuous":
        return forward_marginal(values, spec["continuous_transform"])
    if family == "affine":
        return (values - spec["center"]) / spec["scale"]
    if family == "wide_bound_logit":
        unit = (values - spec["lower"]) / (spec["upper"] - spec["lower"])
        if np.any((unit <= 0.0) | (unit >= 1.0)):
            raise ValueError(
                "wide_bound_logit received a value outside its open bounds"
            )
        raw = np.log(unit) - np.log1p(-unit)
        return (raw - spec["center"]) / spec["scale"]
    if family == "asinh":
        raw = spec["lambda"] * np.arcsinh(values / spec["lambda"])
        return (raw - spec["center"]) / spec["scale"]
    if family == "shifted_asinh":
        raw = spec["lambda"] * np.arcsinh((values - spec["shift"]) / spec["lambda"])
        return (raw - spec["center"]) / spec["scale"]
    if family == "quantile_spline":
        return _interpolate_extrapolate(
            values,
            np.asarray(spec["theta_knots"], dtype=np.float64),
            np.asarray(spec["normal_knots"], dtype=np.float64),
        )
    if family == "atom_centered_asinh":
        raw = spec["lambda"] * np.arcsinh(
            (values - spec["atom_value"]) / spec["lambda"]
        )
        return raw / spec["output_scale"]
    raise ValueError(f"Unsupported marginal transform family: {family}")


def inverse_marginal(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Map one normalized coordinate back to physical space."""
    values = np.asarray(values, dtype=np.float64)
    family = str(spec["family"])
    if family == "mixed_atom_continuous":
        return inverse_marginal(values, spec["continuous_transform"])
    if family == "affine":
        return spec["center"] + spec["scale"] * values
    if family == "wide_bound_logit":
        raw = spec["center"] + spec["scale"] * values
        unit = np.exp(-np.logaddexp(0.0, -raw))
        return spec["lower"] + (spec["upper"] - spec["lower"]) * unit
    if family == "asinh":
        raw = spec["center"] + spec["scale"] * values
        return spec["lambda"] * np.sinh(raw / spec["lambda"])
    if family == "shifted_asinh":
        raw = spec["center"] + spec["scale"] * values
        return spec["shift"] + spec["lambda"] * np.sinh(raw / spec["lambda"])
    if family == "quantile_spline":
        return _interpolate_extrapolate(
            values,
            np.asarray(spec["normal_knots"], dtype=np.float64),
            np.asarray(spec["theta_knots"], dtype=np.float64),
        )
    if family == "atom_centered_asinh":
        return spec["atom_value"] + spec["lambda"] * np.sinh(
            spec["output_scale"] * values / spec["lambda"]
        )
    raise ValueError(f"Unsupported marginal transform family: {family}")


def forward_matrix(
    theta: np.ndarray,
    names: tuple[str, ...],
    transforms: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Apply the configured marginal transforms to a physical matrix."""
    theta = _validate_matrix(theta, names)
    x = np.column_stack(
        [
            forward_marginal(theta[:, index], transforms[name])
            for index, name in enumerate(names)
        ]
    )
    if not np.isfinite(x).all():
        raise ValueError("Marginal normalization produced non-finite values")
    return x


def inverse_matrix(
    x: np.ndarray,
    names: tuple[str, ...],
    transforms: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Invert the configured marginal transforms for a normalized matrix."""
    x = _validate_matrix(x, names)
    theta = np.column_stack(
        [
            inverse_marginal(x[:, index], transforms[name])
            for index, name in enumerate(names)
        ]
    )
    if not np.isfinite(theta).all():
        raise ValueError("Inverse marginal normalization produced non-finite values")
    return theta


def shared_atom_mask(
    theta: np.ndarray,
    names: tuple[str, ...],
    transforms: dict[str, dict[str, Any]],
) -> np.ndarray:
    """Return the exact shared four-parameter Diffstar atom mask."""
    theta = _validate_matrix(theta, names)
    masks = []
    for name in ATOM_PARAMETER_NAMES:
        if name not in names:
            raise ValueError(f"Hybrid normalization requires {name}")
        spec = transforms[name]
        if spec.get("family") not in {"mixed_atom_continuous", "atom_centered_asinh"}:
            raise ValueError(f"Transform for {name} does not define an exact atom")
        values = theta[:, names.index(name)]
        masks.append(values == float(spec["atom_value"]))
    reference = masks[0]
    if any(not np.array_equal(reference, mask) for mask in masks[1:]):
        raise ValueError("The four Diffstar atom masks are not identical")
    return reference


def non_atom_names(names: tuple[str, ...]) -> tuple[str, ...]:
    """Return coordinates modeled by the atom-branch continuous flow."""
    atom_names = set(ATOM_PARAMETER_NAMES)
    return tuple(name for name in names if name not in atom_names)


def _validate_matrix(values: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"Expected matrix with shape [n, {len(names)}]")
    return values


def _interpolate_extrapolate(
    values: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    if len(source) < 2 or np.any(np.diff(source) <= 0.0):
        raise ValueError("Spline knots must be strictly increasing")
    result = np.interp(values, source, target)
    low_slope = (target[1] - target[0]) / (source[1] - source[0])
    high_slope = (target[-1] - target[-2]) / (source[-1] - source[-2])
    low = values < source[0]
    high = values > source[-1]
    result[low] = target[0] + low_slope * (values[low] - source[0])
    result[high] = target[-1] + high_slope * (values[high] - source[-1])
    return result
