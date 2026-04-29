"""Single-galaxy fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .io import GalaxyObservation
from .likelihood import chi2_mag
from .model import DspsContext, ModelResult, run_dsps_model


@dataclass(frozen=True)
class FitResult:
    success: bool
    message: str
    best_parameters: dict[str, float]
    chi2: float
    n_bands: int
    trace: list[dict[str, float]]
    model_result: ModelResult


def fit_one_galaxy(
    context: DspsContext,
    observation: GalaxyObservation,
    base_params: dict[str, float],
    fit_config: dict[str, Any],
) -> FitResult:
    """Fit configured DSPS parameters with scipy.optimize."""
    free = fit_config["free_parameters"]
    names = list(free)
    x0 = np.asarray([_initial_value(free[name], name, base_params) for name in names], dtype=float)
    bounds = [tuple(float(x) for x in free[name]["bounds"]) for name in names]
    trace: list[dict[str, float]] = []

    def unpack(x: np.ndarray) -> dict[str, float]:
        params = dict(base_params)
        params.update({name: float(value) for name, value in zip(names, x)})
        return params

    def objective(x: np.ndarray) -> float:
        params = unpack(x)
        try:
            result = run_dsps_model(context, params)
            value = chi2_mag(observation, result)
        except Exception:
            value = 1e30
        entry = {name: float(value) for name, value in zip(names, x)}
        entry["chi2"] = float(value)
        trace.append(entry)
        return float(value)

    opt = minimize(
        objective,
        x0,
        method=fit_config.get("method", "L-BFGS-B"),
        bounds=bounds,
        options={"maxiter": int(fit_config.get("maxiter", 80))},
    )
    best_params = unpack(opt.x)
    best_model = run_dsps_model(context, best_params)
    return FitResult(
        success=bool(opt.success),
        message=str(opt.message),
        best_parameters=best_params,
        chi2=chi2_mag(observation, best_model),
        n_bands=len(observation.bands),
        trace=trace,
        model_result=best_model,
    )


def _initial_value(spec: dict[str, Any], name: str, base_params: dict[str, float]) -> float:
    """Allow YAML configs to use `initial: from_base` for row-dependent values."""
    value = spec.get("initial", base_params.get(name, 0.0))
    if isinstance(value, str):
        if value != "from_base":
            raise ValueError(f"Unsupported initial value for {name}: {value}")
        value = base_params[name]
    return float(value)
