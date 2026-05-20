"""Single-galaxy fitting helpers."""

# ruff: noqa: I001, E402

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize as jax_minimize

from .io import GalaxyObservation
from .model import DspsContext, ModelResult, model_mags_jax, run_dsps_model


_FIT_DTYPE = jnp.float32


@dataclass(frozen=True)
class FitResult:
    success: bool
    message: str
    best_parameters: dict[str, float]
    chi2: float
    n_bands: int
    trace: list[dict[str, float]]
    model_result: ModelResult
    gradient_norm: float


@dataclass(frozen=True)
class BatchFitResult:
    success: np.ndarray
    message: str
    parameter_names: list[str]
    free_parameter_names: list[str]
    best_parameter_matrix: np.ndarray
    chi2: np.ndarray
    gradient_norm: np.ndarray
    model_mags: np.ndarray
    trace: list[dict[str, float]]
    device: str


@dataclass(frozen=True)
class PopulationFitResult:
    batch: BatchFitResult
    hyper_mu: dict[str, float]
    hyper_sigma: dict[str, float]
    hyper_relations: list[dict[str, Any]]
    loss: float


def fit_one_galaxy(
    context: DspsContext,
    observation: GalaxyObservation,
    base_params: dict[str, float],
    fit_config: dict[str, Any],
) -> FitResult:
    """Fit configured DSPS parameters with pure-JAX gradients."""
    method = str(fit_config.get("method", "jax_adam")).lower()
    if method in {"jax_adam", "jax_adam_vmap"}:
        observed_mag, sigma_mag, _ = _observation_arrays(observation)
        batch = fit_galaxy_batch_adam(
            context=context,
            base_params_rows=[base_params],
            observed_mag=np.asarray(observed_mag)[None, :],
            sigma_mag=np.asarray(sigma_mag)[None, :],
            fit_config=fit_config,
        )
        best_params = {
            name: float(batch.best_parameter_matrix[0, index])
            for index, name in enumerate(batch.parameter_names)
        }
        best_model = run_dsps_model(context, best_params)
        return FitResult(
            success=bool(batch.success[0]),
            message=batch.message,
            best_parameters=best_params,
            chi2=float(batch.chi2[0]),
            n_bands=len(observation.bands),
            trace=batch.trace,
            model_result=best_model,
            gradient_norm=float(batch.gradient_norm[0]),
        )

    free = fit_config["free_parameters"]
    names = list(free)
    x0 = jnp.asarray(
        [_initial_value(free[name], name, base_params) for name in names], dtype=float
    )
    bounds = jnp.asarray(
        [tuple(float(x) for x in free[name]["bounds"]) for name in names], dtype=float
    )
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    x0 = jnp.clip(x0, lower, upper)
    base_parameter_names = list(base_params)
    prior_arrays = _prepare_prior_arrays(
        fit_config.get("priors", {}),
        names,
        base_parameter_names,
        np.asarray(
            [[float(base_params[name]) for name in base_parameter_names]], dtype=float
        ),
        np.asarray(bounds, dtype=float),
    )
    prior_arrays_jax = {
        key: jnp.asarray(value[0]) for key, value in prior_arrays.items()
    }
    observed_mag, sigma_mag, finite_mask = _observation_arrays(observation)
    maxiter = int(fit_config.get("maxiter", 80))
    learning_rate = float(fit_config.get("learning_rate", 0.03))
    tolerance = float(fit_config.get("tolerance", 1.0e-5))
    patience = int(fit_config.get("patience", 12))
    trace: list[dict[str, float]] = []

    def unpack_jax(x: jnp.ndarray) -> dict[str, Any]:
        params: dict[str, Any] = {
            key: jnp.asarray(value) for key, value in base_params.items()
        }
        params.update({name: value for name, value in zip(names, x, strict=True)})
        return params

    def objective(x: jnp.ndarray) -> jnp.ndarray:
        params = unpack_jax(x)
        model_mag = model_mags_jax(context, params)
        chi = jnp.where(finite_mask, (observed_mag - model_mag) / sigma_mag, 0.0)
        chi2 = jnp.sum(chi**2)
        prior = _physical_prior_penalty(
            x,
            lower,
            upper,
            prior_arrays_jax["prior_gaussian_mask"],
            prior_arrays_jax["prior_gaussian_loc"],
            prior_arrays_jax["prior_gaussian_scale"],
            prior_arrays_jax["prior_beta_mask"],
            prior_arrays_jax["prior_beta_alpha"],
            prior_arrays_jax["prior_beta_beta"],
        )
        objective_value = chi2 + float(fit_config.get("prior_weight", 1.0)) * prior
        return jnp.nan_to_num(objective_value, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    if method == "jax_adam":
        best_x, state, best_value, best_grad_norm, success, message = _fit_bounded_adam(
            value_and_grad=value_and_grad,
            x0=x0,
            lower=lower,
            upper=upper,
            maxiter=maxiter,
            learning_rate=learning_rate,
            tolerance=tolerance,
            patience=patience,
            names=names,
            trace=trace,
        )
    elif method == "jax_bfgs":
        best_x, state, best_value, best_grad_norm, success, message = _fit_bounded_bfgs(
            objective=objective,
            value_and_grad=value_and_grad,
            x0=x0,
            lower=lower,
            upper=upper,
            maxiter=maxiter,
            tolerance=tolerance,
            names=names,
            trace=trace,
        )
    else:
        raise ValueError(f"Unsupported fit.method: {fit_config.get('method')}")

    best_params = _unpack_numpy(best_x, names, base_params)
    best_model = run_dsps_model(context, best_params)
    return FitResult(
        success=success,
        message=message,
        best_parameters=best_params,
        chi2=float(best_value),
        n_bands=len(observation.bands),
        trace=trace,
        model_result=best_model,
        gradient_norm=float(best_grad_norm),
    )


def _initial_value(
    spec: dict[str, Any], name: str, base_params: dict[str, float]
) -> float:
    """Allow YAML configs to use `initial: from_base` for row-dependent values."""
    value = spec.get("initial", base_params.get(name, 0.0))
    if isinstance(value, str):
        if value != "from_base":
            raise ValueError(f"Unsupported initial value for {name}: {value}")
        value = base_params[name]
    return float(value)


def _observation_arrays(
    observation: GalaxyObservation,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    observed_mag = jnp.asarray([band.mag_ab for band in observation.bands], dtype=float)
    sigma_mag = jnp.asarray([band.sigma_mag for band in observation.bands], dtype=float)
    finite_mask = jnp.isfinite(observed_mag) & jnp.isfinite(sigma_mag) & (sigma_mag > 0)
    return observed_mag, sigma_mag, finite_mask


def _fit_bounded_adam(
    value_and_grad,
    x0: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    maxiter: int,
    learning_rate: float,
    tolerance: float,
    patience: int,
    names: list[str],
    trace: list[dict[str, float]],
):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    x = x0
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    best_x = x
    best_value = np.inf
    best_grad_norm = np.inf
    stalled = 0

    for iteration in range(1, maxiter + 1):
        value, grad = value_and_grad(x)
        value_f = float(value)
        grad_norm = float(jnp.linalg.norm(grad))
        trace.append(_trace_entry(names, x, value_f, grad_norm, iteration))

        if value_f + tolerance < best_value:
            best_x = x
            best_value = value_f
            best_grad_norm = grad_norm
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break

        grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        x = jnp.clip(x - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps), lower, upper)

    state = {"iteration": iteration, "x": x}
    success = bool(np.isfinite(best_value)) and stalled < patience
    message = (
        f"jax_adam converged: chi2 improvement < {tolerance} for {patience} steps"
        if success
        else f"jax_adam stopped after {state['iteration']} iterations"
    )
    return best_x, state, best_value, best_grad_norm, success, message


def _fit_bounded_bfgs(
    objective,
    value_and_grad,
    x0: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    maxiter: int,
    tolerance: float,
    names: list[str],
    trace: list[dict[str, float]],
):
    y0 = _bounded_to_unconstrained(x0, lower, upper)

    def unconstrained_objective(y: jnp.ndarray) -> jnp.ndarray:
        return objective(_unconstrained_to_bounded(y, lower, upper))

    initial_value, initial_grad = value_and_grad(x0)
    trace.append(
        _trace_entry(
            names, x0, float(initial_value), float(jnp.linalg.norm(initial_grad)), 0
        )
    )
    opt = jax_minimize(
        jax.jit(unconstrained_objective),
        y0,
        method="BFGS",
        tol=tolerance,
        options={"maxiter": maxiter},
    )
    best_x = _unconstrained_to_bounded(opt.x, lower, upper)
    best_value, best_grad = value_and_grad(best_x)
    best_value_f = float(best_value)
    best_grad_norm = float(jnp.linalg.norm(best_grad))
    trace.append(
        _trace_entry(
            names,
            best_x,
            best_value_f,
            best_grad_norm,
            int(getattr(opt, "nit", maxiter)),
        )
    )
    success = bool(getattr(opt, "success", False)) and np.isfinite(best_value_f)
    message = f"jax_bfgs status={int(getattr(opt, 'status', -1))}, nit={int(getattr(opt, 'nit', -1))}"
    state = {"iteration": int(getattr(opt, "nit", -1)), "x": best_x}
    return best_x, state, best_value_f, best_grad_norm, success, message


def _bounded_to_unconstrained(
    x: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray
) -> jnp.ndarray:
    eps = 1.0e-6
    scaled = jnp.clip((x - lower) / (upper - lower), eps, 1.0 - eps)
    return jnp.log(scaled) - jnp.log1p(-scaled)


def _unconstrained_to_bounded(
    y: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray
) -> jnp.ndarray:
    return lower + (upper - lower) * jax.nn.sigmoid(y)


def _trace_entry(
    names: list[str], x: jnp.ndarray, chi2: float, grad_norm: float, iteration: int
) -> dict[str, float]:
    entry = {
        name: float(value) for name, value in zip(names, np.asarray(x), strict=True)
    }
    entry["iteration"] = float(iteration)
    entry["chi2"] = float(chi2)
    entry["gradient_norm"] = float(grad_norm)
    return entry


def _unpack_numpy(
    x: jnp.ndarray, names: list[str], base_params: dict[str, float]
) -> dict[str, float]:
    params = dict(base_params)
    params.update(
        {name: float(value) for name, value in zip(names, np.asarray(x), strict=True)}
    )
    return params


_OPTIMIZER_CACHE = {}


def fit_galaxy_batch_adam(
    context: DspsContext,
    base_params_rows: list[dict[str, float]],
    observed_mag: np.ndarray,
    sigma_mag: np.ndarray,
    fit_config: dict[str, Any],
    truth_theta: np.ndarray | None = None,
) -> BatchFitResult:
    """Fit many independent galaxies in one JAX-vmapped Adam run."""
    setup = _prepare_batch_fit(base_params_rows, fit_config)
    truth_theta_arr, has_truth = _prepare_truth_theta(setup["theta0"], truth_theta)
    theta0 = jnp.asarray(setup["theta0"], dtype=_FIT_DTYPE)
    base_matrix = jnp.asarray(setup["base_matrix"], dtype=_FIT_DTYPE)
    observed = jnp.asarray(observed_mag, dtype=_FIT_DTYPE)
    sigma = jnp.asarray(sigma_mag, dtype=_FIT_DTYPE)
    mask = jnp.isfinite(observed) & jnp.isfinite(sigma) & (sigma > 0)
    lower = jnp.asarray(setup["lower"], dtype=_FIT_DTYPE)
    upper = jnp.asarray(setup["upper"], dtype=_FIT_DTYPE)
    maxiter = int(fit_config.get("maxiter", 80))
    learning_rate = float(fit_config.get("learning_rate", 0.03))
    prior_weight = float(fit_config.get("prior_weight", 1.0))
    fast_warmstart_only = bool(fit_config.get("fast_warmstart_only", False))
    fast_grid_search = bool(fit_config.get("fast_grid_search", False))
    redshift_grid_size = int(fit_config.get("redshift_grid_size", 7))
    redshift_grid_width = float(fit_config.get("redshift_grid_width", 0.4))
    fast_grid_parameters = tuple(fit_config.get("fast_grid_parameters", ()))
    fast_grid_prior_width = float(fit_config.get("fast_grid_prior_width", 1.0))

    cache_key = (
        (
            "independent_warmstart"
            if fast_warmstart_only
            else "independent_grid"
            if fast_grid_search
            else "independent"
        ),
        id(context),
        tuple(setup["parameter_names"]),
        tuple(setup["free_names"]),
        0 if (fast_warmstart_only or fast_grid_search) else maxiter,
        0.0 if (fast_warmstart_only or fast_grid_search) else learning_rate,
        0.0 if fast_warmstart_only else prior_weight,
        redshift_grid_size if fast_grid_search else 0,
        redshift_grid_width if fast_grid_search else 0.0,
        fast_grid_parameters if fast_grid_search else (),
        fast_grid_prior_width if fast_grid_search else 0.0,
    )
    if cache_key not in _OPTIMIZER_CACHE:
        if fast_warmstart_only:
            _OPTIMIZER_CACHE[cache_key] = _build_independent_warmstart_optimizer(
                context=context,
                parameter_names=setup["parameter_names"],
                free_indices=setup["free_indices"],
            )
        elif fast_grid_search:
            _OPTIMIZER_CACHE[cache_key] = _build_independent_grid_optimizer(
                context=context,
                parameter_names=setup["parameter_names"],
                free_indices=setup["free_indices"],
                prior_weight=prior_weight,
                redshift_grid_size=redshift_grid_size,
                redshift_grid_width=redshift_grid_width,
                fast_grid_parameters=fast_grid_parameters,
                fast_grid_prior_width=fast_grid_prior_width,
            )
        else:
            _OPTIMIZER_CACHE[cache_key] = _build_independent_adam_optimizer(
                context=context,
                parameter_names=setup["parameter_names"],
                free_indices=setup["free_indices"],
                maxiter=maxiter,
                learning_rate=learning_rate,
                prior_weight=prior_weight,
            )
    optimize = _OPTIMIZER_CACHE[cache_key]

    best_theta, chi2, grad_norm, model_mags, trace_arrays = optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        jnp.asarray(setup["prior_gaussian_mask"]),
        jnp.asarray(setup["prior_gaussian_loc"], dtype=_FIT_DTYPE),
        jnp.asarray(setup["prior_gaussian_scale"], dtype=_FIT_DTYPE),
        jnp.asarray(setup["prior_beta_mask"]),
        jnp.asarray(setup["prior_beta_alpha"], dtype=_FIT_DTYPE),
        jnp.asarray(setup["prior_beta_beta"], dtype=_FIT_DTYPE),
        jnp.asarray(truth_theta_arr, dtype=_FIT_DTYPE),
    )
    best_matrix = _apply_free_values(
        base_matrix, best_theta, jnp.asarray(setup["free_indices"])
    )
    trace = _batch_trace_from_arrays(
        trace_arrays, setup["free_names"], include_truth_metrics=has_truth
    )
    return BatchFitResult(
        success=np.isfinite(np.asarray(chi2)),
        message=(
            f"jax_warmstart_only device={_jax_device()}"
            if fast_warmstart_only
            else (
                f"jax_grid_warmstart n_grid={redshift_grid_size}, "
                f"params={','.join(fast_grid_parameters)}, device={_jax_device()}"
            )
            if fast_grid_search
            else f"jax_adam_vmap maxiter={maxiter}, device={_jax_device()}"
        ),
        parameter_names=setup["parameter_names"],
        free_parameter_names=setup["free_names"],
        best_parameter_matrix=np.asarray(best_matrix),
        chi2=np.asarray(chi2),
        gradient_norm=np.asarray(grad_norm),
        model_mags=np.asarray(model_mags),
        trace=trace,
        device=_jax_device(),
    )


def fit_population_batch_adam(
    context: DspsContext,
    base_params_rows: list[dict[str, float]],
    observed_mag: np.ndarray,
    sigma_mag: np.ndarray,
    fit_config: dict[str, Any],
    initial_theta: np.ndarray | None = None,
    truth_theta: np.ndarray | None = None,
) -> PopulationFitResult:
    """Joint MAP fit with a Gaussian population prior over free parameters."""
    setup = _prepare_batch_fit(
        base_params_rows, fit_config, initial_theta=initial_theta
    )
    truth_theta_arr, has_truth = _prepare_truth_theta(setup["theta0"], truth_theta)
    theta0 = jnp.asarray(setup["theta0"])
    base_matrix = jnp.asarray(setup["base_matrix"])
    observed = jnp.asarray(observed_mag)
    sigma = jnp.asarray(sigma_mag)
    mask = jnp.isfinite(observed) & jnp.isfinite(sigma) & (sigma > 0)
    lower = jnp.asarray(setup["lower"])
    upper = jnp.asarray(setup["upper"])
    maxiter = int(fit_config.get("maxiter", 80))
    learning_rate = float(fit_config.get("learning_rate", 0.03))
    pop = fit_config.get("population", {})

    sigma_floor = float(pop.get("sigma_floor", 0.03))
    prior_weight = float(pop.get("prior_weight", 1.0))
    hyper_mu_scale = float(pop.get("hyper_mu_scale", 5.0))
    physical_prior_weight = float(fit_config.get("prior_weight", 1.0))

    cache_key = (
        "population",
        id(context),
        tuple(setup["parameter_names"]),
        tuple(setup["free_names"]),
        maxiter,
        learning_rate,
        sigma_floor,
        prior_weight,
        hyper_mu_scale,
        physical_prior_weight,
        tuple(setup["population_relation_default_mask"]),
        tuple(setup["population_relation_target_pos"]),
        tuple(setup["population_relation_predictor_free_pos"]),
        tuple(setup["population_relation_predictor_base_index"]),
    )

    if cache_key not in _OPTIMIZER_CACHE:
        _OPTIMIZER_CACHE[cache_key] = _build_population_adam_optimizer(
            context=context,
            parameter_names=setup["parameter_names"],
            free_indices=setup["free_indices"],
            maxiter=maxiter,
            learning_rate=learning_rate,
            sigma_floor=sigma_floor,
            prior_weight=prior_weight,
            hyper_mu_scale=hyper_mu_scale,
            physical_prior_weight=physical_prior_weight,
            relation_default_mask=jnp.asarray(
                setup["population_relation_default_mask"]
            ),
            relation_target_pos=jnp.asarray(setup["population_relation_target_pos"]),
            relation_predictor_free_pos=jnp.asarray(
                setup["population_relation_predictor_free_pos"]
            ),
            relation_predictor_base_index=jnp.asarray(
                setup["population_relation_predictor_base_index"]
            ),
            relation_pivot=jnp.asarray(setup["population_relation_pivot"]),
            relation_intercept0=jnp.asarray(setup["population_relation_intercept0"]),
            relation_slope0=jnp.asarray(setup["population_relation_slope0"]),
            relation_sigma0=jnp.asarray(setup["population_relation_sigma0"]),
            relation_slope_scale=jnp.asarray(setup["population_relation_slope_scale"]),
        )
    optimize = _OPTIMIZER_CACHE[cache_key]

    (
        best_theta,
        mu,
        sigma_pop,
        relation_intercept,
        relation_slope,
        relation_sigma,
        loss,
        chi2,
        grad_norm,
        model_mags,
        trace_arrays,
    ) = optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        jnp.asarray(setup["prior_gaussian_mask"]),
        jnp.asarray(setup["prior_gaussian_loc"]),
        jnp.asarray(setup["prior_gaussian_scale"]),
        jnp.asarray(setup["prior_beta_mask"]),
        jnp.asarray(setup["prior_beta_alpha"]),
        jnp.asarray(setup["prior_beta_beta"]),
        jnp.asarray(truth_theta_arr),
    )
    best_matrix = _apply_free_values(
        base_matrix, best_theta, jnp.asarray(setup["free_indices"])
    )
    batch = BatchFitResult(
        success=np.isfinite(np.asarray(chi2)),
        message=f"jax_population_adam maxiter={maxiter}, device={_jax_device()}",
        parameter_names=setup["parameter_names"],
        free_parameter_names=setup["free_names"],
        best_parameter_matrix=np.asarray(best_matrix),
        chi2=np.asarray(chi2),
        gradient_norm=np.asarray(grad_norm),
        model_mags=np.asarray(model_mags),
        trace=_batch_trace_from_arrays(
            trace_arrays, setup["free_names"], include_truth_metrics=has_truth
        ),
        device=_jax_device(),
    )
    return PopulationFitResult(
        batch=batch,
        hyper_mu={
            name: float(value)
            for name, value in zip(setup["free_names"], np.asarray(mu), strict=True)
        },
        hyper_sigma={
            name: float(value)
            for name, value in zip(
                setup["free_names"], np.asarray(sigma_pop), strict=True
            )
        },
        hyper_relations=_population_relation_hyper_rows(
            setup,
            np.asarray(relation_intercept),
            np.asarray(relation_slope),
            np.asarray(relation_sigma),
        ),
        loss=float(loss),
    )


def _prepare_batch_fit(
    base_params_rows: list[dict[str, float]],
    fit_config: dict[str, Any],
    initial_theta: np.ndarray | None = None,
) -> dict[str, Any]:
    if not base_params_rows:
        raise ValueError("Cannot fit an empty batch")
    free = fit_config["free_parameters"]
    free_names = list(free)
    parameter_names = _parameter_names_for_rows(base_params_rows)
    missing = [name for name in free_names if name not in parameter_names]
    if missing:
        raise ValueError(f"Free parameters missing from base params: {missing}")
    base_matrix = np.asarray(
        [
            [float(row.get(name, np.nan)) for name in parameter_names]
            for row in base_params_rows
        ],
        dtype=float,
    )
    bounds = np.asarray(
        [tuple(float(x) for x in free[name]["bounds"]) for name in free_names],
        dtype=float,
    )
    if initial_theta is None:
        theta0 = np.asarray(
            [
                [_initial_value(free[name], name, row) for name in free_names]
                for row in base_params_rows
            ],
            dtype=float,
        )
    else:
        theta0 = np.asarray(initial_theta, dtype=float)
        if theta0.shape != (len(base_params_rows), len(free_names)):
            raise ValueError(
                f"initial_theta shape must be {(len(base_params_rows), len(free_names))}, got {theta0.shape}"
            )
    theta0 = np.clip(theta0, bounds[:, 0], bounds[:, 1])
    prior_arrays = _prepare_prior_arrays(
        fit_config.get("priors", {}),
        free_names,
        parameter_names,
        base_matrix,
        bounds,
    )
    relation_arrays = _prepare_population_relation_arrays(
        fit_config.get("population", {}),
        free_names,
        parameter_names,
        base_matrix,
        theta0,
    )
    return {
        "parameter_names": parameter_names,
        "free_names": free_names,
        "free_indices": np.asarray(
            [parameter_names.index(name) for name in free_names], dtype=np.int32
        ),
        "base_matrix": base_matrix,
        "theta0": theta0,
        "lower": bounds[:, 0],
        "upper": bounds[:, 1],
        **prior_arrays,
        **relation_arrays,
    }


def _parameter_names_for_rows(base_params_rows: list[dict[str, float]]) -> list[str]:
    """Build a stable schema when optional per-row prior/catalog keys are missing."""
    names = list(base_params_rows[0])
    seen = set(names)
    for row in base_params_rows[1:]:
        for name in row:
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def _prepare_prior_arrays(
    priors: dict[str, Any],
    free_names: list[str],
    parameter_names: list[str],
    base_matrix: np.ndarray,
    bounds: np.ndarray,
) -> dict[str, np.ndarray]:
    n_rows = base_matrix.shape[0]
    n_free = len(free_names)
    gaussian_mask = np.zeros((n_rows, n_free), dtype=bool)
    gaussian_loc = np.zeros((n_rows, n_free), dtype=float)
    gaussian_scale = np.ones((n_rows, n_free), dtype=float)
    beta_mask = np.zeros((n_rows, n_free), dtype=bool)
    beta_alpha = np.ones((n_rows, n_free), dtype=float)
    beta_beta = np.ones((n_rows, n_free), dtype=float)

    for index, name in enumerate(free_names):
        spec = priors.get(name)
        if not spec:
            continue
        prior_type = str(spec.get("type", "normal"))
        if prior_type == "uniform":
            continue
        if prior_type in {"normal", "truncated_normal"}:
            loc = spec.get("loc", 0.0)
            if loc == "from_base":
                loc_values = base_matrix[:, parameter_names.index(name)]
            else:
                loc_values = np.full(n_rows, float(loc), dtype=float)
            gaussian_mask[:, index] = True
            gaussian_loc[:, index] = loc_values
            scale = spec.get("scale", 1.0)
            if scale == "from_base":
                scale_name = str(spec.get("scale_parameter", f"{name}_prior_sigma"))
                if scale_name not in parameter_names:
                    raise ValueError(
                        f"fit.priors.{name}.scale='from_base' needs parameter "
                        f"{scale_name!r} in base parameters"
                    )
                scale_values = np.asarray(
                    base_matrix[:, parameter_names.index(scale_name)], dtype=float
                )
                scale_values = np.where(
                    np.isfinite(scale_values) & (scale_values > 0),
                    scale_values,
                    1.0,
                )
                gaussian_scale[:, index] = np.maximum(scale_values, 1.0e-6)
            else:
                gaussian_scale[:, index] = max(float(scale), 1.0e-6)
        elif prior_type == "scaled_beta":
            beta_mask[:, index] = True
            beta_alpha[:, index] = max(float(spec.get("alpha", 1.0)), 1.0e-6)
            beta_beta[:, index] = max(float(spec.get("beta", 1.0)), 1.0e-6)
        else:
            raise ValueError(f"Unsupported fit prior type for {name}: {prior_type}")

    return {
        "prior_gaussian_mask": gaussian_mask,
        "prior_gaussian_loc": gaussian_loc,
        "prior_gaussian_scale": gaussian_scale,
        "prior_beta_mask": beta_mask,
        "prior_beta_alpha": beta_alpha,
        "prior_beta_beta": beta_beta,
    }


def _prepare_population_relation_arrays(
    population: dict[str, Any],
    free_names: list[str],
    parameter_names: list[str],
    base_matrix: np.ndarray,
    theta0: np.ndarray,
) -> dict[str, np.ndarray]:
    relations = population.get("relations", {}) if isinstance(population, dict) else {}
    target_pos: list[int] = []
    predictor_free_pos: list[int] = []
    predictor_base_index: list[int] = []
    pivot: list[float] = []
    intercept0: list[float] = []
    slope0: list[float] = []
    sigma0: list[float] = []
    slope_scale: list[float] = []
    default_mask = np.ones(len(free_names), dtype=float)
    relation_names: list[tuple[str, str]] = []

    if not isinstance(relations, dict):
        relations = {}
    for target_name, spec in relations.items():
        if not spec or not bool(spec.get("enabled", True)):
            continue
        if target_name not in free_names:
            raise ValueError(
                f"fit.population.relations target {target_name!r} must be a free parameter"
            )
        predictor_name = str(spec.get("predictor", ""))
        if predictor_name not in parameter_names:
            raise ValueError(
                f"fit.population.relations.{target_name}.predictor={predictor_name!r} "
                "must exist in model parameters"
            )
        target_index = free_names.index(target_name)
        target_pos.append(target_index)
        default_mask[target_index] = 0.0
        predictor_base_index.append(parameter_names.index(predictor_name))
        predictor_free_pos.append(
            free_names.index(predictor_name) if predictor_name in free_names else -1
        )
        predictor_values = base_matrix[:, parameter_names.index(predictor_name)]
        raw_pivot = spec.get("pivot", "median")
        pivot.append(
            float(np.nanmedian(predictor_values))
            if raw_pivot == "median"
            else float(raw_pivot)
        )
        target_values = theta0[:, target_index]
        intercept0.append(
            float(spec.get("intercept_initial", np.nanmean(target_values)))
        )
        slope0.append(float(spec.get("slope_initial", 0.0)))
        sigma0.append(
            max(
                float(spec.get("sigma_initial", np.nanstd(target_values) + 0.2)), 1.0e-3
            )
        )
        slope_scale.append(max(float(spec.get("slope_scale", 2.0)), 1.0e-6))
        relation_names.append((str(target_name), predictor_name))

    return {
        "population_relation_default_mask": default_mask,
        "population_relation_target_pos": np.asarray(target_pos, dtype=np.int32),
        "population_relation_predictor_free_pos": np.asarray(
            predictor_free_pos, dtype=np.int32
        ),
        "population_relation_predictor_base_index": np.asarray(
            predictor_base_index, dtype=np.int32
        ),
        "population_relation_pivot": np.asarray(pivot, dtype=float),
        "population_relation_intercept0": np.asarray(intercept0, dtype=float),
        "population_relation_slope0": np.asarray(slope0, dtype=float),
        "population_relation_sigma0": np.asarray(sigma0, dtype=float),
        "population_relation_slope_scale": np.asarray(slope_scale, dtype=float),
        "population_relation_names": np.asarray(relation_names, dtype=object),
    }


def _population_relation_hyper_rows(
    setup: dict[str, Any],
    intercept: np.ndarray,
    slope: np.ndarray,
    sigma: np.ndarray,
) -> list[dict[str, Any]]:
    names = np.asarray(setup["population_relation_names"], dtype=object)
    pivots = np.asarray(setup["population_relation_pivot"], dtype=float)
    rows: list[dict[str, Any]] = []
    for index in range(len(intercept)):
        target, predictor = names[index]
        rows.append(
            {
                "target_parameter": str(target),
                "predictor_parameter": str(predictor),
                "pivot": float(pivots[index]),
                "intercept": float(intercept[index]),
                "slope": float(slope[index]),
                "sigma": float(sigma[index]),
            }
        )
    return rows


def _prepare_truth_theta(
    theta0: np.ndarray, truth_theta: np.ndarray | None
) -> tuple[np.ndarray, bool]:
    if truth_theta is None:
        return np.full_like(theta0, np.nan, dtype=float), False
    truth = np.asarray(truth_theta, dtype=float)
    if truth.shape != theta0.shape:
        raise ValueError(f"truth_theta shape must be {theta0.shape}, got {truth.shape}")
    return truth, bool(np.isfinite(truth).any())


def _build_independent_adam_optimizer(
    context: DspsContext,
    parameter_names: list[str],
    free_indices: np.ndarray,
    maxiter: int,
    learning_rate: float,
    prior_weight: float,
):
    free_indices_jax = jnp.asarray(free_indices)

    def single_chi2(theta, base, observed, sigma, mask):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        model_mag = model_mags_jax(context, params)
        chi = jnp.where(mask, (observed - model_mag) / sigma, 0.0)
        return jnp.nan_to_num(jnp.sum(chi**2), nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    def single_objective(
        theta,
        base,
        observed,
        sigma,
        mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
    ):
        chi2 = single_chi2(theta, base, observed, sigma, mask)
        prior = _physical_prior_penalty(
            theta,
            lower,
            upper,
            prior_gaussian_mask,
            prior_gaussian_loc,
            prior_gaussian_scale,
            prior_beta_mask,
            prior_beta_alpha,
            prior_beta_beta,
        )
        return jnp.nan_to_num(
            chi2 + prior_weight * prior,
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )

    def single_mags(theta, base):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        return model_mags_jax(context, params)

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0))
    amplitude_free_pos = _free_position(
        parameter_names, free_indices, "log10_formed_mass_msun"
    )
    if amplitude_free_pos is None:
        amplitude_free_pos = _free_position(parameter_names, free_indices, "log10_sfr")

    @jax.jit
    def optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
        truth_theta,
    ):
        theta0 = _warm_start_amplitude(
            theta0,
            base_matrix,
            observed,
            mask,
            lower,
            upper,
            batch_mags,
            amplitude_free_pos,
        )
        y0 = _bounded_to_unconstrained(theta0, lower, upper)
        m0 = jnp.zeros_like(y0)
        v0 = jnp.zeros_like(y0)
        best_objective0 = jnp.full((theta0.shape[0],), jnp.inf)
        best_grad0 = jnp.full((theta0.shape[0],), jnp.inf)
        carry0 = (y0, m0, v0, theta0, best_objective0, best_grad0)

        def single_objective_y(
            y,
            base,
            observed_i,
            sigma_i,
            mask_i,
            prior_gaussian_mask_i,
            prior_gaussian_loc_i,
            prior_gaussian_scale_i,
            prior_beta_mask_i,
            prior_beta_alpha_i,
            prior_beta_beta_i,
        ):
            theta = _unconstrained_to_bounded(y, lower, upper)
            return single_objective(
                theta,
                base,
                observed_i,
                sigma_i,
                mask_i,
                lower,
                upper,
                prior_gaussian_mask_i,
                prior_gaussian_loc_i,
                prior_gaussian_scale_i,
                prior_beta_mask_i,
                prior_beta_alpha_i,
                prior_beta_beta_i,
            )

        batch_objective_y_grad = jax.vmap(
            jax.value_and_grad(single_objective_y, argnums=0),
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

        def step(carry, iteration):
            y, m, v, best_theta, best_objective, best_grad = carry
            theta = _unconstrained_to_bounded(y, lower, upper)
            objective, grad = batch_objective_y_grad(
                y,
                base_matrix,
                observed,
                sigma,
                mask,
                prior_gaussian_mask,
                prior_gaussian_loc,
                prior_gaussian_scale,
                prior_beta_mask,
                prior_beta_alpha,
                prior_beta_beta,
            )
            grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = jnp.linalg.norm(grad, axis=1)
            improved = objective < best_objective
            best_theta = jnp.where(improved[:, None], theta, best_theta)
            best_objective = jnp.where(improved, objective, best_objective)
            best_grad = jnp.where(improved, grad_norm, best_grad)

            t = iteration + 1.0
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad**2)
            m_hat = m / (1.0 - 0.9**t)
            v_hat = v / (1.0 - 0.999**t)
            y = jnp.clip(
                y - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1.0e-8), -30.0, 30.0
            )
            chi2 = jax.vmap(single_chi2, in_axes=(0, 0, 0, 0, 0))(
                theta, base_matrix, observed, sigma, mask
            )
            metrics = jnp.concatenate(
                [
                    jnp.asarray(
                        [jnp.nanmean(chi2), jnp.nanmedian(chi2), jnp.nanmean(grad_norm)]
                    ),
                    _truth_metric_vector(theta, truth_theta),
                ]
            )
            return (y, m, v, best_theta, best_objective, best_grad), metrics

        (_, _, _, best_theta, _, best_grad), metrics = jax.lax.scan(
            step,
            carry0,
            jnp.arange(maxiter, dtype=jnp.int32),
        )
        best_chi2 = jax.vmap(single_chi2, in_axes=(0, 0, 0, 0, 0))(
            best_theta, base_matrix, observed, sigma, mask
        )
        model_mags = batch_mags(best_theta, base_matrix)
        return best_theta, best_chi2, best_grad, model_mags, metrics

    return optimize


def _build_independent_warmstart_optimizer(
    context: DspsContext,
    parameter_names: list[str],
    free_indices: np.ndarray,
):
    free_indices_jax = jnp.asarray(free_indices)

    def single_mags(theta, base):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        return model_mags_jax(context, params)

    def single_chi2(theta, base, observed, sigma, mask):
        model_mag = single_mags(theta, base)
        chi = jnp.where(mask, (observed - model_mag) / sigma, 0.0)
        return jnp.nan_to_num(jnp.sum(chi**2), nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0))
    amplitude_free_pos = _free_position(
        parameter_names, free_indices, "log10_formed_mass_msun"
    )
    if amplitude_free_pos is None:
        amplitude_free_pos = _free_position(parameter_names, free_indices, "log10_sfr")

    @jax.jit
    def optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
        truth_theta,
    ):
        del (
            prior_gaussian_mask,
            prior_gaussian_loc,
            prior_gaussian_scale,
            prior_beta_mask,
            prior_beta_alpha,
            prior_beta_beta,
        )
        best_theta = _warm_start_amplitude(
            theta0,
            base_matrix,
            observed,
            mask,
            lower,
            upper,
            batch_mags,
            amplitude_free_pos,
        )
        best_chi2 = jax.vmap(single_chi2, in_axes=(0, 0, 0, 0, 0))(
            best_theta, base_matrix, observed, sigma, mask
        )
        model_mags = batch_mags(best_theta, base_matrix)
        grad_norm = jnp.zeros((best_theta.shape[0],), dtype=best_theta.dtype)
        metrics = jnp.concatenate(
            [
                jnp.asarray(
                    [jnp.nanmean(best_chi2), jnp.nanmedian(best_chi2), 0.0],
                    dtype=best_theta.dtype,
                ),
                _truth_metric_vector(best_theta, truth_theta),
            ]
        )[None, :]
        return best_theta, best_chi2, grad_norm, model_mags, metrics

    return optimize


def _build_independent_grid_optimizer(
    context: DspsContext,
    parameter_names: list[str],
    free_indices: np.ndarray,
    prior_weight: float,
    redshift_grid_size: int,
    redshift_grid_width: float,
    fast_grid_parameters: tuple[str, ...],
    fast_grid_prior_width: float,
):
    free_indices_jax = jnp.asarray(free_indices)
    n_grid = max(int(redshift_grid_size), 1)
    grid_free_positions = tuple(
        pos
        for name in fast_grid_parameters
        if (pos := _free_position(parameter_names, free_indices, name)) is not None
    )
    z_free_pos = _free_position(parameter_names, free_indices, "z_obs")
    amplitude_free_pos = _free_position(
        parameter_names, free_indices, "log10_formed_mass_msun"
    )
    if amplitude_free_pos is None:
        amplitude_free_pos = _free_position(parameter_names, free_indices, "log10_sfr")

    def single_mags(theta, base):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        return model_mags_jax(context, params)

    def single_chi2(theta, base, observed, sigma, mask):
        model_mag = single_mags(theta, base)
        chi = jnp.where(mask, (observed - model_mag) / sigma, 0.0)
        return jnp.nan_to_num(jnp.sum(chi**2), nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    def single_objective(
        theta,
        base,
        observed,
        sigma,
        mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
    ):
        chi2 = single_chi2(theta, base, observed, sigma, mask)
        prior = _physical_prior_penalty(
            theta,
            lower,
            upper,
            prior_gaussian_mask,
            prior_gaussian_loc,
            prior_gaussian_scale,
            prior_beta_mask,
            prior_beta_alpha,
            prior_beta_beta,
        )
        return jnp.nan_to_num(
            chi2 + prior_weight * prior,
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0))
    batch_chi2 = jax.vmap(single_chi2, in_axes=(0, 0, 0, 0, 0))
    batch_objective = jax.vmap(
        single_objective,
        in_axes=(0, 0, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0),
    )

    @jax.jit
    def optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
        truth_theta,
    ):
        n_rows = theta0.shape[0]
        frac_values = jnp.linspace(0.0, 1.0, n_grid, dtype=theta0.dtype)
        best_theta = theta0
        best_objective = jnp.full((n_rows,), jnp.inf, dtype=theta0.dtype)

        def evaluate(theta):
            theta = _warm_start_amplitude(
                theta,
                base_matrix,
                observed,
                mask,
                lower,
                upper,
                batch_mags,
                amplitude_free_pos,
            )
            objective = batch_objective(
                theta,
                base_matrix,
                observed,
                sigma,
                mask,
                lower,
                upper,
                prior_gaussian_mask,
                prior_gaussian_loc,
                prior_gaussian_scale,
                prior_beta_mask,
                prior_beta_alpha,
                prior_beta_beta,
            )
            return objective.astype(theta0.dtype), theta

        best_objective, best_theta = evaluate(best_theta)

        def scan_one_parameter(best_objective, best_theta, free_pos: int):
            lo, hi = _fast_grid_bounds(
                best_theta,
                free_pos,
                lower,
                upper,
                prior_gaussian_mask,
                prior_gaussian_loc,
                prior_gaussian_scale,
                redshift_grid_width,
                fast_grid_prior_width,
                z_free_pos,
            )

            def grid_step(carry, frac):
                current_best_objective, current_best_theta = carry
                candidate = current_best_theta
                value = lo + frac * (hi - lo)
                candidate = candidate.at[:, free_pos].set(value.astype(theta0.dtype))
                objective, candidate = evaluate(candidate)
                improved = objective < current_best_objective
                current_best_objective = jnp.where(
                    improved, objective, current_best_objective
                )
                current_best_theta = jnp.where(
                    improved[:, None], candidate, current_best_theta
                )
                return (current_best_objective, current_best_theta), None

            (best_objective, best_theta), _ = jax.lax.scan(
                grid_step,
                (best_objective, best_theta),
                frac_values,
            )
            return best_objective, best_theta

        for free_pos in grid_free_positions:
            best_objective, best_theta = scan_one_parameter(
                best_objective, best_theta, free_pos
            )
        best_chi2 = batch_chi2(best_theta, base_matrix, observed, sigma, mask)
        model_mags = batch_mags(best_theta, base_matrix)
        grad_norm = jnp.zeros((n_rows,), dtype=theta0.dtype)
        metrics = jnp.concatenate(
            [
                jnp.asarray(
                    [jnp.nanmean(best_chi2), jnp.nanmedian(best_chi2), 0.0],
                    dtype=theta0.dtype,
                ),
                _truth_metric_vector(best_theta, truth_theta),
            ]
        )[None, :]
        return best_theta, best_chi2, grad_norm, model_mags, metrics

    return optimize


def _fast_grid_bounds(
    theta,
    free_pos: int,
    lower,
    upper,
    prior_gaussian_mask,
    prior_gaussian_loc,
    prior_gaussian_scale,
    redshift_grid_width: float,
    fast_grid_prior_width: float,
    redshift_free_pos: int | None,
):
    center = theta[:, free_pos]
    is_redshift = redshift_free_pos is not None and free_pos == redshift_free_pos
    gaussian_loc = prior_gaussian_loc[:, free_pos]
    gaussian_scale = prior_gaussian_scale[:, free_pos]
    valid_gaussian = (
        prior_gaussian_mask[:, free_pos]
        & jnp.isfinite(gaussian_loc)
        & jnp.isfinite(gaussian_scale)
        & (gaussian_scale > 0)
    )
    prior_width = jnp.asarray(fast_grid_prior_width, dtype=theta.dtype)
    grid_width = jnp.asarray(redshift_grid_width, dtype=theta.dtype)
    lo_gaussian = gaussian_loc - prior_width * gaussian_scale
    hi_gaussian = gaussian_loc + prior_width * gaussian_scale
    fallback_width = jnp.where(
        is_redshift,
        grid_width,
        jnp.asarray(0.25, dtype=theta.dtype) * (upper[free_pos] - lower[free_pos]),
    )
    lo = jnp.where(valid_gaussian, lo_gaussian, center - fallback_width)
    hi = jnp.where(valid_gaussian, hi_gaussian, center + fallback_width)
    lo = jnp.maximum(lo, lower[free_pos])
    hi = jnp.minimum(hi, upper[free_pos])
    hi = jnp.maximum(hi, lo + jnp.asarray(1.0e-4, dtype=theta.dtype))
    return lo, hi


def _build_population_adam_optimizer(
    context: DspsContext,
    parameter_names: list[str],
    free_indices: np.ndarray,
    maxiter: int,
    learning_rate: float,
    sigma_floor: float,
    prior_weight: float,
    hyper_mu_scale: float,
    physical_prior_weight: float,
    relation_default_mask: jnp.ndarray,
    relation_target_pos: jnp.ndarray,
    relation_predictor_free_pos: jnp.ndarray,
    relation_predictor_base_index: jnp.ndarray,
    relation_pivot: jnp.ndarray,
    relation_intercept0: jnp.ndarray,
    relation_slope0: jnp.ndarray,
    relation_sigma0: jnp.ndarray,
    relation_slope_scale: jnp.ndarray,
):
    free_indices_jax = jnp.asarray(free_indices)

    def single_chi2(theta, base, observed, sigma, mask):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        model_mag = model_mags_jax(context, params)
        chi = jnp.where(mask, (observed - model_mag) / sigma, 0.0)
        return jnp.nan_to_num(jnp.sum(chi**2), nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    batch_chi2_grad = jax.vmap(
        single_chi2,
        in_axes=(0, 0, 0, 0, 0),
    )

    def loss(
        theta,
        mu,
        raw_sigma,
        relation_intercept,
        relation_slope,
        relation_raw_sigma,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
    ):
        chi2 = batch_chi2_grad(theta, base_matrix, observed, sigma, mask)
        sigma_pop = jax.nn.softplus(raw_sigma) + sigma_floor
        default_prior = 0.5 * jnp.sum(
            relation_default_mask
            * (((theta - mu) / sigma_pop) ** 2 + 2.0 * jnp.log(sigma_pop)),
            axis=1,
        )
        relation_prior = _population_relation_prior_penalty(
            theta,
            base_matrix,
            relation_intercept,
            relation_slope,
            relation_raw_sigma,
            sigma_floor,
            relation_target_pos,
            relation_predictor_free_pos,
            relation_predictor_base_index,
            relation_pivot,
        )
        physical_prior = _physical_prior_penalty(
            theta,
            lower,
            upper,
            prior_gaussian_mask,
            prior_gaussian_loc,
            prior_gaussian_scale,
            prior_beta_mask,
            prior_beta_alpha,
            prior_beta_beta,
        )
        hyper = 0.5 * jnp.sum(relation_default_mask * (mu / hyper_mu_scale) ** 2)
        relation_hyper = 0.5 * jnp.sum(
            (relation_intercept / hyper_mu_scale) ** 2
            + (relation_slope / relation_slope_scale) ** 2
        )
        return (
            0.5 * jnp.sum(chi2)
            + prior_weight * (jnp.sum(default_prior) + jnp.sum(relation_prior))
            + physical_prior_weight * jnp.sum(physical_prior)
            + hyper
            + relation_hyper
        )

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))

    def single_mags(theta, base):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        return model_mags_jax(context, params)

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0))
    amplitude_free_pos = _free_position(
        parameter_names, free_indices, "log10_formed_mass_msun"
    )
    if amplitude_free_pos is None:
        amplitude_free_pos = _free_position(parameter_names, free_indices, "log10_sfr")

    @jax.jit
    def optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
        truth_theta,
    ):
        theta0 = _warm_start_amplitude(
            theta0,
            base_matrix,
            observed,
            mask,
            lower,
            upper,
            batch_mags,
            amplitude_free_pos,
        )
        y0 = _bounded_to_unconstrained(theta0, lower, upper)
        mu0 = jnp.nanmean(theta0, axis=0)
        raw_sigma0 = _softplus_inverse(jnp.nanstd(theta0, axis=0) + 0.2)
        relation_raw_sigma0 = _softplus_inverse(relation_sigma0)
        m_y = jnp.zeros_like(y0)
        v_y = jnp.zeros_like(y0)
        m_mu = jnp.zeros_like(mu0)
        v_mu = jnp.zeros_like(mu0)
        m_sigma = jnp.zeros_like(raw_sigma0)
        v_sigma = jnp.zeros_like(raw_sigma0)
        m_relation_intercept = jnp.zeros_like(relation_intercept0)
        v_relation_intercept = jnp.zeros_like(relation_intercept0)
        m_relation_slope = jnp.zeros_like(relation_slope0)
        v_relation_slope = jnp.zeros_like(relation_slope0)
        m_relation_sigma = jnp.zeros_like(relation_raw_sigma0)
        v_relation_sigma = jnp.zeros_like(relation_raw_sigma0)
        best_theta = theta0
        best_mu = mu0
        best_raw_sigma = raw_sigma0
        best_relation_intercept = relation_intercept0
        best_relation_slope = relation_slope0
        best_relation_raw_sigma = relation_raw_sigma0
        best_loss = jnp.inf
        carry0 = (
            y0,
            mu0,
            raw_sigma0,
            relation_intercept0,
            relation_slope0,
            relation_raw_sigma0,
            m_y,
            v_y,
            m_mu,
            v_mu,
            m_sigma,
            v_sigma,
            m_relation_intercept,
            v_relation_intercept,
            m_relation_slope,
            v_relation_slope,
            m_relation_sigma,
            v_relation_sigma,
            best_theta,
            best_mu,
            best_raw_sigma,
            best_relation_intercept,
            best_relation_slope,
            best_relation_raw_sigma,
            best_loss,
        )

        def adam_update(value, grad, m, v, iteration):
            grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            t = iteration + 1.0
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad**2)
            m_hat = m / (1.0 - 0.9**t)
            v_hat = v / (1.0 - 0.999**t)
            return value - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1.0e-8), m, v

        def step(carry, iteration):
            (
                y,
                mu,
                raw_sigma,
                relation_intercept,
                relation_slope,
                relation_raw_sigma,
                m_y,
                v_y,
                m_mu,
                v_mu,
                m_sigma,
                v_sigma,
                m_relation_intercept,
                v_relation_intercept,
                m_relation_slope,
                v_relation_slope,
                m_relation_sigma,
                v_relation_sigma,
                best_theta,
                best_mu,
                best_raw_sigma,
                best_relation_intercept,
                best_relation_slope,
                best_relation_raw_sigma,
                best_loss,
            ) = carry
            theta = _unconstrained_to_bounded(y, lower, upper)
            value, grads = value_and_grad(
                theta,
                mu,
                raw_sigma,
                relation_intercept,
                relation_slope,
                relation_raw_sigma,
                base_matrix,
                observed,
                sigma,
                mask,
                lower,
                upper,
                prior_gaussian_mask,
                prior_gaussian_loc,
                prior_gaussian_scale,
                prior_beta_mask,
                prior_beta_alpha,
                prior_beta_beta,
            )
            (
                grad_theta_direct,
                grad_mu,
                grad_sigma,
                grad_relation_intercept,
                grad_relation_slope,
                grad_relation_sigma,
            ) = grads
            _, grad_y = jax.value_and_grad(
                lambda yy: loss(
                    _unconstrained_to_bounded(yy, lower, upper),
                    mu,
                    raw_sigma,
                    relation_intercept,
                    relation_slope,
                    relation_raw_sigma,
                    base_matrix,
                    observed,
                    sigma,
                    mask,
                    lower,
                    upper,
                    prior_gaussian_mask,
                    prior_gaussian_loc,
                    prior_gaussian_scale,
                    prior_beta_mask,
                    prior_beta_alpha,
                    prior_beta_beta,
                )
            )(y)
            improved = value < best_loss
            best_theta = jnp.where(improved, theta, best_theta)
            best_mu = jnp.where(improved, mu, best_mu)
            best_raw_sigma = jnp.where(improved, raw_sigma, best_raw_sigma)
            best_relation_intercept = jnp.where(
                improved, relation_intercept, best_relation_intercept
            )
            best_relation_slope = jnp.where(
                improved, relation_slope, best_relation_slope
            )
            best_relation_raw_sigma = jnp.where(
                improved, relation_raw_sigma, best_relation_raw_sigma
            )
            best_loss = jnp.where(improved, value, best_loss)
            y, m_y, v_y = adam_update(y, grad_y, m_y, v_y, iteration)
            mu, m_mu, v_mu = adam_update(mu, grad_mu, m_mu, v_mu, iteration)
            raw_sigma, m_sigma, v_sigma = adam_update(
                raw_sigma, grad_sigma, m_sigma, v_sigma, iteration
            )
            relation_intercept, m_relation_intercept, v_relation_intercept = (
                adam_update(
                    relation_intercept,
                    grad_relation_intercept,
                    m_relation_intercept,
                    v_relation_intercept,
                    iteration,
                )
            )
            relation_slope, m_relation_slope, v_relation_slope = adam_update(
                relation_slope,
                grad_relation_slope,
                m_relation_slope,
                v_relation_slope,
                iteration,
            )
            relation_raw_sigma, m_relation_sigma, v_relation_sigma = adam_update(
                relation_raw_sigma,
                grad_relation_sigma,
                m_relation_sigma,
                v_relation_sigma,
                iteration,
            )
            y = jnp.clip(y, -30.0, 30.0)
            raw_sigma = jnp.clip(raw_sigma, -8.0, 4.0)
            relation_raw_sigma = jnp.clip(relation_raw_sigma, -8.0, 4.0)
            metrics = jnp.concatenate(
                [
                    jnp.asarray(
                        [
                            value,
                            jnp.nanmean(
                                batch_chi2_grad(
                                    theta, base_matrix, observed, sigma, mask
                                )
                            ),
                            jnp.nanmean(jnp.linalg.norm(grad_theta_direct, axis=1)),
                        ]
                    ),
                    _truth_metric_vector(theta, truth_theta),
                ]
            )
            return (
                y,
                mu,
                raw_sigma,
                relation_intercept,
                relation_slope,
                relation_raw_sigma,
                m_y,
                v_y,
                m_mu,
                v_mu,
                m_sigma,
                v_sigma,
                m_relation_intercept,
                v_relation_intercept,
                m_relation_slope,
                v_relation_slope,
                m_relation_sigma,
                v_relation_sigma,
                best_theta,
                best_mu,
                best_raw_sigma,
                best_relation_intercept,
                best_relation_slope,
                best_relation_raw_sigma,
                best_loss,
            ), metrics

        (
            (
                *_,
                best_theta,
                best_mu,
                best_raw_sigma,
                best_relation_intercept,
                best_relation_slope,
                best_relation_raw_sigma,
                best_loss,
            ),
            metrics,
        ) = jax.lax.scan(
            step,
            carry0,
            jnp.arange(maxiter, dtype=jnp.int32),
        )
        sigma_pop = jax.nn.softplus(best_raw_sigma) + sigma_floor
        relation_sigma = jax.nn.softplus(best_relation_raw_sigma) + sigma_floor
        chi2, grad = jax.vmap(
            jax.value_and_grad(single_chi2, argnums=0),
            in_axes=(0, 0, 0, 0, 0),
        )(best_theta, base_matrix, observed, sigma, mask)
        model_mags = batch_mags(best_theta, base_matrix)
        return (
            best_theta,
            best_mu,
            sigma_pop,
            best_relation_intercept,
            best_relation_slope,
            relation_sigma,
            best_loss,
            chi2,
            jnp.linalg.norm(grad, axis=1),
            model_mags,
            metrics,
        )

    return optimize


def _params_from_vectors(
    theta, base, parameter_names: list[str], free_indices: jnp.ndarray
) -> dict[str, Any]:
    values = _apply_free_values(base, theta, free_indices)
    return {name: values[index] for index, name in enumerate(parameter_names)}


def _physical_prior_penalty(
    theta,
    lower,
    upper,
    gaussian_mask,
    gaussian_loc,
    gaussian_scale,
    beta_mask,
    beta_alpha,
    beta_beta,
):
    gaussian_scale = jnp.maximum(gaussian_scale, 1.0e-6)
    gaussian = 0.5 * ((theta - gaussian_loc) / gaussian_scale) ** 2 + jnp.log(
        gaussian_scale
    )
    gaussian = jnp.where(gaussian_mask, gaussian, 0.0)

    scaled = jnp.clip((theta - lower) / (upper - lower), 1.0e-6, 1.0 - 1.0e-6)
    beta = -(
        (beta_alpha - 1.0) * jnp.log(scaled) + (beta_beta - 1.0) * jnp.log1p(-scaled)
    )
    beta = jnp.where(beta_mask, beta, 0.0)
    return jnp.sum(gaussian + beta, axis=-1)


def _population_relation_prior_penalty(
    theta,
    base_matrix,
    relation_intercept,
    relation_slope,
    relation_raw_sigma,
    sigma_floor,
    target_pos,
    predictor_free_pos,
    predictor_base_index,
    pivot,
):
    if relation_intercept.shape[0] == 0:
        return jnp.zeros(theta.shape[0])
    target_values = theta[:, target_pos]
    free_predictor = jnp.take(theta, jnp.maximum(predictor_free_pos, 0), axis=1)
    base_predictor = jnp.take(base_matrix, jnp.maximum(predictor_base_index, 0), axis=1)
    predictor = jnp.where(
        predictor_free_pos[None, :] >= 0, free_predictor, base_predictor
    )
    loc = relation_intercept[None, :] + relation_slope[None, :] * (
        predictor - pivot[None, :]
    )
    sigma_rel = jax.nn.softplus(relation_raw_sigma) + sigma_floor
    penalty = 0.5 * ((target_values - loc) / sigma_rel[None, :]) ** 2 + jnp.log(
        sigma_rel[None, :]
    )
    return jnp.sum(penalty, axis=1)


def _apply_free_values(base, theta, free_indices):
    return base.at[..., free_indices].set(theta)


def _free_position(
    parameter_names: list[str], free_indices: np.ndarray, name: str
) -> int | None:
    if name not in parameter_names:
        return None
    parameter_index = parameter_names.index(name)
    positions = np.flatnonzero(free_indices == parameter_index)
    return int(positions[0]) if len(positions) else None


def _warm_start_amplitude(
    theta0, base_matrix, observed, mask, lower, upper, batch_mags, free_pos: int | None
):
    if free_pos is None:
        return theta0
    model_mag = batch_mags(theta0, base_matrix)
    delta_mag = jnp.where(mask, model_mag - observed, jnp.nan)
    delta_log10_sfr = (
        jnp.nanmedian(delta_mag, axis=1) / jnp.asarray(2.5, dtype=theta0.dtype)
    ).astype(theta0.dtype)
    warmed = theta0.at[:, free_pos].set(
        theta0[:, free_pos] + jnp.nan_to_num(delta_log10_sfr)
    )
    return jnp.clip(warmed, lower, upper)


def _softplus_inverse(value):
    value = jnp.maximum(value, 1.0e-6)
    return jnp.log(jnp.expm1(value))


def _truth_metric_vector(theta, truth_theta):
    diff = theta - truth_theta
    diff = jnp.where(jnp.isfinite(truth_theta), diff, jnp.nan)
    sq = diff**2
    mse = jnp.nanmean(sq)
    mae = jnp.nanmean(jnp.abs(diff))
    per_parameter_mse = jnp.nanmean(sq, axis=0)
    return jnp.concatenate([jnp.asarray([mse, jnp.sqrt(mse), mae]), per_parameter_mse])


def _batch_trace_from_arrays(
    metrics,
    free_names: list[str] | None = None,
    include_truth_metrics: bool = False,
) -> list[dict[str, float]]:
    arr = np.asarray(metrics)
    rows = []
    for index, row in enumerate(arr):
        entry = {
            "iteration": float(index + 1),
            "mean_chi2_or_loss": float(row[0]),
            "median_chi2": float(row[1]),
            "mean_gradient_norm": float(row[2]),
        }
        if include_truth_metrics and len(row) >= 6:
            entry.update(
                {
                    "truth_mse": float(row[3]),
                    "truth_rmse": float(row[4]),
                    "truth_mae": float(row[5]),
                }
            )
            for offset, name in enumerate(free_names or []):
                metric_index = 6 + offset
                if metric_index < len(row):
                    entry[f"truth_mse_{name}"] = float(row[metric_index])
        rows.append(entry)
    return rows


def _jax_device() -> str:
    device = jax.devices()[0]
    return f"{device.platform}:{device.id}"
