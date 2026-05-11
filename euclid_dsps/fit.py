"""Single-galaxy fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize as jax_minimize

from .io import GalaxyObservation
from .model import DspsContext, ModelResult, model_mags_jax, run_dsps_model


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
        return jnp.nan_to_num(chi2, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

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
    theta0 = jnp.asarray(setup["theta0"])
    base_matrix = jnp.asarray(setup["base_matrix"])
    observed = jnp.asarray(observed_mag)
    sigma = jnp.asarray(sigma_mag)
    mask = jnp.isfinite(observed) & jnp.isfinite(sigma) & (sigma > 0)
    lower = jnp.asarray(setup["lower"])
    upper = jnp.asarray(setup["upper"])
    maxiter = int(fit_config.get("maxiter", 80))
    learning_rate = float(fit_config.get("learning_rate", 0.03))

    optimize = _build_independent_adam_optimizer(
        context=context,
        parameter_names=setup["parameter_names"],
        free_indices=setup["free_indices"],
        maxiter=maxiter,
        learning_rate=learning_rate,
    )
    best_theta, chi2, grad_norm, model_mags, trace_arrays = optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        lower,
        upper,
        jnp.asarray(truth_theta_arr),
    )
    best_matrix = _apply_free_values(
        base_matrix, best_theta, jnp.asarray(setup["free_indices"])
    )
    trace = _batch_trace_from_arrays(
        trace_arrays, setup["free_names"], include_truth_metrics=has_truth
    )
    return BatchFitResult(
        success=np.isfinite(np.asarray(chi2)),
        message=f"jax_adam_vmap maxiter={maxiter}, device={_jax_device()}",
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

    optimize = _build_population_adam_optimizer(
        context=context,
        parameter_names=setup["parameter_names"],
        free_indices=setup["free_indices"],
        maxiter=maxiter,
        learning_rate=learning_rate,
        sigma_floor=float(pop.get("sigma_floor", 0.03)),
        prior_weight=float(pop.get("prior_weight", 1.0)),
        hyper_mu_scale=float(pop.get("hyper_mu_scale", 5.0)),
    )
    best_theta, mu, sigma_pop, loss, chi2, grad_norm, model_mags, trace_arrays = (
        optimize(
            theta0,
            base_matrix,
            observed,
            sigma,
            mask,
            lower,
            upper,
            jnp.asarray(truth_theta_arr),
        )
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
    parameter_names = list(base_params_rows[0])
    missing = [name for name in free_names if name not in parameter_names]
    if missing:
        raise ValueError(f"Free parameters missing from base params: {missing}")
    base_matrix = np.asarray(
        [[float(row[name]) for name in parameter_names] for row in base_params_rows],
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
    }


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
):
    free_indices_jax = jnp.asarray(free_indices)

    def single_chi2(theta, base, observed, sigma, mask):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        model_mag = model_mags_jax(context, params)
        chi = jnp.where(mask, (observed - model_mag) / sigma, 0.0)
        return jnp.nan_to_num(jnp.sum(chi**2), nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    def single_mags(theta, base):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        return model_mags_jax(context, params)

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0))
    log_sfr_free_pos = _free_position(parameter_names, free_indices, "log10_sfr")

    @jax.jit
    def optimize(theta0, base_matrix, observed, sigma, mask, lower, upper, truth_theta):
        theta0 = _warm_start_log10_sfr(
            theta0,
            base_matrix,
            observed,
            mask,
            lower,
            upper,
            batch_mags,
            log_sfr_free_pos,
        )
        y0 = _bounded_to_unconstrained(theta0, lower, upper)
        m0 = jnp.zeros_like(y0)
        v0 = jnp.zeros_like(y0)
        best_chi20 = jnp.full((theta0.shape[0],), jnp.inf)
        best_grad0 = jnp.full((theta0.shape[0],), jnp.inf)
        carry0 = (y0, m0, v0, theta0, best_chi20, best_grad0)

        def single_chi2_y(y, base, observed_i, sigma_i, mask_i):
            theta = _unconstrained_to_bounded(y, lower, upper)
            return single_chi2(theta, base, observed_i, sigma_i, mask_i)

        batch_chi2_y_grad = jax.vmap(
            jax.value_and_grad(single_chi2_y, argnums=0),
            in_axes=(0, 0, 0, 0, 0),
        )

        def step(carry, iteration):
            y, m, v, best_theta, best_chi2, best_grad = carry
            theta = _unconstrained_to_bounded(y, lower, upper)
            chi2, grad = batch_chi2_y_grad(y, base_matrix, observed, sigma, mask)
            grad = jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = jnp.linalg.norm(grad, axis=1)
            improved = chi2 < best_chi2
            best_theta = jnp.where(improved[:, None], theta, best_theta)
            best_chi2 = jnp.where(improved, chi2, best_chi2)
            best_grad = jnp.where(improved, grad_norm, best_grad)

            t = iteration + 1.0
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad**2)
            m_hat = m / (1.0 - 0.9**t)
            v_hat = v / (1.0 - 0.999**t)
            y = jnp.clip(
                y - learning_rate * m_hat / (jnp.sqrt(v_hat) + 1.0e-8), -30.0, 30.0
            )
            metrics = jnp.concatenate(
                [
                    jnp.asarray(
                        [jnp.nanmean(chi2), jnp.nanmedian(chi2), jnp.nanmean(grad_norm)]
                    ),
                    _truth_metric_vector(theta, truth_theta),
                ]
            )
            return (y, m, v, best_theta, best_chi2, best_grad), metrics

        (_, _, _, best_theta, best_chi2, best_grad), metrics = jax.lax.scan(
            step,
            carry0,
            jnp.arange(maxiter, dtype=jnp.float64),
        )
        model_mags = batch_mags(best_theta, base_matrix)
        return best_theta, best_chi2, best_grad, model_mags, metrics

    return optimize


def _build_population_adam_optimizer(
    context: DspsContext,
    parameter_names: list[str],
    free_indices: np.ndarray,
    maxiter: int,
    learning_rate: float,
    sigma_floor: float,
    prior_weight: float,
    hyper_mu_scale: float,
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

    def loss(theta, mu, raw_sigma, base_matrix, observed, sigma, mask):
        chi2 = batch_chi2_grad(theta, base_matrix, observed, sigma, mask)
        sigma_pop = jax.nn.softplus(raw_sigma) + sigma_floor
        prior = 0.5 * jnp.sum(
            ((theta - mu) / sigma_pop) ** 2 + 2.0 * jnp.log(sigma_pop), axis=1
        )
        hyper = 0.5 * jnp.sum((mu / hyper_mu_scale) ** 2)
        return 0.5 * jnp.sum(chi2) + prior_weight * jnp.sum(prior) + hyper

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1, 2))

    def single_mags(theta, base):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        return model_mags_jax(context, params)

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0))
    log_sfr_free_pos = _free_position(parameter_names, free_indices, "log10_sfr")

    @jax.jit
    def optimize(theta0, base_matrix, observed, sigma, mask, lower, upper, truth_theta):
        theta0 = _warm_start_log10_sfr(
            theta0,
            base_matrix,
            observed,
            mask,
            lower,
            upper,
            batch_mags,
            log_sfr_free_pos,
        )
        y0 = _bounded_to_unconstrained(theta0, lower, upper)
        mu0 = jnp.nanmean(theta0, axis=0)
        raw_sigma0 = _softplus_inverse(jnp.nanstd(theta0, axis=0) + 0.2)
        m_y = jnp.zeros_like(y0)
        v_y = jnp.zeros_like(y0)
        m_mu = jnp.zeros_like(mu0)
        v_mu = jnp.zeros_like(mu0)
        m_sigma = jnp.zeros_like(raw_sigma0)
        v_sigma = jnp.zeros_like(raw_sigma0)
        best_theta = theta0
        best_mu = mu0
        best_raw_sigma = raw_sigma0
        best_loss = jnp.inf
        carry0 = (
            y0,
            mu0,
            raw_sigma0,
            m_y,
            v_y,
            m_mu,
            v_mu,
            m_sigma,
            v_sigma,
            best_theta,
            best_mu,
            best_raw_sigma,
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
                m_y,
                v_y,
                m_mu,
                v_mu,
                m_sigma,
                v_sigma,
                best_theta,
                best_mu,
                best_raw_sigma,
                best_loss,
            ) = carry
            theta = _unconstrained_to_bounded(y, lower, upper)
            value, grads = value_and_grad(
                theta, mu, raw_sigma, base_matrix, observed, sigma, mask
            )
            grad_theta_direct, grad_mu, grad_sigma = grads
            _, grad_y = jax.value_and_grad(
                lambda yy: loss(
                    _unconstrained_to_bounded(yy, lower, upper),
                    mu,
                    raw_sigma,
                    base_matrix,
                    observed,
                    sigma,
                    mask,
                )
            )(y)
            improved = value < best_loss
            best_theta = jnp.where(improved, theta, best_theta)
            best_mu = jnp.where(improved, mu, best_mu)
            best_raw_sigma = jnp.where(improved, raw_sigma, best_raw_sigma)
            best_loss = jnp.where(improved, value, best_loss)
            y, m_y, v_y = adam_update(y, grad_y, m_y, v_y, iteration)
            mu, m_mu, v_mu = adam_update(mu, grad_mu, m_mu, v_mu, iteration)
            raw_sigma, m_sigma, v_sigma = adam_update(
                raw_sigma, grad_sigma, m_sigma, v_sigma, iteration
            )
            y = jnp.clip(y, -30.0, 30.0)
            raw_sigma = jnp.clip(raw_sigma, -8.0, 4.0)
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
                m_y,
                v_y,
                m_mu,
                v_mu,
                m_sigma,
                v_sigma,
                best_theta,
                best_mu,
                best_raw_sigma,
                best_loss,
            ), metrics

        (*_, best_theta, best_mu, best_raw_sigma, best_loss), metrics = jax.lax.scan(
            step,
            carry0,
            jnp.arange(maxiter, dtype=jnp.float64),
        )
        sigma_pop = jax.nn.softplus(best_raw_sigma) + sigma_floor
        chi2, grad = jax.vmap(
            jax.value_and_grad(single_chi2, argnums=0),
            in_axes=(0, 0, 0, 0, 0),
        )(best_theta, base_matrix, observed, sigma, mask)
        model_mags = batch_mags(best_theta, base_matrix)
        return (
            best_theta,
            best_mu,
            sigma_pop,
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


def _warm_start_log10_sfr(
    theta0, base_matrix, observed, mask, lower, upper, batch_mags, free_pos: int | None
):
    if free_pos is None:
        return theta0
    model_mag = batch_mags(theta0, base_matrix)
    delta_mag = jnp.where(mask, model_mag - observed, jnp.nan)
    delta_log10_sfr = jnp.nanmedian(delta_mag, axis=1) / 2.5
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
