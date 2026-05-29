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
from .model import (
    DspsContext,
    ModelResult,
    dynamic_model_args,
    gas_metallicity_constraint_penalty_jax,
    model_mags_jax_dynamic,
    run_dsps_model,
)
from .photometry import abmag_to_fnu_cgs, abmag_to_fnu_cgs_jax, magerr_to_fluxerr_fnu_cgs


_FIT_DTYPE = jnp.float32


@dataclass(frozen=True)
class FitResult:
    success: bool
    message: str
    best_parameters: dict[str, float]
    chi2: float
    photometric_objective: float
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
    photometric_objective: np.ndarray
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
    if method == "jax_adam_vmap":
        observed_mag, sigma_mag, _ = _observation_arrays(observation)
        observed_flux, flux_error = _observation_flux_arrays(observation)
        batch = fit_galaxy_batch_adam(
            context=context,
            base_params_rows=[base_params],
            observed_mag=np.asarray(observed_mag)[None, :],
            sigma_mag=np.asarray(sigma_mag)[None, :],
            fit_config=fit_config,
            observed_flux=np.asarray(observed_flux)[None, :],
            flux_error=np.asarray(flux_error)[None, :],
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
            photometric_objective=float(batch.photometric_objective[0]),
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
    observed_mag, sigma_mag, _ = _observation_arrays(observation)
    observed_flux, flux_error = _observation_flux_arrays(observation)
    observed_fit, sigma_fit, finite_mask = _fit_arrays_for_likelihood(
        np.asarray(observed_mag)[None, :],
        np.asarray(sigma_mag)[None, :],
        fit_config,
        observed_flux=np.asarray(observed_flux)[None, :],
        flux_error=np.asarray(flux_error)[None, :],
    )
    observed_fit = jnp.asarray(observed_fit[0], dtype=float)
    sigma_fit = jnp.asarray(sigma_fit[0], dtype=float)
    finite_mask = jnp.asarray(finite_mask[0])
    model_args = dynamic_model_args(context)
    maxiter = int(fit_config.get("maxiter", 80))
    learning_rate = float(fit_config.get("learning_rate", 0.03))
    tolerance = float(fit_config.get("tolerance", 1.0e-5))
    patience = int(fit_config.get("patience", 12))
    photometric_likelihood = _photometric_likelihood(fit_config)
    student_t_dof = _student_t_dof(fit_config)
    trace: list[dict[str, float]] = []

    def unpack_jax(x: jnp.ndarray) -> dict[str, Any]:
        params: dict[str, Any] = {
            key: jnp.asarray(value) for key, value in base_params.items()
        }
        params.update({name: value for name, value in zip(names, x, strict=True)})
        return params

    def photometric_metrics(
        x: jnp.ndarray, args: tuple[Any, ...]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        params = unpack_jax(x)
        model_mag = model_mags_jax_dynamic(context, args, params)
        model_values = _model_values_for_likelihood(model_mag, fit_config)
        chi = jnp.where(finite_mask, (observed_fit - model_values) / sigma_fit, 0.0)
        chi2 = jnp.sum(chi**2)
        photometric_objective = _photometric_objective_from_chi(
            chi, photometric_likelihood, student_t_dof
        )
        return photometric_objective, chi2

    def objective(x: jnp.ndarray, args: tuple[Any, ...]) -> jnp.ndarray:
        photometric_objective, _ = photometric_metrics(x, args)
        params = unpack_jax(x)
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
        objective_value = (
            photometric_objective + float(fit_config.get("prior_weight", 1.0)) * prior
            + gas_metallicity_constraint_penalty_jax(params, context.model_config)
        )
        return jnp.nan_to_num(objective_value, nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    value_and_grad_fn = jax.value_and_grad(objective, argnums=0)
    if _should_jit_single_fit(context, fit_config):
        compiled_value_and_grad = jax.jit(value_and_grad_fn)

        def value_and_grad(x: jnp.ndarray):
            return compiled_value_and_grad(x, model_args)

    else:

        def value_and_grad(x: jnp.ndarray):
            return value_and_grad_fn(x, model_args)
    if method == "jax_adam":
        best_x, state, _best_value, best_grad_norm, success, message = _fit_bounded_adam(
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
        best_x, state, _best_value, best_grad_norm, success, message = _fit_bounded_bfgs(
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
    best_photometric_objective, best_chi2 = photometric_metrics(best_x, model_args)
    return FitResult(
        success=success,
        message=message,
        best_parameters=best_params,
        chi2=float(best_chi2),
        photometric_objective=float(best_photometric_objective),
        n_bands=len(observation.bands),
        trace=trace,
        model_result=best_model,
        gradient_norm=float(best_grad_norm),
    )


def _should_jit_single_fit(context: DspsContext, fit_config: dict[str, Any]) -> bool:
    """Use JIT unless explicitly disabled.

    Large context arrays are passed as dynamic arguments to compiled model calls,
    so the gas grid size no longer needs to disable single-galaxy JIT.
    """
    if "jit" in fit_config:
        return bool(fit_config["jit"])
    return True


def _context_gas_grid_nbytes(context: DspsContext) -> int:
    grid = context.ssp_flux_gas_grid_jax
    if grid is None:
        return 0
    shape = getattr(grid, "shape", ())
    dtype = np.dtype(getattr(grid, "dtype", np.float32))
    return int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


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


def _observation_flux_arrays(observation: GalaxyObservation) -> tuple[np.ndarray, np.ndarray]:
    observed_flux = np.asarray(
        [band.flux_fnu_cgs for band in observation.bands], dtype=float
    )
    flux_error = np.asarray(
        [
            (
                band.flux_error_fnu_cgs
                if band.flux_error_fnu_cgs is not None
                else magerr_to_fluxerr_fnu_cgs(band.flux_fnu_cgs, band.sigma_mag)
            )
            for band in observation.bands
        ],
        dtype=float,
    )
    return observed_flux, flux_error


def _likelihood_space(fit_config: dict[str, Any]) -> str:
    return str(fit_config.get("likelihood_space", "flux")).lower()


def _photometric_likelihood(fit_config: dict[str, Any]) -> str:
    name = str(fit_config.get("photometric_likelihood", "gaussian")).lower()
    name = name.replace("-", "_")
    aliases = {
        "chi2": "gaussian",
        "gaussian_chi2": "gaussian",
        "normal": "gaussian",
        "student": "student_t",
        "studentt": "student_t",
    }
    return aliases.get(name, name)


def _student_t_dof(fit_config: dict[str, Any]) -> float:
    dof = float(fit_config.get("student_t_dof", 2.0))
    if dof <= 0.0:
        raise ValueError("fit.student_t_dof must be positive")
    return dof


def _photometric_objective_from_chi(
    chi: jnp.ndarray,
    photometric_likelihood: str,
    student_t_dof: float,
) -> jnp.ndarray:
    chi2 = chi**2
    if photometric_likelihood == "gaussian":
        return jnp.sum(chi2)
    if photometric_likelihood == "student_t":
        return jnp.sum(
            (student_t_dof + 1.0) * jnp.log1p(chi2 / student_t_dof)
        )
    raise ValueError(f"Unsupported fit.photometric_likelihood: {photometric_likelihood}")


def _fit_arrays_for_likelihood(
    observed_mag: np.ndarray,
    sigma_mag: np.ndarray,
    fit_config: dict[str, Any],
    observed_flux: np.ndarray | None = None,
    flux_error: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    likelihood_space = _likelihood_space(fit_config)
    if likelihood_space == "mag":
        observed = np.asarray(observed_mag, dtype=float)
        sigma = np.asarray(sigma_mag, dtype=float)
    elif likelihood_space == "flux":
        if observed_flux is None:
            observed = np.asarray(abmag_to_fnu_cgs(observed_mag), dtype=float)
        else:
            observed = np.asarray(observed_flux, dtype=float)
        if flux_error is None:
            sigma = np.asarray(
                magerr_to_fluxerr_fnu_cgs(observed, sigma_mag), dtype=float
            )
        else:
            sigma = np.asarray(flux_error, dtype=float)
        floor_frac = float(fit_config.get("flux_error_floor_frac", 0.0))
        jitter = float(fit_config.get("flux_error_jitter", 0.0))
        sigma = np.sqrt(sigma**2 + (floor_frac * observed) ** 2 + jitter**2)
    else:
        raise ValueError(f"Unsupported fit.likelihood_space: {likelihood_space}")
    mask = np.isfinite(observed) & np.isfinite(sigma) & (sigma > 0.0)
    return observed, sigma, mask


def _model_values_for_likelihood(model_mag: jnp.ndarray, fit_config: dict[str, Any]):
    model_mag = _apply_band_calibration(model_mag, fit_config)
    if _likelihood_space(fit_config) == "flux":
        return abmag_to_fnu_cgs_jax(model_mag)
    return model_mag


def _apply_band_calibration(model_mag: jnp.ndarray, fit_config: dict[str, Any]):
    offsets = fit_config.get("band_calibration_offsets_mag") or []
    if not offsets:
        return model_mag
    return model_mag + jnp.asarray(offsets, dtype=model_mag.dtype)


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
        f"jax_adam converged: objective improvement < {tolerance} for {patience} steps"
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
    message = (
        f"jax_bfgs status={int(getattr(opt, 'status', -1))}, "
        f"nit={int(getattr(opt, 'nit', -1))}"
    )
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
    names: list[str], x: jnp.ndarray, objective: float, grad_norm: float, iteration: int
) -> dict[str, float]:
    entry = {
        name: float(value) for name, value in zip(names, np.asarray(x), strict=True)
    }
    entry["iteration"] = float(iteration)
    entry["objective"] = float(objective)
    entry["chi2"] = float(objective)
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
    observed_flux: np.ndarray | None = None,
    flux_error: np.ndarray | None = None,
    initial_theta: np.ndarray | None = None,
) -> BatchFitResult:
    """Fit many independent galaxies in one JAX-vmapped Adam run."""
    setup = _prepare_batch_fit(base_params_rows, fit_config, initial_theta=initial_theta)
    truth_theta_arr, has_truth = _prepare_truth_theta(setup["theta0"], truth_theta)
    theta0 = jnp.asarray(setup["theta0"], dtype=_FIT_DTYPE)
    base_matrix = jnp.asarray(setup["base_matrix"], dtype=_FIT_DTYPE)
    observed_values, sigma_values, valid_mask = _fit_arrays_for_likelihood(
        observed_mag,
        sigma_mag,
        fit_config,
        observed_flux=observed_flux,
        flux_error=flux_error,
    )
    observed = jnp.asarray(observed_values, dtype=_FIT_DTYPE)
    sigma = jnp.asarray(sigma_values, dtype=_FIT_DTYPE)
    mask = jnp.asarray(valid_mask)
    model_args = dynamic_model_args(context)
    warm_observed_mag = jnp.asarray(observed_mag, dtype=_FIT_DTYPE)
    warm_mask = jnp.isfinite(warm_observed_mag) & jnp.isfinite(
        jnp.asarray(sigma_mag, dtype=_FIT_DTYPE)
    )
    lower = jnp.asarray(setup["lower"], dtype=_FIT_DTYPE)
    upper = jnp.asarray(setup["upper"], dtype=_FIT_DTYPE)
    maxiter = int(fit_config.get("maxiter", 80))
    learning_rate = float(fit_config.get("learning_rate", 0.03))
    prior_weight = float(fit_config.get("prior_weight", 1.0))
    photometric_likelihood = _photometric_likelihood(fit_config)
    student_t_dof = _student_t_dof(fit_config)
    band_offsets_mag = tuple(
        float(value) for value in fit_config.get("band_calibration_offsets_mag", [])
    )

    cache_key = (
        "independent",
        id(context),
        tuple(setup["parameter_names"]),
        tuple(setup["free_names"]),
        maxiter,
        learning_rate,
        prior_weight,
        _likelihood_space(fit_config),
        photometric_likelihood,
        student_t_dof,
        band_offsets_mag,
    )
    if cache_key not in _OPTIMIZER_CACHE:
        _OPTIMIZER_CACHE[cache_key] = _build_independent_adam_optimizer(
            context=context,
            parameter_names=setup["parameter_names"],
            free_indices=setup["free_indices"],
            maxiter=maxiter,
            learning_rate=learning_rate,
            prior_weight=prior_weight,
            likelihood_space=_likelihood_space(fit_config),
            photometric_likelihood=photometric_likelihood,
            student_t_dof=student_t_dof,
            band_offsets_mag=band_offsets_mag,
        )
    optimize = _OPTIMIZER_CACHE[cache_key]

    best_theta, chi2, photometric_objective, grad_norm, model_mags, trace_arrays = optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        warm_observed_mag,
        warm_mask,
        lower,
        upper,
        jnp.asarray(setup["prior_gaussian_mask"]),
        jnp.asarray(setup["prior_gaussian_loc"], dtype=_FIT_DTYPE),
        jnp.asarray(setup["prior_gaussian_scale"], dtype=_FIT_DTYPE),
        jnp.asarray(setup["prior_beta_mask"]),
        jnp.asarray(setup["prior_beta_alpha"], dtype=_FIT_DTYPE),
        jnp.asarray(setup["prior_beta_beta"], dtype=_FIT_DTYPE),
        jnp.asarray(truth_theta_arr, dtype=_FIT_DTYPE),
        model_args,
    )
    best_matrix = _apply_free_values(
        base_matrix, best_theta, jnp.asarray(setup["free_indices"])
    )
    trace = _batch_trace_from_arrays(
        trace_arrays, setup["free_names"], include_truth_metrics=has_truth
    )
    return BatchFitResult(
        success=np.isfinite(np.asarray(chi2))
        & np.isfinite(np.asarray(photometric_objective)),
        message=f"jax_adam_vmap maxiter={maxiter}, device={_jax_device()}",
        parameter_names=setup["parameter_names"],
        free_parameter_names=setup["free_names"],
        best_parameter_matrix=np.asarray(best_matrix),
        chi2=np.asarray(chi2),
        photometric_objective=np.asarray(photometric_objective),
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
    observed_flux: np.ndarray | None = None,
    flux_error: np.ndarray | None = None,
) -> PopulationFitResult:
    """Joint MAP fit with a Gaussian population prior over free parameters."""
    setup = _prepare_batch_fit(
        base_params_rows, fit_config, initial_theta=initial_theta
    )
    truth_theta_arr, has_truth = _prepare_truth_theta(setup["theta0"], truth_theta)
    theta0 = jnp.asarray(setup["theta0"])
    base_matrix = jnp.asarray(setup["base_matrix"])
    observed_values, sigma_values, valid_mask = _fit_arrays_for_likelihood(
        observed_mag,
        sigma_mag,
        fit_config,
        observed_flux=observed_flux,
        flux_error=flux_error,
    )
    observed = jnp.asarray(observed_values)
    sigma = jnp.asarray(sigma_values)
    mask = jnp.asarray(valid_mask)
    model_args = dynamic_model_args(context)
    warm_observed_mag = jnp.asarray(observed_mag)
    warm_mask = jnp.isfinite(warm_observed_mag) & jnp.isfinite(jnp.asarray(sigma_mag))
    lower = jnp.asarray(setup["lower"])
    upper = jnp.asarray(setup["upper"])
    maxiter = int(fit_config.get("maxiter", 80))
    learning_rate = float(fit_config.get("learning_rate", 0.03))
    pop = fit_config.get("population", {})

    sigma_floor = float(pop.get("sigma_floor", 0.03))
    prior_weight = float(pop.get("prior_weight", 1.0))
    hyper_mu_scale = float(pop.get("hyper_mu_scale", 5.0))
    physical_prior_weight = float(fit_config.get("prior_weight", 1.0))
    photometric_likelihood = _photometric_likelihood(fit_config)
    student_t_dof = _student_t_dof(fit_config)
    band_offsets_mag = tuple(
        float(value) for value in fit_config.get("band_calibration_offsets_mag", [])
    )

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
        _likelihood_space(fit_config),
        photometric_likelihood,
        student_t_dof,
        band_offsets_mag,
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
            likelihood_space=_likelihood_space(fit_config),
            photometric_likelihood=photometric_likelihood,
            student_t_dof=student_t_dof,
            band_offsets_mag=band_offsets_mag,
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
        photometric_objective,
        grad_norm,
        model_mags,
        trace_arrays,
    ) = optimize(
        theta0,
        base_matrix,
        observed,
        sigma,
        mask,
        warm_observed_mag,
        warm_mask,
        lower,
        upper,
        jnp.asarray(setup["prior_gaussian_mask"]),
        jnp.asarray(setup["prior_gaussian_loc"]),
        jnp.asarray(setup["prior_gaussian_scale"]),
        jnp.asarray(setup["prior_beta_mask"]),
        jnp.asarray(setup["prior_beta_alpha"]),
        jnp.asarray(setup["prior_beta_beta"]),
        jnp.asarray(truth_theta_arr),
        model_args,
    )
    best_matrix = _apply_free_values(
        base_matrix, best_theta, jnp.asarray(setup["free_indices"])
    )
    batch = BatchFitResult(
        success=np.isfinite(np.asarray(chi2))
        & np.isfinite(np.asarray(photometric_objective)),
        message=f"jax_population_adam maxiter={maxiter}, device={_jax_device()}",
        parameter_names=setup["parameter_names"],
        free_parameter_names=setup["free_names"],
        best_parameter_matrix=np.asarray(best_matrix),
        chi2=np.asarray(chi2),
        photometric_objective=np.asarray(photometric_objective),
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
                "initial_theta shape must be "
                f"{(len(base_params_rows), len(free_names))}, got {theta0.shape}"
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
    likelihood_space: str,
    photometric_likelihood: str,
    student_t_dof: float,
    band_offsets_mag: tuple[float, ...],
):
    free_indices_jax = jnp.asarray(free_indices)
    band_offsets_jax = (
        jnp.asarray(band_offsets_mag, dtype=_FIT_DTYPE)
        if band_offsets_mag
        else None
    )

    def single_chi(theta, base, observed, sigma, mask, model_args):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        model_mag = model_mags_jax_dynamic(context, model_args, params)
        if band_offsets_jax is not None:
            model_mag = model_mag + band_offsets_jax
        model_values = (
            abmag_to_fnu_cgs_jax(model_mag)
            if likelihood_space == "flux"
            else model_mag
        )
        return jnp.where(mask, (observed - model_values) / sigma, 0.0)

    def single_chi2(theta, base, observed, sigma, mask, model_args):
        chi = single_chi(theta, base, observed, sigma, mask, model_args)
        return jnp.nan_to_num(jnp.sum(chi**2), nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    def single_photometric_objective(theta, base, observed, sigma, mask, model_args):
        chi = single_chi(theta, base, observed, sigma, mask, model_args)
        return jnp.nan_to_num(
            _photometric_objective_from_chi(
                chi, photometric_likelihood, student_t_dof
            ),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )

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
        model_args,
    ):
        photometric_objective = single_photometric_objective(
            theta, base, observed, sigma, mask, model_args
        )
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
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
            photometric_objective
            + prior_weight * prior
            + gas_metallicity_constraint_penalty_jax(params, context.model_config),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )

    def single_mags(theta, base, model_args):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        model_mag = model_mags_jax_dynamic(context, model_args, params)
        if band_offsets_jax is not None:
            model_mag = model_mag + band_offsets_jax
        return model_mag

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0, None))
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
        warm_observed_mag,
        warm_mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
        truth_theta,
        model_args,
    ):
        def batch_mags_with_args(theta, base):
            return batch_mags(theta, base, model_args)

        theta0 = _warm_start_amplitude(
            theta0,
            base_matrix,
            warm_observed_mag,
            warm_mask,
            lower,
            upper,
            batch_mags_with_args,
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
                model_args,
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
            photometric_objective = jax.vmap(
                single_photometric_objective,
                in_axes=(0, 0, 0, 0, 0, None),
            )(
                theta, base_matrix, observed, sigma, mask, model_args
            )
            chi2 = jax.vmap(single_chi2, in_axes=(0, 0, 0, 0, 0, None))(
                theta, base_matrix, observed, sigma, mask, model_args
            )
            metrics = jnp.concatenate(
                [
                    jnp.asarray(
                        [
                            jnp.nanmean(photometric_objective),
                            jnp.nanmedian(chi2),
                            jnp.nanmean(grad_norm),
                        ]
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
        best_chi2 = jax.vmap(single_chi2, in_axes=(0, 0, 0, 0, 0, None))(
            best_theta, base_matrix, observed, sigma, mask, model_args
        )
        best_photometric_objective = jax.vmap(
            single_photometric_objective,
            in_axes=(0, 0, 0, 0, 0, None),
        )(best_theta, base_matrix, observed, sigma, mask, model_args)
        model_mags = batch_mags(best_theta, base_matrix, model_args)
        return (
            best_theta,
            best_chi2,
            best_photometric_objective,
            best_grad,
            model_mags,
            metrics,
        )

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
    physical_prior_weight: float,
    likelihood_space: str,
    photometric_likelihood: str,
    student_t_dof: float,
    band_offsets_mag: tuple[float, ...],
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
    band_offsets_jax = (
        jnp.asarray(band_offsets_mag, dtype=_FIT_DTYPE)
        if band_offsets_mag
        else None
    )

    def single_chi(theta, base, observed, sigma, mask, model_args):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        model_mag = model_mags_jax_dynamic(context, model_args, params)
        if band_offsets_jax is not None:
            model_mag = model_mag + band_offsets_jax
        model_values = (
            abmag_to_fnu_cgs_jax(model_mag)
            if likelihood_space == "flux"
            else model_mag
        )
        return jnp.where(mask, (observed - model_values) / sigma, 0.0)

    def single_chi2(theta, base, observed, sigma, mask, model_args):
        chi = single_chi(theta, base, observed, sigma, mask, model_args)
        return jnp.nan_to_num(jnp.sum(chi**2), nan=1.0e30, posinf=1.0e30, neginf=1.0e30)

    batch_chi2_grad = jax.vmap(
        single_chi2,
        in_axes=(0, 0, 0, 0, 0, None),
    )

    def single_photometric_objective(theta, base, observed, sigma, mask, model_args):
        chi = single_chi(theta, base, observed, sigma, mask, model_args)
        return jnp.nan_to_num(
            _photometric_objective_from_chi(
                chi, photometric_likelihood, student_t_dof
            ),
            nan=1.0e30,
            posinf=1.0e30,
            neginf=1.0e30,
        )

    batch_photometric_objective = jax.vmap(
        single_photometric_objective,
        in_axes=(0, 0, 0, 0, 0, None),
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
        warm_observed_mag,
        warm_mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
        model_args,
    ):
        photometric_objective = batch_photometric_objective(
            theta, base_matrix, observed, sigma, mask, model_args
        )
        gas_constraints = jax.vmap(
            lambda theta_i, base_i: gas_metallicity_constraint_penalty_jax(
                _params_from_vectors(theta_i, base_i, parameter_names, free_indices_jax),
                context.model_config,
            )
        )(theta, base_matrix)
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
            0.5 * jnp.sum(photometric_objective)
            + jnp.sum(gas_constraints)
            + prior_weight * (jnp.sum(default_prior) + jnp.sum(relation_prior))
            + physical_prior_weight * jnp.sum(physical_prior)
            + hyper
            + relation_hyper
        )

    value_and_grad = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4, 5))

    def single_mags(theta, base, model_args):
        params = _params_from_vectors(theta, base, parameter_names, free_indices_jax)
        model_mag = model_mags_jax_dynamic(context, model_args, params)
        if band_offsets_jax is not None:
            model_mag = model_mag + band_offsets_jax
        return model_mag

    batch_mags = jax.vmap(single_mags, in_axes=(0, 0, None))
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
        warm_observed_mag,
        warm_mask,
        lower,
        upper,
        prior_gaussian_mask,
        prior_gaussian_loc,
        prior_gaussian_scale,
        prior_beta_mask,
        prior_beta_alpha,
        prior_beta_beta,
        truth_theta,
        model_args,
    ):
        def batch_mags_with_args(theta, base):
            return batch_mags(theta, base, model_args)

        theta0 = _warm_start_amplitude(
            theta0,
            base_matrix,
            warm_observed_mag,
            warm_mask,
            lower,
            upper,
            batch_mags_with_args,
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
                warm_observed_mag,
                warm_mask,
                lower,
                upper,
                prior_gaussian_mask,
                prior_gaussian_loc,
                prior_gaussian_scale,
                prior_beta_mask,
                prior_beta_alpha,
                prior_beta_beta,
                model_args,
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
                    warm_observed_mag,
                    warm_mask,
                    lower,
                    upper,
                    prior_gaussian_mask,
                    prior_gaussian_loc,
                    prior_gaussian_scale,
                    prior_beta_mask,
                    prior_beta_alpha,
                    prior_beta_beta,
                    model_args,
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
                                    theta, base_matrix, observed, sigma, mask, model_args
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
        chi2 = batch_chi2_grad(best_theta, base_matrix, observed, sigma, mask, model_args)
        photometric_objective, grad = jax.vmap(
            jax.value_and_grad(single_photometric_objective, argnums=0),
            in_axes=(0, 0, 0, 0, 0, None),
        )(best_theta, base_matrix, observed, sigma, mask, model_args)
        model_mags = batch_mags(best_theta, base_matrix, model_args)
        return (
            best_theta,
            best_mu,
            sigma_pop,
            best_relation_intercept,
            best_relation_slope,
            relation_sigma,
            best_loss,
            chi2,
            photometric_objective,
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
            "mean_fit_quality": float(row[0]),
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
