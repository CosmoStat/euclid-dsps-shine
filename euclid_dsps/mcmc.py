"""MCMC posterior sampling for selected galaxies."""

# ruff: noqa: I001, E402

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.distributions import constraints
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from .fit import _initial_value
from .io import GalaxyObservation
from .model import (
    DspsContext,
    model_mags_jax,
    predict_batch_derived,
    predict_batch_mags,
)


class ScaledBetaDistribution(dist.Distribution):
    """Beta distribution scaled to a finite interval."""

    arg_constraints = {
        "alpha": constraints.positive,
        "beta": constraints.positive,
        "low": constraints.real,
        "high": constraints.real,
    }
    reparametrized_params = ["alpha", "beta"]

    def __init__(
        self,
        alpha: float,
        beta: float,
        low: float,
        high: float,
        validate_args: bool | None = None,
    ):
        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        self._beta = dist.Beta(self.alpha, self.beta)
        super().__init__(batch_shape=(), event_shape=(), validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)

    def sample(self, key, sample_shape=()):
        unit = self._beta.sample(key, sample_shape)
        return self.low + (self.high - self.low) * unit

    def log_prob(self, value):
        unit = (value - self.low) / (self.high - self.low)
        return self._beta.log_prob(unit) - jnp.log(self.high - self.low)


@dataclass(frozen=True)
class MCMCResult:
    samples: dict[str, np.ndarray]
    derived_samples: dict[str, np.ndarray]
    summary: list[dict[str, float | str]]
    posterior_model_mags: np.ndarray
    observed_mag: np.ndarray
    sigma_mag: np.ndarray
    band_names: list[str]
    diagnostics: dict[str, Any]


def sample_one_galaxy(
    context: DspsContext,
    observation: GalaxyObservation,
    base_params: dict[str, float],
    fit_config: dict[str, Any],
    sample_config: dict[str, Any],
    initial_params: dict[str, float] | None = None,
) -> MCMCResult:
    """Sample posterior over configured free parameters with NumPyro HMC/NUTS."""
    observed_mag = jnp.asarray([band.mag_ab for band in observation.bands], dtype=float)
    sigma_mag = jnp.asarray([band.sigma_mag for band in observation.bands], dtype=float)
    finite = jnp.isfinite(observed_mag) & jnp.isfinite(sigma_mag) & (sigma_mag > 0)
    band_names = [band.name for band in observation.bands]

    free = fit_config["free_parameters"]
    free_names = list(free)
    priors = sample_config.get("priors", {})

    def model():
        params = {key: jnp.asarray(value) for key, value in base_params.items()}
        for name in free_names:
            prior_spec = priors.get(name, {})
            params[name] = numpyro.sample(
                name,
                _prior_distribution(name, free[name], prior_spec, base_params),
            )
        model_mag = model_mags_jax(context, params)
        numpyro.deterministic("model_mag", model_mag)
        numpyro.sample(
            "obs", dist.Normal(model_mag, sigma_mag).mask(finite), obs=observed_mag
        )

    init_params = _initial_params(initial_params, free, free_names)
    kernel_kwargs = {}
    if init_params:
        kernel_kwargs["init_strategy"] = init_to_value(values=init_params)
    sampler = str(sample_config.get("sampler", "nuts")).lower()
    kernel = _build_kernel(model, sampler, sample_config, kernel_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=int(sample_config.get("num_warmup", 100)),
        num_samples=int(sample_config.get("num_samples", 200)),
        num_chains=int(sample_config.get("num_chains", 1)),
        chain_method=str(sample_config.get("chain_method", "parallel")),
        progress_bar=bool(sample_config.get("progress_bar", True)),
        jit_model_args=bool(sample_config.get("jit_model_args", False)),
    )
    mcmc.run(
        random.PRNGKey(int(sample_config.get("seed", 42))),
        extra_fields=("diverging", "accept_prob", "num_steps"),
    )
    samples = {
        name: np.asarray(values)
        for name, values in mcmc.get_samples().items()
        if name in free_names
    }
    posterior_model_mags = _posterior_model_mags(context, base_params, samples)
    derived_samples = _posterior_derived(context, base_params, samples)
    return MCMCResult(
        samples=samples,
        derived_samples=derived_samples,
        summary=_sample_summary(samples),
        posterior_model_mags=posterior_model_mags,
        observed_mag=np.asarray(observed_mag),
        sigma_mag=np.asarray(sigma_mag),
        band_names=band_names,
        diagnostics=_diagnostics(
            mcmc,
            sample_config=sample_config,
            sampler=sampler,
            initial_params=init_params,
        ),
    )


def _build_kernel(
    model,
    sampler: str,
    sample_config: dict[str, Any],
    kernel_kwargs: dict[str, Any],
):
    common_kwargs = {
        "target_accept_prob": float(sample_config.get("target_accept_prob", 0.85)),
        "dense_mass": bool(sample_config.get("dense_mass", False)),
        **kernel_kwargs,
    }
    step_size = sample_config.get("step_size")
    if step_size is not None:
        common_kwargs["step_size"] = float(step_size)

    if sampler == "nuts":
        return NUTS(
            model,
            max_tree_depth=int(sample_config.get("max_tree_depth", 10)),
            **common_kwargs,
        )
    if sampler == "hmc":
        hmc_kwargs = dict(common_kwargs)
        num_steps = sample_config.get("num_steps")
        trajectory_length = sample_config.get("trajectory_length")
        if num_steps is not None:
            hmc_kwargs["num_steps"] = int(num_steps)
            hmc_kwargs["trajectory_length"] = None
        elif trajectory_length is not None:
            hmc_kwargs["trajectory_length"] = float(trajectory_length)
        else:
            hmc_kwargs["num_steps"] = 8
            hmc_kwargs["trajectory_length"] = None
        return HMC(model, **hmc_kwargs)
    raise ValueError(f"Unsupported MCMC sampler: {sampler}. Use 'nuts' or 'hmc'.")


def _initial_params(
    initial_params: dict[str, float] | None,
    free: dict[str, Any],
    free_names: list[str],
) -> dict[str, jnp.ndarray] | None:
    if not initial_params:
        return None
    values = {}
    for name in free_names:
        if name not in initial_params or not np.isfinite(initial_params[name]):
            return None
        low, high = [float(value) for value in free[name]["bounds"]]
        eps = max((high - low) * 1.0e-6, 1.0e-8)
        values[name] = jnp.asarray(
            np.clip(float(initial_params[name]), low + eps, high - eps)
        )
    return values


def _prior_distribution(
    name: str,
    fit_spec: dict[str, Any],
    prior_spec: dict[str, Any],
    base_params: dict[str, float],
):
    low, high = [float(value) for value in fit_spec["bounds"]]
    loc = _prior_location(name, fit_spec, prior_spec, base_params)
    scale = float(prior_spec.get("scale", max((high - low) / 4.0, 1.0e-3)))
    prior_type = str(prior_spec.get("type", "truncated_normal"))
    if prior_type == "uniform":
        return dist.Uniform(low, high)
    if prior_type == "normal":
        return dist.Normal(loc, scale)
    if prior_type == "truncated_normal":
        return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
    if prior_type == "scaled_beta":
        alpha = float(prior_spec.get("alpha", 1.0))
        beta = float(prior_spec.get("beta", 1.0))
        return ScaledBetaDistribution(alpha=alpha, beta=beta, low=low, high=high)
    raise ValueError(f"Unsupported prior type for {name}: {prior_type}")


def _prior_location(
    name: str,
    fit_spec: dict[str, Any],
    prior_spec: dict[str, Any],
    base_params: dict[str, float],
) -> float:
    value = prior_spec.get("loc", _initial_value(fit_spec, name, base_params))
    if value == "from_base":
        return float(base_params[name])
    return float(value)


def _posterior_model_mags(
    context: DspsContext, base_params: dict[str, float], samples: dict[str, np.ndarray]
) -> np.ndarray:
    parameter_names, matrix = _posterior_parameter_matrix(base_params, samples)
    return predict_batch_mags(context, parameter_names, matrix)


def _posterior_derived(
    context: DspsContext, base_params: dict[str, float], samples: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    parameter_names, matrix = _posterior_parameter_matrix(base_params, samples)
    return predict_batch_derived(context, parameter_names, matrix)


def _posterior_parameter_matrix(
    base_params: dict[str, float], samples: dict[str, np.ndarray]
) -> tuple[list[str], np.ndarray]:
    parameter_names = list(base_params)
    n_samples = len(next(iter(samples.values())))
    matrix = np.asarray(
        [[float(base_params[name]) for name in parameter_names]] * n_samples,
        dtype=float,
    )
    for name, values in samples.items():
        matrix[:, parameter_names.index(name)] = values
    return parameter_names, matrix


def _sample_summary(samples: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    rows = []
    for name, values in samples.items():
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "parameter": name,
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "q05": float(np.quantile(finite, 0.05)),
                "q16": float(np.quantile(finite, 0.16)),
                "median": float(np.quantile(finite, 0.50)),
                "q84": float(np.quantile(finite, 0.84)),
                "q95": float(np.quantile(finite, 0.95)),
            }
        )
    return rows


def _diagnostics(
    mcmc: MCMC,
    sample_config: dict[str, Any],
    sampler: str,
    initial_params: dict[str, jnp.ndarray] | None = None,
) -> dict[str, Any]:
    extra = mcmc.get_extra_fields()
    diagnostics: dict[str, Any] = {}
    if "diverging" in extra:
        diagnostics["n_divergent"] = int(np.asarray(extra["diverging"]).sum())
    if "accept_prob" in extra:
        diagnostics["mean_accept_prob"] = float(np.asarray(extra["accept_prob"]).mean())
    if "num_steps" in extra:
        num_steps = np.asarray(extra["num_steps"])
        diagnostics["mean_num_steps"] = float(np.mean(num_steps))
        diagnostics["max_num_steps"] = int(np.max(num_steps))
    diagnostics["n_samples"] = int(len(next(iter(mcmc.get_samples().values()))))
    diagnostics["backend"] = f"numpyro_{sampler}"
    diagnostics["sampler"] = sampler
    diagnostics["num_warmup"] = int(sample_config.get("num_warmup", 100))
    diagnostics["num_chains"] = int(sample_config.get("num_chains", 1))
    diagnostics["chain_method"] = str(sample_config.get("chain_method", "parallel"))
    if sampler == "nuts":
        diagnostics["max_tree_depth"] = int(sample_config.get("max_tree_depth", 10))
    if sampler == "hmc":
        diagnostics["num_steps"] = int(sample_config.get("num_steps", 8))
    diagnostics["device"] = f"{jax.devices()[0].platform}:{jax.devices()[0].id}"
    diagnostics["initialized_from_map"] = bool(initial_params)
    if initial_params:
        diagnostics["initial_parameters"] = {
            name: float(value) for name, value in initial_params.items()
        }
    return diagnostics
