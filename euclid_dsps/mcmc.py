"""MCMC posterior sampling for selected galaxies."""

# ruff: noqa: I001, E402

from __future__ import annotations

from dataclasses import dataclass
import sys
import time
from typing import Any

from .jax_runtime import configure_jax_runtime

configure_jax_runtime()

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from .fit import _initial_value, _photometric_likelihood, _student_t_dof
from .io import GalaxyObservation
from .model import (
    DspsContext,
    dynamic_model_args,
    gas_metallicity_constraint_penalty_jax,
    model_mags_jax_dynamic,
    predict_batch_derived,
    predict_batch_mags,
)
from .photometry import abmag_to_fnu_cgs_jax, magerr_to_fluxerr_fnu_cgs
from .posterior_target import (
    build_posterior_target,
    initial_unconstrained_position,
)


@dataclass(frozen=True)
class MCMCResult:
    samples: dict[str, np.ndarray]
    derived_samples: dict[str, np.ndarray]
    summary: list[dict[str, float | str]]
    posterior_model_mags: np.ndarray
    observed_mag: np.ndarray
    sigma_mag: np.ndarray
    observed_flux_fnu_cgs: np.ndarray
    flux_error_fnu_cgs: np.ndarray
    band_names: list[str]
    diagnostics: dict[str, Any]


def _numpyro_modules():
    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import HMC, MCMC, NUTS
        from numpyro.infer.initialization import init_to_value
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "NumPyro posterior samplers require a NumPyro version compatible "
            "with the active JAX installation. Use sample.sampler='mclmc' or "
            "install a compatible NumPyro/JAX pair."
        ) from exc
    return numpyro, dist, HMC, MCMC, NUTS, init_to_value


def _scaled_beta_distribution(dist, constraints):
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
            super().__init__(
                batch_shape=(), event_shape=(), validate_args=validate_args
            )

        @constraints.dependent_property
        def support(self):
            return constraints.interval(self.low, self.high)

        def sample(self, key, sample_shape=()):
            unit = self._beta.sample(key, sample_shape)
            return self.low + (self.high - self.low) * unit

        def log_prob(self, value):
            unit = (value - self.low) / (self.high - self.low)
            return self._beta.log_prob(unit) - jnp.log(self.high - self.low)

    return ScaledBetaDistribution


def sample_one_galaxy(
    context: DspsContext,
    observation: GalaxyObservation,
    base_params: dict[str, float],
    fit_config: dict[str, Any],
    sample_config: dict[str, Any],
    initial_params: dict[str, float] | None = None,
) -> MCMCResult:
    """Sample posterior over configured free parameters."""
    sampler = str(sample_config.get("sampler", "nuts")).lower()
    if sampler == "mclmc":
        return _sample_one_galaxy_mclmc(
            context,
            observation,
            base_params,
            fit_config,
            sample_config,
            initial_params=initial_params,
        )

    numpyro, dist, _HMC, MCMC, _NUTS, init_to_value = _numpyro_modules()
    observed_mag = jnp.asarray([band.mag_ab for band in observation.bands], dtype=float)
    sigma_mag = jnp.asarray([band.sigma_mag for band in observation.bands], dtype=float)
    observed_flux = jnp.asarray(
        [band.flux_fnu_cgs for band in observation.bands], dtype=float
    )
    flux_error = jnp.asarray(
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
    likelihood_space = str(fit_config.get("likelihood_space", "flux")).lower()
    photometric_likelihood = _photometric_likelihood(fit_config)
    student_t_dof = _student_t_dof(fit_config)
    if likelihood_space == "flux":
        floor_frac = float(fit_config.get("flux_error_floor_frac", 0.0))
        jitter = float(fit_config.get("flux_error_jitter", 0.0))
        observed = observed_flux
        sigma = jnp.sqrt(flux_error**2 + (floor_frac * observed_flux) ** 2 + jitter**2)
        finite = jnp.isfinite(observed) & jnp.isfinite(sigma) & (sigma > 0)
    else:
        observed = observed_mag
        sigma = sigma_mag
        finite = jnp.isfinite(observed_mag) & jnp.isfinite(sigma_mag) & (sigma_mag > 0)
    band_names = [band.name for band in observation.bands]
    band_offsets = jnp.asarray(
        fit_config.get("band_calibration_offsets_mag", []), dtype=float
    )

    free = fit_config["free_parameters"]
    free_names = list(free)
    priors = sample_config.get("priors", {})
    model_args = dynamic_model_args(context)

    def model(model_args):
        params = {key: jnp.asarray(value) for key, value in base_params.items()}
        for name in free_names:
            prior_spec = priors.get(name, {})
            params[name] = numpyro.sample(
                name,
                _prior_distribution(name, free[name], prior_spec, base_params),
            )
        numpyro.factor(
            "gas_metallicity_constraint",
            -gas_metallicity_constraint_penalty_jax(
                params, context.model_config, penalty=jnp.inf
            ),
        )
        model_mag = model_mags_jax_dynamic(context, model_args, params)
        if band_offsets.size:
            model_mag = model_mag + band_offsets
        numpyro.deterministic("model_mag", model_mag)
        if likelihood_space == "flux":
            model_obs = abmag_to_fnu_cgs_jax(model_mag)
            numpyro.deterministic("model_flux_fnu_cgs", model_obs)
        else:
            model_obs = model_mag
        if photometric_likelihood == "student_t":
            obs_dist = dist.StudentT(student_t_dof, loc=model_obs, scale=sigma)
        else:
            obs_dist = dist.Normal(model_obs, sigma)
        numpyro.sample("obs", obs_dist.mask(finite), obs=observed)

    init_params = _initial_params(initial_params, free, free_names)
    kernel_kwargs = {}
    if init_params:
        kernel_kwargs["init_strategy"] = init_to_value(values=init_params)
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
        model_args,
        extra_fields=("diverging", "accept_prob", "num_steps"),
    )
    samples = {
        name: np.asarray(values)
        for name, values in mcmc.get_samples().items()
        if name in free_names
    }
    posterior_model_mags = _posterior_model_mags(
        context, base_params, samples, fit_config
    )
    derived_samples = _posterior_derived(context, base_params, samples)
    return MCMCResult(
        samples=samples,
        derived_samples=derived_samples,
        summary=_sample_summary(samples),
        posterior_model_mags=posterior_model_mags,
        observed_mag=np.asarray(observed_mag),
        sigma_mag=np.asarray(sigma_mag),
        observed_flux_fnu_cgs=np.asarray(observed_flux),
        flux_error_fnu_cgs=np.asarray(flux_error),
        band_names=band_names,
        diagnostics=_diagnostics(
            mcmc,
            sample_config=sample_config,
            sampler=sampler,
            initial_params=init_params,
            likelihood_space=likelihood_space,
            photometric_likelihood=photometric_likelihood,
            student_t_dof=student_t_dof,
        ),
    )


def _sample_one_galaxy_mclmc(
    context: DspsContext,
    observation: GalaxyObservation,
    base_params: dict[str, float],
    fit_config: dict[str, Any],
    sample_config: dict[str, Any],
    initial_params: dict[str, float] | None = None,
) -> MCMCResult:
    """Sample one posterior with experimental BlackJAX MCLMC."""
    try:
        import blackjax
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "sample.sampler='mclmc' requires BlackJAX. Install the optional "
            "samplers extra or install blackjax in this environment."
        ) from exc

    observed_mag = np.asarray([band.mag_ab for band in observation.bands], dtype=float)
    sigma_mag = np.asarray([band.sigma_mag for band in observation.bands], dtype=float)
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
    model_args = dynamic_model_args(context)
    target = build_posterior_target(
        context=context,
        model_args=model_args,
        base_params=base_params,
        fit_config=fit_config,
        sample_config=sample_config,
        observed_mag=observed_mag,
        sigma_mag=sigma_mag,
        observed_flux=observed_flux,
        flux_error=flux_error,
    )
    y0 = initial_unconstrained_position(target, initial_params, fit_config)
    num_warmup = int(sample_config.get("num_warmup", 100))
    num_samples = int(sample_config.get("num_samples", 200))
    seed = int(sample_config.get("seed", 42))
    progress_bar = bool(sample_config.get("progress_bar", True))
    debug = bool(sample_config.get("mclmc_debug", False))
    progress_chunk_size = int(sample_config.get("mclmc_progress_chunk_size", 16))
    if progress_chunk_size <= 0:
        raise ValueError("sample.mclmc_progress_chunk_size must be positive")
    dim = int(y0.shape[0])
    L = _resolve_mclmc_float(
        sample_config.get("mclmc_l", sample_config.get("L")), np.sqrt(float(dim))
    )
    step_size = _resolve_mclmc_float(
        sample_config.get("mclmc_step_size", sample_config.get("step_size")),
        min(0.10, 1.0 / np.sqrt(float(dim))),
    )
    inverse_mass_matrix = _mclmc_inverse_mass_matrix(sample_config, dim)

    algorithm = blackjax.mclmc(
        logdensity_fn=target.logdensity,
        L=L,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        desired_energy_var_max_ratio=float(
            sample_config.get("mclmc_desired_energy_var_max_ratio", np.inf)
        ),
    )
    rng_key = random.PRNGKey(seed)
    init_key, compile_key, warmup_key, sample_key = random.split(rng_key, 4)
    _mclmc_log(
        progress_bar or debug,
        "backend="
        f"{jax.default_backend()} devices={_jax_device_strings()} "
        f"dim={dim} bands={len(observation.bands)} warmup={num_warmup} "
        f"samples={num_samples} L={L:.6g} step_size={step_size:.6g} "
        f"chunk={progress_chunk_size}",
    )
    _mclmc_log(progress_bar or debug, "compiling BlackJAX MCLMC step")
    compile_start = time.perf_counter()
    state = algorithm.init(y0, init_key)
    # Keep the raw BlackJAX step inside lax.scan. A separately jitted step nested
    # in scan can trigger CUDA graph-capture failures with this forward model.
    step_fn = algorithm.step
    probe_steps = min(
        progress_chunk_size,
        max(num_warmup if num_warmup > 0 else 0, num_samples, 1),
    )
    compiled_state, _ = _scan_mclmc(step_fn, state, compile_key, probe_steps)
    jax.block_until_ready(compiled_state.position)
    compile_time = time.perf_counter() - compile_start
    _mclmc_log(progress_bar or debug, f"compile done in {compile_time:.3f}s")

    warmup_start = time.perf_counter()
    state, warmup_info = _run_mclmc_steps(
        step_fn,
        state,
        warmup_key,
        num_warmup,
        phase="warmup",
        progress_bar=progress_bar,
        debug=debug,
        chunk_size=progress_chunk_size,
    )
    jax.block_until_ready(state.position)
    warmup_time = time.perf_counter() - warmup_start
    _mclmc_log(progress_bar or debug, f"warmup done in {warmup_time:.3f}s")

    sample_start = time.perf_counter()
    state, sample_info = _run_mclmc_steps(
        step_fn,
        state,
        sample_key,
        num_samples,
        phase="sample",
        progress_bar=progress_bar,
        debug=debug,
        chunk_size=progress_chunk_size,
    )
    positions = sample_info["position"]
    jax.block_until_ready(positions)
    sampling_time = time.perf_counter() - sample_start
    _mclmc_log(progress_bar or debug, f"sampling done in {sampling_time:.3f}s")

    theta = jax.vmap(target.theta_from_unconstrained)(positions)
    theta_np = np.asarray(theta)
    samples = {
        name: theta_np[:, index]
        for index, name in enumerate(target.free_names)
    }
    posterior_model_mags = _posterior_model_mags(
        context, base_params, samples, fit_config
    )
    derived_samples = _posterior_derived(context, base_params, samples)
    photometric_likelihood = _photometric_likelihood(fit_config)
    return MCMCResult(
        samples=samples,
        derived_samples=derived_samples,
        summary=_sample_summary(samples),
        posterior_model_mags=posterior_model_mags,
        observed_mag=np.asarray(observed_mag),
        sigma_mag=np.asarray(sigma_mag),
        observed_flux_fnu_cgs=np.asarray(observed_flux),
        flux_error_fnu_cgs=np.asarray(flux_error),
        band_names=[band.name for band in observation.bands],
        diagnostics=_mclmc_diagnostics(
            sample_config=sample_config,
            initial_params=initial_params,
            likelihood_space=target.likelihood_space,
            photometric_likelihood=photometric_likelihood,
            student_t_dof=_student_t_dof(fit_config),
            num_warmup=num_warmup,
            num_samples=num_samples,
            L=L,
            step_size=step_size,
            progress_chunk_size=progress_chunk_size,
            compile_time=compile_time,
            warmup_time=warmup_time,
            sampling_time=sampling_time,
            warmup_info=warmup_info,
            sample_info=sample_info,
        ),
    )


def _scan_mclmc(step_fn, state, rng_key, num_steps: int):
    if num_steps <= 0:
        return state, _empty_mclmc_info(state)
    return _scan_mclmc_keys(step_fn, state, random.split(rng_key, num_steps))


def _scan_mclmc_keys(step_fn, state, keys):
    if keys.shape[0] <= 0:
        return state, _empty_mclmc_info(state)

    def one_step(carry, key):
        new_state, info = step_fn(key, carry)
        return new_state, (
            new_state.position,
            info.logdensity,
            info.energy_change,
            info.kinetic_change,
            info.nonans,
        )

    final_state, values = jax.lax.scan(one_step, state, keys)
    position, logdensity, energy_change, kinetic_change, nonans = values
    return final_state, {
        "position": position,
        "logdensity": logdensity,
        "energy_change": energy_change,
        "kinetic_change": kinetic_change,
        "nonans": nonans,
    }


def _run_mclmc_steps(
    step_fn,
    state,
    rng_key,
    num_steps: int,
    *,
    phase: str,
    progress_bar: bool,
    debug: bool,
    chunk_size: int,
):
    if num_steps <= 0:
        return state, _empty_mclmc_info(state)
    if chunk_size <= 0:
        raise ValueError("sample.mclmc_progress_chunk_size must be positive")
    if not progress_bar and not debug:
        return _scan_mclmc(step_fn, state, rng_key, num_steps)

    keys = random.split(rng_key, num_steps)
    chunks: dict[str, list[jnp.ndarray]] = {
        "position": [],
        "logdensity": [],
        "energy_change": [],
        "kinetic_change": [],
        "nonans": [],
    }
    pbar = _mclmc_progress_bar(
        enabled=progress_bar,
        total=num_steps,
        desc=f"mclmc-{phase}",
    )
    try:
        for start in range(0, num_steps, chunk_size):
            end = min(start + chunk_size, num_steps)
            state, info = _scan_mclmc_keys(step_fn, state, keys[start:end])
            jax.block_until_ready(info["position"])
            for name, values in info.items():
                chunks[name].append(values)
            if pbar is not None:
                pbar.update(end - start)
            if debug:
                _mclmc_log(
                    True,
                    f"{phase} {end}/{num_steps} "
                    f"logdensity={float(info['logdensity'][-1]):.6g} "
                    f"nonans={float(jnp.mean(info['nonans'])):.3f}",
                )
    finally:
        if pbar is not None:
            pbar.close()

    return state, {
        name: jnp.concatenate(values, axis=0)
        for name, values in chunks.items()
    }


def _empty_mclmc_info(state):
    empty = jnp.empty((0,) + state.position.shape, dtype=state.position.dtype)
    return {
        "position": empty,
        "logdensity": jnp.empty((0,), dtype=state.logdensity.dtype),
        "energy_change": jnp.empty((0,), dtype=state.logdensity.dtype),
        "kinetic_change": jnp.empty((0,), dtype=state.logdensity.dtype),
        "nonans": jnp.empty((0,), dtype=bool),
    }


def _mclmc_progress_bar(*, enabled: bool, total: int, desc: str):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - only if tqdm is absent
        return None
    return tqdm(total=total, desc=desc, unit="step", dynamic_ncols=True)


def _mclmc_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[mclmc] {message}", file=sys.stderr, flush=True)


def _jax_device_strings() -> list[str]:
    return [f"{device.platform}:{device.id}" for device in jax.devices()]


def _resolve_mclmc_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        if value.lower() == "auto":
            return float(default)
        return float(value)
    return float(value)


def _mclmc_inverse_mass_matrix(sample_config: dict[str, Any], dim: int):
    raw = sample_config.get("mclmc_inverse_mass_matrix")
    if raw is None:
        return 1.0
    if isinstance(raw, str):
        if raw.lower() == "identity":
            return 1.0
        if raw.lower() == "ones":
            return jnp.ones((dim,), dtype=jnp.float32)
    arr = jnp.asarray(raw, dtype=jnp.float32)
    if arr.shape == ():
        return float(arr)
    if arr.shape != (dim,):
        raise ValueError(
            "sample.mclmc_inverse_mass_matrix must be scalar or length "
            f"{dim}, got shape {arr.shape}"
        )
    return arr


def _build_kernel(
    model,
    sampler: str,
    sample_config: dict[str, Any],
    kernel_kwargs: dict[str, Any],
):
    _numpyro, _dist, HMC, _MCMC, NUTS, _init_to_value = _numpyro_modules()
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
    raise ValueError(
        f"Unsupported MCMC sampler: {sampler}. Use 'nuts', 'hmc', or 'mclmc'."
    )


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
    _numpyro, dist, _HMC, _MCMC, _NUTS, _init_to_value = _numpyro_modules()
    low, high = [float(value) for value in fit_spec["bounds"]]
    loc = _prior_location(name, fit_spec, prior_spec, base_params)
    scale = _prior_scale(name, prior_spec, base_params, max((high - low) / 4.0, 1.0e-3))
    prior_type = str(prior_spec.get("type", "truncated_normal"))
    if prior_type == "uniform":
        return dist.Uniform(low, high)
    if prior_type == "normal":
        return dist.Normal(loc, scale)
    if prior_type == "truncated_normal":
        return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)
    if prior_type == "scaled_beta":
        from numpyro.distributions import constraints

        scaled_beta = _scaled_beta_distribution(dist, constraints)
        alpha = float(prior_spec.get("alpha", 1.0))
        beta = float(prior_spec.get("beta", 1.0))
        return scaled_beta(alpha=alpha, beta=beta, low=low, high=high)
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


def _prior_scale(
    name: str,
    prior_spec: dict[str, Any],
    base_params: dict[str, float],
    fallback: float,
) -> float:
    value = prior_spec.get("scale", fallback)
    if value == "from_base":
        scale_name = str(prior_spec.get("scale_parameter", f"{name}_prior_sigma"))
        return max(float(base_params.get(scale_name, fallback)), 1.0e-6)
    return max(float(value), 1.0e-6)


def _posterior_model_mags(
    context: DspsContext,
    base_params: dict[str, float],
    samples: dict[str, np.ndarray],
    fit_config: dict[str, Any],
) -> np.ndarray:
    parameter_names, matrix = _posterior_parameter_matrix(base_params, samples)
    mags = predict_batch_mags(context, parameter_names, matrix)
    offsets = np.asarray(fit_config.get("band_calibration_offsets_mag", []), dtype=float)
    if offsets.size:
        mags = mags + offsets
    return mags


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
    mcmc,
    sample_config: dict[str, Any],
    sampler: str,
    initial_params: dict[str, jnp.ndarray] | None = None,
    likelihood_space: str = "flux",
    photometric_likelihood: str = "gaussian",
    student_t_dof: float = 2.0,
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
    diagnostics["likelihood_space"] = likelihood_space
    diagnostics["photometric_likelihood"] = photometric_likelihood
    diagnostics["student_t_dof"] = float(student_t_dof)
    diagnostics["num_warmup"] = int(sample_config.get("num_warmup", 100))
    diagnostics["num_chains"] = int(sample_config.get("num_chains", 1))
    diagnostics["chain_method"] = str(sample_config.get("chain_method", "parallel"))
    if sampler == "nuts":
        diagnostics["max_tree_depth"] = int(sample_config.get("max_tree_depth", 10))
    if sampler == "hmc":
        diagnostics["num_steps"] = int(sample_config.get("num_steps", 8))
    diagnostics["jax_backend"] = str(jax.default_backend())
    diagnostics["jax_devices"] = _jax_device_strings()
    diagnostics["device"] = f"{jax.devices()[0].platform}:{jax.devices()[0].id}"
    diagnostics["initialized_from_map"] = bool(initial_params)
    if initial_params:
        diagnostics["initial_parameters"] = {
            name: float(value) for name, value in initial_params.items()
        }
    return diagnostics


def _mclmc_diagnostics(
    *,
    sample_config: dict[str, Any],
    initial_params: dict[str, float] | None,
    likelihood_space: str,
    photometric_likelihood: str,
    student_t_dof: float,
    num_warmup: int,
    num_samples: int,
    L: float,
    step_size: float,
    progress_chunk_size: int,
    compile_time: float,
    warmup_time: float,
    sampling_time: float,
    warmup_info: dict[str, jnp.ndarray],
    sample_info: dict[str, jnp.ndarray],
) -> dict[str, Any]:
    energy = np.asarray(sample_info["energy_change"])
    kinetic = np.asarray(sample_info["kinetic_change"])
    nonans = np.asarray(sample_info["nonans"])
    warmup_nonans = np.asarray(warmup_info["nonans"])
    diagnostics: dict[str, Any] = {
        "backend": "blackjax_mclmc",
        "sampler": "mclmc",
        "likelihood_space": likelihood_space,
        "photometric_likelihood": photometric_likelihood,
        "student_t_dof": float(student_t_dof),
        "num_warmup": int(num_warmup),
        "num_samples": int(num_samples),
        "num_chains": 1,
        "chain_method": "single",
        "mclmc_l": float(L),
        "mclmc_step_size": float(step_size),
        "mclmc_progress_chunk_size": int(progress_chunk_size),
        "compile_time_s": float(compile_time),
        "warmup_time_s": float(warmup_time),
        "sampling_time_s": float(sampling_time),
        "samples_per_second": float(num_samples / sampling_time)
        if sampling_time > 0.0
        else float("inf"),
        "jax_backend": str(jax.default_backend()),
        "jax_devices": _jax_device_strings(),
        "device": f"{jax.devices()[0].platform}:{jax.devices()[0].id}",
        "initialized_from_map": bool(initial_params),
        "experimental_warning": (
            "BlackJAX unadjusted MCLMC is experimental here and must be "
            "benchmarked against HMC/NUTS before science use."
        ),
    }
    if energy.size:
        abs_energy = np.abs(energy[np.isfinite(energy)])
        if abs_energy.size:
            diagnostics["mean_abs_energy_change"] = float(abs_energy.mean())
            diagnostics["p95_abs_energy_change"] = float(np.quantile(abs_energy, 0.95))
    if kinetic.size:
        abs_kinetic = np.abs(kinetic[np.isfinite(kinetic)])
        if abs_kinetic.size:
            diagnostics["mean_abs_kinetic_change"] = float(abs_kinetic.mean())
            diagnostics["p95_abs_kinetic_change"] = float(
                np.quantile(abs_kinetic, 0.95)
            )
    if nonans.size:
        diagnostics["fraction_nonans"] = float(nonans.mean())
    if warmup_nonans.size:
        diagnostics["warmup_fraction_nonans"] = float(warmup_nonans.mean())
    if initial_params:
        diagnostics["initial_parameters"] = {
            name: float(value) for name, value in initial_params.items()
        }
    return diagnostics
