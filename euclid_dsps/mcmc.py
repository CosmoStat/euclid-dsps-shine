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
import jax.scipy as jsp
import jax.scipy.stats as jstats
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
    BoundedParameterTransform,
    _gas_metallicity_constraint_indices,
    _masked_observation_logprob,
    _resolved_prior_spec,
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
    chain_ids: np.ndarray | None = None


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
        context,
        base_params,
        samples,
        fit_config,
        batch_size=int(sample_config.get("posterior_predictive_batch_size", 512)),
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
    num_warmup = int(sample_config.get("num_warmup", 100))
    num_samples = int(sample_config.get("num_samples", 200))
    num_chains = int(sample_config.get("num_chains", 1))
    if num_chains <= 0:
        raise ValueError("sample.num_chains must be positive")
    seed = int(sample_config.get("seed", 42))
    progress_bar = bool(sample_config.get("progress_bar", True))
    debug = bool(sample_config.get("mclmc_debug", False))
    progress_chunk_size = int(sample_config.get("mclmc_progress_chunk_size", 16))
    if progress_chunk_size <= 0:
        raise ValueError("sample.mclmc_progress_chunk_size must be positive")
    dim = len(target.free_names)
    L = _resolve_mclmc_float(
        sample_config.get("mclmc_l", sample_config.get("L")), np.sqrt(float(dim))
    )
    step_size = _resolve_mclmc_float(
        sample_config.get("mclmc_step_size", sample_config.get("step_size")),
        min(0.10, 1.0 / np.sqrt(float(dim))),
    )
    inverse_mass_matrix = _mclmc_batch_inverse_mass_matrix(sample_config, 1, dim)

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
    chain_keys = random.split(rng_key, num_chains)
    init_position_keys = random.split(random.fold_in(rng_key, 87231), num_chains)
    init_strategy = _mclmc_init_strategy(sample_config, initial_params)
    chain_y0s = _mclmc_initial_positions(
        target,
        initial_params,
        fit_config,
        sample_config,
        init_position_keys,
        init_strategy=init_strategy,
    )
    chain_initial_theta = np.asarray(
        jax.vmap(target.theta_from_unconstrained)(chain_y0s)
    )
    _mclmc_log(
        progress_bar or debug,
        "backend="
        f"{jax.default_backend()} devices={_jax_device_strings()} "
        f"dim={dim} bands={len(observation.bands)} chains={num_chains} "
        f"warmup={num_warmup} samples={num_samples} "
        f"L={L:.6g} step_size={step_size:.6g} "
        f"chunk={progress_chunk_size} init={init_strategy}",
    )
    _mclmc_log(progress_bar or debug, "compiling BlackJAX MCLMC step")
    # Keep the raw BlackJAX step inside lax.scan. A separately jitted step nested
    # in scan can trigger CUDA graph-capture failures with this forward model.
    step_fn = algorithm.step
    probe_steps = min(
        progress_chunk_size,
        max(num_warmup if num_warmup > 0 else 0, num_samples, 1),
    )
    first_init_key, first_compile_key = random.split(chain_keys[0])
    first_state = algorithm.init(chain_y0s[0], first_init_key)
    compile_start = time.perf_counter()
    compiled_state, _ = _scan_mclmc(
        step_fn, first_state, first_compile_key, probe_steps
    )
    jax.block_until_ready(compiled_state.position)
    compile_time = time.perf_counter() - compile_start
    _mclmc_log(progress_bar or debug, f"compile done in {compile_time:.3f}s")

    theta_chains: list[np.ndarray] = []
    warmup_infos = []
    sample_infos = []
    chain_summaries = []
    total_warmup_time = 0.0
    total_sampling_time = 0.0
    for chain_index, chain_key in enumerate(chain_keys):
        init_key, warmup_key, sample_key = random.split(chain_key, 3)
        state = algorithm.init(chain_y0s[chain_index], init_key)
        chain_label = f"chain {chain_index + 1}/{num_chains}"
        _mclmc_log(progress_bar or debug, f"{chain_label} warmup start")
        warmup_start = time.perf_counter()
        state, warmup_info = _run_mclmc_steps(
            step_fn,
            state,
            warmup_key,
            num_warmup,
            phase=f"mclmc-c{chain_index + 1}-warmup",
            progress_bar=progress_bar,
            debug=debug,
            chunk_size=progress_chunk_size,
        )
        _raise_if_mclmc_all_invalid(warmup_info, phase=f"{chain_label} warmup")
        jax.block_until_ready(state.position)
        warmup_time = time.perf_counter() - warmup_start
        total_warmup_time += warmup_time
        _mclmc_log(
            progress_bar or debug,
            f"{chain_label} warmup done in {warmup_time:.3f}s",
        )

        _mclmc_log(progress_bar or debug, f"{chain_label} sampling start")
        sample_start = time.perf_counter()
        state, sample_info = _run_mclmc_steps(
            step_fn,
            state,
            sample_key,
            num_samples,
            phase=f"mclmc-c{chain_index + 1}-sample",
            progress_bar=progress_bar,
            debug=debug,
            chunk_size=progress_chunk_size,
        )
        _raise_if_mclmc_all_invalid(sample_info, phase=f"{chain_label} sampling")
        positions = sample_info["position"]
        jax.block_until_ready(positions)
        sampling_time = time.perf_counter() - sample_start
        total_sampling_time += sampling_time
        _mclmc_log(
            progress_bar or debug,
            f"{chain_label} sampling done in {sampling_time:.3f}s",
        )
        theta = jax.vmap(target.theta_from_unconstrained)(positions)
        theta_chains.append(np.asarray(theta))
        warmup_infos.append(warmup_info)
        sample_infos.append(sample_info)
        chain_summaries.append(
            _mclmc_chain_summary(
                chain_index=chain_index,
                warmup_time=warmup_time,
                sampling_time=sampling_time,
                warmup_info=warmup_info,
                sample_info=sample_info,
            )
        )

    theta_np = np.concatenate(theta_chains, axis=0)
    chain_ids = np.repeat(np.arange(num_chains, dtype=np.int32), num_samples)
    warmup_info = _concat_mclmc_infos(warmup_infos)
    sample_info = _concat_mclmc_infos(sample_infos)
    samples = {name: theta_np[:, index] for index, name in enumerate(target.free_names)}
    posterior_model_mags = _posterior_model_mags(
        context,
        base_params,
        samples,
        fit_config,
        batch_size=int(sample_config.get("posterior_predictive_batch_size", 512)),
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
            num_chains=num_chains,
            L=L,
            step_size=step_size,
            progress_chunk_size=progress_chunk_size,
            compile_time=compile_time,
            warmup_time=total_warmup_time,
            sampling_time=total_sampling_time,
            warmup_info=warmup_info,
            sample_info=sample_info,
            chain_summaries=chain_summaries,
            init_strategy=init_strategy,
            init_jitter_scale=float(sample_config.get("init_jitter_scale", 0.25)),
            chain_initial_theta=chain_initial_theta,
            free_names=target.free_names,
        ),
        chain_ids=chain_ids,
    )


def sample_galaxy_batch_mclmc(
    context: DspsContext,
    observations: list[GalaxyObservation],
    base_params_rows: list[dict[str, float]],
    fit_config: dict[str, Any],
    sample_config: dict[str, Any],
    initial_params_rows: list[dict[str, float] | None] | None = None,
) -> list[MCMCResult]:
    """Sample multiple independent galaxy posteriors in one joint MCLMC state.

    The target factorizes over galaxies, but BlackJAX advances a single joint
    state with shape ``(n_galaxies, n_free_parameters)``. This keeps the output
    contract identical to one result per galaxy while letting each transition
    evaluate the DSPS forward model with JAX batching.
    """
    try:
        import blackjax
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "sample.sampler='mclmc' requires BlackJAX. Install the optional "
            "samplers extra or install blackjax in this environment."
        ) from exc

    n_galaxies = len(observations)
    if n_galaxies == 0:
        return []
    if len(base_params_rows) != n_galaxies:
        raise ValueError("base_params_rows must match observations length")
    if initial_params_rows is None:
        initial_params_rows = [None] * n_galaxies
    if len(initial_params_rows) != n_galaxies:
        raise ValueError("initial_params_rows must match observations length")
    if n_galaxies == 1:
        return [
            _sample_one_galaxy_mclmc(
                context,
                observations[0],
                base_params_rows[0],
                fit_config,
                sample_config,
                initial_params=initial_params_rows[0],
            )
        ]

    free = fit_config["free_parameters"]
    free_names = tuple(free)
    bounds = np.asarray(
        [tuple(float(value) for value in free[name]["bounds"]) for name in free_names],
        dtype=float,
    )
    lower = jnp.asarray(bounds[:, 0], dtype=jnp.float32)
    upper = jnp.asarray(bounds[:, 1], dtype=jnp.float32)
    transform = BoundedParameterTransform(
        names=free_names,
        lower=lower,
        upper=upper,
        gas_metallicity_constraint=_gas_metallicity_constraint_indices(free_names),
    )

    parameter_names = _mclmc_batch_parameter_names(base_params_rows)
    base_matrix = jnp.asarray(
        [
            [float(row.get(name, np.nan)) for name in parameter_names]
            for row in base_params_rows
        ],
        dtype=jnp.float32,
    )
    (
        observed_mag,
        sigma_mag,
        observed_flux,
        flux_error,
        band_names,
    ) = _mclmc_observation_batch_arrays(observations)
    observed, sigma, finite, likelihood_space = _mclmc_likelihood_batch_arrays(
        fit_config,
        observed_mag=observed_mag,
        sigma_mag=sigma_mag,
        observed_flux=observed_flux,
        flux_error=flux_error,
    )
    prior_arrays = _mclmc_prior_batch_arrays(
        free_names=free_names,
        free=free,
        sample_config=sample_config,
        base_params_rows=base_params_rows,
    )
    band_offsets = jnp.asarray(
        fit_config.get("band_calibration_offsets_mag", []), dtype=jnp.float32
    )
    model_args = dynamic_model_args(context)
    photometric_likelihood = _photometric_likelihood(fit_config)
    student_t_dof = _student_t_dof(fit_config)

    def params_from_base_theta(base, theta):
        params = {name: base[index] for index, name in enumerate(parameter_names)}
        params.update({name: theta[index] for index, name in enumerate(free_names)})
        return params

    def single_logdensity(
        y,
        base,
        observed_i,
        sigma_i,
        finite_i,
        prior_code_i,
        prior_low_i,
        prior_high_i,
        prior_loc_i,
        prior_scale_i,
        prior_alpha_i,
        prior_beta_i,
    ):
        theta = transform.to_bounded(y)
        params = params_from_base_theta(base, theta)
        model_mag = model_mags_jax_dynamic(context, model_args, params)
        if band_offsets.size:
            model_mag = model_mag + band_offsets
        model_obs = (
            abmag_to_fnu_cgs_jax(model_mag) if likelihood_space == "flux" else model_mag
        )
        loglike = _masked_observation_logprob(
            observed=observed_i,
            model_obs=model_obs,
            sigma=sigma_i,
            finite_mask=finite_i,
            photometric_likelihood=photometric_likelihood,
            student_t_dof=student_t_dof,
        )
        logprior = _mclmc_batched_bounded_log_prior(
            theta=theta,
            prior_code=prior_code_i,
            low=prior_low_i,
            high=prior_high_i,
            loc=prior_loc_i,
            scale=prior_scale_i,
            alpha=prior_alpha_i,
            beta=prior_beta_i,
        )
        logjac = transform.log_abs_det_jacobian(y)
        if transform.gas_metallicity_constraint is None:
            gas_penalty = gas_metallicity_constraint_penalty_jax(
                params, context.model_config, penalty=jnp.inf
            )
        else:
            gas_penalty = jnp.asarray(0.0, dtype=theta.dtype)
        return loglike + logprior + logjac - gas_penalty

    def joint_logdensity(y_flat):
        y_batch = jnp.reshape(y_flat, (n_galaxies, len(free_names)))
        logdensity = jax.vmap(
            single_logdensity,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )(
            y_batch,
            base_matrix,
            observed,
            sigma,
            finite,
            prior_arrays["code"],
            prior_arrays["low"],
            prior_arrays["high"],
            prior_arrays["loc"],
            prior_arrays["scale"],
            prior_arrays["alpha"],
            prior_arrays["beta"],
        )
        return jnp.sum(logdensity)

    num_warmup = int(sample_config.get("num_warmup", 100))
    num_samples = int(sample_config.get("num_samples", 200))
    num_chains = int(sample_config.get("num_chains", 1))
    if num_chains <= 0:
        raise ValueError("sample.num_chains must be positive")
    seed = int(sample_config.get("seed", 42))
    progress_bar = bool(sample_config.get("progress_bar", True))
    debug = bool(sample_config.get("mclmc_debug", False))
    progress_chunk_size = int(sample_config.get("mclmc_progress_chunk_size", 16))
    if progress_chunk_size <= 0:
        raise ValueError("sample.mclmc_progress_chunk_size must be positive")

    dim = n_galaxies * len(free_names)
    L = _resolve_mclmc_float(
        sample_config.get("mclmc_l", sample_config.get("L")), np.sqrt(float(dim))
    )
    step_size = _resolve_mclmc_float(
        sample_config.get("mclmc_step_size", sample_config.get("step_size")),
        min(0.10, 1.0 / np.sqrt(float(dim))),
    )
    inverse_mass_matrix = _mclmc_inverse_mass_matrix(sample_config, dim)
    algorithm = blackjax.mclmc(
        logdensity_fn=joint_logdensity,
        L=L,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        desired_energy_var_max_ratio=float(
            sample_config.get("mclmc_desired_energy_var_max_ratio", np.inf)
        ),
    )

    rng_key = random.PRNGKey(seed)
    chain_keys = random.split(rng_key, num_chains)
    init_position_keys = random.split(random.fold_in(rng_key, 87231), num_chains)
    has_initial = any(row for row in initial_params_rows)
    init_strategy = _mclmc_batch_init_strategy(sample_config, has_initial)
    chain_y0s_matrix = _mclmc_batch_initial_positions(
        transform=transform,
        free_names=free_names,
        base_params_rows=base_params_rows,
        initial_params_rows=initial_params_rows,
        fit_config=fit_config,
        sample_config=sample_config,
        keys=init_position_keys,
        init_strategy=init_strategy,
    )
    chain_y0s = jnp.reshape(chain_y0s_matrix, (num_chains, dim))
    chain_initial_theta = np.asarray(
        jax.vmap(jax.vmap(transform.to_bounded))(chain_y0s_matrix)
    )
    _mclmc_log(
        progress_bar or debug,
        "backend="
        f"{jax.default_backend()} devices={_jax_device_strings()} "
        f"batch={n_galaxies} dim={dim} bands={len(band_names)} "
        f"chains={num_chains} warmup={num_warmup} samples={num_samples} "
        f"L={L:.6g} step_size={step_size:.6g} "
        f"chunk={progress_chunk_size} init={init_strategy}",
    )
    _mclmc_log(progress_bar or debug, "compiling batched BlackJAX MCLMC step")
    step_fn = algorithm.step
    probe_steps = min(
        progress_chunk_size,
        max(num_warmup if num_warmup > 0 else 0, num_samples, 1),
    )
    first_init_key, first_compile_key = random.split(chain_keys[0])
    first_state = algorithm.init(chain_y0s[0], first_init_key)
    compile_start = time.perf_counter()
    compiled_state, _ = _scan_mclmc(
        step_fn, first_state, first_compile_key, probe_steps
    )
    jax.block_until_ready(compiled_state.position)
    compile_time = time.perf_counter() - compile_start
    _mclmc_log(progress_bar or debug, f"compile done in {compile_time:.3f}s")

    theta_chains: list[np.ndarray] = []
    warmup_infos = []
    sample_infos = []
    chain_summaries = []
    total_warmup_time = 0.0
    total_sampling_time = 0.0
    for chain_index, chain_key in enumerate(chain_keys):
        init_key, warmup_key, sample_key = random.split(chain_key, 3)
        state = algorithm.init(chain_y0s[chain_index], init_key)
        chain_label = f"chain {chain_index + 1}/{num_chains}"
        _mclmc_log(progress_bar or debug, f"{chain_label} warmup start")
        warmup_start = time.perf_counter()
        state, warmup_info = _run_mclmc_steps(
            step_fn,
            state,
            warmup_key,
            num_warmup,
            phase=f"mclmc-batch-c{chain_index + 1}-warmup",
            progress_bar=progress_bar,
            debug=debug,
            chunk_size=progress_chunk_size,
        )
        _raise_if_mclmc_all_invalid(warmup_info, phase=f"{chain_label} warmup")
        jax.block_until_ready(state.position)
        warmup_time = time.perf_counter() - warmup_start
        total_warmup_time += warmup_time
        _mclmc_log(
            progress_bar or debug,
            f"{chain_label} warmup done in {warmup_time:.3f}s",
        )

        _mclmc_log(progress_bar or debug, f"{chain_label} sampling start")
        sample_start = time.perf_counter()
        state, sample_info = _run_mclmc_steps(
            step_fn,
            state,
            sample_key,
            num_samples,
            phase=f"mclmc-batch-c{chain_index + 1}-sample",
            progress_bar=progress_bar,
            debug=debug,
            chunk_size=progress_chunk_size,
        )
        _raise_if_mclmc_all_invalid(sample_info, phase=f"{chain_label} sampling")
        positions = sample_info["position"]
        jax.block_until_ready(positions)
        sampling_time = time.perf_counter() - sample_start
        total_sampling_time += sampling_time
        _mclmc_log(
            progress_bar or debug,
            f"{chain_label} sampling done in {sampling_time:.3f}s",
        )
        positions_matrix = jnp.reshape(
            positions, (positions.shape[0], n_galaxies, len(free_names))
        )
        theta = jax.vmap(jax.vmap(transform.to_bounded))(positions_matrix)
        theta_chains.append(np.asarray(theta))
        warmup_infos.append(warmup_info)
        sample_infos.append(sample_info)
        chain_summaries.append(
            _mclmc_chain_summary(
                chain_index=chain_index,
                warmup_time=warmup_time,
                sampling_time=sampling_time,
                warmup_info=warmup_info,
                sample_info=sample_info,
            )
        )

    theta_np = np.concatenate(theta_chains, axis=0)
    chain_ids = np.repeat(np.arange(num_chains, dtype=np.int32), num_samples)
    warmup_info = _concat_mclmc_infos(warmup_infos)
    sample_info = _concat_mclmc_infos(sample_infos)
    diagnostics_template = _mclmc_diagnostics(
        sample_config=sample_config,
        initial_params=next((row for row in initial_params_rows if row), None),
        likelihood_space=likelihood_space,
        photometric_likelihood=photometric_likelihood,
        student_t_dof=student_t_dof,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        L=L,
        step_size=step_size,
        progress_chunk_size=progress_chunk_size,
        compile_time=compile_time,
        warmup_time=total_warmup_time,
        sampling_time=total_sampling_time,
        warmup_info=warmup_info,
        sample_info=sample_info,
        chain_summaries=chain_summaries,
        init_strategy=init_strategy,
        init_jitter_scale=float(sample_config.get("init_jitter_scale", 0.25)),
        chain_initial_theta=np.empty((0,)),
        free_names=list(free_names),
    )

    results: list[MCMCResult] = []
    for galaxy_index, observation in enumerate(observations):
        samples = {
            name: theta_np[:, galaxy_index, param_index]
            for param_index, name in enumerate(free_names)
        }
        posterior_model_mags = _posterior_model_mags(
            context,
            base_params_rows[galaxy_index],
            samples,
            fit_config,
            batch_size=int(sample_config.get("posterior_predictive_batch_size", 512)),
        )
        derived_samples = _posterior_derived(
            context, base_params_rows[galaxy_index], samples
        )
        initial_params = initial_params_rows[galaxy_index]
        diagnostics = {
            **diagnostics_template,
            "chain_method": "joint_batch_sequential_chains",
            "mclmc_batch_size": int(n_galaxies),
            "mclmc_batch_index": int(galaxy_index),
            "mclmc_joint_dimension": int(dim),
            "row_index": int(observation.row_index),
            "initialized_from_map": bool(initial_params),
            "aggregate_galaxy_samples_per_second": (
                float((n_galaxies * num_chains * num_samples) / total_sampling_time)
                if total_sampling_time > 0.0
                else float("inf")
            ),
            "chain_initial_parameters": [
                {
                    name: float(chain_initial_theta[chain, galaxy_index, param_index])
                    for param_index, name in enumerate(free_names)
                }
                for chain in range(chain_initial_theta.shape[0])
            ],
        }
        if initial_params:
            diagnostics["initial_parameters"] = {
                name: float(value) for name, value in initial_params.items()
            }
        else:
            diagnostics.pop("initial_parameters", None)
        results.append(
            MCMCResult(
                samples=samples,
                derived_samples=derived_samples,
                summary=_sample_summary(samples),
                posterior_model_mags=posterior_model_mags,
                observed_mag=np.asarray(observed_mag[galaxy_index]),
                sigma_mag=np.asarray(sigma_mag[galaxy_index]),
                observed_flux_fnu_cgs=np.asarray(observed_flux[galaxy_index]),
                flux_error_fnu_cgs=np.asarray(flux_error[galaxy_index]),
                band_names=band_names,
                diagnostics=diagnostics,
                chain_ids=chain_ids,
            )
        )
    return results


def _mclmc_batch_parameter_names(
    base_params_rows: list[dict[str, float]],
) -> list[str]:
    names = list(base_params_rows[0])
    for row in base_params_rows[1:]:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _mclmc_observation_batch_arrays(
    observations: list[GalaxyObservation],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    band_names = [band.name for band in observations[0].bands]
    for observation in observations[1:]:
        names = [band.name for band in observation.bands]
        if names != band_names:
            raise ValueError("All batched MCLMC observations must have the same bands")
    observed_mag = np.asarray(
        [[band.mag_ab for band in observation.bands] for observation in observations],
        dtype=float,
    )
    sigma_mag = np.asarray(
        [
            [band.sigma_mag for band in observation.bands]
            for observation in observations
        ],
        dtype=float,
    )
    observed_flux = np.asarray(
        [
            [band.flux_fnu_cgs for band in observation.bands]
            for observation in observations
        ],
        dtype=float,
    )
    flux_error = np.asarray(
        [
            [
                (
                    band.flux_error_fnu_cgs
                    if band.flux_error_fnu_cgs is not None
                    else magerr_to_fluxerr_fnu_cgs(band.flux_fnu_cgs, band.sigma_mag)
                )
                for band in observation.bands
            ]
            for observation in observations
        ],
        dtype=float,
    )
    return observed_mag, sigma_mag, observed_flux, flux_error, band_names


def _mclmc_likelihood_batch_arrays(
    fit_config: dict[str, Any],
    *,
    observed_mag: np.ndarray,
    sigma_mag: np.ndarray,
    observed_flux: np.ndarray,
    flux_error: np.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, str]:
    likelihood_space = str(fit_config.get("likelihood_space", "flux")).lower()
    if likelihood_space == "flux":
        floor_frac = float(fit_config.get("flux_error_floor_frac", 0.0))
        jitter = float(fit_config.get("flux_error_jitter", 0.0))
        observed = np.asarray(observed_flux, dtype=float)
        sigma = np.sqrt(
            np.asarray(flux_error, dtype=float) ** 2
            + (floor_frac * np.asarray(observed_flux, dtype=float)) ** 2
            + jitter**2
        )
    elif likelihood_space == "mag":
        observed = np.asarray(observed_mag, dtype=float)
        sigma = np.asarray(sigma_mag, dtype=float)
    else:
        raise ValueError(f"Unsupported fit.likelihood_space: {likelihood_space}")
    finite = np.isfinite(observed) & np.isfinite(sigma) & (sigma > 0.0)
    return (
        jnp.asarray(observed, dtype=jnp.float32),
        jnp.asarray(sigma, dtype=jnp.float32),
        jnp.asarray(finite),
        likelihood_space,
    )


def _mclmc_prior_batch_arrays(
    *,
    free_names: tuple[str, ...],
    free: dict[str, Any],
    sample_config: dict[str, Any],
    base_params_rows: list[dict[str, float]],
) -> dict[str, jnp.ndarray]:
    priors = sample_config.get("priors", {}) or {}
    type_codes = {
        "uniform": 0,
        "normal": 1,
        "truncated_normal": 2,
        "scaled_beta": 3,
    }
    code = np.zeros((len(base_params_rows), len(free_names)), dtype=np.int32)
    low = np.zeros_like(code, dtype=np.float32)
    high = np.zeros_like(low)
    loc = np.zeros_like(low)
    scale = np.ones_like(low)
    alpha = np.ones_like(low)
    beta = np.ones_like(low)
    for row_index, base_params in enumerate(base_params_rows):
        for param_index, name in enumerate(free_names):
            spec = _resolved_prior_spec(
                name,
                free[name],
                priors.get(name, {}),
                base_params,
            )
            prior_type = str(spec["type"])
            if prior_type not in type_codes:
                raise ValueError(f"Unsupported sample prior type: {prior_type}")
            code[row_index, param_index] = type_codes[prior_type]
            low[row_index, param_index] = float(spec["low"])
            high[row_index, param_index] = float(spec["high"])
            loc[row_index, param_index] = float(spec["loc"])
            scale[row_index, param_index] = float(spec["scale"])
            alpha[row_index, param_index] = float(spec["alpha"])
            beta[row_index, param_index] = float(spec["beta"])
    return {
        "code": jnp.asarray(code),
        "low": jnp.asarray(low),
        "high": jnp.asarray(high),
        "loc": jnp.asarray(loc),
        "scale": jnp.asarray(scale),
        "alpha": jnp.asarray(alpha),
        "beta": jnp.asarray(beta),
    }


def _mclmc_batched_bounded_log_prior(
    *,
    theta: jnp.ndarray,
    prior_code: jnp.ndarray,
    low: jnp.ndarray,
    high: jnp.ndarray,
    loc: jnp.ndarray,
    scale: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    span = high - low
    scale = jnp.maximum(scale, 1.0e-6)
    uniform = -jnp.log(span)
    normal = jstats.norm.logpdf(theta, loc, scale)
    norm = jnp.maximum(
        jsp.special.ndtr((high - loc) / scale) - jsp.special.ndtr((low - loc) / scale),
        1.0e-12,
    )
    truncated = normal - jnp.log(norm)
    unit = jnp.clip((theta - low) / span, 1.0e-6, 1.0 - 1.0e-6)
    scaled_beta = (
        (alpha - 1.0) * jnp.log(unit)
        + (beta - 1.0) * jnp.log1p(-unit)
        + (beta - 1.0) * jnp.log1p(-unit)
        - jsp.special.betaln(alpha, beta)
        - jnp.log(span)
    )
    logprob = jnp.where(
        prior_code == 0,
        uniform,
        jnp.where(
            prior_code == 1,
            normal,
            jnp.where(prior_code == 2, truncated, scaled_beta),
        ),
    )
    return jnp.sum(logprob)


def _mclmc_batch_init_strategy(sample_config: dict[str, Any], has_initial: bool) -> str:
    fallback = "map" if has_initial else "config"
    strategy = str(sample_config.get("init_strategy", fallback)).lower()
    if strategy == "map" and not has_initial:
        return "config"
    return strategy


def _mclmc_batch_initial_positions(
    *,
    transform: BoundedParameterTransform,
    free_names: tuple[str, ...],
    base_params_rows: list[dict[str, float]],
    initial_params_rows: list[dict[str, float] | None],
    fit_config: dict[str, Any],
    sample_config: dict[str, Any],
    keys: jnp.ndarray,
    init_strategy: str,
) -> jnp.ndarray:
    n_chains = int(keys.shape[0])
    n_galaxies = len(base_params_rows)
    if init_strategy == "random_uniform":
        return jax.vmap(
            lambda key: jax.vmap(
                lambda subkey: _mclmc_random_uniform_position_for_transform(
                    transform, subkey
                )
            )(random.split(key, n_galaxies))
        )(keys)

    theta0 = _mclmc_batch_initial_theta(
        transform=transform,
        free_names=free_names,
        base_params_rows=base_params_rows,
        initial_params_rows=initial_params_rows,
        fit_config=fit_config,
    )
    base_y0 = jax.vmap(transform.to_unconstrained)(theta0)
    if init_strategy in {"map", "config"}:
        return jnp.repeat(base_y0[None, :, :], n_chains, axis=0)
    if init_strategy == "map_jitter":
        scale = float(sample_config.get("init_jitter_scale", 0.25))
        noise = random.normal(
            keys[0],
            (n_chains, n_galaxies, len(free_names)),
            dtype=base_y0.dtype,
        )
        return base_y0[None, :, :] + scale * noise
    raise ValueError(
        "sample.init_strategy must be one of "
        "['map', 'config', 'map_jitter', 'random_uniform']"
    )


def _mclmc_batch_initial_theta(
    *,
    transform: BoundedParameterTransform,
    free_names: tuple[str, ...],
    base_params_rows: list[dict[str, float]],
    initial_params_rows: list[dict[str, float] | None],
    fit_config: dict[str, Any],
) -> jnp.ndarray:
    free = fit_config["free_parameters"]
    rows = []
    for base_params, initial_params in zip(
        base_params_rows, initial_params_rows, strict=True
    ):
        values = []
        for name in free_names:
            if (
                initial_params
                and name in initial_params
                and np.isfinite(initial_params[name])
            ):
                value = float(initial_params[name])
            else:
                value = _initial_value(free[name], name, base_params)
            values.append(value)
        rows.append(values)
    theta = jnp.asarray(rows, dtype=jnp.float32)
    eps = jnp.maximum((transform.upper - transform.lower) * 1.0e-6, 1.0e-7)
    return jnp.clip(theta, transform.lower + eps, transform.upper - eps)


def _mclmc_random_uniform_position_for_transform(
    transform: BoundedParameterTransform, key: jnp.ndarray
) -> jnp.ndarray:
    lower = transform.lower
    upper = transform.upper
    eps = jnp.asarray(1.0e-4, dtype=lower.dtype)
    unit = eps + (1.0 - 2.0 * eps) * random.uniform(key, lower.shape, dtype=lower.dtype)
    theta = lower + (upper - lower) * unit
    constraint = transform.gas_metallicity_constraint
    if constraint is None:
        return transform.to_unconstrained(theta)

    stellar_index, gas_index = constraint
    stellar_key, gas_key = random.split(random.fold_in(key, 4319))
    stellar_low = lower[stellar_index]
    stellar_high = jnp.minimum(upper[stellar_index], upper[gas_index])
    stellar_unit = eps + (1.0 - 2.0 * eps) * random.uniform(
        stellar_key, (), dtype=lower.dtype
    )
    stellar = stellar_low + (stellar_high - stellar_low) * stellar_unit
    gas_low = jnp.maximum(lower[gas_index], stellar)
    gas_span = jnp.maximum(upper[gas_index] - gas_low, 1.0e-6)
    gas_unit = eps + (1.0 - 2.0 * eps) * random.uniform(gas_key, (), dtype=lower.dtype)
    gas = gas_low + gas_span * gas_unit
    theta = theta.at[stellar_index].set(stellar).at[gas_index].set(gas)
    return transform.to_unconstrained(theta)


def _mclmc_batch_inverse_mass_matrix(
    sample_config: dict[str, Any], n_galaxies: int, n_free: int
):
    raw = sample_config.get("mclmc_inverse_mass_matrix")
    if raw is None:
        return 1.0
    if isinstance(raw, str):
        if raw.lower() == "identity":
            return 1.0
        if raw.lower() == "ones":
            return jnp.ones((n_galaxies * n_free,), dtype=jnp.float32)
    arr = jnp.asarray(raw, dtype=jnp.float32)
    if arr.shape == ():
        return float(arr)
    if arr.shape == (n_free,):
        return jnp.tile(arr, n_galaxies)
    if arr.shape != (n_galaxies * n_free,):
        raise ValueError(
            "sample.mclmc_inverse_mass_matrix must be scalar, length n_free, "
            f"or length batch*n_free ({n_galaxies * n_free}); got shape {arr.shape}"
        )
    return arr


def _mclmc_init_strategy(
    sample_config: dict[str, Any], initial_params: dict[str, float] | None
) -> str:
    fallback = "map" if initial_params else "config"
    strategy = str(sample_config.get("init_strategy", fallback)).lower()
    if strategy == "map" and not initial_params:
        return "config"
    return strategy


def _mclmc_initial_positions(
    target,
    initial_params: dict[str, float] | None,
    fit_config: dict[str, Any],
    sample_config: dict[str, Any],
    keys: jnp.ndarray,
    *,
    init_strategy: str,
) -> jnp.ndarray:
    n_chains = int(keys.shape[0])
    if init_strategy == "random_uniform":
        return jax.vmap(lambda key: _mclmc_random_uniform_position(target, key))(keys)
    base_y0 = initial_unconstrained_position(target, initial_params, fit_config)
    if init_strategy in {"map", "config"}:
        return jnp.repeat(base_y0[None, :], n_chains, axis=0)
    if init_strategy == "map_jitter":
        scale = float(sample_config.get("init_jitter_scale", 0.25))
        noise = random.normal(
            keys[0], (n_chains, base_y0.shape[0]), dtype=base_y0.dtype
        )
        return base_y0[None, :] + scale * noise
    raise ValueError(
        "sample.init_strategy must be one of "
        "['map', 'config', 'map_jitter', 'random_uniform']"
    )


def _mclmc_random_uniform_position(target, key: jnp.ndarray) -> jnp.ndarray:
    lower = target.transform.lower
    upper = target.transform.upper
    eps = jnp.asarray(1.0e-4, dtype=lower.dtype)
    unit = eps + (1.0 - 2.0 * eps) * random.uniform(key, lower.shape, dtype=lower.dtype)
    theta = lower + (upper - lower) * unit
    constraint = target.transform.gas_metallicity_constraint
    if constraint is None:
        return target.transform.to_unconstrained(theta)

    stellar_index, gas_index = constraint
    stellar_key, gas_key = random.split(random.fold_in(key, 4319))
    stellar_low = lower[stellar_index]
    stellar_high = jnp.minimum(upper[stellar_index], upper[gas_index])
    stellar_unit = eps + (1.0 - 2.0 * eps) * random.uniform(
        stellar_key, (), dtype=lower.dtype
    )
    stellar = stellar_low + (stellar_high - stellar_low) * stellar_unit
    gas_low = jnp.maximum(lower[gas_index], stellar)
    gas_span = jnp.maximum(upper[gas_index] - gas_low, 1.0e-6)
    gas_unit = eps + (1.0 - 2.0 * eps) * random.uniform(gas_key, (), dtype=lower.dtype)
    gas = gas_low + gas_span * gas_unit
    theta = theta.at[stellar_index].set(stellar).at[gas_index].set(gas)
    return target.transform.to_unconstrained(theta)


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
        desc=phase,
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
        name: jnp.concatenate(values, axis=0) for name, values in chunks.items()
    }


def _concat_mclmc_infos(infos: list[dict[str, jnp.ndarray]]) -> dict[str, jnp.ndarray]:
    if not infos:
        raise ValueError("No MCLMC info blocks to concatenate")
    return {
        name: jnp.concatenate([info[name] for info in infos], axis=0)
        for name in infos[0]
    }


def _raise_if_mclmc_all_invalid(info: dict[str, jnp.ndarray], *, phase: str) -> None:
    nonans = np.asarray(info["nonans"])
    if nonans.size and not bool(np.any(nonans)):
        raise RuntimeError(
            f"MCLMC produced zero valid transitions during {phase}. "
            "The chain is stuck at the initial state; reduce the step size, "
            "increase progress chunk size to reduce recompilation pressure, "
            "or rerun with sample.mclmc_debug=true to inspect logdensity."
        )


def _mclmc_chain_summary(
    *,
    chain_index: int,
    warmup_time: float,
    sampling_time: float,
    warmup_info: dict[str, jnp.ndarray],
    sample_info: dict[str, jnp.ndarray],
) -> dict[str, Any]:
    warmup_nonans = np.asarray(warmup_info["nonans"])
    sample_nonans = np.asarray(sample_info["nonans"])
    sample_logdensity = np.asarray(sample_info["logdensity"])
    summary: dict[str, Any] = {
        "chain": int(chain_index),
        "warmup_time_s": float(warmup_time),
        "sampling_time_s": float(sampling_time),
    }
    if warmup_nonans.size:
        summary["warmup_fraction_nonans"] = float(warmup_nonans.mean())
    if sample_nonans.size:
        summary["fraction_nonans"] = float(sample_nonans.mean())
    finite_logdensity = sample_logdensity[np.isfinite(sample_logdensity)]
    if finite_logdensity.size:
        summary["first_logdensity"] = float(finite_logdensity[0])
        summary["last_logdensity"] = float(finite_logdensity[-1])
        summary["mean_logdensity"] = float(finite_logdensity.mean())
    return summary


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
    batch_size: int | None = None,
) -> np.ndarray:
    parameter_names, matrix = _posterior_parameter_matrix(base_params, samples)
    if batch_size is None or batch_size <= 0 or matrix.shape[0] <= batch_size:
        mags = predict_batch_mags(context, parameter_names, matrix)
    else:
        chunks = [
            predict_batch_mags(
                context, parameter_names, matrix[start : start + batch_size]
            )
            for start in range(0, matrix.shape[0], batch_size)
        ]
        mags = np.concatenate(chunks, axis=0)
    offsets = np.asarray(
        fit_config.get("band_calibration_offsets_mag", []), dtype=float
    )
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
    num_chains: int,
    L: float,
    step_size: float,
    progress_chunk_size: int,
    compile_time: float,
    warmup_time: float,
    sampling_time: float,
    warmup_info: dict[str, jnp.ndarray],
    sample_info: dict[str, jnp.ndarray],
    chain_summaries: list[dict[str, Any]],
    init_strategy: str,
    init_jitter_scale: float,
    chain_initial_theta: np.ndarray,
    free_names: list[str],
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
        "num_warmup_per_chain": int(num_warmup),
        "num_samples": int(num_chains * num_samples),
        "num_samples_per_chain": int(num_samples),
        "num_chains": int(num_chains),
        "chain_method": "sequential",
        "mclmc_l": float(L),
        "mclmc_step_size": float(step_size),
        "mclmc_progress_chunk_size": int(progress_chunk_size),
        "init_strategy": init_strategy,
        "init_jitter_scale": float(init_jitter_scale),
        "compile_time_s": float(compile_time),
        "warmup_time_s": float(warmup_time),
        "sampling_time_s": float(sampling_time),
        "samples_per_second": (
            float((num_chains * num_samples) / sampling_time)
            if sampling_time > 0.0
            else float("inf")
        ),
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
    if chain_initial_theta.size:
        diagnostics["chain_initial_parameters"] = [
            {name: float(theta[index]) for index, name in enumerate(free_names)}
            for theta in np.asarray(chain_initial_theta)
        ]
    diagnostics["chains"] = chain_summaries
    return diagnostics
