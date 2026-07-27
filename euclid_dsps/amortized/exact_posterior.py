"""Exact-posterior sampling utilities for learned-prior FENIKS benchmarks."""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


class TargetValues(NamedTuple):
    """Decomposed learned-prior posterior density values."""

    loglike: jnp.ndarray
    logprior: jnp.ndarray
    logtarget: jnp.ndarray


@dataclass(frozen=True)
class NUTSSettings:
    warmup_steps: int = 500
    sample_chunks: tuple[int, ...] = (100, 500, 1000)
    target_accept: float = 0.65
    max_num_doublings: int = 10


@dataclass(frozen=True)
class MCLMCSettings:
    tune_steps: int = 500
    sample_chunks: tuple[int, ...] = (100, 500, 1000)
    thinning: int = 8
    target_accept: float = 0.8
    initial_step_size: float = 1.0e-4
    frac_tune1: float = 0.4
    frac_tune2: float = 0.4
    frac_tune3: float = 0.2
    diagonal_preconditioning: bool = True
    collapse_ratio: float = 1.0e-4
    desired_energy_var: float = 5.0e-4


def normalized_importance_weights(
    logtarget: np.ndarray,
    logq: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Return stable raw importance weights and deterministic PSIS smoothing."""
    log_weight = np.asarray(logtarget, dtype=np.float64) - np.asarray(
        logq, dtype=np.float64
    )
    finite = np.isfinite(log_weight)
    if not finite.any():
        raise ValueError("Importance sampling has no finite log weights")
    floor = float(np.min(log_weight[finite]) - 100.0)
    clean = np.where(finite, log_weight, floor)
    shifted = clean - float(np.max(clean))
    raw = np.exp(shifted)
    raw /= np.sum(raw)
    raw_ess = effective_sample_size_from_weights(raw)
    smoothed, pareto_k = _pareto_smooth_weights(clean)
    return {
        "log_weight": log_weight,
        "weight": raw,
        "raw_ess": float(raw_ess),
        "raw_ess_fraction": float(raw_ess / len(raw)),
        "psis_weight": smoothed,
        "psis_ess": float(effective_sample_size_from_weights(smoothed)),
        "pareto_k": float(pareto_k),
    }


def effective_sample_size_from_weights(weight: np.ndarray) -> float:
    """Return self-normalized importance-weight ESS."""
    weight = np.asarray(weight, dtype=np.float64)
    total = float(np.sum(weight))
    if not np.isfinite(total) or total <= 0.0:
        return 0.0
    normalized = weight / total
    return float(1.0 / np.sum(normalized**2))


def systematic_resample(
    weight: np.ndarray,
    n_samples: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return deterministic-seed systematic-resampling indices."""
    weight = np.asarray(weight, dtype=np.float64)
    weight = weight / np.sum(weight)
    rng = np.random.default_rng(int(seed))
    positions = (rng.random() + np.arange(int(n_samples))) / int(n_samples)
    cumulative = np.cumsum(weight)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _pareto_smooth_weights(log_weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Smooth the upper weight tail with a fitted generalized Pareto law.

    This implements the core PSIS tail replacement. It intentionally preserves
    the unsmoothed log weights as the scientific audit artifact.
    """
    from scipy.stats import genpareto

    log_weight = np.asarray(log_weight, dtype=np.float64)
    n = int(log_weight.size)
    if n < 20:
        weight = np.exp(log_weight - np.max(log_weight))
        return weight / np.sum(weight), float("nan")
    order = np.argsort(log_weight)
    sorted_log = log_weight[order]
    tail_n = min(max(int(math.ceil(0.2 * n)), 20), int(math.ceil(3.0 * np.sqrt(n))))
    tail_start = n - tail_n
    threshold = sorted_log[tail_start - 1]
    excess = np.exp(sorted_log[tail_start:] - threshold) - 1.0
    try:
        shape, _location, scale = genpareto.fit(excess, floc=0.0)
    except (ValueError, FloatingPointError):
        shape, scale = float("nan"), float("nan")
    smoothed_log = sorted_log.copy()
    if np.isfinite(shape) and np.isfinite(scale) and scale > 0.0:
        probability = (np.arange(tail_n, dtype=np.float64) + 0.5) / tail_n
        fitted = genpareto.ppf(probability, c=shape, loc=0.0, scale=scale)
        fitted = np.maximum(fitted, 0.0)
        fitted_log = threshold + np.log1p(fitted)
        fitted_log = np.minimum(fitted_log, float(np.max(sorted_log)))
        smoothed_log[tail_start:] = fitted_log
    restored = np.empty_like(smoothed_log)
    restored[order] = smoothed_log
    weight = np.exp(restored - np.max(restored))
    weight /= np.sum(weight)
    return weight, float(shape)


def run_nuts_chain(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_position: jnp.ndarray,
    *,
    seed: int,
    settings: NUTSSettings,
    out_dir: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Adapt and sample one resumable BlackJAX NUTS chain."""
    _enforce_float32_sampling()
    import blackjax

    sampling_logdensity = _float32_logdensity(logdensity_fn)
    _validate_chunks(settings.sample_chunks)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "sampling_state.pkl"
    params_path = out / "tuned_parameters.npz"
    manifest_path = out / "chain_manifest.json"
    key = jax.random.PRNGKey(int(seed))
    key, warmup_key = jax.random.split(key)
    started = time.perf_counter()
    if resume and state_path.exists() and params_path.exists():
        payload = _read_pickle(state_path)
        state = payload["state"]
        key = payload["key"]
        params_np = np.load(params_path)
        parameters = {
            "step_size": jnp.asarray(params_np["step_size"]),
            "inverse_mass_matrix": jnp.asarray(params_np["inverse_mass_matrix"]),
        }
        warmup_elapsed = float(payload.get("warmup_elapsed_s", 0.0))
    else:
        adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            sampling_logdensity,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=float(settings.target_accept),
            progress_bar=False,
            max_num_doublings=int(settings.max_num_doublings),
        )
        warmup_started = time.perf_counter()
        (state, parameters), adaptation_info = adaptation.run(
            warmup_key,
            jnp.asarray(initial_position, dtype=jnp.float32),
            int(settings.warmup_steps),
        )
        jax.block_until_ready(state.position)
        warmup_elapsed = time.perf_counter() - warmup_started
        np.savez(
            params_path,
            step_size=np.asarray(jax.device_get(parameters["step_size"])),
            inverse_mass_matrix=np.asarray(
                jax.device_get(parameters["inverse_mass_matrix"])
            ),
        )
        _write_warmup_summary(
            out / "warmup_summary.json",
            sampler="nuts",
            elapsed_s=warmup_elapsed,
            nominal_steps=int(settings.warmup_steps),
            parameters=parameters,
            adaptation_info=adaptation_info,
        )
        _write_pickle_atomic(
            state_path,
            {
                "state": state,
                "key": key,
                "warmup_elapsed_s": warmup_elapsed,
            },
        )
    algorithm = blackjax.nuts(
        sampling_logdensity,
        step_size=parameters["step_size"],
        inverse_mass_matrix=parameters["inverse_mass_matrix"],
        max_num_doublings=int(settings.max_num_doublings),
    )
    chunks = []
    for chunk_id, n_samples in enumerate(settings.sample_chunks):
        chunk_path = out / "chunks" / f"part_{chunk_id:06d}.parquet"
        info_path = out / "chunks" / f"part_{chunk_id:06d}_info.parquet"
        if resume and chunk_path.exists() and info_path.exists():
            chunks.append(_chunk_record(chunk_id, n_samples, chunk_path, info_path))
            continue
        key, chunk_key = jax.random.split(key)
        chunk_started = time.perf_counter()
        state, positions, infos = _run_algorithm_steps(
            algorithm, state, chunk_key, int(n_samples), thinning=1
        )
        jax.block_until_ready(state.position)
        elapsed = time.perf_counter() - chunk_started
        _write_chain_chunk(
            chunk_path,
            info_path,
            positions,
            infos,
            elapsed_s=elapsed,
            thinning=1,
        )
        _write_pickle_atomic(
            state_path,
            {
                "state": state,
                "key": key,
                "warmup_elapsed_s": warmup_elapsed,
            },
        )
        chunks.append(_chunk_record(chunk_id, n_samples, chunk_path, info_path))
    manifest = {
        "sampler": "nuts",
        "seed": int(seed),
        "warmup_steps": int(settings.warmup_steps),
        "target_accept": float(settings.target_accept),
        "max_num_doublings": int(settings.max_num_doublings),
        "sample_chunks": list(settings.sample_chunks),
        "stored_samples": int(sum(settings.sample_chunks)),
        "kernel_transitions": int(sum(settings.sample_chunks)),
        "warmup_elapsed_s": warmup_elapsed,
        "total_elapsed_s": time.perf_counter() - started,
        "chunks": chunks,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def run_adjusted_mclmc_chain(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_position: jnp.ndarray,
    *,
    seed: int,
    settings: MCLMCSettings,
    out_dir: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Adapt and sample one resumable Metropolis-adjusted MCLMC chain."""
    _enforce_float32_sampling()
    import blackjax
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    sampling_logdensity = _float32_logdensity(logdensity_fn)
    _validate_chunks(settings.sample_chunks)
    if int(settings.thinning) <= 0:
        raise ValueError("MCLMC thinning must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "sampling_state.pkl"
    params_path = out / "tuned_parameters.npz"
    key = jax.random.PRNGKey(int(seed))
    key, init_key, warmup_key = jax.random.split(key, 3)
    dim = int(np.asarray(initial_position).size)
    started = time.perf_counter()
    if resume and state_path.exists() and params_path.exists():
        payload = _read_pickle(state_path)
        state = payload["state"]
        key = payload["key"]
        saved = np.load(params_path)
        parameters = {
            "L": jnp.asarray(saved["L"]),
            "step_size": jnp.asarray(saved["step_size"]),
            "inverse_mass_matrix": jnp.asarray(saved["inverse_mass_matrix"]),
        }
        warmup_elapsed = float(payload.get("warmup_elapsed_s", 0.0))
        tuning_integrator_steps = int(payload.get("tuning_integrator_steps", 0))
    else:
        state = blackjax.mcmc.adjusted_mclmc.init(
            jnp.asarray(initial_position, dtype=jnp.float32), sampling_logdensity
        )
        initial_params = MCLMCAdaptationState(
            L=jnp.asarray(max(math.sqrt(dim), 1.0), dtype=jnp.float32),
            step_size=jnp.asarray(settings.initial_step_size, dtype=jnp.float32),
            inverse_mass_matrix=jnp.ones((dim,), dtype=jnp.float32),
        )

        def adaptation_kernel(
            rng_key,
            state,
            avg_num_integration_steps,
            step_size,
            inverse_mass_matrix,
        ):
            kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
                sampling_logdensity,
                inverse_mass_matrix=inverse_mass_matrix,
            )
            integration_steps = jnp.maximum(
                1, jnp.ceil(avg_num_integration_steps).astype(jnp.int32)
            )
            return kernel(
                rng_key,
                state,
                step_size,
                integration_steps,
            )

        warmup_started = time.perf_counter()
        state, tuned, tuning_integrator_steps = (
            blackjax.adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=adaptation_kernel,
                num_steps=int(settings.tune_steps),
                state=state,
                rng_key=warmup_key,
                target=float(settings.target_accept),
                frac_tune1=float(settings.frac_tune1),
                frac_tune2=float(settings.frac_tune2),
                frac_tune3=float(settings.frac_tune3),
                diagonal_preconditioning=bool(settings.diagonal_preconditioning),
                params=initial_params,
            )
        )
        jax.block_until_ready(state.position)
        warmup_elapsed = time.perf_counter() - warmup_started
        parameters = {
            "L": tuned.L,
            "step_size": tuned.step_size,
            "inverse_mass_matrix": tuned.inverse_mass_matrix,
        }
        _validate_mclmc_parameters(parameters, settings)
        np.savez(
            params_path,
            **{
                name: np.asarray(jax.device_get(value))
                for name, value in parameters.items()
            },
        )
        _write_warmup_summary(
            out / "warmup_summary.json",
            sampler="adjusted_mclmc",
            elapsed_s=warmup_elapsed,
            nominal_steps=int(settings.tune_steps),
            actual_integrator_steps=int(tuning_integrator_steps),
            parameters=parameters,
        )
        _write_pickle_atomic(
            state_path,
            {
                "state": state,
                "key": key,
                "warmup_elapsed_s": warmup_elapsed,
                "tuning_integrator_steps": int(tuning_integrator_steps),
            },
        )
    _validate_mclmc_parameters(parameters, settings)
    integration_steps = max(
        1,
        int(
            math.ceil(
                float(np.asarray(parameters["L"]))
                / float(np.asarray(parameters["step_size"]))
            )
        ),
    )
    algorithm = blackjax.adjusted_mclmc(
        sampling_logdensity,
        step_size=parameters["step_size"],
        inverse_mass_matrix=parameters["inverse_mass_matrix"],
        num_integration_steps=integration_steps,
    )
    chunks = []
    for chunk_id, n_samples in enumerate(settings.sample_chunks):
        chunk_path = out / "chunks" / f"part_{chunk_id:06d}.parquet"
        info_path = out / "chunks" / f"part_{chunk_id:06d}_info.parquet"
        if resume and chunk_path.exists() and info_path.exists():
            chunks.append(_chunk_record(chunk_id, n_samples, chunk_path, info_path))
            continue
        key, chunk_key = jax.random.split(key)
        chunk_started = time.perf_counter()
        state, positions, infos = _run_algorithm_steps(
            algorithm,
            state,
            chunk_key,
            int(n_samples),
            thinning=int(settings.thinning),
        )
        jax.block_until_ready(state.position)
        elapsed = time.perf_counter() - chunk_started
        _write_chain_chunk(
            chunk_path,
            info_path,
            positions,
            infos,
            elapsed_s=elapsed,
            thinning=int(settings.thinning),
        )
        _write_pickle_atomic(
            state_path,
            {
                "state": state,
                "key": key,
                "warmup_elapsed_s": warmup_elapsed,
                "tuning_integrator_steps": int(tuning_integrator_steps),
            },
        )
        chunks.append(_chunk_record(chunk_id, n_samples, chunk_path, info_path))
    stored = int(sum(settings.sample_chunks))
    manifest = {
        "sampler": "adjusted_mclmc",
        "seed": int(seed),
        "tune_steps": int(settings.tune_steps),
        "actual_tuning_integrator_steps": int(tuning_integrator_steps),
        "target_accept": float(settings.target_accept),
        "frac_tune": [
            float(settings.frac_tune1),
            float(settings.frac_tune2),
            float(settings.frac_tune3),
        ],
        "initial_step_size": float(settings.initial_step_size),
        "L": float(np.asarray(parameters["L"])),
        "step_size": float(np.asarray(parameters["step_size"])),
        "integration_steps_per_transition": int(integration_steps),
        "thinning": int(settings.thinning),
        "sample_chunks": list(settings.sample_chunks),
        "stored_samples": stored,
        "kernel_transitions": stored * int(settings.thinning),
        "integrator_steps_after_warmup": (
            stored * int(settings.thinning) * int(integration_steps)
        ),
        "warmup_elapsed_s": warmup_elapsed,
        "total_elapsed_s": time.perf_counter() - started,
        "chunks": chunks,
    }
    _write_json_atomic(out / "chain_manifest.json", manifest)
    return manifest


def run_unadjusted_mclmc_chain(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_position: jnp.ndarray,
    *,
    seed: int,
    settings: MCLMCSettings,
    out_dir: str | Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Run an explicitly labelled unadjusted MCLMC diagnostic chain."""
    _enforce_float32_sampling()
    import blackjax
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    sampling_logdensity = _float32_logdensity(logdensity_fn)
    _validate_chunks(settings.sample_chunks)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "sampling_state.pkl"
    params_path = out / "tuned_parameters.npz"
    key = jax.random.PRNGKey(int(seed))
    key, init_key, warmup_key = jax.random.split(key, 3)
    dim = int(np.asarray(initial_position).size)
    started = time.perf_counter()
    if resume and state_path.exists() and params_path.exists():
        payload = _read_pickle(state_path)
        state = payload["state"]
        key = payload["key"]
        saved = np.load(params_path)
        parameters = {
            "L": jnp.asarray(saved["L"]),
            "step_size": jnp.asarray(saved["step_size"]),
            "inverse_mass_matrix": jnp.asarray(saved["inverse_mass_matrix"]),
        }
        warmup_elapsed = float(payload.get("warmup_elapsed_s", 0.0))
        tuning_integrator_steps = int(payload.get("tuning_integrator_steps", 0))
    else:
        state = blackjax.mcmc.mclmc.init(
            jnp.asarray(initial_position, dtype=jnp.float32),
            sampling_logdensity,
            init_key,
        )
        initial_params = MCLMCAdaptationState(
            L=jnp.asarray(max(math.sqrt(dim), 1.0), dtype=jnp.float32),
            step_size=jnp.asarray(settings.initial_step_size, dtype=jnp.float32),
            inverse_mass_matrix=jnp.ones((dim,), dtype=jnp.float32),
        )

        def kernel_factory(inverse_mass_matrix):
            return blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=sampling_logdensity,
                inverse_mass_matrix=inverse_mass_matrix,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                desired_energy_var=float(settings.desired_energy_var),
            )

        warmup_started = time.perf_counter()
        state, tuned, tuning_integrator_steps = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel_factory,
            num_steps=int(settings.tune_steps),
            state=state,
            rng_key=warmup_key,
            frac_tune1=float(settings.frac_tune1),
            frac_tune2=float(settings.frac_tune2),
            frac_tune3=float(settings.frac_tune3),
            desired_energy_var=float(settings.desired_energy_var),
            diagonal_preconditioning=bool(settings.diagonal_preconditioning),
            params=initial_params,
        )
        jax.block_until_ready(state.position)
        warmup_elapsed = time.perf_counter() - warmup_started
        parameters = {
            "L": tuned.L,
            "step_size": tuned.step_size,
            "inverse_mass_matrix": tuned.inverse_mass_matrix,
        }
        _validate_mclmc_parameters(parameters, settings)
        np.savez(
            params_path,
            **{
                name: np.asarray(jax.device_get(value))
                for name, value in parameters.items()
            },
        )
        _write_warmup_summary(
            out / "warmup_summary.json",
            sampler="unadjusted_mclmc",
            elapsed_s=warmup_elapsed,
            nominal_steps=int(settings.tune_steps),
            actual_integrator_steps=int(tuning_integrator_steps),
            parameters=parameters,
        )
        _write_pickle_atomic(
            state_path,
            {
                "state": state,
                "key": key,
                "warmup_elapsed_s": warmup_elapsed,
                "tuning_integrator_steps": int(tuning_integrator_steps),
            },
        )
    _validate_mclmc_parameters(parameters, settings)
    algorithm = blackjax.mclmc(sampling_logdensity, **parameters)
    chunks = []
    for chunk_id, n_samples in enumerate(settings.sample_chunks):
        chunk_path = out / "chunks" / f"part_{chunk_id:06d}.parquet"
        info_path = out / "chunks" / f"part_{chunk_id:06d}_info.parquet"
        if resume and chunk_path.exists() and info_path.exists():
            chunks.append(_chunk_record(chunk_id, n_samples, chunk_path, info_path))
            continue
        key, chunk_key = jax.random.split(key)
        chunk_started = time.perf_counter()
        state, positions, infos = _run_algorithm_steps(
            algorithm,
            state,
            chunk_key,
            int(n_samples),
            thinning=int(settings.thinning),
        )
        jax.block_until_ready(state.position)
        elapsed = time.perf_counter() - chunk_started
        _write_chain_chunk(
            chunk_path,
            info_path,
            positions,
            infos,
            elapsed_s=elapsed,
            thinning=int(settings.thinning),
        )
        _write_pickle_atomic(
            state_path,
            {
                "state": state,
                "key": key,
                "warmup_elapsed_s": warmup_elapsed,
                "tuning_integrator_steps": int(tuning_integrator_steps),
            },
        )
        chunks.append(_chunk_record(chunk_id, n_samples, chunk_path, info_path))
    stored = int(sum(settings.sample_chunks))
    manifest = {
        "sampler": "unadjusted_mclmc",
        "scientific_role": "efficiency_and_energy_diagnostic_only",
        "seed": int(seed),
        "tune_steps": int(settings.tune_steps),
        "actual_tuning_integrator_steps": int(tuning_integrator_steps),
        "desired_energy_var": float(settings.desired_energy_var),
        "frac_tune": [
            float(settings.frac_tune1),
            float(settings.frac_tune2),
            float(settings.frac_tune3),
        ],
        "L": float(np.asarray(parameters["L"])),
        "step_size": float(np.asarray(parameters["step_size"])),
        "thinning": int(settings.thinning),
        "stored_samples": stored,
        "kernel_transitions": stored * int(settings.thinning),
        "warmup_elapsed_s": warmup_elapsed,
        "total_elapsed_s": time.perf_counter() - started,
        "chunks": chunks,
    }
    _write_json_atomic(out / "chain_manifest.json", manifest)
    return manifest


def combine_chain_diagnostics(
    chain_directories: list[str | Path],
    *,
    parameter_names: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Combine chunks and report rank-normalized split-Rhat and ESS."""
    chains = []
    for directory in chain_directories:
        paths = sorted(
            path
            for path in (Path(directory) / "chunks").glob("part_*.parquet")
            if not path.name.endswith("_info.parquet")
        )
        if not paths:
            raise ValueError(f"No chain chunks found in {directory}")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        x_columns = [f"x_{index:02d}" for index in range(len(parameter_names))]
        chains.append(frame[x_columns].to_numpy(dtype=np.float64))
    min_draws = min(chain.shape[0] for chain in chains)
    values = np.stack([chain[:min_draws] for chain in chains], axis=0)
    rows = []
    for index, name in enumerate(parameter_names):
        scalar = values[:, :, index]
        rows.append(
            {
                "parameter": name,
                "rhat": split_rhat(scalar),
                "bulk_ess": autocorrelation_ess(_rank_normalize(scalar)),
                "tail_ess": tail_indicator_ess(scalar),
                "chains": int(values.shape[0]),
                "draws_per_chain": int(values.shape[1]),
            }
        )
    diagnostics = pd.DataFrame(rows)
    summary = {
        "chains": int(values.shape[0]),
        "draws_per_chain": int(values.shape[1]),
        "max_rhat": float(diagnostics["rhat"].max()),
        "min_bulk_ess": float(diagnostics["bulk_ess"].min()),
        "min_tail_ess": float(diagnostics["tail_ess"].min()),
        "passes_rhat_1_01": bool((diagnostics["rhat"] <= 1.01).all()),
        "passes_bulk_ess_400": bool((diagnostics["bulk_ess"] >= 400.0).all()),
        "passes_tail_ess_400": bool((diagnostics["tail_ess"] >= 400.0).all()),
    }
    return diagnostics, summary


def split_rhat(chains: np.ndarray) -> float:
    """Return rank-normalized folded split-Rhat for ``[chain, draw]``."""
    values = np.asarray(chains, dtype=np.float64)
    rank_rhat = _basic_split_rhat(_rank_normalize(values))
    folded = np.abs(values - np.median(values))
    folded_rhat = _basic_split_rhat(_rank_normalize(folded))
    return float(max(rank_rhat, folded_rhat))


def _basic_split_rhat(chains: np.ndarray) -> float:
    values = np.asarray(chains, dtype=np.float64)
    n = values.shape[1] // 2
    if values.shape[0] < 2 or n < 2:
        return float("nan")
    split = np.concatenate([values[:, :n], values[:, -n:]], axis=0)
    chain_mean = np.mean(split, axis=1)
    within = np.mean(np.var(split, axis=1, ddof=1))
    between = n * np.var(chain_mean, ddof=1)
    variance = (n - 1.0) / n * within + between / n
    if within <= 0.0:
        return 1.0 if variance <= 0.0 else float("inf")
    return float(np.sqrt(variance / within))


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri
    from scipy.stats import rankdata

    values = np.asarray(chains, dtype=np.float64)
    ranks = rankdata(values.reshape(-1), method="average").reshape(values.shape)
    probability = (ranks - 3.0 / 8.0) / (values.size + 1.0 / 4.0)
    return ndtri(probability)


def autocorrelation_ess(chains: np.ndarray) -> float:
    """Estimate ESS using Geyer's positive paired autocorrelation sequence."""
    values = np.asarray(chains, dtype=np.float64)
    m, n = values.shape
    if m < 1 or n < 4:
        return float("nan")
    centered = values - np.mean(values, axis=1, keepdims=True)
    variance = np.mean(np.var(values, axis=1, ddof=1))
    if variance <= 0.0:
        return float(m * n)
    rho = []
    for lag in range(1, n):
        covariance = np.mean(
            np.sum(centered[:, :-lag] * centered[:, lag:], axis=1) / (n - lag)
        )
        rho.append(float(covariance / variance))
    paired_sum = 0.0
    for index in range(0, len(rho) - 1, 2):
        pair = rho[index] + rho[index + 1]
        if pair < 0.0:
            break
        paired_sum += pair
    return float(min(m * n, m * n / max(1.0 + 2.0 * paired_sum, 1.0e-12)))


def tail_indicator_ess(chains: np.ndarray) -> float:
    """Return the minimum ESS of lower/upper 5% tail indicators."""
    values = np.asarray(chains, dtype=np.float64)
    flat = values.reshape(-1)
    lower, upper = np.quantile(flat, [0.05, 0.95])
    return min(
        autocorrelation_ess((values <= lower).astype(np.float64)),
        autocorrelation_ess((values >= upper).astype(np.float64)),
    )


def _run_algorithm_steps(algorithm, state, key, n_samples: int, *, thinning: int):
    keys = jax.random.split(key, int(n_samples))

    def stored_step(current, draw_key):
        transition_keys = jax.random.split(draw_key, int(thinning))

        def transition(one_state, one_key):
            return algorithm.step(one_key, one_state)

        final_state, infos = jax.lax.scan(transition, current, transition_keys)
        final_info = jax.tree.map(lambda value: value[-1], infos)
        return final_state, (final_state.position, final_info)

    return_value = jax.jit(lambda s, k: jax.lax.scan(stored_step, s, k))(state, keys)
    final_state, (positions, infos) = return_value
    return final_state, positions, infos


def _write_chain_chunk(
    samples_path: Path,
    info_path: Path,
    positions,
    infos,
    *,
    elapsed_s: float,
    thinning: int,
) -> None:
    position = np.asarray(jax.device_get(positions))
    samples = pd.DataFrame(
        {f"x_{index:02d}": position[:, index] for index in range(position.shape[1])}
    )
    samples.insert(0, "draw", np.arange(len(samples), dtype=np.int64))
    samples["elapsed_s"] = float(elapsed_s)
    samples["thinning"] = int(thinning)
    info_values = _info_columns(infos, len(samples))
    info = pd.DataFrame(info_values)
    info.insert(0, "draw", np.arange(len(info), dtype=np.int64))
    _write_parquet_atomic(samples, samples_path)
    _write_parquet_atomic(info, info_path)


def _info_columns(infos, n_rows: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in (
        "acceptance_rate",
        "is_divergent",
        "num_integration_steps",
        "energy",
        "proposal_energy",
    ):
        if not hasattr(infos, name):
            continue
        value = np.asarray(jax.device_get(getattr(infos, name)))
        if value.ndim == 1 and len(value) == n_rows:
            result[name] = value
    if not result:
        result["valid"] = np.ones(n_rows, dtype=bool)
    return result


def _float32_logdensity(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Keep BlackJAX states and target gradients on one explicit dtype."""

    def wrapped(position):
        value = logdensity_fn(jnp.asarray(position, dtype=jnp.float32))
        return jnp.asarray(value, dtype=jnp.float32)

    return wrapped


def _enforce_float32_sampling() -> None:
    """Disable JAX x64 before BlackJAX constructs any adaptation state."""
    if jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", False)
    if jax.config.x64_enabled:
        raise RuntimeError("BlackJAX benchmark requires jax_enable_x64=False")


def _validate_mclmc_parameters(
    parameters: dict[str, jnp.ndarray],
    settings: MCLMCSettings,
) -> None:
    step_size = float(np.asarray(parameters["step_size"]))
    length = float(np.asarray(parameters["L"]))
    inverse_mass = np.asarray(parameters["inverse_mass_matrix"])
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise FloatingPointError(f"Invalid adapted MCLMC step size: {step_size}")
    if step_size < float(settings.initial_step_size) * float(settings.collapse_ratio):
        raise FloatingPointError(
            "MCLMC step size collapsed: "
            f"initial={settings.initial_step_size} adapted={step_size}"
        )
    if not np.isfinite(length) or length <= 0.0:
        raise FloatingPointError(f"Invalid adapted MCLMC L: {length}")
    if not np.isfinite(inverse_mass).all() or np.any(inverse_mass <= 0.0):
        raise FloatingPointError("Invalid adapted MCLMC inverse mass matrix")


def _write_warmup_summary(
    path: Path,
    *,
    sampler: str,
    elapsed_s: float,
    nominal_steps: int,
    parameters: dict[str, Any],
    actual_integrator_steps: int | None = None,
    adaptation_info: Any | None = None,
) -> None:
    payload = {
        "sampler": sampler,
        "elapsed_s": float(elapsed_s),
        "nominal_steps": int(nominal_steps),
        "actual_integrator_steps": (
            None if actual_integrator_steps is None else int(actual_integrator_steps)
        ),
        "parameters": {
            name: np.asarray(jax.device_get(value)).tolist()
            for name, value in parameters.items()
        },
    }
    if adaptation_info is not None:
        payload["adaptation_diagnostics"] = _summarize_adaptation_info(adaptation_info)
    _write_json_atomic(path, payload)


def _summarize_adaptation_info(info: Any) -> dict[str, Any]:
    result = {}
    for name in ("acceptance_rate", "is_divergent", "num_integration_steps"):
        values = _find_tree_field(info, name)
        if values is None:
            continue
        array = np.asarray(jax.device_get(values))
        if name == "is_divergent":
            result["divergences"] = int(np.sum(array))
        else:
            result[f"{name}_mean"] = float(np.mean(array))
    return result


def _find_tree_field(tree: Any, name: str):
    if hasattr(tree, name):
        return getattr(tree, name)
    if isinstance(tree, (tuple, list)):
        for value in tree:
            found = _find_tree_field(value, name)
            if found is not None:
                return found
    return None


def _validate_chunks(chunks: tuple[int, ...]) -> None:
    if not chunks or any(int(value) <= 0 for value in chunks):
        raise ValueError("sample_chunks must contain positive integers")


def _chunk_record(
    chunk_id: int,
    n_samples: int,
    samples_path: Path,
    info_path: Path,
) -> dict[str, Any]:
    return {
        "chunk_id": int(chunk_id),
        "stored_samples": int(n_samples),
        "samples": str(samples_path),
        "info": str(info_path),
    }


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_pickle_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(jax.device_get(payload), stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return jax.tree.map(jnp.asarray, payload)
