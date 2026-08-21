"""Exact-posterior sampling utilities for learned-prior FENIKS benchmarks."""

from __future__ import annotations

import inspect
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
class BatchedTargetNUTSResult:
    """In-memory result from a multi-target, multi-chain NUTS probe."""

    positions: jnp.ndarray
    infos: Any
    step_size: jnp.ndarray
    inverse_mass_matrix: jnp.ndarray
    target_validation_elapsed_s: float
    warmup_elapsed_s: float
    sampling_elapsed_s: float


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
        "raw_max_weight": float(np.max(raw)),
        "raw_weight_entropy": float(-np.sum(raw[raw > 0.0] * np.log(raw[raw > 0.0]))),
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
    _enforce_float64_sampling()
    import blackjax

    sampling_logdensity = jax.jit(_float64_logdensity(logdensity_fn), inline=False)
    target_started = time.perf_counter()
    print("[exact-sampler:nuts] validating target value and gradient", flush=True)
    initial_position = _validate_sampling_target(sampling_logdensity, initial_position)
    print(
        "[exact-sampler:nuts] target ready "
        f"elapsed_s={time.perf_counter() - target_started:.1f}",
        flush=True,
    )
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
        state = _float64_sampling_state(payload["state"])
        key = payload["key"]
        params_np = np.load(params_path)
        parameters = {
            "step_size": jnp.asarray(params_np["step_size"], dtype=jnp.float64),
            "inverse_mass_matrix": jnp.asarray(
                params_np["inverse_mass_matrix"], dtype=jnp.float64
            ),
        }
        warmup_elapsed = float(payload.get("warmup_elapsed_s", 0.0))
    else:
        adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            sampling_logdensity,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=float(settings.target_accept),
            max_num_doublings=int(settings.max_num_doublings),
            initial_step_size=jnp.asarray(1.0, dtype=jnp.float64),
        )
        warmup_started = time.perf_counter()
        print(
            f"[exact-sampler:nuts] warmup start steps={settings.warmup_steps}",
            flush=True,
        )
        (state, parameters), adaptation_info = adaptation.run(
            warmup_key,
            initial_position,
            int(settings.warmup_steps),
        )
        jax.block_until_ready(state.position)
        warmup_elapsed = time.perf_counter() - warmup_started
        print(
            f"[exact-sampler:nuts] warmup done elapsed_s={warmup_elapsed:.1f}",
            flush=True,
        )
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
        print(
            f"[exact-sampler:nuts] chunk={chunk_id} start draws={n_samples}",
            flush=True,
        )
        state, positions, infos = _run_algorithm_steps(
            algorithm, state, chunk_key, int(n_samples), thinning=1
        )
        jax.block_until_ready(state.position)
        elapsed = time.perf_counter() - chunk_started
        print(
            f"[exact-sampler:nuts] chunk={chunk_id} done elapsed_s={elapsed:.1f}",
            flush=True,
        )
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
        "sampling_dtype": "float64",
        "target_dtype": "float32",
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


def run_batched_nuts_chains(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_positions: jnp.ndarray,
    *,
    seeds: tuple[int, ...],
    settings: NUTSSettings,
    out_dirs: tuple[str | Path, ...],
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Adapt and sample resumable independent NUTS chains with ``vmap``."""
    _enforce_float64_sampling()
    import blackjax

    positions = jnp.asarray(initial_positions, dtype=jnp.float64)
    n_chains = int(positions.shape[0])
    if positions.ndim != 2 or n_chains < 1:
        raise ValueError("initial_positions must have shape [chain, parameter]")
    if len(seeds) != n_chains or len(out_dirs) != n_chains:
        raise ValueError("seeds and out_dirs must match the number of chains")
    _validate_chunks(settings.sample_chunks)
    outputs = tuple(Path(path) for path in out_dirs)
    for output in outputs:
        output.mkdir(parents=True, exist_ok=True)
    parents = {output.parent for output in outputs}
    if len(parents) != 1:
        raise ValueError("Batched NUTS chain directories must share one parent")
    parent = next(iter(parents))
    group_name = "-".join(output.name for output in outputs)
    resume_path = parent / f".{group_name}.batched_nuts_state.pkl"
    contract_path = parent / f".{group_name}.batched_nuts_contract.json"
    contract = {
        "version": 1,
        "sampler": "nuts",
        "execution": "vmap_batched_chains",
        "chain_directories": [output.name for output in outputs],
        "seeds": [int(seed) for seed in seeds],
        "warmup_steps": int(settings.warmup_steps),
        "target_accept": float(settings.target_accept),
        "max_num_doublings": int(settings.max_num_doublings),
        "sample_chunks": list(settings.sample_chunks),
    }
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != contract:
            raise ValueError(
                "Incompatible batched NUTS resume contract: "
                f"actual={existing_contract} expected={contract}"
            )
    else:
        _write_json_atomic(contract_path, contract)

    sampling_logdensity = jax.jit(_float64_logdensity(logdensity_fn), inline=False)
    target_started = time.perf_counter()
    print(
        f"[exact-sampler:nuts-batched] validating {n_chains} targets and gradients",
        flush=True,
    )
    positions = jnp.stack(
        [
            _validate_sampling_target(sampling_logdensity, position)
            for position in positions
        ]
    )
    print(
        "[exact-sampler:nuts-batched] targets ready "
        f"elapsed_s={time.perf_counter() - target_started:.1f}",
        flush=True,
    )

    resumed = False
    completed_chunks = 0
    sampling_elapsed = 0.0
    adaptation_info = None
    if resume and resume_path.exists():
        payload = _read_pickle(resume_path)
        states = _float64_sampling_state(payload["states"])
        keys = jnp.asarray(payload["keys"])
        parameters = {
            "step_size": jnp.asarray(
                payload["parameters"]["step_size"], dtype=jnp.float64
            ),
            "inverse_mass_matrix": jnp.asarray(
                payload["parameters"]["inverse_mass_matrix"],
                dtype=jnp.float64,
            ),
        }
        warmup_elapsed = float(payload["warmup_elapsed_s"])
        sampling_elapsed = float(payload.get("sampling_elapsed_s", 0.0))
        completed_chunks = int(payload.get("completed_chunks", 0))
        resumed = True
    else:
        state_paths = [output / "sampling_state.pkl" for output in outputs]
        parameter_paths = [output / "tuned_parameters.npz" for output in outputs]
        resumable = [
            state_path.exists() and parameter_path.exists()
            for state_path, parameter_path in zip(
                state_paths, parameter_paths, strict=True
            )
        ]
        if resume and all(resumable):
            payloads = [_read_pickle(path) for path in state_paths]
            states = jax.tree.map(
                lambda *values: jnp.stack(values),
                *[_float64_sampling_state(payload["state"]) for payload in payloads],
            )
            keys = jnp.stack([jnp.asarray(payload["key"]) for payload in payloads])
            loaded_parameters = [np.load(path) for path in parameter_paths]
            parameters = {
                "step_size": jnp.stack(
                    [
                        jnp.asarray(values["step_size"], dtype=jnp.float64)
                        for values in loaded_parameters
                    ]
                ),
                "inverse_mass_matrix": jnp.stack(
                    [
                        jnp.asarray(values["inverse_mass_matrix"], dtype=jnp.float64)
                        for values in loaded_parameters
                    ]
                ),
            }
            warmup_elapsed = max(
                float(payload.get("warmup_elapsed_s", 0.0)) for payload in payloads
            )
            completed_chunks, sampling_elapsed = _completed_batched_nuts_chunks(
                outputs,
                settings.sample_chunks,
            )
            resumed = True
            _write_pickle_atomic(
                resume_path,
                {
                    "states": states,
                    "keys": keys,
                    "parameters": parameters,
                    "warmup_elapsed_s": warmup_elapsed,
                    "sampling_elapsed_s": sampling_elapsed,
                    "completed_chunks": completed_chunks,
                },
            )
        elif any(resumable) or any(
            output.exists() and any(output.iterdir()) for output in outputs
        ):
            raise FileExistsError(
                "Batched NUTS found incomplete, non-resumable chain artifacts: "
                + ", ".join(map(str, outputs))
            )
        else:
            base_keys = jnp.stack([jax.random.PRNGKey(int(seed)) for seed in seeds])
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(base_keys)
            keys = split_keys[:, 0]
            warmup_keys = split_keys[:, 1]
            adaptation = blackjax.window_adaptation(
                blackjax.nuts,
                sampling_logdensity,
                is_mass_matrix_diagonal=True,
                target_acceptance_rate=float(settings.target_accept),
                max_num_doublings=int(settings.max_num_doublings),
                initial_step_size=jnp.asarray(1.0, dtype=jnp.float64),
            )

            warmup_started = time.perf_counter()
            print(
                "[exact-sampler:nuts-batched] warmup start "
                f"chains={n_chains} steps={settings.warmup_steps} "
                f"max_doublings={settings.max_num_doublings}",
                flush=True,
            )

            def adapt_one(key, position):
                return adaptation.run(key, position, int(settings.warmup_steps))

            (states, parameters), adaptation_info = jax.vmap(adapt_one)(
                warmup_keys,
                positions,
            )
            jax.block_until_ready(states.position)
            warmup_elapsed = time.perf_counter() - warmup_started
            print(
                "[exact-sampler:nuts-batched] warmup done "
                f"chains={n_chains} elapsed_s={warmup_elapsed:.1f}",
                flush=True,
            )
            _write_pickle_atomic(
                resume_path,
                {
                    "states": states,
                    "keys": keys,
                    "parameters": parameters,
                    "warmup_elapsed_s": warmup_elapsed,
                    "sampling_elapsed_s": 0.0,
                    "completed_chunks": 0,
                },
            )

    if completed_chunks < 0 or completed_chunks > len(settings.sample_chunks):
        raise ValueError(f"Invalid completed batched chunk count: {completed_chunks}")
    if resumed:
        print(
            "[exact-sampler:nuts-batched] resume "
            f"chains={n_chains} completed_chunks={completed_chunks} "
            f"stored_draws={sum(settings.sample_chunks[:completed_chunks])}",
            flush=True,
        )
    _validate_or_rollback_batched_nuts_chunks(
        outputs,
        settings.sample_chunks,
        completed_chunks=completed_chunks,
    )

    for chain_index, output in enumerate(outputs):
        chain_parameters = jax.tree.map(
            lambda value, index=chain_index: value[index],
            parameters,
        )
        np.savez(
            output / "tuned_parameters.npz",
            step_size=np.asarray(jax.device_get(chain_parameters["step_size"])),
            inverse_mass_matrix=np.asarray(
                jax.device_get(chain_parameters["inverse_mass_matrix"])
            ),
        )
        warmup_path = output / "warmup_summary.json"
        if adaptation_info is not None:
            chain_info = jax.tree.map(
                lambda value, index=chain_index: value[index],
                adaptation_info,
            )
            _write_warmup_summary(
                warmup_path,
                sampler="nuts_batched",
                elapsed_s=warmup_elapsed,
                nominal_steps=int(settings.warmup_steps),
                parameters=chain_parameters,
                adaptation_info=chain_info,
            )
        elif not warmup_path.exists():
            _write_warmup_summary(
                warmup_path,
                sampler="nuts_batched",
                elapsed_s=warmup_elapsed,
                nominal_steps=int(settings.warmup_steps),
                parameters=chain_parameters,
            )
        _write_pickle_atomic(
            output / "sampling_state.pkl",
            {
                "state": jax.tree.map(
                    lambda value, index=chain_index: value[index],
                    states,
                ),
                "key": keys[chain_index],
                "warmup_elapsed_s": warmup_elapsed,
            },
        )

    chunks_by_chain = [
        [
            _chunk_record(
                chunk_id,
                settings.sample_chunks[chunk_id],
                output / "chunks" / f"part_{chunk_id:06d}.parquet",
                output / "chunks" / f"part_{chunk_id:06d}_info.parquet",
            )
            for chunk_id in range(completed_chunks)
        ]
        for output in outputs
    ]
    for chunk_id, n_samples in enumerate(settings.sample_chunks):
        if chunk_id < completed_chunks:
            continue
        key_pairs = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
        keys = key_pairs[:, 0]
        chunk_keys = key_pairs[:, 1]
        chunk_started = time.perf_counter()
        print(
            "[exact-sampler:nuts-batched] "
            f"chunk={chunk_id} start chains={n_chains} draws={n_samples}",
            flush=True,
        )
        states, (chunk_positions, infos) = _run_batched_nuts_steps(
            sampling_logdensity,
            states,
            chunk_keys,
            parameters["step_size"],
            parameters["inverse_mass_matrix"],
            n_samples=int(n_samples),
            max_num_doublings=int(settings.max_num_doublings),
        )
        jax.block_until_ready(states.position)
        elapsed = time.perf_counter() - chunk_started
        print(
            "[exact-sampler:nuts-batched] "
            f"chunk={chunk_id} done chains={n_chains} elapsed_s={elapsed:.1f}",
            flush=True,
        )
        for chain_index, output in enumerate(outputs):
            chunk_path = output / "chunks" / f"part_{chunk_id:06d}.parquet"
            info_path = output / "chunks" / f"part_{chunk_id:06d}_info.parquet"
            _write_chain_chunk(
                chunk_path,
                info_path,
                chunk_positions[:, chain_index],
                jax.tree.map(
                    lambda value, index=chain_index: value[:, index],
                    infos,
                ),
                elapsed_s=elapsed,
                thinning=1,
            )
        sampling_elapsed += elapsed
        _write_pickle_atomic(
            resume_path,
            {
                "states": states,
                "keys": keys,
                "parameters": parameters,
                "warmup_elapsed_s": warmup_elapsed,
                "sampling_elapsed_s": sampling_elapsed,
                "completed_chunks": chunk_id + 1,
            },
        )
        for chain_index, output in enumerate(outputs):
            _write_pickle_atomic(
                output / "sampling_state.pkl",
                {
                    "state": jax.tree.map(
                        lambda value, index=chain_index: value[index],
                        states,
                    ),
                    "key": keys[chain_index],
                    "warmup_elapsed_s": warmup_elapsed,
                },
            )
            chunks_by_chain[chain_index].append(
                _chunk_record(
                    chunk_id,
                    n_samples,
                    chunk_path,
                    info_path,
                )
            )

    total_elapsed = warmup_elapsed + sampling_elapsed
    manifests = []
    for chain_index, output in enumerate(outputs):
        manifest = {
            "sampler": "nuts",
            "execution": "vmap_batched_chains",
            "batched_chain_count": n_chains,
            "sampling_dtype": "float64",
            "target_dtype": "float32",
            "seed": int(seeds[chain_index]),
            "warmup_steps": int(settings.warmup_steps),
            "target_accept": float(settings.target_accept),
            "max_num_doublings": int(settings.max_num_doublings),
            "sample_chunks": list(settings.sample_chunks),
            "stored_samples": int(sum(settings.sample_chunks)),
            "kernel_transitions": int(sum(settings.sample_chunks)),
            "warmup_elapsed_s": warmup_elapsed,
            "total_elapsed_s": total_elapsed,
            "resumed": resumed,
            "resumed_from_chunks": completed_chunks,
            "chunks": chunks_by_chain[chain_index],
        }
        _write_json_atomic(output / "chain_manifest.json", manifest)
        manifests.append(manifest)
    return manifests


def run_batched_nuts_targets(
    logdensity_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
    initial_positions: jnp.ndarray,
    target_data: Any,
    *,
    seeds: jnp.ndarray,
    settings: NUTSSettings,
) -> BatchedTargetNUTSResult:
    """Run a short NUTS probe over distinct targets and chains with ``vmap``.

    ``initial_positions`` has shape ``[target, chain, parameter]``. Every leaf
    in ``target_data`` has a matching leading target axis. This intentionally
    keeps all results in memory: production runs should continue to use the
    resumable chain writers above.
    """
    _enforce_float64_sampling()
    import blackjax

    positions = jnp.asarray(initial_positions, dtype=jnp.float64)
    if positions.ndim != 3:
        raise ValueError("initial_positions must have shape [target, chain, parameter]")
    n_targets, n_chains, n_parameters = map(int, positions.shape)
    if n_targets < 1 or n_chains < 1 or n_parameters < 1:
        raise ValueError("initial_positions dimensions must all be positive")
    if len(settings.sample_chunks) != 1:
        raise ValueError("multi-target NUTS probes require exactly one sample chunk")
    _validate_chunks(settings.sample_chunks)

    leaves = jax.tree.leaves(target_data)
    if not leaves:
        raise ValueError("target_data must contain at least one array leaf")
    for leaf in leaves:
        value = jnp.asarray(leaf)
        if value.ndim < 1 or int(value.shape[0]) != n_targets:
            raise ValueError(
                "every target_data leaf must have the target count as its "
                "leading dimension"
            )

    seed_array = jnp.asarray(seeds, dtype=jnp.uint32)
    if seed_array.shape != (n_targets, n_chains):
        raise ValueError("seeds must have shape [target, chain]")

    flat_positions = positions.reshape((-1, n_parameters))
    flat_targets = jax.tree.map(
        lambda value: jnp.repeat(jnp.asarray(value), n_chains, axis=0),
        target_data,
    )
    flat_seeds = seed_array.reshape((-1,))
    sampling_logdensity = _float64_conditional_logdensity(logdensity_fn)

    target_started = time.perf_counter()
    print(
        "[exact-sampler:nuts-multitarget] validating "
        f"targets={n_targets} chains_per_target={n_chains}",
        flush=True,
    )
    value_and_grad = jax.jit(
        jax.vmap(jax.value_and_grad(sampling_logdensity, argnums=0))
    )
    values, gradients = value_and_grad(flat_positions, flat_targets)
    values, gradients = jax.device_get((values, gradients))
    if np.asarray(values).dtype != np.dtype(np.float64):
        raise TypeError(
            f"logdensity dtype must be float64, got {np.asarray(values).dtype}"
        )
    if np.asarray(gradients).dtype != np.dtype(np.float64):
        raise TypeError(
            "logdensity gradient dtype must be float64, got "
            f"{np.asarray(gradients).dtype}"
        )
    if not np.isfinite(values).all() or not np.isfinite(gradients).all():
        raise ValueError("Initial sampling target values and gradients must be finite")
    target_elapsed = time.perf_counter() - target_started
    print(
        "[exact-sampler:nuts-multitarget] targets ready "
        f"elapsed_s={target_elapsed:.1f}",
        flush=True,
    )

    base_keys = jax.vmap(jax.random.PRNGKey)(flat_seeds)
    split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(base_keys)
    sample_keys = split_keys[:, 0]
    warmup_keys = split_keys[:, 1]

    def adapt_one(key, position, target):
        def conditional_logdensity(x):
            return sampling_logdensity(x, target)

        adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            conditional_logdensity,
            is_mass_matrix_diagonal=True,
            target_acceptance_rate=float(settings.target_accept),
            max_num_doublings=int(settings.max_num_doublings),
            initial_step_size=jnp.asarray(1.0, dtype=jnp.float64),
            adaptation_info_fn=lambda _state, info, _adaptation_state: (
                info.acceptance_rate,
                info.is_divergent,
            ),
        )
        return adaptation.run(key, position, int(settings.warmup_steps))

    warmup_started = time.perf_counter()
    print(
        "[exact-sampler:nuts-multitarget] warmup start "
        f"targets={n_targets} total_chains={n_targets * n_chains} "
        f"steps={settings.warmup_steps} "
        f"max_doublings={settings.max_num_doublings}",
        flush=True,
    )
    (states, parameters), _adaptation_info = jax.jit(jax.vmap(adapt_one))(
        warmup_keys,
        flat_positions,
        flat_targets,
    )
    jax.block_until_ready(states.position)
    warmup_elapsed = time.perf_counter() - warmup_started
    print(
        f"[exact-sampler:nuts-multitarget] warmup done elapsed_s={warmup_elapsed:.1f}",
        flush=True,
    )

    n_samples = int(settings.sample_chunks[0])
    sampling_started = time.perf_counter()
    print(
        f"[exact-sampler:nuts-multitarget] sampling start draws={n_samples}",
        flush=True,
    )
    states, (sample_positions, infos) = _run_batched_nuts_target_steps(
        sampling_logdensity,
        states,
        sample_keys,
        parameters["step_size"],
        parameters["inverse_mass_matrix"],
        flat_targets,
        n_samples=n_samples,
        max_num_doublings=int(settings.max_num_doublings),
    )
    jax.block_until_ready(states.position)
    sampling_elapsed = time.perf_counter() - sampling_started
    print(
        "[exact-sampler:nuts-multitarget] sampling done "
        f"elapsed_s={sampling_elapsed:.1f}",
        flush=True,
    )

    return BatchedTargetNUTSResult(
        positions=sample_positions.reshape(
            (n_samples, n_targets, n_chains, n_parameters)
        ),
        infos=jax.tree.map(
            lambda value: value.reshape(
                (n_samples, n_targets, n_chains, *value.shape[2:])
            ),
            infos,
        ),
        step_size=parameters["step_size"].reshape((n_targets, n_chains)),
        inverse_mass_matrix=parameters["inverse_mass_matrix"].reshape(
            (n_targets, n_chains, n_parameters)
        ),
        target_validation_elapsed_s=target_elapsed,
        warmup_elapsed_s=warmup_elapsed,
        sampling_elapsed_s=sampling_elapsed,
    )


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
    _enforce_float64_sampling()
    import blackjax
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    sampling_logdensity = jax.jit(_float64_logdensity(logdensity_fn), inline=False)
    target_started = time.perf_counter()
    print("[exact-sampler:mclmc] validating target value and gradient", flush=True)
    initial_position = _validate_sampling_target(sampling_logdensity, initial_position)
    print(
        "[exact-sampler:mclmc] target ready "
        f"elapsed_s={time.perf_counter() - target_started:.1f}",
        flush=True,
    )
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
        state = _float64_sampling_state(payload["state"])
        key = payload["key"]
        saved = np.load(params_path)
        parameters = {
            "L": jnp.asarray(saved["L"], dtype=jnp.float64),
            "step_size": jnp.asarray(saved["step_size"], dtype=jnp.float64),
            "inverse_mass_matrix": jnp.asarray(
                saved["inverse_mass_matrix"], dtype=jnp.float64
            ),
        }
        warmup_elapsed = float(payload.get("warmup_elapsed_s", 0.0))
        tuning_integrator_steps = int(payload.get("tuning_integrator_steps", 0))
    else:
        state = blackjax.mcmc.adjusted_mclmc.init(initial_position, sampling_logdensity)
        initial_params = MCLMCAdaptationState(
            L=jnp.asarray(max(math.sqrt(dim), 1.0), dtype=jnp.float64),
            step_size=jnp.asarray(settings.initial_step_size, dtype=jnp.float64),
            inverse_mass_matrix=jnp.ones((dim,), dtype=jnp.float64),
        )

        if int(settings.tune_steps) == 0:
            fixed_integration_steps = 4
            tuned = initial_params._replace(
                L=initial_params.step_size * fixed_integration_steps
            )
            tuning_integrator_steps = 0
            warmup_elapsed = 0.0
            print(
                "[exact-sampler:mclmc] fixed smoke geometry "
                f"step_size={float(initial_params.step_size):.6g} "
                f"integration_steps={fixed_integration_steps}",
                flush=True,
            )
        else:
            adaptation_kernel, adaptation_extra, adaptation_api = (
                _adjusted_mclmc_adaptation_adapter(
                    blackjax,
                    sampling_logdensity,
                )
            )

            warmup_started = time.perf_counter()
            print(
                "[exact-sampler:mclmc] adaptation start "
                f"steps={settings.tune_steps} api={adaptation_api}",
                flush=True,
            )
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
                    **adaptation_extra,
                )
            )
            jax.block_until_ready(state.position)
            warmup_elapsed = time.perf_counter() - warmup_started
            print(
                "[exact-sampler:mclmc] adaptation done "
                f"elapsed_s={warmup_elapsed:.1f} "
                f"integrator_steps={int(tuning_integrator_steps)}",
                flush=True,
            )
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
        print(
            f"[exact-sampler:mclmc] chunk={chunk_id} start "
            f"draws={n_samples} thinning={settings.thinning}",
            flush=True,
        )
        state, positions, infos = _run_algorithm_steps(
            algorithm,
            state,
            chunk_key,
            int(n_samples),
            thinning=int(settings.thinning),
        )
        jax.block_until_ready(state.position)
        elapsed = time.perf_counter() - chunk_started
        print(
            f"[exact-sampler:mclmc] chunk={chunk_id} done elapsed_s={elapsed:.1f}",
            flush=True,
        )
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
        "sampling_dtype": "float64",
        "target_dtype": "float32",
        "adaptation_mode": (
            "fixed_geometry_smoke"
            if int(settings.tune_steps) == 0
            else "blackjax_three_phase"
        ),
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
    _enforce_float64_sampling()
    import blackjax
    from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

    sampling_logdensity = jax.jit(_float64_logdensity(logdensity_fn), inline=False)
    initial_position = _validate_sampling_target(sampling_logdensity, initial_position)
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
        state = _float64_sampling_state(payload["state"])
        key = payload["key"]
        saved = np.load(params_path)
        parameters = {
            "L": jnp.asarray(saved["L"], dtype=jnp.float64),
            "step_size": jnp.asarray(saved["step_size"], dtype=jnp.float64),
            "inverse_mass_matrix": jnp.asarray(
                saved["inverse_mass_matrix"], dtype=jnp.float64
            ),
        }
        warmup_elapsed = float(payload.get("warmup_elapsed_s", 0.0))
        tuning_integrator_steps = int(payload.get("tuning_integrator_steps", 0))
    else:
        state = blackjax.mcmc.mclmc.init(
            initial_position,
            sampling_logdensity,
            init_key,
        )
        initial_params = MCLMCAdaptationState(
            L=jnp.asarray(max(math.sqrt(dim), 1.0), dtype=jnp.float64),
            step_size=jnp.asarray(settings.initial_step_size, dtype=jnp.float64),
            inverse_mass_matrix=jnp.ones((dim,), dtype=jnp.float64),
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
        "sampling_dtype": "float64",
        "target_dtype": "float32",
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


def _adjusted_mclmc_adaptation_adapter(
    blackjax_module: Any,
    sampling_logdensity: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[Callable[..., Any], dict[str, Any], str]:
    """Bridge the released and development BlackJAX adjusted-MCLMC APIs."""
    adaptation = blackjax_module.adjusted_mclmc_find_L_and_step_size
    if "logdensity_fn" in inspect.signature(adaptation).parameters:

        def explicit_logdensity_kernel(
            rng_key,
            state,
            logdensity_fn,
            step_size,
            inverse_mass_matrix,
            integration_steps_params,
        ):
            kernel = blackjax_module.mcmc.adjusted_mclmc.build_kernel()
            return kernel(
                rng_key=rng_key,
                state=state,
                logdensity_fn=logdensity_fn,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                integration_steps_params=integration_steps_params,
            )

        return (
            explicit_logdensity_kernel,
            {"logdensity_fn": sampling_logdensity},
            "explicit_logdensity",
        )

    def closed_logdensity_kernel(
        rng_key,
        state,
        avg_num_integration_steps,
        step_size,
        inverse_mass_matrix,
    ):
        kernel = blackjax_module.mcmc.adjusted_mclmc.build_kernel(
            sampling_logdensity,
            inverse_mass_matrix=inverse_mass_matrix,
        )
        integration_steps = jnp.maximum(
            1,
            jnp.ceil(avg_num_integration_steps).astype(jnp.int32),
        )
        return kernel(
            rng_key,
            state,
            step_size,
            integration_steps,
        )

    return closed_logdensity_kernel, {}, "closed_logdensity"


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
    rhat = diagnostics["rhat"].to_numpy(dtype=np.float64)
    bulk_ess = diagnostics["bulk_ess"].to_numpy(dtype=np.float64)
    tail_ess = diagnostics["tail_ess"].to_numpy(dtype=np.float64)
    finite_rhat = np.isfinite(rhat)
    finite_bulk_ess = np.isfinite(bulk_ess)
    finite_tail_ess = np.isfinite(tail_ess)
    summary = {
        "chains": int(values.shape[0]),
        "draws_per_chain": int(values.shape[1]),
        "max_rhat": float(np.max(rhat)) if finite_rhat.all() else None,
        "min_bulk_ess": (float(np.min(bulk_ess)) if finite_bulk_ess.all() else None),
        "min_tail_ess": (float(np.min(tail_ess)) if finite_tail_ess.all() else None),
        "finite_rhat_parameters": int(finite_rhat.sum()),
        "finite_bulk_ess_parameters": int(finite_bulk_ess.sum()),
        "finite_tail_ess_parameters": int(finite_tail_ess.sum()),
        "passes_rhat_1_01": bool(finite_rhat.all() and (rhat <= 1.01).all()),
        "passes_bulk_ess_400": bool(
            finite_bulk_ess.all() and (bulk_ess >= 400.0).all()
        ),
        "passes_tail_ess_400": bool(
            finite_tail_ess.all() and (tail_ess >= 400.0).all()
        ),
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


def _run_batched_nuts_steps(
    logdensity_fn,
    states,
    keys,
    step_sizes,
    inverse_mass_matrices,
    *,
    n_samples: int,
    max_num_doublings: int,
):
    import blackjax

    def one_step(key, state, step_size, inverse_mass_matrix):
        algorithm = blackjax.nuts(
            logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            max_num_doublings=int(max_num_doublings),
        )
        return algorithm.step(key, state)

    batched_step = jax.vmap(one_step)
    draw_keys = jax.vmap(lambda key: jax.random.split(key, int(n_samples)))(keys)
    draw_keys = jnp.swapaxes(draw_keys, 0, 1)

    def stored_step(current_states, current_keys):
        next_states, infos = batched_step(
            current_keys,
            current_states,
            step_sizes,
            inverse_mass_matrices,
        )
        return next_states, (next_states.position, infos)

    return jax.jit(
        lambda initial_states, all_keys: jax.lax.scan(
            stored_step,
            initial_states,
            all_keys,
        )
    )(states, draw_keys)


def _run_batched_nuts_target_steps(
    logdensity_fn,
    states,
    keys,
    step_sizes,
    inverse_mass_matrices,
    target_data,
    *,
    n_samples: int,
    max_num_doublings: int,
):
    import blackjax

    kernel = blackjax.nuts.build_kernel()

    def one_step(
        key,
        state,
        step_size,
        inverse_mass_matrix,
        target,
    ):
        def conditional_logdensity(x):
            return logdensity_fn(x, target)

        return kernel(
            key,
            state,
            conditional_logdensity,
            step_size,
            inverse_mass_matrix,
            max_num_doublings=int(max_num_doublings),
        )

    batched_step = jax.vmap(one_step)
    draw_keys = jax.vmap(lambda key: jax.random.split(key, int(n_samples)))(keys)
    draw_keys = jnp.swapaxes(draw_keys, 0, 1)

    def stored_step(current_states, current_keys):
        next_states, infos = batched_step(
            current_keys,
            current_states,
            step_sizes,
            inverse_mass_matrices,
            target_data,
        )
        return next_states, (next_states.position, infos)

    return jax.jit(
        lambda initial_states, all_keys: jax.lax.scan(
            stored_step,
            initial_states,
            all_keys,
        )
    )(states, draw_keys)


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


def _float64_logdensity(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Use a float64 sampler around the native float32 DSPS target."""

    def wrapped(position):
        value = logdensity_fn(jnp.asarray(position, dtype=jnp.float32))
        return jnp.asarray(value, dtype=jnp.float64)

    return wrapped


def _float64_conditional_logdensity(
    logdensity_fn: Callable[[jnp.ndarray, Any], jnp.ndarray],
) -> Callable[[jnp.ndarray, Any], jnp.ndarray]:
    """Use a float64 sampler around a native float32 conditional target."""

    def wrapped(position, target):
        value = logdensity_fn(
            jnp.asarray(position, dtype=jnp.float32),
            target,
        )
        return jnp.asarray(value, dtype=jnp.float64)

    return wrapped


def _enforce_float64_sampling() -> None:
    """Enable x64 before BlackJAX constructs any adaptation state."""
    if not jax.config.x64_enabled:
        jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("BlackJAX benchmark requires jax_enable_x64=True")


def _validate_sampling_target(
    logdensity_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_position: jnp.ndarray,
) -> jnp.ndarray:
    """Compile one target evaluation and reject mixed sampler dtypes early."""
    position = jnp.asarray(initial_position, dtype=jnp.float64)
    value, gradient = jax.value_and_grad(logdensity_fn)(position)
    value, gradient = jax.device_get((value, gradient))
    if np.asarray(value).dtype != np.dtype(np.float64):
        raise TypeError(
            f"logdensity dtype must be float64, got {np.asarray(value).dtype}"
        )
    if np.asarray(gradient).dtype != np.dtype(np.float64):
        raise TypeError(
            f"logdensity gradient dtype must be float64, got {np.asarray(gradient).dtype}"
        )
    if not np.isfinite(np.asarray(value)).all() or not np.isfinite(gradient).all():
        raise ValueError("Initial sampling target value and gradient must be finite")
    return position


def _float64_sampling_state(state: Any) -> Any:
    """Promote floating leaves in a resumed BlackJAX state to float64."""

    def promote(value):
        array = jnp.asarray(value)
        if jnp.issubdtype(array.dtype, jnp.floating):
            return array.astype(jnp.float64)
        return value

    return jax.tree.map(promote, state)


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


def _completed_batched_nuts_chunks(
    outputs: tuple[Path, ...],
    sample_chunks: tuple[int, ...],
) -> tuple[int, float]:
    """Infer a complete legacy batch prefix from standard chain artifacts."""
    completed = 0
    elapsed_s = 0.0
    found_gap = False
    for chunk_id, n_samples in enumerate(sample_chunks):
        pairs = [
            (
                output / "chunks" / f"part_{chunk_id:06d}.parquet",
                output / "chunks" / f"part_{chunk_id:06d}_info.parquet",
            )
            for output in outputs
        ]
        pair_complete = [samples.exists() and info.exists() for samples, info in pairs]
        pair_partial = [samples.exists() != info.exists() for samples, info in pairs]
        if any(pair_partial) or (any(pair_complete) and not all(pair_complete)):
            raise RuntimeError(
                f"Partial legacy batched NUTS chunk {chunk_id}; "
                "a common rollback checkpoint is unavailable"
            )
        if not any(pair_complete):
            found_gap = True
            continue
        if found_gap:
            raise RuntimeError(f"Non-contiguous legacy batched NUTS chunk {chunk_id}")
        for samples, info in pairs:
            _validate_nuts_chunk_rows(samples, info, n_samples)
        timing = pd.read_parquet(pairs[0][0], columns=["elapsed_s"])
        values = timing["elapsed_s"].to_numpy(dtype=np.float64)
        if values.size and np.isfinite(values).all():
            elapsed_s += float(values[0])
        completed += 1
    return completed, elapsed_s


def _validate_or_rollback_batched_nuts_chunks(
    outputs: tuple[Path, ...],
    sample_chunks: tuple[int, ...],
    *,
    completed_chunks: int,
) -> None:
    """Validate committed chunks and remove files beyond the batch checkpoint."""
    for chunk_id, n_samples in enumerate(sample_chunks):
        for output in outputs:
            for temporary in (output / "chunks").glob(f"part_{chunk_id:06d}*.tmp-*"):
                temporary.unlink(missing_ok=True)
            samples = output / "chunks" / f"part_{chunk_id:06d}.parquet"
            info = output / "chunks" / f"part_{chunk_id:06d}_info.parquet"
            if chunk_id < completed_chunks:
                if not samples.exists() or not info.exists():
                    raise FileNotFoundError(
                        f"Committed batched NUTS chunk is incomplete: {samples}, {info}"
                    )
                _validate_nuts_chunk_rows(samples, info, n_samples)
            else:
                samples.unlink(missing_ok=True)
                info.unlink(missing_ok=True)


def _validate_nuts_chunk_rows(
    samples_path: Path,
    info_path: Path,
    expected_rows: int,
) -> None:
    sample_rows = len(pd.read_parquet(samples_path, columns=["draw"]))
    info_rows = len(pd.read_parquet(info_path, columns=["draw"]))
    if sample_rows != int(expected_rows) or info_rows != int(expected_rows):
        raise ValueError(
            "Invalid NUTS chunk row count: "
            f"samples={sample_rows} info={info_rows} expected={expected_rows}"
        )


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
