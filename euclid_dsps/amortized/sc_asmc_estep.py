"""Full-catalogue hierarchical E-step and integrated budget preflight."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import numpy as np

from euclid_dsps.io import ensure_dir, write_json

from .data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .features import feature_stats_hash
from .hierarchical_e_step import (
    build_pmap_hierarchy_kernels,
    compiled_hierarchy_memory_analysis,
    hierarchy_result_to_bank_shard,
    run_pmap_model_hierarchical_e_step,
)
from .latent import latent_spec_hash
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    OBSERVED_SELECTION_CONTRACT,
    POSTERIOR_METHOD_CODES,
    PosteriorBankProvenance,
    is_posterior_bank_shard_complete,
    read_posterior_bank_shard,
    sha256_file,
    validate_posterior_bank_shard_provenance,
    write_posterior_bank_shard,
)
from .sc_asmc_config import (
    sc_asmc_em_config_hash,
    sc_asmc_em_hierarchy,
    sc_asmc_em_schedule,
    validate_sc_asmc_em_config,
)
from .sc_asmc_em import HierarchyDispatch, evaluate_budget_preflight
from .sc_asmc_training import (
    RuntimeBundle,
    load_sc_model,
    validate_component_checkpoint,
)
from .train import _loss_batch


def posterior_bank_provenance(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    run_manifest: dict[str, Any],
    *,
    q_checkpoint: str | Path,
    q_ema_checkpoint: str | Path,
    prior_checkpoint: str | Path,
) -> PosteriorBankProvenance:
    validate_sc_asmc_em_config(config)
    for checkpoint in (q_checkpoint, q_ema_checkpoint, prior_checkpoint):
        validate_component_checkpoint(
            checkpoint,
            sha256_file(checkpoint),
            runtime,
        )
    return PosteriorBankProvenance(
        dataset_hash=str(run_manifest["dataset"]["sha256"]),
        workflow_config_hash=str(run_manifest["config_sha256"]),
        q_checkpoint_hash=sha256_file(q_checkpoint),
        q_ema_hash=sha256_file(q_ema_checkpoint),
        prior_checkpoint_hash=sha256_file(prior_checkpoint),
        latent_transform_hash=latent_spec_hash(runtime.latent_spec),
        feature_stats_hash=feature_stats_hash(runtime.feature_stats),
        likelihood_contract={
            "family": "gaussian",
            "equation": "flux | x,fluxerr ~ Normal(DSPS(T(x)), fluxerr^2)",
            "error_floor_frac": 0.0,
            "error_jitter": 0.0,
            "shared_by": ["object posterior", "sleep", "selection completeness"],
        },
        selection_contract={
            "event": OBSERVED_SELECTION_CONTRACT,
            "beta": "P(A=1 | x) under Gaussian PhotoErr",
            "alpha": "E_{x~p_eta}[beta(x)]",
            "enters_object_weights": False,
        },
        code_commit=_git_commit(),
        upstream_selection_provenance=dict(
            run_manifest["dataset"]["upstream_selection_provenance"]
        ),
    )


def run_sc_estep_rows(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    run_manifest: dict[str, Any],
    *,
    row_indices: np.ndarray,
    bank_root: str | Path,
    worker_id: int,
    iteration: int,
    q_checkpoint: str | Path,
    q_ema_checkpoint: str | Path,
    prior_checkpoint: str | Path,
    seed: int,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run IS/SMC over every requested row and persist fixed-size shards."""
    validate_sc_asmc_em_config(config)
    if int(iteration) not in {0, 1, 2}:
        raise ValueError("E-step iteration must be preflight(0), EM1(1), or EM2(2)")
    rows = np.sort(np.asarray(row_indices, dtype=np.int64))
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("E-step rows must be non-empty and unique")
    output = ensure_dir(bank_root)
    receipt_path = output / f"worker_{int(worker_id):02d}_receipt.json"
    provenance = posterior_bank_provenance(
        config,
        runtime,
        run_manifest,
        q_checkpoint=q_checkpoint,
        q_ema_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
    )
    if receipt_path.is_file() and resume:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        _validate_worker_receipt(payload, rows, provenance)
        return payload
    model = load_sc_model(
        config,
        runtime,
        q_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
    )
    hierarchy = sc_asmc_em_hierarchy(config)
    schedule = sc_asmc_em_schedule(config)
    performance = dict(
        ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
            "performance", {}
        )
        or {}
    )
    devices = tuple(jax.local_devices())
    requested_batch_size = _selected_micro_batch_size(
        requested=int(schedule.posterior_bank_shard_objects),
        n_devices=len(devices),
        performance=performance,
    )
    arrays = load_photometry_arrays_from_config(
        runtime.config,
        batch_size=int(
            (config.get("amortized", {}) or {})
            .get("data", {})
            .get("catalog_batch_size", 10_000)
        ),
        row_indices=rows,
    )
    if arrays.truth:
        raise RuntimeError("E-step loaded truth columns")
    first_photometry = next(
        iter_photometry_batches_from_arrays(
            arrays,
            batch_size=requested_batch_size,
            feature_stats=runtime.feature_stats,
            truth_names=None,
        )
    )
    autotune_path = (
        Path(run_manifest["artifacts"]["selected_rows"]["path"]).resolve().parents[1]
        / "runtime"
        / "estep_micro_batch_autotune.json"
    )
    batch_size, kernels, autotune = _autotune_hierarchy_micro_batch(
        config=config,
        runtime=runtime,
        model=model,
        hierarchy=hierarchy,
        sample_batch=_loss_batch(first_photometry),
        requested=requested_batch_size,
        devices=devices,
        performance=performance,
        receipt_path=autotune_path,
        key=jax.random.fold_in(jax.random.PRNGKey(int(seed)), 91_000),
    )
    started = time.perf_counter()
    futures: list[Future] = []
    shard_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="posterior-bank") as pool:
        batches = iter_photometry_batches_from_arrays(
            arrays,
            batch_size=batch_size,
            feature_stats=runtime.feature_stats,
            truth_names=None,
        )
        for batch_index, photometry in enumerate(
            _prefetch_iterator(
                batches,
                depth=int(performance.get("prefetch_host_batches", 2)),
            )
        ):
            global_shard_id = int(worker_id) * 100_000 + batch_index
            shard_path = output / "shards" / f"shard_{global_shard_id:05d}"
            if resume and is_posterior_bank_shard_complete(
                shard_path, validate_arrays=True
            ):
                validate_posterior_bank_shard_provenance(shard_path, provenance)
                shard_records.append(
                    _shard_record(shard_path, global_shard_id, resumed=True)
                )
                continue
            batch_started = time.perf_counter()
            result = run_pmap_model_hierarchical_e_step(
                model_snapshot=model,
                batch=_loss_batch(photometry),
                key=jax.random.fold_in(jax.random.PRNGKey(int(seed)), batch_index),
                kernels=kernels,
            )
            shard = hierarchy_result_to_bank_shard(
                result,
                model_snapshot=model,
                row_index=np.asarray(photometry.row_index, dtype=np.int64),
                object_id=np.asarray(photometry.object_id).astype(str),
                features=np.asarray(jax.device_get(photometry.features)),
                particle_capacity=128,
                primary_config=hierarchy.primary,
                fallback_config=hierarchy.fallback,
                extended_config=hierarchy.extended,
            )
            future = pool.submit(
                write_posterior_bank_shard,
                output,
                global_shard_id,
                shard,
                provenance,
                resume=resume,
            )
            futures.append(future)
            shard_records.append(
                {
                    "shard_id": global_shard_id,
                    "path": str(shard_path.resolve()),
                    "rows": int(shard.object_count),
                    "row_index_min": int(np.min(shard.row_index)),
                    "row_index_max": int(np.max(shard.row_index)),
                    "elapsed_compute_seconds": time.perf_counter() - batch_started,
                    "resumed": False,
                }
            )
            if verbose:
                method = np.asarray(shard.method)
                print(
                    "[sc-asmc][e-step] "
                    f"iteration={iteration} worker={worker_id} batch={batch_index} "
                    f"rows={shard.object_count} resolved={np.mean(shard.resolved):.3f} "
                    f"IS={np.mean(method == POSTERIOR_METHOD_CODES['IS']):.3f} "
                    f"extended={np.mean(method == POSTERIOR_METHOD_CODES['extended SMC']):.3f}",
                    flush=True,
                )
        for future in futures:
            future.result()
    elapsed = time.perf_counter() - started
    payload = {
        "status": "complete",
        "phase": "preflight_e_step" if int(iteration) == 0 else f"e_step_{iteration}",
        "iteration": int(iteration),
        "worker_id": int(worker_id),
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "q_frozen": True,
        "prior_frozen": True,
        "rows": int(len(rows)),
        "row_indices_sha256": _array_hash(rows),
        "elapsed_seconds": elapsed,
        "objects_per_second": len(rows) / max(elapsed, 1.0e-12),
        "devices": [str(device) for device in devices],
        "fixed_shape_batches": {
            "ordinary_and_primary": batch_size,
            "K64": True,
            "K128": True,
            "fallback": kernels.fallback_batch_size,
            "extended_hard_only": kernels.extended_batch_size,
        },
        "micro_batch_autotune": autotune,
        "host_prefetch_depth": int(performance.get("prefetch_host_batches", 2)),
        "shards": shard_records,
        "provenance": {
            "q_checkpoint_hash": provenance.q_checkpoint_hash,
            "q_ema_hash": provenance.q_ema_hash,
            "prior_checkpoint_hash": provenance.prior_checkpoint_hash,
        },
    }
    write_json(receipt_path, payload)
    return payload


def run_integrated_budget_preflight(
    config: dict[str, Any],
    runtime: RuntimeBundle,
    run_manifest: dict[str, Any],
    *,
    preflight_rows: np.ndarray,
    out_dir: str | Path,
    attempt: int,
    q_checkpoint: str | Path,
    q_ema_checkpoint: str | Path,
    prior_checkpoint: str | Path,
    seed: int,
    parallel_shards: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the same hierarchy as production and apply the fail-closed gate."""
    if int(attempt) not in {1, 2}:
        raise ValueError("integrated preflight attempt must be one or two")
    rows = np.asarray(preflight_rows, dtype=np.int64)
    if len(rows) != 512:
        raise ValueError("integrated budget preflight requires exactly 512 rows")
    output = ensure_dir(out_dir)
    bank_root = output / f"attempt_{int(attempt)}" / "bank"
    worker = run_sc_estep_rows(
        config,
        runtime,
        run_manifest,
        row_indices=rows,
        bank_root=bank_root,
        worker_id=90 + int(attempt),
        iteration=0,
        q_checkpoint=q_checkpoint,
        q_ema_checkpoint=q_ema_checkpoint,
        prior_checkpoint=prior_checkpoint,
        seed=int(seed) + int(attempt),
        resume=True,
        verbose=verbose,
    )
    shards = [read_posterior_bank_shard(record["path"]) for record in worker["shards"]]
    method = np.concatenate([shard.method for shard in shards])
    resolved = np.concatenate([shard.resolved for shard in shards])
    dispatch = HierarchyDispatch(
        method=method,
        resolved=resolved,
        primary_attempted=method != POSTERIOR_METHOD_CODES["IS"],
        fallback_attempted=np.isin(
            method,
            [
                POSTERIOR_METHOD_CODES["fallback SMC"],
                POSTERIOR_METHOD_CODES["extended SMC"],
                POSTERIOR_METHOD_CODES["unresolved"],
            ],
        ),
        extended_attempted=np.isin(
            method,
            [
                POSTERIOR_METHOD_CODES["extended SMC"],
                POSTERIOR_METHOD_CODES["unresolved"],
            ],
        ),
    )
    schedule = sc_asmc_em_schedule(config)
    preflight_config = dict(
        ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
            "preflight", {}
        )
        or {}
    )
    gate = evaluate_budget_preflight(
        dispatch,
        elapsed_seconds=float(worker["elapsed_seconds"]),
        dsps_evaluations=np.concatenate([shard.dsps_evaluations for shard in shards]),
        stage_count=np.concatenate([shard.stage_count for shard in shards]),
        mutation_acceptance=np.concatenate([shard.acceptance for shard in shards]),
        ancestry_ess=np.concatenate([shard.ancestor_ess for shard in shards]),
        movement_squared=np.concatenate([shard.movement_squared for shard in shards]),
        beta_final=np.concatenate([shard.beta_final for shard in shards]),
        full_catalogue_objects=int(run_manifest["objects"]["selected"]),
        e_step_iterations=2,
        parallel_shards=int(parallel_shards),
        job_budget_seconds=float(schedule.job_budget_seconds),
        non_estep_overhead_fraction=float(
            preflight_config.get("projected_non_estep_overhead_fraction", 0.20)
        ),
        attempt=int(attempt),
    )
    payload = {
        "status": gate.status,
        "phase": "budget_preflight",
        "attempt": int(attempt),
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "scientific_training_subset": False,
        "same_hierarchy_as_full_e_step": True,
        "continue_full_catalogue": gate.continue_full_catalogue,
        "active_bootstrap_required": gate.active_bootstrap_required,
        "checks": gate.checks,
        "metrics": gate.metrics,
        "method_counts": gate.method_counts,
        "projected_full_run_wall_seconds": gate.projected_full_run_wall_seconds,
        "worker_receipt": str(
            (bank_root / f"worker_{91 + int(attempt) - 1:02d}_receipt.json").resolve()
        ),
    }
    receipt = output / f"preflight_attempt_{int(attempt)}_receipt.json"
    write_json(receipt, payload)
    if gate.continue_full_catalogue:
        write_json(output / "PREFLIGHT_PASS.json", payload)
    elif gate.status == "ABORT":
        write_json(output / "PREFLIGHT_ABORT.json", payload)
    return payload


def summarize_bank_shards(shard_paths: list[str | Path]) -> dict[str, Any]:
    method = []
    resolved = []
    evaluations = []
    for path in shard_paths:
        shard = read_posterior_bank_shard(path)
        method.append(shard.method)
        resolved.append(shard.resolved)
        evaluations.append(shard.dsps_evaluations)
    methods = np.concatenate(method)
    okay = np.concatenate(resolved)
    evals = np.concatenate(evaluations)
    return {
        "objects": int(len(methods)),
        "resolved_fraction": float(np.mean(okay)),
        "method_counts": {
            name: int(np.sum(methods == code))
            for name, code in POSTERIOR_METHOD_CODES.items()
        },
        "mean_dsps_evaluations": float(np.mean(evals)),
        "total_dsps_evaluations": int(np.sum(evals)),
    }


def _selected_micro_batch_size(
    *,
    requested: int,
    n_devices: int,
    performance: dict[str, Any],
) -> int:
    if int(n_devices) <= 0:
        raise RuntimeError("no local devices for E-step")
    override = performance.get("object_micro_batch_size")
    size = int(override if override is not None else requested)
    size = max(int(n_devices), size - size % int(n_devices))
    if size <= 0:
        raise ValueError("object micro-batch cannot be made device-divisible")
    return size


def _autotune_hierarchy_micro_batch(
    *,
    config: dict[str, Any],
    runtime: RuntimeBundle,
    model: Any,
    hierarchy: Any,
    sample_batch: Any,
    requested: int,
    devices: tuple[Any, ...],
    performance: dict[str, Any],
    receipt_path: Path,
    key: jax.Array,
) -> tuple[int, Any, dict[str, Any]]:
    device_kinds = [str(device.device_kind) for device in devices]
    input_contract = {
        "workflow_config_hash": sc_asmc_em_config_hash(config),
        "latent_transform_hash": latent_spec_hash(runtime.latent_spec),
        "feature_stats_hash": feature_stats_hash(runtime.feature_stats),
    }
    enabled = bool(performance.get("autotune_object_micro_batch", True))
    override = performance.get("object_micro_batch_size")
    if receipt_path.is_file():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError("E-step micro-batch autotune receipt is incomplete")
        if int(payload.get("requested_batch_size", -1)) != int(requested):
            raise ValueError("cached E-step autotune has a different requested size")
        if payload.get("device_kinds") != device_kinds:
            raise ValueError(
                "cached E-step autotune belongs to another device topology"
            )
        if any(payload.get(name) != value for name, value in input_contract.items()):
            raise ValueError("cached E-step autotune inputs changed")
        selected = int(payload["selected_batch_size"])
        return (
            selected,
            _build_hierarchy_kernels(runtime, hierarchy, selected, devices),
            payload,
        )

    if override is not None or not enabled:
        selected = int(requested)
        payload = {
            "status": "complete",
            "mode": "configured_override" if override is not None else "disabled",
            "requested_batch_size": int(requested),
            "selected_batch_size": selected,
            "device_kinds": device_kinds,
            "device_count": len(devices),
            **input_contract,
            "truth_used": False,
            "c0_scope_statement": C0_SCOPE_STATEMENT,
        }
        write_json(receipt_path, payload)
        return (
            selected,
            _build_hierarchy_kernels(runtime, hierarchy, selected, devices),
            payload,
        )

    target = float(performance.get("target_device_memory_fraction", 0.88))
    maximum = float(performance.get("maximum_compiled_memory_fraction", 0.90))
    if not 0.0 < target <= maximum <= 0.95:
        raise ValueError("E-step memory target must satisfy 0 < target <= max <= 0.95")
    memory_limits = _device_memory_limits(devices)
    usable_limit = min(memory_limits) if memory_limits else None
    attempts = []
    selected = None
    selected_kernels = None
    for candidate in _micro_batch_candidates(int(requested), len(devices)):
        kernels = _build_hierarchy_kernels(runtime, hierarchy, candidate, devices)
        try:
            analysis = compiled_hierarchy_memory_analysis(
                model_snapshot=model,
                batch=sample_batch,
                key=jax.random.fold_in(key, candidate),
                kernels=kernels,
            )
        except (RuntimeError, MemoryError) as error:
            attempts.append(
                {
                    "batch_size": candidate,
                    "status": "compile_failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        peak = max(value["peak_memory_in_bytes"] for value in analysis.values())
        fraction = None if usable_limit is None else peak / float(usable_limit)
        attempts.append(
            {
                "batch_size": candidate,
                "status": "compiled",
                "compiler_peak_memory_bytes": peak,
                "compiler_peak_memory_fraction": fraction,
                "kernels": analysis,
            }
        )
        if usable_limit is None or peak <= maximum * usable_limit:
            selected = candidate
            selected_kernels = kernels
            break
    if selected is None or selected_kernels is None:
        raise RuntimeError("no E-step micro-batch fits the compiled memory budget")
    selected_attempt = attempts[-1]
    selected_fraction = selected_attempt.get("compiler_peak_memory_fraction")
    payload = {
        "status": "complete",
        "mode": "xla_compiled_memory",
        "requested_batch_size": int(requested),
        "selected_batch_size": int(selected),
        "candidate_batch_sizes": _micro_batch_candidates(int(requested), len(devices)),
        "target_device_memory_fraction": target,
        "maximum_compiled_memory_fraction": maximum,
        "selected_compiler_peak_memory_fraction": selected_fraction,
        "allocator_preallocation_fraction": os.environ.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION"
        ),
        "device_memory_limits_bytes": memory_limits,
        "device_kinds": device_kinds,
        "device_count": len(devices),
        "attempts": attempts,
        "selection_reason": (
            "largest fixed shape under compiler memory ceiling"
            if selected_fraction is not None
            else "device memory limit unavailable; largest compiled fixed shape"
        ),
        "truth_used": False,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "config_contract": validate_sc_asmc_em_config(config)["status"],
        **input_contract,
    }
    write_json(receipt_path, payload)
    return int(selected), selected_kernels, payload


def _build_hierarchy_kernels(
    runtime: RuntimeBundle,
    hierarchy: Any,
    batch_size: int,
    devices: tuple[Any, ...],
) -> Any:
    return build_pmap_hierarchy_kernels(
        latent_spec=runtime.jit_latent_spec,
        context=runtime.context,
        model_args=runtime.model_args,
        parameter_names=runtime.parameter_names,
        likelihood_config=runtime.likelihood_config,
        calibration_config=runtime.calibration_config,
        primary_config=hierarchy.primary,
        fallback_config=hierarchy.fallback,
        extended_config=hierarchy.extended,
        proposal_config=hierarchy.proposal,
        minimum_is_ess_fraction=hierarchy.minimum_is_ess_fraction,
        maximum_is_weight=hierarchy.maximum_is_weight,
        primary_batch_size=int(batch_size),
        fallback_batch_size=max(len(devices), min(32, int(batch_size))),
        extended_batch_size=max(len(devices), min(16, int(batch_size))),
        devices=devices,
    )


def _micro_batch_candidates(requested: int, n_devices: int) -> list[int]:
    if int(requested) <= 0 or int(n_devices) <= 0:
        raise ValueError("micro-batch candidates require positive sizes")
    raw = (
        int(requested),
        3 * int(requested) // 4,
        int(requested) // 2,
        int(requested) // 4,
        4 * int(n_devices),
        int(n_devices),
    )
    candidates = {max(int(n_devices), value - value % int(n_devices)) for value in raw}
    return sorted(
        (value for value in candidates if value <= int(requested)), reverse=True
    )


def _device_memory_limits(devices: tuple[Any, ...]) -> list[int]:
    limits = []
    for device in devices:
        stats = device.memory_stats() or {}
        value = stats.get("bytes_limit")
        if value is None:
            return []
        limits.append(int(value))
    return limits


def _prefetch_iterator(values: Iterator[Any], *, depth: int) -> Iterator[Any]:
    if int(depth) <= 0:
        yield from values
        return
    iterator = iter(values)
    sentinel = object()

    def next_or_sentinel() -> Any:
        try:
            return next(iterator)
        except StopIteration:
            return sentinel

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="photometry-prefetch"
    ) as pool:
        futures = [pool.submit(next_or_sentinel) for _ in range(int(depth))]
        while futures:
            future = futures.pop(0)
            value = future.result()
            if value is sentinel:
                for pending in futures:
                    pending.cancel()
                return
            futures.append(pool.submit(next_or_sentinel))
            yield value


def _shard_record(path: Path, shard_id: int, *, resumed: bool) -> dict[str, Any]:
    shard = read_posterior_bank_shard(path)
    return {
        "shard_id": int(shard_id),
        "path": str(path.resolve()),
        "rows": shard.object_count,
        "row_index_min": int(np.min(shard.row_index)),
        "row_index_max": int(np.max(shard.row_index)),
        "elapsed_compute_seconds": 0.0,
        "resumed": bool(resumed),
    }


def _validate_worker_receipt(
    payload: dict[str, Any],
    rows: np.ndarray,
    provenance: PosteriorBankProvenance,
) -> None:
    if payload.get("status") != "complete":
        raise ValueError("E-step worker receipt is not complete")
    if payload.get("row_indices_sha256") != _array_hash(rows):
        raise ValueError("E-step resume rows do not match worker receipt")
    records = payload.get("shards", [])
    if not records:
        raise ValueError("E-step worker receipt contains no shards")
    resumed_rows = []
    for record in records:
        if not is_posterior_bank_shard_complete(record["path"], validate_arrays=True):
            raise ValueError("E-step worker receipt references an incomplete shard")
        validate_posterior_bank_shard_provenance(record["path"], provenance)
        resumed_rows.append(read_posterior_bank_shard(record["path"]).row_index)
    if not np.array_equal(np.sort(np.concatenate(resumed_rows)), np.sort(rows)):
        raise ValueError("E-step worker receipt does not cover its assigned rows")


def _array_hash(values: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown-non-git-checkout"
