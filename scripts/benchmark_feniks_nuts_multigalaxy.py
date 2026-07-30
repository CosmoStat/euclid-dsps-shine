#!/usr/bin/env python3
"""Benchmark distinct-galaxy NUTS batching on one accelerator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("EUCLID_DSPS_DISABLE_JAX_PLUGIN_AUTOLOAD", "0")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset

from euclid_dsps.amortized.exact_posterior import (
    NUTSSettings,
    run_batched_nuts_targets,
)
from euclid_dsps.amortized.latent import x_to_theta
from euclid_dsps.amortized.posterior import sample_posterior
from euclid_dsps.config import load_config

try:
    from scripts.run_feniks_exact_posterior_benchmark import (
        _conditional_logdensity_fn,
        _load_runtime_rows,
    )
except ModuleNotFoundError as error:
    if error.name not in {
        "scripts",
        "scripts.run_feniks_exact_posterior_benchmark",
    }:
        raise
    from run_feniks_exact_posterior_benchmark import (
        _conditional_logdensity_fn,
        _load_runtime_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-cohort")
    prepare.add_argument("--dataset", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--max-galaxies", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=260730)

    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--feature-stats", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--batch-size", type=int, required=True)
    run.add_argument("--chains", type=int, default=4)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--draws", type=int, default=10)
    run.add_argument("--max-num-doublings", type=int, default=4)
    run.add_argument("--target-accept", type=float, default=0.65)
    run.add_argument("--seed", type=int, default=260730)
    run.add_argument("--gpu-metrics", type=Path)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--out", type=Path, required=True)
    summarize.add_argument("--batch-sizes", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare-cohort":
        prepare_cohort(args)
    elif args.command == "run":
        run_probe(args)
    else:
        summarize_probe(args)


def prepare_cohort(args: argparse.Namespace) -> None:
    if args.max_galaxies < 1:
        raise ValueError("--max-galaxies must be positive")
    dataset = pyarrow_dataset.dataset(str(args.dataset), format="parquet")
    n_rows = int(dataset.count_rows())
    if args.max_galaxies > n_rows:
        raise ValueError(
            f"Requested {args.max_galaxies} galaxies from a {n_rows}-row dataset"
        )
    rng = np.random.default_rng(int(args.seed))
    row_indices = rng.choice(
        n_rows,
        size=int(args.max_galaxies),
        replace=False,
    )
    cohort = pd.DataFrame(
        {
            "cohort_index": np.arange(len(row_indices), dtype=np.int64),
            "row_index": row_indices.astype(np.int64),
        }
    )
    args.out.mkdir(parents=True, exist_ok=False)
    cohort.to_csv(args.out / "cohort.csv", index=False)
    cohort.to_parquet(args.out / "cohort.parquet", index=False)
    _write_json(
        args.out / "capacity_contract.json",
        {
            "status": "cohort_prepared",
            "purpose": "performance_only_not_convergence",
            "dataset": str(args.dataset.resolve()),
            "dataset_rows": n_rows,
            "max_galaxies": int(args.max_galaxies),
            "selection": "seeded_without_replacement",
            "seed": int(args.seed),
            "row_indices": row_indices.astype(int).tolist(),
            "code_commit": _git_commit(),
        },
    )
    (args.out / "COHORT_DONE").touch()


def run_probe(args: argparse.Namespace) -> None:
    batch_dir = args.out / f"batch_g{int(args.batch_size):04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_probe(args, batch_dir)
    except BaseException as error:
        _write_json(
            batch_dir / "failure.json",
            {
                "status": "failed",
                "batch_size": int(args.batch_size),
                "exception_type": type(error).__name__,
                "exception": str(error),
                "gpu": _gpu_summary(args.gpu_metrics),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def _run_probe(args: argparse.Namespace, batch_dir: Path) -> None:
    if args.batch_size < 1 or args.chains < 1:
        raise ValueError("--batch-size and --chains must be positive")
    if args.warmup < 1 or args.draws < 1:
        raise ValueError("--warmup and --draws must be positive")
    cohort = pd.read_parquet(args.out / "cohort.parquet")
    if args.batch_size > len(cohort):
        raise ValueError(
            f"Batch size {args.batch_size} exceeds cohort size {len(cohort)}"
        )
    selected = cohort.iloc[: int(args.batch_size)].copy()
    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    runtime = _load_runtime_rows(
        args,
        config,
        selected["row_index"].to_numpy(dtype=np.int64),
    )
    actual_rows = np.asarray(runtime.batch.row_index, dtype=np.int64)
    if not np.array_equal(
        actual_rows,
        selected["row_index"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Loaded batch order differs from the capacity cohort")

    key = jax.random.PRNGKey(int(args.seed) + int(args.batch_size) * 1_000)
    key, encoder_key = jax.random.split(key)
    posterior = sample_posterior(
        runtime.model,
        encoder_key,
        runtime.batch.features,
        int(args.chains),
    )
    initial_positions = jnp.swapaxes(posterior.x, 0, 1)
    target_data = (
        runtime.batch.flux,
        runtime.batch.flux_err,
        runtime.batch.mask,
    )
    seeds = (
        int(args.seed)
        + np.arange(int(args.batch_size), dtype=np.uint32)[:, None] * 10_000
        + np.arange(int(args.chains), dtype=np.uint32)[None, :] * 101
    ).astype(np.uint32)
    settings = NUTSSettings(
        warmup_steps=int(args.warmup),
        sample_chunks=(int(args.draws),),
        target_accept=float(args.target_accept),
        max_num_doublings=int(args.max_num_doublings),
    )
    _write_json(
        batch_dir / "run_contract.json",
        {
            "status": "running",
            "purpose": "performance_only_not_convergence",
            "batch_size_galaxies": int(args.batch_size),
            "chains_per_galaxy": int(args.chains),
            "total_chains": int(args.batch_size) * int(args.chains),
            "warmup_steps": int(args.warmup),
            "draws_per_chain": int(args.draws),
            "target_accept": float(args.target_accept),
            "max_num_doublings": int(args.max_num_doublings),
            "config": str(args.config.resolve()),
            "dataset": str(args.dataset.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "feature_stats": str(args.feature_stats.resolve()),
            "row_indices": actual_rows.astype(int).tolist(),
            "object_ids": [
                str(value) for value in np.asarray(runtime.batch.object_id)
            ],
            "code_commit": _git_commit(),
        },
    )

    result = run_batched_nuts_targets(
        _conditional_logdensity_fn(runtime),
        initial_positions,
        target_data,
        seeds=jnp.asarray(seeds),
        settings=settings,
    )
    positions = np.asarray(jax.device_get(result.positions))
    theta = np.asarray(
        jax.device_get(x_to_theta(result.positions, runtime.latent_spec))
    )
    infos = jax.device_get(result.infos)
    _write_samples(
        batch_dir / "samples.parquet",
        positions,
        theta,
        parameter_names=runtime.latent_spec.names,
        row_indices=actual_rows,
        object_ids=np.asarray(runtime.batch.object_id),
    )
    info_frame = _write_infos(
        batch_dir / "nuts_info.parquet",
        infos,
        row_indices=actual_rows,
        object_ids=np.asarray(runtime.batch.object_id),
    )
    np.savez_compressed(
        batch_dir / "tuned_parameters.npz",
        step_size=np.asarray(jax.device_get(result.step_size)),
        inverse_mass_matrix=np.asarray(
            jax.device_get(result.inverse_mass_matrix)
        ),
    )
    selected["object_id"] = [
        str(value) for value in np.asarray(runtime.batch.object_id)
    ]
    selected.to_csv(batch_dir / "cohort_used.csv", index=False)
    selected.to_parquet(batch_dir / "cohort_used.parquet", index=False)

    sampling_transitions = (
        int(args.batch_size) * int(args.chains) * int(args.draws)
    )
    galaxy_sweeps = int(args.batch_size) * int(args.draws)
    summary = {
        "status": "completed",
        "purpose": "performance_only_not_convergence",
        "batch_size_galaxies": int(args.batch_size),
        "chains_per_galaxy": int(args.chains),
        "total_chains": int(args.batch_size) * int(args.chains),
        "warmup_steps": int(args.warmup),
        "draws_per_chain": int(args.draws),
        "max_num_doublings": int(args.max_num_doublings),
        "target_validation_elapsed_s": result.target_validation_elapsed_s,
        "warmup_elapsed_s": result.warmup_elapsed_s,
        "sampling_elapsed_s": result.sampling_elapsed_s,
        "total_sampler_elapsed_s": (
            result.target_validation_elapsed_s
            + result.warmup_elapsed_s
            + result.sampling_elapsed_s
        ),
        "sampling_transitions": sampling_transitions,
        "sampling_transitions_per_s": (
            sampling_transitions / result.sampling_elapsed_s
        ),
        "galaxy_sweeps_per_s": galaxy_sweeps / result.sampling_elapsed_s,
        "mean_acceptance_rate": _finite_mean(
            info_frame.get("acceptance_rate")
        ),
        "divergences": int(
            info_frame.get(
                "is_divergent",
                pd.Series(dtype=bool),
            ).sum()
        ),
        "mean_integration_steps": _finite_mean(
            info_frame.get("num_integration_steps")
        ),
        "gpu": _gpu_summary(args.gpu_metrics),
        "artifacts": {
            "samples": "samples.parquet",
            "nuts_info": "nuts_info.parquet",
            "tuned_parameters": "tuned_parameters.npz",
            "cohort": "cohort_used.parquet",
        },
    }
    _write_json(batch_dir / "benchmark.json", summary)
    (batch_dir / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)


def summarize_probe(args: argparse.Namespace) -> None:
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        batch_dir = args.out / f"batch_g{batch_size:04d}"
        benchmark = batch_dir / "benchmark.json"
        failure = batch_dir / "failure.json"
        if benchmark.is_file() and (batch_dir / "DONE").is_file():
            payload = json.loads(benchmark.read_text(encoding="utf-8"))
            gpu = payload.get("gpu", {})
            rows.append(
                {
                    "batch_size_galaxies": batch_size,
                    "status": "completed",
                    "total_chains": payload["total_chains"],
                    "total_sampler_elapsed_s": payload[
                        "total_sampler_elapsed_s"
                    ],
                    "sampling_transitions_per_s": payload[
                        "sampling_transitions_per_s"
                    ],
                    "galaxy_sweeps_per_s": payload["galaxy_sweeps_per_s"],
                    "peak_hbm_mib": gpu.get("peak_memory_used_mib"),
                    "hbm_total_mib": gpu.get("memory_total_mib"),
                    "peak_hbm_fraction": gpu.get("peak_memory_fraction"),
                    "median_gpu_utilization_percent": gpu.get(
                        "median_utilization_gpu_percent"
                    ),
                    "divergences": payload["divergences"],
                }
            )
        elif failure.is_file():
            payload = json.loads(failure.read_text(encoding="utf-8"))
            rows.append(
                {
                    "batch_size_galaxies": batch_size,
                    "status": "failed",
                    "failure_type": payload.get("exception_type"),
                    "failure": payload.get("exception"),
                }
            )
        else:
            rows.append(
                {
                    "batch_size_galaxies": batch_size,
                    "status": "missing",
                }
            )
    frame = pd.DataFrame(rows).sort_values("batch_size_galaxies")
    frame.to_csv(args.out / "capacity_summary.csv", index=False)
    completed = frame[frame["status"] == "completed"]
    if "peak_hbm_fraction" in completed:
        memory_safe = completed[
            pd.to_numeric(
                completed["peak_hbm_fraction"],
                errors="coerce",
            )
            <= 0.8
        ]
    else:
        memory_safe = completed.iloc[:0]
    largest_memory_safe = (
        int(memory_safe["batch_size_galaxies"].max())
        if len(memory_safe)
        else None
    )
    if len(completed):
        throughput = pd.to_numeric(
            completed["sampling_transitions_per_s"],
            errors="coerce",
        )
        throughput_optimal = (
            int(completed.loc[throughput.idxmax(), "batch_size_galaxies"])
            if throughput.notna().any()
            else None
        )
    else:
        throughput_optimal = None
    largest_tested = max(batch_sizes)
    largest_completed = (
        int(completed["batch_size_galaxies"].max())
        if len(completed)
        else None
    )
    summary = {
        "status": "complete",
        "purpose": "performance_only_not_convergence",
        "tested_batch_sizes": batch_sizes,
        "completed_batch_sizes": completed["batch_size_galaxies"]
        .astype(int)
        .tolist(),
        "failed_batch_sizes": frame.loc[
            frame["status"] == "failed", "batch_size_galaxies"
        ]
        .astype(int)
        .tolist(),
        "largest_completed_batch_size": largest_completed,
        "largest_tested_batch_at_most_80pct_hbm": largest_memory_safe,
        "throughput_optimal_tested_batch_size": throughput_optimal,
        "capacity_limit_observed": bool(
            largest_completed != largest_tested
            or largest_memory_safe != largest_tested
        ),
        "larger_followup_needed": bool(
            largest_completed == largest_tested
            and largest_memory_safe == largest_tested
        ),
        "recommendation_requires_scientific_warmup_validation": True,
    }
    _write_json(args.out / "capacity_summary.json", summary)
    _write_capacity_plot(frame, args.out / "capacity_scaling.png")
    (args.out / "DONE").touch()
    print(json.dumps(summary, indent=2), flush=True)


def _write_samples(
    path: Path,
    positions: np.ndarray,
    theta: np.ndarray,
    *,
    parameter_names: tuple[str, ...],
    row_indices: np.ndarray,
    object_ids: np.ndarray,
) -> None:
    n_draws, n_galaxies, n_chains, n_parameters = positions.shape
    draw, galaxy, chain = np.indices(
        (n_draws, n_galaxies, n_chains)
    )
    frame = pd.DataFrame(
        {
            "draw": draw.reshape(-1),
            "galaxy_index": galaxy.reshape(-1),
            "chain": chain.reshape(-1),
            "row_index": row_indices[galaxy.reshape(-1)],
            "object_id": object_ids[galaxy.reshape(-1)].astype(str),
        }
    )
    flat_x = positions.reshape((-1, n_parameters))
    flat_theta = theta.reshape((-1, n_parameters))
    for index, name in enumerate(parameter_names):
        frame[f"x_{name}"] = flat_x[:, index]
        frame[name] = flat_theta[:, index]
    frame.to_parquet(path, index=False)


def _write_infos(
    path: Path,
    infos: Any,
    *,
    row_indices: np.ndarray,
    object_ids: np.ndarray,
) -> pd.DataFrame:
    first = np.asarray(infos.acceptance_rate)
    n_draws, n_galaxies, n_chains = first.shape
    draw, galaxy, chain = np.indices(
        (n_draws, n_galaxies, n_chains)
    )
    frame = pd.DataFrame(
        {
            "draw": draw.reshape(-1),
            "galaxy_index": galaxy.reshape(-1),
            "chain": chain.reshape(-1),
            "row_index": row_indices[galaxy.reshape(-1)],
            "object_id": object_ids[galaxy.reshape(-1)].astype(str),
        }
    )
    for name in (
        "acceptance_rate",
        "is_divergent",
        "num_integration_steps",
        "energy",
        "proposal_energy",
    ):
        if hasattr(infos, name):
            frame[name] = np.asarray(getattr(infos, name)).reshape(-1)
    frame.to_parquet(path, index=False)
    return frame


def _gpu_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file() or path.stat().st_size == 0:
        return {"samples": 0}
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return {"samples": 0}
    numeric_names = (
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
        "power_draw_w",
    )
    for name in numeric_names:
        if name in frame:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    memory_used = frame.get("memory_used_mib", pd.Series(dtype=float))
    memory_total = frame.get("memory_total_mib", pd.Series(dtype=float))
    utilization = frame.get(
        "utilization_gpu_percent", pd.Series(dtype=float)
    )
    power = frame.get("power_draw_w", pd.Series(dtype=float))
    peak_memory = _finite_max(memory_used)
    total_memory = _finite_max(memory_total)
    return {
        "samples": int(len(frame)),
        "peak_memory_used_mib": peak_memory,
        "memory_total_mib": total_memory,
        "peak_memory_fraction": (
            peak_memory / total_memory
            if peak_memory is not None
            and total_memory is not None
            and total_memory > 0
            else None
        ),
        "median_utilization_gpu_percent": _finite_median(utilization),
        "peak_utilization_gpu_percent": _finite_max(utilization),
        "median_power_draw_w": _finite_median(power),
    }


def _write_capacity_plot(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    completed = frame[frame["status"] == "completed"].copy()
    if not len(completed):
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    x = completed["batch_size_galaxies"].to_numpy(dtype=float)
    axes[0].plot(
        x,
        completed["sampling_transitions_per_s"],
        marker="o",
    )
    axes[0].set_ylabel("NUTS transitions / s")
    axes[1].plot(x, completed["peak_hbm_mib"] / 1024.0, marker="o")
    axes[1].set_ylabel("Peak HBM [GiB]")
    axes[2].plot(
        x,
        completed["median_gpu_utilization_percent"],
        marker="o",
    )
    axes[2].set_ylabel("Median GPU utilization [%]")
    for axis in axes:
        axis.set_xlabel("Galaxies per H100")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    fig.suptitle("FENIKS NUTS multi-galaxy capacity probe")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_batch_sizes(value: str) -> list[int]:
    parsed = [
        int(item)
        for item in value.replace(",", ":").split(":")
        if item
    ]
    if not parsed or any(item < 1 for item in parsed):
        raise ValueError("batch sizes must be positive")
    if len(set(parsed)) != len(parsed):
        raise ValueError("batch sizes must be unique")
    return parsed


def _finite_mean(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else None


def _finite_max(values: pd.Series) -> float | None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if len(finite) else None


def _finite_median(values: pd.Series) -> float | None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if len(finite) else None


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
