#!/usr/bin/env python3
"""Compare DSPS dense, DSPS compressed, and FSPS/Prospector photometry.

Each engine/level runs in a subprocess so an out-of-memory dense resident run
is reported as a failed engine instead of killing the whole benchmark driver.
Successful workers write photometry rows plus wall-time and peak-RSS metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_against_fsps_prospector import (  # noqa: E402
    AGN_AUDIT_LEVELS,
    _model_mags_with_lazy_fsps_component_grid_agn,
    _parameters_for_level,
    reference_mags_fsps_prospector,
    sample_parameter_points,
)
from benchmark_dense_vs_compressed_spectral_assets import (  # noqa: E402
    AGN_LEVELS,
    GAS_LEVELS,
    apply_compressed_overrides,
    apply_runtime_override,
    load_level_context,
    model_without_agn,
)
from euclid_dsps.config import load_config  # noqa: E402
from euclid_dsps.filters import load_filters  # noqa: E402
from euclid_dsps.jax_runtime import apply_jax_runtime_env  # noqa: E402
from euclid_dsps.photometry import abmag_to_fnu_cgs  # noqa: E402

ENGINES = (
    "dsps_dense_lazy",
    "dsps_dense_resident",
    "dsps_compressed",
    "fsps_prospector",
)
DEFAULT_ENGINES = ("dsps_dense_lazy", "dsps_compressed", "fsps_prospector")
DEFAULT_LEVELS = ("stellar_plus_gas", "full_noagn", "stellar_plus_agn", "full_agn")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/popcosmos_binned.yaml")
    parser.add_argument("--compressed-gas-grid", default=None)
    parser.add_argument("--compressed-ssp", default=None)
    parser.add_argument("--compressed-agn-component-grid", default=None)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=ENGINES,
        default=list(DEFAULT_ENGINES),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=AGN_AUDIT_LEVELS,
        default=list(DEFAULT_LEVELS),
    )
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--runtime",
        choices=("config", "auto", "cpu", "gpu"),
        default="cpu",
    )
    parser.add_argument("--stellar-ssp", default=None)
    parser.add_argument(
        "--out",
        default="outputs/benchmarks/photometry_engines",
    )
    parser.add_argument(
        "--memory-guard-fraction",
        type=float,
        default=0.70,
        help=(
            "Skip dsps_dense_resident workers when estimated resident payload "
            "times overhead exceeds this fraction of MemAvailable."
        ),
    )
    parser.add_argument(
        "--memory-overhead-factor",
        type=float,
        default=1.8,
        help="Safety multiplier applied to static dense resident payload estimates.",
    )
    parser.add_argument(
        "--no-memory-guard",
        action="store_true",
        help="Disable resident-memory preflight checks.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engine", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--level", choices=AGN_AUDIT_LEVELS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)
    return parent_main(args)


def parent_main(args: argparse.Namespace) -> int:
    out = Path(args.out)
    worker_dir = out / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    results = []
    rows = []
    total = len(args.engines) * len(args.levels)
    done = 0
    for engine in args.engines:
        for level in args.levels:
            done += 1
            if not args.no_progress:
                print(
                    f"[{done}/{total}] engine={engine} level={level}",
                    file=sys.stderr,
                    flush=True,
                )
            worker_out = worker_dir / f"{engine}_{level}.json"
            guarded = memory_guard_status(args, engine, level)
            if guarded is not None:
                results.append(guarded)
                if not args.no_progress:
                    print(
                        f"[skip] {engine} {level}: {guarded['note']}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            status = run_worker(args, engine, level, worker_out)
            if status["status"] == "ok":
                rows.extend(status.pop("rows", []))
            results.append(status)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(out / "photometry_points.csv", index=False)
    summary = {
        "engine_runs": results,
        "pairwise": pairwise_summary(frame) if not frame.empty else {},
        "config": str(args.config),
        "compressed_gas_grid": str(args.compressed_gas_grid or ""),
        "compressed_ssp": str(args.compressed_ssp or ""),
        "compressed_agn_component_grid": str(args.compressed_agn_component_grid or ""),
        "n": int(args.n),
        "seed": int(args.seed),
        "runtime": str(args.runtime),
    }
    (out / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_optional_plots(frame, results, out)
    print(f"wrote {out}")
    nonfatal = {"ok", "skipped_memory_guard"}
    return 0 if all(item["status"] in nonfatal for item in results) else 1


def memory_guard_status(
    args: argparse.Namespace, engine: str, level: str
) -> dict[str, Any] | None:
    if args.no_memory_guard or engine != "dsps_dense_resident":
        return None
    estimate_mib = estimate_dense_resident_payload_mib(args.config, level)
    available_mib = mem_available_mib()
    required_mib = estimate_mib * float(args.memory_overhead_factor)
    allowed_mib = available_mib * float(args.memory_guard_fraction)
    if required_mib <= allowed_mib:
        return None
    return {
        "engine": engine,
        "level": level,
        "status": "skipped_memory_guard",
        "estimated_static_payload_mib": estimate_mib,
        "memory_overhead_factor": float(args.memory_overhead_factor),
        "estimated_required_mib": required_mib,
        "mem_available_mib": available_mib,
        "memory_guard_fraction": float(args.memory_guard_fraction),
        "allowed_mib": allowed_mib,
        "note": (
            f"estimated {required_mib:.0f} MiB required exceeds guard "
            f"{allowed_mib:.0f} MiB; use dsps_dense_lazy or --no-memory-guard "
            "if you intentionally want to risk the resident run"
        ),
    }


def estimate_dense_resident_payload_mib(config_path: str, level: str) -> float:
    config = load_config(config_path)
    model = config.get("model", {}) or {}
    total = hdf5_dataset_float32_mib(config["ssp_path"], "ssp_flux")
    if level in GAS_LEVELS and str(model.get("nebular_model", "fixed_ssp")) == "gas_grid":
        total += hdf5_dataset_float32_mib(model.get("gas_grid_path"), "ssp_flux")
    if (
        level in AGN_LEVELS
        and str(model.get("agn_model", "none")) == "fsps_component_grid"
    ):
        total += hdf5_dataset_float32_mib(
            model.get("agn_component_grid_path"),
            "agn_lnu_per_mformed",
        )
    return total


def hdf5_dataset_float32_mib(path_value: Any, dataset_name: str) -> float:
    if not path_value:
        return 0.0
    path = Path(str(path_value)).expanduser()
    if not path.exists():
        return 0.0
    import h5py

    with h5py.File(path, "r") as handle:
        if dataset_name not in handle:
            return 0.0
        dataset = handle[dataset_name]
        return float(dataset.size * 4) / 1024.0**2


def mem_available_mib() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                return float(parts[1]) / 1024.0
    return float("inf")


def run_worker(
    args: argparse.Namespace,
    engine: str,
    level: str,
    worker_out: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--engine",
        engine,
        "--level",
        level,
        "--config",
        str(args.config),
        "--n",
        str(args.n),
        "--seed",
        str(args.seed),
        "--runtime",
        str(args.runtime),
        "--out",
        str(args.out),
        "--worker-out",
        str(worker_out),
    ]
    if args.stellar_ssp:
        cmd.extend(["--stellar-ssp", str(args.stellar_ssp)])
    if args.compressed_gas_grid:
        cmd.extend(["--compressed-gas-grid", str(args.compressed_gas_grid)])
    if args.compressed_ssp:
        cmd.extend(["--compressed-ssp", str(args.compressed_ssp)])
    if args.compressed_agn_component_grid:
        cmd.extend(
            [
                "--compressed-agn-component-grid",
                str(args.compressed_agn_component_grid),
            ]
        )
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode == 0 and worker_out.exists():
        status = json.loads(worker_out.read_text(encoding="utf-8"))
        status["subprocess_elapsed_s"] = elapsed
        return status
    return {
        "engine": engine,
        "level": level,
        "status": "failed",
        "returncode": int(proc.returncode),
        "subprocess_elapsed_s": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "note": "worker failed or was killed before writing metrics",
    }


def worker_main(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    config = load_config(args.config)
    config = apply_runtime_override(config, args.runtime)
    if args.engine == "dsps_compressed":
        config = apply_compressed_overrides(config, args)
    apply_jax_runtime_env(config.get("runtime", {}))
    rng = np.random.default_rng(int(args.seed))
    points = sample_parameter_points(config, rng, int(args.n))
    rows = evaluate_engine(args, config, points)
    elapsed = time.perf_counter() - start
    status = {
        "engine": args.engine,
        "level": args.level,
        "status": "ok",
        "elapsed_s": elapsed,
        "seconds_per_point": elapsed / max(len(points), 1),
        "peak_rss_mib": peak_rss_mib(),
        "n_points": len(points),
        "n_rows": len(rows),
        "rows": rows,
    }
    Path(args.worker_out).write_text(
        json.dumps(status, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def evaluate_engine(
    args: argparse.Namespace,
    config: dict[str, Any],
    points: list[dict[str, float]],
) -> list[dict[str, Any]]:
    if args.engine == "fsps_prospector":
        return evaluate_fsps(args, config, points)
    return evaluate_dsps(args, config, points)


def evaluate_dsps(
    args: argparse.Namespace,
    config: dict[str, Any],
    points: list[dict[str, float]],
) -> list[dict[str, Any]]:
    from euclid_dsps.model import model_mags_jax, run_dsps_model_jax

    filters = load_filters(config["bands"])
    dense_agn_mode = "lazy" if args.engine == "dsps_dense_lazy" else "resident"
    context = load_level_context(
        config,
        filters,
        args.level,
        args.stellar_ssp,
        dense_agn_mode=dense_agn_mode,
    )
    rows = []
    for point_index, params in enumerate(points):
        level_params, level_model = _parameters_for_level(
            params,
            config["model"],
            args.level,
        )
        eval_context = copy.copy(context)
        use_lazy_agn = (
            args.engine == "dsps_dense_lazy"
            and args.level in AGN_LEVELS
            and str(level_model.get("agn_model", "none")) == "fsps_component_grid"
        )
        eval_context.model_config = (
            model_without_agn(level_model) if use_lazy_agn else level_model
        )
        mags = (
            _model_mags_with_lazy_fsps_component_grid_agn(
                eval_context,
                level_params,
                level_model,
                run_dsps_model_jax,
            )
            if use_lazy_agn
            else np.asarray(model_mags_jax(eval_context, level_params), dtype=float)
        )
        fluxes = abmag_to_fnu_cgs(mags)
        rows.extend(
            rows_for_mags(args.engine, args.level, point_index, config, mags, fluxes)
        )
    return rows


def evaluate_fsps(
    args: argparse.Namespace,
    config: dict[str, Any],
    points: list[dict[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for point_index, params in enumerate(points):
        level_params, level_model = _parameters_for_level(
            params,
            config["model"],
            args.level,
        )
        mags = np.asarray(
            reference_mags_fsps_prospector(config, level_model, level_params),
            dtype=float,
        )
        fluxes = abmag_to_fnu_cgs(mags)
        rows.extend(
            rows_for_mags(args.engine, args.level, point_index, config, mags, fluxes)
        )
    return rows


def rows_for_mags(
    engine: str,
    level: str,
    point_index: int,
    config: dict[str, Any],
    mags: np.ndarray,
    fluxes: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "engine": engine,
            "level": level,
            "point_index": point_index,
            "band": band["name"],
            "mag": float(mag),
            "fnu_cgs": float(flux),
        }
        for band, mag, flux in zip(config["bands"], mags, fluxes, strict=True)
    ]


def pairwise_summary(frame: pd.DataFrame) -> dict[str, Any]:
    wide = frame.pivot_table(
        index=["level", "point_index", "band"],
        columns="engine",
        values=["mag", "fnu_cgs"],
        aggfunc="first",
    )
    engines = sorted(frame["engine"].unique())
    summary: dict[str, Any] = {}
    for reference in engines:
        for candidate in engines:
            if reference == candidate:
                continue
            key = f"{candidate}_minus_{reference}"
            level_summary: dict[str, Any] = {}
            for level in sorted(frame["level"].unique()):
                band_summary: dict[str, Any] = {}
                for band in sorted(frame["band"].unique()):
                    index = (level, slice(None), band)
                    try:
                        ref_mag = wide.loc[index, ("mag", reference)].to_numpy(float)
                        cand_mag = wide.loc[index, ("mag", candidate)].to_numpy(float)
                        ref_flux = wide.loc[index, ("fnu_cgs", reference)].to_numpy(float)
                        cand_flux = wide.loc[index, ("fnu_cgs", candidate)].to_numpy(float)
                    except KeyError:
                        continue
                    delta_mag = cand_mag - ref_mag
                    delta_flux = (cand_flux - ref_flux) / ref_flux
                    finite = np.isfinite(delta_mag)
                    if finite.sum() == 0:
                        continue
                    band_summary[band] = {
                        "n": int(finite.sum()),
                        "median_abs_delta_mag": float(
                            np.nanmedian(np.abs(delta_mag[finite]))
                        ),
                        "p95_abs_delta_mag": float(
                            np.nanpercentile(np.abs(delta_mag[finite]), 95)
                        ),
                        "mean_delta_mag": float(np.nanmean(delta_mag[finite])),
                        "median_abs_delta_flux_over_flux": float(
                            np.nanmedian(np.abs(delta_flux[finite]))
                        ),
                    }
                if band_summary:
                    level_summary[level] = band_summary
            if level_summary:
                summary[key] = level_summary
    return summary


def write_optional_plots(
    frame: pd.DataFrame, results: list[dict[str, Any]], out: Path
) -> None:
    if frame.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = pd.DataFrame(results)
    ok = metrics[metrics["status"] == "ok"].copy()
    if not ok.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = ok["engine"] + "\n" + ok["level"]
        axes[0].bar(labels, ok["seconds_per_point"])
        axes[0].set_ylabel("seconds / point")
        axes[0].tick_params(axis="x", rotation=45)
        axes[1].bar(labels, ok["peak_rss_mib"])
        axes[1].set_ylabel("peak RSS MiB")
        axes[1].tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(out / "engine_time_memory.png", dpi=150)
        plt.close(fig)


def peak_rss_mib() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / 1024**2
    return float(rss) / 1024.0


if __name__ == "__main__":
    raise SystemExit(main())
