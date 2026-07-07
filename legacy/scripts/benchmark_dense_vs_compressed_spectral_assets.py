#!/usr/bin/env python3
"""Benchmark dense versus compressed DSPS/JAX spectral assets.

This script compares two DSPS configurations at identical sampled parameters.
The dense model is treated as the reference; the compressed model should use
`nebular_model: compressed_gas_grid` and/or
`agn_model: compressed_fsps_component_grid`.

The benchmark can be memory-heavy when dense gas or dense AGN levels are
enabled. Use `--runtime cpu` for first science residual checks and use small
`--n` before running large GPU batch-capacity tests.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
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
    _recent_sfr_diagnostics,
    _stellar_context_model_config,
    _stellar_ssp_path,
    _write_optional_plots,
    sample_parameter_points,
    summarize_benchmark,
)
from euclid_dsps.config import load_config  # noqa: E402
from euclid_dsps.filters import load_filters  # noqa: E402
from euclid_dsps.jax_runtime import apply_jax_runtime_env  # noqa: E402
from euclid_dsps.photometry import abmag_to_fnu_cgs  # noqa: E402

DEFAULT_LEVELS = (
    "stellar_plus_gas",
    "full_noagn",
    "stellar_plus_agn",
    "full_agn",
)
SUPPORTED_LEVELS = AGN_AUDIT_LEVELS
AGN_LEVELS = {
    "stellar_plus_agn",
    "stellar_plus_dust_plus_agn",
    "stellar_plus_gas_plus_agn",
    "full_agn",
}
GAS_LEVELS = {
    "stellar_plus_gas",
    "stellar_plus_gas_plus_agn",
    "full_noagn",
    "full_agn",
}
STELLAR_CONTEXT_LEVELS = {
    "stellar_only",
    "stellar_plus_dust",
    "stellar_plus_agn",
    "stellar_plus_dust_plus_agn",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-config",
        default="configs/popcosmos_binned.yaml",
        help="Dense reference config.",
    )
    parser.add_argument(
        "--compressed-config",
        default=None,
        help=(
            "Compressed config. If omitted, the dense config is copied and "
            "--compressed-gas-grid/--compressed-agn-component-grid overrides "
            "are applied."
        ),
    )
    parser.add_argument(
        "--compressed-gas-grid",
        default=None,
        help="Compressed gas asset path to inject into the compressed config.",
    )
    parser.add_argument(
        "--compressed-ssp",
        default=None,
        help="Compressed base SSP asset path to inject into the compressed config.",
    )
    parser.add_argument(
        "--compressed-agn-component-grid",
        default=None,
        help="Compressed AGN component asset path to inject into the compressed config.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=SUPPORTED_LEVELS,
        default=list(DEFAULT_LEVELS),
        help="Benchmark levels.",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of sampled points.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--runtime",
        choices=("config", "auto", "cpu", "gpu"),
        default="cpu",
        help="Runtime override applied to both configs before JAX import.",
    )
    parser.add_argument(
        "--dense-agn-mode",
        choices=("lazy", "resident"),
        default="lazy",
        help=(
            "lazy reads dense AGN HDF5 slices per point instead of loading the "
            "full dense AGN grid into JAX. resident measures the true dense "
            "resident mode and may require much more RAM."
        ),
    )
    parser.add_argument(
        "--stellar-ssp",
        default=None,
        help="Pure-stellar SSP path used for stellar-only levels.",
    )
    parser.add_argument(
        "--out",
        default="outputs/benchmarks/dense_vs_compressed_popcosmos",
        help="Output directory.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dense_config = load_config(args.dense_config)
    compressed_config = (
        load_config(args.compressed_config)
        if args.compressed_config
        else copy.deepcopy(dense_config)
    )
    compressed_config = apply_compressed_overrides(compressed_config, args)
    dense_config = apply_runtime_override(dense_config, args.runtime)
    compressed_config = apply_runtime_override(compressed_config, args.runtime)
    apply_jax_runtime_env(dense_config.get("runtime", {}))
    run_benchmark(args, dense_config, compressed_config)
    return 0


def apply_compressed_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    model = config["model"]
    if args.compressed_gas_grid:
        model["nebular_model"] = "compressed_gas_grid"
        model["compressed_gas_grid_path"] = str(args.compressed_gas_grid)
    if args.compressed_ssp:
        model["ssp_model"] = "compressed_basis"
        model["compressed_ssp_path"] = str(args.compressed_ssp)
    if args.compressed_agn_component_grid:
        model["agn_model"] = "compressed_fsps_component_grid"
        model["compressed_agn_component_grid_path"] = str(
            args.compressed_agn_component_grid
        )
    return config


def apply_runtime_override(config: dict[str, Any], runtime: str) -> dict[str, Any]:
    if runtime == "config":
        return config
    config = copy.deepcopy(config)
    presets = {
        "auto": {
            "jax_platforms": "auto",
            "disable_jax_plugin_autoload": False,
            "xla_python_client_preallocate": False,
            "require_gpu": False,
        },
        "cpu": {
            "jax_platforms": "cpu",
            "disable_jax_plugin_autoload": True,
            "xla_python_client_preallocate": False,
            "require_gpu": False,
        },
        "gpu": {
            "jax_platforms": "cuda",
            "disable_jax_plugin_autoload": False,
            "xla_python_client_preallocate": False,
            "require_gpu": True,
        },
    }
    config["runtime"].update(presets[runtime])
    return config


def run_benchmark(
    args: argparse.Namespace,
    dense_config: dict[str, Any],
    compressed_config: dict[str, Any],
) -> None:
    from euclid_dsps.model import model_mags_jax, run_dsps_model_jax

    levels = tuple(args.levels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    points = sample_parameter_points(dense_config, rng, int(args.n))
    filters = load_filters(dense_config["bands"])

    rows = []
    progress = progress_bar(
        total=len(points) * len(levels),
        enabled=not args.no_progress,
    )
    for level in levels:
        print(f"[dense-vs-compressed] loading level {level}", file=sys.stderr, flush=True)
        dense_context = load_level_context(
            dense_config,
            filters,
            level,
            args.stellar_ssp,
            dense_agn_mode=args.dense_agn_mode,
        )
        compressed_context = load_level_context(
            compressed_config,
            filters,
            level,
            args.stellar_ssp,
            dense_agn_mode="resident",
        )
        try:
            for point_index, params in enumerate(points):
                diagnostics = _recent_sfr_diagnostics(params)
                dense_params, dense_level_model = _parameters_for_level(
                    params,
                    dense_config["model"],
                    level,
                )
                compressed_params, compressed_level_model = _parameters_for_level(
                    params,
                    compressed_config["model"],
                    level,
                )
                dense_eval_context = copy.copy(dense_context)
                compressed_eval_context = copy.copy(compressed_context)
                use_lazy_dense_agn = (
                    args.dense_agn_mode == "lazy"
                    and level in AGN_LEVELS
                    and str(dense_level_model.get("agn_model", "none"))
                    == "fsps_component_grid"
                )
                dense_eval_context.model_config = (
                    model_without_agn(dense_level_model)
                    if use_lazy_dense_agn
                    else dense_level_model
                )
                compressed_eval_context.model_config = compressed_level_model
                dense_mags = (
                    _model_mags_with_lazy_fsps_component_grid_agn(
                        dense_eval_context,
                        dense_params,
                        dense_level_model,
                        run_dsps_model_jax,
                    )
                    if use_lazy_dense_agn
                    else np.asarray(
                        model_mags_jax(dense_eval_context, dense_params),
                        dtype=float,
                    )
                )
                compressed_mags = np.asarray(
                    model_mags_jax(compressed_eval_context, compressed_params),
                    dtype=float,
                )
                dense_flux = abmag_to_fnu_cgs(dense_mags)
                compressed_flux = abmag_to_fnu_cgs(compressed_mags)
                for band, dense_mag, comp_mag, ref_fnu, comp_fnu in zip(
                    dense_config["bands"],
                    dense_mags,
                    compressed_mags,
                    dense_flux,
                    compressed_flux,
                    strict=True,
                ):
                    rows.append(
                        {
                            "point_index": point_index,
                            "level": level,
                            "observable_type": "photometry",
                            "band": band["name"],
                            "rest_wave_angstrom": np.nan,
                            "dsps_mag": float(comp_mag),
                            "reference_mag": float(dense_mag),
                            "compressed_mag": float(comp_mag),
                            "dense_mag": float(dense_mag),
                            "delta_mag": float(comp_mag - dense_mag),
                            "delta_flux_over_flux": float(
                                (comp_fnu - ref_fnu) / ref_fnu
                            ),
                            "dsps_agn_lnu": np.nan,
                            "reference_agn_lnu": np.nan,
                            "delta_log10_lnu": np.nan,
                            **diagnostics,
                            **{name: float(params[name]) for name in params},
                        }
                    )
                update_progress(progress)
        finally:
            del dense_context
            del compressed_context
            clear_runtime_caches()
    close_progress(progress)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "benchmark_points.csv", index=False)
    summary = summarize_benchmark(frame)
    summary["criteria"] = {
        "target_median_abs_delta_mag": 0.005,
        "target_p95_abs_delta_mag": 0.015,
        "prototype_ceiling_p95_abs_delta_mag": 0.03,
    }
    summary["reference"] = {
        "engine": "dense DSPS/JAX spectral assets",
        "comparison": "compressed_mag - dense_mag",
        "dense_config": str(args.dense_config),
        "compressed_config": str(args.compressed_config or args.dense_config),
        "compressed_gas_grid": str(args.compressed_gas_grid or ""),
        "compressed_ssp": str(args.compressed_ssp or ""),
        "compressed_agn_component_grid": str(
            args.compressed_agn_component_grid or ""
        ),
        "dense_agn_mode": str(args.dense_agn_mode),
    }
    (out / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_optional_plots(frame, out)


def load_level_context(
    config: dict[str, Any],
    filters: dict[str, Any],
    level: str,
    stellar_ssp: str | None,
    dense_agn_mode: str = "resident",
) -> Any:
    from euclid_dsps.model import load_context

    if level in STELLAR_CONTEXT_LEVELS:
        model = _stellar_context_model_config(config["model"])
        if level not in AGN_LEVELS or should_load_without_dense_agn(
            model,
            level,
            dense_agn_mode,
        ):
            model = model_without_agn(model)
        return load_context(
            _stellar_ssp_path(config, stellar_ssp),
            filters,
            n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
            cosmos_config=config.get("cosmos_sed"),
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=model,
        )
    if level in GAS_LEVELS:
        model = dict(config["model"])
        if level not in AGN_LEVELS or should_load_without_dense_agn(
            model,
            level,
            dense_agn_mode,
        ):
            model = model_without_agn(model)
        return load_context(
            config["ssp_path"],
            filters,
            n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
            cosmos_config=config.get("cosmos_sed"),
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=model,
        )
    raise RuntimeError(f"Unsupported benchmark level: {level}")


def model_without_agn(model: dict[str, Any]) -> dict[str, Any]:
    local = dict(model)
    local["agn_model"] = "none"
    local.pop("agn_template_path", None)
    local.pop("agn_component_grid_path", None)
    local.pop("compressed_agn_component_grid_path", None)
    return local


def should_load_without_dense_agn(
    model: dict[str, Any], level: str, dense_agn_mode: str
) -> bool:
    return (
        dense_agn_mode == "lazy"
        and level in AGN_LEVELS
        and str(model.get("agn_model", "none")) == "fsps_component_grid"
    )


def progress_bar(total: int, enabled: bool) -> Any:
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="dense-vs-compressed")


def update_progress(progress: Any) -> None:
    if progress is not None:
        progress.update(1)


def close_progress(progress: Any) -> None:
    if progress is not None:
        progress.close()


def clear_runtime_caches() -> None:
    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
