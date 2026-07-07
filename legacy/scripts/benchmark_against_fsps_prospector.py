#!/usr/bin/env python3
"""Benchmark DSPS/JAX PopCosmos-like photometry against FSPS/Prospector.

The script deliberately does not use DSPS as a stand-in reference. If
python-fsps or Prospector is unavailable, it exits with an actionable error.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.jax_runtime import apply_jax_runtime_env
from euclid_dsps.photometry import abmag_to_fnu_cgs

_C_LIGHT_KM_PER_S = 299_792.458
_DSPS_OM0 = 0.3075
_DSPS_H = 0.6774
_MIN_AGE_YR = 1.0e5
DEFAULT_STELLAR_ONLY_SSP = "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
_REFERENCE_SPS_CACHE: dict[str, Any] = {}
_REFERENCE_FILTER_CACHE: dict[tuple[tuple[str, str], ...], list[Any]] = {}
_AGN_COMPONENT_AXIS_CACHE: dict[str, dict[str, np.ndarray]] = {}

NOAGN_BENCHMARK_LEVELS = (
    "stellar_only",
    "stellar_plus_dust",
    "stellar_plus_gas",
    "full_noagn",
)
AGN_AUDIT_LEVELS = NOAGN_BENCHMARK_LEVELS + (
    "stellar_plus_agn",
    "stellar_plus_dust_plus_agn",
    "stellar_plus_gas_plus_agn",
    "full_agn",
)
AGN_COMPONENT_LEVELS = ("agn_component_only",)
ALL_BENCHMARK_LEVELS = AGN_AUDIT_LEVELS + AGN_COMPONENT_LEVELS
_AGN_PHOT_LEVELS = {
    "stellar_plus_agn",
    "stellar_plus_dust_plus_agn",
    "stellar_plus_gas_plus_agn",
    "full_agn",
}

CORRELATION_PARAMETERS = (
    "z_obs",
    "tau2",
    "dust_index_n",
    "log10_stellar_metallicity",
    "log10_gas_metallicity",
    "log10_gas_ionization",
    "log10_recent_sfr_msun_per_yr",
)
EFFECTIVE_FAINT_MAG_THRESHOLD = 80.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PopCosmos-like DSPS/JAX photometry against an "
            "FSPS/Prospector reference for sampled config parameters."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/popcosmos_binned_noagn.yaml",
        help="PopCosmos-like no-AGN config to benchmark.",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of random points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--out",
        default="outputs/benchmarks/popcosmos_binned_noagn_fsps",
        help="Output directory.",
    )
    parser.add_argument(
        "--runtime",
        choices=("config", "auto", "cpu", "gpu"),
        default="config",
        help=(
            "JAX runtime override for this benchmark. Use cpu for large "
            "component-grid audits that do not fit on GPU."
        ),
    )
    parser.add_argument(
        "--stellar-ssp",
        default=None,
        help=(
            "Pure-stellar Chabrier SSP used for DSPS stellar_only and "
            "stellar_plus_dust levels. Defaults to model.stellar_only_ssp_path "
            f"or {DEFAULT_STELLAR_ONLY_SSP}."
        ),
    )
    parser.add_argument(
        "--agn-template",
        default=None,
        help=(
            "Override model.agn_template_path for AGN audit levels without "
            "editing the science config."
        ),
    )
    parser.add_argument(
        "--agn-component-grid",
        default=None,
        help=(
            "Use model.agn_model='fsps_component_grid' with this "
            "model.agn_component_grid_path for AGN audit levels."
        ),
    )
    parser.add_argument(
        "--compressed-gas-grid",
        default=None,
        help=(
            "Use model.nebular_model='compressed_gas_grid' with this "
            "model.compressed_gas_grid_path without editing the config."
        ),
    )
    parser.add_argument(
        "--compressed-ssp",
        default=None,
        help=(
            "Use model.ssp_model='compressed_basis' with this "
            "model.compressed_ssp_path without editing the config."
        ),
    )
    parser.add_argument(
        "--compressed-agn-component-grid",
        default=None,
        help=(
            "Use model.agn_model='compressed_fsps_component_grid' with this "
            "model.compressed_agn_component_grid_path without editing the config."
        ),
    )
    parser.add_argument(
        "--agn-host-attenuation",
        choices=("none", "diffuse", "prospector_fsps", "fsps_diffuse_unit_tau"),
        default=None,
        help=(
            "Override model.agn_host_attenuation for AGN audit levels without "
            "editing the science config."
        ),
    )
    parser.add_argument(
        "--agn-host-attenuation-scale",
        type=float,
        default=None,
        help=(
            "Scale factor multiplying the host-dust optical depth applied to "
            "the AGN component for AGN audit levels."
        ),
    )
    parser.add_argument(
        "--agn-igm-order",
        choices=("pre_igm", "fsps_after_igm"),
        default=None,
        help=(
            "Override model.agn_igm_order for AGN audit levels. "
            "Use fsps_after_igm to match FSPS compsp.f90 ordering."
        ),
    )
    parser.add_argument(
        "--agn-baked-attenuation",
        choices=("none", "fsps_powerlaw_unit_tau"),
        default=None,
        help=(
            "Override model.agn_baked_attenuation. Existing FSPS AGN assets "
            "are baked with dust_type=0 unit-tau attenuation."
        ),
    )
    parser.add_argument(
        "--agn-baked-dust-index",
        type=float,
        default=None,
        help="Override model.agn_baked_dust_index for baked FSPS dust_type=0 assets.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=ALL_BENCHMARK_LEVELS,
        default=None,
        help=(
            "Benchmark levels to run. Defaults to no-AGN levels for no-AGN "
            "configs, and adds stellar_plus_agn/full_agn for configs with an "
            "active AGN model."
        ),
    )
    parser.add_argument(
        "--agn-component-wavelength-samples",
        type=int,
        default=256,
        help=(
            "Number of rest-frame wavelength samples written for "
            "agn_component_only."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
        config = _apply_runtime_override(config, args.runtime)
        apply_jax_runtime_env(config.get("runtime", {}))
        _require_reference_packages()
        run_benchmark(args, config)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _require_reference_packages() -> None:
    missing = []
    try:
        import fsps  # noqa: F401
    except Exception:
        missing.append("python-fsps")
    try:
        import prospect  # noqa: F401
    except Exception:
        missing.append("Prospector")
    if missing:
        raise RuntimeError(
            "FSPS/Prospector benchmark requires "
            f"{', '.join(missing)}. Install them in the active environment before "
            "using this benchmark; DSPS is not used as a reference fallback."
        )


def run_benchmark(args: argparse.Namespace, config: dict[str, Any] | None = None) -> None:
    if config is None:
        _log_progress(f"loading config {args.config}")
        config = load_config(args.config)
        config = _apply_runtime_override(config, args.runtime)
        apply_jax_runtime_env(config.get("runtime", {}))
    agn_override_count = sum(
        bool(value)
        for value in (
            args.agn_template,
            args.agn_component_grid,
            args.compressed_agn_component_grid,
        )
    )
    if agn_override_count > 1:
        raise RuntimeError(
            "--agn-template, --agn-component-grid, and "
            "--compressed-agn-component-grid are mutually exclusive"
        )
    if args.compressed_gas_grid:
        config = copy.deepcopy(config)
        config["model"]["nebular_model"] = "compressed_gas_grid"
        config["model"]["compressed_gas_grid_path"] = str(args.compressed_gas_grid)
    if args.compressed_ssp:
        config = copy.deepcopy(config)
        config["model"]["ssp_model"] = "compressed_basis"
        config["model"]["compressed_ssp_path"] = str(args.compressed_ssp)
    if args.agn_template:
        config = copy.deepcopy(config)
        config["model"]["agn_model"] = "template_grid"
        config["model"]["agn_template_path"] = str(args.agn_template)
    if args.agn_component_grid:
        config = copy.deepcopy(config)
        config["model"]["agn_model"] = "fsps_component_grid"
        config["model"]["agn_component_grid_path"] = str(args.agn_component_grid)
    if args.compressed_agn_component_grid:
        config = copy.deepcopy(config)
        config["model"]["agn_model"] = "compressed_fsps_component_grid"
        config["model"]["compressed_agn_component_grid_path"] = str(
            args.compressed_agn_component_grid
        )
    if args.agn_host_attenuation:
        config = copy.deepcopy(config)
        config["model"]["agn_host_attenuation"] = str(args.agn_host_attenuation)
    if args.agn_host_attenuation_scale is not None:
        if float(args.agn_host_attenuation_scale) < 0.0:
            raise RuntimeError("--agn-host-attenuation-scale must be >= 0")
        config = copy.deepcopy(config)
        config["model"]["agn_host_attenuation_scale"] = float(
            args.agn_host_attenuation_scale
        )
    if args.agn_igm_order:
        config = copy.deepcopy(config)
        config["model"]["agn_igm_order"] = str(args.agn_igm_order)
    if args.agn_baked_attenuation:
        config = copy.deepcopy(config)
        config["model"]["agn_baked_attenuation"] = str(args.agn_baked_attenuation)
    if args.agn_baked_dust_index is not None:
        config = copy.deepcopy(config)
        config["model"]["agn_baked_dust_index"] = float(args.agn_baked_dust_index)

    from euclid_dsps.model import (
        load_context,
        model_mags_jax,
        run_dsps_model_jax,
    )

    levels = _benchmark_levels(config["model"], args.levels)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _log_progress(f"sampling {int(args.n)} parameter points")
    rng = np.random.default_rng(int(args.seed))
    points = sample_parameter_points(config, rng, int(args.n))
    _log_progress("loading filters")
    filters = load_filters(config["bands"])
    gas_context = None
    if _needs_gas_context(levels):
        _log_progress(
            "loading DSPS gas context from "
            f"{config['ssp_path']} and {config['model'].get('gas_grid_path')}"
        )
        gas_context_model = config.get("model")
        if _uses_lazy_fsps_component_grid(config["model"]):
            gas_context_model = _model_config_without_agn(config["model"])
        gas_context = load_context(
            config["ssp_path"],
            filters,
            n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
            cosmos_config=config.get("cosmos_sed"),
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=gas_context_model,
        )
    stellar_ssp_path = _stellar_ssp_path(config, args.stellar_ssp)
    stellar_context = None
    if _needs_stellar_context(levels):
        stellar_model_config = _stellar_context_model_config(config["model"])
        if _uses_lazy_fsps_component_grid(config["model"]):
            stellar_model_config = _model_config_without_agn(stellar_model_config)
        _log_progress(f"loading DSPS stellar-only context from {stellar_ssp_path}")
        stellar_context = load_context(
            stellar_ssp_path,
            filters,
            n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
            cosmos_config=config.get("cosmos_sed"),
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=stellar_model_config,
        )
    _log_progress("starting FSPS/Prospector comparison loop")

    rows = []
    for point_index, params in enumerate(points):
        diagnostics = _recent_sfr_diagnostics(params)
        for level in levels:
            print(
                f"benchmark point {point_index + 1}/{len(points)} level {level}",
                file=sys.stderr,
                flush=True,
            )
            level_params, level_model = _parameters_for_level(params, config["model"], level)
            level_context = copy.copy(_context_for_level(level, gas_context, stellar_context))
            if level == "agn_component_only":
                if _uses_lazy_fsps_component_grid(level_model):
                    level_context.model_config = _model_config_without_agn(level_model)
                    base_result = run_dsps_model_jax(level_context, level_params)
                    dsps_wave = np.asarray(base_result.wave, dtype=float)
                    dsps_agn = _lazy_fsps_component_grid_agn_sed(
                        level_context,
                        base_result,
                        level_params,
                        level_model,
                    )
                else:
                    level_context.model_config = level_model
                    dsps_result = run_dsps_model_jax(level_context, level_params)
                    dsps_wave = np.asarray(dsps_result.wave, dtype=float)
                    dsps_agn = np.asarray(dsps_result.agn_sed, dtype=float)
                component_wave = _component_wavelength_grid(
                    dsps_wave, int(args.agn_component_wavelength_samples)
                )
                reference_wave, reference_agn = reference_agn_component_fsps_prospector(
                    config,
                    level_model,
                    level_params,
                )
                dsps_component = np.interp(component_wave, dsps_wave, dsps_agn)
                reference_component = np.interp(
                    component_wave, reference_wave, reference_agn
                )
                for wave_angstrom, dsps_lnu, reference_lnu in zip(
                    component_wave,
                    dsps_component,
                    reference_component,
                    strict=True,
                ):
                    rows.append(
                        _agn_component_row(
                            point_index,
                            level,
                            float(wave_angstrom),
                            float(dsps_lnu),
                            float(reference_lnu),
                            diagnostics,
                            params,
                        )
                )
                continue
            if _uses_lazy_fsps_component_grid(level_model) and level in _AGN_PHOT_LEVELS:
                level_context.model_config = _model_config_without_agn(level_model)
                dsps_mags = _model_mags_with_lazy_fsps_component_grid_agn(
                    level_context,
                    level_params,
                    level_model,
                    run_dsps_model_jax,
                )
            else:
                level_context.model_config = level_model
                dsps_mags = np.asarray(
                    model_mags_jax(level_context, level_params), dtype=float
                )
            reference_mags = reference_mags_fsps_prospector(
                config,
                level_model,
                level_params,
            )
            dsps_flux = abmag_to_fnu_cgs(dsps_mags)
            ref_flux = abmag_to_fnu_cgs(reference_mags)
            for band, dsps_mag, ref_mag, dsps_fnu, ref_fnu in zip(
                config["bands"], dsps_mags, reference_mags, dsps_flux, ref_flux, strict=True
            ):
                rows.append(
                    {
                        "point_index": point_index,
                        "level": level,
                        "observable_type": "photometry",
                        "band": band["name"],
                        "rest_wave_angstrom": np.nan,
                        "dsps_mag": float(dsps_mag),
                        "reference_mag": float(ref_mag),
                        "delta_mag": float(dsps_mag - ref_mag),
                        "delta_flux_over_flux": float((dsps_fnu - ref_fnu) / ref_fnu),
                        "dsps_agn_lnu": np.nan,
                        "reference_agn_lnu": np.nan,
                        "delta_log10_lnu": np.nan,
                        **diagnostics,
                        **{name: float(params[name]) for name in params},
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "benchmark_points.csv", index=False)
    summary = summarize_benchmark(frame)
    summary["criteria"] = {
        "broadbands_median_abs_delta_mag": 0.02,
        "broadbands_p95_abs_delta_mag": 0.05,
        "requires_no_monotone_bias": True,
    }
    summary["reference"] = {
        "engine": "prospect.sources.FastStepBasis + python-fsps + sedpy",
        "imf_type": 1,
        "imf_name": "chabrier",
        "sfh": "PopCosmos lookback bins mapped to Prospector agebins",
        "dust": "FSPS dust_type=4 for dust_model=prospector_fsps",
        "igm": "FSPS native IGM when config igm_model is not none",
        "agn": _agn_reference_note(config["model"], levels),
        "cosmology": {
            "Om0": _DSPS_OM0,
            "h": _DSPS_H,
            "note": "Luminosity distance and age are computed with a local flat-LambdaCDM integral matching the DSPS default parameters.",
        },
        "level_note": (
            "stellar_only and stellar_plus_dust use the pure-stellar DSPS SSP "
            f"{stellar_ssp_path}; stellar_plus_gas and full_noagn use the "
            "configured Chabrier gas grid. agn_component_only compares the "
            "pre-IGM DSPS AGN component to the FSPS/Prospector finite-difference "
            "AGN spectrum on rest-frame wavelength samples."
        ),
        "dsps_stellar_only_ssp": stellar_ssp_path,
        "dsps_base_ssp": str(config["ssp_path"]),
    }
    (out / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_optional_plots(frame, out)


def _stellar_ssp_path(config: dict[str, Any], cli_path: str | None) -> str:
    if cli_path:
        return str(cli_path)
    return str(config["model"].get("stellar_only_ssp_path") or DEFAULT_STELLAR_ONLY_SSP)


def _apply_runtime_override(config: dict[str, Any], runtime: str) -> dict[str, Any]:
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
            "expected_gpu_name": "NVIDIA",
        },
    }
    runtime_config = dict(config.get("runtime") or {})
    runtime_config.update(presets[runtime])
    config["runtime"] = runtime_config
    return config


def _log_progress(message: str) -> None:
    print(f"[benchmark] {message}", file=sys.stderr, flush=True)


def _component_wavelength_grid(wave: np.ndarray, n_samples: int) -> np.ndarray:
    wave = np.asarray(wave, dtype=float)
    finite = wave[np.isfinite(wave) & (wave > 0.0)]
    if finite.size == 0:
        raise RuntimeError("DSPS wavelength grid is empty for agn_component_only")
    n = max(2, min(int(n_samples), finite.size))
    indices = np.unique(np.linspace(0, finite.size - 1, n, dtype=int))
    return finite[indices]


def _agn_component_row(
    point_index: int,
    level: str,
    wave_angstrom: float,
    dsps_lnu: float,
    reference_lnu: float,
    diagnostics: dict[str, float],
    params: dict[str, float],
) -> dict[str, Any]:
    finite_positive = (
        np.isfinite(dsps_lnu)
        and np.isfinite(reference_lnu)
        and dsps_lnu > 0.0
        and reference_lnu > 0.0
    )
    if finite_positive:
        delta_mag = -2.5 * math.log10(dsps_lnu / reference_lnu)
        delta_log10_lnu = math.log10(dsps_lnu) - math.log10(reference_lnu)
        delta_flux_over_flux = (dsps_lnu - reference_lnu) / reference_lnu
    else:
        delta_mag = np.nan
        delta_log10_lnu = np.nan
        delta_flux_over_flux = np.nan
    return {
        "point_index": point_index,
        "level": level,
        "observable_type": "agn_component_sed",
        "band": f"rest_{wave_angstrom:.1f}A",
        "rest_wave_angstrom": wave_angstrom,
        "dsps_mag": np.nan,
        "reference_mag": np.nan,
        "delta_mag": float(delta_mag),
        "delta_flux_over_flux": float(delta_flux_over_flux),
        "dsps_agn_lnu": float(dsps_lnu),
        "reference_agn_lnu": float(reference_lnu),
        "delta_log10_lnu": float(delta_log10_lnu),
        **diagnostics,
        **{name: float(params[name]) for name in params},
    }


def _model_mags_with_lazy_fsps_component_grid_agn(
    context: Any,
    params: dict[str, float],
    agn_model_config: dict[str, Any],
    run_dsps_model_jax: Any,
) -> np.ndarray:
    """Evaluate AGN photometry without loading the full component grid into JAX."""
    import jax.numpy as jnp

    from euclid_dsps.model import combine_agn_and_igm_jax, predict_mags_jax

    base_result = run_dsps_model_jax(context, params)
    agn_sed = _lazy_fsps_component_grid_agn_sed(
        context,
        base_result,
        params,
        agn_model_config,
    )
    wave = jnp.asarray(base_result.wave, dtype=jnp.float32)
    _, post_igm = combine_agn_and_igm_jax(
        wave,
        jnp.asarray(base_result.pre_igm_sed, dtype=jnp.float32),
        jnp.asarray(agn_sed, dtype=jnp.float32),
        jnp.asarray(params["z_obs"], dtype=jnp.float32),
        agn_model_config,
    )
    return np.asarray(predict_mags_jax(context, wave, post_igm, params["z_obs"]), dtype=float)


def _lazy_fsps_component_grid_agn_sed(
    context: Any,
    base_result: Any,
    params: dict[str, float],
    agn_model_config: dict[str, Any],
) -> np.ndarray:
    """Read only the HDF5 slices needed for one AGN component interpolation."""
    import h5py
    import jax.numpy as jnp

    from euclid_dsps.model import (
        apply_agn_host_attenuation_jax,
        log10_stellar_metallicity_to_absolute_jax,
        popcosmos_age_weights_jax,
    )

    path = str(agn_model_config.get("agn_component_grid_path") or "")
    if not path:
        raise RuntimeError("fsps_component_grid requires model.agn_component_grid_path")
    axes = _agn_component_axes(path)
    context_wave = np.asarray(context.ssp_wave_jax, dtype=float)
    _validate_lazy_component_axes(path, axes, context)
    fagn = float(np.exp(float(params["ln_fagn"])))
    tauagn = float(np.exp(float(params["ln_tauagn"])))
    lgmet_abs = float(
        log10_stellar_metallicity_to_absolute_jax(
            params["log10_stellar_metallicity"],
            context.z_sun,
        )
    )
    age_weights = np.asarray(
        popcosmos_age_weights_jax(
            base_result.t_obs_gyr,
            params,
            context.ssp_lg_age_gyr_jax,
        ),
        dtype=np.float32,
    )
    formed_mass = float(np.asarray(base_result.formed_mass_msun, dtype=float))
    component_by_age = np.zeros(
        (len(axes["ssp_lg_age_gyr"]), len(axes["ssp_wave"])),
        dtype=np.float32,
    )
    fagn_pairs = _axis_interp_pairs(axes["fagn_grid"], fagn, "fagn_grid", path)
    tau_pairs = _axis_interp_pairs(axes["agn_tau_grid"], tauagn, "agn_tau_grid", path)
    lgmet_pairs = _axis_interp_pairs(axes["ssp_lgmet"], lgmet_abs, "ssp_lgmet", path)
    with h5py.File(path, "r") as handle:
        grid = handle["agn_lnu_per_mformed"]
        for fagn_index, fagn_weight in fagn_pairs:
            for tau_index, tau_weight in tau_pairs:
                for lgmet_index, lgmet_weight in lgmet_pairs:
                    weight = fagn_weight * tau_weight * lgmet_weight
                    if weight == 0.0:
                        continue
                    component_by_age += weight * np.asarray(
                        grid[fagn_index, tau_index, lgmet_index, :, :],
                        dtype=np.float32,
                    )
    agn_native = np.sum(component_by_age * age_weights[:, None] * formed_mass, axis=0)
    agn_lnu = np.interp(
        context_wave,
        axes["ssp_wave"],
        agn_native,
        left=0.0,
        right=0.0,
    ).astype(np.float32)
    attenuated = apply_agn_host_attenuation_jax(
        jnp.asarray(context_wave, dtype=jnp.float32),
        jnp.asarray(agn_lnu, dtype=jnp.float32),
        params,
        agn_model_config,
    )
    return np.asarray(attenuated, dtype=np.float32)


def _agn_component_axes(path: str) -> dict[str, np.ndarray]:
    cached = _AGN_COMPONENT_AXIS_CACHE.get(path)
    if cached is not None:
        return cached
    import h5py

    with h5py.File(path, "r") as handle:
        axes = {
            "ssp_wave": np.asarray(handle["ssp_wave"], dtype=float),
            "ssp_lg_age_gyr": np.asarray(handle["ssp_lg_age_gyr"], dtype=float),
            "ssp_lgmet": np.asarray(handle["ssp_lgmet"], dtype=float),
            "fagn_grid": np.asarray(handle["fagn_grid"], dtype=float),
            "agn_tau_grid": np.asarray(handle["agn_tau_grid"], dtype=float),
        }
    _AGN_COMPONENT_AXIS_CACHE[path] = axes
    return axes


def _validate_lazy_component_axes(path: str, axes: dict[str, np.ndarray], context: Any) -> None:
    expected = {
        "ssp_wave": np.asarray(context.ssp_wave_jax, dtype=float),
        "ssp_lg_age_gyr": np.asarray(context.ssp_lg_age_gyr_jax, dtype=float),
        "ssp_lgmet": np.asarray(context.ssp_lgmet_jax, dtype=float),
    }
    for name, expected_axis in expected.items():
        actual = axes[name]
        if actual.shape != expected_axis.shape or not np.allclose(
            actual,
            expected_axis,
            rtol=1.0e-6,
            atol=1.0e-4,
        ):
            raise RuntimeError(
                f"AGN component grid {path} {name} axis does not match the "
                "active DSPS context"
            )


def _axis_interp_pairs(
    axis: np.ndarray,
    value: float,
    axis_name: str,
    path: str,
) -> tuple[tuple[int, float], ...]:
    axis = np.asarray(axis, dtype=float)
    value = float(value)
    if axis.ndim != 1 or axis.size < 1 or not np.all(np.isfinite(axis)):
        raise RuntimeError(f"AGN component grid {path} has invalid {axis_name}")
    tolerance = max(1.0e-8, 1.0e-6 * max(abs(float(axis[0])), abs(float(axis[-1]))))
    if value < float(axis[0]) - tolerance or value > float(axis[-1]) + tolerance:
        raise RuntimeError(
            f"AGN component grid {path} {axis_name} does not cover sampled value "
            f"{value:g}; axis range is [{float(axis[0]):g}, {float(axis[-1]):g}]"
        )
    if axis.size == 1:
        return ((0, 1.0),)
    clipped = float(np.clip(value, axis[0], axis[-1]))
    hi = int(np.searchsorted(axis, clipped, side="right"))
    hi = min(max(hi, 1), axis.size - 1)
    lo = hi - 1
    width = float(axis[hi] - axis[lo])
    weight_hi = 0.0 if width <= 0.0 else (clipped - float(axis[lo])) / width
    weight_hi = float(np.clip(weight_hi, 0.0, 1.0))
    weight_lo = 1.0 - weight_hi
    if hi == lo:
        return ((lo, 1.0),)
    return ((lo, weight_lo), (hi, weight_hi))


def _benchmark_levels(
    model_config: dict[str, Any], requested: list[str] | None
) -> tuple[str, ...]:
    if requested:
        levels = tuple(requested)
    elif _is_active_agn_model(model_config):
        levels = AGN_AUDIT_LEVELS
    else:
        levels = NOAGN_BENCHMARK_LEVELS
    if any(level in _AGN_PHOT_LEVELS or level == "agn_component_only" for level in levels):
        if not _is_active_agn_model(model_config):
            raise RuntimeError(
                "AGN audit levels require model.agn_model='template_grid', "
                "'fsps_component_grid', or 'compressed_fsps_component_grid'."
            )
        _require_agn_level_model(model_config, "AGN audit")
    return levels


def _agn_reference_note(model_config: dict[str, Any], levels: tuple[str, ...]) -> str:
    if not any(level in _AGN_PHOT_LEVELS or level == "agn_component_only" for level in levels):
        return "disabled for all benchmark levels"
    return (
        "AGN audit mode: DSPS uses the configured AGN model while the "
        "reference uses FSPS/Prospector fagn and agn_tau. This checks the "
        "current AGN convention but is not a final "
        "PopCosmos AGN validation."
    )


def _is_active_agn_model(model_config: dict[str, Any]) -> bool:
    return str(model_config.get("agn_model", "none")) in {
        "template_grid",
        "fsps_component_grid",
        "compressed_fsps_component_grid",
    }


def _stellar_context_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    local_model = dict(model_config)
    local_model["nebular_model"] = "fixed_ssp"
    local_model.pop("gas_grid_path", None)
    local_model["emission_line_corrections"] = "none"
    return local_model


def _stellar_only_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    local_model = _stellar_context_model_config(model_config)
    local_model["agn_model"] = "none"
    local_model.pop("agn_template_path", None)
    local_model.pop("agn_component_grid_path", None)
    local_model.pop("compressed_agn_component_grid_path", None)
    return local_model


def _uses_lazy_fsps_component_grid(model_config: dict[str, Any]) -> bool:
    return str(model_config.get("agn_model", "none")) == "fsps_component_grid"


def _model_config_without_agn(model_config: dict[str, Any]) -> dict[str, Any]:
    local_model = dict(model_config)
    local_model["agn_model"] = "none"
    local_model.pop("agn_template_path", None)
    local_model.pop("agn_component_grid_path", None)
    local_model.pop("compressed_agn_component_grid_path", None)
    return local_model


def _needs_stellar_context(levels: tuple[str, ...]) -> bool:
    return any(
        level
        in {
            "stellar_only",
            "stellar_plus_dust",
            "stellar_plus_agn",
            "stellar_plus_dust_plus_agn",
            "agn_component_only",
        }
        for level in levels
    )


def _needs_gas_context(levels: tuple[str, ...]) -> bool:
    return any(
        level in {"stellar_plus_gas", "stellar_plus_gas_plus_agn", "full_noagn", "full_agn"}
        for level in levels
    )


def _context_for_level(level: str, gas_context: Any | None, stellar_context: Any | None) -> Any:
    if level in {
        "stellar_only",
        "stellar_plus_dust",
        "stellar_plus_agn",
        "stellar_plus_dust_plus_agn",
        "agn_component_only",
    }:
        if stellar_context is None:
            raise RuntimeError(f"{level} requires a loaded stellar-only DSPS context")
        return stellar_context
    if level in {"stellar_plus_gas", "stellar_plus_gas_plus_agn", "full_noagn", "full_agn"}:
        if gas_context is None:
            raise RuntimeError(f"{level} requires a loaded gas DSPS context")
        return gas_context
    raise RuntimeError(f"Unsupported benchmark level: {level}")


def sample_parameter_points(
    config: dict[str, Any], rng: np.random.Generator, n_points: int
) -> list[dict[str, float]]:
    free = config["fit"]["free_parameters"]
    fixed = {
        key: float(value)
        for key, value in config["model"]["fixed_parameters"].items()
        if np.isscalar(value)
    }
    points = []
    for _ in range(n_points):
        params = dict(fixed)
        for name, spec in free.items():
            low, high = [float(value) for value in spec["bounds"]]
            params[name] = float(rng.uniform(low, high))
        if "log10_gas_metallicity" in params and "log10_stellar_metallicity" in params:
            params["log10_gas_metallicity"] = max(
                params["log10_gas_metallicity"],
                params["log10_stellar_metallicity"],
            )
        points.append(params)
    return points


def _parameters_for_level(
    params: dict[str, float], model_config: dict[str, Any], level: str
) -> tuple[dict[str, float], dict[str, Any]]:
    local_params = dict(params)
    local_model = dict(model_config)
    if level == "stellar_only":
        local_params["tau2"] = 0.0
        local_params["tau1_over_tau2"] = 0.0
        local_model["nebular_model"] = "fixed_ssp"
    elif level == "stellar_plus_dust":
        local_model["nebular_model"] = "fixed_ssp"
    elif level == "stellar_plus_gas":
        local_params["tau2"] = 0.0
        local_params["tau1_over_tau2"] = 0.0
    elif level == "full_noagn":
        pass
    elif level == "stellar_plus_agn":
        local_params["tau2"] = 0.0
        local_params["tau1_over_tau2"] = 0.0
        local_model["nebular_model"] = "fixed_ssp"
        local_model["agn_host_attenuation"] = "none"
        _require_agn_level_model(local_model, level)
        return local_params, local_model
    elif level == "stellar_plus_dust_plus_agn":
        local_model["nebular_model"] = "fixed_ssp"
        _require_agn_level_model(local_model, level)
        return local_params, local_model
    elif level == "stellar_plus_gas_plus_agn":
        local_params["tau2"] = 0.0
        local_params["tau1_over_tau2"] = 0.0
        local_model["agn_host_attenuation"] = "none"
        _require_agn_level_model(local_model, level)
        return local_params, local_model
    elif level == "agn_component_only":
        local_params["tau2"] = 0.0
        local_params["tau1_over_tau2"] = 0.0
        local_model["nebular_model"] = "fixed_ssp"
        local_model["igm_model"] = "none"
        local_model["agn_host_attenuation"] = "none"
        _require_agn_level_model(local_model, level)
        return local_params, local_model
    elif level == "full_agn":
        _require_agn_level_model(local_model, level)
        return local_params, local_model
    else:
        raise RuntimeError(f"Unsupported benchmark level: {level}")
    local_model["agn_model"] = "none"
    return local_params, local_model


def _require_agn_level_model(model_config: dict[str, Any], level: str) -> None:
    agn_model = str(model_config.get("agn_model", "none"))
    if agn_model == "template_grid":
        if not model_config.get("agn_template_path"):
            raise RuntimeError(f"{level} requires model.agn_template_path")
        return
    if agn_model == "fsps_component_grid":
        if not model_config.get("agn_component_grid_path"):
            raise RuntimeError(f"{level} requires model.agn_component_grid_path")
        return
    if agn_model == "compressed_fsps_component_grid":
        if not model_config.get("compressed_agn_component_grid_path"):
            raise RuntimeError(
                f"{level} requires model.compressed_agn_component_grid_path"
            )
        return
    raise RuntimeError(
        f"{level} requires model.agn_model='template_grid', 'fsps_component_grid', "
        "or 'compressed_fsps_component_grid'"
    )


def _recent_sfr_diagnostics(params: dict[str, float]) -> dict[str, float]:
    zred = float(params["z_obs"])
    t_obs = _cosmic_age_gyr_flat_lcdm(zred)
    agebins, mass_shape = _popcosmos_agebins_and_mass_shape(t_obs, params)
    target_mstar = 10.0 ** float(params["log10_stellar_mass"])
    mass = mass_shape / np.maximum(mass_shape.sum(), 1.0e-30) * target_mstar
    widths_yr = 10.0 ** agebins[:, 1] - 10.0 ** agebins[:, 0]
    sfr = mass / np.maximum(widths_yr, 1.0)
    recent = float(np.clip(sfr[0], 1.0e-300, np.inf))
    old = float(np.clip(sfr[-1], 1.0e-300, np.inf))
    return {
        "recent_sfr_msun_per_yr": recent,
        "log10_recent_sfr_msun_per_yr": float(np.log10(recent)),
        "recent_to_old_sfr_ratio": float(recent / old),
    }


def reference_mags_fsps_prospector(
    config: dict[str, Any],
    model_config: dict[str, Any],
    params: dict[str, float],
) -> np.ndarray:
    """Evaluate an independent FSPS/Prospector photometric reference.

    The reference uses Prospector's ``FastStepBasis`` wrapper around python-FSPS
    with a tabular SFH matching the PopCosmos lookback-bin convention. Photometry
    is integrated with sedpy filters, not DSPS photometry kernels.
    """
    _validate_reference_model_support(model_config, params)
    from prospect.sources import FastStepBasis

    sps = _REFERENCE_SPS_CACHE.get("faststep")
    if sps is None:
        sps = FastStepBasis(zcontinuous=1, imf_type=1, compute_vega_mags=False)
        _REFERENCE_SPS_CACHE["faststep"] = sps

    filters = _sedpy_filters(config["bands"])
    reference_params = _prospector_params(model_config, params)
    _sync_fsps_boolean_params(sps, reference_params)
    _spec, phot_maggies, _mfrac = sps.get_spectrum(
        filters=filters,
        peraa=False,
        **reference_params,
    )
    phot = np.asarray(phot_maggies, dtype=float)
    if phot.shape != (len(config["bands"]),):
        raise RuntimeError(
            "Prospector returned photometry with shape "
            f"{phot.shape}; expected {(len(config['bands']),)}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        mags = -2.5 * np.log10(phot)
    if not np.all(np.isfinite(mags)):
        raise RuntimeError(
            "Prospector reference produced non-finite magnitudes. Check sampled "
            "parameters, filters, and FSPS setup."
        )
    return mags


def reference_agn_component_fsps_prospector(
    config: dict[str, Any],
    model_config: dict[str, Any],
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the FSPS/Prospector AGN finite-difference spectrum.

    The output spectrum is rest-frame ``Lsun/Hz`` after applying Prospector's
    mass normalization, but before redshift, luminosity-distance dimming, IGM,
    filters, or photometric integration.
    """
    _validate_reference_model_support(model_config, params)
    if not _is_active_agn_model(model_config):
        raise RuntimeError("agn_component_only requires an active AGN model")
    from prospect.sources import FastStepBasis

    sps = _REFERENCE_SPS_CACHE.get("faststep")
    if sps is None:
        sps = FastStepBasis(zcontinuous=1, imf_type=1, compute_vega_mags=False)
        _REFERENCE_SPS_CACHE["faststep"] = sps

    reference_params = _prospector_params(model_config, params)
    reference_params["add_igm_absorption"] = False
    reference_params["igm_factor"] = 0.0
    with_agn = dict(reference_params)
    without_agn = dict(reference_params)
    without_agn["fagn"] = 0.0
    without_agn["add_dust_emission"] = False

    _sync_fsps_boolean_params(sps, without_agn)
    base_wave, base_spec_per_mformed, base_mfrac = sps.get_galaxy_spectrum(
        **without_agn
    )
    _sync_fsps_boolean_params(sps, with_agn)
    agn_wave, agn_spec_per_mformed, agn_mfrac = sps.get_galaxy_spectrum(**with_agn)
    wave = np.asarray(base_wave, dtype=float)
    if np.asarray(agn_wave).shape != wave.shape or not np.allclose(agn_wave, wave):
        raise RuntimeError("Prospector AGN component wavelength grid changed")
    base_mass_norm = _prospector_mass_normalization(without_agn, float(base_mfrac))
    agn_mass_norm = _prospector_mass_normalization(with_agn, float(agn_mfrac))
    component = (
        np.asarray(agn_spec_per_mformed, dtype=float) * agn_mass_norm
        - np.asarray(base_spec_per_mformed, dtype=float) * base_mass_norm
    )
    if wave.ndim != 1 or component.ndim != 1 or wave.shape != component.shape:
        raise RuntimeError(
            "Prospector AGN component spectrum has incompatible wave/spec shapes"
        )
    return wave, component


def _prospector_mass_normalization(params: dict[str, Any], mfrac: float) -> float:
    mass = float(np.sum(np.asarray(params.get("mass", 1.0), dtype=float)))
    if np.all(np.asarray(params.get("mass_units", "mformed")) == "mstar"):
        mass /= max(float(mfrac), 1.0e-30)
    return mass


def _validate_reference_model_support(
    model_config: dict[str, Any], params: dict[str, float]
) -> None:
    sfh_model = str(model_config.get("sfh_model", "lognormal"))
    if sfh_model != "popcosmos_bins":
        raise RuntimeError(
            "FSPS/Prospector benchmark reference currently supports "
            "model.sfh_model='popcosmos_bins' only."
        )
    agn_model = str(model_config.get("agn_model", "none"))
    if agn_model not in {
        "none",
        "template_grid",
        "fsps_component_grid",
        "compressed_fsps_component_grid",
    }:
        raise RuntimeError(f"Unsupported benchmark AGN model: {agn_model}")
    if _is_active_agn_model(model_config):
        if "ln_fagn" not in params or "ln_tauagn" not in params:
            raise RuntimeError(
                "AGN benchmark levels require sampled ln_fagn and ln_tauagn"
            )
    dust_model = str(model_config.get("dust_model", "prospector_fsps"))
    if dust_model not in {"prospector_fsps", "charlot_fall_powerlaw"}:
        raise RuntimeError(f"Unsupported benchmark dust model: {dust_model}")
    dust_requested = (
        float(params.get("tau2", 0.0)) > 0.0
        or float(params.get("tau1_over_tau2", 0.0)) > 0.0
    )
    if dust_model == "charlot_fall_powerlaw" and dust_requested:
        raise RuntimeError(
            "The independent FSPS/Prospector reference cannot reproduce "
            "dust_model='charlot_fall_powerlaw'. Use dust_model='prospector_fsps' "
            "for dust benchmark levels."
        )
    if str(model_config.get("emission_line_corrections", "none")) != "none":
        raise RuntimeError(
            "FSPS/Prospector reference does not yet apply repository emission-line "
            "correction tables. Use emission_line_corrections: none."
        )
    if str(model_config.get("igm_model", "none")) not in {
        "none",
        "madau95_approx",
        "fsps_madau95",
    }:
        raise RuntimeError(f"Unsupported benchmark IGM model: {model_config['igm_model']}")


def _prospector_params(
    model_config: dict[str, Any], params: dict[str, float]
) -> dict[str, Any]:
    zred = float(params["z_obs"])
    t_obs = _cosmic_age_gyr_flat_lcdm(zred)
    agebins, mass_shape = _popcosmos_agebins_and_mass_shape(t_obs, params)
    target_mstar = 10.0 ** float(params["log10_stellar_mass"])
    mass = mass_shape / np.maximum(mass_shape.sum(), 1.0e-30) * target_mstar
    uses_gas = str(model_config.get("nebular_model", "fixed_ssp")) in {
        "gas_grid",
        "compressed_gas_grid",
    }
    uses_igm = str(model_config.get("igm_model", "none")) != "none"
    uses_agn = _is_active_agn_model(model_config)
    dust_model = str(model_config.get("dust_model", "prospector_fsps"))
    uses_dust = dust_model == "prospector_fsps" and (
        float(params.get("tau2", 0.0)) > 0.0
        or float(params.get("tau1_over_tau2", 0.0)) > 0.0
    )
    tau2 = max(float(params.get("tau2", 0.0)), 0.0) if uses_dust else 0.0
    tau1 = max(float(params.get("tau1_over_tau2", 0.0)), 0.0) * tau2

    return {
        "zred": zred,
        "lumdist": _luminosity_distance_mpc_flat_lcdm(zred),
        "mass": mass,
        "mass_units": "mstar",
        "agebins": agebins,
        "imf_type": 1,
        "logzsol": float(params["log10_stellar_metallicity"]),
        "dust_type": 4 if uses_dust else 0,
        "dust2": tau2,
        "dust1": tau1,
        "dust_index": float(params.get("dust_index_n", -0.7)),
        "dust1_index": float(model_config.get("dust1_index", -1.0)),
        "dust_tesc": float(model_config.get("dust_tesc_logyr", 7.0)),
        "add_dust_emission": uses_agn,
        "add_neb_emission": uses_gas,
        "add_neb_continuum": uses_gas,
        "nebemlineinspec": uses_gas,
        "gas_logz": float(params.get("log10_gas_metallicity", 0.0)),
        "gas_logu": float(params.get("log10_gas_ionization", -2.0)),
        "add_igm_absorption": uses_igm,
        "igm_factor": 1.0 if uses_igm else 0.0,
        "fagn": float(np.exp(float(params["ln_fagn"]))) if uses_agn else 0.0,
        "agn_tau": float(np.exp(float(params["ln_tauagn"]))) if uses_agn else 10.0,
    }


def _sync_fsps_boolean_params(sps: Any, params: dict[str, Any]) -> None:
    """Set FSPS booleans before Prospector update casts numpy scalars."""
    for name in (
        "add_neb_emission",
        "add_neb_continuum",
        "nebemlineinspec",
        "add_igm_absorption",
        "add_dust_emission",
    ):
        if name in params and name in sps.ssp.params.all_params:
            sps.ssp.params[name] = bool(params[name])


def _sedpy_filters(bands: list[dict[str, Any]]) -> list[Any]:
    from sedpy.observate import Filter

    key = tuple(
        (str(band["name"]), json.dumps(band.get("filter", {}), sort_keys=True))
        for band in bands
    )
    cached = _REFERENCE_FILTER_CACHE.get(key)
    if cached is not None:
        return cached
    filters = load_filters(bands)
    sedpy_filters = []
    for band in bands:
        curve = filters[band["name"]]
        wave, transmission = _pad_filter_edges(curve.wave, curve.transmission)
        sedpy_filters.append(
            Filter(kname=str(band["name"]), data=(wave, transmission))
        )
    _REFERENCE_FILTER_CACHE[key] = sedpy_filters
    return sedpy_filters


def _pad_filter_edges(
    wave: np.ndarray, transmission: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(wave, dtype=float)
    transmission = np.asarray(transmission, dtype=float)
    if wave.ndim != 1 or transmission.ndim != 1 or wave.size != transmission.size:
        raise RuntimeError("Filter wave/transmission arrays must be matching 1D arrays")
    order = np.argsort(wave)
    wave = wave[order]
    transmission = np.clip(transmission[order], 0.0, np.inf)
    finite = np.isfinite(wave) & np.isfinite(transmission)
    wave = wave[finite]
    transmission = transmission[finite]
    if wave.size < 2 or np.nanmax(transmission) <= 0.0:
        raise RuntimeError("Benchmark filters must have at least two positive samples")
    width = float(wave[-1] - wave[0])
    pad = max(width * 1.0e-3, 1.0)
    return (
        np.concatenate([[wave[0] - pad], wave, [wave[-1] + pad]]),
        np.concatenate([[0.0], transmission, [0.0]]),
    )


def _popcosmos_agebins_and_mass_shape(
    t_obs_gyr: float, params: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    edges_gyr = _popcosmos_lookback_edges_gyr(t_obs_gyr)
    dlogsfr = np.asarray(
        [float(params[f"dlog10_sfr_{index}"]) for index in range(1, 7)],
        dtype=float,
    )
    sfr_shape = _logsfr_ratios_to_sfr_bins(dlogsfr)
    low_yr = edges_gyr[:-1] * 1.0e9
    high_yr = edges_gyr[1:] * 1.0e9
    low_yr[0] = max(low_yr[0], _MIN_AGE_YR)
    high_yr = np.maximum(high_yr, low_yr + 1.01e6)
    agebins = np.column_stack([np.log10(low_yr), np.log10(high_yr)])
    mass_shape = sfr_shape * np.maximum(high_yr - low_yr, 1.0)
    return agebins, np.clip(mass_shape, 1.0e-30, np.inf)


def _popcosmos_lookback_edges_gyr(t_obs_gyr: float) -> np.ndarray:
    t_safe = max(float(t_obs_gyr), 1.0e-5)
    if t_safe > 0.13:
        log_edges = np.logspace(np.log10(0.10), np.log10(max(0.85 * t_safe, 0.100001)), 5)
        edges = np.concatenate([[0.0, 0.03], log_edges, [t_safe]])
    else:
        edges = t_safe * np.asarray([0.0, 0.03, 0.10, 0.20, 0.35, 0.55, 0.85, 1.0])
    return np.maximum.accumulate(edges)


def _logsfr_ratios_to_sfr_bins(dlog10_sfr: np.ndarray) -> np.ndarray:
    older = np.cumprod(10.0 ** (-np.asarray(dlog10_sfr, dtype=float)))
    return np.clip(np.concatenate([[1.0], older]), 1.0e-30, 1.0e30)


def _cosmic_age_gyr_flat_lcdm(zred: float) -> float:
    a_obs = 1.0 / (1.0 + max(float(zred), 0.0))
    a_grid = np.linspace(1.0e-6, a_obs, 4096)
    ez = np.sqrt(_DSPS_OM0 / a_grid**3 + (1.0 - _DSPS_OM0))
    h0_inv_gyr = 9.778131 / _DSPS_H
    return float(h0_inv_gyr * np.trapezoid(1.0 / (a_grid * ez), a_grid))


def _luminosity_distance_mpc_flat_lcdm(zred: float) -> float:
    z = max(float(zred), 0.0)
    if z == 0.0:
        return 1.0e-5
    z_grid = np.linspace(0.0, z, 4096)
    ez = np.sqrt(_DSPS_OM0 * (1.0 + z_grid) ** 3 + (1.0 - _DSPS_OM0))
    dc = (_C_LIGHT_KM_PER_S / (100.0 * _DSPS_H)) * np.trapezoid(1.0 / ez, z_grid)
    return float((1.0 + z) * dc)


def summarize_benchmark(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"levels": {}}
    for (level, band), group in frame.groupby(["level", "band"], sort=True):
        observable_type = str(group.get("observable_type", pd.Series(["photometry"])).iloc[0])
        if observable_type == "agn_component_sed":
            dsps = group["dsps_agn_lnu"].to_numpy(dtype=float)
            reference = group["reference_agn_lnu"].to_numpy(dtype=float)
        else:
            dsps = group["dsps_mag"].to_numpy(dtype=float)
            reference = group["reference_mag"].to_numpy(dtype=float)
        delta = group["delta_mag"].to_numpy(dtype=float)
        finite_dsps = np.isfinite(dsps)
        finite_reference = np.isfinite(reference)
        finite_delta = np.isfinite(delta)
        finite_both = finite_dsps & finite_reference & finite_delta
        finite_delta_values = delta[finite_both]
        if observable_type == "agn_component_sed":
            effectively_faint_dsps = np.zeros_like(finite_dsps, dtype=bool)
            effectively_faint_reference = np.zeros_like(finite_reference, dtype=bool)
        else:
            effectively_faint_dsps = finite_dsps & (dsps >= EFFECTIVE_FAINT_MAG_THRESHOLD)
            effectively_faint_reference = finite_reference & (
                reference >= EFFECTIVE_FAINT_MAG_THRESHOLD
            )
        bright_finite_both = (
            finite_both
            & ~effectively_faint_dsps
            & ~effectively_faint_reference
        )
        bright_delta_values = delta[bright_finite_both]
        level_summary = summary["levels"].setdefault(level, {})
        level_summary[band] = {
            "n_total": int(len(group)),
            "n_finite_both": int(finite_both.sum()),
            "n_nonfinite_dsps": int((~finite_dsps).sum()),
            "n_nonfinite_reference": int((~finite_reference).sum()),
            "n_nonfinite_delta": int((~finite_delta).sum()),
            "effective_faint_mag_threshold": EFFECTIVE_FAINT_MAG_THRESHOLD,
            "n_effectively_faint_dsps": int(effectively_faint_dsps.sum()),
            "n_effectively_faint_reference": int(effectively_faint_reference.sum()),
            "n_bright_finite_both": int(bright_finite_both.sum()),
            "median_abs_delta_mag": _nan_safe_stat(
                finite_delta_values, lambda values: np.nanmedian(np.abs(values))
            ),
            "p95_abs_delta_mag": _nan_safe_stat(
                finite_delta_values, lambda values: np.nanpercentile(np.abs(values), 95)
            ),
            "mean_delta_mag": _nan_safe_stat(
                finite_delta_values, lambda values: np.nanmean(values)
            ),
            "bright_median_abs_delta_mag": _nan_safe_stat(
                bright_delta_values, lambda values: np.nanmedian(np.abs(values))
            ),
            "bright_p95_abs_delta_mag": _nan_safe_stat(
                bright_delta_values,
                lambda values: np.nanpercentile(np.abs(values), 95),
            ),
            "bright_mean_delta_mag": _nan_safe_stat(
                bright_delta_values, lambda values: np.nanmean(values)
            ),
        }
    summary["residual_correlations"] = {
        name: _correlation(frame, name) for name in CORRELATION_PARAMETERS if name in frame
    }
    summary["residual_correlations_by_level_band"] = {}
    for (level, band), group in frame.groupby(["level", "band"], sort=True):
        by_level = summary["residual_correlations_by_level_band"].setdefault(level, {})
        by_level[band] = {
            name: _correlation(group, name)
            for name in CORRELATION_PARAMETERS
            if name in group
        }
    if "observable_type" in frame and (frame["observable_type"] == "agn_component_sed").any():
        component = frame[frame["observable_type"] == "agn_component_sed"]
        delta = component["delta_mag"].to_numpy(dtype=float)
        finite = np.isfinite(delta)
        summary["agn_component_only"] = {
            "n_total": int(len(component)),
            "n_finite_ratio": int(finite.sum()),
            "median_abs_delta_mag_equivalent": _nan_safe_stat(
                delta[finite], lambda values: np.nanmedian(np.abs(values))
            ),
            "p95_abs_delta_mag_equivalent": _nan_safe_stat(
                delta[finite], lambda values: np.nanpercentile(np.abs(values), 95)
            ),
            "note": (
                "For agn_component_only, delta_mag is -2.5*log10("
                "DSPS_AGN_lnu / FSPS_AGN_lnu) at matched rest wavelength."
            ),
        }
    return summary


def _nan_safe_stat(values: np.ndarray, fn: Any) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(fn(finite))


def _correlation(frame: pd.DataFrame, name: str) -> float | None:
    x = frame[name].to_numpy(dtype=float)
    y = frame["delta_mag"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.nanstd(x[valid]) == 0.0 or np.nanstd(y[valid]) == 0.0:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _write_optional_plots(frame: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if frame.empty:
        return
    photometry = (
        frame[frame["observable_type"] == "photometry"]
        if "observable_type" in frame
        else frame
    )
    if not photometry.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        photometry.boxplot(column="delta_mag", by="band", ax=ax, rot=45)
        ax.set_title("DSPS - FSPS/Prospector")
        ax.set_ylabel("delta mag")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(out / "delta_mag_by_band.png", dpi=150)
        plt.close(fig)
    _write_diagnostic_plots(frame, out, plt)


def _write_diagnostic_plots(frame: pd.DataFrame, out: Path, plt: Any) -> None:
    diagnostic_dir = out / "diagnostics"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    for level, level_frame in frame.groupby("level", sort=True):
        if (
            "observable_type" in level_frame
            and (level_frame["observable_type"] == "agn_component_sed").any()
        ):
            _write_agn_component_diagnostic_plots(level_frame, diagnostic_dir, plt)
            continue
        bands = sorted(str(value) for value in level_frame["band"].dropna().unique())
        if not bands:
            continue
        ncols = min(3, len(bands))
        nrows = int(math.ceil(len(bands) / ncols))
        for parameter in CORRELATION_PARAMETERS:
            if parameter not in level_frame:
                continue
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(4.0 * ncols, 3.0 * nrows),
                squeeze=False,
                sharey=True,
            )
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, band in zip(axes.flat, bands, strict=False):
                ax.set_visible(True)
                band_frame = level_frame[level_frame["band"] == band]
                x = band_frame[parameter].to_numpy(dtype=float)
                y = band_frame["delta_mag"].to_numpy(dtype=float)
                valid = np.isfinite(x) & np.isfinite(y)
                ax.axhline(0.0, color="0.75", linewidth=0.8)
                ax.scatter(x[valid], y[valid], s=8, alpha=0.45)
                ax.set_title(f"{band} ({int(valid.sum())}/{len(band_frame)})")
                ax.set_xlabel(parameter)
                ax.set_ylabel("Delta mag")
            fig.suptitle(f"{level}: Delta mag vs {parameter}")
            fig.tight_layout()
            fig.savefig(
                diagnostic_dir
                / f"{_safe_filename(level)}_delta_mag_vs_{_safe_filename(parameter)}.png",
                dpi=140,
            )
            plt.close(fig)


def _write_agn_component_diagnostic_plots(
    level_frame: pd.DataFrame, diagnostic_dir: Path, plt: Any
) -> None:
    component = level_frame[level_frame["observable_type"] == "agn_component_sed"]
    if component.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for point_index, group in component.groupby("point_index", sort=True):
        wave = group["rest_wave_angstrom"].to_numpy(dtype=float)
        delta = group["delta_mag"].to_numpy(dtype=float)
        valid = np.isfinite(wave) & np.isfinite(delta)
        if not valid.any():
            continue
        ax.plot(wave[valid], delta[valid], alpha=0.18, linewidth=0.8)
        if point_index >= 60:
            break
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Rest wavelength [Angstrom]")
    ax.set_ylabel("-2.5 log10(DSPS AGN / FSPS AGN)")
    ax.set_title("agn_component_only SED ratio audit")
    fig.tight_layout()
    fig.savefig(diagnostic_dir / "agn_component_only_sed_ratio_by_wavelength.png", dpi=150)
    plt.close(fig)

    for parameter in ("ln_fagn", "ln_tauagn", "z_obs", "tau2"):
        if parameter not in component:
            continue
        grouped = component.groupby("point_index", sort=True).agg(
            parameter=(parameter, "first"),
            median_abs_delta=("delta_mag", lambda values: np.nanmedian(np.abs(values))),
        )
        x = grouped["parameter"].to_numpy(dtype=float)
        y = grouped["median_abs_delta"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x[valid], y[valid], s=14, alpha=0.65)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Median |equivalent delta mag| over wavelength")
        ax.set_title(f"agn_component_only vs {parameter}")
        fig.tight_layout()
        fig.savefig(
            diagnostic_dir
            / f"agn_component_only_median_abs_delta_vs_{_safe_filename(parameter)}.png",
            dpi=150,
        )
        plt.close(fig)


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


if __name__ == "__main__":
    raise SystemExit(main())
