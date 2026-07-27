#!/usr/bin/env python3
"""Generate an optional DSPS-compatible FSPS AGN template grid.

The template is isolated as:

    (spectrum[fagn=fagn_norm] - spectrum[fagn=0]) / (fagn_norm * Lbol_stellar)

with spectra from python-fsps in Lsun/Hz and Lbol_stellar computed by integrating
the fagn=0 Lnu spectrum over frequency. The output template therefore has units
of Lnu per stellar bolometric luminosity and matches the current
euclid_dsps.model.add_agn_component_jax convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from fsps_grid_common import (
    DEFAULT_REFERENCE_SSP,
    POPCOSMOS_Z_SUN,
    FspsGridError,
    ensure_output_path,
    fail,
    fsps_metadata,
    lbol_from_lnu,
    parse_grid,
    progress_bar,
    require_fsps,
    validate_agn_grid_hdf5,
    validate_agn_grid_with_model,
    write_attrs,
)

DEFAULT_OUTPUT = "Data/popcosmos_chabrier_agn_template_grid.h5"
DEFAULT_AGN_TAU_GRID = [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DSPS-compatible HDF5 AGN template grid from FSPS "
            "fagn/agn_tau spectra. Requires python-fsps and SPS_HOME unless "
            "--validate-only is used."
        )
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HDF5 path.")
    parser.add_argument(
        "--base-ssp",
        default=DEFAULT_REFERENCE_SSP,
        help="Base SSP HDF5 used for load_context synthetic validation.",
    )
    parser.add_argument(
        "--agn-tau-grid",
        type=float,
        nargs="+",
        default=DEFAULT_AGN_TAU_GRID,
        help="AGN torus optical-depth grid passed to FSPS agn_tau.",
    )
    parser.add_argument(
        "--fagn-normalization",
        type=float,
        default=1.0,
        help="FSPS fagn value used for finite-difference AGN isolation.",
    )
    parser.add_argument(
        "--fagn-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional FSPS fagn grid for audit templates. When set, the output "
            "template is 5D: fagn x agn_tau x tage_gyr x stellar_logzsol x wave."
        ),
    )
    parser.add_argument(
        "--tage-gyr",
        type=float,
        default=1.0,
        help="SSP age used to define the stellar bolometric normalization.",
    )
    parser.add_argument(
        "--tage-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional SSP-age grid in Gyr for audit templates. When set, the "
            "output template is 5D."
        ),
    )
    parser.add_argument(
        "--stellar-logzsol",
        type=float,
        default=0.0,
        help="Stellar log(Z/Zsun) used for the normalization spectrum.",
    )
    parser.add_argument(
        "--stellar-logzsol-grid",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional stellar log(Z/Zsun) grid for audit templates. When set, "
            "the output template is 5D."
        ),
    )
    parser.add_argument(
        "--signed-delta",
        action="store_true",
        help=(
            "Store signed FSPS finite differences instead of clipping negative "
            "AGN deltas to zero. Recommended for audit templates."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Stored template dtype. DSPS loads the grid as float32.",
    )
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression for template_lnu_per_lbol.",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=4,
        help="gzip compression level when --compression=gzip.",
    )
    parser.add_argument(
        "--require-exact-normalization",
        action="store_true",
        help=(
            "Fail instead of writing an approximate template. Exact FSPS CLUMPY "
            "bolometric normalization has not been independently audited here."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --output HDF5 and exit without importing python-fsps.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during generation and validation.",
    )
    parser.add_argument(
        "--skip-model-validation",
        action="store_true",
        help="Skip load_context/run_dsps_model_jax synthetic validation.",
    )
    parser.add_argument(
        "--skip-forward-run",
        action="store_true",
        help="Load the grid into a context but skip the JAX forward model run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            shape = validate_agn_grid(args)
            print(f"validated AGN grid {args.output}: template shape {shape}")
            return 0
        output = generate_agn_grid(args)
        args.output = str(output)
        shape = validate_agn_grid(args)
        print(f"wrote {output}")
        print(f"validated AGN grid: template shape {shape}")
        return 0
    except FspsGridError as exc:
        return fail(str(exc))


def generate_agn_grid(args: argparse.Namespace) -> Path:
    if args.require_exact_normalization:
        raise FspsGridError(
            "Exact FSPS/CLUMPY bolometric AGN normalization is not independently "
            "audited in this repository. Rerun without --require-exact-normalization "
            "to write an asset explicitly marked approximate."
        )
    if not np.isfinite(args.fagn_normalization) or args.fagn_normalization <= 0.0:
        raise FspsGridError("--fagn-normalization must be positive and finite")
    if not np.isfinite(args.tage_gyr) or args.tage_gyr <= 0.0:
        raise FspsGridError("--tage-gyr must be positive and finite")

    output = ensure_output_path(args.output, args.overwrite)
    agn_tau_grid = parse_grid(args.agn_tau_grid, "--agn-tau-grid", min_size=1)
    fagn_grid = parse_grid(
        args.fagn_grid if args.fagn_grid is not None else [args.fagn_normalization],
        "--fagn-grid",
        min_size=1,
    )
    tage_grid = parse_grid(
        args.tage_grid if args.tage_grid is not None else [args.tage_gyr],
        "--tage-grid",
        min_size=1,
    )
    if np.any(tage_grid <= 0.0):
        raise FspsGridError("--tage-grid values must be positive")
    stellar_logzsol_grid = parse_grid(
        (
            args.stellar_logzsol_grid
            if args.stellar_logzsol_grid is not None
            else [args.stellar_logzsol]
        ),
        "--stellar-logzsol-grid",
        min_size=1,
    )
    audit_grid = (
        args.signed_delta
        or args.fagn_grid is not None
        or args.tage_grid is not None
        or args.stellar_logzsol_grid is not None
    )
    progress = progress_bar(
        total=int(
            len(fagn_grid)
            * len(agn_tau_grid)
            * len(tage_grid)
            * len(stellar_logzsol_grid)
        )
        + 3,
        enabled=not args.no_progress,
        desc="FSPS AGN grid",
        unit="step",
    )
    try:
        if progress is not None:
            progress.set_postfix(stage="import_fsps")
        fsps = require_fsps()
        if progress is not None:
            progress.update(1)

        if progress is not None:
            progress.set_postfix(stage="initialize")
        sp = fsps.StellarPopulation(
            zcontinuous=1,
            sfh=0,
            imf_type=1,
            add_neb_emission=0,
            add_igm_absorption=0,
            add_dust_emission=1,
            dust_type=0,
            dust1=0.0,
            dust2=0.0,
            logzsol=float(stellar_logzsol_grid[0]),
        )
        sp.params["fagn"] = 0.0
        sp.params["logzsol"] = float(stellar_logzsol_grid[0])
        if progress is not None:
            progress.update(1)

        if progress is not None:
            progress.set_postfix(stage="discover_wave")
        wave, stellar = sp.get_spectrum(tage=float(tage_grid[0]), peraa=False)
        wave = np.asarray(wave, dtype=np.float64)
        stellar = np.asarray(stellar, dtype=np.float64)
        if stellar.ndim != 1:
            raise FspsGridError(
                "FSPS returned an unexpected non-1D spectrum for AGN normalization"
            )
        if progress is not None:
            progress.update(1)

        template_shape = (
            (
                len(fagn_grid),
                len(agn_tau_grid),
                len(tage_grid),
                len(stellar_logzsol_grid),
                len(wave),
            )
            if audit_grid
            else (len(agn_tau_grid), len(wave))
        )
        template = np.zeros(template_shape, dtype=np.dtype(args.dtype))
        lbol_grid = np.zeros(
            (len(tage_grid), len(stellar_logzsol_grid)), dtype=np.float64
        )
        for age_index, tage_gyr in enumerate(tage_grid):
            for logz_index, stellar_logzsol in enumerate(stellar_logzsol_grid):
                sp.params["fagn"] = 0.0
                sp.params["logzsol"] = float(stellar_logzsol)
                wave_base, stellar = sp.get_spectrum(tage=float(tage_gyr), peraa=False)
                if np.asarray(wave_base).shape != wave.shape or not np.allclose(
                    wave_base, wave
                ):
                    raise FspsGridError(
                        "FSPS AGN wavelength grid changed between evaluations"
                    )
                stellar = np.asarray(stellar, dtype=np.float64)
                lbol_stellar = lbol_from_lnu(wave, stellar)
                if not np.isfinite(lbol_stellar) or lbol_stellar <= 0.0:
                    raise FspsGridError(
                        "Could not compute a positive stellar bolometric luminosity"
                    )
                lbol_grid[age_index, logz_index] = lbol_stellar
                for fagn_index, fagn_normalization in enumerate(fagn_grid):
                    for tau_index, agn_tau in enumerate(agn_tau_grid):
                        if progress is not None:
                            progress.set_postfix(
                                stage="template",
                                fagn=f"{float(fagn_normalization):g}",
                                agn_tau=f"{float(agn_tau):g}",
                                tage=f"{float(tage_gyr):g}",
                                logz=f"{float(stellar_logzsol):g}",
                            )
                        sp.params["agn_tau"] = float(agn_tau)
                        sp.params["fagn"] = float(fagn_normalization)
                        wave_agn, with_agn = sp.get_spectrum(
                            tage=float(tage_gyr), peraa=False
                        )
                        if np.asarray(wave_agn).shape != wave.shape or not np.allclose(
                            wave_agn, wave
                        ):
                            raise FspsGridError(
                                "FSPS AGN wavelength grid changed between evaluations"
                            )
                        delta = np.asarray(with_agn, dtype=np.float64) - stellar
                        normalized = delta / (float(fagn_normalization) * lbol_stellar)
                        if not args.signed_delta:
                            normalized = np.clip(normalized, 0.0, np.inf)
                        if audit_grid:
                            template[
                                fagn_index, tau_index, age_index, logz_index, :
                            ] = normalized
                        else:
                            template[tau_index, :] = normalized
                        if progress is None:
                            print(
                                "generated "
                                f"fagn={float(fagn_normalization):g} "
                                f"agn_tau={float(agn_tau):g} "
                                f"tage={float(tage_gyr):g} "
                                f"logzsol={float(stellar_logzsol):g}",
                                flush=True,
                            )
                        else:
                            progress.update(1)
                        sp.params["fagn"] = 0.0

        if progress is not None:
            progress.set_postfix(stage="write_hdf5")
        _write_agn_grid(
            output,
            args,
            fsps,
            sp,
            fagn_grid,
            agn_tau_grid,
            tage_grid,
            stellar_logzsol_grid,
            wave,
            template,
            lbol_grid,
            audit_grid,
        )
        if progress is not None:
            progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return output


def validate_agn_grid(args: argparse.Namespace) -> tuple[int, ...]:
    total = 1
    if not args.skip_model_validation:
        total += 1 if args.skip_forward_run else 2
    progress = progress_bar(
        total=total,
        enabled=not args.no_progress,
        desc="AGN grid validation",
        unit="step",
    )
    try:
        if progress is not None:
            progress.set_postfix(stage="hdf5")
        shape = validate_agn_grid_hdf5(args.output)
        if progress is not None:
            progress.update(1)
        if not args.skip_model_validation:
            validate_agn_grid_with_model(
                args.output,
                args.base_ssp,
                skip_run=args.skip_forward_run,
                progress=progress,
            )
        return shape
    finally:
        if progress is not None:
            progress.close()


def _write_agn_grid(
    output: Path,
    args: argparse.Namespace,
    fsps: Any,
    sp: Any,
    fagn_grid: np.ndarray,
    agn_tau_grid: np.ndarray,
    tage_grid: np.ndarray,
    stellar_logzsol_grid: np.ndarray,
    wave: np.ndarray,
    template: np.ndarray,
    lbol_grid: np.ndarray,
    audit_grid: bool,
) -> None:
    compression, compression_opts = _compression_options(args)
    baked_dust_index = float(sp.params.get("dust_index", -0.7))
    if output.exists():
        output.unlink()
    with h5py.File(output, "w") as handle:
        handle["wave"] = np.asarray(wave, dtype=np.float32)
        if audit_grid:
            handle["fagn_grid"] = np.asarray(fagn_grid, dtype=np.float32)
            handle["fagn_normalization_grid"] = np.asarray(fagn_grid, dtype=np.float32)
            handle["tage_gyr_grid"] = np.asarray(tage_grid, dtype=np.float32)
            handle["stellar_logzsol_grid"] = np.asarray(
                stellar_logzsol_grid, dtype=np.float32
            )
            handle["stellar_lbol_lsun_grid"] = np.asarray(lbol_grid, dtype=np.float32)
        handle["agn_tau_grid"] = agn_tau_grid
        chunks = (1, 1, 1, 1, len(wave)) if audit_grid else (1, len(wave))
        handle.create_dataset(
            "template_lnu_per_lbol",
            data=template,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
        )
        attrs = fsps_metadata(
            fsps, sp, sys.argv if sys.argv else ["generate_fsps_agn_grid.py"]
        )
        attrs.update(
            {
                "asset_kind": (
                    "popcosmos_chabrier_agn_fspsdiff_audit_grid"
                    if audit_grid
                    else "popcosmos_chabrier_agn_template_grid"
                ),
                "imf_type": 1,
                "imf_name": "chabrier",
                "z_sun": POPCOSMOS_Z_SUN,
                "dust_type": 0,
                "agn_baked_attenuation": "fsps_powerlaw_unit_tau",
                "agn_baked_dust_index": baked_dust_index,
                "fsps_controls": {
                    "sfh": 0,
                    "imf_type": 1,
                    "imf_name": "chabrier",
                    "add_neb_emission": 0,
                    "add_dust_emission": 1,
                    "dust_type": 0,
                    "dust_index": baked_dust_index,
                    "peraa": False,
                },
                "units_wave": "Angstrom",
                "units_template_lnu_per_lbol": "Lsun/Hz per Lsun bolometric",
                "agn_tau_grid": agn_tau_grid,
                "fagn_normalization": float(fagn_grid[0]),
                "fagn_grid": fagn_grid,
                "tage_gyr": float(tage_grid[0]),
                "tage_gyr_grid": tage_grid,
                "stellar_logzsol": float(stellar_logzsol_grid[0]),
                "stellar_logzsol_grid": stellar_logzsol_grid,
                "stellar_lbol_lsun": float(lbol_grid[0, 0]),
                "signed_delta": bool(args.signed_delta),
                "template_shape": tuple(int(value) for value in template.shape),
                "normalization_convention": (
                    "template = (FSPS spectrum with fagn=fagn_normalization - "
                    "FSPS spectrum with fagn=0) / "
                    "(fagn_normalization * integrated stellar Lbol)"
                ),
                "normalization_status": "audit" if audit_grid else "approximate",
                "scientific_caveat": (
                    "FSPS documents fagn as AGN luminosity divided by stellar "
                    "bolometric luminosity, but this asset uses a repository-local "
                    "finite wavelength integral to match add_agn_component_jax. "
                    "Treat as an optional additive CLUMPY-derived template until "
                    "normalization is independently audited."
                ),
            }
        )
        write_attrs(handle, attrs)


def _compression_options(args: argparse.Namespace) -> tuple[str | None, int | None]:
    if args.compression == "none":
        return None, None
    if args.compression == "gzip":
        return "gzip", int(args.gzip_level)
    return "lzf", None


if __name__ == "__main__":
    raise SystemExit(main())
