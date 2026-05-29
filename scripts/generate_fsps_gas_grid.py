#!/usr/bin/env python3
"""Generate a DSPS-compatible gas SSP grid with python-fsps.

The output HDF5 layout is consumed by euclid_dsps.model._load_gas_ssp_grid:

    ssp_flux[gas_lgmet, gas_lgu, stellar_lgmet, age, wave]

python-fsps is intentionally imported only after argument parsing so
``python scripts/generate_fsps_gas_grid.py --help`` works without FSPS/SPS_HOME.
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
    assert_wave_matches,
    axes_from_reference_or_fsps,
    build_metallicity_plan,
    ensure_output_path,
    fail,
    fsps_metadata,
    parse_grid,
    require_fsps,
    validate_gas_grid_hdf5,
    validate_gas_grid_with_model,
    write_attrs,
)

DEFAULT_OUTPUT = "Data/popcosmos_chabrier_gas_ssp_grid.h5"
DEFAULT_GAS_LGMET_GRID = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
DEFAULT_GAS_LGU_GRID = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a DSPS-compatible HDF5 SSP grid over FSPS gas_logz and "
            "gas_logu. Requires python-fsps and SPS_HOME unless --validate-only "
            "is used."
        )
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HDF5 path.")
    parser.add_argument(
        "--reference-ssp",
        default=DEFAULT_REFERENCE_SSP,
        help=(
            "Existing DSPS SSP HDF5 whose stellar metallicity, age, and wavelength "
            "axes must be matched. If the path is absent, axes are discovered from "
            "the active FSPS build."
        ),
    )
    parser.add_argument(
        "--base-ssp",
        default=DEFAULT_REFERENCE_SSP,
        help="Base SSP HDF5 used for load_context synthetic validation.",
    )
    parser.add_argument(
        "--gas-lgmet-grid",
        type=float,
        nargs="+",
        default=DEFAULT_GAS_LGMET_GRID,
        help="Gas log10(Zgas/Zsun) grid passed to FSPS gas_logz.",
    )
    parser.add_argument(
        "--gas-lgu-grid",
        type=float,
        nargs="+",
        default=DEFAULT_GAS_LGU_GRID,
        help="Gas ionization log10(U) grid passed to FSPS gas_logu.",
    )
    parser.add_argument(
        "--stellar-lgmet-grid",
        type=float,
        nargs="+",
        help=(
            "Optional absolute log10 stellar metallicity grid. Defaults to the "
            "reference SSP ssp_lgmet axis or FSPS zlegend."
        ),
    )
    parser.add_argument(
        "--metallicity-mode",
        choices=["auto", "discrete", "continuous"],
        default="auto",
        help=(
            "How to set stellar metallicity in FSPS. auto/discrete use integer "
            "zmet when the requested grid matches FSPS zlegend; continuous uses "
            "logzsol and requires --fsps-z-sun."
        ),
    )
    parser.add_argument(
        "--fsps-z-sun",
        type=float,
        help="FSPS solar metallicity for --metallicity-mode continuous.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Stored ssp_flux dtype. DSPS loads the grid as float32.",
    )
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression for ssp_flux.",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=4,
        help="gzip compression level when --compression=gzip.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --output HDF5 and exit without importing python-fsps.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar during generation.",
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
            shape = validate_gas_grid_hdf5(args.output, args.reference_ssp)
            if not args.skip_model_validation:
                validate_gas_grid_with_model(
                    args.output, args.base_ssp, skip_run=args.skip_forward_run
                )
            print(f"validated gas grid {args.output}: ssp_flux shape {shape}")
            return 0
        output = generate_gas_grid(args)
        shape = validate_gas_grid_hdf5(output, args.reference_ssp)
        if not args.skip_model_validation:
            validate_gas_grid_with_model(
                output, args.base_ssp, skip_run=args.skip_forward_run
            )
        print(f"wrote {output}")
        print(f"validated gas grid: ssp_flux shape {shape}")
        return 0
    except FspsGridError as exc:
        return fail(str(exc))


def generate_gas_grid(args: argparse.Namespace) -> Path:
    fsps = require_fsps()
    output = ensure_output_path(args.output, args.overwrite)
    gas_lgmet_grid = parse_grid(args.gas_lgmet_grid, "--gas-lgmet-grid", min_size=1)
    gas_lgu_grid = parse_grid(args.gas_lgu_grid, "--gas-lgu-grid", min_size=1)

    probe = _build_stellar_population(fsps, zcontinuous=0)
    axes = axes_from_reference_or_fsps(args.reference_ssp, probe)
    if args.stellar_lgmet_grid:
        axes["ssp_lgmet"] = parse_grid(
            args.stellar_lgmet_grid, "--stellar-lgmet-grid", min_size=1
        )
    metal_plan = build_metallicity_plan(
        probe,
        axes["ssp_lgmet"],
        args.metallicity_mode,
        args.fsps_z_sun,
    )
    sp = _build_stellar_population(
        fsps,
        zcontinuous=0 if metal_plan.mode == "discrete" else 1,
    )

    shape = (
        len(gas_lgmet_grid),
        len(gas_lgu_grid),
        len(axes["ssp_lgmet"]),
        len(axes["ssp_lg_age_gyr"]),
        len(axes["ssp_wave"]),
    )
    compression, compression_opts = _compression_options(args)
    if output.exists():
        output.unlink()

    with h5py.File(output, "w") as handle:
        handle["ssp_wave"] = np.asarray(axes["ssp_wave"], dtype=np.float32)
        handle["ssp_lg_age_gyr"] = np.asarray(axes["ssp_lg_age_gyr"], dtype=np.float32)
        handle["ssp_lgmet"] = np.asarray(axes["ssp_lgmet"], dtype=np.float32)
        handle["gas_lgmet_grid"] = gas_lgmet_grid
        handle["gas_lgu_grid"] = gas_lgu_grid
        flux_out = handle.create_dataset(
            "ssp_flux",
            shape=shape,
            dtype=np.dtype(args.dtype),
            chunks=(1, 1, 1, len(axes["ssp_lg_age_gyr"]), len(axes["ssp_wave"])),
            compression=compression,
            compression_opts=compression_opts,
        )

        progress = _progress_bar(
            total=len(gas_lgmet_grid) * len(gas_lgu_grid) * len(axes["ssp_lgmet"]),
            enabled=not args.no_progress,
            desc="FSPS gas SSP grid",
        )
        try:
            for gas_z_index, gas_lgmet in enumerate(gas_lgmet_grid):
                sp.params["gas_logz"] = float(gas_lgmet)
                for gas_u_index, gas_lgu in enumerate(gas_lgu_grid):
                    sp.params["gas_logu"] = float(gas_lgu)
                    for met_index in range(len(axes["ssp_lgmet"])):
                        wave, spectra = _spectrum_for_stellar_metallicity(
                            sp, metal_plan, met_index
                        )
                        assert_wave_matches("FSPS gas SSP", wave, axes["ssp_wave"])
                        spectra = np.asarray(spectra, dtype=np.dtype(args.dtype))
                        if spectra.shape != shape[3:]:
                            raise FspsGridError(
                                f"FSPS returned spectra with shape {spectra.shape}; "
                                f"expected {shape[3:]} from target age/wave axes"
                            )
                        flux_out[gas_z_index, gas_u_index, met_index, :, :] = np.clip(
                            spectra, 0.0, np.inf
                        )
                        if progress is not None:
                            progress.set_postfix(
                                gas_logz=f"{float(gas_lgmet):.3g}",
                                gas_logu=f"{float(gas_lgu):.3g}",
                                stellar_lgmet=f"{float(axes['ssp_lgmet'][met_index]):.3g}",
                            )
                            progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        attrs = fsps_metadata(fsps, sp, sys.argv if sys.argv else ["generate_fsps_gas_grid.py"])
        attrs.update(
            {
                "asset_kind": "popcosmos_chabrier_gas_ssp_grid",
                "imf_type": 1,
                "imf_name": "chabrier",
                "z_sun": POPCOSMOS_Z_SUN,
                "dust_type": 0,
                "units_ssp_flux": "Lsun/Hz/Msun formed",
                "units_ssp_wave": "Angstrom",
                "units_ssp_lg_age_gyr": "log10(age/Gyr)",
                "units_ssp_lgmet": "log10(absolute stellar metallicity mass fraction)",
                "units_gas_lgmet_grid": "log10(Zgas/Zsun), passed to FSPS gas_logz",
                "units_gas_lgu_grid": "log10 ionization parameter, passed to FSPS gas_logu",
                "gas_grid_axes": {
                    "ssp_wave": "Angstrom",
                    "ssp_lg_age_gyr": "log10(age/Gyr)",
                    "ssp_lgmet": "log10(absolute stellar metallicity mass fraction)",
                    "gas_lgmet_grid": "log10(Zgas/Zsun)",
                    "gas_lgu_grid": "log10 ionization parameter U",
                },
                "fsps_controls": {
                    "sfh": 0,
                    "imf_type": 1,
                    "imf_name": "chabrier",
                    "add_neb_emission": 1,
                    "add_neb_continuum": 1,
                    "peraa": False,
                },
                "stellar_metallicity_mode": metal_plan.mode,
                "stellar_zmet_indices": list(metal_plan.zmet_indices),
                "stellar_logzsol_values": list(metal_plan.logzsol_values),
                "fsps_z_sun_for_continuous_mode": metal_plan.fsps_z_sun,
                "gas_lgmet_grid": gas_lgmet_grid,
                "gas_lgu_grid": gas_lgu_grid,
                "scientific_caveat": (
                    "FSPS nebular emission is most self-consistent when gas_logz "
                    "tracks stellar metallicity. Varying gas_logz independently "
                    "at fixed stellar metallicity can bias non-hydrogenic line "
                    "ratios; python-fsps documents typical 1-15 percent accuracy "
                    "for this mode."
                ),
            }
        )
        write_attrs(handle, attrs)

    return output


def _spectrum_for_stellar_metallicity(
    sp: Any, metal_plan: Any, met_index: int
) -> tuple[np.ndarray, np.ndarray]:
    if metal_plan.mode == "discrete":
        return sp.get_spectrum(
            zmet=metal_plan.zmet_indices[met_index],
            tage=0.0,
            peraa=False,
        )
    sp.params["logzsol"] = metal_plan.logzsol_values[met_index]
    return sp.get_spectrum(tage=0.0, peraa=False)


def _build_stellar_population(fsps: Any, zcontinuous: int) -> Any:
    return fsps.StellarPopulation(
        zcontinuous=zcontinuous,
        sfh=0,
        imf_type=1,
        add_neb_emission=1,
        add_neb_continuum=1,
        add_igm_absorption=0,
        add_dust_emission=0,
        dust_type=0,
        dust1=0.0,
        dust2=0.0,
        fagn=0.0,
    )


def _progress_bar(total: int, enabled: bool, desc: str) -> Any | None:
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        print("tqdm is not installed; continuing without progress bar.", file=sys.stderr)
        return None

    return tqdm(total=total, desc=desc, unit="spectrum")


def _compression_options(args: argparse.Namespace) -> tuple[str | None, int | None]:
    if args.compression == "none":
        return None, None
    if args.compression == "gzip":
        return "gzip", int(args.gzip_level)
    return "lzf", None


if __name__ == "__main__":
    raise SystemExit(main())
