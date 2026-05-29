#!/usr/bin/env python3
"""Generate an FSPS-native AGN component SSP grid for DSPS.

The grid stores the additive AGN component directly in SSP units:

    agn_lnu_per_mformed = FSPS(fagn, agn_tau) - FSPS(fagn=0)

for each ``fagn``, ``agn_tau``, stellar metallicity, SSP age, and wavelength.
DSPS can then convolve this component with the same SFH age weights used for
the stellar SSP, avoiding the older repository-local ``fagn * Lbol * template``
normalization convention.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from fsps_grid_common import (
    DEFAULT_STELLAR_ONLY_SSP,
    POPCOSMOS_Z_SUN,
    FspsGridError,
    assert_wave_matches,
    ensure_output_path,
    fail,
    fsps_metadata,
    progress_bar,
    read_ssp_axes,
    require_fsps,
    write_attrs,
)

DEFAULT_OUTPUT = "Data/popcosmos_chabrier_agn_component_ssp_grid.h5"
DEFAULT_AGN_TAU_GRID = [5.0, 10.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 150.0]
DEFAULT_FAGN_GRID = [
    8.315287e-7,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    2.718282,
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PopCosmos-like Chabrier FSPS AGN component SSP grid. "
            "Requires SPS_HOME and python-fsps unless --validate-only is used."
        )
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HDF5 path.")
    parser.add_argument(
        "--reference-ssp",
        default=DEFAULT_STELLAR_ONLY_SSP,
        help="Reference SSP whose wave/age/metallicity axes must match the grid.",
    )
    parser.add_argument(
        "--fagn-grid",
        type=float,
        nargs="+",
        default=DEFAULT_FAGN_GRID,
        help="FSPS fagn grid.",
    )
    parser.add_argument(
        "--agn-tau-grid",
        type=float,
        nargs="+",
        default=DEFAULT_AGN_TAU_GRID,
        help="FSPS agn_tau grid.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Stored grid dtype. DSPS loads the grid as float32.",
    )
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression for agn_lnu_per_mformed.",
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
        "--skip-reference-axis-check",
        action="store_true",
        help="Skip checking axes against --reference-ssp during validation.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars during generation.",
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
            shape = validate_component_grid(args)
            print(f"validated AGN component grid {args.output}: shape {shape}")
            return 0
        output = generate_component_grid(args)
        args.output = str(output)
        shape = validate_component_grid(args)
        print(f"wrote {output}")
        print(f"validated AGN component grid: shape {shape}")
        return 0
    except FspsGridError as exc:
        return fail(str(exc))


def generate_component_grid(args: argparse.Namespace) -> Path:
    output = ensure_output_path(args.output, args.overwrite)
    axes = read_ssp_axes(args.reference_ssp)
    fagn_grid = _parse_positive_grid(args.fagn_grid, "--fagn-grid")
    tau_grid = _parse_positive_grid(args.agn_tau_grid, "--agn-tau-grid")
    fsps = require_fsps()
    sp = _build_stellar_population(fsps)
    compression, compression_opts = _compression_options(args)

    shape = (
        len(fagn_grid),
        len(tau_grid),
        len(axes["ssp_lgmet"]),
        len(axes["ssp_lg_age_gyr"]),
        len(axes["ssp_wave"]),
    )
    if output.exists():
        output.unlink()

    with h5py.File(output, "w") as handle:
        handle["ssp_wave"] = np.asarray(axes["ssp_wave"], dtype=np.float32)
        handle["ssp_lg_age_gyr"] = np.asarray(
            axes["ssp_lg_age_gyr"], dtype=np.float32
        )
        handle["ssp_lgmet"] = np.asarray(axes["ssp_lgmet"], dtype=np.float32)
        handle["fagn_grid"] = np.asarray(fagn_grid, dtype=np.float32)
        handle["agn_tau_grid"] = np.asarray(tau_grid, dtype=np.float32)
        component_out = handle.create_dataset(
            "agn_lnu_per_mformed",
            shape=shape,
            dtype=np.dtype(args.dtype),
            chunks=(1, 1, 1, shape[3], shape[4]),
            compression=compression,
            compression_opts=compression_opts,
        )
        progress = progress_bar(
            total=shape[0] * shape[1] * shape[2],
            enabled=not args.no_progress,
            desc="FSPS AGN component SSP",
            unit="grid",
        )
        try:
            for met_index in range(shape[2]):
                sp.params["fagn"] = 0.0
                wave, base = sp.get_spectrum(
                    zmet=met_index + 1,
                    tage=0.0,
                    peraa=False,
                )
                assert_wave_matches("FSPS AGN component", wave, axes["ssp_wave"])
                base = np.asarray(base, dtype=np.float64)
                if base.shape != shape[3:]:
                    raise FspsGridError(
                        f"FSPS returned base spectra with shape {base.shape}; "
                        f"expected {shape[3:]}"
                    )
                for fagn_index, fagn in enumerate(fagn_grid):
                    for tau_index, tau in enumerate(tau_grid):
                        sp.params["fagn"] = float(fagn)
                        sp.params["agn_tau"] = float(tau)
                        wave_agn, with_agn = sp.get_spectrum(
                            zmet=met_index + 1,
                            tage=0.0,
                            peraa=False,
                        )
                        assert_wave_matches(
                            "FSPS AGN component", wave_agn, axes["ssp_wave"]
                        )
                        delta = np.asarray(with_agn, dtype=np.float64) - base
                        component_out[fagn_index, tau_index, met_index, :, :] = delta
                        if progress is None:
                            print(
                                "generated "
                                f"fagn={float(fagn):g} "
                                f"agn_tau={float(tau):g} "
                                f"zmet={met_index + 1}",
                                flush=True,
                            )
                        else:
                            progress.set_postfix(
                                fagn=f"{float(fagn):g}",
                                agn_tau=f"{float(tau):g}",
                                zmet=str(met_index + 1),
                            )
                            progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        attrs = fsps_metadata(
            fsps, sp, sys.argv if sys.argv else ["generate_fsps_agn_component_grid.py"]
        )
        baked_dust_index = float(sp.params.get("dust_index", -0.7))
        attrs.update(
            {
                "asset_kind": "popcosmos_chabrier_agn_component_ssp_grid",
                "imf_type": 1,
                "imf_name": "chabrier",
                "z_sun": POPCOSMOS_Z_SUN,
                "dust_type": 0,
                "agn_baked_attenuation": "fsps_powerlaw_unit_tau",
                "agn_baked_dust_index": baked_dust_index,
                "add_neb_emission": 0,
                "add_neb_continuum": 0,
                "add_igm_absorption": 0,
                "add_dust_emission": 1,
                "units_ssp_wave": "Angstrom",
                "units_ssp_lg_age_gyr": "log10(age/Gyr)",
                "units_ssp_lgmet": "log10(absolute stellar metallicity mass fraction)",
                "units_agn_lnu_per_mformed": "Lsun/Hz/Msun formed",
                "component_definition": "FSPS(fagn, agn_tau) - FSPS(fagn=0)",
                "normalization_status": "fsps_native_component",
                "fsps_controls": {
                    "sfh": 0,
                    "imf_type": 1,
                    "imf_name": "chabrier",
                    "zcontinuous": 0,
                    "add_neb_emission": 0,
                    "add_neb_continuum": 0,
                    "add_igm_absorption": 0,
                    "add_dust_emission": 1,
                    "dust_type": 0,
                    "dust_index": baked_dust_index,
                    "peraa": False,
                },
            }
        )
        write_attrs(handle, attrs)
    return output


def validate_component_grid(args: argparse.Namespace) -> tuple[int, ...]:
    path = Path(args.output).expanduser()
    if not path.exists():
        raise FspsGridError(f"AGN component grid not found: {path}")
    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "fagn_grid",
        "agn_tau_grid",
        "agn_lnu_per_mformed",
    )
    with h5py.File(path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"AGN component grid {path} is missing datasets: {', '.join(missing)}"
            )
        wave = np.asarray(handle["ssp_wave"])
        age = np.asarray(handle["ssp_lg_age_gyr"])
        lgmet = np.asarray(handle["ssp_lgmet"])
        fagn = np.asarray(handle["fagn_grid"])
        tau = np.asarray(handle["agn_tau_grid"])
        shape = tuple(handle["agn_lnu_per_mformed"].shape)
        expected = (len(fagn), len(tau), len(lgmet), len(age), len(wave))
        if shape != expected:
            raise FspsGridError(
                "agn_lnu_per_mformed must have shape "
                "(n_fagn, n_agn_tau, n_ssp_lgmet, n_ssp_lg_age_gyr, n_wave)"
            )
        _validate_monotonic_axis("ssp_wave", wave)
        _validate_monotonic_axis("ssp_lg_age_gyr", age)
        _validate_monotonic_axis("ssp_lgmet", lgmet)
        _validate_monotonic_axis("fagn_grid", fagn)
        _validate_monotonic_axis("agn_tau_grid", tau)
        if not args.skip_reference_axis_check:
            reference = read_ssp_axes(args.reference_ssp)
            for key, actual in {
                "ssp_wave": wave,
                "ssp_lg_age_gyr": age,
                "ssp_lgmet": lgmet,
            }.items():
                expected_axis = reference[key]
                if actual.shape != expected_axis.shape or not np.allclose(
                    actual, expected_axis, rtol=0.0, atol=1.0e-4
                ):
                    raise FspsGridError(
                        f"{key} axis does not match reference SSP {args.reference_ssp}"
                    )
    return shape


def _build_stellar_population(fsps: Any) -> Any:
    return fsps.StellarPopulation(
        zcontinuous=0,
        sfh=0,
        imf_type=1,
        add_neb_emission=0,
        add_neb_continuum=0,
        add_igm_absorption=0,
        add_dust_emission=1,
        dust_type=0,
        dust1=0.0,
        dust2=0.0,
        fagn=0.0,
    )


def _parse_positive_grid(values: list[float], name: str) -> np.ndarray:
    grid = np.asarray(values, dtype=np.float32)
    if grid.ndim != 1 or grid.size < 1:
        raise FspsGridError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise FspsGridError(f"{name} values must be positive and finite")
    if grid.size > 1 and not np.all(np.diff(grid) > 0.0):
        raise FspsGridError(f"{name} must be strictly increasing")
    return grid


def _validate_monotonic_axis(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise FspsGridError(f"{name} must be a finite 1D axis")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise FspsGridError(f"{name} must be strictly increasing")


def _compression_options(args: argparse.Namespace) -> tuple[str | None, int | None]:
    if args.compression == "none":
        return None, None
    if args.compression == "gzip":
        return "gzip", int(args.gzip_level)
    return "lzf", None


if __name__ == "__main__":
    raise SystemExit(main())
