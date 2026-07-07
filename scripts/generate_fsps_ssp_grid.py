#!/usr/bin/env python3
"""Generate a DSPS-compatible PopCosmos Chabrier reference SSP with python-fsps.

The output HDF5 layout matches ``dsps.load_ssp_templates``:

    ssp_flux[stellar_lgmet, age, wave]

python-fsps is imported only after argument parsing so
``python scripts/generate_fsps_ssp_grid.py --help`` works without FSPS/SPS_HOME.
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
    DEFAULT_STELLAR_ONLY_SSP,
    POPCOSMOS_Z_SUN,
    FspsGridError,
    assert_wave_matches,
    ensure_output_path,
    fail,
    fsps_metadata,
    progress_bar,
    require_fsps,
    validate_ssp_grid_hdf5,
    write_attrs,
)

DEFAULT_OUTPUT = DEFAULT_REFERENCE_SSP
DEFAULT_LINE_NAME_SOURCE = DEFAULT_REFERENCE_SSP


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the PopCosmos-like Chabrier base SSP HDF5 from python-fsps. "
            "Requires SPS_HOME and python-fsps unless --validate-only is used."
        )
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output HDF5 path. Defaults to the fixed-nebular PopCosmos-like SSP, "
            "or to Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5 when "
            "--stellar-only is set."
        ),
    )
    parser.add_argument(
        "--stellar-only",
        action="store_true",
        help=(
            "Generate a pure-stellar Chabrier SSP with FSPS nebular emission and "
            "nebular continuum disabled. This is the correct DSPS asset for "
            "benchmark stellar_only and stellar_plus_dust levels."
        ),
    )
    parser.add_argument(
        "--gas-logu",
        type=float,
        default=-2.0,
        help="Fixed FSPS gas_logu used for the reference nebular SSP.",
    )
    parser.add_argument(
        "--gas-logz",
        type=float,
        default=0.0,
        help="Fixed FSPS gas_logz used for the reference nebular SSP.",
    )
    parser.add_argument(
        "--reference-line-names",
        default=DEFAULT_LINE_NAME_SOURCE,
        help=(
            "Optional existing SSP HDF5 whose ssp_emline_name dataset is reused. "
            "If absent, generic line_### names are written."
        ),
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
        help="HDF5 compression for SSP flux datasets.",
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
        help="Disable tqdm progress bars during generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = DEFAULT_STELLAR_ONLY_SSP if args.stellar_only else DEFAULT_OUTPUT
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            shape = validate_ssp_grid_hdf5(args.output)
            print(f"validated SSP grid {args.output}: ssp_flux shape {shape}")
            return 0
        output = generate_ssp_grid(args)
        shape = validate_ssp_grid_hdf5(output)
        print(f"wrote {output}")
        print(f"validated SSP grid: ssp_flux shape {shape}")
        return 0
    except FspsGridError as exc:
        return fail(str(exc))


def generate_ssp_grid(args: argparse.Namespace) -> Path:
    output = ensure_output_path(args.output, args.overwrite)
    fsps = require_fsps()
    sp = _build_stellar_population(fsps, args)
    include_nebular = not bool(args.stellar_only)
    axes, n_line = _discover_axes_and_line_count(sp, include_nebular=include_nebular)
    line_names = _line_names(args.reference_line_names, n_line)

    shape = (
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
        flux_out = handle.create_dataset(
            "ssp_flux",
            shape=shape,
            dtype=np.dtype(args.dtype),
            chunks=(1, len(axes["ssp_lg_age_gyr"]), len(axes["ssp_wave"])),
            compression=compression,
            compression_opts=compression_opts,
        )
        surviving_out = handle.create_dataset(
            "ssp_surviving_mstar",
            shape=shape[:2],
            dtype=np.dtype(args.dtype),
            chunks=(1, len(axes["ssp_lg_age_gyr"])),
            compression=compression,
            compression_opts=compression_opts,
        )
        line_out = None
        if n_line:
            handle["ssp_emline_wave"] = np.asarray(
                sp.emline_wavelengths, dtype=np.float32
            )
            handle["ssp_emline_name"] = line_names
            line_out = handle.create_dataset(
                "ssp_emline_luminosity",
                shape=(shape[0], shape[1], n_line),
                dtype=np.dtype(args.dtype),
                chunks=(1, shape[1], n_line),
                compression=compression,
                compression_opts=compression_opts,
            )

        progress = progress_bar(
            total=shape[0],
            enabled=not args.no_progress,
            desc="FSPS Chabrier stellar SSP" if args.stellar_only else "FSPS Chabrier SSP",
            unit="metallicity",
        )
        try:
            for met_index in range(shape[0]):
                wave, spectra = sp.get_spectrum(
                    zmet=met_index + 1,
                    tage=0.0,
                    peraa=False,
                )
                assert_wave_matches("FSPS Chabrier SSP", wave, axes["ssp_wave"])
                spectra = np.asarray(spectra, dtype=np.dtype(args.dtype))
                if spectra.shape != shape[1:]:
                    raise FspsGridError(
                        f"FSPS returned spectra with shape {spectra.shape}; "
                        f"expected {shape[1:]} from target age/wave axes"
                    )
                flux_out[met_index, :, :] = np.clip(spectra, 0.0, np.inf)
                surviving_out[met_index, :] = _surviving_mstar(
                    sp, shape[1], np.dtype(args.dtype)
                )
                if line_out is not None:
                    line_out[met_index, :, :] = _emline_luminosity(
                        sp, shape[1], n_line, np.dtype(args.dtype)
                    )
                if progress is not None:
                    progress.set_postfix(
                        zmet=str(met_index + 1),
                        stellar_lgmet=f"{float(axes['ssp_lgmet'][met_index]):.3g}",
                    )
                    progress.update(1)
        finally:
            if progress is not None:
                progress.close()

        attrs = fsps_metadata(fsps, sp, sys.argv if sys.argv else ["generate_fsps_ssp_grid.py"])
        asset_kind = (
            "popcosmos_chabrier_stellar_only_ssp"
            if args.stellar_only
            else "popcosmos_chabrier_base_ssp"
        )
        nebular_note = (
            "Pure-stellar SSP: FSPS nebular emission and nebular continuum are disabled. "
            "Use this asset for stellar_only and stellar_plus_dust benchmark levels."
            if args.stellar_only
            else (
                "Base SSP includes FSPS nebular emission at fixed gas_logu "
                f"{float(args.gas_logu):g} and gas_logz {float(args.gas_logz):g}. "
                "Production PopCosmos-like gas runs should use the separate "
                "popcosmos_chabrier_gas_ssp_grid.h5."
            )
        )
        attrs.update(
            {
                "asset_kind": asset_kind,
                "imf_type": 1,
                "imf_name": "chabrier",
                "z_sun": POPCOSMOS_Z_SUN,
                "dust_type": 0,
                "add_neb_emission": int(include_nebular),
                "add_neb_continuum": int(include_nebular),
                "gas_logu": float(args.gas_logu) if include_nebular else "",
                "gas_logz": float(args.gas_logz) if include_nebular else "",
                "units_ssp_flux": "Lsun/Hz/Msun formed",
                "units_ssp_wave": "Angstrom",
                "units_ssp_lg_age_gyr": "log10(age/Gyr)",
                "units_ssp_lgmet": "log10(absolute stellar metallicity mass fraction)",
                "units_ssp_surviving_mstar": "Msun surviving stellar mass per Msun formed",
                "units_ssp_emline_luminosity": "Lsun/Msun formed",
                "units_ssp_emline_wave": "Angstrom",
                "fsps_controls": {
                    "sfh": 0,
                    "imf_type": 1,
                    "imf_name": "chabrier",
                    "zcontinuous": 0,
                    "add_neb_emission": int(include_nebular),
                    "add_neb_continuum": int(include_nebular),
                    "gas_logu": float(args.gas_logu) if include_nebular else None,
                    "gas_logz": float(args.gas_logz) if include_nebular else None,
                    "add_igm_absorption": 0,
                    "add_dust_emission": 0,
                    "dust_type": 0,
                    "peraa": False,
                },
                "nebular_reference": nebular_note,
            }
        )
        write_attrs(handle, attrs)

    return output


def _build_stellar_population(fsps: Any, args: argparse.Namespace) -> Any:
    include_nebular = not bool(args.stellar_only)
    sp = fsps.StellarPopulation(
        zcontinuous=0,
        sfh=0,
        imf_type=1,
        add_neb_emission=int(include_nebular),
        add_neb_continuum=int(include_nebular),
        add_igm_absorption=0,
        add_dust_emission=0,
        dust_type=0,
        dust1=0.0,
        dust2=0.0,
        fagn=0.0,
    )
    if include_nebular:
        sp.params["gas_logu"] = float(args.gas_logu)
        sp.params["gas_logz"] = float(args.gas_logz)
    return sp


def _discover_axes_and_line_count(
    sp: Any, *, include_nebular: bool = True
) -> tuple[dict[str, np.ndarray], int]:
    wave, spectra = sp.get_spectrum(zmet=1, tage=0.0, peraa=False)
    spectra = np.asarray(spectra)
    if spectra.ndim != 2:
        raise FspsGridError(
            "FSPS get_spectrum(tage=0.0) did not return an age-by-wavelength grid"
        )
    ssp_ages = getattr(sp, "ssp_ages", None)
    if ssp_ages is None:
        raise FspsGridError("Could not discover the FSPS SSP age grid")
    zlegend = getattr(sp, "zlegend", None)
    if zlegend is None:
        raise FspsGridError("Could not discover the FSPS stellar metallicity grid")
    if len(ssp_ages) != spectra.shape[0]:
        raise FspsGridError(
            "FSPS ssp_ages length does not match the returned SSP spectra grid"
        )
    axes = {
        "ssp_wave": np.asarray(wave, dtype=np.float32),
        "ssp_lg_age_gyr": np.asarray(ssp_ages, dtype=np.float32) - 9.0,
        "ssp_lgmet": np.log10(np.asarray(zlegend, dtype=np.float32)),
    }
    if not include_nebular:
        return axes, 0
    emline_wave = np.asarray(getattr(sp, "emline_wavelengths", []), dtype=np.float32)
    emline_luminosity = np.asarray(getattr(sp, "emline_luminosity", []))
    if emline_wave.size == 0 and emline_luminosity.size == 0:
        return axes, 0
    if emline_luminosity.ndim != 2 or emline_luminosity.shape[0] != spectra.shape[0]:
        raise FspsGridError(
            "FSPS emline_luminosity must have shape (n_age, n_line) after "
            "get_spectrum(tage=0.0)"
        )
    if len(emline_wave) != emline_luminosity.shape[1]:
        raise FspsGridError(
            "FSPS emline_wavelengths length does not match emline_luminosity"
        )
    return axes, int(emline_luminosity.shape[1])


def _emline_luminosity(
    sp: Any, n_age: int, n_line: int, dtype: np.dtype
) -> np.ndarray:
    luminosity = np.asarray(getattr(sp, "emline_luminosity", []), dtype=dtype)
    expected = (n_age, n_line)
    if luminosity.shape != expected:
        raise FspsGridError(
            f"FSPS emline_luminosity shape {luminosity.shape} does not match {expected}"
        )
    return np.clip(luminosity, 0.0, np.inf)


def _surviving_mstar(sp: Any, n_age: int, dtype: np.dtype) -> np.ndarray:
    surviving = np.asarray(getattr(sp, "stellar_mass", []), dtype=dtype)
    if surviving.shape != (n_age,):
        raise FspsGridError(
            f"FSPS stellar_mass shape {surviving.shape} does not match {(n_age,)}"
        )
    return np.clip(surviving, 1.0e-4, np.inf)


def _line_names(reference_line_names: str | Path | None, n_line: int) -> np.ndarray:
    if n_line == 0:
        return np.asarray([], dtype="S16")
    if reference_line_names:
        reference = Path(reference_line_names).expanduser()
        if reference.exists():
            with h5py.File(reference, "r") as handle:
                if "ssp_emline_name" in handle and len(handle["ssp_emline_name"]) == n_line:
                    return np.asarray(handle["ssp_emline_name"][:], dtype="S64")
    return np.asarray(
        [f"line_{index:03d}".encode() for index in range(n_line)], dtype="S16"
    )


def _compression_options(args: argparse.Namespace) -> tuple[str | None, int | None]:
    if args.compression == "none":
        return None, None
    if args.compression == "gzip":
        return "gzip", int(args.gzip_level)
    return "lzf", None


if __name__ == "__main__":
    raise SystemExit(main())
