"""Convert Euclid throughput FITS passbands to DSPS HDF5 format."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fits", nargs="+", help="Input Euclid throughput FITS files.")
    parser.add_argument("--out", default="filters/converted", help="Output directory.")
    parser.add_argument("--wave-column", default="WAVE", help="FITS wavelength column.")
    parser.add_argument("--throughput-column", default="T_TOTAL", help="FITS throughput column.")
    parser.add_argument("--wave-unit", default="nm", choices=["angstrom", "nm", "micron"], help="Input wavelength unit.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.fits:
        src = Path(item)
        dst = out_dir / output_name(src)
        convert_one(
            src,
            dst,
            wave_column=args.wave_column,
            throughput_column=args.throughput_column,
            wave_unit=args.wave_unit,
        )
        print(dst)


def convert_one(src: Path, dst: Path, wave_column: str, throughput_column: str, wave_unit: str) -> None:
    data = fits.getdata(src, 1)
    wave = np.asarray(data[wave_column], dtype=float) * wave_unit_to_angstrom(wave_unit)
    transmission = np.clip(np.asarray(data[throughput_column], dtype=float), 0.0, 1.0)
    mask = np.isfinite(wave) & np.isfinite(transmission)
    order = np.argsort(wave[mask])
    with h5py.File(dst, "w") as hdf:
        hdf.create_dataset("wave", data=wave[mask][order])
        hdf.create_dataset("transmission", data=transmission[mask][order])
        hdf.attrs["source"] = str(src)
        hdf.attrs["wave_unit"] = "Angstrom"
        hdf.attrs["throughput_column"] = throughput_column


def wave_unit_to_angstrom(unit: str) -> float:
    if unit == "angstrom":
        return 1.0
    if unit == "nm":
        return 10.0
    if unit == "micron":
        return 10_000.0
    raise ValueError(f"Unsupported wavelength unit: {unit}")


def output_name(src: Path) -> str:
    stem = src.stem.lower()
    for token, band in (("-y_", "y"), ("-j_", "j"), ("-h_", "h")):
        if token in stem:
            return f"euclid_nisp_{band}_throughput.h5"
    return f"{stem}.h5"


if __name__ == "__main__":
    main()
