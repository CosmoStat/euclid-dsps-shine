#!/usr/bin/env python3
"""Build a low-rank gas SSP grid consumed directly by JAX.

The input dense grid is:

    ssp_flux[gas_lgmet, gas_lgu, stellar_lgmet, age, wave]

The output compressed grid stores:

    gas_scale[gas_lgmet, gas_lgu, stellar_lgmet, age]
    gas_coeff[gas_lgmet, gas_lgu, stellar_lgmet, age, k]
    gas_basis[k, wave]

This is a linear-flux compression of the current dense gas spectra. It reduces
resident JAX payload size, but it is still a prototype for science use because
line-rich gas spectra should ultimately be represented as continuum plus sparse
line luminosities when enriched assets are available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from fsps_grid_common import FspsGridError, ensure_output_path, fail, write_attrs

DEFAULT_INPUT = "Data/popcosmos_chabrier_gas_ssp_grid.h5"
DEFAULT_OUTPUT = "Data/popcosmos_chabrier_gas_grid_basis_k64.h5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Dense gas grid HDF5.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Compressed HDF5.")
    parser.add_argument("--k", type=int, default=64, help="Number of basis vectors.")
    parser.add_argument(
        "--oversample",
        type=int,
        default=16,
        help="Randomized SVD oversampling dimension.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalization",
        choices=["none", "l2", "max"],
        default="l2",
        help="Per-curve normalization before basis fitting.",
    )
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression for coefficient datasets.",
    )
    parser.add_argument(
        "--basis-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Stored dtype for gas_basis.",
    )
    parser.add_argument(
        "--coeff-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Stored dtype for gas_coeff.",
    )
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --output and exit.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            summary = validate_compressed_grid(args.output)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        output = build_compressed_grid(args)
        summary = validate_compressed_grid(output)
        print(f"wrote {output}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except FspsGridError as exc:
        return fail(str(exc))


def build_compressed_grid(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FspsGridError(f"Dense gas SSP grid not found: {input_path}")
    output = ensure_output_path(args.output, args.overwrite)
    if args.k < 1:
        raise FspsGridError("--k must be positive")
    if args.oversample < 0:
        raise FspsGridError("--oversample must be non-negative")

    compression, compression_opts = compression_options(args)
    rng = np.random.default_rng(int(args.seed))

    with h5py.File(input_path, "r") as src:
        required = (
            "ssp_wave",
            "ssp_lg_age_gyr",
            "ssp_lgmet",
            "gas_lgmet_grid",
            "gas_lgu_grid",
            "ssp_flux",
        )
        missing = [key for key in required if key not in src]
        if missing:
            raise FspsGridError(
                f"Dense gas grid {input_path} is missing datasets: {', '.join(missing)}"
            )
        dense = src["ssp_flux"]
        if dense.ndim != 5:
            raise FspsGridError(
                "ssp_flux must have shape "
                "(gas_lgmet, gas_lgu, stellar_lgmet, age, wave)"
            )
        ngas_z, ngas_u, nmet, nage, nwave = dense.shape
        if args.k > nwave:
            raise FspsGridError("--k cannot exceed the wavelength dimension")
        n_curves = ngas_z * ngas_u * nmet * nage
        rank = min(args.k + args.oversample, nwave, n_curves)

        scale = compute_curve_scale(dense, args.normalization)
        omega = rng.normal(size=(nwave, rank)).astype(np.float32)
        y = np.empty((n_curves, rank), dtype=np.float32)
        flat_scale = scale.reshape(-1)
        for start, stop, block in iter_dense_curve_blocks(dense):
            block_norm = normalize_block(block, flat_scale[start:stop])
            y[start:stop] = block_norm @ omega
        q, _ = np.linalg.qr(y, mode="reduced")
        q = q.astype(np.float32, copy=False)

        b = np.zeros((rank, nwave), dtype=np.float32)
        for start, stop, block in iter_dense_curve_blocks(dense):
            block_norm = normalize_block(block, flat_scale[start:stop])
            b += q[start:stop].T @ block_norm
        _u_hat, _singular_values, vt = np.linalg.svd(b, full_matrices=False)
        basis = vt[: args.k].astype(np.float32, copy=False)
        basis_dtype = np.dtype(getattr(args, "basis_dtype", "float32"))
        coeff_dtype = np.dtype(getattr(args, "coeff_dtype", "float32"))

        if output.exists():
            output.unlink()
        with h5py.File(output, "w") as dst:
            for key in (
                "ssp_wave",
                "ssp_lg_age_gyr",
                "ssp_lgmet",
                "gas_lgmet_grid",
                "gas_lgu_grid",
            ):
                dst[key] = np.asarray(src[key], dtype=np.float32)
            dst.create_dataset(
                "gas_basis",
                data=basis.astype(basis_dtype),
                compression=compression,
                compression_opts=compression_opts,
            )
            coeff_out = dst.create_dataset(
                "gas_coeff",
                shape=(ngas_z, ngas_u, nmet, nage, args.k),
                dtype=coeff_dtype,
                chunks=(1, 1, 1, nage, args.k),
                compression=compression,
                compression_opts=compression_opts,
            )
            dst.create_dataset(
                "gas_scale",
                data=scale.reshape(ngas_z, ngas_u, nmet, nage).astype(np.float32),
                chunks=(1, 1, 1, nage),
                compression=compression,
                compression_opts=compression_opts,
            )
            for start, stop, block in iter_dense_curve_blocks(dense):
                block_norm = normalize_block(block, flat_scale[start:stop])
                coeff = (block_norm @ basis.T).astype(coeff_dtype, copy=False)
                write_coeff_block(coeff_out, start, stop, coeff)
            attrs = {key: decode_attr(value) for key, value in src.attrs.items()}
            attrs.update(
                {
                    "asset_kind": "popcosmos_chabrier_compressed_gas_grid",
                    "compression_kind": "randomized_linear_svd",
                    "compression_version": 1,
                    "basis_space": "linear_flux",
                    "k_basis": int(args.k),
                    "normalization": str(args.normalization),
                    "compressed_dtypes": {
                        "gas_basis": str(basis_dtype),
                        "gas_coeff": str(coeff_dtype),
                        "gas_scale": "float32",
                    },
                    "line_handling": "line_rich_full_spectrum_basis_prototype",
                    "source_grid_path": str(input_path),
                    "source_grid_size_bytes": int(input_path.stat().st_size),
                    "source_grid_mtime_ns": int(input_path.stat().st_mtime_ns),
                    "dense_main_dataset": "ssp_flux",
                    "units_gas_basis": "same normalized units as ssp_flux",
                    "units_gas_coeff": "dimensionless projection coefficient",
                    "units_gas_scale": attrs.get(
                        "units_ssp_flux", "Lsun/Hz/Msun formed"
                    ),
                    "scientific_status": (
                        "prototype_full_spectrum_basis; benchmark against dense "
                        "grid before science use"
                    ),
                    "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "generation_command": " ".join(sys.argv),
                }
            )
            write_attrs(dst, attrs)
    return output


def compute_curve_scale(dense: h5py.Dataset, normalization: str) -> np.ndarray:
    shape = dense.shape
    scale = np.ones(shape[:-1], dtype=np.float32)
    if normalization == "none":
        return scale
    flat = scale.reshape(-1)
    for start, stop, block in iter_dense_curve_blocks(dense):
        if normalization == "l2":
            values = np.sqrt(
                np.mean(np.asarray(block, dtype=np.float64) ** 2, axis=1)
            )
        elif normalization == "max":
            values = np.max(np.abs(block), axis=1)
        else:  # pragma: no cover - argparse constrains this
            raise FspsGridError(f"Unsupported normalization: {normalization}")
        values = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(values) & (values > 0.0)
        flat[start:stop] = np.where(finite, values, 1.0).astype(np.float32)
    return scale


def normalize_block(block: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32)
    return np.asarray(block, dtype=np.float32) / np.maximum(scale[:, None], 1.0e-30)


def iter_dense_curve_blocks(dense: h5py.Dataset):
    ngas_z, ngas_u, nmet, nage, nwave = dense.shape
    for i in range(ngas_z):
        for j in range(ngas_u):
            for k in range(nmet):
                start = (((i * ngas_u) + j) * nmet + k) * nage
                stop = start + nage
                block = np.asarray(dense[i, j, k, :, :], dtype=np.float32).reshape(
                    nage, nwave
                )
                yield start, stop, block


def write_coeff_block(
    coeff_out: h5py.Dataset, start: int, stop: int, coeff: np.ndarray
) -> None:
    ngas_z, ngas_u, nmet, nage, _nbasis = coeff_out.shape
    first = start // nage
    last = (stop - 1) // nage
    if first != last:
        raise FspsGridError("Coefficient block crosses natural HDF5 block boundary")
    i = first // (ngas_u * nmet)
    rem = first % (ngas_u * nmet)
    j = rem // nmet
    k = rem % nmet
    coeff_out[i, j, k, :, :] = coeff


def validate_compressed_grid(path: str | Path) -> dict[str, Any]:
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FspsGridError(f"Compressed gas SSP grid not found: {grid_path}")
    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "gas_lgmet_grid",
        "gas_lgu_grid",
        "gas_basis",
        "gas_coeff",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"Compressed gas grid {grid_path} is missing datasets: "
                f"{', '.join(missing)}"
            )
        wave = handle["ssp_wave"]
        age = handle["ssp_lg_age_gyr"]
        lgmet = handle["ssp_lgmet"]
        gas_lgmet = handle["gas_lgmet_grid"]
        gas_lgu = handle["gas_lgu_grid"]
        basis = handle["gas_basis"]
        coeff = handle["gas_coeff"]
        expected_coeff = (
            len(gas_lgmet),
            len(gas_lgu),
            len(lgmet),
            len(age),
            basis.shape[0],
        )
        if basis.ndim != 2 or basis.shape[1] != len(wave):
            raise FspsGridError("gas_basis must have shape (n_basis, n_wave)")
        if coeff.ndim != 5 or coeff.shape != expected_coeff:
            raise FspsGridError(
                "gas_coeff must have shape "
                "(n_gas_lgmet, n_gas_lgu, n_ssp_lgmet, n_ssp_lg_age_gyr, n_basis)"
            )
        if "gas_scale" in handle and handle["gas_scale"].shape != expected_coeff[:-1]:
            raise FspsGridError(
                "gas_scale must have shape "
                "(n_gas_lgmet, n_gas_lgu, n_ssp_lgmet, n_ssp_lg_age_gyr)"
            )
        attrs = {key: decode_attr(value) for key, value in handle.attrs.items()}
        return {
            "path": str(grid_path),
            "asset_kind": attrs.get("asset_kind", ""),
            "compression_kind": attrs.get("compression_kind", ""),
            "k_basis": int(basis.shape[0]),
            "n_wave": int(len(wave)),
            "n_age": int(len(age)),
            "n_ssp_lgmet": int(len(lgmet)),
            "n_gas_lgmet": int(len(gas_lgmet)),
            "n_gas_lgu": int(len(gas_lgu)),
            "basis_dtype": str(np.dtype(basis.dtype)),
            "coeff_dtype": str(np.dtype(coeff.dtype)),
            "resident_payload_bytes": int(
                basis.size * basis.dtype.itemsize
                + coeff.size * coeff.dtype.itemsize
                + (
                    handle["gas_scale"].size * handle["gas_scale"].dtype.itemsize
                    if "gas_scale" in handle
                    else 0
                )
            ),
            "source_grid_path": attrs.get("source_grid_path", ""),
            "line_handling": attrs.get("line_handling", ""),
        }


def compression_options(args: argparse.Namespace) -> tuple[str | None, int | None]:
    if args.compression == "none":
        return None, None
    if args.compression == "gzip":
        return "gzip", int(args.gzip_level)
    return "lzf", None


def decode_attr(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [decode_attr(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
