#!/usr/bin/env python3
"""Build a low-rank base SSP grid consumed directly by JAX.

The dense input is:

    ssp_flux[stellar_lgmet, age, wave]

The compressed output stores:

    ssp_scale[stellar_lgmet, age]
    ssp_coeff[stellar_lgmet, age, k]
    ssp_basis[k, wave]

This lowers resident GPU payload for PopCosmos-like stellar-only/dust paths.
It is not used by the legacy lognormal/MDF DSPS helper, which expects the full
dense SSP tensor.
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

DEFAULT_INPUT = "Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"
DEFAULT_OUTPUT = "Data/popcosmos_chabrier_stellar_ssp_basis_k64.h5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Dense SSP HDF5.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Compressed HDF5.")
    parser.add_argument("--k", type=int, default=64, help="Number of basis vectors.")
    parser.add_argument("--oversample", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalization",
        choices=["none", "l2", "max"],
        default="l2",
        help="Per-curve normalization before basis fitting.",
    )
    parser.add_argument(
        "--basis-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Stored dtype for ssp_basis.",
    )
    parser.add_argument(
        "--coeff-dtype",
        choices=["float32", "float16"],
        default="float16",
        help="Stored dtype for ssp_coeff.",
    )
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
    )
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only:
            print(
                json.dumps(
                    validate_compressed_grid(args.output), indent=2, sort_keys=True
                )
            )
            return 0
        output = build_compressed_grid(args)
        print(f"wrote {output}")
        print(json.dumps(validate_compressed_grid(output), indent=2, sort_keys=True))
        return 0
    except FspsGridError as exc:
        return fail(str(exc))


def build_compressed_grid(args: argparse.Namespace) -> Path:
    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FspsGridError(f"Dense SSP grid not found: {input_path}")
    output = ensure_output_path(args.output, args.overwrite)
    if args.k < 1:
        raise FspsGridError("--k must be positive")
    if args.oversample < 0:
        raise FspsGridError("--oversample must be non-negative")
    compression, compression_opts = compression_options(args)
    rng = np.random.default_rng(int(args.seed))
    basis_dtype = np.dtype(args.basis_dtype)
    coeff_dtype = np.dtype(args.coeff_dtype)

    with h5py.File(input_path, "r") as src:
        required = ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet", "ssp_flux")
        missing = [key for key in required if key not in src]
        if missing:
            raise FspsGridError(
                f"Dense SSP grid {input_path} is missing datasets: {', '.join(missing)}"
            )
        dense = src["ssp_flux"]
        if dense.ndim != 3:
            raise FspsGridError("ssp_flux must have shape (stellar_lgmet, age, wave)")
        nmet, nage, nwave = dense.shape
        n_curves = nmet * nage
        rank = min(args.k + args.oversample, nwave, n_curves)
        if args.k > rank:
            raise FspsGridError("--k cannot exceed min(n_curve, n_wave)")

        scale = compute_curve_scale(dense, args.normalization)
        flat_scale = scale.reshape(-1)
        omega = rng.normal(size=(nwave, rank)).astype(np.float32)
        y = np.empty((n_curves, rank), dtype=np.float32)
        for start, stop, block in iter_curve_blocks(dense):
            y[start:stop] = normalize_block(block, flat_scale[start:stop]) @ omega
        q, _ = np.linalg.qr(y, mode="reduced")
        q = q.astype(np.float32, copy=False)
        b = np.zeros((rank, nwave), dtype=np.float32)
        for start, stop, block in iter_curve_blocks(dense):
            b += q[start:stop].T @ normalize_block(block, flat_scale[start:stop])
        _u_hat, _singular_values, vt = np.linalg.svd(b, full_matrices=False)
        basis = vt[: args.k].astype(np.float32, copy=False)

        if output.exists():
            output.unlink()
        with h5py.File(output, "w") as dst:
            for key in ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet"):
                # Axes are tiny compared with the SSP payload. Preserve their
                # source precision so compressed assets validate exactly against
                # the dense grid, including very long wavelength tails.
                dst[key] = np.asarray(src[key])
            dst.create_dataset(
                "ssp_basis",
                data=basis.astype(basis_dtype),
                compression=compression,
                compression_opts=compression_opts,
            )
            coeff_out = dst.create_dataset(
                "ssp_coeff",
                shape=(nmet, nage, args.k),
                dtype=coeff_dtype,
                chunks=(1, nage, args.k),
                compression=compression,
                compression_opts=compression_opts,
            )
            dst.create_dataset(
                "ssp_scale",
                data=scale.reshape(nmet, nage).astype(np.float32),
                chunks=(1, nage),
                compression=compression,
                compression_opts=compression_opts,
            )
            for start, stop, block in iter_curve_blocks(dense):
                coeff = (
                    normalize_block(block, flat_scale[start:stop]) @ basis.T
                ).astype(coeff_dtype, copy=False)
                met_index = start // nage
                coeff_out[met_index, :, :] = coeff
            attrs = {key: decode_attr(value) for key, value in src.attrs.items()}
            attrs.update(
                {
                    "asset_kind": "popcosmos_chabrier_compressed_stellar_ssp",
                    "compression_kind": "randomized_linear_svd",
                    "compression_version": 1,
                    "basis_space": "linear_flux",
                    "k_basis": int(args.k),
                    "normalization": str(args.normalization),
                    "compressed_dtypes": {
                        "ssp_basis": str(basis_dtype),
                        "ssp_coeff": str(coeff_dtype),
                        "ssp_scale": "float32",
                    },
                    "source_grid_path": str(input_path),
                    "source_grid_size_bytes": int(input_path.stat().st_size),
                    "source_grid_mtime_ns": int(input_path.stat().st_mtime_ns),
                    "dense_main_dataset": "ssp_flux",
                    "scientific_status": (
                        "compressed base SSP; benchmark stellar-only and dust "
                        "photometry before science use"
                    ),
                    "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "generation_command": " ".join(sys.argv),
                }
            )
            write_attrs(dst, attrs)
    return output


def compute_curve_scale(dense: h5py.Dataset, normalization: str) -> np.ndarray:
    scale = np.ones(dense.shape[:-1], dtype=np.float32)
    if normalization == "none":
        return scale
    flat = scale.reshape(-1)
    for start, stop, block in iter_curve_blocks(dense):
        if normalization == "l2":
            values = np.sqrt(np.mean(np.asarray(block, dtype=np.float64) ** 2, axis=1))
        elif normalization == "max":
            values = np.max(np.abs(block), axis=1)
        else:  # pragma: no cover - argparse constrains this
            raise FspsGridError(f"Unsupported normalization: {normalization}")
        values = np.asarray(values, dtype=np.float32)
        flat[start:stop] = np.where(np.isfinite(values) & (values > 0.0), values, 1.0)
    return scale


def normalize_block(block: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(block, dtype=np.float32) / np.maximum(scale[:, None], 1.0e-30)


def iter_curve_blocks(dense: h5py.Dataset):
    _nmet, nage, _nwave = dense.shape
    for met_index in range(dense.shape[0]):
        start = met_index * nage
        stop = start + nage
        yield start, stop, np.asarray(dense[met_index, :, :], dtype=np.float32)


def validate_compressed_grid(path: str | Path) -> dict[str, Any]:
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FspsGridError(f"Compressed SSP grid not found: {grid_path}")
    required = ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet", "ssp_basis", "ssp_coeff")
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"Compressed SSP grid {grid_path} missing datasets: {', '.join(missing)}"
            )
        wave = handle["ssp_wave"]
        age = handle["ssp_lg_age_gyr"]
        lgmet = handle["ssp_lgmet"]
        basis = handle["ssp_basis"]
        coeff = handle["ssp_coeff"]
        expected_coeff = (len(lgmet), len(age), basis.shape[0])
        if basis.ndim != 2 or basis.shape[1] != len(wave):
            raise FspsGridError("ssp_basis must have shape (n_basis, n_wave)")
        if coeff.ndim != 3 or coeff.shape != expected_coeff:
            raise FspsGridError(
                "ssp_coeff must have shape (n_ssp_lgmet, n_ssp_lg_age_gyr, n_basis)"
            )
        if "ssp_scale" in handle and handle["ssp_scale"].shape != expected_coeff[:-1]:
            raise FspsGridError(
                "ssp_scale must have shape (n_ssp_lgmet, n_ssp_lg_age_gyr)"
            )
        attrs = {key: decode_attr(value) for key, value in handle.attrs.items()}
        payload = basis.size * basis.dtype.itemsize + coeff.size * coeff.dtype.itemsize
        if "ssp_scale" in handle:
            payload += handle["ssp_scale"].size * handle["ssp_scale"].dtype.itemsize
        return {
            "path": str(grid_path),
            "asset_kind": attrs.get("asset_kind", ""),
            "compression_kind": attrs.get("compression_kind", ""),
            "k_basis": int(basis.shape[0]),
            "basis_dtype": str(np.dtype(basis.dtype)),
            "coeff_dtype": str(np.dtype(coeff.dtype)),
            "resident_payload_bytes": int(payload),
            "source_grid_path": attrs.get("source_grid_path", ""),
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
    if isinstance(value, (dict, list, tuple)):
        return value
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
