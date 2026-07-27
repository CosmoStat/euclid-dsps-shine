#!/usr/bin/env python3
"""Build a low-rank AGN component grid consumed directly by JAX.

The input dense grid is:

    agn_lnu_per_mformed[fagn, agn_tau, stellar_lgmet, age, wave]

The output compressed grid stores:

    agn_scale[fagn, agn_tau, stellar_lgmet, age]
    agn_coeff[fagn, agn_tau, stellar_lgmet, age, k]
    agn_basis[k, wave]

so JAX can interpolate/sum in coefficient space and reconstruct only the
per-galaxy AGN SED. The script streams the dense HDF5 in age-by-wavelength
blocks and does not load the full dense tensor at once.
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

DEFAULT_INPUT = "Data/popcosmos_chabrier_agn_component_ssp_grid.h5"
DEFAULT_OUTPUT = "Data/popcosmos_chabrier_agn_component_basis_k32.h5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Dense AGN grid HDF5.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Compressed HDF5.")
    parser.add_argument("--k", type=int, default=32, help="Number of basis vectors.")
    parser.add_argument(
        "--oversample",
        type=int,
        default=12,
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
        "--factor-fagn",
        action="store_true",
        help=(
            "Store spectra per unit fagn and drop the fagn coefficient axis. "
            "The runtime model multiplies by fagn after interpolation."
        ),
    )
    parser.add_argument(
        "--basis-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Stored dtype for agn_basis.",
    )
    parser.add_argument(
        "--coeff-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Stored dtype for agn_coeff.",
    )
    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression for coefficient datasets.",
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
        raise FspsGridError(f"Dense AGN component grid not found: {input_path}")
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
            "fagn_grid",
            "agn_tau_grid",
            "agn_lnu_per_mformed",
        )
        missing = [key for key in required if key not in src]
        if missing:
            raise FspsGridError(
                f"Dense AGN grid {input_path} is missing datasets: {', '.join(missing)}"
            )
        dense = src["agn_lnu_per_mformed"]
        if dense.ndim != 5:
            raise FspsGridError(
                "agn_lnu_per_mformed must have shape "
                "(fagn, agn_tau, stellar_lgmet, age, wave)"
            )
        nfagn, ntau, nmet, nage, nwave = dense.shape
        if args.k > nwave:
            raise FspsGridError("--k cannot exceed the wavelength dimension")
        fagn_grid = np.asarray(src["fagn_grid"], dtype=np.float32)
        factor_fagn = bool(getattr(args, "factor_fagn", False))
        if factor_fagn:
            ref_index = int(np.argmin(np.abs(fagn_grid - 1.0)))
            ref_fagn = float(fagn_grid[ref_index])
            if not np.isfinite(ref_fagn) or ref_fagn <= 0.0:
                raise FspsGridError("--factor-fagn requires a positive fagn reference")
            curve_shape = (ntau, nmet, nage)
            n_curves = ntau * nmet * nage
            block_iter = lambda: iter_factored_fagn_curve_blocks(  # noqa: E731
                dense, ref_index, ref_fagn
            )
        else:
            ref_index = None
            ref_fagn = None
            curve_shape = (nfagn, ntau, nmet, nage)
            n_curves = nfagn * ntau * nmet * nage
            block_iter = lambda: iter_dense_curve_blocks(dense)  # noqa: E731
        rank = min(args.k + args.oversample, nwave, n_curves)
        scale = compute_curve_scale(curve_shape, block_iter, args.normalization)
        omega = rng.normal(size=(nwave, rank)).astype(np.float32)
        y = np.empty((n_curves, rank), dtype=np.float32)
        flat_scale = scale.reshape(-1)
        for start, stop, block in block_iter():
            block_norm = normalize_block(block, flat_scale[start:stop])
            y[start:stop] = block_norm @ omega
        q, _ = np.linalg.qr(y, mode="reduced")
        q = q.astype(np.float32, copy=False)
        b = np.zeros((rank, nwave), dtype=np.float32)
        for start, stop, block in block_iter():
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
                "fagn_grid",
                "agn_tau_grid",
            ):
                dst[key] = np.asarray(src[key], dtype=np.float32)
            dst.create_dataset(
                "agn_basis",
                data=basis.astype(basis_dtype),
                compression=compression,
                compression_opts=compression_opts,
            )
            coeff_shape = curve_shape + (args.k,)
            coeff_out = dst.create_dataset(
                "agn_coeff",
                shape=coeff_shape,
                dtype=coeff_dtype,
                chunks=agn_coeff_chunks(coeff_shape),
                compression=compression,
                compression_opts=compression_opts,
            )
            scale_out = dst.create_dataset(
                "agn_scale",
                data=scale.reshape(curve_shape).astype(np.float32),
                chunks=agn_scale_chunks(curve_shape),
                compression=compression,
                compression_opts=compression_opts,
            )
            for start, stop, block in block_iter():
                block_norm = normalize_block(block, flat_scale[start:stop])
                coeff = (block_norm @ basis.T).astype(coeff_dtype, copy=False)
                write_coeff_block(coeff_out, start, stop, coeff, curve_shape)
            attrs = {key: decode_attr(value) for key, value in src.attrs.items()}
            attrs.update(
                {
                    "asset_kind": "popcosmos_chabrier_compressed_agn_component_grid",
                    "compression_kind": "randomized_linear_svd",
                    "compression_version": 2 if factor_fagn else 1,
                    "basis_space": "linear_flux",
                    "k_basis": int(args.k),
                    "normalization": str(args.normalization),
                    "fagn_handling": (
                        "linear_runtime_multiplier"
                        if factor_fagn
                        else "grid_interpolation"
                    ),
                    "fagn_reference_index": (
                        -1 if ref_index is None else int(ref_index)
                    ),
                    "fagn_reference": ("" if ref_fagn is None else float(ref_fagn)),
                    "compressed_dtypes": {
                        "agn_basis": str(basis_dtype),
                        "agn_coeff": str(coeff_dtype),
                        "agn_scale": "float32",
                    },
                    "line_handling": "not_applicable_agn_continuum_component",
                    "source_grid_path": str(input_path),
                    "source_grid_size_bytes": int(input_path.stat().st_size),
                    "source_grid_mtime_ns": int(input_path.stat().st_mtime_ns),
                    "dense_main_dataset": "agn_lnu_per_mformed",
                    "units_agn_basis": "same normalized units as agn_lnu_per_mformed",
                    "units_agn_coeff": "dimensionless projection coefficient",
                    "units_agn_scale": attrs.get(
                        "units_agn_lnu_per_mformed", "Lsun/Hz/Msun formed"
                    ),
                    "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "generation_command": " ".join(sys.argv),
                }
            )
            write_attrs(dst, attrs)
            del scale_out
    return output


def compute_curve_scale(
    curve_shape: tuple[int, ...], block_iter: Any, normalization: str
) -> np.ndarray:
    scale = np.ones(curve_shape, dtype=np.float32)
    if normalization == "none":
        return scale
    flat = scale.reshape(-1)
    for start, stop, block in block_iter():
        if normalization == "l2":
            values = np.sqrt(np.mean(np.asarray(block, dtype=np.float64) ** 2, axis=1))
        elif normalization == "max":
            values = np.max(np.abs(block), axis=1)
        else:  # pragma: no cover - argparse constrains this
            raise FspsGridError(f"Unsupported normalization: {normalization}")
        values = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(values) & (values > 0.0)
        values = np.where(finite, values, 1.0).astype(np.float32)
        flat[start:stop] = values
    return scale


def normalize_block(block: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float32)
    return np.asarray(block, dtype=np.float32) / np.maximum(scale[:, None], 1.0e-30)


def iter_dense_curve_blocks(dense: h5py.Dataset):
    nfagn, ntau, nmet, nage, nwave = dense.shape
    for i in range(nfagn):
        for j in range(ntau):
            for k in range(nmet):
                start = (((i * ntau) + j) * nmet + k) * nage
                stop = start + nage
                block = np.asarray(dense[i, j, k, :, :], dtype=np.float32).reshape(
                    nage, nwave
                )
                yield start, stop, block


def iter_factored_fagn_curve_blocks(
    dense: h5py.Dataset, ref_index: int, ref_fagn: float
):
    _nfagn, ntau, nmet, nage, _nwave = dense.shape
    for j in range(ntau):
        for k in range(nmet):
            start = (j * nmet + k) * nage
            stop = start + nage
            block = np.asarray(dense[ref_index, j, k, :, :], dtype=np.float32).reshape(
                nage, -1
            )
            yield start, stop, block / np.float32(ref_fagn)


def agn_coeff_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 5:
        return (1, 1, 1, shape[3], shape[4])
    if len(shape) == 4:
        return (1, 1, shape[2], shape[3])
    raise FspsGridError(f"Unsupported agn_coeff shape: {shape}")


def agn_scale_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 4:
        return (1, 1, 1, shape[3])
    if len(shape) == 3:
        return (1, 1, shape[2])
    raise FspsGridError(f"Unsupported agn_scale shape: {shape}")


def write_coeff_block(
    coeff_out: h5py.Dataset,
    start: int,
    stop: int,
    coeff: np.ndarray,
    curve_shape: tuple[int, ...] | None = None,
) -> None:
    shape = curve_shape or coeff_out.shape[:-1]
    nage = shape[-1]
    first = start // nage
    last = (stop - 1) // nage
    if first != last:
        raise FspsGridError("Coefficient block crosses natural HDF5 block boundary")
    if len(shape) == 4:
        _nfagn, ntau, nmet, _nage = shape
        i = first // (ntau * nmet)
        rem = first % (ntau * nmet)
        j = rem // nmet
        k = rem % nmet
        coeff_out[i, j, k, :, :] = coeff
        return
    if len(shape) == 3:
        _ntau, nmet, _nage = shape
        j = first // nmet
        k = first % nmet
        coeff_out[j, k, :, :] = coeff
        return
    raise FspsGridError(f"Unsupported AGN coefficient curve shape: {shape}")


def validate_compressed_grid(path: str | Path) -> dict[str, Any]:
    grid_path = Path(path).expanduser()
    if not grid_path.exists():
        raise FspsGridError(f"Compressed AGN component grid not found: {grid_path}")
    required = (
        "ssp_wave",
        "ssp_lg_age_gyr",
        "ssp_lgmet",
        "fagn_grid",
        "agn_tau_grid",
        "agn_basis",
        "agn_coeff",
        "agn_scale",
    )
    with h5py.File(grid_path, "r") as handle:
        missing = [key for key in required if key not in handle]
        if missing:
            raise FspsGridError(
                f"Compressed AGN component grid {grid_path} missing datasets: "
                f"{', '.join(missing)}"
            )
        wave = np.asarray(handle["ssp_wave"])
        age = np.asarray(handle["ssp_lg_age_gyr"])
        lgmet = np.asarray(handle["ssp_lgmet"])
        fagn = np.asarray(handle["fagn_grid"])
        tau = np.asarray(handle["agn_tau_grid"])
        basis_shape = tuple(handle["agn_basis"].shape)
        coeff_shape = tuple(handle["agn_coeff"].shape)
        scale_shape = tuple(handle["agn_scale"].shape)
        attrs = {key: decode_attr(value) for key, value in handle.attrs.items()}
        fagn_handling = str(attrs.get("fagn_handling", "grid_interpolation"))
        if fagn_handling == "linear_runtime_multiplier":
            expected_coeff = (len(tau), len(lgmet), len(age), basis_shape[0])
        else:
            expected_coeff = (
                len(fagn),
                len(tau),
                len(lgmet),
                len(age),
                basis_shape[0],
            )
        expected_scale = expected_coeff[:-1]
        if len(basis_shape) != 2 or basis_shape[1] != len(wave):
            raise FspsGridError("agn_basis must have shape (n_basis, n_wave)")
        if coeff_shape != expected_coeff:
            raise FspsGridError(
                f"agn_coeff shape {coeff_shape} does not match {expected_coeff}"
            )
        if scale_shape != expected_scale:
            raise FspsGridError(
                f"agn_scale shape {scale_shape} does not match {expected_scale}"
            )
        for name, values in {
            "ssp_wave": wave,
            "ssp_lg_age_gyr": age,
            "ssp_lgmet": lgmet,
            "fagn_grid": fagn,
            "agn_tau_grid": tau,
        }.items():
            validate_axis(name, values)
    return {
        "path": str(grid_path),
        "basis_shape": list(basis_shape),
        "coeff_shape": list(coeff_shape),
        "scale_shape": list(scale_shape),
        "k_basis": int(basis_shape[0]),
        "fagn_handling": attrs.get("fagn_handling", "grid_interpolation"),
        "basis_dtype": str(handle_dtype(grid_path, "agn_basis")),
        "coeff_dtype": str(handle_dtype(grid_path, "agn_coeff")),
        "asset_kind": attrs.get("asset_kind", ""),
        "compression_kind": attrs.get("compression_kind", ""),
        "normalization": attrs.get("normalization", ""),
    }


def handle_dtype(path: Path, name: str) -> np.dtype:
    with h5py.File(path, "r") as handle:
        return np.dtype(handle[name].dtype)


def validate_axis(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise FspsGridError(f"{name} must be a finite 1D axis")
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        raise FspsGridError(f"{name} must be strictly increasing")


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
