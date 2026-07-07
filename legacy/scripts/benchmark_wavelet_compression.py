#!/usr/bin/env python3
"""Compare current low-rank SVD assets against sparse Haar-wavelet spectra.

This is a sampling-based benchmark. It does not build new runtime assets.
It reads selected dense spectra, compares:

- the current compressed SVD-style representation already used by DSPS;
- an oracle sparse Haar-wavelet representation with top-m coefficients.

The wavelet candidate stores, per curve:

    scale + top_m(index, coeff)

where the Haar basis is fixed by the transform and therefore does not need to
be stored as a dense ``basis[k, wave]`` array. The payload estimate includes
one uint16 index and one coefficient per retained wavelet coefficient.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_WAVELET_M = (16, 32, 64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-ssp",
        default="Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5",
        help="Dense stellar SSP HDF5.",
    )
    parser.add_argument(
        "--compressed-ssp",
        default="Data/popcosmos_chabrier_stellar_ssp_basis_k64_coeff16.h5",
        help="Current compressed stellar SSP HDF5.",
    )
    parser.add_argument(
        "--dense-gas-grid",
        default="Data/popcosmos_chabrier_gas_ssp_grid.h5",
        help="Dense gas grid HDF5.",
    )
    parser.add_argument(
        "--compressed-gas-grid",
        default="Data/popcosmos_chabrier_gas_grid_basis_k64_mixed16.h5",
        help="Current compressed gas grid HDF5.",
    )
    parser.add_argument(
        "--dense-agn-component-grid",
        default="Data/popcosmos_chabrier_agn_component_ssp_grid.h5",
        help="Dense AGN component HDF5.",
    )
    parser.add_argument(
        "--compressed-agn-component-grid",
        default="Data/popcosmos_chabrier_agn_component_basis_k12_fagnlinear_coeff16.h5",
        help="Current compressed AGN component HDF5.",
    )
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wavelet-m",
        type=int,
        nargs="+",
        default=list(DEFAULT_WAVELET_M),
        help="Numbers of retained sparse Haar coefficients per spectrum.",
    )
    parser.add_argument(
        "--wavelet-coeff-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--out",
        default="outputs/report/wavelet_compression_benchmark_2026-05-31",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    examples: dict[str, dict[str, np.ndarray]] = {}

    rows.extend(audit_ssp(args, rng, examples))
    rows.extend(audit_gas(args, rng, examples))
    rows.extend(audit_agn(args, rng, examples))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No benchmark rows were produced")
    frame = frame.sort_values(
        ["component", "family", "payload_mib", "spectral_p95_rel_median"]
    )
    frame.to_csv(out / "wavelet_vs_svd_tradeoff.csv", index=False)
    summary = summarize(frame)
    (out / "wavelet_vs_svd_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plots(frame, examples, out)
    write_report(frame, summary, args, out)
    print(f"wrote {out}")
    return 0


def audit_ssp(
    args: argparse.Namespace, rng: np.random.Generator, examples: dict[str, Any]
) -> list[dict[str, Any]]:
    dense_path = Path(args.dense_ssp)
    comp_path = Path(args.compressed_ssp)
    if not dense_path.exists() or not comp_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with h5py.File(dense_path, "r") as dense_h5, h5py.File(comp_path, "r") as comp_h5:
        dense = dense_h5["ssp_flux"]
        basis = np.asarray(comp_h5["ssp_basis"], dtype=np.float32)
        coeff = np.asarray(comp_h5["ssp_coeff"], dtype=np.float32)
        scale = np.asarray(comp_h5["ssp_scale"], dtype=np.float32)
        samples = sample_indices(rng, dense.shape[:-1], args.n_samples)
        rows.append(
            evaluate_candidate(
                component="stellar_ssp",
                family="current_svd",
                method=f"current_svd_k{basis.shape[0]}",
                dense=dense,
                samples=samples,
                reconstruct=lambda idx: reconstruct_lowrank(
                    basis, coeff[idx], scale[idx]
                ),
                dense_bytes=dense_nbytes(dense),
                payload_bytes=dataset_payload_bytes(comp_h5, ("ssp_basis", "ssp_coeff", "ssp_scale")),
                n_curves=int(np.prod(dense.shape[:-1])),
            )
        )
        rows.extend(
            evaluate_wavelet_candidates(
                component="stellar_ssp",
                dense=dense,
                samples=samples,
                m_values=args.wavelet_m,
                coeff_dtype=args.wavelet_coeff_dtype,
                n_curves=int(np.prod(dense.shape[:-1])),
            )
        )
        examples["stellar_ssp"] = example_curves(
            dense,
            samples[0],
            current_reconstruct=lambda idx: reconstruct_lowrank(
                basis, coeff[idx], scale[idx]
            ),
            wavelet_m=best_example_m(args.wavelet_m),
            coeff_dtype=args.wavelet_coeff_dtype,
        )
    return rows


def audit_gas(
    args: argparse.Namespace, rng: np.random.Generator, examples: dict[str, Any]
) -> list[dict[str, Any]]:
    dense_path = Path(args.dense_gas_grid)
    comp_path = Path(args.compressed_gas_grid)
    if not dense_path.exists() or not comp_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with h5py.File(dense_path, "r") as dense_h5, h5py.File(comp_path, "r") as comp_h5:
        dense = dense_h5["ssp_flux"]
        basis = np.asarray(comp_h5["gas_basis"], dtype=np.float32)
        coeff = np.asarray(comp_h5["gas_coeff"], dtype=np.float32)
        scale = np.asarray(comp_h5["gas_scale"], dtype=np.float32)
        samples = sample_indices(rng, dense.shape[:-1], args.n_samples)
        rows.append(
            evaluate_candidate(
                component="gas",
                family="current_svd",
                method=f"current_svd_k{basis.shape[0]}_mixed16",
                dense=dense,
                samples=samples,
                reconstruct=lambda idx: reconstruct_lowrank(
                    basis, coeff[idx], scale[idx]
                ),
                dense_bytes=dense_nbytes(dense),
                payload_bytes=dataset_payload_bytes(comp_h5, ("gas_basis", "gas_coeff", "gas_scale")),
                n_curves=int(np.prod(dense.shape[:-1])),
            )
        )
        rows.extend(
            evaluate_wavelet_candidates(
                component="gas",
                dense=dense,
                samples=samples,
                m_values=args.wavelet_m,
                coeff_dtype=args.wavelet_coeff_dtype,
                n_curves=int(np.prod(dense.shape[:-1])),
            )
        )
        examples["gas"] = example_curves(
            dense,
            samples[0],
            current_reconstruct=lambda idx: reconstruct_lowrank(
                basis, coeff[idx], scale[idx]
            ),
            wavelet_m=best_example_m(args.wavelet_m),
            coeff_dtype=args.wavelet_coeff_dtype,
        )
    return rows


def audit_agn(
    args: argparse.Namespace, rng: np.random.Generator, examples: dict[str, Any]
) -> list[dict[str, Any]]:
    dense_path = Path(args.dense_agn_component_grid)
    comp_path = Path(args.compressed_agn_component_grid)
    if not dense_path.exists() or not comp_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with h5py.File(dense_path, "r") as dense_h5, h5py.File(comp_path, "r") as comp_h5:
        dense = dense_h5["agn_lnu_per_mformed"]
        fagn_grid = np.asarray(dense_h5["fagn_grid"], dtype=np.float32)
        ref_fagn_index = int(np.argmin(np.abs(fagn_grid - 1.0)))
        ref_fagn = float(fagn_grid[ref_fagn_index])
        basis = np.asarray(comp_h5["agn_basis"], dtype=np.float32)
        coeff = np.asarray(comp_h5["agn_coeff"], dtype=np.float32)
        scale = np.asarray(comp_h5["agn_scale"], dtype=np.float32)
        samples = sample_indices(rng, dense.shape[:-1], args.n_samples)

        def current_reconstruct(idx):
            fagn_index, tau_index, met_index, age_index = idx
            per_unit = reconstruct_lowrank(
                basis,
                coeff[tau_index, met_index, age_index],
                scale[tau_index, met_index, age_index],
            )
            return float(fagn_grid[fagn_index]) * per_unit

        rows.append(
            evaluate_candidate(
                component="agn",
                family="current_svd",
                method=f"current_fagn_factored_svd_k{basis.shape[0]}",
                dense=dense,
                samples=samples,
                reconstruct=current_reconstruct,
                dense_bytes=dense_nbytes(dense),
                payload_bytes=dataset_payload_bytes(comp_h5, ("agn_basis", "agn_coeff", "agn_scale")),
                n_curves=int(np.prod(dense.shape[1:-1])),
                notes="payload excludes fagn axis because current asset uses linear_runtime_multiplier",
            )
        )
        rows.extend(
            evaluate_wavelet_candidates(
                component="agn",
                dense=dense,
                samples=samples,
                m_values=args.wavelet_m,
                coeff_dtype=args.wavelet_coeff_dtype,
                n_curves=int(np.prod(dense.shape[1:-1])),
                dense_transform=lambda idx: (
                    dense[
                        ref_fagn_index,
                        int(idx[1]),
                        int(idx[2]),
                        int(idx[3]),
                        :,
                    ]
                    / ref_fagn
                    * float(fagn_grid[int(idx[0])])
                ),
                family="haar_wavelet_fagn_factored",
                method_prefix="haar_fagn_factored",
            )
        )
        examples["agn"] = example_curves(
            dense,
            samples[0],
            current_reconstruct=current_reconstruct,
            wavelet_m=best_example_m(args.wavelet_m),
            coeff_dtype=args.wavelet_coeff_dtype,
            wavelet_truth=lambda idx: (
                dense[
                    ref_fagn_index,
                    int(idx[1]),
                    int(idx[2]),
                    int(idx[3]),
                    :,
                ]
                / ref_fagn
                * float(fagn_grid[int(idx[0])])
            ),
        )
    return rows


def evaluate_wavelet_candidates(
    *,
    component: str,
    dense: h5py.Dataset,
    samples: list[tuple[int, ...]],
    m_values: list[int],
    coeff_dtype: str,
    n_curves: int,
    dense_transform: Any | None = None,
    family: str = "haar_wavelet_sparse",
    method_prefix: str = "haar",
) -> list[dict[str, Any]]:
    rows = []
    coeff_bytes = np.dtype(coeff_dtype).itemsize
    wave_n = int(dense.shape[-1])
    pad_n = next_power_of_two(wave_n)
    index_bytes = 2 if pad_n <= np.iinfo(np.uint16).max else 4
    dense_bytes = dense_nbytes(dense)
    for m in sorted(set(int(v) for v in m_values if int(v) > 0)):
        if m > pad_n:
            continue
        metrics = []
        for idx in samples:
            truth = np.asarray(
                dense_transform(idx) if dense_transform is not None else dense[idx],
                dtype=np.float32,
            )
            pred = reconstruct_sparse_haar(
                truth, m=m, coeff_dtype=coeff_dtype
            )
            metrics.append(robust_relative_errors(truth, pred))
        arr = np.asarray(metrics, dtype=float)
        payload_bytes = wavelet_payload_bytes(
            n_curves=n_curves,
            m=m,
            coeff_bytes=coeff_bytes,
            index_bytes=index_bytes,
            scale_bytes=4,
        )
        rows.append(
            {
                "component": component,
                "family": family,
                "method": f"{method_prefix}_top{m}_{coeff_dtype}_idx{index_bytes * 8}",
                "payload_mib": payload_bytes / 1024.0**2,
                "dense_mib": dense_bytes / 1024.0**2,
                "compression_factor": dense_bytes / max(payload_bytes, 1),
                "spectral_median_rel_median": float(np.nanmedian(arr[:, 0])),
                "spectral_p95_rel_median": float(np.nanmedian(arr[:, 1])),
                "spectral_p95_rel_p95": float(np.nanpercentile(arr[:, 1], 95)),
                "metric_kind": "robust_relative_flux_on_sampled_dense_spectra",
                "n_samples": len(samples),
                "n_curves_for_payload": n_curves,
                "wavelet_pad_n": pad_n,
                "notes": "oracle sparse Haar top-m per spectrum; runtime sparse implementation not yet built",
            }
        )
    return rows


def evaluate_candidate(
    *,
    component: str,
    family: str,
    method: str,
    dense: h5py.Dataset,
    samples: list[tuple[int, ...]],
    reconstruct: Any,
    dense_bytes: int,
    payload_bytes: int,
    n_curves: int,
    notes: str = "",
) -> dict[str, Any]:
    metrics = []
    for idx in samples:
        truth = np.asarray(dense[idx], dtype=np.float32)
        pred = np.asarray(reconstruct(idx), dtype=np.float32)
        metrics.append(robust_relative_errors(truth, pred))
    arr = np.asarray(metrics, dtype=float)
    return {
        "component": component,
        "family": family,
        "method": method,
        "payload_mib": payload_bytes / 1024.0**2,
        "dense_mib": dense_bytes / 1024.0**2,
        "compression_factor": dense_bytes / max(payload_bytes, 1),
        "spectral_median_rel_median": float(np.nanmedian(arr[:, 0])),
        "spectral_p95_rel_median": float(np.nanmedian(arr[:, 1])),
        "spectral_p95_rel_p95": float(np.nanpercentile(arr[:, 1], 95)),
        "metric_kind": "robust_relative_flux_on_sampled_dense_spectra",
        "n_samples": len(samples),
        "n_curves_for_payload": n_curves,
        "wavelet_pad_n": np.nan,
        "notes": notes,
    }


def reconstruct_lowrank(
    basis: np.ndarray, coeff: np.ndarray, scale: np.ndarray | float
) -> np.ndarray:
    return np.asarray(scale, dtype=np.float32) * (
        np.asarray(coeff, dtype=np.float32) @ np.asarray(basis, dtype=np.float32)
    )


def reconstruct_sparse_haar(
    spectrum: np.ndarray, m: int, coeff_dtype: str = "float16"
) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=np.float32)
    scale = curve_scale(spectrum)
    normed = spectrum / max(scale, 1.0e-30)
    coeffs = haar_forward_padded(normed)
    if m < coeffs.size:
        keep = np.argpartition(np.abs(coeffs), -m)[-m:]
        sparse = np.zeros_like(coeffs)
        values = coeffs[keep]
        if coeff_dtype == "float16":
            values = values.astype(np.float16).astype(np.float32)
        sparse[keep] = values.astype(np.float32, copy=False)
    else:
        sparse = coeffs
    recon = haar_inverse_padded(sparse, original_size=spectrum.size)
    return np.asarray(recon * scale, dtype=np.float32)


def haar_forward_padded(values: np.ndarray) -> np.ndarray:
    original = np.asarray(values, dtype=np.float32)
    n = next_power_of_two(original.size)
    padded = np.zeros(n, dtype=np.float32)
    padded[: original.size] = original
    output = padded.copy()
    temp = np.empty_like(output)
    size = n
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    while size > 1:
        half = size // 2
        even = output[:size:2]
        odd = output[1:size:2]
        temp[:half] = (even + odd) * inv_sqrt2
        temp[half:size] = (even - odd) * inv_sqrt2
        output[:size] = temp[:size]
        size = half
    return output


def haar_inverse_padded(coeffs: np.ndarray, original_size: int) -> np.ndarray:
    output = np.asarray(coeffs, dtype=np.float32).copy()
    n = output.size
    temp = np.empty_like(output)
    size = 1
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    while size < n:
        avg = output[:size]
        diff = output[size : 2 * size]
        temp[: 2 * size : 2] = (avg + diff) * inv_sqrt2
        temp[1 : 2 * size : 2] = (avg - diff) * inv_sqrt2
        output[: 2 * size] = temp[: 2 * size]
        size *= 2
    return output[:original_size]


def curve_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return 1.0
    scale = float(np.sqrt(np.mean(values[finite] ** 2)))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def robust_relative_errors(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(pred)
    if not finite.any():
        return np.nan, np.nan
    denom_floor = max(float(np.nanpercentile(np.abs(truth[finite]), 95)) * 1.0e-4, 1.0e-30)
    rel = np.abs(pred[finite] - truth[finite]) / np.maximum(
        np.abs(truth[finite]), denom_floor
    )
    return float(np.nanmedian(rel)), float(np.nanpercentile(rel, 95))


def sample_indices(
    rng: np.random.Generator, shape: tuple[int, ...], n_samples: int
) -> list[tuple[int, ...]]:
    total = int(np.prod(shape))
    n = min(max(int(n_samples), 1), total)
    flat = rng.choice(total, size=n, replace=False)
    return [tuple(int(v) for v in np.unravel_index(int(i), shape)) for i in flat]


def dense_nbytes(dataset: h5py.Dataset) -> int:
    return int(dataset.size * dataset.dtype.itemsize)


def dataset_payload_bytes(handle: h5py.File, names: tuple[str, ...]) -> int:
    return int(sum(handle[name].size * handle[name].dtype.itemsize for name in names))


def wavelet_payload_bytes(
    *, n_curves: int, m: int, coeff_bytes: int, index_bytes: int, scale_bytes: int
) -> int:
    return int(n_curves * (scale_bytes + m * (coeff_bytes + index_bytes)))


def next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def best_example_m(values: list[int]) -> int:
    values = sorted(set(int(v) for v in values if int(v) > 0))
    if 128 in values:
        return 128
    return values[len(values) // 2] if values else 128


def example_curves(
    dense: h5py.Dataset,
    idx: tuple[int, ...],
    current_reconstruct: Any,
    wavelet_m: int,
    coeff_dtype: str,
    wavelet_truth: Any | None = None,
) -> dict[str, np.ndarray]:
    truth = np.asarray(wavelet_truth(idx) if wavelet_truth else dense[idx], dtype=np.float32)
    current = np.asarray(current_reconstruct(idx), dtype=np.float32)
    wavelet = reconstruct_sparse_haar(truth, m=wavelet_m, coeff_dtype=coeff_dtype)
    return {
        "truth": truth,
        "current": current,
        "wavelet": wavelet,
        "idx": np.asarray(idx, dtype=np.int32),
        "wavelet_m": np.asarray([wavelet_m], dtype=np.int32),
    }


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for component, group in frame.groupby("component"):
        current = group[group["family"] == "current_svd"].sort_values("payload_mib")
        wavelet = group[group["family"].str.contains("haar")].sort_values(
            ["spectral_p95_rel_median", "payload_mib"]
        )
        entry: dict[str, Any] = {"component": component}
        if not current.empty:
            row = current.iloc[0]
            entry.update(
                {
                    "current_method": str(row["method"]),
                    "current_payload_mib": float(row["payload_mib"]),
                    "current_compression_factor": float(row["compression_factor"]),
                    "current_spectral_p95_rel_median": float(
                        row["spectral_p95_rel_median"]
                    ),
                }
            )
        if not wavelet.empty:
            row = wavelet.iloc[0]
            entry.update(
                {
                    "best_wavelet_method_by_loss": str(row["method"]),
                    "best_wavelet_payload_mib": float(row["payload_mib"]),
                    "best_wavelet_compression_factor": float(row["compression_factor"]),
                    "best_wavelet_spectral_p95_rel_median": float(
                        row["spectral_p95_rel_median"]
                    ),
                }
            )
        rows.append(entry)
    return {"components": rows}


def write_plots(
    frame: pd.DataFrame, examples: dict[str, dict[str, np.ndarray]], out: Path
) -> None:
    colors = {
        "current_svd": "#2563eb",
        "haar_wavelet_sparse": "#c2410c",
        "haar_wavelet_fagn_factored": "#c2410c",
    }
    markers = {"stellar_ssp": "o", "gas": "s", "agn": "^"}
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for _, row in frame.iterrows():
        ax.scatter(
            row["compression_factor"],
            row["spectral_p95_rel_median"],
            color=colors.get(row["family"], "#555555"),
            marker=markers.get(row["component"], "o"),
            s=58 if row["family"] == "current_svd" else 36,
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compression factor versus dense payload")
    ax.set_ylabel("median p95 relative spectral error")
    ax.set_title("Current SVD assets versus sparse Haar wavelet candidates")
    ax.grid(alpha=0.25, which="both")
    handles = [
        plt.Line2D([0], [0], color="#2563eb", marker="o", linestyle="", label="current SVD"),
        plt.Line2D([0], [0], color="#c2410c", marker="o", linestyle="", label="sparse Haar wavelet"),
    ]
    ax.legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "wavelet_vs_svd_tradeoff.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=False)
    for ax, component in zip(axes, ("stellar_ssp", "gas", "agn"), strict=True):
        subset = frame[frame["component"] == component].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset["label"] = subset["method"].str.replace("_", "\n", regex=False)
        subset = subset.sort_values("payload_mib").tail(8)
        ax.barh(
            np.arange(len(subset)),
            subset["payload_mib"],
            color=[
                "#2563eb" if fam == "current_svd" else "#c2410c"
                for fam in subset["family"]
            ],
            alpha=0.85,
        )
        ax.set_yticks(np.arange(len(subset)))
        ax.set_yticklabels(subset["label"], fontsize=6)
        ax.set_xscale("log")
        ax.set_title(component)
        ax.set_xlabel("payload [MiB]")
        ax.grid(axis="x", alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out / "wavelet_vs_svd_payloads.png", dpi=170)
    plt.close(fig)

    for component, curves in examples.items():
        truth = curves["truth"]
        current = curves["current"]
        wavelet = curves["wavelet"]
        x = np.arange(len(truth))
        finite = np.isfinite(truth) & np.isfinite(current) & np.isfinite(wavelet)
        if not finite.any():
            continue
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
        axes[0].plot(x, truth, color="black", lw=1.0, label="dense")
        axes[0].plot(x, current, color="#2563eb", lw=0.8, alpha=0.85, label="current SVD")
        axes[0].plot(x, wavelet, color="#c2410c", lw=0.8, alpha=0.85, label="sparse Haar")
        axes[0].set_yscale("symlog", linthresh=max(np.nanpercentile(np.abs(truth[finite]), 95) * 1e-5, 1e-30))
        axes[0].set_ylabel("flux")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)
        denom = np.maximum(
            np.abs(truth),
            max(np.nanpercentile(np.abs(truth[finite]), 95) * 1.0e-4, 1.0e-30),
        )
        axes[1].plot(x, np.abs(current - truth) / denom, color="#2563eb", lw=0.8, label="SVD rel err")
        axes[1].plot(x, np.abs(wavelet - truth) / denom, color="#c2410c", lw=0.8, label="Haar rel err")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("wavelength index")
        axes[1].set_ylabel("relative error")
        axes[1].grid(alpha=0.2, which="both")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{component}: dense vs current SVD vs sparse Haar")
        fig.tight_layout()
        fig.savefig(out / f"{component}_example_reconstruction.png", dpi=170)
        plt.close(fig)


def write_report(
    frame: pd.DataFrame, summary: dict[str, Any], args: argparse.Namespace, out: Path
) -> None:
    current_rows = frame[frame["family"] == "current_svd"]
    best_wavelet = (
        frame[frame["family"].str.contains("haar")]
        .sort_values(["component", "spectral_p95_rel_median", "payload_mib"])
        .groupby("component", as_index=False)
        .head(1)
    )
    lines = [
        "# Wavelet Versus Current SVD Compression Benchmark",
        "",
        "This is a sampling-based benchmark. It compares the current DSPS low-rank",
        "SVD-style compressed assets against an oracle sparse Haar-wavelet",
        "representation on the same sampled dense spectra.",
        "",
        "PyWavelets is not installed in the local `shine` environment, so the",
        "benchmark uses a built-in orthonormal Haar DWT implemented in NumPy.",
        "",
        "## Command",
        "",
        "```bash",
        "python scripts/benchmark_wavelet_compression.py \\",
        f"  --n-samples {int(args.n_samples)} \\",
        f"  --seed {int(args.seed)} \\",
        f"  --out {out}",
        "```",
        "",
        "## Current SVD Assets",
        "",
        current_rows[
            [
                "component",
                "method",
                "payload_mib",
                "compression_factor",
                "spectral_p95_rel_median",
                "spectral_p95_rel_p95",
            ]
        ].to_markdown(index=False),
        "",
        "## Best Sparse Haar Candidates By Spectral Loss",
        "",
        best_wavelet[
            [
                "component",
                "method",
                "payload_mib",
                "compression_factor",
                "spectral_p95_rel_median",
                "spectral_p95_rel_p95",
            ]
        ].to_markdown(index=False),
        "",
        "## How To Read This",
        "",
        "- `current_svd` is the representation currently consumed by DSPS.",
        "- `haar_*` is an oracle sparse wavelet representation: for each dense",
        "  spectrum, the benchmark keeps the top-m Haar coefficients and their",
        "  indices.",
        "- The wavelet payload estimate includes one scale, one coefficient, and",
        "  one index per retained wavelet coefficient per curve.",
        "- This benchmark does not prove wavelets are ready for JAX runtime. A",
        "  production wavelet representation would require sparse gather/scatter",
        "  kernels or a fixed-layout coefficient tensor.",
        "",
        "## Plots",
        "",
        "- `wavelet_vs_svd_tradeoff.png`",
        "- `wavelet_vs_svd_payloads.png`",
        "- `stellar_ssp_example_reconstruction.png`",
        "- `gas_example_reconstruction.png`",
        "- `agn_example_reconstruction.png`",
        "",
        "## Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
