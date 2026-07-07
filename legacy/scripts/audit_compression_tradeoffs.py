#!/usr/bin/env python3
"""Audit compression factor versus spectral reconstruction loss.

This script is intentionally sampling-based for the multi-GiB gas/AGN grids:
it reads only selected dense spectra and compares candidate representations
against them. It does not build new assets. The output is a ranked CSV plus
plots showing which methods give the best size/loss tradeoff.
"""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-gas-grid", default="Data/popcosmos_chabrier_gas_ssp_grid.h5")
    parser.add_argument("--compressed-gas-grid", default="Data/popcosmos_chabrier_gas_grid_basis_k64.h5")
    parser.add_argument(
        "--dense-agn-component-grid",
        default="Data/popcosmos_chabrier_agn_component_ssp_grid.h5",
    )
    parser.add_argument(
        "--compressed-agn-component-grid",
        default="Data/popcosmos_chabrier_agn_component_basis_k32.h5",
    )
    parser.add_argument(
        "--ssp-metrics",
        default="outputs/ssp_shape_investigation/chabrier_noNE/compression_metrics.json",
    )
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/ssp_compression/tradeoffs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []
    rows.extend(audit_gas(args, rng))
    rows.extend(audit_agn(args, rng))
    rows.extend(audit_ssp(args))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No audit rows were produced")
    frame = frame.sort_values(
        ["component", "spectral_p95_rel_median", "payload_mib"], ascending=[True, True, True]
    )
    frame.to_csv(out / "compression_tradeoff.csv", index=False)
    summary = summarize(frame)
    (out / "compression_tradeoff_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plots(frame, out)
    print(f"wrote {out}")
    return 0


def audit_gas(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
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
        dense_bytes = int(dense.size * dense.dtype.itemsize)
        k_values = [k for k in (16, 24, 32, 48, 64, basis.shape[0]) if k <= basis.shape[0]]
        k_values = sorted(set(k_values))
        samples = sample_indices(rng, dense.shape[:-1], args.n_samples)
        for k in k_values:
            rows.append(
                evaluate_candidate(
                    component="gas",
                    method=f"full_spectrum_svd_k{k}_fp32",
                    dense=dense,
                    samples=samples,
                    reconstruct=lambda idx, kk=k: reconstruct_lowrank(
                        basis[:kk], coeff[idx][..., :kk], scale[idx]
                    ),
                    dense_bytes=dense_bytes,
                    payload_bytes=payload_lowrank_bytes(
                        basis.shape[1], int(np.prod(coeff.shape[:-1])), k, 4, 4, scale.size * 4
                    ),
                )
            )
            rows.append(
                evaluate_candidate(
                    component="gas",
                    method=f"full_spectrum_svd_k{k}_mixed_f16",
                    dense=dense,
                    samples=samples,
                    reconstruct=lambda idx, kk=k: reconstruct_lowrank(
                        basis[:kk].astype(np.float16).astype(np.float32),
                        coeff[idx][..., :kk].astype(np.float16).astype(np.float32),
                        scale[idx],
                    ),
                    dense_bytes=dense_bytes,
                    payload_bytes=payload_lowrank_bytes(
                        basis.shape[1], int(np.prod(coeff.shape[:-1])), k, 2, 2, scale.size * 4
                    ),
                )
            )
    return rows


def audit_agn(args: argparse.Namespace, rng: np.random.Generator) -> list[dict[str, Any]]:
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
        dense_bytes = int(dense.size * dense.dtype.itemsize)
        k_values = [k for k in (8, 12, 16, 24, 32, basis.shape[0]) if k <= basis.shape[0]]
        k_values = sorted(set(k_values))
        samples = sample_indices(rng, dense.shape[:-1], args.n_samples)
        for k in k_values:
            rows.append(
                evaluate_candidate(
                    component="agn",
                    method=f"grid_fagn_svd_k{k}_fp32",
                    dense=dense,
                    samples=samples,
                    reconstruct=lambda idx, kk=k: reconstruct_lowrank(
                        basis[:kk], coeff[idx][..., :kk], scale[idx]
                    ),
                    dense_bytes=dense_bytes,
                    payload_bytes=payload_lowrank_bytes(
                        basis.shape[1], int(np.prod(coeff.shape[:-1])), k, 4, 4, scale.size * 4
                    ),
                )
            )
            rows.append(
                evaluate_candidate(
                    component="agn",
                    method=f"fagn_factored_svd_k{k}_coeff16_basis32",
                    dense=dense,
                    samples=samples,
                    reconstruct=lambda idx, kk=k: reconstruct_factored_agn(
                        basis[:kk],
                        coeff,
                        scale,
                        idx,
                        ref_fagn_index,
                        ref_fagn,
                        fagn_grid[int(idx[0])],
                        coeff_dtype=np.float16,
                        basis_dtype=np.float32,
                    ),
                    dense_bytes=dense_bytes,
                    payload_bytes=payload_lowrank_bytes(
                        basis.shape[1],
                        int(np.prod(coeff.shape[1:-1])),
                        k,
                        4,
                        2,
                        int(np.prod(scale.shape[1:])) * 4,
                    ),
                )
            )
    return rows


def audit_ssp(args: argparse.Namespace) -> list[dict[str, Any]]:
    metrics_path = Path(args.ssp_metrics)
    if not metrics_path.exists():
        return []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for row in metrics.get("svd", {}).get("rows", []):
        k = int(row.get("k", 0))
        if k not in {16, 32, 64, 128}:
            continue
        compression = float(row.get("nominal_compression", np.nan))
        if not np.isfinite(compression) or compression <= 0.0:
            continue
        dense_mib = 54.6086
        payload_mib = dense_mib / compression
        rows.append(
            {
                "component": "stellar_ssp",
                "method": f"stellar_svd_k{k}_logflux_metric",
                "payload_mib": payload_mib,
                "dense_mib": dense_mib,
                "compression_factor": compression,
                "spectral_median_rel_median": np.nan,
                "spectral_p95_rel_median": float(row.get("p95_abs_dex", np.nan)),
                "spectral_p95_rel_p95": float(row.get("p99_abs_dex", np.nan)),
                "metric_kind": "abs_dex_from_existing_ssp_diagnostics",
                "n_samples": np.nan,
            }
        )
    return rows


def evaluate_candidate(
    *,
    component: str,
    method: str,
    dense: h5py.Dataset,
    samples: list[tuple[int, ...]],
    reconstruct: Any,
    dense_bytes: int,
    payload_bytes: int,
) -> dict[str, Any]:
    metrics = []
    for idx in samples:
        truth = np.asarray(dense[idx], dtype=np.float32)
        pred = np.asarray(reconstruct(idx), dtype=np.float32)
        metrics.append(robust_relative_errors(truth, pred))
    arr = np.asarray(metrics, dtype=float)
    return {
        "component": component,
        "method": method,
        "payload_mib": payload_bytes / 1024.0**2,
        "dense_mib": dense_bytes / 1024.0**2,
        "compression_factor": dense_bytes / max(payload_bytes, 1),
        "spectral_median_rel_median": float(np.nanmedian(arr[:, 0])),
        "spectral_p95_rel_median": float(np.nanmedian(arr[:, 1])),
        "spectral_p95_rel_p95": float(np.nanpercentile(arr[:, 1], 95)),
        "metric_kind": "robust_relative_flux_on_sampled_dense_spectra",
        "n_samples": len(samples),
    }


def robust_relative_errors(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    finite = np.isfinite(truth) & np.isfinite(pred)
    if not finite.any():
        return np.nan, np.nan
    ref = np.abs(truth[finite])
    floor = np.nanmedian(ref[ref > 0.0]) if np.any(ref > 0.0) else 1.0
    denom = np.maximum(ref, max(float(floor), 1.0e-300))
    err = np.abs(pred[finite] - truth[finite]) / denom
    return float(np.nanmedian(err)), float(np.nanpercentile(err, 95))


def reconstruct_lowrank(basis: np.ndarray, coeff: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(coeff, dtype=np.float32)[None, :] @ np.asarray(basis, dtype=np.float32)).ravel() * np.float32(scale)


def reconstruct_factored_agn(
    basis: np.ndarray,
    coeff: np.ndarray,
    scale: np.ndarray,
    idx: tuple[int, ...],
    ref_fagn_index: int,
    ref_fagn: float,
    target_fagn: float,
    *,
    coeff_dtype: Any,
    basis_dtype: Any,
) -> np.ndarray:
    _ifagn, itau, imet, iage = idx
    c = coeff[ref_fagn_index, itau, imet, iage, : basis.shape[0]] / np.float32(ref_fagn)
    s = scale[ref_fagn_index, itau, imet, iage]
    b = basis.astype(basis_dtype).astype(np.float32)
    c = c.astype(coeff_dtype).astype(np.float32)
    return (c[None, :] @ b).ravel() * np.float32(s) * np.float32(target_fagn)


def sample_indices(
    rng: np.random.Generator, shape: tuple[int, ...], n_samples: int
) -> list[tuple[int, ...]]:
    total = int(np.prod(shape))
    n = min(int(n_samples), total)
    flat = rng.choice(total, size=n, replace=False)
    return [np.unravel_index(int(index), shape) for index in flat]


def payload_lowrank_bytes(
    n_wave: int,
    n_curves: int,
    k: int,
    basis_itemsize: int,
    coeff_itemsize: int,
    scale_bytes: int,
) -> int:
    return int(k * n_wave * basis_itemsize + n_curves * k * coeff_itemsize + scale_bytes)


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component, group in frame.groupby("component"):
        finite = group[np.isfinite(group["spectral_p95_rel_median"])]
        if finite.empty:
            continue
        best = finite.sort_values(
            ["spectral_p95_rel_median", "compression_factor"],
            ascending=[True, False],
        ).iloc[0]
        compact = finite.sort_values(
            ["compression_factor", "spectral_p95_rel_median"],
            ascending=[False, True],
        ).iloc[0]
        result[str(component)] = {
            "lowest_loss_method": best.to_dict(),
            "highest_compression_method": compact.to_dict(),
        }
    return result


def write_plots(frame: pd.DataFrame, out: Path) -> None:
    plot_frame = frame.copy()
    plot_frame = plot_frame[np.isfinite(plot_frame["spectral_p95_rel_median"])]
    if plot_frame.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for component, group in plot_frame.groupby("component"):
        ax.scatter(
            group["compression_factor"],
            group["spectral_p95_rel_median"],
            s=42,
            alpha=0.82,
            label=component,
        )
        for _, row in group.iterrows():
            label = short_label(str(row["method"]))
            ax.annotate(label, (row["compression_factor"], row["spectral_p95_rel_median"]), fontsize=6, alpha=0.75)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compression factor versus dense tensor")
    ax.set_ylabel("median p95 spectral loss")
    ax.set_title("Compression factor versus sampled reconstruction loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "compression_factor_vs_loss.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ordered = plot_frame.sort_values("payload_mib")
    labels = [f"{c}\\n{short_label(m)}" for c, m in zip(ordered["component"], ordered["method"], strict=False)]
    ax.bar(np.arange(len(ordered)), ordered["payload_mib"], color="#4c78a8")
    ax.set_yscale("log")
    ax.set_ylabel("estimated resident payload [MiB]")
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6)
    ax.set_title("Estimated resident payload by candidate")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "candidate_payload_mib.png", dpi=170)
    plt.close(fig)


def short_label(method: str) -> str:
    return (
        method.replace("full_spectrum_", "")
        .replace("grid_fagn_", "")
        .replace("fagn_factored_", "fagn_fact_")
        .replace("_logflux_metric", "")
        .replace("stellar_", "")
    )


if __name__ == "__main__":
    raise SystemExit(main())
