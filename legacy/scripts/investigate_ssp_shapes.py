#!/usr/bin/env python3
"""Plot SSP shapes and test simple compression approximations.

The diagnostics focus on the wavelength interval used by the PopCosmos-like
SED comparisons by default. Full FSPS SSP files extend far into the IR, so
compression numbers should be interpreted for the selected wave range.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SSP_PATHS = (
    Path("Data/fsps_v0.4.7_mist_c3k_a_chabrier_noNE.h5"),
    Path("Data/fsps_v0.4.7_mist_c3k_a_chabrier_wNE_logGasU-2.0_logGasZ0.0.h5"),
)
DEFAULT_GAS_GRID_PATH = Path("Data/popcosmos_chabrier_gas_ssp_grid.h5")
DEFAULT_AGN_COMPONENT_PATH = Path("Data/popcosmos_chabrier_agn_component_ssp_grid.h5")
DEFAULT_OUT = Path("outputs/ssp_shape_investigation")
TARGET_AGES_GYR = (0.001, 0.01, 0.1, 1.0, 5.0, 10.0)
TARGET_TRACK_WAVES = (1500.0, 3600.0, 5500.0, 10000.0, 16000.0)
SVD_K = (1, 2, 4, 8, 16, 32, 64, 128)
QUADRATIC_SEGMENTS = (16, 32, 64, 128, 256, 512)
HAAR_KEEP = (32, 64, 128, 256, 512, 1024)
INTERP_KNOTS = (128, 256, 512, 1024)


@dataclass(frozen=True)
class SspData:
    path: Path
    label: str
    attrs: dict[str, Any]
    wave: np.ndarray
    lg_age_gyr: np.ndarray
    lgmet: np.ndarray
    flux: np.ndarray
    surviving_mstar: np.ndarray | None

    @property
    def age_gyr(self) -> np.ndarray:
        return np.power(10.0, self.lg_age_gyr)


def main() -> None:
    args = parse_args()
    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    summaries: list[dict[str, Any]] = []
    compression: dict[str, Any] = {}
    all_outputs: list[Path] = []

    ssps = [load_ssp(path) for path in args.ssp]
    for ssp in ssps:
        asset_dir = out_dir / ssp.label
        asset_dir.mkdir(parents=True, exist_ok=True)
        print(f"[ssp] {ssp.path} -> {asset_dir}")
        summaries.append(summarize_ssp(ssp, args.wave_min, args.wave_max))
        all_outputs.extend(plot_dimension_diagnostics(ssp, asset_dir, args))
        compression[ssp.label] = compression_diagnostics(ssp, asset_dir, args)
        all_outputs.extend(sorted(asset_dir.glob("*.png")))

    if len(ssps) >= 2:
        comparison_path = plot_pairwise_ratio(ssps[0], ssps[1], out_dir, args)
        all_outputs.append(comparison_path)
    related_assets = inspect_related_assets(args)
    if related_assets:
        all_outputs.append(plot_asset_size_summary(related_assets, out_dir, args))

    summary_path = out_dir / "ssp_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "wave_range_angstrom": [args.wave_min, args.wave_max],
                "log_wave_grid_size": args.log_grid_size,
                "summaries": summaries,
                "related_assets": related_assets,
                "compression": compression,
                "plots": sorted({str(path) for path in all_outputs}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = write_report(out_dir, summaries, compression, args)
    print(f"[ssp] wrote {summary_path}")
    print(f"[ssp] wrote {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ssp",
        action="append",
        type=Path,
        default=None,
        help=(
            "SSP HDF5 path. May be repeated. Defaults to the active Chabrier "
            "pure-stellar and fixed-nebular assets."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--wave-min", type=float, default=900.0)
    parser.add_argument("--wave-max", type=float, default=30000.0)
    parser.add_argument(
        "--log-grid-size",
        type=int,
        default=2048,
        help="Power-of-two log-wavelength grid size for compression tests.",
    )
    parser.add_argument(
        "--gas-grid",
        type=Path,
        default=DEFAULT_GAS_GRID_PATH,
        help="Related PopCosmos gas SSP grid to summarize without loading fully.",
    )
    parser.add_argument(
        "--agn-component-grid",
        type=Path,
        default=DEFAULT_AGN_COMPONENT_PATH,
        help="Related PopCosmos AGN component grid to summarize without loading fully.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    if args.ssp is None:
        args.ssp = list(DEFAULT_SSP_PATHS)
    if args.wave_min <= 0 or args.wave_max <= args.wave_min:
        parser.error("--wave-min/--wave-max must define a positive increasing range")
    if args.log_grid_size < 128 or not _is_power_of_two(args.log_grid_size):
        parser.error("--log-grid-size must be a power of two >= 128")
    return args


def load_ssp(path: Path) -> SspData:
    path = path.expanduser()
    with h5py.File(path, "r") as handle:
        required = ("ssp_wave", "ssp_lg_age_gyr", "ssp_lgmet", "ssp_flux")
        missing = [key for key in required if key not in handle]
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"{path} is missing required datasets: {joined}")
        attrs = {key: _jsonable_attr(handle.attrs[key]) for key in handle.attrs.keys()}
        surviving = (
            np.asarray(handle["ssp_surviving_mstar"], dtype=np.float32)
            if "ssp_surviving_mstar" in handle
            else None
        )
        return SspData(
            path=path,
            label=_label_from_path(path, attrs),
            attrs=attrs,
            wave=np.asarray(handle["ssp_wave"], dtype=np.float64),
            lg_age_gyr=np.asarray(handle["ssp_lg_age_gyr"], dtype=np.float64),
            lgmet=np.asarray(handle["ssp_lgmet"], dtype=np.float64),
            flux=np.asarray(handle["ssp_flux"], dtype=np.float32),
            surviving_mstar=surviving,
        )


def summarize_ssp(ssp: SspData, wave_min: float, wave_max: float) -> dict[str, Any]:
    wave_mask = _wave_mask(ssp.wave, wave_min, wave_max)
    raw_bytes = int(np.prod(ssp.flux.shape) * ssp.flux.dtype.itemsize)
    file_bytes = ssp.path.stat().st_size if ssp.path.exists() else None
    return {
        "label": ssp.label,
        "path": str(ssp.path),
        "asset_kind": ssp.attrs.get("asset_kind", ""),
        "shape": list(ssp.flux.shape),
        "dtype": str(ssp.flux.dtype),
        "raw_flux_bytes": raw_bytes,
        "hdf5_file_bytes": file_bytes,
        "wave_min_angstrom": float(np.nanmin(ssp.wave)),
        "wave_max_angstrom": float(np.nanmax(ssp.wave)),
        "wave_points_total": int(ssp.wave.size),
        "wave_points_selected": int(np.count_nonzero(wave_mask)),
        "age_gyr_min": float(np.nanmin(ssp.age_gyr)),
        "age_gyr_max": float(np.nanmax(ssp.age_gyr)),
        "n_age": int(ssp.lg_age_gyr.size),
        "lgmet_min": float(np.nanmin(ssp.lgmet)),
        "lgmet_max": float(np.nanmax(ssp.lgmet)),
        "n_lgmet": int(ssp.lgmet.size),
        "imf_name": ssp.attrs.get("imf_name", ""),
        "add_neb_emission": ssp.attrs.get("add_neb_emission", ""),
        "add_neb_continuum": ssp.attrs.get("add_neb_continuum", ""),
        "z_sun": ssp.attrs.get("z_sun", ""),
    }


def inspect_related_assets(args: argparse.Namespace) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for role, path in (
        ("base SSP", DEFAULT_SSP_PATHS[1]),
        ("stellar-only SSP", DEFAULT_SSP_PATHS[0]),
        ("gas SSP grid", args.gas_grid),
        ("AGN component grid", args.agn_component_grid),
    ):
        expanded = Path(path).expanduser()
        if expanded.exists():
            assets.append(inspect_hdf5_asset(expanded, role))
    return assets


def inspect_hdf5_asset(path: Path, role: str) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        attrs = {key: _jsonable_attr(handle.attrs[key]) for key in handle.attrs.keys()}
        for key, ds in handle.items():
            if isinstance(ds, h5py.Dataset):
                raw_bytes = int(np.prod(ds.shape) * ds.dtype.itemsize)
                datasets[key] = {
                    "shape": list(ds.shape),
                    "dtype": str(ds.dtype),
                    "raw_bytes": raw_bytes,
                }
    main_key = _main_flux_key(datasets)
    return {
        "role": role,
        "path": str(path),
        "file_bytes": int(path.stat().st_size),
        "asset_kind": attrs.get("asset_kind", ""),
        "main_dataset": main_key,
        "main_shape": datasets.get(main_key, {}).get("shape", []),
        "main_raw_bytes": datasets.get(main_key, {}).get("raw_bytes", 0),
        "datasets": datasets,
    }


def _main_flux_key(datasets: dict[str, Any]) -> str:
    for key in ("ssp_flux", "agn_lnu_per_mformed", "template_lnu_per_lbol"):
        if key in datasets:
            return key
    return next(iter(datasets), "")


def plot_asset_size_summary(
    assets: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace
) -> Path:
    labels = [asset["role"] for asset in assets]
    file_mib = [asset["file_bytes"] / 1024**2 for asset in assets]
    raw_mib = [asset["main_raw_bytes"] / 1024**2 for asset in assets]
    x = np.arange(len(assets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, file_mib, width, label="HDF5 file")
    ax.bar(x + width / 2, raw_mib, width, label="main tensor float payload")
    ax.set_yscale("log")
    ax.set_ylabel("size [MiB, log scale]")
    ax.set_title("PopCosmos spectral assets: storage and resident tensor size")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    for index, asset in enumerate(assets):
        shape = " x ".join(str(v) for v in asset["main_shape"])
        ax.text(
            index,
            max(file_mib[index], raw_mib[index]) * 1.2,
            shape,
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    fig.tight_layout()
    path = out_dir / "popcosmos_asset_size_summary.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dimension_diagnostics(
    ssp: SspData, out_dir: Path, args: argparse.Namespace
) -> list[Path]:
    paths = [
        plot_wavelength_dimension(ssp, out_dir, args),
        plot_age_tracks(ssp, out_dir, args),
        plot_metallicity_tracks(ssp, out_dir, args),
        plot_age_wavelength_heatmaps(ssp, out_dir, args),
        plot_metallicity_wavelength_heatmaps(ssp, out_dir, args),
    ]
    if ssp.surviving_mstar is not None:
        paths.append(plot_surviving_mass(ssp, out_dir, args))
    return paths


def plot_wavelength_dimension(
    ssp: SspData, out_dir: Path, args: argparse.Namespace
) -> Path:
    wave_mask = _wave_mask(ssp.wave, args.wave_min, args.wave_max)
    wave = ssp.wave[wave_mask]
    met_indices = _representative_metallicity_indices(ssp)
    age_indices = _nearest_indices(ssp.age_gyr, TARGET_AGES_GYR)

    fig, axes = plt.subplots(len(met_indices), 1, figsize=(10.8, 9.2), sharex=True)
    if len(met_indices) == 1:
        axes = [axes]
    floor = _positive_floor(ssp.flux[:, :, wave_mask])
    for ax, met_idx in zip(axes, met_indices, strict=True):
        for age_idx in age_indices:
            y = np.clip(ssp.flux[met_idx, age_idx, wave_mask], floor, np.inf)
            ax.plot(
                wave,
                y,
                lw=1.0,
                label=f"{ssp.age_gyr[age_idx]:.3g} Gyr",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Lnu / Msun")
        ax.set_title(f"log10(Z)={ssp.lgmet[met_idx]:.3f}")
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("rest wavelength [Angstrom]")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"{ssp.label}: wavelength slices by age", y=0.995)
    fig.tight_layout()
    path = out_dir / "dimension_wavelength_sample_spectra.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def plot_age_tracks(ssp: SspData, out_dir: Path, args: argparse.Namespace) -> Path:
    wave_indices = _nearest_indices(ssp.wave, TARGET_TRACK_WAVES)
    met_indices = _representative_metallicity_indices(ssp)
    fig, axes = plt.subplots(1, len(wave_indices), figsize=(15.5, 3.8), sharey=False)
    if len(wave_indices) == 1:
        axes = [axes]
    floor = _positive_floor(ssp.flux)
    for ax, wave_idx in zip(axes, wave_indices, strict=True):
        for met_idx in met_indices:
            y = np.log10(np.clip(ssp.flux[met_idx, :, wave_idx], floor, np.inf))
            ax.plot(
                ssp.age_gyr,
                y,
                marker="o",
                ms=2.0,
                lw=1.0,
                label=f"logZ={ssp.lgmet[met_idx]:.2f}",
            )
        ax.set_xscale("log")
        ax.set_title(f"{ssp.wave[wave_idx]:.0f} A")
        ax.set_xlabel("age [Gyr]")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("log10 Lnu / Msun")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"{ssp.label}: age dimension at fixed wavelengths", y=1.02)
    fig.tight_layout()
    path = out_dir / "dimension_age_tracks.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def plot_metallicity_tracks(
    ssp: SspData, out_dir: Path, args: argparse.Namespace
) -> Path:
    wave_indices = _nearest_indices(ssp.wave, TARGET_TRACK_WAVES)
    age_indices = _nearest_indices(ssp.age_gyr, (0.01, 0.1, 1.0, 10.0))
    fig, axes = plt.subplots(1, len(wave_indices), figsize=(15.5, 3.8), sharey=False)
    if len(wave_indices) == 1:
        axes = [axes]
    floor = _positive_floor(ssp.flux)
    for ax, wave_idx in zip(axes, wave_indices, strict=True):
        for age_idx in age_indices:
            y = np.log10(np.clip(ssp.flux[:, age_idx, wave_idx], floor, np.inf))
            ax.plot(
                ssp.lgmet,
                y,
                marker="o",
                ms=3.0,
                lw=1.0,
                label=f"{ssp.age_gyr[age_idx]:.3g} Gyr",
            )
        ax.set_title(f"{ssp.wave[wave_idx]:.0f} A")
        ax.set_xlabel("log10 absolute Z")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("log10 Lnu / Msun")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"{ssp.label}: metallicity dimension at fixed wavelengths", y=1.02)
    fig.tight_layout()
    path = out_dir / "dimension_metallicity_tracks.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def plot_age_wavelength_heatmaps(
    ssp: SspData, out_dir: Path, args: argparse.Namespace
) -> Path:
    wave_mask = _wave_mask(ssp.wave, args.wave_min, args.wave_max)
    wave = ssp.wave[wave_mask]
    x_edges = _log_edges(wave)
    y_edges = _log_edges(ssp.age_gyr)
    met_indices = _representative_metallicity_indices(ssp)
    log_flux = _safe_log10(ssp.flux[:, :, wave_mask])

    fig, axes = plt.subplots(1, len(met_indices), figsize=(15.0, 4.4), sharey=True)
    if len(met_indices) == 1:
        axes = [axes]
    vmin, vmax = _shared_limits([log_flux[idx] for idx in met_indices])
    mesh = None
    for ax, met_idx in zip(axes, met_indices, strict=True):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            log_flux[met_idx],
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"log10(Z)={ssp.lgmet[met_idx]:.3f}")
        ax.set_xlabel("rest wavelength [Angstrom]")
    axes[0].set_ylabel("age [Gyr]")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="log10 Lnu / Msun", shrink=0.85)
    fig.suptitle(f"{ssp.label}: age x wavelength", y=1.02)
    path = out_dir / "dimension_age_wavelength_heatmaps.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metallicity_wavelength_heatmaps(
    ssp: SspData, out_dir: Path, args: argparse.Namespace
) -> Path:
    wave_mask = _wave_mask(ssp.wave, args.wave_min, args.wave_max)
    wave = ssp.wave[wave_mask]
    x_edges = _log_edges(wave)
    y_edges = _linear_edges(ssp.lgmet)
    age_indices = _nearest_indices(ssp.age_gyr, (0.01, 0.1, 1.0, 10.0))
    log_flux = _safe_log10(ssp.flux[:, :, wave_mask])

    fig, axes = plt.subplots(1, len(age_indices), figsize=(16.0, 4.1), sharey=True)
    if len(age_indices) == 1:
        axes = [axes]
    vmin, vmax = _shared_limits([log_flux[:, idx, :] for idx in age_indices])
    mesh = None
    for ax, age_idx in zip(axes, age_indices, strict=True):
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            log_flux[:, age_idx, :],
            shading="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xscale("log")
        ax.set_title(f"{ssp.age_gyr[age_idx]:.3g} Gyr")
        ax.set_xlabel("rest wavelength [Angstrom]")
    axes[0].set_ylabel("log10 absolute Z")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label="log10 Lnu / Msun", shrink=0.85)
    fig.suptitle(f"{ssp.label}: metallicity x wavelength", y=1.02)
    path = out_dir / "dimension_metallicity_wavelength_heatmaps.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_surviving_mass(ssp: SspData, out_dir: Path, args: argparse.Namespace) -> Path:
    assert ssp.surviving_mstar is not None
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for met_idx in _representative_metallicity_indices(ssp):
        ax.plot(
            ssp.age_gyr,
            ssp.surviving_mstar[met_idx],
            lw=1.2,
            label=f"logZ={ssp.lgmet[met_idx]:.2f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("age [Gyr]")
    ax.set_ylabel("surviving stellar mass / formed mass")
    ax.set_title(f"{ssp.label}: surviving mass auxiliary grid")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "surviving_mass_by_age_metallicity.png"
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    return path


def plot_pairwise_ratio(
    base: SspData, other: SspData, out_dir: Path, args: argparse.Namespace
) -> Path:
    if base.flux.shape != other.flux.shape or not np.allclose(base.wave, other.wave):
        return out_dir / "pairwise_ratio_skipped_shape_mismatch.txt"
    wave_mask = _wave_mask(base.wave, args.wave_min, args.wave_max)
    wave = base.wave[wave_mask]
    met_indices = _representative_metallicity_indices(base)
    age_indices = _nearest_indices(base.age_gyr, (0.001, 0.01, 0.1, 1.0, 10.0))
    floor = min(_positive_floor(base.flux), _positive_floor(other.flux))

    fig, axes = plt.subplots(len(met_indices), 1, figsize=(11.0, 9.0), sharex=True)
    if len(met_indices) == 1:
        axes = [axes]
    for ax, met_idx in zip(axes, met_indices, strict=True):
        for age_idx in age_indices:
            numerator = np.clip(other.flux[met_idx, age_idx, wave_mask], floor, np.inf)
            denominator = np.clip(base.flux[met_idx, age_idx, wave_mask], floor, np.inf)
            log_num = np.log10(numerator)
            log_den = np.log10(denominator)
            valid = (log_num >= np.nanmax(log_num) - 8.0) & (
                log_den >= np.nanmax(log_den) - 8.0
            )
            ratio = np.full_like(log_num, np.nan)
            ratio[valid] = log_num[valid] - log_den[valid]
            ax.plot(
                wave,
                ratio,
                lw=0.9,
                label=f"{base.age_gyr[age_idx]:.3g} Gyr",
            )
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.6)
        ax.set_xscale("log")
        ax.set_ylabel("Delta log10 Lnu")
        ax.set_title(f"log10(Z)={base.lgmet[met_idx]:.3f}")
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("rest wavelength [Angstrom]")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle(f"{other.label} relative to {base.label}", y=0.995)
    fig.tight_layout()
    path = out_dir / f"pairwise_log_ratio_{other.label}_over_{base.label}.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def compression_diagnostics(
    ssp: SspData, out_dir: Path, args: argparse.Namespace
) -> dict[str, Any]:
    print(f"[ssp] compression diagnostics for {ssp.label}")
    x_grid, y_grid, metadata = resample_log_flux(ssp, args)
    relevant = relevant_flux_mask(y_grid)
    interpolation = interpolation_diagnostics(x_grid, y_grid, relevant)
    svd = svd_diagnostics(x_grid, y_grid, relevant)
    quadratic = quadratic_diagnostics(x_grid, y_grid, relevant)
    haar = haar_diagnostics(x_grid, y_grid, relevant)
    plot_compression_tradeoffs(interpolation, svd, quadratic, haar, out_dir, args)
    plot_reconstruction_examples(
        ssp, x_grid, y_grid, interpolation, svd, quadratic, haar, out_dir, args
    )

    metrics = {
        "resampled_log_wave": metadata,
        "log_wave_linear_interpolation": _jsonable_metrics(interpolation),
        "svd": _jsonable_metrics(svd),
        "piecewise_quadratic": _jsonable_metrics(quadratic),
        "haar_wavelet_proxy": _jsonable_metrics(haar),
    }
    metrics_path = out_dir / "compression_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def resample_log_flux(
    ssp: SspData, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    wave_mask = _wave_mask(ssp.wave, args.wave_min, args.wave_max)
    wave = ssp.wave[wave_mask]
    log_wave = np.log10(wave)
    x_grid = np.linspace(log_wave[0], log_wave[-1], args.log_grid_size)
    floor = _positive_floor(ssp.flux[:, :, wave_mask])
    log_flux = np.log10(np.clip(ssp.flux[:, :, wave_mask], floor, np.inf))
    flat = log_flux.reshape(-1, log_flux.shape[-1])
    y_grid = np.empty((flat.shape[0], x_grid.size), dtype=np.float32)
    for i, curve in enumerate(flat):
        y_grid[i] = np.interp(x_grid, log_wave, curve)
    metadata = {
        "n_curves": int(y_grid.shape[0]),
        "n_grid": int(y_grid.shape[1]),
        "wave_min_angstrom": float(10.0**x_grid[0]),
        "wave_max_angstrom": float(10.0**x_grid[-1]),
        "source_wave_points": int(np.count_nonzero(wave_mask)),
        "metric_excludes_points_fainter_than_peak_minus_dex": 8.0,
    }
    return x_grid, y_grid, metadata


def interpolation_diagnostics(
    x_grid: np.ndarray, y_grid: np.ndarray, relevant: np.ndarray
) -> dict[str, Any]:
    rows = []
    recon_by_knots: dict[int, np.ndarray] = {}
    n_curves, n_wave = y_grid.shape
    raw_values = y_grid.size
    for n_knots in INTERP_KNOTS:
        if n_knots >= n_wave:
            continue
        knot_index = np.unique(np.rint(np.linspace(0, n_wave - 1, n_knots)).astype(int))
        x_knots = x_grid[knot_index]
        y_knots = y_grid[:, knot_index]
        recon = np.empty_like(y_grid)
        for index, curve in enumerate(y_knots):
            recon[index] = np.interp(x_grid, x_knots, curve)
        recon_by_knots[int(knot_index.size)] = recon
        stats = error_stats(y_grid, recon, relevant)
        stored_values = knot_index.size * (n_curves + 1)
        rows.append(
            {
                "knots": int(knot_index.size),
                "nominal_compression": float(raw_values / stored_values),
                "representation": "store log-flux values at shared log-wavelength knots and linearly interpolate",
                **stats,
            }
        )
    return {
        "rows": rows,
        "reconstruction_by_knots": recon_by_knots,
        "x_grid": x_grid,
    }


def svd_diagnostics(
    x_grid: np.ndarray, y_grid: np.ndarray, relevant: np.ndarray
) -> dict[str, Any]:
    mean = y_grid.mean(axis=0, keepdims=True)
    centered = y_grid - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = np.cumsum(singular**2) / np.sum(singular**2)
    rows = []
    recon_by_k: dict[int, np.ndarray] = {}
    raw_values = y_grid.size
    n_curves, n_wave = y_grid.shape
    for k in SVD_K:
        if k > singular.size:
            continue
        recon = (u[:, :k] * singular[:k]) @ vt[:k] + mean
        recon_by_k[k] = recon.astype(np.float32)
        stats = error_stats(y_grid, recon, relevant)
        stored_values = n_wave + k * (n_curves + n_wave + 1)
        rows.append(
            {
                "k": int(k),
                "energy": float(energy[k - 1]),
                "nominal_compression": float(raw_values / stored_values),
                **stats,
            }
        )
    return {
        "rows": rows,
        "singular_values": singular.astype(float).tolist(),
        "energy": energy.astype(float).tolist(),
        "mean": mean.astype(np.float32),
        "vt": vt.astype(np.float32),
        "u": u.astype(np.float32),
        "singular_array": singular.astype(np.float32),
        "reconstruction_by_k": recon_by_k,
        "x_grid": x_grid,
    }


def quadratic_diagnostics(
    x_grid: np.ndarray, y_grid: np.ndarray, relevant: np.ndarray
) -> dict[str, Any]:
    rows = []
    recon_by_segments: dict[int, np.ndarray] = {}
    for n_segments in QUADRATIC_SEGMENTS:
        if n_segments >= y_grid.shape[1] // 3:
            continue
        recon = piecewise_quadratic_reconstruct(y_grid, n_segments)
        recon_by_segments[n_segments] = recon
        stats = error_stats(y_grid, recon, relevant)
        rows.append(
            {
                "segments": int(n_segments),
                "coefficients_per_curve": int(3 * n_segments),
                "nominal_compression": float(y_grid.shape[1] / (3 * n_segments)),
                **stats,
            }
        )
    return {
        "rows": rows,
        "reconstruction_by_segments": recon_by_segments,
        "x_grid": x_grid,
    }


def piecewise_quadratic_reconstruct(
    y_grid: np.ndarray, n_segments: int
) -> np.ndarray:
    n_wave = y_grid.shape[1]
    edges = np.linspace(0, n_wave, n_segments + 1, dtype=int)
    recon = np.empty_like(y_grid)
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        length = end - start
        if length <= 0:
            continue
        if length < 3:
            recon[:, start:end] = y_grid[:, start:end]
            continue
        t = np.linspace(-1.0, 1.0, length, dtype=np.float64)
        design = np.column_stack([np.ones(length), t, t**2])
        coeff = np.linalg.pinv(design) @ y_grid[:, start:end].T
        recon[:, start:end] = (design @ coeff).T.astype(np.float32)
    return recon


def haar_diagnostics(
    x_grid: np.ndarray, y_grid: np.ndarray, relevant: np.ndarray
) -> dict[str, Any]:
    coeff = haar_transform(y_grid)
    rows = []
    recon_by_keep: dict[int, np.ndarray] = {}
    raw_values = y_grid.shape[1]
    for keep in HAAR_KEEP:
        if keep > coeff.shape[1]:
            continue
        sparse = keep_largest_per_row(coeff, keep)
        recon = haar_inverse(sparse)
        recon_by_keep[keep] = recon
        stats = error_stats(y_grid, recon, relevant)
        rows.append(
            {
                "coefficients_per_curve": int(keep),
                "nominal_compression_values_only": float(raw_values / keep),
                "approx_compression_float32_plus_uint16_index": float(
                    (4 * raw_values) / (6 * keep)
                ),
                **stats,
            }
        )
    return {
        "rows": rows,
        "coefficients": coeff,
        "reconstruction_by_keep": recon_by_keep,
        "x_grid": x_grid,
    }


def haar_transform(values: np.ndarray) -> np.ndarray:
    coeff = values.astype(np.float32, copy=True)
    length = coeff.shape[1]
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    while length > 1:
        half = length // 2
        even = coeff[:, :length:2]
        odd = coeff[:, 1:length:2]
        avg = (even + odd) * inv_sqrt2
        diff = (even - odd) * inv_sqrt2
        coeff[:, :half] = avg
        coeff[:, half:length] = diff
        length = half
    return coeff


def haar_inverse(coefficients: np.ndarray) -> np.ndarray:
    values = coefficients.astype(np.float32, copy=True)
    length = 1
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    n_wave = values.shape[1]
    while length < n_wave:
        avg = values[:, :length].copy()
        diff = values[:, length : 2 * length].copy()
        values[:, : 2 * length : 2] = (avg + diff) * inv_sqrt2
        values[:, 1 : 2 * length : 2] = (avg - diff) * inv_sqrt2
        length *= 2
    return values


def keep_largest_per_row(coeff: np.ndarray, keep: int) -> np.ndarray:
    sparse = np.zeros_like(coeff)
    if keep >= coeff.shape[1]:
        return coeff.copy()
    idx = np.argpartition(np.abs(coeff), -keep, axis=1)[:, -keep:]
    rows = np.arange(coeff.shape[0])[:, None]
    sparse[rows, idx] = coeff[rows, idx]
    return sparse


def plot_compression_tradeoffs(
    interpolation: dict[str, Any],
    svd: dict[str, Any],
    quadratic: dict[str, Any],
    haar: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), sharey=True)
    for ax, rows, x_key, title in (
        (
            axes[0, 0],
            interpolation["rows"],
            "nominal_compression",
            "log-wave knots + linear interpolation",
        ),
        (axes[0, 1], svd["rows"], "nominal_compression", "SVD shared basis"),
        (
            axes[1, 0],
            quadratic["rows"],
            "nominal_compression",
            "piecewise quadratic",
        ),
        (
            axes[1, 1],
            haar["rows"],
            "approx_compression_float32_plus_uint16_index",
            "Haar proxy",
        ),
    ):
        x = [row[x_key] for row in rows]
        median = [row["median_abs_dex"] for row in rows]
        p95 = [row["p95_abs_dex"] for row in rows]
        p99 = [row["p99_abs_dex"] for row in rows]
        ax.plot(x, median, marker="o", label="median")
        ax.plot(x, p95, marker="o", label="p95")
        ax.plot(x, p99, marker="o", label="p99")
        ax.axhline(0.01, color="black", lw=0.7, alpha=0.35)
        ax.axhline(0.05, color="black", lw=0.7, alpha=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("approx. compression ratio")
        ax.set_title(title)
        ax.grid(alpha=0.22)
    axes[0, 0].set_ylabel("absolute log10 flux error [dex]")
    axes[1, 0].set_ylabel("absolute log10 flux error [dex]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Compression error tradeoffs on selected log-wavelength grid", y=1.02)
    fig.tight_layout()
    path = out_dir / "compression_error_tradeoffs.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for label, rows, ratio_key, marker in (
        ("log-wave knots", interpolation["rows"], "nominal_compression", "o"),
        ("SVD", svd["rows"], "nominal_compression", "s"),
        ("quadratic", quadratic["rows"], "nominal_compression", "^"),
        ("Haar", haar["rows"], "approx_compression_float32_plus_uint16_index", "D"),
    ):
        ratio = [row[ratio_key] for row in rows]
        p95 = [row["p95_abs_dex"] for row in rows]
        ax.plot(ratio, p95, marker=marker, lw=1.4, label=label)
    ax.axhline(0.01, color="black", lw=0.8, alpha=0.35, label="0.01 dex")
    ax.axhline(0.05, color="black", lw=0.8, alpha=0.2, label="0.05 dex")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("approx. compression ratio")
    ax.set_ylabel("p95 absolute log10 flux error [dex]")
    ax.set_title("Compression frontier")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    frontier_path = out_dir / "compression_frontier_p95.png"
    fig.savefig(frontier_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    energy = np.asarray(svd["energy"], dtype=float)
    ax.plot(np.arange(1, energy.size + 1), energy, lw=1.2)
    ax.scatter([row["k"] for row in svd["rows"]], [row["energy"] for row in svd["rows"]])
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.001)
    ax.set_xlabel("SVD components")
    ax.set_ylabel("cumulative variance fraction")
    ax.set_title("SVD energy in log10 SSP flux")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    energy_path = out_dir / "svd_energy.png"
    fig.savefig(energy_path, dpi=args.dpi)
    plt.close(fig)
    return path


def plot_reconstruction_examples(
    ssp: SspData,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    interpolation: dict[str, Any],
    svd: dict[str, Any],
    quadratic: dict[str, Any],
    haar: dict[str, Any],
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    wave = np.power(10.0, x_grid)
    examples = [
        (
            _nearest_index(ssp.lgmet, np.nanmin(ssp.lgmet)),
            _nearest_index(ssp.age_gyr, 0.01),
        ),
        (
            _nearest_index(ssp.lgmet, np.log10(float(ssp.attrs.get("z_sun", 0.0142)))),
            _nearest_index(ssp.age_gyr, 1.0),
        ),
        (
            _nearest_index(ssp.lgmet, np.nanmax(ssp.lgmet)),
            _nearest_index(ssp.age_gyr, 10.0),
        ),
    ]
    interp_knots = _available_key(interpolation["reconstruction_by_knots"], 512)
    svd_k = _available_key(svd["reconstruction_by_k"], 16)
    quad_segments = _available_key(quadratic["reconstruction_by_segments"], 128)
    haar_keep = _available_key(haar["reconstruction_by_keep"], 256)
    recon_specs = [
        (
            f"interp knots={interp_knots}",
            interpolation["reconstruction_by_knots"][interp_knots],
        ),
        (f"SVD k={svd_k}", svd["reconstruction_by_k"][svd_k]),
        (
            f"quadratic seg={quad_segments}",
            quadratic["reconstruction_by_segments"][quad_segments],
        ),
        (f"Haar keep={haar_keep}", haar["reconstruction_by_keep"][haar_keep]),
    ]

    fig, axes = plt.subplots(len(examples), 2, figsize=(12.0, 8.6), sharex=True)
    for row, (met_idx, age_idx) in enumerate(examples):
        curve_idx = met_idx * len(ssp.lg_age_gyr) + age_idx
        ax_flux = axes[row, 0]
        ax_resid = axes[row, 1]
        original = y_grid[curve_idx]
        ax_flux.plot(wave, original, color="black", lw=1.4, label="original")
        for label, recon in recon_specs:
            ax_flux.plot(wave, recon[curve_idx], lw=0.9, alpha=0.85, label=label)
            ax_resid.plot(wave, recon[curve_idx] - original, lw=0.9, label=label)
        title = f"logZ={ssp.lgmet[met_idx]:.2f}, age={ssp.age_gyr[age_idx]:.3g} Gyr"
        ax_flux.set_title(title)
        ax_flux.set_ylabel("log10 Lnu / Msun")
        ax_flux.set_xscale("log")
        ax_flux.grid(alpha=0.22)
        ax_resid.axhline(0.0, color="black", lw=0.7, alpha=0.5)
        ax_resid.axhline(0.01, color="black", lw=0.6, alpha=0.25)
        ax_resid.axhline(-0.01, color="black", lw=0.6, alpha=0.25)
        ax_resid.set_xscale("log")
        ax_resid.set_ylabel("residual dex")
        ax_resid.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    axes[-1, 0].set_xlabel("rest wavelength [Angstrom]")
    axes[-1, 1].set_xlabel("rest wavelength [Angstrom]")
    fig.suptitle(f"{ssp.label}: reconstruction examples", y=1.01)
    fig.tight_layout()
    path = out_dir / "compression_reconstruction_examples.png"
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def error_stats(
    y_true: np.ndarray, y_pred: np.ndarray, relevant: np.ndarray
) -> dict[str, float]:
    err = np.abs(y_pred - y_true)
    values = err[relevant & np.isfinite(err)]
    if values.size == 0:
        return {
            "median_abs_dex": float("nan"),
            "p95_abs_dex": float("nan"),
            "p99_abs_dex": float("nan"),
            "max_abs_dex": float("nan"),
            "frac_gt_0p01_dex": float("nan"),
            "frac_gt_0p05_dex": float("nan"),
        }
    return {
        "median_abs_dex": float(np.nanmedian(values)),
        "p95_abs_dex": float(np.nanpercentile(values, 95)),
        "p99_abs_dex": float(np.nanpercentile(values, 99)),
        "max_abs_dex": float(np.nanmax(values)),
        "frac_gt_0p01_dex": float(np.mean(values > 0.01)),
        "frac_gt_0p05_dex": float(np.mean(values > 0.05)),
    }


def relevant_flux_mask(y_grid: np.ndarray) -> np.ndarray:
    row_peak = np.nanmax(y_grid, axis=1, keepdims=True)
    return np.isfinite(y_grid) & (y_grid >= row_peak - 8.0)


def write_report(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    compression: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    lines = [
        "# SSP Shape Investigation",
        "",
        f"Wave range used for compression diagnostics: {args.wave_min:g}-{args.wave_max:g} Angstrom.",
        f"Compression grid: {args.log_grid_size} uniformly spaced log-wavelength samples.",
        "Errors are measured in log10 flux space and ignore samples more than 8 dex below each curve peak.",
        "",
        "## Assets",
        "",
        "| label | shape | selected wave points | file size | raw flux size | nebular |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            "| {label} | {shape} | {wave_points_selected} / {wave_points_total} | "
            "{file_size} | {raw_size} | {nebular} |".format(
                label=item["label"],
                shape="x".join(str(v) for v in item["shape"]),
                wave_points_selected=item["wave_points_selected"],
                wave_points_total=item["wave_points_total"],
                file_size=_format_bytes(item["hdf5_file_bytes"]),
                raw_size=_format_bytes(item["raw_flux_bytes"]),
                nebular=f"em={item['add_neb_emission']}, cont={item['add_neb_continuum']}",
            )
        )
    lines.extend(["", "## Compression Snapshot", ""])
    for label, metrics in compression.items():
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            _best_threshold_sentence(
                "Log-wave interpolation",
                metrics["log_wave_linear_interpolation"]["rows"],
                "knots",
            )
        )
        lines.append(_best_threshold_sentence("SVD", metrics["svd"]["rows"], "k"))
        lines.append(
            _best_threshold_sentence(
                "Piecewise quadratic",
                metrics["piecewise_quadratic"]["rows"],
                "segments",
            )
        )
        lines.append(
            _best_threshold_sentence(
                "Haar proxy",
                metrics["haar_wavelet_proxy"]["rows"],
                "coefficients_per_curve",
            )
        )
        lines.append("")
        lines.append("| method | setting | compression | median dex | p95 dex | p99 dex |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method, rows, setting_key, ratio_key in (
            (
                "log-wave interp",
                metrics["log_wave_linear_interpolation"]["rows"],
                "knots",
                "nominal_compression",
            ),
            ("SVD", metrics["svd"]["rows"], "k", "nominal_compression"),
            (
                "quadratic",
                metrics["piecewise_quadratic"]["rows"],
                "segments",
                "nominal_compression",
            ),
            (
                "Haar",
                metrics["haar_wavelet_proxy"]["rows"],
                "coefficients_per_curve",
                "approx_compression_float32_plus_uint16_index",
            ),
        ):
            for row in rows:
                lines.append(
                    f"| {method} | {row[setting_key]} | {row[ratio_key]:.2f}x | "
                    f"{row['median_abs_dex']:.4g} | {row['p95_abs_dex']:.4g} | "
                    f"{row['p99_abs_dex']:.4g} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Practical Reading",
            "",
            "- Pure stellar SSPs are smooth enough that shared low-rank bases are the first compression candidate to test for broad-band photometry.",
            "- Storing fewer log-wavelength knots and interpolating is the simplest implementation path, but it needs many knots once nebular lines are included.",
            "- Fixed-nebular SSPs contain narrow line spikes; treating lines as a separate sparse component is safer than forcing a continuum-only polynomial model to fit them.",
            "- Piecewise quadratics in log wavelength are useful as a baseline, but their error is dominated by sharp spectral structure unless many segments are kept.",
            "- The Haar proxy is intentionally simple. If it wins for a target tolerance, a real wavelet codec with shared metadata and quantization would be worth testing next.",
            "- A compressed representation only saves VRAM if the JAX model consumes it directly. Decoding the full SSP/gas tensor onto the device before fitting recovers disk space but not batch-size headroom.",
        ]
    )
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _best_threshold_sentence(method: str, rows: list[dict[str, Any]], key: str) -> str:
    parts = []
    for threshold in (0.01, 0.02, 0.05):
        match = next((row for row in rows if row["p95_abs_dex"] <= threshold), None)
        if match is None:
            parts.append(f"p95<={threshold:g} dex: not reached")
        else:
            ratio = match.get("nominal_compression")
            if ratio is None:
                ratio = match.get("approx_compression_float32_plus_uint16_index")
            parts.append(
                f"p95<={threshold:g} dex: {key}={match[key]} ({ratio:.2f}x)"
            )
    return f"- {method}: " + "; ".join(parts) + "."


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key
        not in {
            "mean",
            "vt",
            "u",
            "singular_array",
            "reconstruction_by_knots",
            "reconstruction_by_k",
            "reconstruction_by_segments",
            "reconstruction_by_keep",
            "coefficients",
            "x_grid",
        }
    }


def _configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
        }
    )


def _wave_mask(wave: np.ndarray, wave_min: float, wave_max: float) -> np.ndarray:
    return np.isfinite(wave) & (wave >= wave_min) & (wave <= wave_max)


def _positive_floor(values: np.ndarray) -> float:
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size == 0:
        return float(np.finfo(np.float32).tiny)
    return float(max(np.nanpercentile(finite_positive, 0.001) * 1.0e-4, 1.0e-45))


def _safe_log10(values: np.ndarray) -> np.ndarray:
    floor = _positive_floor(values)
    return np.log10(np.clip(values, floor, np.inf))


def _representative_metallicity_indices(ssp: SspData) -> list[int]:
    z_sun = float(ssp.attrs.get("z_sun") or 0.0142)
    targets = (float(np.nanmin(ssp.lgmet)), math.log10(z_sun), float(np.nanmax(ssp.lgmet)))
    return _nearest_indices(ssp.lgmet, targets)


def _nearest_indices(values: np.ndarray, targets: tuple[float, ...]) -> list[int]:
    indices: list[int] = []
    for target in targets:
        idx = _nearest_index(values, target)
        if idx not in indices:
            indices.append(idx)
    return indices


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(values - target)))


def _log_edges(values: np.ndarray) -> np.ndarray:
    log_values = np.log10(np.asarray(values, dtype=float))
    edges = _linear_edges(log_values)
    return np.power(10.0, edges)


def _linear_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        delta = max(abs(float(values[0])) * 0.05, 1.0)
        return np.asarray([values[0] - delta, values[0] + delta])
    mid = 0.5 * (values[1:] + values[:-1])
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _shared_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    merged = np.concatenate([np.ravel(a[np.isfinite(a)]) for a in arrays])
    return (float(np.nanpercentile(merged, 2)), float(np.nanpercentile(merged, 98)))


def _label_from_path(path: Path, attrs: dict[str, Any]) -> str:
    stem = path.stem
    asset_kind = str(attrs.get("asset_kind", ""))
    if "chabrier_noNE" in stem or "stellar_only" in asset_kind:
        return "chabrier_noNE"
    if "chabrier_wNE" in stem or "base_ssp" in asset_kind:
        match = re.search(r"logGasU([+-]?\d+(?:\.\d+)?)", stem)
        suffix = f"_gasU{match.group(1)}" if match else ""
        return f"chabrier_wNE{suffix}".replace(".", "p").replace("-", "m")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return safe[:80]


def _jsonable_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _format_bytes(value: int | None) -> str:
    if value is None:
        return ""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def _available_key(mapping: dict[int, np.ndarray], preferred: int) -> int:
    if preferred in mapping:
        return preferred
    return sorted(mapping)[len(mapping) // 2]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


if __name__ == "__main__":
    main()
