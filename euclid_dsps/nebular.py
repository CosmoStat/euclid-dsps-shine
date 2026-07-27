"""Nebular-line diagnostics for SSP assets and fitted redshifts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import ensure_dir, write_json


def write_nebular_diagnostic_outputs(
    context: Any,
    fits: pd.DataFrame,
    out_dir: str | Path,
    label: str = "batch_fit",
    top_n_lines: int = 25,
    max_modes: int = 12,
    make_plots: bool = True,
) -> None:
    """Write diagnostic-only nebular tables.

    This does not alter the DSPS likelihood. It only connects fitted-redshift
    attractors to strong emission-line/filter crossings.
    """
    out = ensure_dir(out_dir)
    inventory = emline_inventory(context)
    summary = {
        "nebular_emission_mode": getattr(context, "nebular_emission_mode", "ssp_flux"),
        "has_emline_luminosity": bool(
            getattr(context, "ssp_emline_luminosity", None) is not None
        ),
        "has_emline_wave": bool(getattr(context, "ssp_emline_wave", None) is not None),
        "n_lines": int(len(inventory)),
        "forward_model_note": (
            "Current DSPS photometry uses ssp_flux. emline_table diagnostics are "
            "not added to the likelihood, to avoid double-counting lines that may "
            "already be present in ssp_flux."
        ),
    }
    write_json(out / f"{label}_nebular_summary.json", summary)
    if inventory.empty:
        return
    inventory.to_csv(out / f"{label}_nebular_line_inventory.csv", index=False)
    modes = fitted_redshift_modes(fits, max_modes=max_modes)
    if modes.empty:
        return
    modes.to_csv(out / f"{label}_nebular_redshift_modes.csv", index=False)
    crossings = line_filter_crossings(
        context, inventory, modes, top_n_lines=top_n_lines
    )
    if crossings.empty:
        return
    crossings.to_csv(out / f"{label}_nebular_line_filter_crossings.csv", index=False)
    if make_plots:
        plot_line_filter_crossings(
            crossings, context, out / f"{label}_nebular_line_filter_crossings.png"
        )


def emline_inventory(context: Any) -> pd.DataFrame:
    luminosity = getattr(context, "ssp_emline_luminosity", None)
    wave = getattr(context, "ssp_emline_wave", None)
    if luminosity is None or wave is None:
        return pd.DataFrame()
    luminosity = np.asarray(luminosity, dtype=float)
    wave = np.asarray(wave, dtype=float)
    if luminosity.ndim < 1 or len(wave) != luminosity.shape[-1]:
        return pd.DataFrame()
    names = tuple(getattr(context, "ssp_emline_name", ()) or ())
    if len(names) != len(wave):
        names = tuple(f"line_{i:03d}" for i in range(len(wave)))
    strength = np.nanmax(np.abs(luminosity.reshape(-1, luminosity.shape[-1])), axis=0)
    rows = [
        {
            "line_index": int(i),
            "line_name": str(names[i]),
            "rest_wavelength_angstrom": float(wave[i]),
            "relative_strength": float(strength[i]),
        }
        for i in range(len(wave))
        if np.isfinite(wave[i])
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values(
        ["relative_strength", "rest_wavelength_angstrom"],
        ascending=[False, True],
    )
    frame["strength_rank"] = np.arange(1, len(frame) + 1)
    return frame.reset_index(drop=True)


def fitted_redshift_modes(
    fits: pd.DataFrame,
    bin_width: float = 0.05,
    min_count: int = 3,
    max_modes: int = 12,
) -> pd.DataFrame:
    column = "fit_z_obs" if "fit_z_obs" in fits.columns else "z_obs"
    if fits.empty or column not in fits:
        return pd.DataFrame()
    z = pd.to_numeric(fits[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    z = z.dropna()
    if z.empty:
        return pd.DataFrame()
    z_bin = (z / bin_width).round() * bin_width
    grouped = z.groupby(z_bin)
    rows = []
    for mode, values in grouped:
        if len(values) < min_count:
            continue
        rows.append(
            {
                "z_fit_bin": float(mode),
                "n_galaxies": int(len(values)),
                "z_fit_median": float(values.median()),
                "z_fit_min": float(values.min()),
                "z_fit_max": float(values.max()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["n_galaxies", "z_fit_bin"], ascending=[False, True])
        .head(max_modes)
        .reset_index(drop=True)
    )


def line_filter_crossings(
    context: Any,
    inventory: pd.DataFrame,
    modes: pd.DataFrame,
    top_n_lines: int = 25,
    transmission_threshold: float = 0.03,
) -> pd.DataFrame:
    if inventory.empty or modes.empty:
        return pd.DataFrame()
    lines = inventory.head(top_n_lines)
    rows: list[dict[str, Any]] = []
    for _, mode in modes.iterrows():
        z = float(mode["z_fit_bin"])
        for _, line in lines.iterrows():
            observed_wave = float(line["rest_wavelength_angstrom"]) * (1.0 + z)
            for band, curve in getattr(context, "filters", {}).items():
                transmission = _normalized_filter_transmission(curve, observed_wave)
                if transmission < transmission_threshold:
                    continue
                rows.append(
                    {
                        "z_fit_bin": z,
                        "n_galaxies_in_mode": int(mode["n_galaxies"]),
                        "line_index": int(line["line_index"]),
                        "line_name": str(line["line_name"]),
                        "line_strength_rank": int(line["strength_rank"]),
                        "line_relative_strength": float(line["relative_strength"]),
                        "rest_wavelength_angstrom": float(
                            line["rest_wavelength_angstrom"]
                        ),
                        "observed_wavelength_angstrom": observed_wave,
                        "band": band,
                        "filter_transmission_norm": transmission,
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["z_fit_bin", "line_strength_rank", "filter_transmission_norm"],
        ascending=[True, True, False],
    )


def plot_line_filter_crossings(
    crossings: pd.DataFrame, context: Any, path: str | Path
) -> None:
    if crossings.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    filters = getattr(context, "filters", {})
    for band, curve in filters.items():
        wave = np.asarray(curve.wave, dtype=float)
        transmission = np.asarray(curve.transmission, dtype=float)
        good = np.isfinite(wave) & np.isfinite(transmission) & (transmission > 0)
        if good.any():
            ax.axhspan(
                float(wave[good].min()),
                float(wave[good].max()),
                alpha=0.05,
                label=band,
            )
    for band, group in crossings.groupby("band"):
        ax.scatter(
            group["z_fit_bin"],
            group["observed_wavelength_angstrom"],
            s=12 + 2 * np.sqrt(group["n_galaxies_in_mode"]),
            alpha=0.65,
            label=f"{band} line",
        )
    ax.set_xlabel("fitted-redshift attractor bin")
    ax.set_ylabel("observed line wavelength [Angstrom]")
    ax.set_title("Strong SSP emission-line crossings near fitted-z attractors")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles, strict=False))
    ax.legend(dedup.values(), dedup.keys(), fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _normalized_filter_transmission(curve: Any, wavelength: float) -> float:
    wave = np.asarray(curve.wave, dtype=float)
    transmission = np.asarray(curve.transmission, dtype=float)
    good = np.isfinite(wave) & np.isfinite(transmission)
    if good.sum() < 2:
        return 0.0
    wave = wave[good]
    transmission = transmission[good]
    max_transmission = float(np.nanmax(transmission))
    if max_transmission <= 0.0:
        return 0.0
    if wavelength < float(wave.min()) or wavelength > float(wave.max()):
        return 0.0
    value = float(np.interp(wavelength, wave, transmission))
    return max(value / max_transmission, 0.0)
