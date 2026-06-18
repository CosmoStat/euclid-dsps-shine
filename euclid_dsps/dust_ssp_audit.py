"""Dust/SSP consistency diagnostics for PopCosmos-like DSPS configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .filters import load_filters
from .io import ensure_dir
from .model import (
    _prospector_fsps_diffuse_shape_jax,
    load_context,
)
from .parameter_vectors import free_parameter_bounds_from_config

_DUST_PARAMS = ("tau2", "dust_index_n", "tau1_over_tau2")


def write_dust_ssp_audit(config: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Write dust-parameter and SSP-grid diagnostics for a configured model."""
    out = ensure_dir(out_dir)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    wave = np.asarray(jax.device_get(context.ssp_wave_jax), dtype=float)
    lg_age = np.asarray(jax.device_get(context.ssp_lg_age_gyr_jax), dtype=float)
    bounds = _dust_bounds(config)
    payload = {
        "dust_model": str((config.get("model", {}) or {}).get("dust_model")),
        "ssp_path": str(config.get("ssp_path")),
        "compressed_ssp_path": str((config.get("model", {}) or {}).get("compressed_ssp_path")),
        "ssp_wave_angstrom": _array_summary(wave),
        "ssp_log_age_gyr": _array_summary(lg_age),
        "ssp_age_logyr": _array_summary(lg_age + 9.0),
        "dust_tesc_logyr": float((config.get("model", {}) or {}).get("dust_tesc_logyr", 7.0)),
        "dust1_index": float((config.get("model", {}) or {}).get("dust1_index", -1.0)),
        "parameter_bounds": bounds,
        "notes": [
            "tau2 is diffuse V-band optical depth.",
            "dust_index_n modifies the diffuse Calzetti/Noll/Drude attenuation shape.",
            "tau1_over_tau2 multiplies tau2 for the young-star birth-cloud component.",
        ],
    }
    (out / "dust_ssp_audit.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    curves = _dust_curve_grid(
        wave,
        bounds,
        dust_tesc_logyr=payload["dust_tesc_logyr"],
        dust1_index=payload["dust1_index"],
    )
    curves.to_csv(out / "dust_transmission_grid.csv", index=False)
    _write_dust_plots(curves, out)
    _write_report(payload, out / "dust_ssp_audit.md")
    return payload


def _dust_bounds(config: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    names = tuple((config.get("fit", {}) or {}).get("free_parameters", {}))
    lower, upper = free_parameter_bounds_from_config(config, names)
    by_name = {
        name: (float(lower[index]), float(upper[index]))
        for index, name in enumerate(names)
    }
    fixed = ((config.get("model", {}) or {}).get("fixed_parameters", {}) or {})
    defaults = {"tau2": 0.3, "dust_index_n": -0.7, "tau1_over_tau2": 1.0}
    payload: dict[str, dict[str, float | str]] = {}
    for name in _DUST_PARAMS:
        if name in by_name:
            lo, hi = by_name[name]
            payload[name] = {
                "mode": "free",
                "lower": lo,
                "mid": 0.5 * (lo + hi),
                "upper": hi,
            }
        else:
            value = float(fixed.get(name, defaults[name]))
            payload[name] = {
                "mode": "fixed",
                "lower": value,
                "mid": value,
                "upper": value,
            }
    return payload


def _dust_curve_grid(
    wave: np.ndarray,
    bounds: dict[str, dict[str, float | str]],
    *,
    dust_tesc_logyr: float,
    dust1_index: float,
) -> pd.DataFrame:
    del dust_tesc_logyr
    quantiles = ("lower", "mid", "upper")
    sample_wave = np.geomspace(max(float(np.nanmin(wave)), 100.0), float(np.nanmax(wave)), 512)
    rows = []
    for label in quantiles:
        tau2 = float(bounds["tau2"][label])
        dust_index_n = float(bounds["dust_index_n"][label])
        tau1_over_tau2 = float(bounds["tau1_over_tau2"][label])
        diffuse_shape = np.asarray(
            jax.device_get(
                _prospector_fsps_diffuse_shape_jax(
                    jnp.asarray(sample_wave, dtype=jnp.float32),
                    jnp.asarray(dust_index_n, dtype=jnp.float32),
                )
            ),
            dtype=float,
        )
        diffuse_tau = max(tau2, 0.0) * diffuse_shape
        tau1 = max(tau1_over_tau2, 0.0) * max(tau2, 0.0)
        birth_tau = tau1 * (sample_wave / 5500.0) ** float(dust1_index)
        old_trans = np.exp(-np.clip(diffuse_tau, 0.0, 80.0))
        young_trans = np.exp(-np.clip(diffuse_tau + birth_tau, 0.0, 80.0))
        for idx, wave_value in enumerate(sample_wave):
            rows.append(
                {
                    "case": label,
                    "wave_angstrom": float(wave_value),
                    "tau2": tau2,
                    "dust_index_n": dust_index_n,
                    "tau1_over_tau2": tau1_over_tau2,
                    "diffuse_tau": float(diffuse_tau[idx]),
                    "birth_tau": float(birth_tau[idx]),
                    "old_transmission": float(old_trans[idx]),
                    "young_transmission": float(young_trans[idx]),
                }
            )
    return pd.DataFrame(rows)


def _array_summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _write_dust_plots(curves: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for case, group in curves.groupby("case", sort=False):
        axes[0].plot(
            group["wave_angstrom"],
            group["old_transmission"],
            label=str(case),
        )
        axes[1].plot(
            group["wave_angstrom"],
            group["young_transmission"],
            label=str(case),
        )
    for ax, title in zip(axes, ("old SSP transmission", "young SSP transmission"), strict=True):
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("rest wavelength [Angstrom]")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("transmission")
    axes[1].legend(title="bound case")
    fig.tight_layout()
    fig.savefig(out / "dust_transmission_grid.png", dpi=150)
    plt.close(fig)


def _write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Dust/SSP Audit",
        "",
        f"- Dust model: `{payload['dust_model']}`",
        f"- SSP path: `{payload['ssp_path']}`",
        f"- Compressed SSP path: `{payload['compressed_ssp_path']}`",
        f"- SSP wavelength Angstrom: `{payload['ssp_wave_angstrom']}`",
        f"- SSP log age Gyr: `{payload['ssp_log_age_gyr']}`",
        f"- dust_tesc_logyr: `{payload['dust_tesc_logyr']}`",
        f"- dust1_index: `{payload['dust1_index']}`",
        "",
        "## Parameters",
        "",
    ]
    for name, bounds in payload["parameter_bounds"].items():
        lines.append(f"- `{name}`: `{bounds}`")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in payload["notes"])
    path.write_text("\n".join(lines), encoding="utf-8")
