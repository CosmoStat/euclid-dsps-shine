"""Validation gates for synthetic Diffsky DSPS closure catalogs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir
from euclid_dsps.model import dynamic_model_args, load_context
from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
from euclid_dsps.photometry import abmag_to_fnu_cgs

from .config import DEFAULT_SPLIT_SIZES, SPLIT_ORDER
from .photometry import GROUND_TRUTH_COLUMNS, theta_from_truth_frame
from .resampling import effective_sample_size
from .selection import (
    apply_proposal_selection,
    normalize_selection,
    photometric_selection_enabled,
)


def validate_dsps_closure_dataset(
    config: dict[str, Any],
    *,
    dataset_dir: str | Path,
    sample_size: int = 256,
    batch_size: int = 256,
) -> Path:
    """Run closure dataset validation gates and write validation_report.json."""
    root = Path(dataset_dir)
    diagnostics = ensure_dir(root / "diagnostics")
    manifest = _read_manifest(root / "manifest.yaml")
    expected_sizes = _expected_sizes(manifest)
    bands = [str(band["name"]) for band in config["bands"]]
    split_frames: dict[str, pd.DataFrame] = {}
    report: dict[str, Any] = {
        "dataset_dir": str(root),
        "parameter_order": list(DIFFSKY_BASIC_PARAMETER_NAMES),
        "n_parameters": len(DIFFSKY_BASIC_PARAMETER_NAMES),
        "splits": {},
        "gates": {},
    }
    errors: list[str] = []
    for split in SPLIT_ORDER:
        path = root / f"{split}.parquet"
        if not path.exists():
            errors.append(f"missing split parquet: {path}")
            continue
        frame = pd.read_parquet(path)
        split_frames[split] = frame
        split_report = _validate_split_frame(
            split,
            frame,
            bands,
            expected_size=int(expected_sizes.get(split, DEFAULT_SPLIT_SIZES[split])),
        )
        report["splits"][split] = split_report
        errors.extend(split_report["errors"])
    errors.extend(_validate_disjoint_identity(split_frames))
    errors.extend(_validate_weighted_nz(root, split_frames))
    if split_frames:
        all_frame = pd.concat(split_frames.values(), ignore_index=True)
        report["noise"] = _noise_residual_report(all_frame, bands)
        if not report["noise"]["pass"]:
            errors.append(report["noise"]["message"])
        report["metallicity"] = _metallicity_population_report(
            all_frame,
            config,
            manifest,
        )
        if not report["metallicity"]["pass"]:
            errors.append(report["metallicity"]["message"])
        report["photometric_selection"] = _photometric_selection_report(
            all_frame,
            bands,
            config,
            manifest,
        )
        if not report["photometric_selection"]["pass"]:
            errors.append(report["photometric_selection"]["message"])
        recompute = _recompute_flux_report(
            config,
            all_frame,
            bands,
            sample_size=int(sample_size),
            batch_size=int(batch_size),
        )
        report["flux_recompute"] = recompute
        if not recompute["pass"]:
            errors.append(recompute["message"])
    report["gates"]["pass"] = not errors
    report["gates"]["errors"] = errors
    report_path = root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    (diagnostics / "validation_errors.txt").write_text(
        "\n".join(errors) + ("\n" if errors else ""),
        encoding="utf-8",
    )
    if errors:
        raise ValueError(
            "Synthetic Diffsky closure validation failed; see "
            f"{report_path}: " + "; ".join(errors[:5])
        )
    return report_path


def _validate_split_frame(
    split: str,
    frame: pd.DataFrame,
    bands: list[str],
    *,
    expected_size: int,
) -> dict[str, Any]:
    errors: list[str] = []
    if len(frame) != int(expected_size):
        errors.append(f"{split}: expected {expected_size} rows, found {len(frame)}")
    truth_columns = [GROUND_TRUTH_COLUMNS[name] for name in DIFFSKY_BASIC_PARAMETER_NAMES]
    missing_truth = [column for column in truth_columns if column not in frame.columns]
    if missing_truth:
        errors.append(f"{split}: missing truth columns {missing_truth}")
    else:
        truth = frame[truth_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(truth).all():
            errors.append(f"{split}: non-finite values in 18 truth columns")
    for band in bands:
        for prefix in ("flux_true", "flux", "fluxerr", "mask"):
            column = f"{prefix}_{band}"
            if column not in frame.columns:
                errors.append(f"{split}: missing {column}")
        if f"flux_true_{band}" in frame:
            values = pd.to_numeric(frame[f"flux_true_{band}"], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                errors.append(f"{split}: non-finite flux_true_{band}")
        if f"fluxerr_{band}" in frame and f"mask_{band}" in frame:
            err = pd.to_numeric(frame[f"fluxerr_{band}"], errors="coerce").to_numpy(float)
            mask = frame[f"mask_{band}"].astype(bool).to_numpy()
            if np.any(mask & (~np.isfinite(err) | (err <= 0.0))):
                errors.append(f"{split}: invalid positive fluxerr for valid {band}")
    return {
        "rows": int(len(frame)),
        "expected_rows": int(expected_size),
        "n_truth_columns": int(len(truth_columns) - len(missing_truth)),
        "errors": errors,
    }


def _validate_disjoint_identity(split_frames: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    object_sets = {
        split: set(frame["object_id"].astype(str))
        for split, frame in split_frames.items()
        if "object_id" in frame
    }
    proposal_sets = {
        split: set(frame["source_proposal_id"].astype(str))
        for split, frame in split_frames.items()
        if "source_proposal_id" in frame
    }
    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1:]:
            if object_sets.get(left, set()) & object_sets.get(right, set()):
                errors.append(f"object_id collision between {left} and {right}")
            if proposal_sets.get(left, set()) & proposal_sets.get(right, set()):
                errors.append(f"source_proposal_id shared between {left} and {right}")
    return errors


def _validate_weighted_nz(root: Path, split_frames: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    manifest = _read_manifest(root / "manifest.yaml")
    gen_cfg = dict(manifest.get("synthetic_diffsky", {}) or {})
    selection = normalize_selection(gen_cfg.get("selection", {}) or {})
    strict_nz_gate = not photometric_selection_enabled(selection)
    z_min = float(gen_cfg.get("z_min", 0.0))
    z_max = float(gen_cfg.get("z_max", 0.35))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_max <= z_min:
        z_min, z_max = 0.0, 0.35
    bins = np.linspace(z_min, z_max, 25)
    for split, final in split_frames.items():
        proposal_dir = root / "proposals" / split
        paths = sorted(proposal_dir.glob("*.parquet"))
        if not paths:
            errors.append(f"{split}: no proposal shards found for weighted n(z)")
            continue
        proposals = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        if not {"redshift_true", "galaxy_weight"} <= set(proposals.columns):
            errors.append(f"{split}: proposals missing redshift_true/galaxy_weight")
            continue
        try:
            proposals, _ = apply_proposal_selection(proposals, selection)
        except ValueError as exc:
            errors.append(f"{split}: proposal selection failed for n(z): {exc}")
            continue
        if len(proposals) == 0:
            errors.append(f"{split}: proposal selection leaves no rows for weighted n(z)")
            continue
        prop_z = proposals["redshift_true"].to_numpy(float)
        weights = proposals["galaxy_weight"].to_numpy(float)
        final_z = final["redshift_true"].to_numpy(float)
        prop_hist, _ = np.histogram(prop_z, bins=bins, weights=weights)
        final_hist, _ = np.histogram(final_z, bins=bins)
        if prop_hist.sum() <= 0 or final_hist.sum() <= 0:
            errors.append(f"{split}: empty n(z) histogram")
            continue
        prop_pdf = prop_hist / prop_hist.sum()
        final_pdf = final_hist / final_hist.sum()
        max_delta = float(np.max(np.abs(prop_pdf - final_pdf)))
        ess = effective_sample_size(weights)
        threshold = max(0.15, 4.0 / np.sqrt(max(len(final), 1)))
        if max_delta > threshold and strict_nz_gate:
            errors.append(
                f"{split}: weighted proposal n(z) and resampled n(z) differ by "
                f"{max_delta:.3g} > {threshold:.3g} (ESS={ess:.3g})"
            )
    return errors


def _noise_residual_report(frame: pd.DataFrame, bands: list[str]) -> dict[str, Any]:
    residuals = []
    for band in bands:
        residuals.append(
            (
                frame[f"flux_{band}"].to_numpy(float)
                - frame[f"flux_true_{band}"].to_numpy(float)
            )
            / frame[f"fluxerr_{band}"].to_numpy(float)
        )
    values = np.concatenate(residuals)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values)) if values.size else float("nan")
    std = float(np.std(values)) if values.size else float("nan")
    mean_threshold = max(0.05, 5.0 / np.sqrt(max(values.size, 1)))
    std_threshold = 0.15
    ok = bool(
        np.isfinite(mean)
        and np.isfinite(std)
        and abs(mean) <= mean_threshold
        and abs(std - 1.0) <= std_threshold
    )
    return {
        "mean": mean,
        "std": std,
        "n": int(values.size),
        "mean_threshold": float(mean_threshold),
        "std_threshold": float(std_threshold),
        "pass": ok,
        "message": (
            "normalized noise residuals pass"
            if ok
            else f"normalized noise residual mean/std out of bounds: {mean}, {std}"
        ),
    }


def _metallicity_population_report(
    frame: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    met = frame["log10_stellar_metallicity_true"].to_numpy(float)
    mass = frame["logsm_true"].to_numpy(float)
    finite = np.isfinite(met) & np.isfinite(mass)
    std = float(np.std(met[finite])) if finite.any() else float("nan")
    corr = (
        float(np.corrcoef(mass[finite], met[finite])[0, 1])
        if finite.sum() > 2 and np.std(mass[finite]) > 0.0 and np.std(met[finite]) > 0.0
        else float("nan")
    )
    clipped = (
        frame["metallicity_clipped"].astype(bool).to_numpy()
        if "metallicity_clipped" in frame
        else np.zeros(len(frame), dtype=bool)
    )
    selection = _selection_from_config_or_manifest(config, manifest)
    errors: list[str] = []
    population_ok = bool(
        np.isfinite(std) and std > 1.0e-4 and np.isfinite(corr) and corr > 0.05
    )
    if not population_ok:
        errors.append("metallicity is constant or lacks a visible positive trend")
    clipped_fraction = float(clipped.mean()) if len(clipped) else 0.0
    max_clipped = selection.get("max_metallicity_clipped_fraction")
    if max_clipped is not None and clipped_fraction > float(max_clipped):
        errors.append(
            f"metallicity clipped fraction {clipped_fraction:.4g} exceeds "
            f"{float(max_clipped):.4g}"
        )
    if bool(selection.get("require_metallicity_unclipped")) and int(clipped.sum()) > 0:
        errors.append(
            f"selection requires unclipped metallicity but found {int(clipped.sum())} rows"
        )
    bounds = _ssp_relative_metallicity_bounds(config)
    out_of_bounds_count = 0
    if bounds is not None:
        rel_lo, rel_hi = bounds
        tol = 1.0e-6
        out_of_bounds = finite & ((met < rel_lo - tol) | (met > rel_hi + tol))
        out_of_bounds_count = int(np.count_nonzero(out_of_bounds))
        if out_of_bounds_count:
            errors.append(
                f"{out_of_bounds_count} metallicity truths are outside SSP relative "
                f"support [{rel_lo:.6g}, {rel_hi:.6g}]"
            )
    max_abs_conversion_delta = 0.0
    if "lgmet_abs_used_true" in frame:
        z_sun = float((config.get("model", {}) or {}).get("z_sun", 0.0142))
        expected = frame["lgmet_abs_used_true"].to_numpy(float) - np.log10(z_sun)
        valid = np.isfinite(expected) & np.isfinite(met)
        if valid.any():
            max_abs_conversion_delta = float(np.max(np.abs(expected[valid] - met[valid])))
            if max_abs_conversion_delta > 1.0e-6:
                errors.append(
                    "log10_stellar_metallicity_true is inconsistent with "
                    "lgmet_abs_used_true - log10(z_sun)"
                )
    ok = not errors
    return {
        "std": std,
        "mass_metallicity_corr": corr,
        "clipped_count": int(clipped.sum()),
        "clipped_fraction": clipped_fraction,
        "ssp_relative_bounds": list(bounds) if bounds is not None else None,
        "out_of_bounds_count": out_of_bounds_count,
        "max_abs_conversion_delta": max_abs_conversion_delta,
        "pass": ok,
        "message": (
            "metallicity population checks pass"
            if ok
            else "; ".join(errors)
        ),
    }


def _photometric_selection_report(
    frame: pd.DataFrame,
    bands: list[str],
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    selection = _selection_from_config_or_manifest(config, manifest)
    min_true = int(selection["min_true_snr_bands"])
    min_observed = int(selection["min_observed_snr_bands"])
    if min_true <= 0 and min_observed <= 0:
        return {"enabled": False, "pass": True, "message": "no S/N selection configured"}
    errors: list[str] = []
    if "n_bands_true_snr_ge_threshold" not in frame:
        errors.append("missing n_bands_true_snr_ge_threshold")
        true_counts = np.empty(0, dtype=float)
    else:
        true_counts = frame["n_bands_true_snr_ge_threshold"].to_numpy(float)
    if "n_bands_observed_snr_ge_threshold" not in frame:
        errors.append("missing n_bands_observed_snr_ge_threshold")
        observed_counts = np.empty(0, dtype=float)
    else:
        observed_counts = frame["n_bands_observed_snr_ge_threshold"].to_numpy(float)
    if min_true > 0 and true_counts.size:
        failures = int(np.count_nonzero(true_counts < min_true))
        if failures:
            errors.append(f"{failures} rows fail min_true_snr_bands={min_true}")
    if min_observed > 0 and observed_counts.size:
        failures = int(np.count_nonzero(observed_counts < min_observed))
        if failures:
            errors.append(f"{failures} rows fail min_observed_snr_bands={min_observed}")
    return {
        "enabled": True,
        "bands": list(bands),
        "snr_threshold": float(selection["snr_threshold"]),
        "min_true_snr_bands": min_true,
        "min_observed_snr_bands": min_observed,
        "true_snr_band_count_min": float(np.min(true_counts))
        if true_counts.size
        else float("nan"),
        "true_snr_band_count_median": float(np.median(true_counts))
        if true_counts.size
        else float("nan"),
        "observed_snr_band_count_min": float(np.min(observed_counts))
        if observed_counts.size
        else float("nan"),
        "observed_snr_band_count_median": float(np.median(observed_counts))
        if observed_counts.size
        else float("nan"),
        "pass": not errors,
        "message": "photometric S/N selection checks pass"
        if not errors
        else "; ".join(errors),
    }


def _ssp_relative_metallicity_bounds(config: dict[str, Any]) -> tuple[float, float] | None:
    path = Path(str(config.get("ssp_path", "")))
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        if "ssp_lgmet" not in handle:
            return None
        grid = np.asarray(handle["ssp_lgmet"][:], dtype=float)
    grid = grid[np.isfinite(grid)]
    if grid.size == 0:
        return None
    z_sun = float((config.get("model", {}) or {}).get("z_sun", 0.0142))
    logzsun = float(np.log10(z_sun))
    return float(np.min(grid) - logzsun), float(np.max(grid) - logzsun)


def _selection_from_config_or_manifest(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_selection = (
        (manifest.get("synthetic_diffsky", {}) or {}).get("selection")
        if isinstance(manifest, dict)
        else None
    )
    if manifest_selection is not None:
        return normalize_selection(manifest_selection)
    return normalize_selection(
        (config.get("synthetic_diffsky", {}) or {}).get("selection", {}) or {}
    )


def _recompute_flux_report(
    config: dict[str, Any],
    frame: pd.DataFrame,
    bands: list[str],
    *,
    sample_size: int,
    batch_size: int,
) -> dict[str, Any]:
    sample_size = min(int(sample_size), len(frame))
    sample = frame.sample(n=sample_size, random_state=260617) if sample_size else frame
    theta = theta_from_truth_frame(sample)
    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int((config.get("model", {}) or {}).get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    chunks = []
    for start in range(0, len(theta), int(batch_size)):
        pred = model_mags_from_theta_matrix_jax(
            context,
            model_args,
            jnp.asarray(theta[start : start + int(batch_size)], dtype=jnp.float32),
            DIFFSKY_BASIC_PARAMETER_NAMES,
        )
        chunks.append(np.asarray(jax.device_get(pred), dtype=float))
    mag = np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(bands)))
    deltas = []
    rel_flux = []
    for index, band in enumerate(bands):
        if f"mag_true_{band}" in sample:
            deltas.append(np.abs(mag[:, index] - sample[f"mag_true_{band}"].to_numpy(float)))
        flux = np.asarray(abmag_to_fnu_cgs(mag[:, index]), dtype=float)
        old = sample[f"flux_true_{band}"].to_numpy(float)
        rel_flux.append(np.abs(flux - old) / np.maximum(np.abs(old), 1.0e-300))
    max_abs_delta_mag = float(np.max(np.concatenate(deltas))) if deltas else 0.0
    max_relative_flux_error = float(np.max(np.concatenate(rel_flux))) if rel_flux else 0.0
    ok = bool(max_abs_delta_mag <= 3.0e-5 and max_relative_flux_error <= 1.0e-4)
    return {
        "sample_size": int(sample_size),
        "max_abs_delta_mag": max_abs_delta_mag,
        "max_relative_flux_error": max_relative_flux_error,
        "pass": ok,
        "message": (
            "DSPS flux recomputation passes"
            if ok
            else "DSPS flux recomputation exceeds tolerance"
        ),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _expected_sizes(manifest: dict[str, Any]) -> dict[str, int]:
    splits = manifest.get("splits", {}) if isinstance(manifest, dict) else {}
    out = dict(DEFAULT_SPLIT_SIZES)
    for name, payload in splits.items():
        if isinstance(payload, dict) and "final_size" in payload:
            out[str(name)] = int(payload["final_size"])
    return out
