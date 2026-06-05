"""Prepare compact OpenUniverse LSST+Roman parquet subsets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.io import ensure_dir

from .access import resolve_openuniverse_paths
from .io import JOIN_ATTR_KEY, read_openuniverse_hpix
from .noise import add_band_snr_noise, add_fractional_snr_noise
from .schema import (
    OU_FLUX_COLUMNS,
    OU_LSST_ROMAN_14_BANDS,
    OU_NATIVE_FLUX_UNIT,
    OU_TRUTH_COLUMNS,
    normalized_flux_column,
    normalized_flux_truth_column,
    normalized_fluxerr_column,
    normalized_mask_column,
)
from .units import photon_flux_to_internal


def prepare_openuniverse_lsst_roman_subset(
    *,
    hpix_ids: Sequence[int],
    input_root: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    min_flux_valid_bands: int = 8,
    noise_model: dict | None = None,
    seed: int = 42,
) -> dict:
    """Prepare a normalized LSST+Roman 14-band OpenUniverse subset parquet."""
    hpix_tuple = tuple(int(hpix) for hpix in hpix_ids)
    if not hpix_tuple:
        raise ValueError("At least one OpenUniverse hpix id is required")
    if int(min_flux_valid_bands) < 1:
        raise ValueError("min_flux_valid_bands must be >= 1")
    if int(min_flux_valid_bands) > len(OU_LSST_ROMAN_14_BANDS):
        raise ValueError(
            "min_flux_valid_bands cannot exceed the 14 LSST+Roman target bands"
        )

    paths = resolve_openuniverse_paths(hpix_tuple, input_root)
    frames: list[pd.DataFrame] = []
    join_reports: list[dict[str, int | str]] = []
    remaining = None if limit is None else max(int(limit), 0)
    for hpix, group in zip(hpix_tuple, paths, strict=True):
        frame = read_openuniverse_hpix(group["main"], group["flux"])
        report = dict(frame.attrs.get(JOIN_ATTR_KEY, {}))
        report["hpix"] = int(hpix)
        join_reports.append(report)
        normalized = _normalize_joined_frame(
            frame,
            min_flux_valid_bands=int(min_flux_valid_bands),
            noise_model=_normalized_noise_model(noise_model),
            seed=int(seed) + len(frames),
        )
        if remaining is not None:
            normalized = normalized.head(remaining)
            remaining -= int(len(normalized))
        if len(normalized):
            frames.append(normalized)
        if remaining is not None and remaining <= 0:
            break
    if not frames:
        raise ValueError("No OpenUniverse rows survived subset preparation")

    output = pd.concat(frames, ignore_index=True)
    out_path = Path(output_path)
    ensure_dir(out_path.parent)
    output.to_parquet(out_path, index=False)

    manifest = _manifest_payload(
        hpix_ids=hpix_tuple,
        input_root=input_root,
        output_path=out_path,
        n_rows=len(output),
        min_flux_valid_bands=min_flux_valid_bands,
        noise_model=_normalized_noise_model(noise_model),
        join_reports=join_reports,
    )
    manifest_path = out_path.with_suffix(".manifest.yaml")
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest


def _normalize_joined_frame(
    frame: pd.DataFrame,
    *,
    min_flux_valid_bands: int,
    noise_model: dict,
    seed: int,
) -> pd.DataFrame:
    required = [
        "galaxy_id",
        "ra",
        "dec",
        OU_TRUTH_COLUMNS["redshift"],
        OU_TRUTH_COLUMNS["redshift_hubble"],
        OU_TRUTH_COLUMNS["stellar_mass"],
        *OU_FLUX_COLUMNS.values(),
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(
            "OpenUniverse joined table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    flux_truth = np.stack(
        [
            photon_flux_to_internal(frame[OU_FLUX_COLUMNS[band]].to_numpy())
            for band in OU_LSST_ROMAN_14_BANDS
        ],
        axis=1,
    )
    truth_mask = np.isfinite(flux_truth) & (flux_truth > 0.0)
    keep = truth_mask.sum(axis=1) >= int(min_flux_valid_bands)
    filtered = frame.loc[keep].reset_index(drop=True).copy()
    flux_truth = flux_truth[keep]
    truth_mask = truth_mask[keep]
    flux_obs, flux_err = _apply_noise_model(
        flux_truth,
        noise_model=noise_model,
        seed=seed,
    )
    mask = truth_mask & np.isfinite(flux_obs) & np.isfinite(flux_err) & (flux_err > 0.0)

    out = pd.DataFrame(
        {
            "galaxy_id": filtered["galaxy_id"].to_numpy(),
            "ra": filtered["ra"].to_numpy(dtype=float),
            "dec": filtered["dec"].to_numpy(dtype=float),
            "redshift": filtered[OU_TRUTH_COLUMNS["redshift"]].to_numpy(dtype=float),
            "redshiftHubble": filtered[OU_TRUTH_COLUMNS["redshift_hubble"]].to_numpy(
                dtype=float
            ),
            "stellar_mass": filtered[OU_TRUTH_COLUMNS["stellar_mass"]].to_numpy(
                dtype=float
            ),
            "redshift_truth": filtered[OU_TRUTH_COLUMNS["redshift"]].to_numpy(
                dtype=float
            ),
            "redshift_hubble_truth": filtered[
                OU_TRUTH_COLUMNS["redshift_hubble"]
            ].to_numpy(dtype=float),
        }
    )
    if OU_TRUTH_COLUMNS["stellar_mass"] not in out:
        out[OU_TRUTH_COLUMNS["stellar_mass"]] = filtered[
            OU_TRUTH_COLUMNS["stellar_mass"]
        ].to_numpy(dtype=float)
    else:
        out["um_source_galaxy_obs_sm"] = filtered[
            OU_TRUTH_COLUMNS["stellar_mass"]
        ].to_numpy(dtype=float)

    for band_index, band in enumerate(OU_LSST_ROMAN_14_BANDS):
        out[normalized_flux_truth_column(band)] = flux_truth[:, band_index]
        out[normalized_flux_column(band)] = flux_obs[:, band_index]
        out[normalized_fluxerr_column(band)] = flux_err[:, band_index]
        out[normalized_mask_column(band)] = mask[:, band_index]
    return out


def _apply_noise_model(
    flux_truth: np.ndarray,
    *,
    noise_model: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    kind = str(noise_model.get("type", "fractional_snr"))
    if kind == "fractional_snr":
        return add_fractional_snr_noise(
            flux_truth,
            snr=float(noise_model.get("snr", 50.0)),
            seed=int(seed),
            min_sigma_fraction=float(noise_model.get("min_sigma_fraction", 1.0e-4)),
        )
    if kind == "band_snr":
        return add_band_snr_noise(
            flux_truth,
            band_snr=dict(noise_model.get("band_snr", {}) or {}),
            band_names=OU_LSST_ROMAN_14_BANDS,
            seed=int(seed),
        )
    if kind in {"none", "truth"}:
        _, flux_err = add_fractional_snr_noise(
            flux_truth,
            snr=float(noise_model.get("snr", 10_000.0)),
            seed=int(seed),
            min_sigma_fraction=float(noise_model.get("min_sigma_fraction", 1.0e-4)),
        )
        return np.asarray(flux_truth, dtype=np.float32), flux_err
    if kind == "depth_like":
        from .noise import add_depth_like_noise

        return add_depth_like_noise(flux_truth, seed=seed, **noise_model)
    raise ValueError(f"Unsupported OpenUniverse noise model type: {kind}")


def _normalized_noise_model(noise_model: dict | None) -> dict:
    if noise_model is None:
        return {"type": "fractional_snr", "snr": 50.0}
    return dict(noise_model)


def _manifest_payload(
    *,
    hpix_ids: tuple[int, ...],
    input_root: str | Path,
    output_path: Path,
    n_rows: int,
    min_flux_valid_bands: int,
    noise_model: dict,
    join_reports: list[dict[str, int | str]],
) -> dict[str, Any]:
    return {
        "dataset": "openuniverse_lsst_roman_14",
        "hpix_ids": list(hpix_ids),
        "input_root": str(input_root),
        "output_path": str(output_path),
        "number_of_rows": int(n_rows),
        "bands": list(OU_LSST_ROMAN_14_BANDS),
        "flux_unit": OU_NATIVE_FLUX_UNIT,
        "noise_model": noise_model,
        "min_flux_valid_bands": int(min_flux_valid_bands),
        "join_reports": join_reports,
        "created_utc": datetime.now(UTC).isoformat(),
        "code_version": _code_version(),
        "truth_policy": {
            "redshift": "truth",
            "redshiftHubble": "truth",
            "stellar_mass": "truth",
            "sfh": "unavailable",
            "dust": "unavailable",
            "metallicity": "unavailable",
            "halo": "unavailable",
        },
        "unit_todo": [
            "Keep native OpenUniverse photon_per_sec_cm2 for PR 1.",
            "Implement DSPS photon-rate photometry or validated conversion before "
            "using a physical decoder against these fluxes.",
            "Verify LSST/Roman filters and unit conventions.",
        ],
    }


def _code_version() -> str | None:
    try:
        return metadata.version("euclid-dsps-shine")
    except metadata.PackageNotFoundError:
        return None
