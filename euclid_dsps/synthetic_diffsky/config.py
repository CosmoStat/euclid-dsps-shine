"""Configuration helpers for Diffsky/FENIKS DSPS closure generation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from euclid_dsps.photometric_uncertainty import default_m5_depth_error_model

SPLIT_ORDER = ("train", "validation", "test")
DEFAULT_SPLIT_SIZES = {"train": 40_000, "validation": 5_000, "test": 5_000}
SMOKE_SPLIT_SIZES = {"train": 240, "validation": 40, "test": 40}


@dataclass(frozen=True)
class SplitGenerationConfig:
    name: str
    n_final: int
    source_seed: int
    noise_seed: int
    resample_seed: int
    object_id_start: int


@dataclass(frozen=True)
class SyntheticDiffskyConfig:
    output_dir: Path
    proposal_backend: str
    calibration_dir: str
    calibration_name: str
    diffsky_commit: str | None
    n_host_halos_per_shard: int
    max_shards: int
    jax_batch_size: int
    z_min: float
    z_max: float
    lgmp_min: float
    lgmp_max: float
    sky_area_degsq: float
    mc_merge: int
    z_phot_table_size: int
    logmp_cutoff: float
    weighted_lc_photdata_kwargs: dict[str, Any]
    mc_lc_phot_kwargs: dict[str, Any]
    metallicity_grid_policy: str
    stellar_metallicity_scatter_dex: float
    max_duplication_fraction: float
    duplication_gate: str
    min_ess_fraction: float
    pool_size_factor: float
    selection: dict[str, Any]
    flux_error_model: dict[str, Any]
    splits: dict[str, SplitGenerationConfig]
    config_hash: str | None = None


def load_synthetic_diffsky_config(
    config: dict[str, Any],
    *,
    smoke: bool = False,
    max_galaxies: int | None = None,
) -> SyntheticDiffskyConfig:
    """Resolve the synthetic Diffsky generation block from a normalized config."""
    raw = copy.deepcopy(config.get("synthetic_diffsky", {}) or {})
    output_dir = Path(
        raw.get(
            "output_dir",
            "Data/diffsky/synthetic/feniks_260617_dsps_closure",
        )
    )
    split_sizes = dict(DEFAULT_SPLIT_SIZES)
    split_sizes.update(raw.get("split_sizes", {}) or {})
    if smoke:
        split_sizes = dict(raw.get("smoke_split_sizes", SMOKE_SPLIT_SIZES) or {})
    if max_galaxies is not None:
        split_sizes = _cap_split_sizes(split_sizes, int(max_galaxies))
    seeds = {
        "train": 26061701,
        "validation": 26061702,
        "test": 26061703,
    }
    seeds.update(raw.get("source_seeds", {}) or {})
    noise_seeds = {
        "train": 26062701,
        "validation": 26062702,
        "test": 26062703,
    }
    noise_seeds.update(raw.get("noise_seeds", {}) or {})
    resample_seeds = {
        "train": 26063701,
        "validation": 26063702,
        "test": 26063703,
    }
    resample_seeds.update(raw.get("resample_seeds", {}) or {})
    starts = {"train": 0, "validation": 1_000_000_000, "test": 2_000_000_000}
    starts.update(raw.get("object_id_starts", {}) or {})
    selection = dict(raw.get("selection", {}) or {})
    if smoke:
        selection.update(dict(raw.get("smoke_selection", {}) or {}))
    splits = {
        name: SplitGenerationConfig(
            name=name,
            n_final=int(split_sizes.get(name, 0)),
            source_seed=int(seeds[name]),
            noise_seed=int(noise_seeds[name]),
            resample_seed=int(resample_seeds[name]),
            object_id_start=int(starts[name]),
        )
        for name in SPLIT_ORDER
    }
    return SyntheticDiffskyConfig(
        output_dir=output_dir,
        proposal_backend=str(raw.get("proposal_backend", "diffsky")),
        calibration_dir=str(raw.get("calibration_dir", "feniks_calibrations")),
        calibration_name=str(raw.get("calibration_name", "feniks_260617")),
        diffsky_commit=(
            None
            if raw.get("diffsky_commit") is None
            else str(raw.get("diffsky_commit"))
        ),
        n_host_halos_per_shard=int(
            _runtime_value(raw, "n_host_halos_per_shard", 2048, smoke=smoke)
        ),
        max_shards=int(_runtime_value(raw, "max_shards", 64, smoke=smoke)),
        jax_batch_size=int(raw.get("jax_batch_size", 256)),
        z_min=float(raw.get("z_min", 0.001)),
        z_max=float(raw.get("z_max", 0.35)),
        lgmp_min=float(raw.get("lgmp_min", 10.5)),
        lgmp_max=float(raw.get("lgmp_max", 15.0)),
        sky_area_degsq=float(raw.get("sky_area_degsq", 10.0)),
        mc_merge=int(raw.get("mc_merge", 0)),
        z_phot_table_size=int(
            _runtime_value(raw, "z_phot_table_size", 64, smoke=smoke)
        ),
        logmp_cutoff=float(raw.get("logmp_cutoff", 11.0)),
        weighted_lc_photdata_kwargs=dict(raw.get("weighted_lc_photdata_kwargs", {}) or {}),
        mc_lc_phot_kwargs=dict(raw.get("mc_lc_phot_kwargs", {}) or {}),
        metallicity_grid_policy=str(raw.get("metallicity_grid_policy", "fail")),
        stellar_metallicity_scatter_dex=float(
            raw.get(
                "stellar_metallicity_scatter_dex",
                (config.get("model", {}) or {}).get(
                    "stellar_metallicity_scatter_dex", 0.2
                ),
            )
        ),
        max_duplication_fraction=float(
            _runtime_value(raw, "max_duplication_fraction", 0.05, smoke=smoke)
        ),
        duplication_gate=str(raw.get("duplication_gate", "fail")),
        min_ess_fraction=float(
            _runtime_value(raw, "min_ess_fraction", 2.0, smoke=smoke)
        ),
        pool_size_factor=float(_runtime_value(raw, "pool_size_factor", 4.0, smoke=smoke)),
        selection=selection,
        flux_error_model=dict(raw.get("flux_error_model", default_m5_depth_error_model()) or {}),
        splits=splits,
    )


def selected_splits(split: str) -> tuple[str, ...]:
    """Resolve a CLI split selector."""
    split = str(split)
    if split == "all":
        return SPLIT_ORDER
    if split not in SPLIT_ORDER:
        supported = ", ".join((*SPLIT_ORDER, "all"))
        raise ValueError(f"Unsupported split {split!r}; use {supported}")
    return (split,)


def _runtime_value(
    raw: dict[str, Any],
    name: str,
    default: Any,
    *,
    smoke: bool,
) -> Any:
    if smoke:
        smoke_name = f"smoke_{name}"
        if smoke_name in raw:
            return raw[smoke_name]
    return raw.get(name, default)


def _cap_split_sizes(split_sizes: dict[str, int], max_galaxies: int) -> dict[str, int]:
    if max_galaxies < 0:
        raise ValueError("--max-galaxies must be non-negative")
    total = sum(int(split_sizes.get(name, 0)) for name in SPLIT_ORDER)
    if total <= max_galaxies:
        return {name: int(split_sizes.get(name, 0)) for name in SPLIT_ORDER}
    if max_galaxies == 0:
        return {name: 0 for name in SPLIT_ORDER}
    capped: dict[str, int] = {}
    remaining = int(max_galaxies)
    for index, name in enumerate(SPLIT_ORDER):
        if index == len(SPLIT_ORDER) - 1:
            capped[name] = remaining
            break
        target = int(round(max_galaxies * int(split_sizes.get(name, 0)) / total))
        target = min(max(target, 0), remaining)
        capped[name] = target
        remaining -= target
    return capped
