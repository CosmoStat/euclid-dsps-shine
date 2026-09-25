"""Weighted raw proposals -> empirical parent, without photometric preselection."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES

TRUTH_COLUMNS = tuple(
    "redshift_true"
    if name == "z_obs"
    else "logsm_true"
    if name == "log10_stellar_mass"
    else f"{name}_true"
    for name in DIFFSKY_BASIC_PARAMETER_NAMES
)
SOURCE_COLUMNS = (
    "source_proposal_id",
    "source_seed",
    "source_split",
    "source_shard",
    "galaxy_weight",
    "metallicity_clipped",
    *TRUTH_COLUMNS,
)


def partition_source_shards(sources: dict, sizes: dict, seed: int) -> tuple[dict, dict]:
    """Hold out whole generator realizations, not merely differently named rows.

    The old generator keyed each shard by source_seed + shard_index. Original
    validation/test seeds overlap train. Use train once as the canonical pool
    only when it covers all saved effective seeds; never count aliases twice.
    """

    def effective(record):
        return {
            int(record["source_seed"]) + int(Path(p).stem.removeprefix("shard_"))
            for p in record["paths"]
        }

    canonical = sources["train"]
    seeds = effective(canonical)
    if len(seeds) != len(canonical["paths"]) or len(seeds) < len(sizes):
        raise ValueError("Insufficient unique canonical source shards")
    if any(not effective(s).issubset(seeds) for s in sources.values()):
        raise ValueError(
            "Canonical train pool does not cover all effective source seeds"
        )
    names = list(sizes)
    fractions = np.array([sizes[s] for s in names], float)
    if not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError("Positive split sizes required")
    counts = len(seeds) * fractions / fractions.sum()
    n = np.floor(counts).astype(int)
    for i in np.argsort(-(counts - n), kind="stable")[: len(seeds) - n.sum()]:
        n[i] += 1
    if np.any(n < 1):
        raise ValueError("Each split requires at least one independent source shard")
    paths = np.random.default_rng(seed).permutation(sorted(canonical["paths"]))
    result, start = {}, 0
    for split, count in zip(names, n, strict=True):
        record = dict(
            source_split="train",
            source_seed=canonical["source_seed"],
            paths=sorted(str(p) for p in paths[start : start + count]),
        )
        record["effective_seeds"] = sorted(effective(record))
        result[split] = record
        start += count
    return result, dict(
        method="whole_effective_seed_holdout_from_canonical_train_pool",
        original_shards=sum(len(s["paths"]) for s in sources.values()),
        canonical_unique_shards=len(seeds),
        ignored_overlapping_shards=sum(len(s["paths"]) for s in sources.values())
        - len(seeds),
        shard_counts={s: len(r["paths"]) for s, r in result.items()},
        effective_seed_overlap=0,
    )


def eligible_proposals(path: Path, split: str, source_seed: int) -> pd.DataFrame:
    """Only the declared SSP-metallicity support restriction is applied here."""
    frame = pd.read_parquet(path, columns=list(SOURCE_COLUMNS))
    shard = int(path.stem.removeprefix("shard_"))
    if (
        frame.empty
        or frame.source_proposal_id.isna().any()
        or frame.source_proposal_id.duplicated().any()
        or not frame.source_split.eq(split).all()
        or not frame.source_seed.eq(source_seed).all()
        or not frame.source_shard.eq(shard).all()
        or not frame.source_proposal_id.str.startswith(
            f"{split}:{source_seed}:{shard}:"
        ).all()
    ):
        raise ValueError(f"Invalid raw proposal identities: {path}")
    w = frame.galaxy_weight.to_numpy(float)
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError(f"Invalid proposal weights: {path}")
    if not np.isfinite(frame[list(TRUTH_COLUMNS)].to_numpy(float)).all():
        raise ValueError(f"Nonfinite native truth: {path}")
    if (
        frame.metallicity_clipped.isna().any()
        or not frame.metallicity_clipped.isin([True, False]).all()
    ):
        raise ValueError(f"Invalid metallicity support flag: {path}")
    rows = frame.source_proposal_id.str.rsplit(":", n=1).str[-1]
    if not rows.str.fullmatch(r"\d+").all():
        raise ValueError(f"Invalid proposal row identity: {path}")
    frame["effective_source_seed"] = source_seed + shard
    frame["effective_proposal_key"] = str(source_seed + shard) + ":" + rows
    return frame.loc[(w > 0) & ~frame.metallicity_clipped.astype(bool)].copy()


def sample_parent(
    paths: list[Path],
    split: str,
    source_seed: int,
    count: int,
    seed: int,
    *,
    sha: Callable[[Path], str],
    progress: Callable[[dict], None],
) -> tuple[pd.DataFrame, dict]:
    """Exact categorical sampling over all shards, without loading the full pool.

    P(shard) = sum(w in shard) / sum(w), then P(row | shard) = w / sum(w in shard).
    Their product is the desired proposal probability. Final rows have weight one.
    """
    if count < 1 or not paths:
        raise ValueError("Nonempty pool and positive sample count required")
    records = []
    for i, path in enumerate(paths):
        digest = sha(path)
        frame = eligible_proposals(path, split, source_seed)
        w = frame.galaxy_weight.to_numpy(float)
        records.append(
            dict(
                path=str(path),
                sha256=digest,
                eligible_rows=len(frame),
                mass=float(w.sum()),
                weight_square_sum=float(w @ w),
            )
        )
        progress(dict(stage="scan", done=i + 1, total=len(paths)))
    masses = np.array([r["mass"] for r in records])
    if not np.isfinite(masses).all() or masses.sum() <= 0:
        raise ValueError("No finite positive proposal mass in declared support")
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(count, masses / masses.sum())
    pieces = []
    for i, (path, n) in enumerate(zip(paths, counts, strict=True)):
        if n:
            if sha(path) != records[i]["sha256"]:
                raise ValueError(f"Proposal changed during sampling: {path}")
            frame = eligible_proposals(path, split, source_seed)
            w = frame.galaxy_weight.to_numpy(float)
            pieces.append(frame.iloc[rng.choice(len(frame), int(n), p=w / w.sum())])
        progress(dict(stage="sample", done=i + 1, total=len(paths)))
    result = (
        pd.concat(pieces, ignore_index=True)
        .iloc[rng.permutation(count)]
        .reset_index(drop=True)
    )
    result = result.rename(columns={"galaxy_weight": "source_proposal_weight"})
    # Preserve the legacy column safely for readers that automatically choose it.
    result["galaxy_weight"] = 1.0
    result["population_weight"] = 1.0
    summary = dict(
        source_files=records,
        count=count,
        unique_proposals=result.effective_proposal_key.nunique(),
        weight_role="resampled_once_unit_row_weights",
        no_photometric_preselection=True,
        support="positive-weight proposals with unclipped SSP metallicity",
        eligible_proposals=sum(r["eligible_rows"] for r in records),
        proposal_ess=float(
            masses.sum() ** 2 / sum(r["weight_square_sum"] for r in records)
        ),
    )
    return result, summary


def observe(
    frame: pd.DataFrame,
    flux: np.ndarray,
    bands: list[dict],
    error_model: dict,
    *,
    seed: int,
    selection: dict,
) -> pd.DataFrame:
    from euclid_dsps.photometric_uncertainty import flux_error_from_model
    from euclid_dsps.photometry import abmag_to_fnu_cgs

    names = [b["name"] for b in bands]
    if (
        flux.shape != (len(frame), len(names))
        or not np.isfinite(flux).all()
        or np.any(flux < 0)
    ):
        raise ValueError("Invalid forward photometry; no rows may be silently dropped")
    if selection["band"] not in names:
        raise ValueError("Selection band absent")
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for i, band in enumerate(bands):
        name = band["name"]
        errors = flux_error_from_model(
            flux[:, i], band.get("error_model") or error_model, band_name=name
        )
        if not np.isfinite(errors).all() or np.any(errors <= 0):
            raise ValueError("Invalid noise scale")
        result[f"flux_true_{name}"] = flux[:, i]
        result[f"fluxerr_{name}"] = errors
        result[f"flux_{name}"] = flux[:, i] + rng.normal(size=len(frame)) * errors
        result[f"mask_{name}"] = True
    band = selection["band"]
    threshold = float(abmag_to_fnu_cgs(selection["max_mag_ab"]))
    result["selected_r29"] = result[f"mask_{band}"] & (
        result[f"flux_{band}"] > threshold
    )
    return result


def catalogue_checks(frame: pd.DataFrame, bands: list[dict], selection: dict) -> dict:
    """Use ALL parent rows for noise and efficiency checks, including rejections."""
    from scipy.special import ndtr

    from euclid_dsps.photometry import abmag_to_fnu_cgs

    if frame.empty or frame.object_id.duplicated().any():
        raise ValueError("Empty or duplicated object identities")
    if not frame.population_weight.eq(1).all() or not frame.galaxy_weight.eq(1).all():
        raise ValueError("Final rows must not be proposal weighted a second time")
    noise = []
    for band in bands:
        name = band["name"]
        f, e, obs = [
            frame[f"{prefix}_{name}"].to_numpy(float)
            for prefix in ("flux_true", "fluxerr", "flux")
        ]
        if not np.isfinite([f, e, obs]).all() or np.any(e <= 0):
            raise ValueError(f"Invalid flux/error values: {name}")
        if not frame[f"mask_{name}"].eq(True).all():
            raise ValueError("This version requires the explicit all-observed mask law")
        z = (obs - f) / e
        noise.append(
            dict(
                band=name,
                mean=float(z.mean()),
                std=float(z.std()),
                fraction_negative_observed=float(np.mean(obs < 0)),
            )
        )
    band = selection["band"]
    threshold = float(abmag_to_fnu_cgs(selection["max_mag_ab"]))
    actual = frame[f"mask_{band}"] & (frame[f"flux_{band}"] > threshold)
    if not np.array_equal(actual, frame.selected_r29):
        raise ValueError("Stored selection differs from observed r selection")
    beta = ndtr((frame[f"flux_true_{band}"] - threshold) / frame[f"fluxerr_{band}"])
    expected = float(np.sum(beta))
    std_count = float(np.sqrt(np.sum(beta * (1 - beta))))
    return dict(
        rows=len(frame),
        selected=int(actual.sum()),
        empirical_alpha=float(actual.mean()),
        analytic_alpha=float(np.mean(beta)),
        selection_count_difference=float(actual.sum() - expected),
        selection_count_std=std_count,
        noise=noise,
    )
