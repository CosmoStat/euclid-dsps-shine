"""MIRA diagnostics for held-out amortized posterior samples.

The implementation follows the finite-sample normalization used by
``mira-score`` while retaining per-object contributions so uncertainty can be
estimated by bootstrapping held-out objects rather than random regions.
"""

from __future__ import annotations

import hashlib
import itertools
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from euclid_dsps.io import ensure_dir, write_json

FENIKS_SPLINE15D_PARAMETERS = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "sfh_dlog_sfr_01",
    "sfh_dlog_sfr_02",
    "sfh_dlog_sfr_03",
    "sfh_dlog_sfr_04",
    "sfh_dlog_sfr_05",
    "sfh_dlog_sfr_06",
    "sfh_dlog_sfr_07",
    "sfh_dlog_sfr_08",
    "sfh_dlog_sfr_09",
    "sfh_dlog_sfr_10",
)

MIRA_IDEAL_SCORE = 2.0 / 3.0
MIRA_PAPER_URL = "https://arxiv.org/abs/2605.02014"
MIRA_UPSTREAM_COMMIT = "c57487198ac30711783b78ac2af6a76758544483"


@dataclass(frozen=True)
class PosteriorInput:
    """Named posterior sample source and its resolved parquet files."""

    name: str
    source: Path
    files: tuple[Path, ...]


@dataclass(frozen=True)
class DensePosterior:
    """Dense posterior samples aligned to the truth-object order."""

    name: str
    values: np.ndarray
    sample_ids: tuple[Any, ...]
    files: tuple[Path, ...]


def parse_posterior_spec(spec: str) -> tuple[str, Path]:
    """Parse ``NAME=PATH`` used by the standalone MIRA evaluator."""
    if "=" not in spec:
        raise ValueError(
            f"Posterior specification must be NAME=PATH, received {spec!r}"
        )
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(
            f"Posterior specification must be NAME=PATH, received {spec!r}"
        )
    return name, Path(raw_path)


def resolve_truth_path(path: str | Path) -> Path:
    """Resolve a truth parquet supplied directly or through an inference dir."""
    source = Path(path)
    if source.is_file():
        return source
    candidate = source / "inference_truth.parquet"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Could not find truth parquet at {source} or {candidate}")


def resolve_posterior_input(name: str, path: str | Path) -> PosteriorInput:
    """Resolve monolithic or sharded encoder-posterior parquet artifacts."""
    source = Path(path)
    if source.is_file():
        files = (source,)
    elif source.is_dir():
        monolithic = source / "posterior_samples.parquet"
        shard_dir = source / "posterior_samples"
        if monolithic.is_file():
            files = (monolithic,)
        elif shard_dir.is_dir():
            files = tuple(sorted(shard_dir.glob("*.parquet")))
        else:
            files = tuple(sorted(source.glob("*.parquet")))
    else:
        raise FileNotFoundError(f"Posterior source does not exist: {source}")
    if not files:
        raise FileNotFoundError(
            "No posterior parquet found. Expected posterior_samples.parquet, "
            f"posterior_samples/*.parquet, or parquet shards under {source}"
        )
    return PosteriorInput(name=name, source=source, files=files)


def resolve_companion_truth(source: PosteriorInput) -> Path | None:
    """Find the inference truth table adjacent to a posterior source."""
    if source.source.is_file():
        candidates = (source.source.parent / "inference_truth.parquet",)
    else:
        candidates = (
            source.source / "inference_truth.parquet",
            source.source.parent / "inference_truth.parquet",
        )
    return next((path for path in candidates if path.is_file()), None)


def feniks_mira_groups(
    parameters: Sequence[str] = FENIKS_SPLINE15D_PARAMETERS,
) -> dict[str, tuple[int, ...]]:
    """Return the full, physical, SFH, and marginal FENIKS score groups."""
    names = tuple(parameters)
    if names != FENIKS_SPLINE15D_PARAMETERS:
        raise ValueError(
            "FENIKS MIRA currently requires the canonical spline15d parameter order"
        )
    groups: dict[str, tuple[int, ...]] = {
        "full_15d": tuple(range(len(names))),
        "physical_5d": tuple(range(5)),
        "sfh_contrasts_10d": tuple(range(5, len(names))),
    }
    groups.update({f"marginal_{name}": (index,) for index, name in enumerate(names)})
    return groups


@jax.jit
def mira_region_contributions(
    truth: jnp.ndarray,
    posterior: jnp.ndarray,
    centers: jnp.ndarray,
    reference_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Return MIRA values with shape ``[regions, models, objects]``.

    ``truth`` is ``[objects, dimensions]`` and ``posterior`` is
    ``[models, objects, samples, dimensions]``. One posterior draw defines the
    radius of each random ball. Strict distance comparison excludes that
    reference draw from the count, matching the continuous-distribution
    construction with ``N = samples - 1``.
    """
    truth = jnp.asarray(truth, dtype=jnp.float32)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    centers = jnp.asarray(centers, dtype=jnp.float32)
    reference_indices = jnp.asarray(reference_indices, dtype=jnp.int32)
    n_samples = posterior.shape[2]
    max_probability = jnp.asarray(n_samples / (n_samples + 1), dtype=posterior.dtype)

    def one_region(
        carry: None,
        region: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[None, jnp.ndarray]:
        center, reference_index = region
        sample_delta = posterior - center[None, :, None, :]
        sample_distance_squared = jnp.sum(sample_delta * sample_delta, axis=-1)
        index = jnp.broadcast_to(
            reference_index[None, :, None],
            (posterior.shape[0], posterior.shape[1], 1),
        )
        radius_squared = jnp.take_along_axis(sample_distance_squared, index, axis=2)[
            ..., 0
        ]
        counts = jnp.sum(
            sample_distance_squared < radius_squared[..., None],
            axis=2,
        )
        truth_delta = truth - center
        truth_distance_squared = jnp.sum(truth_delta * truth_delta, axis=-1)
        truth_inside = truth_distance_squared[None, :] <= radius_squared
        n_candidate = n_samples - 1
        probability_inside = (counts + 1) / (n_candidate + 2)
        probability_outside = (n_candidate - counts + 1) / (n_candidate + 2)
        probability = jnp.where(
            truth_inside,
            probability_inside,
            probability_outside,
        )
        return None, probability / max_probability

    _, contributions = jax.lax.scan(
        one_region,
        None,
        (centers, reference_indices),
    )
    return contributions


def evaluate_feniks_mira(
    *,
    truth_path: str | Path,
    posterior_specs: Sequence[tuple[str, str | Path]],
    out_dir: str | Path,
    num_regions: int = 100,
    num_bootstrap: int = 1000,
    samples_per_object: int | None = 128,
    seed: int = 260730,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate FENIKS encoder posteriors and write reproducible artifacts."""
    if num_regions <= 0:
        raise ValueError("num_regions must be positive")
    if num_bootstrap < 0:
        raise ValueError("num_bootstrap must be non-negative")
    if samples_per_object is not None and samples_per_object < 2:
        raise ValueError("samples_per_object must be at least 2")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if not posterior_specs:
        raise ValueError("At least one posterior specification is required")

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    ensure_dir(out)
    started = time.perf_counter()
    parameters = FENIKS_SPLINE15D_PARAMETERS
    truth_file = resolve_truth_path(truth_path)
    posterior_inputs = [
        resolve_posterior_input(name, source) for name, source in posterior_specs
    ]
    model_names = [item.name for item in posterior_inputs]
    if len(model_names) != len(set(model_names)):
        raise ValueError(f"Posterior model names must be unique: {model_names}")

    initial_manifest = {
        "status": "running",
        "truth_path": str(truth_file),
        "posterior_sources": {
            item.name: [str(path) for path in item.files] for item in posterior_inputs
        },
        "parameters": list(parameters),
        "num_regions": int(num_regions),
        "num_bootstrap": int(num_bootstrap),
        "samples_per_object_requested": samples_per_object,
        "seed": int(seed),
        "limit": limit,
        "paper": MIRA_PAPER_URL,
        "upstream_reference_commit": MIRA_UPSTREAM_COMMIT,
    }
    write_json(out / "mira_manifest.json", initial_manifest)

    truth = _read_truth(truth_file, parameters, limit=limit)
    companion_truths = _validate_companion_truths(
        truth,
        truth_file,
        posterior_inputs,
        parameters,
        limit=limit,
    )
    dense_models = [
        _read_dense_posterior(
            item,
            truth,
            parameters,
            samples_per_object=samples_per_object,
            require_exact_object_set=limit is None,
        )
        for item in posterior_inputs
    ]
    sample_counts = {model.values.shape[1] for model in dense_models}
    if len(sample_counts) != 1:
        raise ValueError(
            "All posterior models must use the same number of samples; "
            f"received {sorted(sample_counts)}"
        )
    posterior = np.stack([model.values for model in dense_models], axis=0)
    truth_values = truth.loc[:, parameters].to_numpy(dtype=np.float32)
    lower = truth_values.min(axis=0)
    upper = truth_values.max(axis=0)
    truth_range = upper - lower
    constant = [
        parameters[index]
        for index, value in enumerate(truth_range)
        if not np.isfinite(value) or value <= 0
    ]
    if constant:
        raise ValueError(
            f"Truth normalization has constant/non-finite dimensions: {constant}"
        )
    scale = truth_range + np.float32(1.0e-8)
    truth_normalized = (truth_values - lower) / scale
    posterior_normalized = (posterior - lower[None, None, None, :]) / scale[
        None, None, None, :
    ]
    _write_normalization_diagnostics(
        out,
        parameters,
        lower,
        upper,
        scale,
        posterior_normalized,
        model_names,
    )

    group_definitions = feniks_mira_groups(parameters)
    score_rows: list[dict[str, Any]] = []
    region_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    contribution_frames: list[pd.DataFrame] = []
    pairwise_rows: list[dict[str, Any]] = []
    object_ids = truth["object_id"].to_numpy()
    row_indices = (
        truth["row_index"].to_numpy()
        if "row_index" in truth
        else np.arange(len(truth), dtype=np.int64)
    )

    for group_index, (group_name, indices) in enumerate(group_definitions.items()):
        group_seed = _derived_seed(seed, group_index, 11)
        bootstrap_seed = _derived_seed(seed, group_index, 29)
        rng = np.random.default_rng(group_seed)
        centers = rng.uniform(
            0.0,
            1.0,
            size=(num_regions, len(truth), len(indices)),
        ).astype(np.float32)
        reference_indices = rng.integers(
            0,
            posterior.shape[2],
            size=(num_regions, len(truth)),
            dtype=np.int32,
        )
        contributions = np.asarray(
            jax.device_get(
                mira_region_contributions(
                    jnp.asarray(truth_normalized[:, indices]),
                    jnp.asarray(posterior_normalized[..., indices]),
                    jnp.asarray(centers),
                    jnp.asarray(reference_indices),
                )
            ),
            dtype=np.float64,
        )
        region_scores = contributions.mean(axis=2)
        object_contributions = contributions.mean(axis=0)
        bootstrap_scores = _bootstrap_object_scores(
            object_contributions,
            num_bootstrap=num_bootstrap,
            seed=bootstrap_seed,
        )

        for model_index, model_name in enumerate(model_names):
            model_bootstrap = bootstrap_scores[:, model_index]
            score = float(object_contributions[model_index].mean())
            score_rows.append(
                {
                    "model": model_name,
                    "group": group_name,
                    "dimensions": len(indices),
                    "parameters": ",".join(parameters[index] for index in indices),
                    "num_objects": len(truth),
                    "num_posterior_samples": posterior.shape[2],
                    "num_regions": num_regions,
                    "score": score,
                    "ideal_score": MIRA_IDEAL_SCORE,
                    "delta_from_ideal": score - MIRA_IDEAL_SCORE,
                    "theoretical_sigma": float(np.sqrt(1.0 / (18.0 * len(truth)))),
                    "region_mc_std": float(
                        region_scores[:, model_index].std(ddof=1)
                        if num_regions > 1
                        else 0.0
                    ),
                    **_bootstrap_summary(model_bootstrap),
                    "group_seed": int(group_seed),
                    "bootstrap_seed": int(bootstrap_seed),
                }
            )
            region_frames.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "group": group_name,
                        "region_id": np.arange(num_regions, dtype=np.int32),
                        "score": region_scores[:, model_index],
                    }
                )
            )
            contribution = pd.DataFrame(
                {
                    "model": model_name,
                    "group": group_name,
                    "object_id": object_ids,
                    "row_index": row_indices,
                    "mira_contribution": object_contributions[model_index],
                }
            )
            contribution_frames.append(contribution)
            if num_bootstrap:
                bootstrap_frames.append(
                    pd.DataFrame(
                        {
                            "model": model_name,
                            "group": group_name,
                            "bootstrap_id": np.arange(num_bootstrap, dtype=np.int32),
                            "score": model_bootstrap,
                        }
                    )
                )

        for first, second in itertools.combinations(range(len(model_names)), 2):
            delta = object_contributions[first] - object_contributions[second]
            bootstrap_delta = bootstrap_scores[:, first] - bootstrap_scores[:, second]
            pairwise_rows.append(
                {
                    "group": group_name,
                    "model_a": model_names[first],
                    "model_b": model_names[second],
                    "score_a_minus_b": float(delta.mean()),
                    **_bootstrap_summary(bootstrap_delta, prefix="delta_"),
                }
            )

    scores = pd.DataFrame(score_rows)
    regions = pd.concat(region_frames, ignore_index=True)
    contributions_frame = pd.concat(contribution_frames, ignore_index=True)
    bootstrap_frame = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame(columns=["model", "group", "bootstrap_id", "score"])
    )
    pairwise = pd.DataFrame(pairwise_rows)
    scores.to_csv(out / "mira_scores.csv", index=False)
    scores.to_parquet(out / "mira_scores.parquet", index=False)
    regions.to_parquet(out / "mira_region_scores.parquet", index=False)
    contributions_frame.to_parquet(
        out / "mira_object_contributions.parquet", index=False
    )
    bootstrap_frame.to_parquet(out / "mira_bootstrap_scores.parquet", index=False)
    pairwise.to_csv(out / "mira_pairwise_differences.csv", index=False)
    _write_score_plot(scores, out / "mira_scores.png")

    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "models": model_names,
        "num_objects": len(truth),
        "num_posterior_samples": int(posterior.shape[2]),
        "num_regions": int(num_regions),
        "num_bootstrap": int(num_bootstrap),
        "companion_truths_checked": sum(
            record["status"] in {"primary_reference", "validated"}
            for record in companion_truths.values()
        ),
        "score_groups": list(group_definitions),
        "full_15d": scores.loc[
            scores["group"].eq("full_15d"),
            [
                "model",
                "score",
                "bootstrap_q025",
                "bootstrap_q975",
                "delta_from_ideal",
            ],
        ].to_dict(orient="records"),
        "ideal_score": MIRA_IDEAL_SCORE,
        "theoretical_sigma": float(np.sqrt(1.0 / (18.0 * len(truth)))),
        "elapsed_seconds": float(elapsed),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "outputs": {
            "scores_csv": "mira_scores.csv",
            "scores_parquet": "mira_scores.parquet",
            "region_scores": "mira_region_scores.parquet",
            "object_contributions": "mira_object_contributions.parquet",
            "bootstrap_scores": "mira_bootstrap_scores.parquet",
            "pairwise_differences": "mira_pairwise_differences.csv",
            "normalization": "mira_normalization.csv",
            "normalization_diagnostics": "mira_normalization_diagnostics.csv",
            "plot": "mira_scores.png",
        },
    }
    write_json(out / "mira_summary.json", summary)
    manifest = {
        **initial_manifest,
        **summary,
        "truth_file": _file_record(truth_file),
        "posterior_files": {
            model.name: [_file_record(path) for path in model.files]
            for model in dense_models
        },
        "companion_truths": companion_truths,
        "normalization": {
            "type": "truth_minmax",
            "range_epsilon": 1.0e-8,
            "posterior_clipped": False,
        },
        "distance": "euclidean_squared_equivalent",
        "center_distribution": "uniform_unit_hypercube",
        "shared_random_regions_across_models": True,
        "reference_draw": "uniform_posterior_sample_index",
        "finite_sample_normalization": "divide by S/(S+1)",
        "bootstrap_unit": "held_out_object",
        "git_sha": _git_sha(),
    }
    write_json(out / "mira_manifest.json", manifest)
    (out / "DONE").touch()
    return summary


def _read_truth(
    path: Path,
    parameters: Sequence[str],
    *,
    limit: int | None,
) -> pd.DataFrame:
    truth = pd.read_parquet(path)
    required = {"object_id", *parameters}
    missing = sorted(required - set(truth.columns))
    if missing:
        raise ValueError(f"Truth parquet is missing columns: {missing}")
    keep = ["object_id"]
    if "row_index" in truth:
        keep.append("row_index")
    keep.extend(parameters)
    truth = truth.loc[:, keep]
    if truth["object_id"].isna().any() or truth["object_id"].duplicated().any():
        raise ValueError("Truth object_id values must be finite and unique")
    if "row_index" in truth and truth["row_index"].duplicated().any():
        raise ValueError("Truth row_index values must be unique")
    if limit is not None:
        truth = truth.iloc[:limit]
    values = truth.loc[:, parameters].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Truth parameter matrix contains non-finite values")
    if len(truth) < 2:
        raise ValueError("MIRA requires at least two held-out truth objects")
    return truth.reset_index(drop=True)


def _validate_companion_truths(
    truth: pd.DataFrame,
    truth_file: Path,
    posterior_inputs: Sequence[PosteriorInput],
    parameters: Sequence[str],
    *,
    limit: int | None,
) -> dict[str, dict[str, Any]]:
    """Check truth tables shipped beside posterior sources against the reference."""
    records: dict[str, dict[str, Any]] = {}
    truth_ids = truth["object_id"].tolist()
    truth_id_set = set(truth_ids)
    for source in posterior_inputs:
        candidate = resolve_companion_truth(source)
        if candidate is None:
            records[source.name] = {"status": "not_found"}
            continue
        record = _file_record(candidate)
        if candidate.resolve() == truth_file.resolve():
            records[source.name] = {"status": "primary_reference", **record}
            continue

        companion = _read_truth(candidate, parameters, limit=None)
        companion_ids = set(companion["object_id"].tolist())
        missing = truth_id_set - companion_ids
        extra = companion_ids - truth_id_set
        if missing or (limit is None and extra):
            raise ValueError(
                f"Companion truth for posterior {source.name!r} has a different "
                f"object set: missing={len(missing)}, extra={len(extra)}"
            )
        companion = (
            companion.set_index("object_id", drop=False)
            .loc[truth_ids]
            .reset_index(drop=True)
        )
        if ("row_index" in truth) != ("row_index" in companion):
            raise ValueError(
                f"Companion truth for posterior {source.name!r} has an incompatible "
                "row_index contract"
            )
        if "row_index" in truth and not np.array_equal(
            truth["row_index"].to_numpy(), companion["row_index"].to_numpy()
        ):
            raise ValueError(
                f"Companion truth for posterior {source.name!r} has different "
                "row_index values"
            )
        reference_values = truth.loc[:, parameters].to_numpy(dtype=np.float64)
        companion_values = companion.loc[:, parameters].to_numpy(dtype=np.float64)
        if not np.array_equal(reference_values, companion_values):
            max_delta = float(np.max(np.abs(reference_values - companion_values)))
            raise ValueError(
                f"Companion truth for posterior {source.name!r} differs from the "
                f"reference truth (max_abs_delta={max_delta:.8g})"
            )
        records[source.name] = {
            "status": "validated",
            "num_objects_compared": len(truth),
            **record,
        }
    return records


def _read_dense_posterior(
    source: PosteriorInput,
    truth: pd.DataFrame,
    parameters: Sequence[str],
    *,
    samples_per_object: int | None,
    require_exact_object_set: bool,
) -> DensePosterior:
    required = {"object_id", "sample_id", *parameters}
    schemas = [set(pq.read_schema(path).names) for path in source.files]
    for path, names in zip(source.files, schemas, strict=True):
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"Posterior parquet {path} is missing columns: {missing}")
    include_row_index = "row_index" in truth and all(
        "row_index" in names for names in schemas
    )
    columns = ["object_id"]
    if include_row_index:
        columns.append("row_index")
    columns.extend(["sample_id", *parameters])
    frames = [pd.read_parquet(path, columns=columns) for path in source.files]
    posterior = pd.concat(frames, ignore_index=True)
    truth_ids = truth["object_id"]
    truth_id_set = set(truth_ids.tolist())
    posterior_id_set = set(posterior["object_id"].drop_duplicates().tolist())
    missing_ids = truth_id_set - posterior_id_set
    extra_ids = posterior_id_set - truth_id_set
    if missing_ids:
        examples = list(missing_ids)[:5]
        raise ValueError(
            f"Posterior {source.name!r} is missing {len(missing_ids)} truth IDs: "
            f"{examples}"
        )
    if require_exact_object_set and extra_ids:
        examples = list(extra_ids)[:5]
        raise ValueError(
            f"Posterior {source.name!r} has {len(extra_ids)} unexpected IDs: {examples}"
        )
    posterior = posterior.loc[posterior["object_id"].isin(truth_id_set)].copy()
    if posterior.duplicated(["object_id", "sample_id"]).any():
        raise ValueError(
            f"Posterior {source.name!r} has duplicate object_id/sample_id rows"
        )
    all_sample_ids = tuple(sorted(posterior["sample_id"].drop_duplicates().tolist()))
    if samples_per_object is not None:
        if len(all_sample_ids) < samples_per_object:
            raise ValueError(
                f"Posterior {source.name!r} has only {len(all_sample_ids)} sample "
                f"IDs, fewer than requested {samples_per_object}"
            )
        selected_sample_ids = all_sample_ids[:samples_per_object]
    else:
        selected_sample_ids = all_sample_ids
    if len(selected_sample_ids) < 2:
        raise ValueError(
            f"Posterior {source.name!r} must provide at least two samples per object"
        )
    posterior = posterior.loc[posterior["sample_id"].isin(selected_sample_ids)].copy()

    object_order = {value: index for index, value in enumerate(truth_ids.tolist())}
    sample_order = {value: index for index, value in enumerate(selected_sample_ids)}
    posterior["_object_order"] = posterior["object_id"].map(object_order)
    posterior["_sample_order"] = posterior["sample_id"].map(sample_order)
    posterior = posterior.sort_values(["_object_order", "_sample_order"], kind="stable")
    n_objects = len(truth)
    n_samples = len(selected_sample_ids)
    if len(posterior) != n_objects * n_samples:
        counts = posterior.groupby("object_id", sort=False)["sample_id"].size()
        bad = counts.loc[counts.ne(n_samples)]
        raise ValueError(
            f"Posterior {source.name!r} has incomplete per-object samples; "
            f"expected {n_samples}, examples={bad.head().to_dict()}"
        )
    object_grid = posterior["_object_order"].to_numpy().reshape(n_objects, n_samples)
    sample_grid = posterior["_sample_order"].to_numpy().reshape(n_objects, n_samples)
    if not np.array_equal(
        object_grid,
        np.arange(n_objects, dtype=object_grid.dtype)[:, None]
        + np.zeros((1, n_samples), dtype=object_grid.dtype),
    ):
        raise ValueError(f"Posterior {source.name!r} object ordering is inconsistent")
    if not np.array_equal(
        sample_grid,
        np.arange(n_samples, dtype=sample_grid.dtype)[None, :]
        + np.zeros((n_objects, 1), dtype=sample_grid.dtype),
    ):
        raise ValueError(
            f"Posterior {source.name!r} sample IDs are inconsistent across objects"
        )
    if include_row_index:
        posterior_rows = posterior["row_index"].to_numpy().reshape(n_objects, n_samples)
        truth_rows = truth["row_index"].to_numpy()
        if not np.all(posterior_rows == truth_rows[:, None]):
            raise ValueError(
                f"Posterior {source.name!r} row_index does not match truth"
            )
    values = (
        posterior.loc[:, parameters]
        .to_numpy(dtype=np.float32)
        .reshape(
            n_objects,
            n_samples,
            len(parameters),
        )
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Posterior {source.name!r} parameter matrix contains non-finite values"
        )
    return DensePosterior(
        name=source.name,
        values=values,
        sample_ids=selected_sample_ids,
        files=source.files,
    )


def _bootstrap_object_scores(
    object_contributions: np.ndarray,
    *,
    num_bootstrap: int,
    seed: int,
    batch_size: int = 100,
) -> np.ndarray:
    n_models, n_objects = object_contributions.shape
    if num_bootstrap == 0:
        return np.empty((0, n_models), dtype=np.float64)
    rng = np.random.default_rng(seed)
    scores = np.empty((num_bootstrap, n_models), dtype=np.float64)
    for start in range(0, num_bootstrap, batch_size):
        stop = min(start + batch_size, num_bootstrap)
        indices = rng.integers(
            0,
            n_objects,
            size=(stop - start, n_objects),
        )
        scores[start:stop] = object_contributions[:, indices].mean(axis=2).T
    return scores


def _bootstrap_summary(
    values: np.ndarray,
    *,
    prefix: str = "bootstrap_",
) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}mean": float("nan"),
            f"{prefix}std": float("nan"),
            f"{prefix}q025": float("nan"),
            f"{prefix}q16": float("nan"),
            f"{prefix}q84": float("nan"),
            f"{prefix}q975": float("nan"),
        }
    return {
        f"{prefix}mean": float(np.mean(values)),
        f"{prefix}std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}q025": float(np.quantile(values, 0.025)),
        f"{prefix}q16": float(np.quantile(values, 0.16)),
        f"{prefix}q84": float(np.quantile(values, 0.84)),
        f"{prefix}q975": float(np.quantile(values, 0.975)),
    }


def _write_normalization_diagnostics(
    out: Path,
    parameters: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    divisor: np.ndarray,
    posterior_normalized: np.ndarray,
    model_names: Sequence[str],
) -> None:
    normalization = pd.DataFrame(
        {
            "parameter": parameters,
            "truth_min": lower,
            "truth_max": upper,
            "truth_range": upper - lower,
            "normalization_divisor": divisor,
        }
    )
    normalization.to_csv(out / "mira_normalization.csv", index=False)
    rows = []
    for model_index, model_name in enumerate(model_names):
        values = posterior_normalized[model_index]
        for parameter_index, parameter in enumerate(parameters):
            coordinate = values[..., parameter_index]
            rows.append(
                {
                    "model": model_name,
                    "parameter": parameter,
                    "fraction_below_truth_min": float(np.mean(coordinate < 0.0)),
                    "fraction_above_truth_max": float(np.mean(coordinate > 1.0)),
                    "normalized_min": float(np.min(coordinate)),
                    "normalized_max": float(np.max(coordinate)),
                }
            )
    pd.DataFrame(rows).to_csv(out / "mira_normalization_diagnostics.csv", index=False)


def _write_score_plot(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = scores["group"].drop_duplicates().tolist()
    models = scores["model"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [1, 2.4]},
        constrained_layout=True,
    )
    for axis, selected_groups, title in (
        (axes[0], groups[:3], "Joint latent-space scores"),
        (axes[1], groups[3:], "One-dimensional marginal scores"),
    ):
        x = np.arange(len(selected_groups), dtype=float)
        width = 0.7 / max(len(models), 1)
        for model_index, model in enumerate(models):
            subset = (
                scores.loc[
                    scores["model"].eq(model) & scores["group"].isin(selected_groups)
                ]
                .set_index("group")
                .loc[selected_groups]
            )
            offset = (model_index - (len(models) - 1) / 2) * width
            lower = subset["score"] - subset["bootstrap_q025"]
            upper = subset["bootstrap_q975"] - subset["score"]
            axis.errorbar(
                x + offset,
                subset["score"],
                yerr=np.vstack([lower, upper]),
                marker="o",
                linestyle="none",
                capsize=3,
                label=model,
            )
        sigma = float(scores["theoretical_sigma"].iloc[0])
        axis.axhspan(
            MIRA_IDEAL_SCORE - sigma,
            MIRA_IDEAL_SCORE + sigma,
            color="#d8d8d8",
            alpha=0.65,
            linewidth=0,
        )
        axis.axhline(MIRA_IDEAL_SCORE, color="#202020", linewidth=1.1)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [group.replace("marginal_", "") for group in selected_groups],
            rotation=35 if len(selected_groups) > 5 else 0,
            ha="right" if len(selected_groups) > 5 else "center",
        )
        axis.set_ylabel("MIRA score")
        axis.set_title(title)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axes[0].legend(frameon=False, ncol=max(1, min(len(models), 3)))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _derived_seed(seed: int, group_index: int, stream: int) -> int:
    sequence = np.random.SeedSequence([int(seed), int(group_index), int(stream)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
