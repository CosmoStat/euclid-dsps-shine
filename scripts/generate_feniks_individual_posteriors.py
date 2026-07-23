#!/usr/bin/env python3
"""Sample and plot per-galaxy posteriors from a trained FENIKS encoder."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd

from euclid_dsps.amortized.catalog import (
    learned_prior_samples_frame,
    posterior_samples_frame,
)
from euclid_dsps.amortized.catalog_identity import (
    available_columns,
    object_id_column_from_config,
    write_truth_snapshot,
)
from euclid_dsps.amortized.data import (
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from euclid_dsps.amortized.diagnostics import (
    _truth_parameter_frame,
    _write_multi_overlay_corner_plot,
)
from euclid_dsps.amortized.features import read_feature_stats
from euclid_dsps.amortized.latent import x_to_theta
from euclid_dsps.amortized.posterior import sample_posterior
from euclid_dsps.amortized.train import (
    _latent_spec_for_amortized_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.io import ensure_dir, truth_column_from_spec, write_json

EXAMPLE_DEFINITIONS = (
    ("typical", "Galaxy closest to the robust 15D population center"),
    ("nearby", "Low-redshift example (3rd percentile)"),
    ("high_z", "High-redshift-tail example (98.5th percentile)"),
    ("massive", "Massive stellar galaxy (99.5th percentile)"),
    ("dusty", "Strongly attenuated galaxy (99.7th percentile)"),
    ("quenched", "Lowest-sSFR galaxy"),
    ("star_forming", "Highest-sSFR galaxy"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--posterior-samples", type=int, default=2048)
    parser.add_argument("--prior-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=260723)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _nearest_unused(
    values: np.ndarray,
    target: float,
    *,
    allowed: np.ndarray,
    used: set[int],
) -> int:
    order = np.argsort(np.abs(np.asarray(values, dtype=float) - float(target)))
    for raw_index in order:
        index = int(raw_index)
        if allowed[index] and index not in used:
            used.add(index)
            return index
    raise ValueError("No unused finite row satisfies the representative selection")


def _extreme_unused(
    values: np.ndarray,
    *,
    largest: bool,
    used: set[int],
) -> int:
    finite = np.isfinite(values)
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    for raw_index in order:
        index = int(raw_index)
        if finite[index] and index not in used:
            used.add(index)
            return index
    raise ValueError("No unused finite row satisfies the extreme selection")


def _truth_columns(config: dict[str, Any]) -> dict[str, str]:
    specs = dict((config.get("truth", {}) or {}).get("parameter_columns") or {})
    return {
        str(name): str(column)
        for name, spec in specs.items()
        if (column := truth_column_from_spec(spec))
    }


def _ssfr_column(columns: set[str]) -> str:
    for column in ("logssfr_true", "log10_ssfr_at_obs", "logsSFR"):
        if column in columns:
            return column
    raise ValueError(
        "Representative quenched/star-forming selection requires one of "
        "logssfr_true, log10_ssfr_at_obs, or logsSFR"
    )


def select_representative_rows(
    config: dict[str, Any],
    dataset: str | Path,
) -> pd.DataFrame:
    """Return deterministic row indices for the explorer's seven example classes."""
    dataset = Path(dataset)
    columns = available_columns(dataset)
    truth_columns = _truth_columns(config)
    required_names = tuple((config.get("fit", {}) or {}).get("free_parameters", {}))
    missing_names = [name for name in required_names if name not in truth_columns]
    if missing_names:
        raise ValueError(
            "Configured free parameters lack truth columns: " + ", ".join(missing_names)
        )
    requested = [truth_columns[name] for name in required_names]
    ssfr_column = _ssfr_column(columns)
    requested.append(ssfr_column)
    id_column = object_id_column_from_config(config)
    if id_column and id_column in columns:
        requested.append(id_column)
    missing_columns = sorted(set(requested) - columns)
    if missing_columns:
        raise ValueError(
            f"{dataset} is missing selection columns: " + ", ".join(missing_columns)
        )
    frame = pd.read_parquet(dataset, columns=sorted(set(requested)))
    if frame.empty:
        raise ValueError(
            f"Cannot select representative rows from empty catalog: {dataset}"
        )

    truth_matrix = np.column_stack(
        [_numeric(frame, truth_columns[name]) for name in required_names]
    )
    finite_truth = np.isfinite(truth_matrix).all(axis=1)
    median = np.nanmedian(truth_matrix, axis=0)
    q16, q84 = np.nanquantile(truth_matrix, [0.16, 0.84], axis=0)
    scale = np.maximum(q84 - q16, 1.0e-6)
    distance = np.sqrt(np.nanmean(((truth_matrix - median) / scale) ** 2, axis=1))

    used: set[int] = set()
    selected: dict[str, int] = {}
    selected["typical"] = _nearest_unused(
        distance, 0.0, allowed=finite_truth, used=used
    )
    z = _numeric(frame, truth_columns["z_obs"])
    finite_z = np.isfinite(z)
    selected["nearby"] = _nearest_unused(
        z, np.nanquantile(z, 0.03), allowed=finite_z, used=used
    )
    selected["high_z"] = _nearest_unused(
        z, np.nanquantile(z, 0.985), allowed=finite_z, used=used
    )
    mass = _numeric(frame, truth_columns["log10_stellar_mass"])
    selected["massive"] = _nearest_unused(
        mass, np.nanquantile(mass, 0.995), allowed=np.isfinite(mass), used=used
    )
    dust = _numeric(frame, truth_columns["dust_av"])
    selected["dusty"] = _nearest_unused(
        dust, np.nanquantile(dust, 0.997), allowed=np.isfinite(dust), used=used
    )
    ssfr = _numeric(frame, ssfr_column)
    selected["quenched"] = _extreme_unused(ssfr, largest=False, used=used)
    selected["star_forming"] = _extreme_unused(ssfr, largest=True, used=used)

    descriptions = dict(EXAMPLE_DEFINITIONS)
    rows = []
    for order, (key, _description) in enumerate(EXAMPLE_DEFINITIONS, start=1):
        row_index = selected[key]
        rows.append(
            {
                "order": order,
                "example_key": key,
                "description": descriptions[key],
                "row_index": row_index,
                "object_id": (
                    frame.iloc[row_index][id_column]
                    if id_column and id_column in frame
                    else row_index
                ),
                "z_true": z[row_index],
                "log10_stellar_mass_true": mass[row_index],
                "dust_av_true": dust[row_index],
                "log10_ssfr_true": ssfr[row_index],
            }
        )
    return pd.DataFrame(rows)


def _safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "object"


def _write_corners(
    posterior: pd.DataFrame,
    prior: pd.DataFrame,
    truth: pd.DataFrame,
    selection: pd.DataFrame,
    out: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []
    for item in selection.itertuples(index=False):
        row_index = int(item.row_index)
        object_posterior = posterior.loc[posterior["row_index"] == row_index]
        object_truth = truth.loc[truth["row_index"] == row_index]
        if object_posterior.empty:
            raise ValueError(f"No posterior samples for row_index={row_index}")
        if len(object_truth) != 1:
            raise ValueError(
                f"Expected one truth row for row_index={row_index}, got {len(object_truth)}"
            )
        filename = (
            f"{int(item.order):02d}_{item.example_key}_"
            f"{_safe_name(item.object_id)}_corner.png"
        )
        path = _write_multi_overlay_corner_plot(
            object_posterior,
            out,
            plt,
            truth=object_truth,
            prior=prior,
            filename=filename,
            title=(
                f"{item.example_key}: object {item.object_id}, catalog row {row_index}"
            ),
            posterior_label="individual encoder posterior",
            config=config,
        )
        if path is None:
            raise ValueError(f"Corner plot was not produced for row_index={row_index}")
        records.append(
            {
                "order": int(item.order),
                "example_key": item.example_key,
                "object_id": item.object_id,
                "row_index": row_index,
                "corner": path.name,
                "posterior_rows": int(len(object_posterior)),
                "truth_rows": int(len(object_truth)),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    if args.posterior_samples <= 0 or args.prior_samples <= 0:
        raise ValueError("posterior-samples and prior-samples must be positive")
    out = Path(args.out)
    done = out / "DONE"
    if done.exists() and not args.overwrite:
        raise FileExistsError(f"Output is already complete: {out}")
    ensure_dir(out)

    config = load_config(args.config)
    config["catalog_path"] = str(args.dataset)
    selection = select_representative_rows(config, args.dataset)
    selection.to_csv(out / "selected_galaxies.csv", index=False)
    (out / "selected_row_indices.txt").write_text(
        "".join(f"{int(value)}\n" for value in selection["row_index"]),
        encoding="utf-8",
    )
    row_indices = selection["row_index"].to_numpy(dtype=np.int64)
    raw_truth = write_truth_snapshot(
        out,
        config,
        row_indices=row_indices,
        limit=None,
    )

    feature_stats = read_feature_stats(args.feature_stats)
    model = load_checkpoint(args.checkpoint, config)
    latent_spec = _latent_spec_for_amortized_config(config)
    arrays = load_photometry_arrays_from_config(
        config,
        batch_size=10_000,
        row_indices=row_indices,
    )
    key = jax.random.PRNGKey(int(args.seed))
    posterior_frames = []
    for batch in iter_photometry_batches_from_arrays(
        arrays,
        batch_size=len(row_indices),
        feature_stats=feature_stats,
    ):
        key, sample_key = jax.random.split(key)
        sample = sample_posterior(
            model,
            sample_key,
            batch.features,
            int(args.posterior_samples),
        )
        theta = x_to_theta(sample.x, latent_spec)
        shape = sample.logq.shape
        posterior_frames.append(
            posterior_samples_frame(
                batch.object_id,
                jax.device_get(theta),
                latent_spec.names,
                jax.device_get(sample.logq),
                jax.device_get(sample.logprior),
                np.full(shape, np.nan, dtype=float),
                row_index=batch.row_index,
            )
        )
    posterior = pd.concat(posterior_frames, ignore_index=True)
    posterior.to_parquet(out / "posterior_samples.parquet", index=False)

    key, prior_key = jax.random.split(key)
    prior_x = model.prior.sample(prior_key, int(args.prior_samples))
    prior_theta = x_to_theta(prior_x, latent_spec)
    prior = learned_prior_samples_frame(
        jax.device_get(prior_x),
        jax.device_get(prior_theta),
        latent_spec.names,
        jax.device_get(model.prior.log_prob(prior_x)),
    )
    prior.to_parquet(out / "learned_prior_samples.parquet", index=False)

    identity = raw_truth[["row_index", "object_id"]].copy()
    truth_summary = _truth_parameter_frame(identity, out, config=config)
    truth_summary.insert(0, "row_index", identity["row_index"].to_numpy())
    truth_summary.insert(1, "object_id", identity["object_id"].to_numpy())
    truth_summary.to_parquet(out / "individual_truth.parquet", index=False)
    corners = _write_corners(
        posterior,
        prior,
        truth_summary,
        selection,
        out,
        config,
    )

    summary = {
        "config": str(args.config),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "feature_stats": str(args.feature_stats),
        "seed": int(args.seed),
        "posterior_samples_per_object": int(args.posterior_samples),
        "prior_samples": int(args.prior_samples),
        "n_objects": int(len(selection)),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "corners": corners,
    }
    write_json(out / "individual_posterior_manifest.json", summary)
    write_json(out / "normalized_config.json", config)
    done.touch()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
