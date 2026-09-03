#!/usr/bin/env python3
"""Finalize projected-parent individual posterior support and closure diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from euclid_dsps.amortized.catalog_identity import write_truth_snapshot
from euclid_dsps.amortized.diagnostics import _write_multi_overlay_corner_plot
from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.config import load_config
from euclid_dsps.io import truth_column_from_spec, truth_value_from_spec

try:
    from scripts.evaluate_redshift_pit_coverage import evaluate as evaluate_redshift
except ModuleNotFoundError:
    from evaluate_redshift_pit_coverage import evaluate as evaluate_redshift


CORE_PARAMETERS = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _runtime_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _validate_runtime_provenance(
    root: Path, manifest: dict[str, Any], repo: Path
) -> dict[str, Any]:
    actual_commit = _runtime_commit(repo)
    manifest_commit = str(manifest["code_commit"])
    if actual_commit == manifest_commit:
        return {
            "mode": "manifest",
            "manifest_code_commit": manifest_commit,
            "runtime_code_commit": actual_commit,
        }
    raw_recovery = os.environ.get("FINALIZER_RECOVERY_RECEIPT")
    if not raw_recovery:
        raise ValueError(
            f"runtime commit mismatch: {actual_commit} != {manifest_commit}"
        )
    recovery_path = Path(raw_recovery).resolve()
    recovery = _read_json(recovery_path)
    manifest_path = root / "RUN_MANIFEST.json"
    expected = {
        "status": "AUTHORIZED",
        "scope": "finalizer_only_nonfinite_json_recovery",
        "inference_code_commit": manifest_commit,
        "finalizer_code_commit": actual_commit,
        "run_manifest_sha256": sha256_file(manifest_path),
        "inference_shards_reused": True,
        "new_inference_submitted": False,
    }
    mismatches = {
        key: {"expected": value, "actual": recovery.get(key)}
        for key, value in expected.items()
        if recovery.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid finalizer recovery authorization: {mismatches}")
    return {
        "mode": "authorized_finalizer_recovery",
        "manifest_code_commit": manifest_commit,
        "runtime_code_commit": actual_commit,
        "authorization": str(recovery_path),
        "authorization_sha256": sha256_file(recovery_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _read_parquet_files(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("no parquet inputs")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _validate_counts(
    frame: pd.DataFrame,
    cohort: np.ndarray,
    expected_draws: int,
    label: str,
) -> None:
    if "row_index" not in frame or "sample_id" not in frame:
        raise ValueError(f"{label} lacks row_index/sample_id")
    counts = frame.groupby("row_index", sort=False)["sample_id"].nunique()
    if set(counts.index.astype(int)) != set(np.asarray(cohort, dtype=int)):
        raise ValueError(f"{label} object cohort mismatch")
    if not counts.eq(int(expected_draws)).all():
        raise ValueError(
            f"{label} requires {expected_draws} unique draws per object; "
            f"observed={sorted(counts.unique().tolist())}"
        )


def _support_summary(diagnostics: pd.DataFrame) -> dict[str, Any]:
    raw_fraction = pd.to_numeric(
        diagnostics["raw_ess_fraction"], errors="coerce"
    ).to_numpy(dtype=float)
    raw_ess = pd.to_numeric(diagnostics["raw_ess"], errors="coerce").to_numpy(
        dtype=float
    )
    pareto = pd.to_numeric(diagnostics["pareto_k"], errors="coerce").to_numpy(
        dtype=float
    )
    max_weight = pd.to_numeric(diagnostics["max_raw_weight"], errors="coerce").to_numpy(
        dtype=float
    )
    finite = np.isfinite(raw_fraction)
    if not finite.any():
        raise ValueError("importance diagnostics contain no finite ESS")
    median_fraction = float(np.nanmedian(raw_fraction))
    bad_k = float(np.mean(~np.isfinite(pareto) | (pareto > 0.7)))
    p90_weight = float(np.nanquantile(max_weight, 0.90))
    passed = median_fraction >= 0.05 and bad_k <= 0.20 and p90_weight <= 0.80
    return {
        "status": "PASS" if passed else "FAIL",
        "objects": int(len(diagnostics)),
        "median_raw_ess": float(np.nanmedian(raw_ess)),
        "q10_raw_ess": float(np.nanquantile(raw_ess, 0.10)),
        "median_raw_ess_fraction": median_fraction,
        "q10_raw_ess_fraction": float(np.nanquantile(raw_fraction, 0.10)),
        "fraction_pareto_k_gt_0p7": bad_k,
        "fraction_pareto_k_gt_1": float(np.mean(~np.isfinite(pareto) | (pareto > 1.0))),
        "median_pareto_k": float(np.nanmedian(pareto)),
        "p90_max_raw_weight": p90_weight,
        "maximum_raw_weight": float(np.nanmax(max_weight)),
        "thresholds": {
            "minimum_median_raw_ess_fraction": 0.05,
            "maximum_fraction_pareto_k_gt_0p7": 0.20,
            "maximum_p90_raw_weight": 0.80,
        },
    }


def _truth_frame(
    *,
    evaluation: Path,
    config: dict[str, Any],
    cohort: np.ndarray,
    parameter_names: tuple[str, ...],
) -> pd.DataFrame:
    raw = write_truth_snapshot(
        evaluation,
        config,
        row_indices=cohort,
        limit=None,
        filename="inference_truth.parquet",
    )
    if len(raw) != len(cohort):
        raise ValueError(f"expected {len(cohort)} truth rows, found {len(raw)}")
    specs = dict((config.get("truth", {}) or {}).get("parameter_columns") or {})
    missing = [name for name in parameter_names if name not in specs]
    if missing:
        raise ValueError(f"truth closure config lacks parameters: {missing}")
    rows = []
    for record in raw.to_dict(orient="records"):
        row = {
            "row_index": int(record["row_index"]),
            "object_id": record.get("object_id", record["row_index"]),
        }
        for name in parameter_names:
            column = truth_column_from_spec(specs[name])
            if not column or column not in record:
                raise ValueError(f"truth column for {name} is unavailable")
            row[name] = truth_value_from_spec(record, specs[name])
        rows.append(row)
    truth = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True)
    if not np.isfinite(truth[list(parameter_names)].to_numpy(dtype=float)).all():
        raise ValueError("truth closure contains non-finite latent parameters")
    truth.to_parquet(evaluation / "individual_truth.parquet", index=False)
    return truth


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    values = values[finite]
    weights = weights[finite]
    if not len(values) or weights.sum() <= 0.0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(np.interp(float(quantile), cumulative, values))


def _ppc_summary(
    residuals: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weight_columns = weights[["row_index", "sample_id", "psis_weight"]].copy()
    merged = residuals.merge(
        weight_columns,
        on=["row_index", "sample_id"],
        how="inner",
        validate="many_to_one",
    )
    merged = merged[merged["valid"].astype(bool)].copy()
    merged["model"] = label
    rows = []
    for band, group in merged.groupby("band", sort=True):
        value = pd.to_numeric(group["chi_likelihood"], errors="coerce").to_numpy(
            dtype=float
        )
        weight = pd.to_numeric(group["psis_weight"], errors="coerce").to_numpy(
            dtype=float
        )
        finite = np.isfinite(value) & np.isfinite(weight) & (weight >= 0.0)
        value = value[finite]
        weight = weight[finite]
        weight = weight / weight.sum()
        rows.append(
            {
                "model": label,
                "band": band,
                "rows": int(len(value)),
                "weighted_mean_chi": float(np.sum(weight * value)),
                "weighted_rms_chi": float(np.sqrt(np.sum(weight * value**2))),
                "weighted_median_abs_chi": _weighted_quantile(
                    np.abs(value), weight, 0.5
                ),
                "weighted_fraction_abs_chi_gt_3": float(
                    np.sum(weight * (np.abs(value) > 3.0))
                ),
                "weighted_fraction_abs_chi_gt_5": float(
                    np.sum(weight * (np.abs(value) > 5.0))
                ),
            }
        )
    return pd.DataFrame(rows), merged


def _equal_q_weights(q: pd.DataFrame) -> pd.DataFrame:
    weights = q[["row_index", "sample_id"]].copy()
    counts = weights.groupby("row_index")["sample_id"].transform("count")
    weights["psis_weight"] = 1.0 / counts.to_numpy(dtype=float)
    return weights


def _plot_support_comparison(
    path: Path, parent: pd.DataFrame, source: pd.DataFrame
) -> None:
    merged = parent[["row_index", "raw_ess", "pareto_k", "max_raw_weight"]].merge(
        source[["row_index", "raw_ess", "pareto_k", "max_raw_weight"]],
        on="row_index",
        suffixes=("_parent", "_source"),
        validate="one_to_one",
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), constrained_layout=True)
    specs = (
        ("raw_ess", "ESS", True),
        ("pareto_k", "Pareto k", False),
        ("max_raw_weight", "maximum raw weight", False),
    )
    for axis, (column, label, log_scale) in zip(axes, specs, strict=True):
        x = merged[f"{column}_source"].to_numpy(dtype=float)
        y = merged[f"{column}_parent"].to_numpy(dtype=float)
        axis.scatter(x, y, s=17, alpha=0.72, color="#0072B2")
        finite = np.isfinite(x) & np.isfinite(y)
        if log_scale:
            finite &= (x > 0.0) & (y > 0.0)
        if not finite.any():
            raise ValueError(f"support comparison has no finite {column} values")
        low = float(min(x[finite].min(), y[finite].min()))
        high = float(max(x[finite].max(), y[finite].max()))
        axis.plot([low, high], [low, high], color="0.35", linestyle="--")
        if log_scale:
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.set(xlabel=f"source prior {label}", ylabel=f"projected parent {label}")
    figure.suptitle("Same-q-draw individual importance support")
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_ppc(path: Path, frames: list[pd.DataFrame]) -> None:
    colors = {
        "q": "#0072B2",
        "source_prior_iw": "#CC79A7",
        "projected_parent_iw": "#009E73",
    }
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    bins = np.linspace(-8.0, 8.0, 81)
    for frame in frames:
        label = str(frame["model"].iloc[0])
        values = pd.to_numeric(frame["chi_likelihood"], errors="coerce").to_numpy(
            dtype=float
        )
        weights = pd.to_numeric(frame["psis_weight"], errors="coerce").to_numpy(
            dtype=float
        )
        finite = np.isfinite(values) & np.isfinite(weights)
        axis.hist(
            np.clip(values[finite], bins[0], bins[-1]),
            bins=bins,
            weights=weights[finite],
            density=True,
            histtype="step",
            linewidth=1.7,
            label=label,
            color=colors.get(label),
        )
    axis.axvline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axis.set(
        xlabel="likelihood-normalized observed minus model flux",
        ylabel="weighted density",
        title="Posterior-predictive residuals",
    )
    axis.legend(frameon=False)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _write_corners(
    *,
    out: Path,
    q: pd.DataFrame,
    parent_iw: pd.DataFrame,
    source_iw: pd.DataFrame,
    prior: pd.DataFrame,
    truth: pd.DataFrame,
    panel_rows: np.ndarray,
    config: dict[str, Any],
    parent_diagnostics: pd.DataFrame,
) -> list[dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    parameter_names = tuple((config.get("fit", {}) or {}).get("free_parameters", {}))
    diagnostic = parent_diagnostics.set_index("row_index")
    records = []
    for order, raw_row_index in enumerate(panel_rows, start=1):
        row_index = int(raw_row_index)
        q_object = q[q["row_index"] == row_index]
        parent_object = parent_iw[parent_iw["row_index"] == row_index]
        source_object = source_iw[source_iw["row_index"] == row_index]
        truth_object = truth[truth["row_index"] == row_index]
        if q_object.empty or parent_object.empty or len(truth_object) != 1:
            raise ValueError(f"incomplete panel inputs for row_index={row_index}")
        support = diagnostic.loc[row_index]
        stem = f"{order:02d}_row_{row_index}"
        core_path = _write_multi_overlay_corner_plot(
            q_object[list(CORE_PARAMETERS)],
            out,
            plt,
            truth=truth_object[list(CORE_PARAMETERS)],
            prior=prior[list(CORE_PARAMETERS)],
            filename=f"{stem}_core5_q_iw_prior_truth.png",
            title=(
                f"row {row_index}: projected-parent ESS={support['raw_ess']:.1f}, "
                f"Pareto k={support['pareto_k']:.2f}"
            ),
            posterior_label="raw conditional q",
            config=config,
            additional_overlays=[
                {
                    "key": "projected_parent_iw",
                    "label": "projected-parent PSIS-IW",
                    "frame": parent_object[list(CORE_PARAMETERS)],
                    "color": "#009E73",
                },
                {
                    "key": "source_prior_iw",
                    "label": "source-prior PSIS-IW",
                    "frame": source_object[list(CORE_PARAMETERS)],
                    "color": "#CC79A7",
                    "linestyle": "--",
                },
            ],
        )
        full_path = _write_multi_overlay_corner_plot(
            q_object[list(parameter_names)],
            out,
            plt,
            truth=truth_object[list(parameter_names)],
            prior=prior[list(parameter_names)],
            filename=f"{stem}_full15_q_iw_prior_truth.png",
            title=f"row {row_index}: full joint posterior diagnostic",
            posterior_label="raw conditional q",
            config=config,
            additional_overlays=[
                {
                    "key": "projected_parent_iw",
                    "label": "projected-parent PSIS-IW",
                    "frame": parent_object[list(parameter_names)],
                    "color": "#009E73",
                }
            ],
        )
        if core_path is None or full_path is None:
            raise ValueError(f"corner plotting failed for row_index={row_index}")
        object_id = truth_object.iloc[0]["object_id"]
        if isinstance(object_id, np.generic):
            object_id = object_id.item()
        records.append(
            {
                "order": order,
                "row_index": row_index,
                "object_id": object_id,
                "projected_parent_raw_ess": float(support["raw_ess"]),
                "projected_parent_pareto_k": float(support["pareto_k"]),
                "core5_plot": str(core_path.resolve()),
                "full15_plot": str(full_path.resolve()),
            }
        )
    _write_json(out / "manifest.json", {"status": "COMPLETE", "panels": records})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    runtime_provenance = _validate_runtime_provenance(root, manifest, repo)
    final_path = root / "INDIVIDUAL_POSTERIOR_DIAGNOSTIC_COMPLETE.json"
    if final_path.is_file():
        print(final_path.read_text(encoding="utf-8"), flush=True)
        return

    cohort_path = Path(manifest["cohort"]["path"])
    panel_path = Path(manifest["cohort"]["panel_rows"])
    if sha256_file(cohort_path) != manifest["cohort"]["sha256"]:
        raise ValueError("cohort SHA256 mismatch")
    if sha256_file(panel_path) != manifest["cohort"]["panel_rows_sha256"]:
        raise ValueError("panel cohort SHA256 mismatch")
    cohort = np.load(cohort_path, allow_pickle=False)
    panel_rows = np.load(panel_path, allow_pickle=False)
    expected_draws = int(manifest["inference"]["posterior_draws_per_object"])
    resample_draws = int(manifest["inference"]["psis_resample_draws_per_object"])

    shards = []
    for record in manifest["cohort"]["shards"]:
        shard = root / "shards" / f"shard_{int(record['shard']):05d}"
        complete = _read_json(shard / "SHARD_COMPLETE.json")
        if (
            complete.get("status") != "COMPLETE"
            or complete.get("truth_used") is not False
        ):
            raise ValueError(f"invalid shard completion receipt: {shard}")
        if not (shard / "DONE").is_file():
            raise FileNotFoundError(shard / "DONE")
        for forbidden in (
            "inference_truth.parquet",
            "redshift_pit.parquet",
            "corner_full_latent_truth_prior_posterior.png",
        ):
            if (shard / "inference" / forbidden).exists():
                raise ValueError(f"truth leaked into inference shard: {forbidden}")
        shards.append(shard)

    q = _read_parquet_files(
        [shard / "inference/posterior_samples/batch_000001.parquet" for shard in shards]
    )
    residuals = _read_parquet_files(
        [
            shard / "inference/posterior_predictive_residuals/batch_000001.parquet"
            for shard in shards
        ]
    )
    priors = _read_parquet_files(
        [shard / "inference/learned_prior_samples.parquet" for shard in shards]
    )
    parent_weighted = _read_parquet_files(
        [
            shard / "projected_parent_iw/weighted_samples/batch_000000.parquet"
            for shard in shards
        ]
    )
    parent_resampled = _read_parquet_files(
        [
            shard / "projected_parent_iw/resampled_samples/batch_000000.parquet"
            for shard in shards
        ]
    )
    source_weighted = _read_parquet_files(
        [
            shard / "source_prior_iw/weighted_samples/batch_000000.parquet"
            for shard in shards
        ]
    )
    source_resampled = _read_parquet_files(
        [
            shard / "source_prior_iw/resampled_samples/batch_000000.parquet"
            for shard in shards
        ]
    )
    parent_diagnostics = _read_parquet_files(
        [
            shard / "projected_parent_iw/importance_diagnostics.parquet"
            for shard in shards
        ]
    )
    source_diagnostics = _read_parquet_files(
        [shard / "source_prior_iw/importance_diagnostics.parquet" for shard in shards]
    )
    _validate_counts(q, cohort, expected_draws, "q posterior")
    _validate_counts(
        parent_weighted, cohort, expected_draws, "parent weighted posterior"
    )
    _validate_counts(
        source_weighted, cohort, expected_draws, "source weighted posterior"
    )
    _validate_counts(parent_resampled, cohort, resample_draws, "parent IW posterior")
    _validate_counts(source_resampled, cohort, resample_draws, "source IW posterior")
    if len(parent_diagnostics) != len(cohort) or len(source_diagnostics) != len(cohort):
        raise ValueError("importance diagnostic object count mismatch")

    evaluation = root / "evaluation"
    if evaluation.exists():
        shutil.rmtree(evaluation)
    evaluation.mkdir(parents=True, exist_ok=False)
    dense = evaluation / "dense_joint_draws"
    dense.mkdir()
    q.to_parquet(dense / "q_k1024.parquet", index=False)
    parent_weighted.to_parquet(dense / "projected_parent_weighted.parquet", index=False)
    parent_resampled.to_parquet(
        dense / "projected_parent_psis_resampled.parquet", index=False
    )
    source_weighted.to_parquet(dense / "source_prior_weighted.parquet", index=False)
    source_resampled.to_parquet(
        dense / "source_prior_psis_resampled.parquet", index=False
    )
    parent_diagnostics.to_csv(evaluation / "projected_parent_support.csv", index=False)
    source_diagnostics.to_csv(evaluation / "source_prior_support.csv", index=False)
    support_objects = parent_diagnostics.merge(
        source_diagnostics,
        on=["row_index", "object_id"],
        suffixes=("_parent", "_source"),
        validate="one_to_one",
    )
    support_objects.to_csv(evaluation / "same_draw_support_comparison.csv", index=False)

    parent_support = _support_summary(parent_diagnostics)
    source_support = _support_summary(source_diagnostics)
    _plot_support_comparison(
        evaluation / "same_draw_support_comparison.png",
        parent_diagnostics,
        source_diagnostics,
    )
    support_receipt = {
        "status": "FROZEN",
        "objects": int(len(cohort)),
        "proposal_draws_per_object": expected_draws,
        "projected_parent_support": parent_support,
        "source_prior_support": source_support,
        "comparison_sha256": sha256_file(
            evaluation / "same_draw_support_comparison.csv"
        ),
        "truth_used": False,
    }
    _write_json(evaluation / "SUPPORT_FROZEN.json", support_receipt)

    # Truth is attached only after the same-draw support receipt is durable.
    truth_config_path = Path(manifest["truth_closure"]["config"])
    if sha256_file(truth_config_path) != manifest["truth_closure"]["config_sha256"]:
        raise ValueError("truth closure config SHA256 mismatch")
    truth_config = load_config(truth_config_path)
    truth_config["catalog_path"] = manifest["dataset"]["path"]
    parameter_names = tuple(
        (truth_config.get("fit", {}) or {}).get("free_parameters", {})
    )
    truth = _truth_frame(
        evaluation=evaluation,
        config=truth_config,
        cohort=cohort,
        parameter_names=parameter_names,
    )

    q_calibration = (
        q.sort_values(["row_index", "sample_id"])
        .groupby("row_index", sort=False, group_keys=False)
        .head(resample_draws)
        .copy()
    )
    q_calibration["sample_id"] = q_calibration.groupby("row_index").cumcount()
    q_calibration_path = dense / "q_calibration_k256.parquet"
    q_calibration.to_parquet(q_calibration_path, index=False)
    calibration = evaluate_redshift(
        truth_path=evaluation / "individual_truth.parquet",
        posterior_specs=[
            ("q", q_calibration_path),
            ("source_prior_iw", dense / "source_prior_psis_resampled.parquet"),
            (
                "projected_parent_iw",
                dense / "projected_parent_psis_resampled.parquet",
            ),
        ],
        out=evaluation / "redshift_calibration",
        truth_column="z_obs",
        samples_per_object=resample_draws,
        bootstrap=500,
        seed=293400,
        expected_objects=len(cohort),
        scope="frozen_observed_flux_stratified_independent_test_closure",
    )

    ppc_tables = []
    ppc_frames = []
    for label, weights in (
        ("q", _equal_q_weights(q)),
        ("source_prior_iw", source_weighted),
        ("projected_parent_iw", parent_weighted),
    ):
        table, frame = _ppc_summary(residuals, weights, label=label)
        ppc_tables.append(table)
        ppc_frames.append(frame)
    ppc = pd.concat(ppc_tables, ignore_index=True)
    ppc.to_csv(evaluation / "posterior_predictive_by_band.csv", index=False)
    _plot_ppc(evaluation / "posterior_predictive_residuals.png", ppc_frames)

    panels = _write_corners(
        out=evaluation / "individual_panels",
        q=q,
        parent_iw=parent_resampled,
        source_iw=source_resampled,
        prior=priors,
        truth=truth,
        panel_rows=panel_rows,
        config=truth_config,
        parent_diagnostics=parent_diagnostics,
    )
    support_delta = {
        "median_raw_ess": parent_support["median_raw_ess"]
        - source_support["median_raw_ess"],
        "median_raw_ess_fraction": parent_support["median_raw_ess_fraction"]
        - source_support["median_raw_ess_fraction"],
        "fraction_pareto_k_gt_0p7": parent_support["fraction_pareto_k_gt_0p7"]
        - source_support["fraction_pareto_k_gt_0p7"],
        "p90_max_raw_weight": parent_support["p90_max_raw_weight"]
        - source_support["p90_max_raw_weight"],
    }
    receipt = {
        "status": "DIAGNOSTIC_COMPLETE",
        "method": manifest["method"],
        "winner": manifest["benchmark"]["winner"],
        "objects": int(len(cohort)),
        "proposal_draws_per_object": expected_draws,
        "calibration_draws_per_object": resample_draws,
        "projected_parent_support": parent_support,
        "source_prior_support": source_support,
        "same_draw_support_delta_parent_minus_source": support_delta,
        "redshift_calibration": calibration["models"],
        "posterior_predictive_by_band": str(
            (evaluation / "posterior_predictive_by_band.csv").resolve()
        ),
        "panels": panels,
        "truth_boundary": manifest["truth_boundary"],
        "truth_used_for_inference_or_support": False,
        "truth_used_for_final_closure": True,
        "scientific_promotion": False,
        "interpretation": (
            "A support failure means projected-parent IW corners and calibration "
            "remain diagnostic; PSIS cannot replace missing proposal support."
        ),
        "artifacts": {
            "support_comparison": str(
                (evaluation / "same_draw_support_comparison.csv").resolve()
            ),
            "support_plot": str(
                (evaluation / "same_draw_support_comparison.png").resolve()
            ),
            "support_receipt": str((evaluation / "SUPPORT_FROZEN.json").resolve()),
            "redshift_calibration": str(
                (
                    evaluation
                    / "redshift_calibration/redshift_calibration_summary.json"
                ).resolve()
            ),
            "redshift_calibration_plot": str(
                (
                    evaluation / "redshift_calibration/redshift_pit_coverage.png"
                ).resolve()
            ),
            "ppc_plot": str(
                (evaluation / "posterior_predictive_residuals.png").resolve()
            ),
            "panel_manifest": str(
                (evaluation / "individual_panels/manifest.json").resolve()
            ),
        },
        "runtime_provenance": runtime_provenance,
    }
    _write_json(final_path, receipt)
    print(
        json.dumps(_json_safe(receipt), indent=2, sort_keys=True, allow_nan=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
