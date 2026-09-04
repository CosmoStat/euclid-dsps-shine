#!/usr/bin/env python3
"""Prepare a frozen individual-posterior diagnostic for a projected parent prior."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from euclid_dsps.amortized.population_vem import sha256_file
from euclid_dsps.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: str | Path) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or value.stat().st_size <= 0:
        raise FileNotFoundError(value)
    return value


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _band_column(config: dict[str, Any], name: str) -> tuple[str, str]:
    matches = [band for band in config.get("bands", ()) if band.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} band, found {len(matches)}")
    band = matches[0]
    column = str(band.get("column", ""))
    if not column:
        raise ValueError(f"band {name!r} has no observed flux column")
    return column, str(band.get("units", "unknown"))


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _select_observed_flux_quantiles(
    dataset: Path,
    selected_rows: np.ndarray,
    *,
    flux_column: str,
    id_column: str | None,
    objects: int,
    panels: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    columns = set(pq.ParquetFile(dataset).schema.names)
    requested = [flux_column]
    if id_column and id_column in columns:
        requested.append(id_column)
    missing = sorted(set(requested) - columns)
    if missing:
        raise ValueError(f"test catalogue is missing observed columns: {missing}")
    frame = pd.read_parquet(dataset, columns=requested).reset_index(drop=True)
    if selected_rows.size == 0 or int(selected_rows.max()) >= len(frame):
        raise ValueError("selected-test cohort contains an invalid catalogue row")
    selected = frame.iloc[selected_rows].copy()
    selected.insert(0, "row_index", selected_rows)
    selected[flux_column] = pd.to_numeric(selected[flux_column], errors="coerce")
    selected = selected[np.isfinite(selected[flux_column])].copy()
    if len(selected) < objects:
        raise ValueError(
            f"only {len(selected)} selected rows have finite {flux_column}; need {objects}"
        )
    selected = selected.sort_values(
        [flux_column, "row_index"], kind="mergesort"
    ).reset_index(drop=True)
    ranks = np.rint(np.linspace(0, len(selected) - 1, objects)).astype(np.int64)
    if len(np.unique(ranks)) != objects:
        raise ValueError("observed-flux quantile selection produced duplicate ranks")
    cohort = selected.iloc[ranks].copy().reset_index(drop=True)
    cohort.insert(0, "cohort_order", np.arange(objects, dtype=np.int64))
    cohort["observed_flux_rank_fraction"] = ranks / max(len(selected) - 1, 1)
    cohort["panel"] = False
    panel_positions = np.rint(np.linspace(0, objects - 1, panels)).astype(np.int64)
    if len(np.unique(panel_positions)) != panels:
        raise ValueError("panel selection produced duplicate cohort positions")
    cohort.loc[panel_positions, "panel"] = True
    if id_column and id_column in cohort:
        cohort = cohort.rename(columns={id_column: "object_id"})
    else:
        cohort["object_id"] = cohort["row_index"]
    return cohort, panel_positions


def prepare(
    *,
    benchmark_root: Path,
    out: Path,
    repo: Path,
    objects: int,
    shards: int,
    panels: int,
    posterior_draws: int,
    resample_draws: int,
    object_batch_size: int = 8,
    prior_draws: int = 512,
    model_receipt: Path | None = None,
) -> dict[str, Any]:
    benchmark_root = benchmark_root.resolve()
    out = out.resolve()
    repo = repo.resolve()
    if objects <= 0 or shards <= 0 or shards > objects:
        raise ValueError("objects/shards must be positive and shards <= objects")
    if panels <= 0 or panels > objects:
        raise ValueError("panels must lie in [1, objects]")
    if posterior_draws <= 0 or resample_draws <= 0:
        raise ValueError("draw counts must be positive")
    if resample_draws > posterior_draws:
        raise ValueError("resample_draws cannot exceed posterior_draws")
    if object_batch_size <= 0 or prior_draws <= 0:
        raise ValueError("object_batch_size and prior_draws must be positive")

    benchmark_manifest_path = _require_file(benchmark_root / "RUN_MANIFEST.json")
    winner_path = _require_file(benchmark_root / "TRUTH_FREE_ARCHITECTURE_WINNER.json")
    closure_path = _require_file(benchmark_root / "POPULATION_PROJECTION_COMPLETE.json")
    fit_path = _require_file(benchmark_root / "PROJECTION_FIT_COMPLETE.json")
    benchmark_manifest = _read_json(benchmark_manifest_path)
    winner = _read_json(winner_path)
    closure = _read_json(closure_path)
    fit = _read_json(fit_path)
    if (
        winner.get("status") != "WINNER_SELECTED"
        or winner.get("truth_used") is not False
        or winner.get("winner_passes_all_truth_free_distribution_gates") is not True
        or winner.get("winner_passes_nll_non_regression_gate") is not True
    ):
        raise ValueError("architecture winner is not a passing truth-free winner")
    if closure.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError("architecture winner closure is incomplete")
    if fit.get("status") != "COMPLETE" or fit.get("truth_used") is not False:
        raise ValueError("winner fit receipt is not complete and truth-free")

    projection_parent = fit["parent"]
    parent = projection_parent
    parent_checkpoint = _require_file(parent["checkpoint"])
    parent_sidecar = _require_file(parent["checkpoint_sidecar"])
    projection_config_path = _require_file(parent["config"])
    if winner["artifacts"]["parent_checkpoint"] != str(parent_checkpoint):
        raise ValueError("winner and fit receipts disagree on the parent checkpoint")
    if sha256_file(parent_checkpoint) != parent["checkpoint_sha256"]:
        raise ValueError("projected-parent checkpoint SHA256 mismatch")
    if sha256_file(parent_sidecar) != parent["checkpoint_sidecar_sha256"]:
        raise ValueError("projected-parent sidecar SHA256 mismatch")
    if sha256_file(projection_config_path) != parent["config_sha256"]:
        raise ValueError("projected-parent config SHA256 mismatch")

    model_receipt_record = None
    if model_receipt is not None:
        model_receipt = _require_file(model_receipt)
        frozen_model = _read_json(model_receipt)
        if (
            frozen_model.get("status") != "FROZEN"
            or frozen_model.get("truth_used_for_training_or_checkpoint_selection")
            is not False
            or frozen_model.get("prior_bitwise_unchanged") is not True
        ):
            raise ValueError("posterior model receipt is not a certified frozen-q receipt")
        parent_checkpoint = _require_file(frozen_model["checkpoint"])
        parent_sidecar = _require_file(frozen_model["checkpoint_sidecar"])
        candidate_config_path = _require_file(frozen_model["config"])
        feature_stats = _require_file(frozen_model["feature_stats"])
        for path, key, label in (
            (parent_checkpoint, "checkpoint_sha256", "posterior checkpoint"),
            (parent_sidecar, "checkpoint_sidecar_sha256", "posterior sidecar"),
            (candidate_config_path, "config_sha256", "posterior config"),
            (feature_stats, "feature_stats_sha256", "posterior feature statistics"),
        ):
            if sha256_file(path) != frozen_model[key]:
                raise ValueError(f"{label} SHA256 mismatch")
        parent = {
            "checkpoint_sha256": frozen_model["checkpoint_sha256"],
            "checkpoint_sidecar_sha256": frozen_model[
                "checkpoint_sidecar_sha256"
            ],
            "config_sha256": frozen_model["config_sha256"],
        }
        model_receipt_record = {
            "path": str(model_receipt),
            "sha256": sha256_file(model_receipt),
        }

    source = benchmark_manifest["source"]
    source_checkpoint = _require_file(source["checkpoint"])
    source_sidecar = _require_file(source["checkpoint_sidecar"])
    source_feature_stats = _require_file(source["feature_stats"])
    for path, expected, label in (
        (source_checkpoint, source["checkpoint_sha256"], "source checkpoint"),
        (source_sidecar, source["checkpoint_sidecar_sha256"], "source sidecar"),
        (
            source_feature_stats,
            source["feature_stats_sha256"],
            "feature statistics",
        ),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"{label} SHA256 mismatch")

    if model_receipt is None:
        candidate_config_path = projection_config_path
        feature_stats = source_feature_stats
    candidate_config = load_config(candidate_config_path)
    source_config_path = _require_file(benchmark_manifest["config"]["path"])
    if sha256_file(source_config_path) != benchmark_manifest["config"]["sha256"]:
        raise ValueError("source inference config SHA256 mismatch")
    source_config = load_config(source_config_path)
    truth_config_path = _require_file(benchmark_manifest["truth_config"]["path"])
    if sha256_file(truth_config_path) != benchmark_manifest["truth_config"]["sha256"]:
        raise ValueError("truth closure config SHA256 mismatch")
    truth_source_config = load_config(truth_config_path)
    test_dataset = benchmark_manifest["datasets"]["test"]
    dataset = _require_file(test_dataset["path"])
    if sha256_file(dataset) != test_dataset["sha256"]:
        raise ValueError("independent test catalogue SHA256 mismatch")
    c0_objects = int(
        test_dataset.get("c0_objects", pq.ParquetFile(dataset).metadata.num_rows)
    )
    if c0_objects <= 0 or c0_objects > pq.ParquetFile(dataset).metadata.num_rows:
        raise ValueError("invalid independent C0 object count")

    calibration_manifest_path = _require_file(source["calibration_manifest"])
    if sha256_file(calibration_manifest_path) != source["calibration_manifest_sha256"]:
        raise ValueError("calibration manifest SHA256 mismatch")
    calibration_manifest = _read_json(calibration_manifest_path)
    q_evaluation = calibration_manifest["banks"]["q_evaluation"]
    selected_rows_path = _require_file(q_evaluation["cohort_path"])
    if sha256_file(selected_rows_path) != q_evaluation["cohort_sha256"]:
        raise ValueError("selected-test cohort SHA256 mismatch")
    selected_rows = np.asarray(
        np.load(selected_rows_path, allow_pickle=False), dtype=np.int64
    ).reshape(-1)
    if len(selected_rows) != int(
        benchmark_manifest["datasets"]["test"]["selected_objects"]
    ):
        raise ValueError("selected-test cohort object count mismatch")

    flux_column, flux_units = _band_column(candidate_config, "lsst_r")
    id_column = str((candidate_config.get("dataset", {}) or {}).get("id_column", ""))
    cohort, panel_positions = _select_observed_flux_quantiles(
        dataset,
        selected_rows,
        flux_column=flux_column,
        id_column=id_column or None,
        objects=objects,
        panels=panels,
    )

    request = {
        "benchmark_root": str(benchmark_root),
        "winner_receipt_sha256": sha256_file(winner_path),
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "selected_test_cohort_sha256": q_evaluation["cohort_sha256"],
        "objects": int(objects),
        "shards": int(shards),
        "panels": int(panels),
        "posterior_draws": int(posterior_draws),
        "resample_draws": int(resample_draws),
        "object_batch_size": int(object_batch_size),
        "prior_draws": int(prior_draws),
        "model_receipt_sha256": (
            None if model_receipt_record is None else model_receipt_record["sha256"]
        ),
        "selection": "deterministic observed lsst_r flux quantiles",
    }
    if out.exists():
        existing = out / "RUN_MANIFEST.json"
        if existing.is_file() and _read_json(existing).get("request") == request:
            print(f"[population-posterior] already prepared: {out}")
            return _read_json(existing)
        raise FileExistsError(f"population posterior output exists: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        manifest_dir = staging / "manifests"
        manifest_dir.mkdir(parents=True)
        cohort_rows = cohort["row_index"].to_numpy(dtype=np.int64)
        np.save(manifest_dir / "cohort.npy", cohort_rows, allow_pickle=False)
        cohort.to_csv(manifest_dir / "cohort_observed_selection.csv", index=False)
        np.save(
            manifest_dir / "panel_rows.npy",
            cohort_rows[panel_positions],
            allow_pickle=False,
        )
        shard_records = []
        for shard, positions in enumerate(np.array_split(np.arange(objects), shards)):
            rows = cohort_rows[positions]
            path = manifest_dir / f"shard_{shard:05d}.npy"
            np.save(path, rows, allow_pickle=False)
            shard_records.append(
                {
                    "shard": shard,
                    "path": str((out / "manifests" / path.name).resolve()),
                    "sha256": sha256_file(path),
                    "objects": int(len(rows)),
                    "observed_flux_rank_min": float(
                        cohort.iloc[positions]["observed_flux_rank_fraction"].min()
                    ),
                    "observed_flux_rank_max": float(
                        cohort.iloc[positions]["observed_flux_rank_fraction"].max()
                    ),
                }
            )

        inference_config = copy.deepcopy(candidate_config)
        inference_config["catalog_path"] = str(dataset)
        inference_config.setdefault("truth", {})["parameter_columns"] = {}
        inference = inference_config.setdefault("amortized", {}).setdefault(
            "inference", {}
        )
        inference.update(
            {
                "write_truth_snapshot": False,
                "write_truth_diagnostics": False,
                "write_posterior_predictive": True,
                "write_residual_samples": True,
                "shard_outputs": True,
                "resume_shards": True,
                "combine_sample_shards": False,
                "combine_summary_shards": True,
            }
        )
        source_inference_config = copy.deepcopy(source_config)
        source_inference_config["catalog_path"] = str(dataset)
        source_inference_config.setdefault("truth", {})["parameter_columns"] = {}
        source_inference = source_inference_config.setdefault(
            "amortized", {}
        ).setdefault("inference", {})
        source_inference.update(
            {"write_truth_snapshot": False, "write_truth_diagnostics": False}
        )
        truth_config = copy.deepcopy(inference_config)
        truth_config["truth"] = copy.deepcopy(truth_source_config["truth"])
        truth_config["amortized"]["inference"].update(
            {"write_truth_snapshot": True, "write_truth_diagnostics": True}
        )
        _write_config(staging / "inference_config.yaml", inference_config)
        _write_config(staging / "source_config.yaml", source_inference_config)
        _write_config(staging / "truth_closure_config.yaml", truth_config)

        manifest = {
            "status": "PREPARED",
            "schema_version": 1,
            "method": "projected_parent_individual_posterior_support_v1",
            "code_commit": _git_commit(repo),
            "request": request,
            "benchmark": {
                "root": str(benchmark_root),
                "manifest": str(benchmark_manifest_path),
                "manifest_sha256": sha256_file(benchmark_manifest_path),
                "winner_receipt": str(winner_path),
                "winner_receipt_sha256": sha256_file(winner_path),
                "closure_receipt": str(closure_path),
                "closure_receipt_sha256": sha256_file(closure_path),
                "winner": winner["winner"],
            },
            "model": {
                "role": (
                    "source conditional q with projected parent prior"
                    if model_receipt_record is None
                    else "frozen prior-sleep-NPE q with projected parent prior"
                ),
                "checkpoint": str(parent_checkpoint),
                "checkpoint_sha256": parent["checkpoint_sha256"],
                "checkpoint_sidecar": str(parent_sidecar),
                "checkpoint_sidecar_sha256": parent["checkpoint_sidecar_sha256"],
                "config": str((out / "inference_config.yaml").resolve()),
                "config_sha256": sha256_file(staging / "inference_config.yaml"),
                "feature_stats": str(feature_stats),
                "feature_stats_sha256": sha256_file(feature_stats),
                "freeze_receipt": model_receipt_record,
            },
            "population_selection": {
                "config": str(projection_config_path),
                "config_sha256": projection_parent["config_sha256"],
                "feature_stats": str(source_feature_stats),
                "feature_stats_sha256": source["feature_stats_sha256"],
                "role": "truth-free beta(theta) evaluation only",
            },
            "source_model": {
                "role": "same q with pre-projection source prior baseline",
                "checkpoint": str(source_checkpoint),
                "checkpoint_sha256": source["checkpoint_sha256"],
                "checkpoint_sidecar": str(source_sidecar),
                "checkpoint_sidecar_sha256": source["checkpoint_sidecar_sha256"],
                "config": str((out / "source_config.yaml").resolve()),
                "config_sha256": sha256_file(staging / "source_config.yaml"),
            },
            "truth_closure": {
                "config": str((out / "truth_closure_config.yaml").resolve()),
                "config_sha256": sha256_file(staging / "truth_closure_config.yaml"),
                "read_after_inference": True,
            },
            "dataset": {
                "path": str(dataset),
                "sha256": test_dataset["sha256"],
                "role": "independent selected test catalogue",
                "c0_objects": c0_objects,
            },
            "cohort": {
                "path": str((out / "manifests" / "cohort.npy").resolve()),
                "sha256": sha256_file(manifest_dir / "cohort.npy"),
                "observed_selection": str(
                    (out / "manifests" / "cohort_observed_selection.csv").resolve()
                ),
                "observed_selection_sha256": sha256_file(
                    manifest_dir / "cohort_observed_selection.csv"
                ),
                "panel_rows": str((out / "manifests" / "panel_rows.npy").resolve()),
                "panel_rows_sha256": sha256_file(manifest_dir / "panel_rows.npy"),
                "objects": int(objects),
                "panels": int(panels),
                "shards": shard_records,
                "selection_variable": flux_column,
                "selection_units": flux_units,
                "truth_used": False,
            },
            "inference": {
                "posterior_draws_per_object": int(posterior_draws),
                "psis_resample_draws_per_object": int(resample_draws),
                "object_batch_size": int(object_batch_size),
                "prior_draws_per_shard": int(prior_draws),
                "support_gate": {
                    "minimum_median_raw_ess_fraction": 0.05,
                    "maximum_fraction_pareto_k_gt_0p7": 0.20,
                    "maximum_p90_raw_weight": 0.80,
                },
            },
            "truth_boundary": {
                "cohort_selection": False,
                "inference": False,
                "importance_support": False,
                "panel_selection": False,
                "final_closure_and_plot_overlay": True,
            },
            "scientific_promotion": False,
        }
        (staging / "RUN_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, out)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"[population-posterior] prepared objects={objects} shards={shards} "
        f"draws={posterior_draws} root={out}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objects", type=int, default=64)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--panels", type=int, default=8)
    parser.add_argument("--posterior-draws", type=int, default=1024)
    parser.add_argument("--resample-draws", type=int, default=256)
    parser.add_argument("--object-batch-size", type=int, default=8)
    parser.add_argument("--prior-draws", type=int, default=512)
    parser.add_argument("--model-receipt", type=Path)
    args = parser.parse_args()
    prepare(
        benchmark_root=args.benchmark_root,
        out=args.out,
        repo=Path(__file__).resolve().parents[1],
        objects=args.objects,
        shards=args.shards,
        panels=args.panels,
        posterior_draws=args.posterior_draws,
        resample_draws=args.resample_draws,
        object_batch_size=args.object_batch_size,
        prior_draws=args.prior_draws,
        model_receipt=args.model_receipt,
    )


if __name__ == "__main__":
    main()
