#!/usr/bin/env python3
"""Finalize the frozen epoch-160 support and population evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.amortized.catalog_identity import write_truth_snapshot
from euclid_dsps.amortized.mira import (
    FENIKS_SPLINE15D_PARAMETERS,
    evaluate_feniks_mira,
)
from euclid_dsps.amortized.tarp import evaluate_feniks_tarp
from euclid_dsps.config import load_config
from euclid_dsps.io import truth_column_from_spec
from euclid_dsps.photometry import abmag_to_fnu_cgs

VARIANTS = ("raw", "ema")
HELDOUT_SHARDS = 4
CATALOGUE_SHARDS = 8


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _require_truth_free_inference(path: Path, *, samples: int) -> dict[str, Any]:
    summary = _read_json(path / "inference_summary.json")
    valid = (
        summary.get("complete") is True
        and int(summary.get("posterior_samples", -1)) == int(samples)
        and summary.get("truth_snapshot_enabled") is False
        and summary.get("truth_diagnostics_enabled") is False
        and summary.get("truth_used_for_inference_or_checkpoint_selection") is False
        and int(summary.get("truth_snapshot_rows", -1)) == 0
        and not (path / "inference_truth.parquet").exists()
    )
    if not valid:
        raise ValueError(f"invalid truth-free inference contract: {path}")
    return summary


def _link_bank(paths: Iterable[Path], destination: Path) -> tuple[Path, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    linked = []
    for index, source in enumerate(paths):
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(source)
        target = destination / f"shard_{index:05d}.parquet"
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"posterior-bank link target changed: {target}")
        else:
            target.symlink_to(source.resolve())
        linked.append(target)
    if not linked:
        raise ValueError(f"no posterior files linked into {destination}")
    return tuple(linked)


def _posterior_files(path: Path) -> list[Path]:
    files = sorted((path / "posterior_samples").glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(path / "posterior_samples")
    return files


def _predictive_files(path: Path) -> list[Path]:
    files = sorted((path / "posterior_predictive_flux").glob("batch_*.parquet"))
    if not files:
        raise FileNotFoundError(path / "posterior_predictive_flux")
    return files


def _heldout_support_summary(root: Path, variant: str) -> dict[str, Any]:
    pieces = []
    for shard in range(HELDOUT_SHARDS):
        path = root / "heldout" / variant / f"shard_{shard}" / "ordinary_iw"
        summary = _read_json(path / "importance_summary.json")
        if (
            summary.get("status") != "complete"
            or int(summary.get("n_joint_draws", -1))
            != int(summary.get("n_objects", -1)) * 1024
        ):
            raise ValueError(f"invalid held-out importance artifact: {path}")
        pieces.append(pd.read_parquet(path / "importance_diagnostics.parquet"))
    diagnostics = pd.concat(pieces, ignore_index=True)
    diagnostics.to_parquet(
        root / "heldout" / f"{variant}_importance_diagnostics.parquet",
        index=False,
    )
    finite_fraction = (
        diagnostics["n_finite_logweights"] / diagnostics["n_proposal_samples"]
    )
    metrics = {
        "objects": int(len(diagnostics)),
        "draws_per_object": 1024,
        "median_raw_ess_fraction": float(diagnostics["raw_ess_fraction"].median()),
        "median_raw_ess": float(diagnostics["raw_ess"].median()),
        "p10_raw_ess_fraction": float(diagnostics["raw_ess_fraction"].quantile(0.1)),
        "fraction_raw_ess_below_0p01": float(
            np.mean(diagnostics["raw_ess_fraction"] < 0.01)
        ),
        "p90_max_raw_weight": float(diagnostics["max_raw_weight"].quantile(0.9)),
        "fraction_pareto_k_gt_0p7": float(
            np.nanmean(diagnostics["pareto_k"] > 0.7)
        ),
        "fraction_pareto_k_gt_1": float(
            np.nanmean(diagnostics["pareto_k"] > 1.0)
        ),
        "minimum_finite_logweight_fraction": float(finite_fraction.min()),
    }
    passed = (
        metrics["median_raw_ess_fraction"] >= 0.05
        and metrics["p10_raw_ess_fraction"] >= 0.01
        and metrics["fraction_raw_ess_below_0p01"] <= 0.10
        and metrics["p90_max_raw_weight"] <= 0.50
        and metrics["fraction_pareto_k_gt_0p7"] <= 0.20
        and metrics["minimum_finite_logweight_fraction"] >= 0.99
    )
    metrics["status"] = "PASS" if passed else "FAIL"
    metrics["role"] = "diagnostic_only_no_checkpoint_selection"
    path = root / "heldout" / f"{variant}_support_summary.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def _heldout_predictive_summary(root: Path, variant: str) -> dict[str, Any]:
    compact_parts = []
    predictive_parts = []
    for shard in range(HELDOUT_SHARDS):
        source = (
            root
            / "heldout"
            / variant
            / f"shard_{shard}"
            / "exact_gaussian_predictive"
        )
        compact_parts.append(
            pd.read_parquet(source / "posterior_predictive_residual_summary.parquet")
        )
        predictive_parts.extend(_predictive_files(source))
    compact = pd.concat(compact_parts, ignore_index=True).drop_duplicates(
        ["row_index", "band"]
    )
    compact = compact.loc[
        compact["valid"].astype(bool)
        & np.isfinite(compact["obs_flux_fnu_cgs"])
        & np.isfinite(compact["obs_err_fnu_cgs"])
        & (compact["obs_err_fnu_cgs"] > 0.0)
    ]
    pieces = []
    for path in predictive_parts:
        frame = pd.read_parquet(
            path, columns=["row_index", "sample_id", "band", "model_flux_fnu_cgs"]
        )
        merged = frame.merge(
            compact,
            on=["row_index", "band"],
            how="inner",
            validate="many_to_one",
        )
        merged["normalized_residual"] = (
            merged["obs_flux_fnu_cgs"] - merged["model_flux_fnu_cgs"]
        ) / merged["obs_err_fnu_cgs"]
        pieces.append(merged[["row_index", "band", "normalized_residual"]])
    residuals = pd.concat(pieces, ignore_index=True)
    residuals = residuals.loc[np.isfinite(residuals["normalized_residual"])]
    by_band = (
        residuals.groupby("band", sort=True)["normalized_residual"]
        .agg(
            draws="size",
            mean="mean",
            median="median",
            rms=lambda values: float(np.sqrt(np.mean(np.square(values)))),
        )
        .reset_index()
    )
    by_band["abs_median"] = np.abs(by_band["median"])
    by_band.to_csv(root / "heldout" / f"{variant}_predictive_by_band.csv", index=False)
    metrics = {
        "objects": int(residuals["row_index"].nunique()),
        "draws": int(len(residuals)),
        "posterior_draws_per_object": 64,
        "median_band_rms": float(by_band["rms"].median()),
        "maximum_band_rms": float(by_band["rms"].max()),
        "median_absolute_band_bias": float(by_band["abs_median"].median()),
        "maximum_absolute_band_bias": float(by_band["abs_median"].max()),
    }
    metrics["status"] = (
        "PASS"
        if metrics["median_band_rms"] <= 2.0
        and metrics["maximum_band_rms"] <= 4.0
        and metrics["median_absolute_band_bias"] <= 1.0
        else "FAIL"
    )
    metrics["role"] = "diagnostic_only_no_checkpoint_selection"
    path = root / "heldout" / f"{variant}_predictive_summary.json"
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def _truth_matrix(
    frame: pd.DataFrame,
    mappings: dict[str, Any],
    parameters: tuple[str, ...],
) -> np.ndarray:
    columns = [truth_column_from_spec(mappings[name]) for name in parameters]
    if any(column is None for column in columns):
        raise ValueError("truth closure mappings must name physical columns")
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"truth closure is missing columns: {missing}")
    return frame.loc[:, columns].to_numpy(dtype=np.float64)


def _population_rows(
    *,
    truth: np.ndarray,
    model: np.ndarray,
    parameters: tuple[str, ...],
    model_name: str,
    target_population: str,
) -> list[dict[str, Any]]:
    probabilities = np.linspace(0.0, 1.0, 1001)
    truth_quantiles = np.quantile(truth, probabilities, axis=0)
    model_quantiles = np.quantile(model, probabilities, axis=0)
    truth_std = np.std(truth, axis=0)
    truth_iqr = truth_quantiles[750] - truth_quantiles[250]
    rows = []
    for index, name in enumerate(parameters):
        wasserstein = float(
            np.mean(np.abs(model_quantiles[:, index] - truth_quantiles[:, index]))
        )
        rows.append(
            {
                "model": model_name,
                "target_population": target_population,
                "parameter": name,
                "truth_objects": int(len(truth)),
                "model_joint_draws": int(len(model)),
                "wasserstein_1d": wasserstein,
                "wasserstein_over_truth_iqr": float(
                    wasserstein / max(float(truth_iqr[index]), 1.0e-12)
                ),
                "mean_difference_over_truth_std": float(
                    (np.mean(model[:, index]) - np.mean(truth[:, index]))
                    / max(float(truth_std[index]), 1.0e-12)
                ),
                "std_ratio": float(
                    np.std(model[:, index])
                    / max(float(truth_std[index]), 1.0e-12)
                ),
                "q05_difference": float(
                    model_quantiles[50, index] - truth_quantiles[50, index]
                ),
                "q50_difference": float(
                    model_quantiles[500, index] - truth_quantiles[500, index]
                ),
                "q95_difference": float(
                    model_quantiles[950, index] - truth_quantiles[950, index]
                ),
            }
        )
    return rows


def _correlation_row(
    *, truth: np.ndarray, model: np.ndarray, model_name: str, target: str
) -> dict[str, Any]:
    truth_correlation = np.corrcoef(truth, rowvar=False)
    model_correlation = np.corrcoef(model, rowvar=False)
    upper = np.triu_indices(truth.shape[1], k=1)
    delta = np.abs(model_correlation[upper] - truth_correlation[upper])
    return {
        "model": model_name,
        "target_population": target,
        "mean_absolute_correlation_error": float(np.mean(delta)),
        "maximum_absolute_correlation_error": float(np.max(delta)),
    }


def deterministic_panel_rows(
    catalog: Path, row_indices: np.ndarray, *, count: int = 16
) -> np.ndarray:
    """Choose an observed-r quantile panel without reading latent truth."""
    flux = pd.read_parquet(catalog, columns=["flux_lsst_r"])[
        "flux_lsst_r"
    ].to_numpy(dtype=np.float64)
    rows = np.asarray(row_indices, dtype=np.int64)
    values = flux[rows]
    finite = np.isfinite(values) & (values > 0.0)
    rows = rows[finite]
    values = values[finite]
    if len(rows) < count:
        raise ValueError("not enough finite observed-r rows for individual panel")
    order = np.argsort(values, kind="stable")
    positions = np.rint(np.linspace(0, len(order) - 1, count)).astype(int)
    return rows[order[positions]]


def _catalogue_source_files(root: Path, variant: str, kind: str) -> list[Path]:
    files = []
    for shard in range(CATALOGUE_SHARDS):
        base = root / "catalogue" / variant / f"shard_{shard}"
        if not (base / "DONE").is_file():
            raise FileNotFoundError(base / "DONE")
        if kind == "q":
            files.extend(_posterior_files(base / "exact_gaussian_k256"))
        else:
            files.append(base / "ordinary_iw" / "resampled_samples/batch_000000.parquet")
    return files


def _write_population_bank(
    *,
    files: list[Path],
    output: Path,
    expected_rows: np.ndarray,
    panel_rows: np.ndarray,
    parameters: tuple[str, ...],
    samples_per_object: int,
    original_samples_per_object: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel_output = output.parent.parent / "individual_panels" / output.name
    if output.is_file() and panel_output.is_file():
        aggregate = pd.read_parquet(output)
        panel = pd.read_parquet(panel_output)
        counts = aggregate.groupby("row_index", sort=False)["sample_id"].size()
        expected = set(np.asarray(expected_rows, dtype=np.int64).tolist())
        actual = set(aggregate["row_index"].astype(np.int64).unique().tolist())
        if actual != expected or not np.all(
            counts.to_numpy() == int(samples_per_object)
        ):
            raise ValueError(f"existing population bank is incomplete: {output}")
        return aggregate, panel, {
            "objects": int(len(counts)),
            "joint_draws": int(len(aggregate)),
            "samples_per_object": int(samples_per_object),
            "source_samples_per_object": int(original_samples_per_object),
            "object_equal_weighting": True,
            "distribution_replaced_by_point_estimates": False,
            "artifact": _file_record(output),
        }
    aggregate_parts = []
    panel_parts = []
    panel_set = set(np.asarray(panel_rows, dtype=np.int64).tolist())
    for path in files:
        frame = pd.read_parquet(path)
        if "row_index" not in frame or "sample_id" not in frame:
            raise ValueError(f"posterior bank lacks joint-draw identities: {path}")
        missing = sorted(set(parameters) - set(frame.columns))
        if missing:
            raise ValueError(f"posterior bank is missing parameters {missing}: {path}")
        panel_parts.append(frame.loc[frame["row_index"].isin(panel_set)].copy())
        aggregate_parts.append(frame.loc[frame["sample_id"] < samples_per_object].copy())
    aggregate = pd.concat(aggregate_parts, ignore_index=True)
    panel = pd.concat(panel_parts, ignore_index=True)
    expected = set(np.asarray(expected_rows, dtype=np.int64).tolist())
    actual = set(aggregate["row_index"].astype(np.int64).unique().tolist())
    if actual != expected:
        raise ValueError(
            f"catalogue posterior rows differ: missing={len(expected-actual)} "
            f"extra={len(actual-expected)}"
        )
    counts = aggregate.groupby("row_index", sort=False)["sample_id"].size()
    if not np.all(counts.to_numpy() == int(samples_per_object)):
        raise ValueError("aggregate posterior is not object-equal")
    duplicates = aggregate.duplicated(["row_index", "sample_id"])
    if duplicates.any():
        raise ValueError("aggregate posterior contains duplicate joint-draw identities")
    output.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_parquet(output, index=False)
    panel.to_parquet(panel_output, index=False)
    summary = {
        "objects": int(len(counts)),
        "joint_draws": int(len(aggregate)),
        "samples_per_object": int(samples_per_object),
        "source_samples_per_object": int(original_samples_per_object),
        "object_equal_weighting": True,
        "distribution_replaced_by_point_estimates": False,
        "artifact": _file_record(output),
    }
    return aggregate, panel, summary


def finalize(
    *,
    root: Path,
    training_config_path: Path,
    truth_config_path: Path,
    test_catalog: Path,
    manifest_root: Path,
    samples_per_object: int = 128,
    num_regions: int = 100,
    num_bootstrap: int = 1000,
    aggregate_samples_per_object: int = 32,
) -> dict[str, Any]:
    completion = root / "EPOCH160_EVALUATION_COMPLETE.json"
    if completion.is_file():
        return _read_json(completion)
    freeze = _read_json(root / "CHECKPOINT_FROZEN.json")
    if (
        freeze.get("status") != "FROZEN"
        or int(freeze.get("epoch", -1)) != 160
        or freeze.get("truth_used_for_training_or_checkpoint_selection") is not False
    ):
        raise ValueError("invalid epoch-160 freeze receipt")
    training_config = load_config(training_config_path)
    if (training_config.get("truth", {}) or {}).get("parameter_columns"):
        raise ValueError("training config passed to finalizer contains truth mappings")
    truth_config = load_config(truth_config_path)
    parameters = tuple(FENIKS_SPLINE15D_PARAMETERS)
    mappings = dict((truth_config.get("truth", {}) or {}).get("parameter_columns") or {})
    if set(mappings) != set(parameters):
        raise ValueError("truth config must map all and only spline15d parameters")

    combined = root / "heldout" / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    posterior_specs = []
    support_summaries = {}
    predictive_summaries = {}
    heldout_objects = 0
    for variant in VARIANTS:
        q_files = []
        iw_files = []
        for shard in range(HELDOUT_SHARDS):
            base = root / "heldout" / variant / f"shard_{shard}"
            if not (base / "DONE").is_file():
                raise FileNotFoundError(base / "DONE")
            support = base / "exact_gaussian_k1024"
            predictive = base / "exact_gaussian_predictive"
            support_summary = _require_truth_free_inference(support, samples=1024)
            _require_truth_free_inference(predictive, samples=64)
            heldout_objects += int(support_summary["n_processed"])
            q_files.extend(_posterior_files(support))
            iw_files.append(base / "ordinary_iw/resampled_samples/batch_000000.parquet")
        q_bank = combined / f"{variant}_q"
        iw_bank = combined / f"{variant}_iw"
        _link_bank(q_files, q_bank)
        _link_bank(iw_files, iw_bank)
        posterior_specs.extend(((f"{variant}_q", q_bank), (f"{variant}_iw", iw_bank)))
        support_summaries[variant] = _heldout_support_summary(root, variant)
        predictive_summaries[variant] = _heldout_predictive_summary(root, variant)
    heldout_objects //= len(VARIANTS)

    truth_config["catalog_path"] = str(test_catalog.resolve())
    heldout_rows = np.load(
        manifest_root / "final_validation_indices.npy", allow_pickle=False
    ).astype(np.int64)
    heldout_truth = write_truth_snapshot(
        combined,
        truth_config,
        row_indices=heldout_rows,
        limit=None,
        filename="inference_truth.parquet",
    )
    if set(heldout_truth["row_index"].astype(int)) != set(heldout_rows.tolist()):
        raise ValueError("held-out truth snapshot differs from frozen row manifest")
    if not (root / "heldout/mira/DONE").is_file():
        evaluate_feniks_mira(
            truth_path=combined / "inference_truth.parquet",
            posterior_specs=posterior_specs,
            out_dir=root / "heldout/mira",
            num_regions=int(num_regions),
            num_bootstrap=int(num_bootstrap),
            samples_per_object=int(samples_per_object),
            seed=261601,
            parameters=parameters,
            drop_nonfinite_truth=True,
        )
    if not (root / "heldout/tarp/DONE").is_file():
        evaluate_feniks_tarp(
            truth_path=combined / "inference_truth.parquet",
            posterior_specs=posterior_specs,
            out_dir=root / "heldout/tarp",
            num_bootstrap=int(num_bootstrap),
            samples_per_object=int(samples_per_object),
            seed=261602,
            parameters=parameters,
            drop_nonfinite_truth=True,
        )

    full_rows = np.load(manifest_root / "full_test_indices.npy", allow_pickle=False)
    panel_rows = deterministic_panel_rows(test_catalog, full_rows, count=16)
    panel_dir = root / "population/individual_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    np.save(panel_dir / "row_indices.npy", panel_rows, allow_pickle=False)
    panel_payload = {
        "selection": (
            "16 deterministic observed-flux_lsst_r quantiles from the independent "
            "selected test catalogue"
        ),
        "truth_used": False,
        "row_indices": panel_rows.tolist(),
    }
    (panel_dir / "manifest.json").write_text(
        json.dumps(panel_payload, indent=2, sort_keys=True) + "\n"
    )

    population_dir = root / "population/posterior_aggregate"
    population_dir.mkdir(parents=True, exist_ok=True)
    population_summaries = {}
    population_frames = {}
    for variant in VARIANTS:
        for kind, source_samples in (("q", 256), ("iw", 32)):
            name = f"{variant}_{kind}"
            frame, _panel, summary = _write_population_bank(
                files=_catalogue_source_files(root, variant, kind),
                output=population_dir / f"{name}.parquet",
                expected_rows=full_rows,
                panel_rows=panel_rows,
                parameters=parameters,
                samples_per_object=int(aggregate_samples_per_object),
                original_samples_per_object=source_samples,
            )
            population_summaries[name] = summary
            population_frames[name] = frame

    marginal_rows = []
    correlation_arrays = {}
    for name, frame in population_frames.items():
        values = frame.loc[:, parameters].to_numpy(dtype=np.float64)
        finite = np.all(np.isfinite(values), axis=1)
        values = values[finite]
        correlation_arrays[name] = np.corrcoef(values, rowvar=False)
        for index, parameter in enumerate(parameters):
            column = values[:, index]
            q05, q50, q95 = np.quantile(column, [0.05, 0.50, 0.95])
            marginal_rows.append(
                {
                    "posterior_aggregate": name,
                    "parameter": parameter,
                    "joint_draws": int(len(values)),
                    "mean": float(np.mean(column)),
                    "std": float(np.std(column)),
                    "q05": float(q05),
                    "q50": float(q50),
                    "q95": float(q95),
                }
            )
    aggregate_marginals = root / "population/posterior_aggregate_marginals.csv"
    aggregate_correlations = root / "population/posterior_aggregate_correlations.npz"
    pd.DataFrame(marginal_rows).to_csv(aggregate_marginals, index=False)
    np.savez_compressed(aggregate_correlations, **correlation_arrays)

    catalogue_truth = write_truth_snapshot(
        root / "population",
        truth_config,
        row_indices=full_rows,
        limit=None,
        filename="catalogue_selected_truth.parquet",
    )
    if set(catalogue_truth["row_index"].astype(int)) != set(full_rows.astype(int)):
        raise ValueError("full catalogue truth closure differs from frozen row manifest")
    catalogue_truth_matrix = _truth_matrix(catalogue_truth, mappings, parameters)
    finite_catalogue = np.all(np.isfinite(catalogue_truth_matrix), axis=1)
    finite_catalogue_rows = set(
        catalogue_truth.loc[finite_catalogue, "row_index"].astype(np.int64).tolist()
    )
    catalogue_truth_matrix = catalogue_truth_matrix[finite_catalogue]

    population_rows = []
    correlation_rows = []
    for name, frame in population_frames.items():
        frame = frame.loc[frame["row_index"].isin(finite_catalogue_rows)]
        values = frame.loc[:, parameters].to_numpy(dtype=np.float64)
        finite = np.all(np.isfinite(values), axis=1)
        values = values[finite]
        population_rows.extend(
            _population_rows(
                truth=catalogue_truth_matrix,
                model=values,
                parameters=parameters,
                model_name=name,
                target_population="empirical observed-selected independent test catalogue",
            )
        )
        correlation_rows.append(
            _correlation_row(
                truth=catalogue_truth_matrix,
                model=values,
                model_name=name,
                target="empirical observed-selected independent test catalogue",
            )
        )

    prior_dir = root / "population/prior"
    prior_receipt = _read_json(prior_dir / "report_receipt.json")
    if prior_receipt.get("truth_used") is not False:
        raise ValueError("epoch-160 prior report used truth")
    test_truth = write_truth_snapshot(
        root / "population",
        truth_config,
        row_indices=None,
        limit=None,
        filename="test_population_truth_C0.parquet",
    )
    test_matrix = _truth_matrix(test_truth, mappings, parameters)
    finite_test = np.all(np.isfinite(test_matrix), axis=1)
    test_matrix = test_matrix[finite_test]
    test_rows = test_truth.loc[finite_test, "row_index"].to_numpy(dtype=np.int64)
    test_flux = pd.read_parquet(test_catalog, columns=["flux_lsst_r"])[
        "flux_lsst_r"
    ].to_numpy(dtype=np.float64)
    selected = test_flux[test_rows] > float(np.asarray(abmag_to_fnu_cgs(29.0)))
    with np.load(prior_dir / "parent_and_selected_prior.npz", allow_pickle=False) as arrays:
        prior_distributions = {
            "learned_parent_prior": (
                test_matrix,
                np.asarray(arrays["theta"])[
                    np.all(np.isfinite(np.asarray(arrays["theta"])), axis=1)
                ],
                "p_eta(theta | C0)",
            ),
            "beta_weighted_selected_prior": (
                test_matrix[selected],
                np.asarray(arrays["selected_theta"])[
                    np.all(np.isfinite(np.asarray(arrays["selected_theta"])), axis=1)
                ],
                "beta(theta) p_eta(theta | C0) / alpha_eta",
            ),
        }
    for name, (truth_values, model_values, target) in prior_distributions.items():
        population_rows.extend(
            _population_rows(
                truth=truth_values,
                model=model_values,
                parameters=parameters,
                model_name=name,
                target_population=target,
            )
        )
        correlation_rows.append(
            _correlation_row(
                truth=truth_values,
                model=model_values,
                model_name=name,
                target=target,
            )
        )
    recovery_path = root / "population/population_recovery.csv"
    correlation_path = root / "population/population_correlation_recovery.csv"
    pd.DataFrame(population_rows).to_csv(recovery_path, index=False)
    pd.DataFrame(correlation_rows).to_csv(correlation_path, index=False)

    receipt = {
        "status": "DIAGNOSTIC_COMPLETE",
        "epoch": 160,
        "scientific_promotion": False,
        "training_continued_independently": True,
        "training_frozen_before_truth": True,
        "truth_used_for_training_or_checkpoint_selection": False,
        "checkpoint": freeze,
        "heldout_objects": int(heldout_objects),
        "heldout_draws_per_object": 1024,
        "catalogue_objects": int(len(full_rows)),
        "catalogue_draws_per_object": 256,
        "population_aggregate_draws_per_object": int(aggregate_samples_per_object),
        "catalogue_cohort": "all observed-selected rows in the independent test catalogue",
        "prior_shared_by_raw_and_ema": True,
        "heldout_support": support_summaries,
        "heldout_predictive": predictive_summaries,
        "population_banks": population_summaries,
        "contracts": {
            "parent": "p_eta(theta | C0) from the learned prior flow",
            "selected_prior": "beta(theta) p_eta(theta | C0) / alpha_eta",
            "posterior_aggregate": (
                "object-equal dense joint posterior mixture over the observed-selected "
                "catalogue; descriptive and not relabeled as the parent prior"
            ),
            "individual": "dense joint draws retained; no pointwise replacement",
            "truth_role": "post-freeze evaluation only",
        },
        "artifacts": {
            "mira": _file_record(root / "heldout/mira/mira_summary.json"),
            "tarp": _file_record(root / "heldout/tarp/tarp_summary.json"),
            "population_recovery": _file_record(recovery_path),
            "population_correlation_recovery": _file_record(correlation_path),
            "posterior_aggregate_marginals": _file_record(aggregate_marginals),
            "posterior_aggregate_correlations": _file_record(
                aggregate_correlations
            ),
            "individual_panel_manifest": _file_record(panel_dir / "manifest.json"),
        },
    }
    temporary = completion.with_name(f".{completion.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, completion)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--truth-config", type=Path, required=True)
    parser.add_argument("--test-catalog", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--samples-per-object", type=int, default=128)
    parser.add_argument("--num-regions", type=int, default=100)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--aggregate-samples-per-object", type=int, default=32)
    args = parser.parse_args()
    result = finalize(
        root=args.root,
        training_config_path=args.training_config,
        truth_config_path=args.truth_config,
        test_catalog=args.test_catalog,
        manifest_root=args.manifest_root,
        samples_per_object=args.samples_per_object,
        num_regions=args.num_regions,
        num_bootstrap=args.num_bootstrap,
        aggregate_samples_per_object=args.aggregate_samples_per_object,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
