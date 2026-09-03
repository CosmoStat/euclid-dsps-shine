#!/usr/bin/env python3
"""Aggregate final population-VEM closure, calibration, and clean plots."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from euclid_dsps.amortized.data import load_photometry_arrays_from_config
from euclid_dsps.amortized.latent import latent_spec_from_config, x_to_theta
from euclid_dsps.amortized.mira import evaluate_feniks_mira
from euclid_dsps.amortized.population_vem import (
    iter_array_bank_shards,
    merge_array_bank_shards,
    require_git_commit,
    resolve_manifest_config,
    sha256_file,
)
from euclid_dsps.amortized.tarp import evaluate_feniks_tarp
from euclid_dsps.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _cohort(manifest: dict[str, Any], name: str) -> np.ndarray:
    record = manifest["banks"][name]
    path = Path(record["cohort_path"])
    if sha256_file(path) != record["cohort_sha256"]:
        raise ValueError(f"cohort provenance changed: {path}")
    return np.load(path, allow_pickle=False).astype(np.int64)


def _theta_from_truth(arrays, names: tuple[str, ...]) -> np.ndarray:
    if not arrays.truth or any(name not in arrays.truth for name in names):
        raise ValueError("final closure is missing truth parameters")
    theta = np.column_stack([np.asarray(arrays.truth[name]) for name in names])
    if not np.all(np.isfinite(theta)):
        raise ValueError("final closure truth contains non-finite values")
    return theta.astype(np.float64)


def _x_to_theta_chunks(x: np.ndarray, latent_spec, chunk: int = 16384) -> np.ndarray:
    parts = []
    for start in range(0, len(x), int(chunk)):
        parts.append(
            np.asarray(
                jax.device_get(
                    x_to_theta(jnp.asarray(x[start : start + int(chunk)]), latent_spec)
                ),
                dtype=np.float64,
            )
        )
    return np.concatenate(parts)


def _comparison_rows(
    *,
    source: np.ndarray,
    target: np.ndarray,
    names: tuple[str, ...],
    comparison: str,
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(names):
        if target_weights is None:
            target_q25, target_q50, target_q75 = np.quantile(
                target[:, index], [0.25, 0.50, 0.75]
            )
        else:
            target_q25, target_q50, target_q75 = (
                _weighted_quantile(target[:, index], target_weights, probability)
                for probability in (0.25, 0.50, 0.75)
            )
        source_median = (
            float(np.quantile(source[:, index], 0.5))
            if source_weights is None
            else _weighted_quantile(source[:, index], source_weights, 0.5)
        )
        target_median = (
            float(target_q50)
            if target_weights is None
            else _weighted_quantile(target[:, index], target_weights, 0.5)
        )
        scale = max(float(target_q75 - target_q25), 1.0e-8)
        distance = float(
            wasserstein_distance(
                source[:, index],
                target[:, index],
                u_weights=source_weights,
                v_weights=target_weights,
            )
        )
        rows.append(
            {
                "comparison": comparison,
                "parameter": name,
                "wasserstein": distance,
                "target_iqr": scale,
                "wasserstein_over_target_iqr": distance / scale,
                "source_median": source_median,
                "target_median": target_median,
                "median_shift_over_target_iqr": (source_median - target_median) / scale,
            }
        )
    return rows


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    if not np.all(np.isfinite(sorted_weights)) or np.any(sorted_weights < 0.0):
        raise ValueError("weighted quantiles require finite non-negative weights")
    if sorted_weights.sum() <= 0.0:
        raise ValueError("weighted quantiles require positive total weight")
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return float(np.interp(float(probability), cumulative, sorted_values))


def _weighted_correlation(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    mean = np.sum(values * normalized[:, None], axis=0)
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 1.0e-30))
    return covariance / np.outer(scale, scale)


def _plot_selected_population(
    path: Path,
    names: tuple[str, ...],
    truth: np.ndarray,
    q: np.ndarray,
    prior: np.ndarray,
    prior_weights: np.ndarray,
) -> None:
    indices = range(min(5, len(names)))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, index in zip(axes.flat, indices, strict=False):
        low, high = np.quantile(truth[:, index], [0.005, 0.995])
        bins = np.linspace(low, high, 45)
        axis.hist(
            truth[:, index],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.20,
            color="#555555",
            label="Selected truth",
        )
        axis.hist(
            q[:, index],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#0072B2",
            label="Aggregate AVI q",
        )
        axis.hist(
            prior[:, index],
            bins=bins,
            weights=prior_weights,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#D55E00",
            label="Selected learned prior",
        )
        axis.set_title(names[index], fontsize=10)
        axis.set_ylabel("Density")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.08))
    figure.suptitle("Selected-population recovery after targeted VEM", fontsize=14)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_parent_population(
    path: Path,
    names: tuple[str, ...],
    truth: np.ndarray,
    prior: np.ndarray,
) -> None:
    indices = range(min(5, len(names)))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, index in zip(axes.flat, indices, strict=False):
        low, high = np.quantile(truth[:, index], [0.005, 0.995])
        bins = np.linspace(low, high, 45)
        axis.hist(
            truth[:, index],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.20,
            color="#555555",
            label="C0 truth",
        )
        axis.hist(
            prior[:, index],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#009E73",
            label="Learned parent prior",
        )
        axis.set_title(names[index], fontsize=10)
        axis.set_ylabel("Density")
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.08))
    figure.suptitle("Parent-population recovery within C0", fontsize=14)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_population_metrics(path: Path, metrics: pd.DataFrame) -> None:
    comparisons = (
        "aggregate_q_vs_selected_truth",
        "selected_prior_vs_selected_truth",
        "aggregate_q_vs_selected_prior",
        "parent_prior_vs_c0_truth",
    )
    labels = {
        "aggregate_q_vs_selected_truth": "AVI aggregate / selected truth",
        "selected_prior_vs_selected_truth": "Selected prior / selected truth",
        "aggregate_q_vs_selected_prior": "AVI aggregate / selected prior",
        "parent_prior_vs_c0_truth": "Parent prior / C0 truth",
    }
    figure, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    width = 0.19
    positions = np.arange(metrics["parameter"].nunique())
    for offset, comparison in enumerate(comparisons):
        subset = metrics.loc[metrics["comparison"].eq(comparison)]
        axis.bar(
            positions + (offset - 1.5) * width,
            subset["wasserstein_over_target_iqr"].to_numpy(),
            width=width,
            label=labels[comparison],
        )
    names = (
        metrics.loc[metrics["comparison"].eq(comparisons[0]), "parameter"]
        .astype(str)
        .tolist()
    )
    axis.axhline(0.2, color="#555555", linestyle="--", linewidth=1.0)
    axis.set_xticks(positions, names, rotation=55, ha="right")
    axis.set_ylabel("Wasserstein distance / target IQR")
    axis.set_title("Population recovery after targeted selection-corrected VEM")
    axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_individuals(
    path: Path,
    names: tuple[str, ...],
    row_index: np.ndarray,
    q: np.ndarray,
    truth_by_row: dict[int, np.ndarray],
    selected_prior: np.ndarray,
    selected_weights: np.ndarray,
) -> list[int]:
    redshift = np.asarray([truth_by_row[int(row)][0] for row in row_index])
    targets = np.quantile(redshift, [0.05, 0.22, 0.40, 0.60, 0.78, 0.95])
    selected_positions = []
    available = set(range(len(row_index)))
    for target in targets:
        position = min(available, key=lambda item: abs(redshift[item] - target))
        selected_positions.append(position)
        available.remove(position)
    dimensions = min(5, len(names))
    figure, axes = plt.subplots(
        len(selected_positions),
        dimensions,
        figsize=(3.0 * dimensions, 2.1 * len(selected_positions)),
        constrained_layout=True,
    )
    for row_number, position in enumerate(selected_positions):
        truth = truth_by_row[int(row_index[position])]
        for dimension in range(dimensions):
            axis = axes[row_number, dimension]
            low, high = np.quantile(
                np.concatenate(
                    (q[position, :, dimension], selected_prior[:, dimension])
                ),
                [0.005, 0.995],
            )
            bins = np.linspace(low, high, 35)
            axis.hist(
                selected_prior[:, dimension],
                bins=bins,
                weights=selected_weights,
                density=True,
                histtype="stepfilled",
                alpha=0.15,
                color="#D55E00",
                label="Selected learned prior",
            )
            axis.hist(
                q[position, :, dimension],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color="#0072B2",
                label="Approximate AVI q",
            )
            axis.axvline(
                truth[dimension],
                color="#111111",
                linewidth=1.2,
                label="Truth",
            )
            if row_number == 0:
                axis.set_title(names[dimension], fontsize=9)
            if dimension == 0:
                axis.set_ylabel(f"row {int(row_index[position])}\nDensity")
            axis.tick_params(labelsize=7)
    figure.suptitle(
        "Individual approximate AVI posteriors (32 joint draws); line = truth",
        fontsize=13,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)
    return [int(row_index[position]) for position in selected_positions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    final_receipt = root / "POPULATION_VEM_COMPLETE.json"
    if final_receipt.is_file():
        existing = _read_json(final_receipt)
        artifacts = existing.get("artifacts", {})
        if existing.get("status") not in {
            "POPULATION_TARGET_PASS",
            "DIAGNOSTIC_COMPLETE",
        } or any(not Path(path).is_file() for path in artifacts.values()):
            raise ValueError("existing population-VEM completion receipt is invalid")
        print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
        return
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    require_git_commit(repo, manifest["code_commit"])
    refresh = _read_json(root / "q_refresh" / "Q_REFRESH_COMPLETE.json")
    if refresh.get("status") != "COMPLETE" or refresh.get("truth_used") is not False:
        raise ValueError("final evaluation requires a certified truth-free q refresh")
    evaluation_rows = _cohort(manifest, "q_evaluation")
    q_bank_manifest = merge_array_bank_shards(
        root / "banks" / "q_evaluation",
        expected_shards=8,
        expected_row_indices=evaluation_rows,
    )
    prior_bank_manifest = merge_array_bank_shards(
        root / "banks" / "prior_evaluation",
        expected_shards=8,
    )
    for name, bank_manifest in (
        ("q_evaluation", q_bank_manifest),
        ("prior_evaluation", prior_bank_manifest),
    ):
        contract = bank_manifest["contract"]
        if contract["checkpoint_sha256"] != refresh["checkpoint_sha256"]:
            raise ValueError(f"{name} used a different refreshed checkpoint")
        if contract["code_commit"] != manifest["code_commit"]:
            raise ValueError(f"{name} used a different code commit")
        expected_dataset = (
            manifest["datasets"]["test"]["sha256"]
            if name == "q_evaluation"
            else manifest["datasets"]["train"]["sha256"]
        )
        if contract["dataset_sha256"] != expected_dataset:
            raise ValueError(f"{name} used a different dataset")
        if contract["truth_used"] is not False:
            raise ValueError(f"truth entered {name}")
        expected_kind = name
        if contract["kind"] != expected_kind:
            raise ValueError(f"{name} has the wrong bank kind")
        if name == "q_evaluation" and contract.get("draws_per_object") != 32:
            raise ValueError("q_evaluation did not use exactly 32 joint draws")
    q_parts = list(
        iter_array_bank_shards(root / "banks" / "q_evaluation" / "bank_manifest.json")
    )
    prior_parts = list(
        iter_array_bank_shards(
            root / "banks" / "prior_evaluation" / "bank_manifest.json"
        )
    )
    q_rows = np.concatenate([part["row_index"] for part in q_parts]).astype(np.int64)
    q_x = np.concatenate([part["x"] for part in q_parts]).astype(np.float32)
    if len(np.unique(q_rows)) != len(q_rows):
        raise ValueError("final q bank contains duplicate objects")
    prior_x = np.concatenate([part["x"] for part in prior_parts]).astype(np.float32)
    log_beta = np.concatenate([part["log_beta"] for part in prior_parts]).astype(
        np.float64
    )
    if len(prior_x) != int(manifest["banks"]["prior_evaluation"]["samples"]):
        raise ValueError("final prior bank sample count mismatch")
    if q_x.shape[1] != int(manifest["banks"]["q_evaluation"]["draws_per_object"]):
        raise ValueError("final q bank draw count mismatch")
    beta = np.where(np.isfinite(log_beta), np.exp(log_beta), 0.0)
    if not np.isfinite(beta.sum()) or beta.sum() <= 0.0:
        raise ValueError("learned prior has zero selected mass")
    selected_weights = beta / beta.sum()

    config = load_config(resolve_manifest_config(manifest, "config", repo))
    truth_config = load_config(resolve_manifest_config(manifest, "truth_config", repo))
    latent_spec = latent_spec_from_config(config)
    names = tuple(latent_spec.names)
    truth_config["catalog_path"] = manifest["datasets"]["test"]["path"]
    selected_arrays = load_photometry_arrays_from_config(
        truth_config,
        batch_size=10_000,
        row_indices=q_rows,
    )
    c0_arrays = load_photometry_arrays_from_config(
        truth_config,
        batch_size=10_000,
        row_indices=np.arange(manifest["datasets"]["test"]["c0_objects"]),
    )
    selected_truth = _theta_from_truth(selected_arrays, names)
    c0_truth = _theta_from_truth(c0_arrays, names)
    truth_by_row = {
        int(row): selected_truth[index]
        for index, row in enumerate(np.asarray(selected_arrays.row_index))
    }
    if set(q_rows.tolist()) != set(truth_by_row):
        raise ValueError("final q bank and selected truth have different row sets")

    q_theta = _x_to_theta_chunks(q_x.reshape(-1, q_x.shape[-1]), latent_spec)
    q_theta_by_object = q_theta.reshape(q_x.shape)
    prior_theta = _x_to_theta_chunks(prior_x, latent_spec)
    selected_truth_ordered = np.stack([truth_by_row[int(row)] for row in q_rows])

    evaluation = root / "evaluation"
    if evaluation.exists():
        shutil.rmtree(evaluation)
    plots = evaluation / "plots"
    posterior_dir = evaluation / "posterior_q"
    for path in (plots, posterior_dir):
        path.mkdir(parents=True, exist_ok=True)
    truth_frame = pd.DataFrame(selected_truth_ordered, columns=names)
    truth_frame.insert(0, "row_index", q_rows)
    truth_frame.insert(0, "object_id", q_rows)
    truth_path = evaluation / "selected_test_truth.parquet"
    truth_frame.to_parquet(truth_path, index=False)
    for shard, part in enumerate(q_parts):
        rows = np.asarray(part["row_index"], dtype=np.int64)
        theta = _x_to_theta_chunks(
            np.asarray(part["x"]).reshape(-1, q_x.shape[-1]), latent_spec
        ).reshape(part["x"].shape)
        frame = pd.DataFrame(theta.reshape(-1, len(names)), columns=names)
        frame.insert(0, "sample_id", np.tile(np.arange(theta.shape[1]), len(rows)))
        frame.insert(0, "row_index", np.repeat(rows, theta.shape[1]))
        frame.insert(0, "object_id", np.repeat(rows, theta.shape[1]))
        frame.to_parquet(posterior_dir / f"shard_{shard:05d}.parquet", index=False)

    metric_rows = []
    metric_rows.extend(
        _comparison_rows(
            source=q_theta,
            target=selected_truth_ordered,
            names=names,
            comparison="aggregate_q_vs_selected_truth",
        )
    )
    metric_rows.extend(
        _comparison_rows(
            source=prior_theta,
            target=selected_truth_ordered,
            names=names,
            comparison="selected_prior_vs_selected_truth",
            source_weights=selected_weights,
        )
    )
    metric_rows.extend(
        _comparison_rows(
            source=q_theta,
            target=prior_theta,
            names=names,
            comparison="aggregate_q_vs_selected_prior",
            target_weights=selected_weights,
        )
    )
    metric_rows.extend(
        _comparison_rows(
            source=prior_theta,
            target=c0_truth,
            names=names,
            comparison="parent_prior_vs_c0_truth",
        )
    )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(evaluation / "population_recovery.csv", index=False)
    _plot_population_metrics(plots / "population_recovery_metrics.png", metrics)
    np.savez_compressed(
        evaluation / "population_correlations.npz",
        selected_truth=np.corrcoef(selected_truth_ordered, rowvar=False),
        aggregate_q=np.corrcoef(q_theta, rowvar=False),
        c0_truth=np.corrcoef(c0_truth, rowvar=False),
        parent_prior=np.corrcoef(prior_theta, rowvar=False),
        selected_prior=_weighted_correlation(prior_theta, selected_weights),
    )
    _plot_selected_population(
        plots / "selected_population_recovery.png",
        names,
        selected_truth_ordered,
        q_theta,
        prior_theta,
        selected_weights,
    )
    _plot_parent_population(
        plots / "parent_population_recovery.png",
        names,
        c0_truth,
        prior_theta,
    )
    individual_rows = _plot_individuals(
        plots / "individual_posteriors.png",
        names,
        q_rows,
        q_theta_by_object,
        truth_by_row,
        prior_theta,
        selected_weights,
    )

    mira = evaluate_feniks_mira(
        truth_path=truth_path,
        posterior_specs=(("refreshed_avi_q32", posterior_dir),),
        out_dir=evaluation / "mira",
        num_regions=64,
        num_bootstrap=256,
        samples_per_object=32,
        seed=267000,
        parameters=names,
    )
    tarp = evaluate_feniks_tarp(
        truth_path=truth_path,
        posterior_specs=(("refreshed_avi_q32", posterior_dir),),
        out_dir=evaluation / "tarp",
        num_alpha_bins=32,
        num_bootstrap=256,
        samples_per_object=32,
        seed=268000,
        parameters=names,
    )
    by_comparison = {
        comparison: {
            "median_wasserstein_over_target_iqr": float(
                group["wasserstein_over_target_iqr"].median()
            ),
            "redshift_wasserstein_over_target_iqr": float(
                group.loc[
                    group["parameter"].eq(names[0]),
                    "wasserstein_over_target_iqr",
                ].iloc[0]
            ),
        }
        for comparison, group in metrics.groupby("comparison")
    }
    population_target_pass = bool(
        by_comparison["aggregate_q_vs_selected_truth"][
            "redshift_wasserstein_over_target_iqr"
        ]
        <= 0.15
        and by_comparison["selected_prior_vs_selected_truth"][
            "redshift_wasserstein_over_target_iqr"
        ]
        <= 0.20
        and by_comparison["parent_prior_vs_c0_truth"][
            "redshift_wasserstein_over_target_iqr"
        ]
        <= 0.30
    )
    receipt = {
        "status": (
            "POPULATION_TARGET_PASS"
            if population_target_pass
            else "DIAGNOSTIC_COMPLETE"
        ),
        "population_target_pass": population_target_pass,
        "scientific_promotion": False,
        "scientific_promotion_limit": (
            "q is an approximate AVI posterior with 32 draws per object; exact "
            "importance-support promotion is outside this time-bounded workflow"
        ),
        "stage": 5,
        "method": manifest["method"],
        "test_objects": int(len(q_rows)),
        "q_draws_per_object": int(q_x.shape[1]),
        "prior_draws": int(len(prior_x)),
        "posterior_role": (
            "approximate stochastic-ELBO AVI; dense joint draws; no importance "
            "correction"
        ),
        "population_contracts": {
            "parent_prior": "p_eta(theta|C0)",
            "selected_prior": "beta(theta) p_eta(theta|C0) / alpha_eta",
            "posterior_aggregate": (
                "object-equal mixture of q(theta|y); never relabeled as parent prior"
            ),
        },
        "truth_used_for_training_or_checkpoint_selection": False,
        "truth_used_for_final_closure": True,
        "metrics": by_comparison,
        "mira_status": mira["status"],
        "tarp_status": tarp["status"],
        "individual_rows": individual_rows,
        "artifacts": {
            "population_recovery": str(
                (evaluation / "population_recovery.csv").resolve()
            ),
            "population_metrics_plot": str(
                (plots / "population_recovery_metrics.png").resolve()
            ),
            "population_metrics_plot_pdf": str(
                (plots / "population_recovery_metrics.pdf").resolve()
            ),
            "selected_population_plot": str(
                (plots / "selected_population_recovery.png").resolve()
            ),
            "selected_population_plot_pdf": str(
                (plots / "selected_population_recovery.pdf").resolve()
            ),
            "parent_population_plot": str(
                (plots / "parent_population_recovery.png").resolve()
            ),
            "parent_population_plot_pdf": str(
                (plots / "parent_population_recovery.pdf").resolve()
            ),
            "individual_plot": str((plots / "individual_posteriors.png").resolve()),
            "individual_plot_pdf": str((plots / "individual_posteriors.pdf").resolve()),
            "mira": str((evaluation / "mira" / "mira_summary.json").resolve()),
            "tarp": str((evaluation / "tarp" / "tarp_summary.json").resolve()),
        },
    }
    _write_json(final_receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
