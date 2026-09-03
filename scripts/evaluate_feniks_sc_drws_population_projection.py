#!/usr/bin/env python3
"""Evaluate direct population projections and object-level posterior calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from euclid_dsps.amortized.data import load_photometry_arrays_from_config
from euclid_dsps.amortized.latent import latent_spec_from_config, x_to_theta
from euclid_dsps.amortized.population_projection import (
    distribution_comparison,
    evaluate_log_beta,
    inverse_selection_weights,
    selection_runtime,
    weighted_cdf_values,
)
from euclid_dsps.amortized.population_vem import (
    iter_array_bank_shards,
    require_git_commit,
    resolve_manifest_config,
    sha256_file,
)
from euclid_dsps.amortized.train import load_checkpoint
from euclid_dsps.config import load_config

try:
    from scripts.evaluate_redshift_pit_coverage import evaluate as evaluate_redshift
except ModuleNotFoundError:
    from evaluate_redshift_pit_coverage import evaluate as evaluate_redshift


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_q(path: Path) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(part["x"], dtype=np.float32).reshape(-1, part["x"].shape[-1])
            for part in iter_array_bank_shards(path)
        ]
    )


def _load_parent_target(path: Path) -> tuple[np.ndarray, np.ndarray]:
    parts = list(iter_array_bank_shards(path))
    x = np.concatenate(
        [
            np.asarray(part["x"], dtype=np.float32).reshape(-1, part["x"].shape[-1])
            for part in parts
        ]
    )
    log_beta = np.concatenate(
        [np.asarray(part["log_beta"], dtype=np.float64).reshape(-1) for part in parts]
    )
    weights, _diagnostics = inverse_selection_weights(log_beta)
    return x, weights


def _x_to_theta(x: np.ndarray, latent_spec) -> np.ndarray:
    pieces = []
    for start in range(0, len(x), 32768):
        pieces.append(
            np.asarray(
                jax.device_get(
                    x_to_theta(jnp.asarray(x[start : start + 32768]), latent_spec)
                ),
                dtype=np.float64,
            )
        )
    return np.concatenate(pieces)


def _theta_from_truth(arrays, names: tuple[str, ...]) -> np.ndarray:
    if not arrays.truth or any(name not in arrays.truth for name in names):
        raise ValueError("population closure is missing truth parameters")
    theta = np.column_stack([np.asarray(arrays.truth[name]) for name in names])
    if not np.all(np.isfinite(theta)):
        raise ValueError("population closure truth contains non-finite values")
    return theta.astype(np.float64)


def _sample_prior(prior, seed: int, samples: int) -> np.ndarray:
    @jax.jit
    def draw(key):
        return prior.sample(key, int(samples))

    return np.asarray(
        jax.device_get(draw(jax.random.PRNGKey(int(seed)))), dtype=np.float32
    )


def _comparison_rows(
    *,
    comparison: str,
    role: str,
    source: np.ndarray,
    target: np.ndarray,
    names: tuple[str, ...],
    source_weights: np.ndarray | None = None,
    target_weights: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "comparison": comparison,
                "role": role,
                "parameter": name,
                **distribution_comparison(
                    source[:, index],
                    target[:, index],
                    source_weights=source_weights,
                    target_weights=target_weights,
                ),
            }
        )
    return rows


def _plot_redshift_projection(
    path: Path,
    q: np.ndarray,
    selected: np.ndarray,
    parent_target: np.ndarray,
    parent_target_weights: np.ndarray,
    parent: np.ndarray,
    parent_selected_weights: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    low = min(np.quantile(q[:, 0], 0.002), np.quantile(parent_target[:, 0], 0.002))
    high = max(np.quantile(q[:, 0], 0.998), np.quantile(parent_target[:, 0], 0.998))
    grid = np.linspace(low, high, 500)
    curves = (
        (q[:, 0], None, "q aggregate target", "#0072B2"),
        (selected[:, 0], None, "direct selected flow", "#D55E00"),
        (
            parent[:, 0],
            parent_selected_weights,
            "beta-weighted parent flow",
            "#009E73",
        ),
    )
    for values, weights, label, color in curves:
        axes[0].plot(
            grid,
            weighted_cdf_values(values, grid, weights),
            label=label,
            color=color,
            linewidth=1.8,
        )
    axes[0].set(
        xlabel="redshift",
        ylabel="CDF",
        title="Selected-population distribution projection",
    )
    axes[0].legend(frameon=False, fontsize=8)
    for values, weights, label, color in (
        (parent_target[:, 0], parent_target_weights, "q / beta target", "#CC79A7"),
        (parent[:, 0], None, "direct parent flow", "#009E73"),
    ):
        axes[1].plot(
            grid,
            weighted_cdf_values(values, grid, weights),
            label=label,
            color=color,
            linewidth=1.8,
        )
    axes[1].set(
        xlabel="redshift",
        ylabel="CDF",
        title="Parent C0 distribution projection",
    )
    axes[1].legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _plot_physical_projection(
    path: Path,
    names: tuple[str, ...],
    q: np.ndarray,
    selected: np.ndarray,
    parent_target: np.ndarray,
    parent_target_weights: np.ndarray,
    parent: np.ndarray,
    parent_selected_weights: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(16, 6.2), constrained_layout=True)
    for dimension in range(min(5, len(names))):
        low, high = np.quantile(
            np.concatenate((q[:, dimension], parent_target[:, dimension])),
            [0.002, 0.998],
        )
        bins = np.linspace(low, high, 60)
        axes[0, dimension].hist(
            q[:, dimension],
            bins=bins,
            density=True,
            histtype="step",
            color="#0072B2",
            label="q aggregate",
        )
        axes[0, dimension].hist(
            selected[:, dimension],
            bins=bins,
            density=True,
            histtype="step",
            color="#D55E00",
            label="selected flow",
        )
        axes[0, dimension].hist(
            parent[:, dimension],
            bins=bins,
            weights=parent_selected_weights,
            density=True,
            histtype="step",
            color="#009E73",
            label="selected parent",
        )
        axes[0, dimension].set_title(names[dimension], fontsize=9)
        axes[1, dimension].hist(
            parent_target[:, dimension],
            bins=bins,
            weights=parent_target_weights,
            density=True,
            histtype="step",
            color="#CC79A7",
            label="q / beta target",
        )
        axes[1, dimension].hist(
            parent[:, dimension],
            bins=bins,
            density=True,
            histtype="step",
            color="#009E73",
            label="parent flow",
        )
        axes[1, dimension].set_xlabel(names[dimension], fontsize=9)
    axes[0, 0].set_ylabel("Selected density")
    axes[1, 0].set_ylabel("Parent density")
    axes[0, 0].legend(frameon=False, fontsize=7)
    axes[1, 0].legend(frameon=False, fontsize=7)
    figure.suptitle("Joint-draw population projections (marginal views)")
    figure.savefig(path, dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-samples", type=int, default=65536)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    require_git_commit(repo, manifest["code_commit"])
    fit = _read_json(root / "PROJECTION_FIT_COMPLETE.json")
    if fit.get("status") != "COMPLETE" or fit.get("truth_used") is not False:
        raise ValueError("evaluation requires a complete truth-free projection fit")
    final_path = root / "POPULATION_PROJECTION_COMPLETE.json"
    if final_path.is_file():
        existing = _read_json(final_path)
        if existing.get("status") != "DIAGNOSTIC_COMPLETE":
            raise ValueError("existing projection completion receipt is invalid")
        print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
        return

    config = load_config(resolve_manifest_config(manifest, "config", repo))
    truth_config = load_config(resolve_manifest_config(manifest, "truth_config", repo))
    latent_spec = latent_spec_from_config(config)
    names = tuple(latent_spec.names)
    selected_checkpoint = Path(fit["selected"]["checkpoint"])
    parent_checkpoint = Path(fit["parent"]["checkpoint"])
    for record, checkpoint in (
        (fit["selected"], selected_checkpoint),
        (fit["parent"], parent_checkpoint),
    ):
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError(f"projection checkpoint SHA256 mismatch: {checkpoint}")
    selected_model = load_checkpoint(selected_checkpoint, config)
    parent_model = load_checkpoint(parent_checkpoint, config)

    q_x = _load_q(Path(manifest["q_banks"]["validation"]["manifest"]))
    parent_target_x, parent_target_weights = _load_parent_target(
        root / "banks" / "beta_validation" / "bank_manifest.json"
    )
    selected_x = _sample_prior(selected_model.prior, 270100, int(args.prior_samples))
    parent_x = _sample_prior(parent_model.prior, 270200, int(args.prior_samples))
    runtime = selection_runtime(config, Path(manifest["source"]["feature_stats"]))
    parent_log_beta = evaluate_log_beta(parent_model, parent_x, runtime, chunk_size=512)
    parent_beta = np.where(np.isfinite(parent_log_beta), np.exp(parent_log_beta), 0.0)
    if parent_beta.sum() <= 0.0 or not np.isfinite(parent_beta.sum()):
        raise ValueError("fitted parent has no finite selected mass")
    parent_selected_weights = parent_beta / parent_beta.sum()

    q_theta = _x_to_theta(q_x, latent_spec)
    parent_target_theta = _x_to_theta(parent_target_x, latent_spec)
    selected_theta = _x_to_theta(selected_x, latent_spec)
    parent_theta = _x_to_theta(parent_x, latent_spec)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _comparison_rows(
            comparison="selected_flow_vs_q_aggregate",
            role="truth_free_validation_distribution",
            source=selected_theta,
            target=q_theta,
            names=names,
        )
    )
    rows.extend(
        _comparison_rows(
            comparison="parent_flow_vs_inverse_beta_q",
            role="truth_free_validation_distribution",
            source=parent_theta,
            target=parent_target_theta,
            names=names,
            target_weights=parent_target_weights,
        )
    )
    rows.extend(
        _comparison_rows(
            comparison="selected_parent_flow_vs_q_aggregate",
            role="truth_free_validation_distribution",
            source=parent_theta,
            target=q_theta,
            names=names,
            source_weights=parent_selected_weights,
        )
    )

    calibration = manifest["independent_posterior_calibration"]
    for record in calibration["posterior_files"]:
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"independent posterior shard changed: {record['path']}")
    if sha256_file(calibration["truth"]) != calibration["truth_sha256"]:
        raise ValueError("independent posterior truth table changed")
    final_evaluation = root / "evaluation"
    if final_evaluation.exists():
        raise FileExistsError(
            f"incomplete projection evaluation already exists: {final_evaluation}"
        )
    evaluation = root / f".evaluation-attempt-{os.environ.get('SLURM_JOB_ID', 'local')}"
    if evaluation.exists():
        raise FileExistsError(
            f"projection evaluation attempt already exists: {evaluation}"
        )
    plots = evaluation / "distribution_projection" / "plots"
    plots.mkdir(parents=True, exist_ok=False)
    posterior_calibration = evaluate_redshift(
        truth_path=Path(calibration["truth"]),
        posterior_specs=[("vem1_refreshed_q32", Path(calibration["posterior"]))],
        out=evaluation / "posterior_calibration" / "redshift",
        truth_column="z_obs",
        samples_per_object=int(calibration["draws_per_object"]),
        bootstrap=512,
        seed=270300,
        expected_objects=int(calibration["objects"]),
        scope="feniks_selected_test_same_object_redshift_posterior_calibration",
    )

    truth_config["catalog_path"] = manifest["datasets"]["test"]["path"]
    c0_arrays = load_photometry_arrays_from_config(
        truth_config,
        batch_size=10000,
        row_indices=np.arange(int(manifest["datasets"]["test"]["c0_objects"])),
    )
    c0_truth = _theta_from_truth(c0_arrays, names)
    selected_truth_frame = pd.read_parquet(calibration["truth"])
    selected_truth = selected_truth_frame.loc[:, names].to_numpy(np.float64)
    rows.extend(
        _comparison_rows(
            comparison="selected_flow_vs_independent_selected_truth",
            role="frozen_population_truth_closure",
            source=selected_theta,
            target=selected_truth,
            names=names,
        )
    )
    rows.extend(
        _comparison_rows(
            comparison="selected_parent_flow_vs_independent_selected_truth",
            role="frozen_population_truth_closure",
            source=parent_theta,
            target=selected_truth,
            names=names,
            source_weights=parent_selected_weights,
        )
    )
    rows.extend(
        _comparison_rows(
            comparison="parent_flow_vs_independent_c0_truth",
            role="frozen_population_truth_closure",
            source=parent_theta,
            target=c0_truth,
            names=names,
        )
    )
    metrics = pd.DataFrame(rows)
    metrics_path = evaluation / "distribution_projection" / "distribution_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    _plot_redshift_projection(
        plots / "redshift_distribution_projection.png",
        q_theta,
        selected_theta,
        parent_target_theta,
        parent_target_weights,
        parent_theta,
        parent_selected_weights,
    )
    _plot_physical_projection(
        plots / "physical_distribution_projection.png",
        names,
        q_theta,
        selected_theta,
        parent_target_theta,
        parent_target_weights,
        parent_theta,
        parent_selected_weights,
    )

    validation = metrics.loc[metrics["role"].eq("truth_free_validation_distribution")]
    by_comparison = {
        comparison: {
            "redshift_cdf_supremum": float(
                group.loc[group["parameter"].eq(names[0]), "cdf_supremum"].iloc[0]
            ),
            "redshift_distribution_rank_uniform_ks": float(
                group.loc[
                    group["parameter"].eq(names[0]),
                    "distribution_rank_uniform_ks",
                ].iloc[0]
            ),
            "maximum_physical_5d_cdf_supremum": float(
                group.loc[group["parameter"].isin(names[:5]), "cdf_supremum"].max()
            ),
            "mean_15d_cdf_supremum": float(group["cdf_supremum"].mean()),
        }
        for comparison, group in validation.groupby("comparison")
    }
    q_calibration = posterior_calibration["models"]["vem1_refreshed_q32"]
    tolerances = {
        "posterior_redshift_pit_ks": 0.05,
        "posterior_redshift_coverage_ece": 0.05,
        "selected_redshift_cdf_supremum": 0.05,
        "parent_redshift_cdf_supremum": 0.07,
        "physical_5d_cdf_supremum": 0.10,
    }
    posterior_calibration_pass = bool(
        q_calibration["pit_ks_uniform"] <= tolerances["posterior_redshift_pit_ks"]
        and q_calibration["coverage_ece"]
        <= tolerances["posterior_redshift_coverage_ece"]
    )
    distribution_projection_pass = bool(
        by_comparison["selected_flow_vs_q_aggregate"]["redshift_cdf_supremum"]
        <= tolerances["selected_redshift_cdf_supremum"]
        and by_comparison["selected_parent_flow_vs_q_aggregate"][
            "redshift_cdf_supremum"
        ]
        <= tolerances["parent_redshift_cdf_supremum"]
        and by_comparison["selected_flow_vs_q_aggregate"][
            "maximum_physical_5d_cdf_supremum"
        ]
        <= tolerances["physical_5d_cdf_supremum"]
    )
    for name, key in (("mira", "mira_sha256"), ("tarp", "tarp_sha256")):
        if sha256_file(calibration[name]) != calibration[key]:
            raise ValueError(f"independent {name.upper()} receipt changed")
    mira_summary = _read_json(Path(calibration["mira"]))
    tarp_summary = _read_json(Path(calibration["tarp"]))
    os.replace(evaluation, final_evaluation)
    final_plots = final_evaluation / "distribution_projection" / "plots"
    final_metrics = (
        final_evaluation / "distribution_projection" / "distribution_metrics.csv"
    )
    receipt = {
        "status": "DIAGNOSTIC_COMPLETE",
        "method": manifest["method"],
        "objects": {
            "selected_train_fit": int(manifest["q_banks"]["fit"]["objects"]),
            "selected_train_validation": int(
                manifest["q_banks"]["validation"]["objects"]
            ),
            "independent_selected_test": int(calibration["objects"]),
            "independent_c0_test": int(manifest["datasets"]["test"]["c0_objects"]),
        },
        "posterior_calibration": {
            "contract": "object-aligned finite-rank PIT and central coverage",
            "redshift": q_calibration,
            "pass": posterior_calibration_pass,
            "mira": calibration["mira"],
            "mira_status": mira_summary.get("status"),
            "tarp": calibration["tarp"],
            "tarp_status": tarp_summary.get("status"),
        },
        "distribution_projection": {
            "contract": (
                "full weighted population distributions; distribution ranks are "
                "not posterior PIT"
            ),
            "comparisons": by_comparison,
            "pass": distribution_projection_pass,
        },
        "tolerances": tolerances,
        "parent_selection": {
            "alpha_monte_carlo": float(np.mean(parent_beta)),
            "selected_weight_ess": float(
                np.square(parent_beta.sum()) / np.square(parent_beta).sum()
            ),
            "selected_weight_ess_fraction": float(
                np.square(parent_beta.sum())
                / np.square(parent_beta).sum()
                / len(parent_beta)
            ),
        },
        "truth_used_for_training_or_checkpoint_selection": False,
        "truth_used_for_final_closure": True,
        "point_estimates_used": False,
        "redshift_median_gate_used": False,
        "scientific_promotion": False,
        "scientific_limit": (
            "the population targets inherit the approximate VEM-1 q; individual "
            "importance support remains a separate limitation"
        ),
        "artifacts": {
            "distribution_metrics": str(final_metrics.resolve()),
            "redshift_distribution_plot": str(
                (final_plots / "redshift_distribution_projection.png").resolve()
            ),
            "physical_distribution_plot": str(
                (final_plots / "physical_distribution_projection.png").resolve()
            ),
            "redshift_pit_summary": str(
                (
                    final_evaluation
                    / "posterior_calibration"
                    / "redshift"
                    / "redshift_calibration_summary.json"
                ).resolve()
            ),
            "redshift_pit_plot": str(
                (
                    final_evaluation
                    / "posterior_calibration"
                    / "redshift"
                    / "redshift_pit_coverage.png"
                ).resolve()
            ),
        },
        "artifact_sha256": {
            "distribution_metrics": sha256_file(final_metrics),
            "redshift_distribution_plot": sha256_file(
                final_plots / "redshift_distribution_projection.png"
            ),
            "physical_distribution_plot": sha256_file(
                final_plots / "physical_distribution_projection.png"
            ),
            "redshift_pit_summary": sha256_file(
                final_evaluation
                / "posterior_calibration"
                / "redshift"
                / "redshift_calibration_summary.json"
            ),
            "redshift_pit_plot": sha256_file(
                final_evaluation
                / "posterior_calibration"
                / "redshift"
                / "redshift_pit_coverage.png"
            ),
        },
    }
    _write_json(final_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
