#!/usr/bin/env python3
"""Compare bootstrap/distilled q and extended SMC against canonical-target NUTS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh


def _galaxy_dir(root: Path, item) -> Path:
    return root / "galaxies" / (
        f"{int(item.order):02d}_{item.example_key}_row{int(item.row_index)}"
    )


def _weighted_quantile(values, weights, probability: float) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return float(np.interp(float(probability), cumulative, sorted_values))


def _moments(frame: pd.DataFrame, columns: list[str], weight=None):
    values = frame[columns].to_numpy(dtype=np.float64)
    if weight is None:
        normalized = np.full(len(values), 1.0 / len(values))
    else:
        normalized = np.array(weight, dtype=np.float64, copy=True)
        normalized /= np.sum(normalized)
    mean = np.sum(values * normalized[:, None], axis=0)
    centered = values - mean
    covariance = (centered * normalized[:, None]).T @ centered
    correction = max(1.0 - float(np.sum(normalized**2)), 1.0e-12)
    covariance /= correction
    quantiles = np.asarray(
        [
            [
                _weighted_quantile(values[:, index], normalized, probability)
                for probability in (0.16, 0.50, 0.84)
            ]
            for index in range(values.shape[1])
        ]
    )
    return mean, covariance, quantiles


def _covariance_ratios(covariance, reference) -> np.ndarray:
    dimension = int(reference.shape[0])
    scale = max(float(np.trace(reference)) / dimension, 1.0e-12)
    regularization = scale * 1.0e-8
    identity = np.eye(dimension)
    return eigvalsh(
        covariance + regularization * identity,
        reference + regularization * identity,
    )


def _finite_median(values) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else None


def _finite_quantile(values, probability: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, probability)) if finite.size else None


def summarize(
    *,
    bootstrap_root: Path,
    distilled_root: Path,
    out: Path,
) -> dict[str, Any]:
    bootstrap_cohort = pd.read_parquet(bootstrap_root / "cohort.parquet")
    distilled_cohort = pd.read_parquet(distilled_root / "cohort.parquet")
    identity_columns = ["order", "example_key", "row_index", "object_id"]
    if not bootstrap_cohort[identity_columns].equals(
        distilled_cohort[identity_columns]
    ):
        raise ValueError("bootstrap and distilled cohorts differ")
    if len(bootstrap_cohort) != 8:
        raise ValueError(f"teacher audit requires exactly 8 objects, got {len(bootstrap_cohort)}")

    posterior_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for item in bootstrap_cohort.itertuples(index=False):
        before_dir = _galaxy_dir(bootstrap_root, item)
        after_dir = _galaxy_dir(distilled_root, item)
        manifest = json.loads((before_dir / "prepare_manifest.json").read_text())
        names = list(manifest["latent_spec"]["names"])
        columns = [f"x_{name}" for name in names]
        frames = {
            "q_bootstrap": pd.read_parquet(before_dir / "encoder_samples.parquet"),
            "q_distilled": pd.read_parquet(after_dir / "encoder_samples.parquet"),
            "teacher_smc": pd.read_parquet(
                before_dir / "adaptive_smc_weighted_samples.parquet"
            ),
            "nuts": pd.read_parquet(before_dir / "nuts/samples.parquet"),
        }
        moments = {}
        for method, frame in frames.items():
            weight = frame["smc_weight"].to_numpy() if method == "teacher_smc" else None
            moments[method] = _moments(frame, columns, weight)
        nuts_mean, nuts_covariance, nuts_quantiles = moments["nuts"]
        nuts_scale = np.maximum(np.sqrt(np.diag(nuts_covariance)), 1.0e-8)
        for method, (mean, covariance, quantiles) in moments.items():
            width = np.maximum(quantiles[:, 2] - quantiles[:, 0], 1.0e-12)
            nuts_width = np.maximum(
                nuts_quantiles[:, 2] - nuts_quantiles[:, 0], 1.0e-12
            )
            for index, name in enumerate(names):
                posterior_rows.append(
                    {
                        "row_index": int(item.row_index),
                        "object_id": str(item.object_id),
                        "parameter": name,
                        "method": method,
                        "mean": float(mean[index]),
                        "std": float(np.sqrt(max(covariance[index, index], 0.0))),
                        "q16": float(quantiles[index, 0]),
                        "q50": float(quantiles[index, 1]),
                        "q84": float(quantiles[index, 2]),
                        "mean_abs_z_vs_nuts": float(
                            abs(mean[index] - nuts_mean[index]) / nuts_scale[index]
                        ),
                        "central68_width_ratio_vs_nuts": float(
                            width[index] / nuts_width[index]
                        ),
                    }
                )
            ratios = _covariance_ratios(covariance, nuts_covariance)
            covariance_rows.append(
                {
                    "row_index": int(item.row_index),
                    "object_id": str(item.object_id),
                    "method": method,
                    "eigen_ratio_min": float(np.min(ratios)),
                    "eigen_ratio_median": float(np.median(ratios)),
                    "eigen_ratio_max": float(np.max(ratios)),
                }
            )

        smc = json.loads((before_dir / "adaptive_smc_diagnostics.json").read_text())
        nuts = json.loads((before_dir / "nuts/diagnostics.json").read_text())
        selected_smc = smc["fallback"] if smc["fallback_attempted"] else smc["primary"]
        object_rows.append(
            {
                "row_index": int(item.row_index),
                "object_id": str(item.object_id),
                "smc_beta_final": selected_smc["beta_final"],
                "smc_eligible": smc["eligible_after_fallback"],
                "smc_acceptance": selected_smc["mutation_acceptance"],
                "smc_ancestor_ess_fraction": selected_smc["ancestor_ess_fraction"],
                "smc_epsilon_squared_jump": selected_smc["epsilon_squared_jump"],
                "smc_mixing_failure": selected_smc["mixing_failure"],
                "nuts_max_rhat": nuts["max_rhat"],
                "nuts_min_bulk_ess": nuts["min_bulk_ess"],
                "nuts_min_tail_ess": nuts["min_tail_ess"],
            }
        )
        for stage, directory in (("bootstrap", before_dir), ("distilled", after_dir)):
            values = json.loads((directory / "importance_diagnostics.json").read_text())
            importance_rows.append(
                {
                    "row_index": int(item.row_index),
                    "object_id": str(item.object_id),
                    "stage": stage,
                    "raw_ess_fraction": values["raw_ess_fraction"],
                    "raw_max_weight": values["raw_max_weight"],
                    "pareto_k": values["pareto_k"],
                }
            )

    posterior = pd.DataFrame(posterior_rows)
    covariance = pd.DataFrame(covariance_rows)
    objects = pd.DataFrame(object_rows)
    importance = pd.DataFrame(importance_rows)
    out.mkdir(parents=True, exist_ok=False)
    posterior.to_csv(out / "posterior_agreement.csv", index=False)
    posterior.to_parquet(out / "posterior_agreement.parquet", index=False)
    covariance.to_csv(out / "covariance_agreement.csv", index=False)
    objects.to_csv(out / "object_diagnostics.csv", index=False)
    importance.to_csv(out / "q_importance_diagnostics.csv", index=False)

    def method_summary(method: str) -> dict[str, float | None]:
        rows = posterior.loc[posterior["method"].eq(method)]
        geometry = covariance.loc[covariance["method"].eq(method)]
        return {
            "median_abs_mean_z_vs_nuts": _finite_median(rows["mean_abs_z_vs_nuts"]),
            "q90_abs_mean_z_vs_nuts": _finite_quantile(rows["mean_abs_z_vs_nuts"], 0.90),
            "median_width_ratio_vs_nuts": _finite_median(
                rows["central68_width_ratio_vs_nuts"]
            ),
            "fraction_width_ratio_between_0p5_and_2": float(
                np.mean(
                    rows["central68_width_ratio_vs_nuts"].between(0.5, 2.0)
                )
            ),
            "median_covariance_eigen_ratio_vs_nuts": _finite_median(
                geometry["eigen_ratio_median"]
            ),
        }

    q_is = {}
    for stage in ("bootstrap", "distilled"):
        rows = importance.loc[importance["stage"].eq(stage)]
        q_is[stage] = {
            "median_ess_fraction": _finite_median(rows["raw_ess_fraction"]),
            "median_max_weight": _finite_median(rows["raw_max_weight"]),
            "median_pareto_k": _finite_median(rows["pareto_k"]),
        }
    method_summaries = {
        method: method_summary(method)
        for method in ("q_bootstrap", "q_distilled", "teacher_smc")
    }
    teacher = method_summaries["teacher_smc"]
    distilled = method_summaries["q_distilled"]
    checks = {
        "nuts_converged": bool(
            objects["nuts_max_rhat"].notna().all()
            and (objects["nuts_max_rhat"] <= 1.01).all()
            and (objects["nuts_min_bulk_ess"] >= 400.0).all()
            and (objects["nuts_min_tail_ess"] >= 400.0).all()
        ),
        "teacher_reached_beta_one": bool(
            np.allclose(objects["smc_beta_final"], 1.0, atol=1.0e-6)
        ),
        "teacher_all_eligible": bool(objects["smc_eligible"].all()),
        "teacher_mean_agreement": bool(
            teacher["median_abs_mean_z_vs_nuts"] <= 0.25
            and teacher["q90_abs_mean_z_vs_nuts"] <= 0.50
        ),
        "teacher_width_agreement": bool(
            0.8 <= teacher["median_width_ratio_vs_nuts"] <= 1.25
            and teacher["fraction_width_ratio_between_0p5_and_2"] >= 0.90
        ),
        "teacher_covariance_agreement": bool(
            0.5 <= teacher["median_covariance_eigen_ratio_vs_nuts"] <= 2.0
        ),
        "distilled_q_is_supported": bool(
            q_is["distilled"]["median_ess_fraction"] >= 0.05
            and q_is["distilled"]["median_max_weight"] <= 0.80
        ),
        "distilled_q_geometry_improved": bool(
            distilled["median_abs_mean_z_vs_nuts"]
            < method_summaries["q_bootstrap"]["median_abs_mean_z_vs_nuts"]
        ),
    }
    teacher_ready = all(
        checks[name]
        for name in (
            "nuts_converged",
            "teacher_reached_beta_one",
            "teacher_all_eligible",
            "teacher_mean_agreement",
            "teacher_width_agreement",
            "teacher_covariance_agreement",
        )
    )
    q_ready = bool(
        checks["distilled_q_is_supported"]
        and checks["distilled_q_geometry_improved"]
    )
    if not checks["nuts_converged"]:
        next_action = "EXTEND_OR_REPAIR_NUTS_REFERENCE"
    elif not teacher_ready:
        next_action = "FIX_OR_REJECT_EXTENDED_SMC_TEACHER"
    elif not q_ready:
        next_action = "FIX_Q_DISTILLATION_OR_POSTERIOR_PARAMETERIZATION"
    else:
        next_action = "RUN_FRESH_PRIOR_MSTEP_SMOKE"
    receipt = {
        "status": "PASS" if teacher_ready and q_ready else "DIAGNOSTIC_COMPLETE",
        "contract": (
            "same eight observations and canonical target; truth is not read; "
            "no optimizer update is performed"
        ),
        "objects": int(len(objects)),
        "method_summaries": method_summaries,
        "q_only_importance": q_is,
        "checks": checks,
        "teacher_ready": teacher_ready,
        "q_ready": q_ready,
        "next_action": next_action,
    }
    (out / "teacher_audit_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "DONE").touch()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-root", type=Path, required=True)
    parser.add_argument("--distilled-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
