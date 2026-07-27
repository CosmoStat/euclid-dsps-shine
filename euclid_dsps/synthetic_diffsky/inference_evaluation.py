"""Evaluate closure posterior outputs against synthetic DSPS truths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir
from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES

from .photometry import GROUND_TRUTH_COLUMNS


def evaluate_closure_inference(
    *,
    run_dir: str | Path,
    dataset_path: str | Path,
    out_dir: str | Path | None = None,
) -> Path:
    """Evaluate posterior summaries/samples against held-out closure truths."""
    run = Path(run_dir)
    out = ensure_dir(out_dir or (run / "closure_evaluation"))
    truth = pd.read_parquet(dataset_path)
    if "object_id" not in truth:
        raise ValueError("Closure test dataset must contain object_id")
    summary_path = run / "posterior_summary.parquet"
    samples_path = run / "posterior_samples.parquet"
    if not summary_path.exists() and not samples_path.exists():
        raise FileNotFoundError(
            f"{run} must contain posterior_summary.parquet or posterior_samples.parquet"
        )
    summary = pd.read_parquet(summary_path) if summary_path.exists() else None
    samples = pd.read_parquet(samples_path) if samples_path.exists() else None
    parameter_metrics = _parameter_metrics(truth, summary, samples)
    parameter_metrics.to_csv(out / "posterior_truth_parameter_metrics.csv", index=False)
    coverage = _coverage_metrics(truth, summary, samples)
    coverage.to_csv(out / "posterior_truth_coverage.csv", index=False)
    residual = _posterior_predictive_residual_payload(run)
    payload: dict[str, Any] = {
        "run_dir": str(run),
        "dataset_path": str(dataset_path),
        "n_truth_rows": int(len(truth)),
        "n_parameters": int(len(DIFFSKY_BASIC_PARAMETER_NAMES)),
        "parameter_metrics": parameter_metrics.to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
        "posterior_predictive": residual,
        "interpretation": (
            "Closure inference is evaluated by calibration and posterior "
            "predictive checks, not by MAP proximity alone."
        ),
    }
    report = out / "closure_inference_evaluation.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    _write_markdown_report(out / "closure_inference_evaluation.md", payload)
    return report


def _parameter_metrics(
    truth: pd.DataFrame,
    summary: pd.DataFrame | None,
    samples: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    truth_by_id = truth.set_index("object_id")
    summary_by_id = (
        summary.set_index("object_id")
        if summary is not None and "object_id" in summary
        else None
    )
    sample_groups = (
        samples.groupby("object_id")
        if samples is not None and "object_id" in samples
        else None
    )
    for name in DIFFSKY_BASIC_PARAMETER_NAMES:
        truth_col = GROUND_TRUTH_COLUMNS[name]
        if truth_col not in truth_by_id:
            continue
        y_true = pd.to_numeric(truth_by_id[truth_col], errors="coerce")
        if sample_groups is not None and name in samples:
            pred = sample_groups[name].mean()
        elif summary_by_id is not None and f"{name}_median" in summary_by_id:
            pred = pd.to_numeric(summary_by_id[f"{name}_median"], errors="coerce")
        else:
            continue
        joined = pd.concat(
            [y_true.rename("truth"), pred.rename("prediction")], axis=1
        ).dropna()
        if joined.empty:
            continue
        delta = joined["prediction"].to_numpy(float) - joined["truth"].to_numpy(float)
        rows.append(
            {
                "parameter": name,
                "n": int(len(joined)),
                "posterior_mean_minus_truth_mean": float(np.mean(delta)),
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "median_abs_error": float(np.median(np.abs(delta))),
            }
        )
    return pd.DataFrame(rows)


def _coverage_metrics(
    truth: pd.DataFrame,
    summary: pd.DataFrame | None,
    samples: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    truth_by_id = truth.set_index("object_id")
    for name in DIFFSKY_BASIC_PARAMETER_NAMES:
        truth_col = GROUND_TRUTH_COLUMNS[name]
        if truth_col not in truth_by_id:
            continue
        y_true = pd.to_numeric(truth_by_id[truth_col], errors="coerce")
        if samples is not None and name in samples and "object_id" in samples:
            grouped = samples.groupby("object_id")[name]
            quantiles = {
                level: grouped.quantile(level)
                for level in (0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975)
            }
            rank_rows = []
            for oid, group in grouped:
                if oid in y_true.index and np.isfinite(y_true.loc[oid]):
                    values = group.to_numpy(float)
                    rank_rows.append(float(np.mean(values <= float(y_true.loc[oid]))))
            for label, low_q, high_q, nominal in (
                ("50", 0.25, 0.75, 0.50),
                ("68", 0.16, 0.84, 0.68),
                ("90", 0.05, 0.95, 0.90),
                ("95", 0.025, 0.975, 0.95),
            ):
                joined = pd.concat(
                    [
                        y_true.rename("truth"),
                        quantiles[low_q].rename("low"),
                        quantiles[high_q].rename("high"),
                    ],
                    axis=1,
                ).dropna()
                rows.append(
                    {
                        "parameter": name,
                        "interval": label,
                        "nominal": nominal,
                        "coverage": (
                            float(
                                np.mean(
                                    (joined["truth"] >= joined["low"])
                                    & (joined["truth"] <= joined["high"])
                                )
                            )
                            if len(joined)
                            else np.nan
                        ),
                        "mean_pit_rank": (
                            float(np.mean(rank_rows)) if rank_rows else np.nan
                        ),
                    }
                )
        elif summary is not None and "object_id" in summary:
            summary_by_id = summary.set_index("object_id")
            low_col = f"{name}_q16"
            high_col = f"{name}_q84"
            if low_col not in summary_by_id or high_col not in summary_by_id:
                continue
            joined = pd.concat(
                [
                    y_true.rename("truth"),
                    summary_by_id[low_col].rename("low"),
                    summary_by_id[high_col].rename("high"),
                ],
                axis=1,
            ).dropna()
            rows.append(
                {
                    "parameter": name,
                    "interval": "68",
                    "nominal": 0.68,
                    "coverage": (
                        float(
                            np.mean(
                                (joined["truth"] >= joined["low"])
                                & (joined["truth"] <= joined["high"])
                            )
                        )
                        if len(joined)
                        else np.nan
                    ),
                    "mean_pit_rank": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _posterior_predictive_residual_payload(run: Path) -> dict[str, Any]:
    candidates = [
        run / "posterior_predictive_flux_residual_summary.parquet",
        run / "posterior_predictive_flux_residual_summary.csv",
        run / "batch_posterior_predictive_flux_residual_summary.parquet",
        run / "batch_posterior_predictive_flux_residual_summary.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        )
        numeric = frame.select_dtypes(include=[np.number])
        payload: dict[str, Any] = {"path": str(path), "rows": int(len(frame))}
        for column in numeric.columns:
            if "residual" in column or "chi" in column:
                values = numeric[column].to_numpy(float)
                values = values[np.isfinite(values)]
                if values.size:
                    payload[f"{column}_mean"] = float(np.mean(values))
                    payload[f"{column}_std"] = float(np.std(values))
        return payload
    return {"path": None, "message": "posterior predictive residual table not found"}


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Closure Inference Evaluation",
        "",
        f"- `run_dir`: {payload['run_dir']}",
        f"- `dataset_path`: {payload['dataset_path']}",
        f"- `n_truth_rows`: {payload['n_truth_rows']}",
        f"- `n_parameters`: {payload['n_parameters']}",
        "",
        "Parameter and coverage metrics are written next to this report as CSV.",
        "Use coverage, PIT/rank behavior and posterior predictive residuals as "
        "the primary closure criteria.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
