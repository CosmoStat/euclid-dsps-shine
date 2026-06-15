"""Redshift posterior metrics and Diffsky ablation reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir


def redshift_metrics_from_samples(
    posterior_samples: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    z_parameter: str = "z_obs",
    truth_column: str = "redshift_true",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute per-object and aggregate redshift posterior metrics."""
    required = {"object_id", z_parameter}
    if not required <= set(posterior_samples.columns):
        missing = ", ".join(sorted(required - set(posterior_samples.columns)))
        raise ValueError(f"posterior_samples missing required columns: {missing}")
    if "object_id" not in truth or truth_column not in truth:
        raise ValueError(f"truth must contain object_id and {truth_column}")
    rows = []
    truth_lookup = truth.set_index("object_id")[truth_column]
    for object_id, group in posterior_samples.groupby("object_id", sort=False):
        if object_id not in truth_lookup:
            continue
        z_true = float(truth_lookup.loc[object_id])
        samples = pd.to_numeric(group[z_parameter], errors="coerce").to_numpy(dtype=float)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0 or not np.isfinite(z_true):
            continue
        q16, q50, q84 = np.quantile(samples, [0.16, 0.5, 0.84])
        q025, q975 = np.quantile(samples, [0.025, 0.975])
        delta = (q50 - z_true) / (1.0 + z_true)
        rows.append(
            {
                "object_id": object_id,
                "z_true": z_true,
                "z_pred_median": float(q50),
                "z_q16": float(q16),
                "z_q84": float(q84),
                "z_q025": float(q025),
                "z_q975": float(q975),
                "delta_z": float(delta),
                "pit": float(np.mean(samples <= z_true)),
                "covered_68": bool(q16 <= z_true <= q84),
                "covered_95": bool(q025 <= z_true <= q975),
                "posterior_width_68": float(q84 - q16),
                "posterior_width_95": float(q975 - q025),
                "n_samples": int(samples.size),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, summarize_redshift_metrics(frame)


def summarize_redshift_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "n_objects": 0,
            "median_bias": float("nan"),
            "sigma_mad": float("nan"),
            "rmse": float("nan"),
            "outlier_fraction_0p15": float("nan"),
            "pit_mean": float("nan"),
            "coverage_68": float("nan"),
            "coverage_95": float("nan"),
            "posterior_width_68_median": float("nan"),
        }
    delta = frame["delta_z"].to_numpy(dtype=float)
    finite = np.isfinite(delta)
    delta = delta[finite]
    if delta.size == 0:
        return summarize_redshift_metrics(pd.DataFrame())
    med = float(np.median(delta))
    return {
        "n_objects": int(delta.size),
        "median_bias": med,
        "sigma_mad": float(1.4826 * np.median(np.abs(delta - med))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "outlier_fraction_0p15": float(np.mean(np.abs(delta) > 0.15)),
        "pit_mean": float(np.nanmean(frame["pit"])),
        "coverage_68": float(np.nanmean(frame["covered_68"].astype(float))),
        "coverage_95": float(np.nanmean(frame["covered_95"].astype(float))),
        "posterior_width_68_median": float(
            np.nanmedian(frame["posterior_width_68"])
        ),
    }


def posterior_truth_metrics(
    posterior_samples: pd.DataFrame,
    truth: pd.DataFrame,
    parameter_pairs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Summarize posterior median recovery for configured parameter/truth pairs."""
    residuals = _posterior_truth_residuals(posterior_samples, truth, parameter_pairs)
    return _summarize_posterior_truth_residuals(residuals)


def _posterior_truth_residuals(
    posterior_samples: pd.DataFrame,
    truth: pd.DataFrame,
    parameter_pairs: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    rows = []
    for parameter, truth_column in parameter_pairs:
        if parameter not in posterior_samples or truth_column not in truth:
            continue
        med = (
            posterior_samples.groupby("object_id", sort=False)[parameter]
            .median()
            .rename("posterior_median")
            .reset_index()
        )
        joined = med.merge(
            truth[["object_id", truth_column]],
            on="object_id",
            how="inner",
        )
        if joined.empty:
            continue
        y_pred = joined["posterior_median"].to_numpy(dtype=float)
        y_true = joined[truth_column].to_numpy(dtype=float)
        finite = np.isfinite(y_pred) & np.isfinite(y_true)
        if not finite.any():
            continue
        residual = y_pred[finite] - y_true[finite]
        for object_id, value in zip(
            joined.loc[finite, "object_id"].to_numpy(),
            residual,
            strict=True,
        ):
            rows.append(
                {
                    "object_id": object_id,
                    "parameter": parameter,
                    "truth_column": truth_column,
                    "metric_name": _posterior_metric_name(parameter),
                    "residual": float(value),
                }
            )
    return pd.DataFrame(rows)


def _summarize_posterior_truth_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if residuals.empty:
        return pd.DataFrame(rows)
    for (parameter, truth_column, metric_name), group in residuals.groupby(
        ["parameter", "truth_column", "metric_name"],
        sort=False,
    ):
        values = group["residual"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        med = float(np.median(values))
        rows.append(
            {
                "parameter": parameter,
                "truth_column": truth_column,
                "metric_name": metric_name,
                "n_objects": int(values.size),
                "bias": float(np.mean(values)),
                "median_bias": med,
                "rmse": float(np.sqrt(np.mean(values**2))),
                "sigma_mad": float(1.4826 * np.median(np.abs(values - med))),
            }
        )
    return pd.DataFrame(rows)


def _posterior_metric_name(parameter: str) -> str:
    if parameter == "log10_stellar_mass_alpha_corrected":
        return "mass_bias_alpha_corrected"
    if parameter == "log10_stellar_mass":
        return "mass_bias_raw"
    return f"{parameter}_bias"


def write_redshift_metrics_for_run(
    *,
    dataset_path: str | Path,
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    label: str = "run",
) -> dict[str, Path]:
    """Compute redshift and posterior-vs-truth metrics for one inference run."""
    run = Path(run_dir)
    out = ensure_dir(out_dir or run)
    truth_cols = ["object_id", "redshift_true", "logsm_true", "logsfr_true", "logssfr_true"]
    truth = _read_existing_columns(dataset_path, truth_cols)
    object_frames = []
    residual_frames = []
    pairs = (
        ("z_obs", "redshift_true"),
        ("log10_stellar_mass", "logsm_true"),
        ("log10_stellar_mass_alpha_corrected", "logsm_true"),
        ("log10_sfr_at_obs", "logsfr_true"),
        ("log10_ssfr_at_obs", "logssfr_true"),
    )
    found_samples = False
    for samples in _iter_posterior_sample_tables(run):
        found_samples = True
        object_frame, _summary = redshift_metrics_from_samples(samples, truth)
        if not object_frame.empty:
            object_frames.append(object_frame)
        residual_frame = _posterior_truth_residuals(samples, truth, pairs)
        if not residual_frame.empty:
            residual_frames.append(residual_frame)
    if not found_samples:
        raise FileNotFoundError(f"Missing posterior sample outputs under {run}")
    object_metrics = (
        pd.concat(object_frames, ignore_index=True)
        if object_frames
        else pd.DataFrame()
    )
    summary = summarize_redshift_metrics(object_metrics)
    object_path = out / "photoz_object_metrics.csv"
    object_metrics.to_csv(object_path, index=False)
    summary_frame = pd.DataFrame(
        [
            {
                **summary,
                "label": label,
                **_read_alpha_summary(run),
                **_read_likelihood_summary(run),
            }
        ]
    )
    photoz_path = out / "photoz_metrics.csv"
    summary_frame.to_csv(photoz_path, index=False)
    residuals = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames
        else pd.DataFrame()
    )
    posterior_metrics = _summarize_posterior_truth_residuals(residuals)
    posterior_path = out / "posterior_vs_truth_metrics.csv"
    posterior_metrics.to_csv(posterior_path, index=False)
    return {
        "photoz_object_metrics": object_path,
        "photoz_metrics": photoz_path,
        "posterior_vs_truth_metrics": posterior_path,
    }


def run_redshift_ablation(
    *,
    dataset_path: str | Path,
    runs: list[tuple[str, Path]],
    out_dir: str | Path,
) -> Path:
    """Compare redshift metrics across multiple inference/closure runs."""
    out = ensure_dir(out_dir)
    rows = []
    object_frames = []
    truth = _read_existing_columns(dataset_path, ["object_id", "redshift_true"])
    for label, run_dir in runs:
        frames = []
        found_samples = False
        for samples in _iter_posterior_sample_tables(run_dir):
            found_samples = True
            object_metrics, _summary = redshift_metrics_from_samples(samples, truth)
            if not object_metrics.empty:
                frames.append(object_metrics)
        if not found_samples:
            continue
        object_metrics = (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )
        summary = summarize_redshift_metrics(object_metrics)
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                **summary,
                **_read_alpha_summary(run_dir),
                **_read_likelihood_summary(run_dir),
            }
        )
        if not object_metrics.empty:
            object_metrics.insert(0, "label", label)
            object_frames.append(object_metrics)
    summary = pd.DataFrame(rows)
    summary_path = out / "redshift_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    objects = pd.concat(object_frames, ignore_index=True) if object_frames else pd.DataFrame()
    if not objects.empty:
        objects.to_csv(out / "redshift_ablation_object_metrics.csv", index=False)
        _write_redshift_plots(objects, out)
    report = out / "redshift_ablation_report.md"
    _write_ablation_report(report, summary)
    return report


def parse_run_specs(specs: list[str]) -> list[tuple[str, Path]]:
    runs = []
    for spec in specs:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).name
        runs.append((label, Path(path)))
    return runs


def _read_existing_columns(path: str | Path, columns: list[str]) -> pd.DataFrame:
    available = pd.read_parquet(path, columns=None).columns
    selected = [column for column in columns if column in available]
    return pd.read_parquet(path, columns=selected)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _iter_posterior_sample_tables(run: Path):
    monolithic = run / "posterior_samples.parquet"
    if monolithic.exists():
        yield _read_table(monolithic)
        return
    manifest_paths = _posterior_sample_paths_from_manifest(run)
    if manifest_paths:
        for path in manifest_paths:
            if path.exists():
                yield pd.read_parquet(path)
        return
    shard_dir = run / "posterior_samples"
    if not shard_dir.exists():
        return
    for path in sorted(shard_dir.glob("*.parquet")):
        yield pd.read_parquet(path)


def _posterior_sample_paths_from_manifest(run: Path) -> list[Path]:
    manifest = run / "posterior_shards_manifest.json"
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = []
    for key in ("shards_written", "shards_skipped"):
        value = payload.get(key, [])
        if isinstance(value, list):
            records.extend(item for item in value if isinstance(item, dict))
    paths = []
    for record in sorted(records, key=lambda item: int(item.get("batch", 0))):
        raw = record.get("samples_path")
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_absolute() or path.exists():
            paths.append(path)
        else:
            paths.append(run / "posterior_samples" / path.name)
    return paths


def _read_alpha_summary(run_dir: Path) -> dict[str, float | bool | str]:
    path = run_dir / "inference_summary.json"
    if not path.exists():
        path = run_dir / "forward_closure_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    alpha = payload.get("global_sed_scale", {})
    if not isinstance(alpha, dict):
        return {}
    keep = (
        "alpha_sed",
        "log_alpha_sed",
        "delta_mag_global",
        "alpha_prior_penalty",
        "large_scale_warning",
    )
    return {key: alpha[key] for key in keep if key in alpha}


def _read_likelihood_summary(run_dir: Path) -> dict[str, float | str]:
    path = run_dir / "normalized_config.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    amortized = payload.get("amortized", {})
    if isinstance(amortized, dict):
        likelihood = amortized.get("likelihood", {})
        if isinstance(likelihood, dict) and likelihood:
            return {
                "likelihood_type": str(likelihood.get("type", "student_t")),
                "student_t_dof": float(likelihood.get("student_t_dof", 2.0)),
                "error_floor_frac": float(likelihood.get("error_floor_frac", 0.02)),
                "error_jitter": float(likelihood.get("error_jitter", 0.0)),
            }
    fit = payload.get("fit", {})
    if isinstance(fit, dict):
        return {
            "likelihood_type": str(fit.get("photometric_likelihood", "student_t")),
            "student_t_dof": float(fit.get("student_t_dof", 2.0)),
            "error_floor_frac": float(fit.get("flux_error_floor_frac", 0.0)),
            "error_jitter": float(fit.get("flux_error_jitter", 0.0)),
        }
    return {}


def _write_redshift_plots(frame: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, group in frame.groupby("label", sort=False):
        ax.scatter(group["z_true"], group["z_pred_median"], s=8, alpha=0.5, label=label)
    lo = float(np.nanmin(frame[["z_true", "z_pred_median"]].to_numpy()))
    hi = float(np.nanmax(frame[["z_true", "z_pred_median"]].to_numpy()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("z_true")
    ax.set_ylabel("z_posterior_median")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "z_pred_vs_z_true.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, group in frame.groupby("label", sort=False):
        ax.hist(group["pit"], bins=20, histtype="step", density=True, label=label)
    ax.set_xlabel("PIT")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pit_histogram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, group in frame.groupby("label", sort=False):
        ax.hist(group["delta_z"], bins=40, histtype="step", density=True, label=label)
    ax.set_xlabel("delta_z")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "delta_z_histogram.png", dpi=160)
    plt.close(fig)


def _write_ablation_report(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Diffsky Redshift Ablation",
        "",
        "This report compares posterior redshift accuracy and calibration.",
        "",
        "## Summary",
        "",
        _markdown_table(summary) if not summary.empty else "_No runs scored._",
        "",
        "Median accuracy alone is insufficient; use PIT, coverage, and posterior "
        "widths to evaluate calibration.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}" if np.isfinite(value) else "nan")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
