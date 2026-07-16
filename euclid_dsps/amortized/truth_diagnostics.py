"""Extended truth/proxy diagnostics for Diffsky amortized runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.io import write_json


@dataclass(frozen=True)
class TruthPair:
    name: str
    pred_column: str
    truth_column: str
    kind: str = "direct"


POSTERIOR_TRUTH_PAIRS = (
    TruthPair("z_obs", "z_obs_median", "redshift_true"),
    TruthPair("log10_stellar_mass", "log10_stellar_mass_median", "logsm_true"),
    TruthPair(
        "log10_stellar_mass_alpha_corrected",
        "log10_stellar_mass_alpha_corrected",
        "logsm_true",
    ),
    TruthPair("log10_sfr_at_obs", "log10_sfr_at_obs_median", "logsfr_true"),
    TruthPair(
        "log10_sfr_at_obs_alpha_corrected",
        "log10_sfr_at_obs_alpha_corrected",
        "logsfr_true",
    ),
    TruthPair("log10_ssfr_at_obs", "log10_ssfr_at_obs_median", "logssfr_true"),
    TruthPair("tau2_proxy", "tau2_median", "tau2_truth_proxy", "proxy"),
    TruthPair(
        "dust_index_n_proxy",
        "dust_index_n_median",
        "dust_index_truth_proxy",
        "proxy",
    ),
)

MAP_TRUTH_PAIRS = (
    TruthPair("z_obs", "z_obs", "redshift_true"),
    TruthPair("log10_stellar_mass", "log10_stellar_mass", "logsm_true"),
    TruthPair(
        "log10_sfr_at_obs_alpha_corrected",
        "log10_sfr_at_obs_alpha_corrected",
        "logsfr_true",
    ),
    TruthPair("log10_ssfr_at_obs", "log10_ssfr_at_obs", "logssfr_true"),
    TruthPair("tau2_proxy", "tau2", "tau2_truth_proxy", "proxy"),
    TruthPair("dust_index_n_proxy", "dust_index_n", "dust_index_truth_proxy", "proxy"),
)

PRIOR_TRUTH_PAIRS = (
    TruthPair("z_obs", "z_obs", "redshift_true"),
    TruthPair("log10_stellar_mass", "log10_stellar_mass", "logsm_true"),
    TruthPair("tau2_proxy", "tau2", "tau2_truth_proxy", "proxy"),
    TruthPair("dust_index_n_proxy", "dust_index_n", "dust_index_truth_proxy", "proxy"),
)


def write_extended_truth_diagnostics(
    run_dir: str | Path,
    *,
    truth_path: str | Path | None = None,
    posterior_summary_path: str | Path | None = None,
    map_estimates_path: str | Path | None = None,
    prior_samples_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write extended truth/proxy tables and plots for available artifacts."""
    run = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run
    truth_file = (
        Path(truth_path) if truth_path is not None else run / "inference_truth.parquet"
    )
    if not truth_file.exists():
        return {}
    truth = _with_truth_proxies(pd.read_parquet(truth_file))
    outputs: dict[str, str] = {}

    posterior_file = (
        Path(posterior_summary_path)
        if posterior_summary_path is not None
        else run / "posterior_summary.parquet"
    )
    posterior = _read_frame(posterior_file)
    if not posterior.empty:
        outputs |= _write_prediction_truth_block(
            posterior,
            truth,
            POSTERIOR_TRUTH_PAIRS,
            out,
            prefix="posterior",
        )

    map_file = (
        Path(map_estimates_path)
        if map_estimates_path is not None
        else run / "map_estimates.parquet"
    )
    map_estimates = _read_frame(map_file)
    if not map_estimates.empty:
        outputs |= _write_prediction_truth_block(
            map_estimates,
            truth,
            MAP_TRUTH_PAIRS,
            out,
            prefix="map",
        )

    prior_file = _resolve_prior_path(run, prior_samples_path)
    prior = _read_frame(prior_file)
    if not prior.empty:
        outputs |= _write_prior_truth_block(prior, truth, out)

    plot_paths = _write_population_overlay_plots(
        truth=truth,
        posterior=posterior,
        map_estimates=map_estimates,
        prior=prior,
        out=out,
    )
    outputs.update({path.name: str(path) for path in plot_paths})
    _write_truth_diagnostics_summary(out, outputs)
    return outputs


def _with_truth_proxies(truth: pd.DataFrame) -> pd.DataFrame:
    work = truth.copy()
    if "dust_av" in work:
        work["tau2_truth_proxy"] = (
            pd.to_numeric(work["dust_av"], errors="coerce") / 1.086
        )
    if "dust_delta" in work:
        work["dust_index_truth_proxy"] = pd.to_numeric(
            work["dust_delta"],
            errors="coerce",
        )
    if "logsfr_true" in work and "logsm_true" in work and "logssfr_true" not in work:
        work["logssfr_true"] = pd.to_numeric(
            work["logsfr_true"],
            errors="coerce",
        ) - pd.to_numeric(work["logsm_true"], errors="coerce")
    return work


def _read_frame(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _resolve_prior_path(run: Path, explicit: str | Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    for name in (
        "learned_prior_samples.parquet",
        "learned_or_loaded_prior_samples.parquet",
    ):
        path = run / name
        if path.exists():
            return path
    return None


def _write_prediction_truth_block(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    pairs: tuple[TruthPair, ...],
    out: Path,
    *,
    prefix: str,
) -> dict[str, str]:
    identity = _identity_column(prediction, truth)
    if identity is None:
        return {}
    rows = []
    residual_rows = []
    truth_columns = [identity]
    if "object_id" in truth and "object_id" not in truth_columns:
        truth_columns.append("object_id")
    truth_columns.extend(
        sorted({pair.truth_column for pair in pairs if pair.truth_column in truth})
    )
    merged = prediction.merge(
        truth[truth_columns],
        on=identity,
        how="inner",
        suffixes=("", "_truth"),
    )
    if merged.empty:
        return {}
    for pair in pairs:
        if pair.pred_column not in merged or pair.truth_column not in merged:
            continue
        pred = pd.to_numeric(merged[pair.pred_column], errors="coerce").to_numpy(float)
        ref = pd.to_numeric(merged[pair.truth_column], errors="coerce").to_numpy(float)
        finite = np.isfinite(pred) & np.isfinite(ref)
        if not finite.any():
            continue
        residual = pred[finite] - ref[finite]
        rows.append(
            _metric_row(pair, residual, int(finite.sum()), pred[finite], ref[finite])
        )
        residual_frame = merged.loc[finite, [identity]].copy()
        if "object_id" in merged:
            residual_frame["object_id"] = merged.loc[finite, "object_id"].to_numpy()
        residual_frame["parameter"] = pair.name
        residual_frame["prediction_column"] = pair.pred_column
        residual_frame["truth_column"] = pair.truth_column
        residual_frame["truth_kind"] = pair.kind
        residual_frame["predicted"] = pred[finite]
        residual_frame["truth"] = ref[finite]
        residual_frame["residual"] = residual
        residual_rows.append(residual_frame)
    outputs: dict[str, str] = {}
    if rows:
        metrics = pd.DataFrame(rows)
        metrics_path = out / f"{prefix}_vs_truth_extended.csv"
        metrics.to_csv(metrics_path, index=False)
        outputs[metrics_path.name] = str(metrics_path)
    if residual_rows:
        residuals = pd.concat(residual_rows, ignore_index=True)
        residual_path = out / f"{prefix}_vs_truth_extended_objects.parquet"
        residuals.to_parquet(residual_path, index=False)
        outputs[residual_path.name] = str(residual_path)
        for path in _write_prediction_truth_plots(residuals, out, prefix=prefix):
            outputs[path.name] = str(path)
    return outputs


def _metric_row(
    pair: TruthPair,
    residual: np.ndarray,
    n_objects: int,
    predicted: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    med = float(np.median(residual))
    corr = (
        float(np.corrcoef(predicted, truth)[0, 1])
        if predicted.size > 1 and np.std(predicted) > 0.0 and np.std(truth) > 0.0
        else float("nan")
    )
    return {
        "parameter": pair.name,
        "prediction_column": pair.pred_column,
        "truth_column": pair.truth_column,
        "truth_kind": pair.kind,
        "n_objects": int(n_objects),
        "bias": float(np.mean(residual)),
        "median_bias": med,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "sigma_mad": float(1.4826 * np.median(np.abs(residual - med))),
        "corr": corr,
    }


def _write_prior_truth_block(
    prior: pd.DataFrame,
    truth: pd.DataFrame,
    out: Path,
) -> dict[str, str]:
    rows = []
    for pair in PRIOR_TRUTH_PAIRS:
        if pair.pred_column not in prior or pair.truth_column not in truth:
            continue
        pred = pd.to_numeric(prior[pair.pred_column], errors="coerce").to_numpy(float)
        ref = pd.to_numeric(truth[pair.truth_column], errors="coerce").to_numpy(float)
        pred = pred[np.isfinite(pred)]
        ref = ref[np.isfinite(ref)]
        if pred.size == 0 or ref.size == 0:
            continue
        rows.append(
            {
                "parameter": pair.name,
                "prior_column": pair.pred_column,
                "truth_column": pair.truth_column,
                "truth_kind": pair.kind,
                "n_prior": int(pred.size),
                "n_truth": int(ref.size),
                "prior_median": float(np.quantile(pred, 0.5)),
                "truth_median": float(np.quantile(ref, 0.5)),
                "median_delta": float(np.quantile(pred, 0.5) - np.quantile(ref, 0.5)),
                "prior_q16": float(np.quantile(pred, 0.16)),
                "truth_q16": float(np.quantile(ref, 0.16)),
                "prior_q84": float(np.quantile(pred, 0.84)),
                "truth_q84": float(np.quantile(ref, 0.84)),
                "quantile_l1": _quantile_l1(pred, ref),
            }
        )
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    path = out / "prior_vs_truth_population.csv"
    frame.to_csv(path, index=False)
    return {path.name: str(path)}


def _quantile_l1(a: np.ndarray, b: np.ndarray) -> float:
    qs = np.linspace(0.01, 0.99, 99)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _identity_column(prediction: pd.DataFrame, truth: pd.DataFrame) -> str | None:
    if "row_index" in prediction and "row_index" in truth:
        return "row_index"
    if "object_id" in prediction and "object_id" in truth:
        return "object_id"
    return None


def _write_prediction_truth_plots(
    residuals: pd.DataFrame,
    out: Path,
    *,
    prefix: str,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths: list[Path] = []
    for parameter, group in residuals.groupby("parameter", sort=False):
        truth = group["truth"].to_numpy(dtype=float)
        pred = group["predicted"].to_numpy(dtype=float)
        residual = group["residual"].to_numpy(dtype=float)
        finite = np.isfinite(truth) & np.isfinite(pred)
        if not finite.any():
            continue
        truth = truth[finite]
        pred = pred[finite]
        residual = residual[finite]
        safe = _safe_name(str(parameter))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(truth, pred, s=10, alpha=0.55)
        lo = float(np.nanmin([truth.min(), pred.min()]))
        hi = float(np.nanmax([truth.max(), pred.max()]))
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, alpha=0.6)
        ax.set_xlabel("truth/proxy")
        ax.set_ylabel(prefix)
        ax.set_title(f"{prefix} vs truth: {parameter}")
        fig.tight_layout()
        path = out / f"truth_vs_{prefix}_{safe}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        ax.scatter(truth, residual, s=10, alpha=0.55)
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.6)
        ax.set_xlabel("truth/proxy")
        ax.set_ylabel(f"{prefix} - truth")
        ax.set_title(f"{prefix} residual: {parameter}")
        fig.tight_layout()
        path = out / f"{prefix}_bias_vs_truth_{safe}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_population_overlay_plots(
    *,
    truth: pd.DataFrame,
    posterior: pd.DataFrame,
    map_estimates: pd.DataFrame,
    prior: pd.DataFrame,
    out: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    paths: list[Path] = []
    overlay_specs = (
        ("z_obs", "redshift_true", "z_obs_median", "z_obs"),
        (
            "log10_stellar_mass",
            "logsm_true",
            "log10_stellar_mass_median",
            "log10_stellar_mass",
        ),
        (
            "log10_sfr_at_obs",
            "logsfr_true",
            "log10_sfr_at_obs_alpha_corrected",
            "log10_sfr_at_obs_alpha_corrected",
        ),
        (
            "log10_ssfr_at_obs",
            "logssfr_true",
            "log10_ssfr_at_obs_median",
            "log10_ssfr_at_obs",
        ),
        ("tau2_proxy", "tau2_truth_proxy", "tau2_median", "tau2"),
        (
            "dust_index_n_proxy",
            "dust_index_truth_proxy",
            "dust_index_n_median",
            "dust_index_n",
        ),
    )
    population_columns: dict[str, dict[str, np.ndarray]] = {}
    for name, truth_col, posterior_col, value_col in overlay_specs:
        series: dict[str, np.ndarray] = {}
        if truth_col in truth:
            series["truth"] = _finite_values(truth[truth_col])
        if not posterior.empty and posterior_col in posterior:
            series["posterior"] = _finite_values(posterior[posterior_col])
        if not map_estimates.empty and value_col in map_estimates:
            series["map"] = _finite_values(map_estimates[value_col])
        if not prior.empty and value_col in prior:
            series["prior"] = _finite_values(prior[value_col])
        series = {key: value for key, value in series.items() if value.size}
        if len(series) < 2:
            continue
        population_columns[name] = series
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for label, values in series.items():
            ax.hist(values, bins=40, density=True, histtype="step", lw=1.5, label=label)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.set_title(f"Population overlay: {name}")
        ax.legend()
        fig.tight_layout()
        path = out / f"population_overlay_{_safe_name(name)}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    return values[np.isfinite(values)]


def _safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "p")
    )


def _write_truth_diagnostics_summary(out: Path, outputs: dict[str, str]) -> None:
    write_json(
        out / "extended_truth_diagnostics_summary.json",
        {
            "n_outputs": int(len(outputs)),
            "outputs": outputs,
        },
    )
