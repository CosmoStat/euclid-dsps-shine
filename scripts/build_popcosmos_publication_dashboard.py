#!/usr/bin/env python3
"""Build the complete FENIKS and Pop-COSMOS publication dashboard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

POPCOSMOS_PAPER = {
    "method": "popcosmos_paper",
    "display_name": "Pop-COSMOS\npaper cohort",
    "cohort": "popcosmos_paper",
    "n_objects": 12_014,
    "median_bias": 1.0e-4,
    "nmad": 7.0e-3,
    "rmse": None,
    "outlier_fraction_0p15": 0.016,
    "coverage_68": None,
    "source": "https://doi.org/10.3847/1538-4357/ad7736",
}
EXPECTED_CALIBRATION_RUNS = {
    ("cosmos_public_specz", "rws26"),
    ("cosmos_public_specz", "rws24"),
    ("feniks_synthetic", "rws_k8_t2_seed2"),
    ("feniks_synthetic", "rws_k8_t2_seed3"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feniks-rws", type=Path, required=True)
    parser.add_argument("--feniks-rws-mixture", type=Path, required=True)
    parser.add_argument("--feniks-smcwake", type=Path, required=True)
    parser.add_argument("--matched-metrics", type=Path, required=True)
    parser.add_argument("--calibration-table", type=Path, required=True)
    parser.add_argument("--feniks-tarp", type=Path, required=True)
    parser.add_argument("--cosmos-tarp", type=Path, required=True)
    parser.add_argument("--timing-26", type=Path, required=True)
    parser.add_argument("--timing-24", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--popcosmos-gpu-seconds", type=float, default=15.0)
    return parser.parse_args()


def _resolve(path: Path, filename: str) -> Path:
    candidate = path / filename if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_feniks_metric(
    path: Path,
    *,
    method: str,
    display_name: str,
) -> dict[str, object]:
    source = _resolve(path, "photoz_metrics.csv")
    frame = pd.read_csv(source)
    if len(frame) != 1:
        raise ValueError(f"Expected one FENIKS metric row in {source}, found {len(frame)}")
    row = frame.iloc[0]
    return {
        "method": method,
        "display_name": display_name,
        "cohort": "feniks_synthetic",
        "n_objects": int(row["n_objects"]),
        "median_bias": float(row["median_bias"]),
        "nmad": float(row["sigma_mad"]),
        "rmse": float(row["rmse"]),
        "outlier_fraction_0p15": float(row["outlier_fraction_0p15"]),
        "coverage_68": float(row["coverage_68"]),
        "source": str(source),
    }


def _read_accuracy_metrics(
    *,
    feniks_rws: Path,
    feniks_rws_mixture: Path,
    feniks_smcwake: Path,
    matched_metrics: Path,
) -> pd.DataFrame:
    rows = [
        _read_feniks_metric(
            feniks_rws,
            method="feniks_rws_k8",
            display_name="FENIKS RWS k8",
        ),
        _read_feniks_metric(
            feniks_rws_mixture,
            method="feniks_rws_mix_k8",
            display_name="FENIKS RWS mix k8",
        ),
        _read_feniks_metric(
            feniks_smcwake,
            method="feniks_smcwake_k4",
            display_name="FENIKS SMC-wake k4",
        ),
    ]
    matched_file = _resolve(matched_metrics, "redshift_method_metrics.csv")
    matched = pd.read_csv(matched_file)
    expected = {"rws26", "rws24", "popcosmos"}
    if set(matched["method"]) != expected or matched["method"].duplicated().any():
        raise ValueError("Matched COSMOS table must contain exactly rws26/rws24/popcosmos")
    labels = {
        "rws26": "COSMOS RWS 26 bands",
        "rws24": "COSMOS RWS 24 bands",
        "popcosmos": "Pop-COSMOS matched",
    }
    for row in matched.itertuples(index=False):
        values: dict[str, object] = {
            "method": row.method,
            "display_name": labels[row.method],
            "cohort": "cosmos_matched_specz",
            "n_objects": int(row.n_spec),
            "median_bias": float(row.median_bias),
            "nmad": float(row.nmad),
            "rmse": float(row.rmse),
            "outlier_fraction_0p15": float(row.outlier_fraction_0p15),
            "coverage_68": float(row.coverage_68),
            "source": str(matched_file),
        }
        for metric in (
            "median_bias",
            "nmad",
            "rmse",
            "outlier_fraction_0p15",
            "coverage_68",
        ):
            for suffix in ("ci95_low", "ci95_high"):
                column = f"{metric}_{suffix}"
                values[column] = (
                    float(getattr(row, column)) if column in matched.columns else np.nan
                )
        rows.append(values)
    rows.append(dict(POPCOSMOS_PAPER))
    result = pd.DataFrame(rows)
    for metric in (
        "median_bias",
        "nmad",
        "rmse",
        "outlier_fraction_0p15",
        "coverage_68",
    ):
        for suffix in ("ci95_low", "ci95_high"):
            column = f"{metric}_{suffix}"
            if column not in result:
                result[column] = np.nan
    return result


def _read_calibration(path: Path) -> pd.DataFrame:
    source = _resolve(path, "redshift_calibration_comparison.csv")
    frame = pd.read_csv(source)
    frame = frame.loc[
        frame["mira_score"].notna() & frame["tarp_atc"].notna()
    ].copy()
    observed = set(frame[["context", "model"]].itertuples(index=False, name=None))
    if observed != EXPECTED_CALIBRATION_RUNS:
        raise ValueError(
            f"Calibration run mismatch: expected={sorted(EXPECTED_CALIBRATION_RUNS)}, "
            f"observed={sorted(observed)}"
        )
    return frame


def _read_tarp_coverage(path: Path, *, context: str) -> pd.DataFrame:
    source = _resolve(path, "tarp_coverage.csv")
    frame = pd.read_csv(source)
    frame = frame.loc[frame["group"].eq("marginal_z_obs")].copy()
    frame.insert(0, "context", context)
    return frame


def _read_timing(path: Path) -> dict[str, object]:
    source = _resolve(path, "timing_summary.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("backend") != "gpu":
        raise ValueError(f"Timing is not a completed GPU result: {source}")
    return payload


def _speed_summary(
    timing_26: dict[str, object],
    timing_24: dict[str, object],
    *,
    popcosmos_gpu_seconds: float,
) -> dict[str, object]:
    variants = {}
    for name, timing in (("rws26", timing_26), ("rws24", timing_24)):
        steady = timing["steady_state"]
        encoder = float(steady["encoder_only"]["median_seconds_per_object"])
        draws = float(steady["posterior_draws"]["median_seconds_per_object"])
        predictive = float(
            steady["posterior_predictive"]["median_seconds_per_object"]
        )
        features = float(timing["feature_construction_seconds"]) / int(
            timing["n_objects"]
        )
        variants[name] = {
            "encoder_seconds_per_object": encoder,
            "posterior_draws_seconds_per_object": draws,
            "amortized_posterior_seconds_per_object": encoder + draws,
            "posterior_predictive_seconds_per_object": predictive,
            "feature_construction_seconds_per_object": features,
            "catalog_pipeline_seconds_per_object": encoder
            + draws
            + predictive
            + features,
            "popcosmos_throughput_ratio": popcosmos_gpu_seconds
            / (encoder + draws + predictive + features),
        }
    return {
        "popcosmos_gpu_seconds_per_object": float(popcosmos_gpu_seconds),
        "popcosmos_source": POPCOSMOS_PAPER["source"],
        "comparison_kind": "descriptive_gpu_throughput_ratio",
        "variants": variants,
    }


def _styles() -> dict[str, dict[str, object]]:
    return {
        "feniks_rws_k8": {"color": "#D55E00", "marker": "D"},
        "feniks_rws_mix_k8": {"color": "#CC79A7", "marker": "^"},
        "feniks_smcwake_k4": {"color": "#8C6D31", "marker": "v"},
        "rws26": {"color": "#0072B2", "marker": "o"},
        "rws24": {"color": "#009E73", "marker": "s"},
        "popcosmos": {"color": "#303030", "marker": "P"},
        "popcosmos_paper": {"color": "#E69F00", "marker": "*"},
        "rws_k8_t2_seed2": {"color": "#D55E00", "marker": "D", "ls": "--"},
        "rws_k8_t2_seed3": {"color": "#CC79A7", "marker": "^", "ls": "--"},
    }


def _plot_accuracy_metric(
    axis,
    frame: pd.DataFrame,
    *,
    metric: str,
    title: str,
    scale: float = 1.0,
    show_labels: bool = False,
    value_format: str = ".3f",
) -> None:
    from matplotlib.transforms import blended_transform_factory

    styles = _styles()
    y = np.arange(len(frame))[::-1]
    values = frame[metric].to_numpy(float) * scale
    missing_y = []
    for index, row in enumerate(frame.itertuples(index=False)):
        value = values[index]
        if not np.isfinite(value):
            missing_y.append(y[index])
            continue
        style = styles[row.method]
        low = getattr(row, f"{metric}_ci95_low") * scale
        high = getattr(row, f"{metric}_ci95_high") * scale
        xerr = None
        if np.isfinite(low) and np.isfinite(high):
            xerr = np.asarray([[value - low], [high - value]])
        axis.errorbar(
            value,
            y[index],
            xerr=xerr,
            fmt=style["marker"],
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7,
            capsize=2.5,
        )
        axis.annotate(
            format(value, value_format),
            (value, y[index]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=style["color"],
        )
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    for missing in missing_y:
        axis.text(
            0.02,
            missing,
            "n/a",
            transform=transform,
            va="center",
            fontsize=7.5,
            color="#777777",
        )
    axis.set_title(title, fontsize=11)
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.axhline(y[2] - 0.5, color="#D3D3D3", linewidth=0.9)
    axis.axhline(y[5] - 0.5, color="#D3D3D3", linewidth=0.9)
    axis.set_yticks(y)
    axis.set_yticklabels(frame["display_name"] if show_labels else [])


def _write_dashboard(
    *,
    accuracy: pd.DataFrame,
    calibration: pd.DataFrame,
    tarp_coverage: pd.DataFrame,
    speed: dict[str, object],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = _styles()
    figure = plt.figure(figsize=(24, 12.8))
    grid = figure.add_gridspec(
        2,
        5,
        height_ratios=(1.05, 1.0),
        left=0.08,
        right=0.985,
        top=0.90,
        bottom=0.10,
        hspace=0.40,
        wspace=0.34,
    )
    accuracy_specs = (
        ("median_bias", "Absolute median bias", 1.0, ".3f"),
        ("nmad", "NMAD", 1.0, ".3f"),
        ("rmse", "RMSE", 1.0, ".3f"),
        ("outlier_fraction_0p15", "Outliers |dz| > 0.15 (%)", 100.0, ".1f"),
        ("coverage_68", "68% coverage (%)", 100.0, ".1f"),
    )
    for column, (metric, title, scale, value_format) in enumerate(accuracy_specs):
        axis = figure.add_subplot(grid[0, column])
        plotted = accuracy.copy()
        if metric == "median_bias":
            plotted[metric] = plotted[metric].abs()
            plotted[f"{metric}_ci95_low"] = np.nan
            plotted[f"{metric}_ci95_high"] = np.nan
        _plot_accuracy_metric(
            axis,
            plotted,
            metric=metric,
            title=title,
            scale=scale,
            show_labels=column == 0,
            value_format=value_format,
        )

    calibration_order = ["rws26", "rws24", "rws_k8_t2_seed2", "rws_k8_t2_seed3"]
    calibration = calibration.set_index("model").loc[calibration_order].reset_index()
    labels = ["COSMOS 26", "COSMOS 24", "FENIKS seed 2", "FENIKS seed 3"]
    x = np.asarray([0.0, 1.0, 2.45, 3.45])

    mira_axis = figure.add_subplot(grid[1, 0])
    for index, row in enumerate(calibration.itertuples(index=False)):
        style = styles[row.model]
        mira_axis.errorbar(
            x[index],
            row.mira_score,
            yerr=np.asarray(
                [[row.mira_score - row.mira_bootstrap_q025],
                 [row.mira_bootstrap_q975 - row.mira_score]]
            ),
            fmt=style["marker"],
            color=style["color"],
            capsize=3,
        )
        mira_axis.annotate(
            f"{row.mira_score:.3f}",
            (x[index], row.mira_score),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=style["color"],
        )
    mira_axis.axhline(2.0 / 3.0, color="#222222", linestyle="--", linewidth=1)
    mira_axis.set_title("Redshift MIRA (ideal = 2/3)", fontsize=11)
    mira_axis.set_ylabel("MIRA score")
    mira_axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)

    tarp_axis = figure.add_subplot(grid[1, 1])
    for index, row in enumerate(calibration.itertuples(index=False)):
        style = styles[row.model]
        tarp_axis.errorbar(
            x[index],
            row.tarp_atc,
            yerr=np.asarray(
                [[row.tarp_atc - row.tarp_bootstrap_atc_q025],
                 [row.tarp_bootstrap_atc_q975 - row.tarp_atc]]
            ),
            fmt=style["marker"],
            color=style["color"],
            capsize=3,
        )
        tarp_axis.annotate(
            f"{row.tarp_atc:+.3f}",
            (x[index], row.tarp_atc),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=style["color"],
        )
    tarp_axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1)
    tarp_axis.set_title("Redshift TARP ATC (ideal = 0)", fontsize=11)
    tarp_axis.set_ylabel("area to curve")
    tarp_axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)

    curve_axis = figure.add_subplot(grid[1, 2:4])
    curve_axis.plot([0, 1], [0, 1], color="#222222", linestyle="--", label="Ideal")
    curve_labels = dict(zip(calibration_order, labels, strict=True))
    for model in calibration_order:
        context = (
            "cosmos_public_specz" if model in {"rws26", "rws24"}
            else "feniks_synthetic"
        )
        curve = tarp_coverage.loc[
            tarp_coverage["context"].eq(context) & tarp_coverage["model"].eq(model)
        ].sort_values("alpha")
        style = styles[model]
        curve_axis.fill_between(
            curve["alpha"],
            curve["bootstrap_q025"],
            curve["bootstrap_q975"],
            color=style["color"],
            alpha=0.10,
            linewidth=0,
        )
        curve_axis.plot(
            curve["alpha"],
            curve["ecp"],
            color=style["color"],
            linestyle=style.get("ls", "-"),
            linewidth=1.6,
            label=curve_labels[model],
        )
    curve_axis.set_xlim(0, 1)
    curve_axis.set_ylim(0, 1)
    curve_axis.set_aspect("equal", adjustable="box")
    curve_axis.set_title("TARP expected coverage", fontsize=11)
    curve_axis.set_xlabel("nominal coverage")
    curve_axis.set_ylabel("expected coverage probability")
    curve_axis.legend(frameon=False, fontsize=8, loc="lower right")

    speed_axis = figure.add_subplot(grid[1, 4])
    speed_axis.axis("off")
    rws26 = speed["variants"]["rws26"]
    posterior_us = 1.0e6 * rws26["amortized_posterior_seconds_per_object"]
    predictive_ms = 1.0e3 * rws26["posterior_predictive_seconds_per_object"]
    pipeline_ms = 1.0e3 * rws26["catalog_pipeline_seconds_per_object"]
    ratio = rws26["popcosmos_throughput_ratio"]
    speed_axis.text(0.0, 0.98, "Inference throughput", fontsize=13, weight="bold", va="top")
    speed_axis.text(
        0.0,
        0.84,
        (
            "Our 26-band RWS on one H100\n"
            f"Encoder + 128 draws: {posterior_us:.1f} us / galaxy\n"
            f"128 DSPS predictions: {predictive_ms:.0f} ms / galaxy\n"
            f"Features + draws + DSPS: {pipeline_ms:.0f} ms / galaxy\n\n"
            "Published Pop-COSMOS\n"
            f"MCMC throughput: {speed['popcosmos_gpu_seconds_per_object']:.0f} GPU-s / galaxy\n\n"
            f"~{ratio:.0f}x lower GPU time / galaxy\n"
            "for the measured catalog pipeline"
        ),
        fontsize=10,
        linespacing=1.45,
        va="top",
    )
    speed_axis.text(
        0.0,
        0.08,
        (
            "Descriptive throughput ratio, not a controlled\n"
            "algorithmic benchmark: amortized RWS vs MCMC,\n"
            "different forward models and GPU generations.\n"
            "Compilation, setup and I/O are excluded."
        ),
        fontsize=8,
        color="#555555",
        va="bottom",
    )

    for axis in (mira_axis, tarp_axis, curve_axis):
        axis.grid(color="#E8E8E8", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "FENIKS and Pop-COSMOS: redshift accuracy, posterior calibration and throughput",
        fontsize=18,
    )
    figure.text(
        0.5,
        0.035,
        (
            "Top: FENIKS synthetic closure, matched COSMOS public spectroscopy (N=1,395), "
            "and the external Pop-COSMOS paper cohort are separated by gray rules. "
            "Bottom: redshift-only MIRA/TARP uses dense 128-draw posteriors; Pop-COSMOS "
            "public quantiles cannot supply these calibration scores."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.savefig(path, dpi=200)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    inputs = {
        "feniks_rws": _resolve(args.feniks_rws, "photoz_metrics.csv"),
        "feniks_rws_mixture": _resolve(args.feniks_rws_mixture, "photoz_metrics.csv"),
        "feniks_smcwake": _resolve(args.feniks_smcwake, "photoz_metrics.csv"),
        "matched_metrics": _resolve(args.matched_metrics, "redshift_method_metrics.csv"),
        "calibration_table": _resolve(
            args.calibration_table, "redshift_calibration_comparison.csv"
        ),
        "feniks_tarp": _resolve(args.feniks_tarp, "tarp_coverage.csv"),
        "cosmos_tarp": _resolve(args.cosmos_tarp, "tarp_coverage.csv"),
        "timing_26": _resolve(args.timing_26, "timing_summary.json"),
        "timing_24": _resolve(args.timing_24, "timing_summary.json"),
    }
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    accuracy = _read_accuracy_metrics(
        feniks_rws=args.feniks_rws,
        feniks_rws_mixture=args.feniks_rws_mixture,
        feniks_smcwake=args.feniks_smcwake,
        matched_metrics=args.matched_metrics,
    )
    calibration = _read_calibration(args.calibration_table)
    tarp_coverage = pd.concat(
        [
            _read_tarp_coverage(args.feniks_tarp, context="feniks_synthetic"),
            _read_tarp_coverage(args.cosmos_tarp, context="cosmos_public_specz"),
        ],
        ignore_index=True,
    )
    speed = _speed_summary(
        _read_timing(args.timing_26),
        _read_timing(args.timing_24),
        popcosmos_gpu_seconds=args.popcosmos_gpu_seconds,
    )
    accuracy.to_csv(args.out / "publication_dashboard_metrics.csv", index=False)
    accuracy.to_parquet(args.out / "publication_dashboard_metrics.parquet", index=False)
    _write_dashboard(
        accuracy=accuracy,
        calibration=calibration,
        tarp_coverage=tarp_coverage,
        speed=speed,
        path=args.out / "feniks_popcosmos_dashboard.png",
    )
    summary = {
        "status": "complete",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in inputs.items()
        },
        "accuracy_rows": json.loads(accuracy.to_json(orient="records")),
        "speed": speed,
        "external_popcosmos_result": POPCOSMOS_PAPER,
        "limitations": [
            "FENIKS is synthetic closure; COSMOS uses real public spectroscopy.",
            "The matched COSMOS benchmark and published Pop-COSMOS cohort differ.",
            "The speed ratio compares different inference algorithms, forward models, and GPUs.",
            "Public Pop-COSMOS quantiles do not permit MIRA or TARP evaluation.",
        ],
    }
    (args.out / "publication_dashboard_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "DONE").touch()
    print(f"[publication-dashboard] methods={len(accuracy)} -> {args.out}")


if __name__ == "__main__":
    main()
