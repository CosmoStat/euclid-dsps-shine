#!/usr/bin/env python3
"""Compare redshift-only MIRA/TARP results across truth cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_DASHBOARD_RUNS = {
    ("cosmos_public_specz", "rws26"),
    ("cosmos_public_specz", "rws24"),
    ("feniks_synthetic", "rws_k8_t2_seed2"),
    ("feniks_synthetic", "rws_k8_t2_seed3"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feniks-mira", type=Path, required=True)
    parser.add_argument("--feniks-tarp", type=Path, required=True)
    parser.add_argument("--cosmos-mira", type=Path, required=True)
    parser.add_argument("--cosmos-tarp", type=Path, required=True)
    parser.add_argument("--photoz-metrics", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _resolve_table(path: Path, filename: str) -> Path:
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


def _read_context(
    *,
    context: str,
    truth_scope: str,
    mira_path: Path,
    tarp_path: Path,
) -> pd.DataFrame:
    mira_file = _resolve_table(mira_path, "mira_scores.csv")
    tarp_file = _resolve_table(tarp_path, "tarp_summary.csv")
    mira = pd.read_csv(mira_file)
    tarp = pd.read_csv(tarp_file)
    mira = mira.loc[mira["group"].eq("marginal_z_obs")].copy()
    tarp = tarp.loc[tarp["group"].eq("marginal_z_obs")].copy()
    if mira.empty or tarp.empty:
        raise ValueError(f"Missing marginal_z_obs results for context {context!r}")
    if mira["model"].duplicated().any() or tarp["model"].duplicated().any():
        raise ValueError(f"Duplicate model rows for context {context!r}")

    mira = mira.loc[
        :,
        [
            "model",
            "num_objects",
            "num_posterior_samples",
            "score",
            "ideal_score",
            "bootstrap_mean",
            "bootstrap_std",
            "bootstrap_q025",
            "bootstrap_q975",
        ],
    ].rename(
        columns={
            "score": "mira_score",
            "ideal_score": "mira_ideal_score",
            "bootstrap_mean": "mira_bootstrap_mean",
            "bootstrap_std": "mira_bootstrap_std",
            "bootstrap_q025": "mira_bootstrap_q025",
            "bootstrap_q975": "mira_bootstrap_q975",
        }
    )
    tarp = tarp.loc[
        :,
        [
            "model",
            "num_objects",
            "num_posterior_samples",
            "atc",
            "ks_pvalue",
            "coverage_rmse",
            "coverage_max_abs_error",
            "bootstrap_atc_mean",
            "bootstrap_atc_std",
            "bootstrap_atc_q025",
            "bootstrap_atc_q975",
        ],
    ].rename(
        columns={
            "num_objects": "tarp_num_objects",
            "num_posterior_samples": "tarp_num_posterior_samples",
            "atc": "tarp_atc",
            "ks_pvalue": "tarp_ks_pvalue",
            "coverage_rmse": "tarp_coverage_rmse",
            "coverage_max_abs_error": "tarp_coverage_max_abs_error",
            "bootstrap_atc_mean": "tarp_bootstrap_atc_mean",
            "bootstrap_atc_std": "tarp_bootstrap_atc_std",
            "bootstrap_atc_q025": "tarp_bootstrap_atc_q025",
            "bootstrap_atc_q975": "tarp_bootstrap_atc_q975",
        }
    )
    merged = mira.merge(tarp, on="model", validate="one_to_one")
    if not merged["num_objects"].eq(merged["tarp_num_objects"]).all():
        raise ValueError(f"MIRA/TARP truth counts differ for context {context!r}")
    if (
        not merged["num_posterior_samples"]
        .eq(merged["tarp_num_posterior_samples"])
        .all()
    ):
        raise ValueError(f"MIRA/TARP sample counts differ for context {context!r}")
    merged = merged.drop(columns=["tarp_num_objects", "tarp_num_posterior_samples"])
    merged.insert(0, "truth_scope", truth_scope)
    merged.insert(0, "context", context)
    return merged


def _read_tarp_coverage(path: Path, *, context: str) -> pd.DataFrame:
    coverage_file = _resolve_table(path, "tarp_coverage.csv")
    coverage = pd.read_csv(coverage_file)
    coverage = coverage.loc[coverage["group"].eq("marginal_z_obs")].copy()
    if coverage.empty:
        raise ValueError(f"Missing marginal_z_obs TARP coverage for {context!r}")
    coverage.insert(0, "context", context)
    return coverage


def _validate_dashboard_contract(
    frame: pd.DataFrame,
    tarp_coverage: pd.DataFrame,
) -> None:
    calibrated = frame.loc[
        frame["mira_score"].notna() & frame["tarp_atc"].notna(),
        ["context", "model"],
    ]
    observed = set(calibrated.itertuples(index=False, name=None))
    if observed != EXPECTED_DASHBOARD_RUNS:
        raise ValueError(
            "Dashboard run contract mismatch: "
            f"expected={sorted(EXPECTED_DASHBOARD_RUNS)!r}, observed={sorted(observed)!r}"
        )
    coverage_runs = set(
        tarp_coverage.loc[:, ["context", "model"]].itertuples(index=False, name=None)
    )
    if coverage_runs != EXPECTED_DASHBOARD_RUNS:
        raise ValueError(
            "TARP coverage run contract mismatch: "
            f"expected={sorted(EXPECTED_DASHBOARD_RUNS)!r}, "
            f"observed={sorted(coverage_runs)!r}"
        )
    if tarp_coverage.duplicated(["context", "model", "alpha"]).any():
        raise ValueError("Duplicate TARP alpha rows in dashboard inputs")


def _write_plot(
    frame: pd.DataFrame,
    path: Path,
    *,
    tarp_coverage: pd.DataFrame | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibrated = frame.loc[
        frame["mira_score"].notna() & frame["tarp_atc"].notna()
    ].copy()
    context_order = {"cosmos_public_specz": 0, "feniks_synthetic": 1}
    model_order = {
        "rws26": 0,
        "rws24": 1,
        "rws_k8_t2_seed2": 2,
        "rws_k8_t2_seed3": 3,
    }
    calibrated["_context_order"] = calibrated["context"].map(context_order)
    calibrated["_model_order"] = calibrated["model"].map(model_order)
    calibrated = calibrated.sort_values(["_context_order", "_model_order"])
    plot_style = {
        "rws26": {
            "label": "COSMOS RWS\n26 bands",
            "legend": "COSMOS RWS, 26 bands",
            "color": "#0072B2",
            "linestyle": "-",
            "marker": "o",
        },
        "rws24": {
            "label": "COSMOS RWS\n24 bands (no IRAC)",
            "legend": "COSMOS RWS, 24 bands (no IRAC)",
            "color": "#009E73",
            "linestyle": "-",
            "marker": "s",
        },
        "rws_k8_t2_seed2": {
            "label": "FENIKS RWS\nseed 2",
            "legend": "FENIKS RWS, seed 2",
            "color": "#D55E00",
            "linestyle": "--",
            "marker": "D",
        },
        "rws_k8_t2_seed3": {
            "label": "FENIKS RWS\nseed 3",
            "legend": "FENIKS RWS, seed 3",
            "color": "#CC79A7",
            "linestyle": "--",
            "marker": "^",
        },
    }
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    models = calibrated["model"].tolist()
    for index, model in enumerate(models):
        plot_style.setdefault(
            model,
            {
                "label": model,
                "legend": model,
                "color": default_colors[index % len(default_colors)],
                "linestyle": "-",
                "marker": "o",
            },
        )
    labels = [
        f"{plot_style[row.model]['label']}\nN={int(row.num_objects):,}"
        for row in calibrated.itertuples(index=False)
    ]
    x = np.arange(len(calibrated), dtype=float)
    if len(x) > 2:
        x[2:] += 0.45
    figure, axes = plt.subplots(
        1, 3, figsize=(18, 6.2), gridspec_kw={"width_ratios": [1.05, 1.25, 1.05]}
    )
    figure.subplots_adjust(
        left=0.055, right=0.985, top=0.83, bottom=0.25, wspace=0.28
    )

    theoretical_sigma = np.sqrt(
        (1.0 / 18.0) / calibrated["num_objects"].to_numpy(float)
    )
    for xi, sigma in zip(x, theoretical_sigma, strict=True):
        axes[0].fill_between(
            [xi - 0.24, xi + 0.24],
            [2.0 / 3.0 - 1.96 * sigma] * 2,
            [2.0 / 3.0 + 1.96 * sigma] * 2,
            color="#B8B8B8",
            alpha=0.42,
            linewidth=0,
            zorder=0,
        )
    for index, row in enumerate(calibrated.itertuples(index=False)):
        style = plot_style[row.model]
        axes[0].errorbar(
            x[index],
            row.mira_score,
            yerr=np.asarray(
                [
                    [row.mira_score - row.mira_bootstrap_q025],
                    [row.mira_bootstrap_q975 - row.mira_score],
                ]
            ),
            fmt=style["marker"],
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7,
            capsize=4,
            linewidth=1.5,
            zorder=3,
        )
        axes[0].annotate(
            f"{row.mira_score:.3f}",
            (x[index], row.mira_score),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=style["color"],
        )
    axes[0].axhline(
        2.0 / 3.0,
        color="#202020",
        linestyle=(0, (5, 3)),
        linewidth=1.2,
        label="Ideal = 2/3",
    )
    axes[0].fill_between(
        [], [], [], color="#B8B8B8", alpha=0.42, label="Ideal 95% range"
    )
    axes[0].set_title("MIRA score")
    axes[0].set_ylabel("MIRA score")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5)

    axes[1].plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color="#202020",
        linestyle=(0, (5, 3)),
        linewidth=1.2,
        label="Ideal",
        zorder=1,
    )
    if tarp_coverage is not None:
        for model in models:
            context = calibrated.loc[calibrated["model"].eq(model), "context"].iloc[0]
            curve = tarp_coverage.loc[
                tarp_coverage["model"].eq(model)
                & tarp_coverage["context"].eq(context)
            ].sort_values("alpha")
            if curve.empty:
                raise ValueError(f"Missing TARP coverage curve for {context}/{model}")
            style = plot_style[model]
            axes[1].fill_between(
                curve["alpha"],
                curve["bootstrap_q025"],
                curve["bootstrap_q975"],
                color=style["color"],
                alpha=0.10,
                linewidth=0,
            )
            axes[1].plot(
                curve["alpha"],
                curve["ecp"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                label=style["legend"],
            )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_title("TARP coverage")
    axes[1].set_xlabel("Nominal coverage, alpha")
    axes[1].set_ylabel("Expected coverage probability")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8.1)

    for index, row in enumerate(calibrated.itertuples(index=False)):
        style = plot_style[row.model]
        axes[2].errorbar(
            x[index],
            row.tarp_atc,
            yerr=np.asarray(
                [
                    [row.tarp_atc - row.tarp_bootstrap_atc_q025],
                    [row.tarp_bootstrap_atc_q975 - row.tarp_atc],
                ]
            ),
            fmt=style["marker"],
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=7,
            capsize=4,
            linewidth=1.5,
            zorder=3,
        )
        axes[2].annotate(
            f"{row.tarp_atc:+.3f}",
            (x[index], row.tarp_atc),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=style["color"],
        )
    axes[2].axhline(
        0.0, color="#202020", linestyle=(0, (5, 3)), linewidth=1.2
    )
    axes[2].set_title("TARP area to curve")
    axes[2].set_ylabel("ATC (ideal = 0)")

    for axis in (axes[0], axes[2]):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8.5)
        if len(x) > 2:
            axis.axvline((x[1] + x[2]) / 2.0, color="#D6D6D6", linewidth=0.8)
        axis.margins(x=0.12)
    for axis in axes:
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Redshift posterior calibration: COSMOS spectroscopy and FENIKS closure",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.055,
        "Points/curves: measured values; error bars and ribbons: object-bootstrap 95% CI. "
        "MIRA gray ranges: 2/3 +/- 1.96 sqrt[(1/18)/N]. Cross-cohort comparisons are descriptive.",
        ha="center",
        fontsize=9,
        color="#3F3F3F",
    )
    figure.savefig(path, dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_paths = {
        "feniks_mira": _resolve_table(args.feniks_mira, "mira_scores.csv"),
        "feniks_tarp": _resolve_table(args.feniks_tarp, "tarp_summary.csv"),
        "cosmos_mira": _resolve_table(args.cosmos_mira, "mira_scores.csv"),
        "cosmos_tarp": _resolve_table(args.cosmos_tarp, "tarp_summary.csv"),
    }
    if args.photoz_metrics is not None:
        input_paths["photoz_metrics"] = _resolve_table(
            args.photoz_metrics, "redshift_method_metrics.csv"
        )

    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [
            _read_context(
                context="feniks_synthetic",
                truth_scope="5000 held-out synthetic galaxies with latent truth",
                mira_path=input_paths["feniks_mira"],
                tarp_path=input_paths["feniks_tarp"],
            ),
            _read_context(
                context="cosmos_public_specz",
                truth_scope="1395 real COSMOS galaxies with public spectroscopy",
                mira_path=input_paths["cosmos_mira"],
                tarp_path=input_paths["cosmos_tarp"],
            ),
        ],
        ignore_index=True,
    )
    if "photoz_metrics" in input_paths:
        photoz = pd.read_csv(input_paths["photoz_metrics"]).rename(
            columns={"method": "model"}
        )
        photoz.insert(0, "context", "cosmos_public_specz")
        frame = frame.merge(photoz, on=["context", "model"], how="outer")
        popcosmos = frame["model"].eq("popcosmos")
        frame.loc[popcosmos, "truth_scope"] = (
            "1395 real COSMOS galaxies with public spectroscopy"
        )
    frame["has_dense_posterior_calibration"] = frame["mira_score"].notna()
    frame.to_csv(out / "redshift_calibration_comparison.csv", index=False)
    frame.to_parquet(out / "redshift_calibration_comparison.parquet", index=False)
    tarp_coverage = pd.concat(
        [
            _read_tarp_coverage(
                args.feniks_tarp,
                context="feniks_synthetic",
            ),
            _read_tarp_coverage(
                args.cosmos_tarp,
                context="cosmos_public_specz",
            ),
        ],
        ignore_index=True,
    )
    _validate_dashboard_contract(frame, tarp_coverage)
    _write_plot(
        frame,
        out / "redshift_calibration_dashboard.png",
        tarp_coverage=tarp_coverage,
    )

    summary = {
        "status": "complete",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "rows": json.loads(frame.to_json(orient="records")),
        "limitations": [
            "FENIKS is synthetic held-out closure; COSMOS is a spectroscopy-selected real subset.",
            "Cross-context distances from the ideal are descriptive, not paired model comparisons.",
            "Public Pop-COSMOS quantiles support photo-z metrics but not MIRA/TARP without chains.",
        ],
        "plot_contract": {
            "runs": [
                {"context": context, "model": model}
                for context, model in sorted(EXPECTED_DASHBOARD_RUNS)
            ],
            "parameter": "z_obs",
            "mira_point": "score",
            "mira_errorbar": "object-bootstrap 95% interval",
            "mira_ideal": 2.0 / 3.0,
            "mira_ideal_variance": "(1/18) / num_objects",
            "tarp_curve": "expected coverage probability versus alpha",
            "tarp_errorbar": "object-bootstrap 95% interval for ATC",
            "tarp_ideal_atc": 0.0,
        },
    }
    (out / "redshift_calibration_comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "DONE").touch()
    print(f"[redshift-calibration-comparison] rows={len(frame)} -> {out}")


if __name__ == "__main__":
    main()
