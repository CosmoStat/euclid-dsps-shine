#!/usr/bin/env python3
"""Compare redshift-only MIRA/TARP results across truth cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def _write_plot(frame: pd.DataFrame, path: Path) -> None:
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
    labels = [
        f"{context.replace('_', ' ')}\n{model}"
        for context, model in zip(
            calibrated["context"], calibrated["model"], strict=True
        )
    ]
    x = np.arange(len(calibrated), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=False)
    figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.25, wspace=0.25)

    mira_center = calibrated["mira_bootstrap_mean"].to_numpy(float)
    axes[0].errorbar(
        x,
        mira_center,
        yerr=np.vstack(
            [
                mira_center - calibrated["mira_bootstrap_q025"].to_numpy(float),
                calibrated["mira_bootstrap_q975"].to_numpy(float) - mira_center,
            ]
        ),
        fmt="o",
        capsize=4,
    )
    axes[0].axhline(2.0 / 3.0, color="#202020", linestyle="--", linewidth=1.1)
    axes[0].set_title("MIRA redshift calibration")
    axes[0].set_ylabel("MIRA score")

    tarp_center = calibrated["tarp_bootstrap_atc_mean"].to_numpy(float)
    axes[1].errorbar(
        x,
        tarp_center,
        yerr=np.vstack(
            [
                tarp_center - calibrated["tarp_bootstrap_atc_q025"].to_numpy(float),
                calibrated["tarp_bootstrap_atc_q975"].to_numpy(float) - tarp_center,
            ]
        ),
        fmt="o",
        capsize=4,
    )
    axes[1].axhline(0.0, color="#202020", linestyle="--", linewidth=1.1)
    axes[1].set_title("TARP redshift calibration")
    axes[1].set_ylabel("Area to curve")

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.margins(x=0.12)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    figure.suptitle("Redshift posterior calibration on available truth cohorts")
    figure.savefig(path, dpi=180)
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
    _write_plot(frame, out / "redshift_calibration_comparison.png")

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
    }
    (out / "redshift_calibration_comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "DONE").touch()
    print(f"[redshift-calibration-comparison] rows={len(frame)} -> {out}")


if __name__ == "__main__":
    main()
