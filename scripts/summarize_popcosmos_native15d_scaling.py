#!/usr/bin/env python3
"""Summarize held-out redshift metrics across native COSMOS RWS stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# ``full`` is the held-out-safe 40k pool in the publication workflow.  Keep
# ``n40k`` as a read alias so older single-chain runs remain summarizable.
STAGES = (("n5k", 5_000), ("n20k", 20_000), ("full", 40_000))
STAGE_PATH_ALIASES = {"full": ("full", "n40k")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def collect_scaling_rows(root: Path) -> list[dict[str, float | int | str]]:
    rows = []
    for stage, train_catalog_size in STAGES:
        candidates = STAGE_PATH_ALIASES.get(stage, (stage,))
        path = next(
            (
                root / candidate / "inference/redshift_metrics.json"
                for candidate in candidates
                if (root / candidate / "inference/redshift_metrics.json").is_file()
            ),
            None,
        )
        if path is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row: dict[str, float | int | str] = {
            "stage": stage,
            "train_catalog_size": train_catalog_size,
            "n_inference": int(payload["n_inference"]),
            **payload["metrics"],
        }
        intervals = payload.get("bootstrap", {}).get(
            "confidence_intervals_95", {}
        )
        for name, interval in intervals.items():
            row[f"{name}_ci95_low"] = float(interval["low"])
            row[f"{name}_ci95_high"] = float(interval["high"])
        rows.append(row)
    return rows


def _write_plot(frame: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    metrics = (
        ("nmad", "NMAD"),
        ("rmse", "RMSE"),
        ("outlier_fraction_0p15", "Outlier fraction"),
        ("coverage_68", "68% coverage"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5), sharex=True)
    x = frame["train_catalog_size"].to_numpy(float)
    for ax, (name, label) in zip(axes.flat, metrics, strict=True):
        y = frame[name].to_numpy(float)
        low_name = f"{name}_ci95_low"
        high_name = f"{name}_ci95_high"
        yerr = None
        if low_name in frame and high_name in frame:
            low = frame[low_name].to_numpy(float)
            high = frame[high_name].to_numpy(float)
            yerr = [y - low, high - y]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3)
        ax.set_xscale("log")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("training catalog size")
    fig.suptitle("Native DSPS RWS held-out COSMOS redshift scaling")
    fig.tight_layout()
    fig.savefig(out / "redshift_scaling.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.out or args.root / "redshift_scaling"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_scaling_rows(args.root)
    if len(rows) != len(STAGES):
        raise RuntimeError(f"Expected all three scaling stages, found {len(rows)}")
    frame = pd.DataFrame(rows).sort_values("train_catalog_size")
    frame.to_csv(out / "redshift_scaling_metrics.csv", index=False)
    payload = {
        "status": "complete",
        "science_target": "z_obs_only",
        "evaluation": "fixed held-out cohort outside the 40k training pool",
        "stages": rows,
    }
    (out / "redshift_scaling_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_plot(frame, out)
    print(f"[cosmos-native15d-scaling] stages=3 -> {out}")


if __name__ == "__main__":
    main()
