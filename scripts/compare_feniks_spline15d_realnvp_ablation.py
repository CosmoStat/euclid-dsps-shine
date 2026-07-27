#!/usr/bin/env python3
"""Rank spline-15D RealNVP ablations using validation diagnostics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from euclid_dsps.io import ensure_dir, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--pattern", default="feniks_spline15d_realnvp_v2_[abcd]*")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/reports/spline15d_realnvp_v2")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = []
    for run_dir in sorted(args.runs_root.glob(args.pattern)):
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation = dict(summary.get("best_selection_diagnostics", {}))
        calibration = dict(summary.get("temperature_calibration", {}))
        test = dict(summary.get("test_temperature_calibrated_metrics", {}))
        rows.append(
            {
                "run": run_dir.name,
                "validation_eligible": bool(
                    summary.get("best_selection_eligible", False)
                ),
                "validation_selection_metric": float(
                    summary.get("best_selection_metric", float("inf"))
                ),
                "best_epoch": int(summary.get("best_epoch", -1)),
                "validation_nll": float(
                    summary.get("best_validation_nll", float("nan"))
                ),
                "validation_novel_nll": float(
                    summary.get("validation_novel_nll", float("nan"))
                ),
                "validation_median_ks": validation.get("median_ks_normalized"),
                "validation_max_ks": validation.get("max_ks_normalized"),
                "validation_correlation_frobenius": validation.get(
                    "correlation_frobenius_physical"
                ),
                "validation_base_std_mean": validation.get("base_std_mean"),
                "selected_temperature": calibration.get("selected_base_temperature"),
                "test_median_ks_calibrated": test.get("median_ks_normalized"),
                "test_max_ks_calibrated": test.get("max_ks_normalized"),
                "test_correlation_frobenius_calibrated": test.get(
                    "correlation_frobenius_physical"
                ),
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No completed runs matching {args.runs_root / args.pattern}"
        )
    frame = pd.DataFrame(rows).sort_values(
        ["validation_eligible", "validation_selection_metric"],
        ascending=[False, True],
    )
    selected = frame.iloc[0]
    ensure_dir(args.out)
    frame.to_csv(args.out / "ablation_ranking.csv", index=False)
    write_json(
        args.out / "selected_run.json",
        {
            "selection_split": "validation",
            "selected_run": selected["run"],
            "validation_eligible": bool(selected["validation_eligible"]),
            "validation_selection_metric": float(
                selected["validation_selection_metric"]
            ),
            "note": "Test columns are reported after selection and never rank runs.",
        },
    )
    print(frame.to_string(index=False))
    print(f"[ablation] selected on validation: {selected['run']}")


if __name__ == "__main__":
    main()
