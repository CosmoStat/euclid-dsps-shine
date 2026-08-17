#!/usr/bin/env python3
"""Select a stable Pop-COSMOS SMC likelihood without using spectroscopy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("floor_0p00", "floor_0p02", "floor_0p05")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-median-abs-logz-seed-delta", type=float, default=0.5)
    parser.add_argument("--max-median-chi2-per-band", type=float, default=10.0)
    parser.add_argument("--max-median-frac-abs-gt-5", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or args.root / "pilot_selection"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite pilot summary: {out}")
    out.mkdir(parents=True)
    rows = []
    candidates = []
    expected_rows = None
    for variant in VARIANTS:
        runs = sorted((args.root / variant).glob("seed_*"))
        if len(runs) != 2:
            raise RuntimeError(
                f"Expected exactly two seeds under {args.root / variant}"
            )
        object_frames = []
        summaries = []
        for run in runs:
            if not (run / "DONE").is_file():
                raise RuntimeError(f"Incomplete SMC run: {run}")
            summary = json.loads((run / "smc_summary.json").read_text())
            objects = pd.read_parquet(run / "smc_object_diagnostics.parquet")
            row_index = objects["row_index"].to_numpy(dtype=np.int64)
            if expected_rows is None:
                expected_rows = row_index
            elif not np.array_equal(expected_rows, row_index):
                raise RuntimeError(f"Cohort/order mismatch in {run}")
            object_frames.append(objects)
            summaries.append(summary)
            rows.append(
                {
                    "variant": variant,
                    "seed": int(summary["seed"]),
                    "support_status": summary["support_gate"]["status"],
                    **summary["metrics"],
                }
            )
        left = object_frames[0].set_index("row_index")
        right = object_frames[1].set_index("row_index")
        logz_delta = left["log_evidence"] - right["log_evidence"]
        median_abs_delta = float(np.median(np.abs(logz_delta)))
        support_pass = all(
            item["support_gate"]["status"] == "PASS" for item in summaries
        )
        median_chi2 = float(
            np.mean(
                [item["metrics"]["median_chi2_per_valid_band"] for item in summaries]
            )
        )
        median_tail = float(
            np.mean([item["metrics"]["median_fraction_abs_gt_5"] for item in summaries])
        )
        replicate_pass = median_abs_delta <= args.max_median_abs_logz_seed_delta
        adequacy_pass = (
            median_chi2 <= args.max_median_chi2_per_band
            and median_tail <= args.max_median_frac_abs_gt_5
        )
        candidates.append(
            {
                "variant": variant,
                "seed_directories": [str(run) for run in runs],
                "support_pass": support_pass,
                "replicate_stability_pass": replicate_pass,
                "photometric_adequacy_pass": adequacy_pass,
                "median_abs_logz_seed_delta": median_abs_delta,
                "mean_seed_mean_log_evidence": float(
                    np.mean(
                        [item["metrics"]["mean_log_evidence"] for item in summaries]
                    )
                ),
                "mean_seed_median_log_evidence": float(
                    np.mean(
                        [item["metrics"]["median_log_evidence"] for item in summaries]
                    )
                ),
                "mean_seed_median_chi2_per_valid_band": median_chi2,
                "mean_seed_median_fraction_abs_gt_5": median_tail,
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    eligible = candidate_frame[
        candidate_frame["support_pass"]
        & candidate_frame["replicate_stability_pass"]
        & candidate_frame["photometric_adequacy_pass"]
    ]
    if len(eligible):
        selected = str(
            eligible.sort_values("mean_seed_mean_log_evidence", ascending=False).iloc[
                0
            ]["variant"]
        )
        status = "PASS"
    else:
        selected = None
        status = "FAIL"
    pd.DataFrame(rows).to_csv(out / "seed_metrics.csv", index=False)
    candidate_frame.to_csv(out / "likelihood_candidates.csv", index=False)
    payload = {
        "status": "complete",
        "selection_status": status,
        "selected_variant": selected,
        "selection_rule": (
            "require both support gates, paired seed log-evidence stability and "
            "photometric adequacy; rank eligible likelihoods by mean seed mean log evidence"
        ),
        "spectroscopy_used": False,
        "thresholds": {
            "max_median_abs_logz_seed_delta": args.max_median_abs_logz_seed_delta,
            "max_median_chi2_per_band": args.max_median_chi2_per_band,
            "max_median_fraction_abs_gt_5": args.max_median_frac_abs_gt_5,
        },
        "candidates": candidates,
    }
    (out / "selection_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "DONE").touch()
    print(f"[posthoc-smc-summary] selection={status} variant={selected} -> {out}")


if __name__ == "__main__":
    main()
