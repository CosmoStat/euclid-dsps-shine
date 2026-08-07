#!/usr/bin/env python3
"""Collect post-hoc calibration decision metrics without collapsing posteriors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.root.rglob("importance_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        redshift = payload.get("redshift_metrics") or {}
        rows.append(
            {
                "artifact": str(path.relative_to(args.root)),
                "n_objects": payload.get("n_objects"),
                "n_joint_draws": payload.get("n_joint_draws"),
                "median_raw_ess_fraction": payload.get(
                    "median_raw_ess_fraction"
                ),
                "median_psis_ess_fraction": payload.get(
                    "median_psis_ess_fraction"
                ),
                "fraction_pareto_k_gt_0p7": payload.get(
                    "fraction_pareto_k_gt_0p7"
                ),
                "redshift_n": redshift.get("n_objects"),
                "redshift_nmad": redshift.get("nmad"),
                "redshift_outlier_fraction_0p15": redshift.get(
                    "outlier_fraction_0p15"
                ),
                "redshift_coverage_68": redshift.get("coverage_68"),
                "redshift_coverage_95": redshift.get("coverage_95"),
                "redshift_pit_ks_uniform": redshift.get("pit_ks_uniform"),
            }
        )
    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "importance_decision_table.csv", index=False)
    frame.to_parquet(args.out / "importance_decision_table.parquet", index=False)
    em = []
    for path in sorted(args.root.rglob("em_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        em.append(
            {
                "artifact": str(path.relative_to(args.root)),
                "n_objects": payload.get("n_objects"),
                "proposal_samples_per_object": payload.get(
                    "proposal_samples_per_object"
                ),
                "best_validation_mean_log_evidence_is": payload.get(
                    "best_validation_mean_log_evidence_is"
                ),
                "weight_kind": payload.get("weight_kind"),
            }
        )
    pd.DataFrame(em).to_csv(args.out / "empirical_bayes_decision_table.csv", index=False)
    summary = {
        "status": "complete",
        "importance_artifacts": len(rows),
        "empirical_bayes_artifacts": len(em),
        "posterior_contract": "Tables contain diagnostics only; distributional conclusions require the linked weighted/resampled joint banks.",
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.out / "DONE").write_text(
        json.dumps({"status": "complete"}, indent=2), encoding="utf-8"
    )
    print(f"[posthoc-summary] importance={len(rows)} em={len(em)} -> {args.out}")


if __name__ == "__main__":
    main()
