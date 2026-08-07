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
        redshift_by_weight = payload.get("redshift_metrics_by_weight") or {}
        redshift_raw = redshift_by_weight.get("raw") or {}
        redshift_psis = (
            redshift_by_weight.get("psis") or payload.get("redshift_metrics") or {}
        )
        support = payload.get("support_gate") or {}
        rows.append(
            {
                "artifact": str(path.relative_to(args.root)),
                "n_objects": payload.get("n_objects"),
                "n_joint_draws": payload.get("n_joint_draws"),
                "median_raw_ess_fraction": payload.get("median_raw_ess_fraction"),
                "median_psis_ess_fraction": payload.get("median_psis_ess_fraction"),
                "fraction_pareto_k_gt_0p7": payload.get("fraction_pareto_k_gt_0p7"),
                "support_gate": support.get("status"),
                "redshift_n": redshift_psis.get("n_objects"),
                "redshift_raw_nmad": redshift_raw.get("nmad"),
                "redshift_raw_outlier_fraction_0p15": redshift_raw.get(
                    "outlier_fraction_0p15"
                ),
                "redshift_raw_coverage_68": redshift_raw.get("coverage_68"),
                "redshift_raw_coverage_95": redshift_raw.get("coverage_95"),
                "redshift_raw_pit_ks_uniform": redshift_raw.get("pit_ks_uniform"),
                "redshift_psis_nmad": redshift_psis.get("nmad"),
                "redshift_psis_outlier_fraction_0p15": redshift_psis.get(
                    "outlier_fraction_0p15"
                ),
                "redshift_psis_coverage_68": redshift_psis.get("coverage_68"),
                "redshift_psis_coverage_95": redshift_psis.get("coverage_95"),
                "redshift_psis_pit_ks_uniform": redshift_psis.get("pit_ks_uniform"),
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
                "selected_candidate": payload.get("selected_candidate"),
                "stopping_reason": payload.get("stopping_reason"),
            }
        )
    pd.DataFrame(em).to_csv(
        args.out / "empirical_bayes_decision_table.csv", index=False
    )
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
