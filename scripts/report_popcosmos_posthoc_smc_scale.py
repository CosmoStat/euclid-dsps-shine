#!/usr/bin/env python3
"""Print a compact decision report for a progressive Pop-COSMOS SMC tier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    cohort = _read_json(root / "cohorts/cohort_manifest.json")
    n_objects = int(cohort["smc_objects"])
    print(f"tier_objects={n_objects}")
    print(f"probe_objects={cohort['proposal_probe_objects']}")
    print(f"progressive_parent={cohort.get('progressive_parent', {}).get('root')}")

    selection_path = root / "pilot_selection/selection_summary.json"
    if not selection_path.is_file():
        print("smc_selection=PENDING")
        print("NEXT_ACTION=WAIT_FOR_SMC_FINALIZER")
        return
    selection = _read_json(selection_path)
    print(f"smc_selection={selection.get('selection_status')}")
    print(f"selected_variant={selection.get('selected_variant')}")
    for seed in (260817, 260818):
        summary_path = root / f"floor_0p05/seed_{seed}/smc_summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        metrics = summary["metrics"]
        print(
            f"seed_{seed}: gate={summary['support_gate']['status']} "
            f"ESS={metrics['median_final_ess_fraction']:.4f} "
            f"anc={metrics['median_unique_ancestor_fraction']:.4f} "
            f"accept={metrics['median_mala_acceptance']:.4f}"
        )
    if selection.get("selection_status") != "PASS":
        print("NEXT_ACTION=STOP_SMC_GATE_FAILED")
        return

    decision_path = root / "proposal_refresh_k2048/refresh_validation_summary.json"
    if not decision_path.is_file():
        print("encoder_refresh=PENDING")
        print("ordinary_is=PENDING")
        print("NEXT_ACTION=WAIT_FOR_REFRESH")
        return
    decision = _read_json(decision_path)
    metrics = decision["candidate_metrics"]
    print(f"encoder_refresh={decision['encoder_refresh_gate']}")
    print(f"ordinary_is={decision['ordinary_importance_support_gate']}")
    print(f"median_raw_ess_fraction={metrics['median_raw_ess_fraction']:.6f}")
    print(f"fraction_pareto_k_gt_0p7={metrics['fraction_pareto_k_gt_0p7']:.6f}")
    passed = (
        decision["encoder_refresh_gate"] == "PASS"
        and decision["ordinary_importance_support_gate"] == "PASS"
    )
    if not passed:
        print("NEXT_ACTION=STOP_PROPOSAL_SUPPORT_FAILED")
    elif n_objects == 512:
        print("NEXT_ACTION=SUBMIT_1024_CONFIRMATION")
    elif n_objects == 1024:
        print("NEXT_ACTION=REVIEW_PRIOR_UPDATE")
    else:
        print("NEXT_ACTION=REVIEW_TIER")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
