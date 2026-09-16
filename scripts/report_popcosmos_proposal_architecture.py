#!/usr/bin/env python3
"""Report progress and fail-closed decisions for posterior Phase 0/1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    root = parse_args().root
    summaries = sorted(root.glob("phase*/**/candidate_summary.json"))
    complete = sum((path.parent / "DONE").is_file() for path in summaries)
    print(f"architecture_candidates_complete={complete}/19")
    for path in summaries:
        if not (path.parent / "DONE").is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["ordinary_is"]["validation"]
        print(
            f"{payload['phase']} {payload['candidate']} seed={payload['seed']}: "
            f"IS={validation['support_status']} "
            f"ESS={validation['median_raw_ess_fraction']:.6f} "
            f"bad_k={validation['fraction_pareto_k_gt_0p7']:.6f}"
        )
    selection_path = root / "architecture_selection.json"
    if not selection_path.is_file():
        print("phase0=PENDING")
        print("phase1=PENDING")
        print("phase2=BLOCKED")
        print("NEXT_ACTION=WAIT_FOR_PHASE01")
        return
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    phase0 = selection["phase0"]
    phase1 = selection["phase1"]
    phase2 = selection["phase2_promotion"]
    print(f"phase0={phase0['diagnostic_status']}")
    print(f"phase1={phase1['selection_status']}")
    print(f"phase1_selected={phase1['selected_candidate']}")
    print(f"phase1_best_diagnostic={phase1['best_diagnostic_candidate']}")
    print(f"phase2={phase2['status']}")
    print(f"NEXT_ACTION={phase2['next_action']}")


if __name__ == "__main__":
    main()
