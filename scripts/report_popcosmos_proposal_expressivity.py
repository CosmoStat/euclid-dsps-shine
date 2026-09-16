#!/usr/bin/env python3
"""Print compact progress or final decisions for proposal diagnostics."""

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
    selection = root / "pilot_selection/selection_summary.json"
    result = root / "proposal_expressivity/proposal_expressivity_summary.json"
    if not selection.is_file():
        done = len(list((root / "floor_0p05").glob("seed_*/shard_*/DONE")))
        print(f"diagnostic_smc=PENDING shards_done={done}")
        print("NEXT_ACTION=WAIT_FOR_DIAGNOSTIC_SMC")
        return
    smc = json.loads(selection.read_text(encoding="utf-8"))
    gates = [candidate["support_pass"] for candidate in smc.get("candidates", [])]
    print(f"diagnostic_smc_support={'PASS' if gates and all(gates) else 'FAIL'}")
    if not result.is_file():
        print("support_diagnosis=PENDING")
        print("expressivity_evidence=PENDING")
        print("NEXT_ACTION=WAIT_FOR_PROPOSAL_EXPRESSIVITY")
        return
    summary = json.loads(result.read_text(encoding="utf-8"))
    diagnosis = summary["support_diagnosis"]
    evidence = summary["expressivity_evidence"]
    print(f"support_diagnosis={diagnosis['evidence_status']}")
    print(f"expressivity_evidence={evidence['evidence_status']}")
    for name, payload in summary["ordinary_is"].items():
        validation = payload["validation"]
        print(
            f"{name}: validation_IS={validation['support_status']} "
            f"ESS={validation['median_raw_ess_fraction']:.6f} "
            f"bad_k={validation['fraction_pareto_k_gt_0p7']:.6f}"
        )
    print(f"NEXT_ACTION={evidence['next_action']}")


if __name__ == "__main__":
    main()
