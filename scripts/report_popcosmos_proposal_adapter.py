#!/usr/bin/env python3
"""Report progress and decisions for the warm-start adapter benchmark."""

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
    summaries = sorted(root.glob("*/seed_*/adapter_summary.json"))
    complete = sum((path.parent / "DONE").is_file() for path in summaries)
    print(f"adapter_candidates_complete={complete}/12")
    for path in summaries:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload["ordinary_is"]["validation"]
        print(
            f"{payload['candidate']} seed={payload['seed']}: "
            f"IS={validation['support_status']} "
            f"ESS={validation['median_raw_ess_fraction']:.6f} "
            f"bad_k={validation['fraction_pareto_k_gt_0p7']:.6f} "
            f"NLL={payload['validation_weighted_smc_b_nll']:.4f}"
        )
    selection_path = root / "adapter_selection.json"
    if not selection_path.is_file():
        print("adapter_diagnosis=PENDING")
        print("NEXT_ACTION=WAIT_FOR_ADAPTER_BENCHMARK")
        return
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    print(f"adapter_diagnosis={selection['diagnosis']}")
    print(f"recommended_representation={selection['recommended_representation']}")
    print(f"ready_for_production={selection['ready_for_production']}")
    print(f"NEXT_ACTION={selection['next_action']}")


if __name__ == "__main__":
    main()
