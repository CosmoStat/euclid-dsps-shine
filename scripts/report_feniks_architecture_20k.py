#!/usr/bin/env python3
"""Print a reconnectable compact report for the FENIKS architecture battle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    path = args.root / "architecture_selection.json"
    if not path.is_file():
        complete = len(list(args.root.glob("*/seed_*/DONE")))
        print(f"architecture_runs_complete={complete}/6")
        print("NEXT_ACTION=WAIT_FOR_ARCHITECTURE_BATTLE")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload["candidates"]:
        print(
            f"{item['candidate']}: all_seed_IS={item['all_seed_support_pass']} "
            f"ESS={item['median_seed_ess_fraction']:.6f} "
            f"worst_bad_k={item['worst_seed_bad_pareto_fraction']:.6f} "
            f"validation={item['mean_best_validation_objective']:.6f}"
        )
    print(f"selection={payload['selection_status']}")
    print(f"selected={payload['selected_architecture']}")
    print(f"diagnostic_leader={payload['diagnostic_leader']}")
    print(f"NEXT_ACTION={payload['next_action']}")


if __name__ == "__main__":
    main()
