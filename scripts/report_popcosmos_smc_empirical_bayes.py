#!/usr/bin/env python3
"""Print the fail-closed decision from a direct SMC empirical-Bayes pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.root / "smc_empirical_bayes_summary.json"
    if not summary_path.is_file():
        print("smc_empirical_bayes=PENDING")
        print("NEXT_ACTION=WAIT_FOR_DIRECT_SMC_MSTEP")
        return
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = payload["validation_gate"]
    print(f"status={payload['status']}")
    print(f"population_contract={payload['population_contract']}")
    print(f"objects={payload['n_objects']}")
    print(f"train_objects={payload['n_train_objects']}")
    print(f"validation_objects={payload['n_validation_objects']}")
    print(f"selection={payload['selection_status']}")
    print(f"selected_candidate={payload['selected_candidate']}")
    print(f"mean_validation_logz_delta={gate['mean_log_evidence_delta']:.6f}")
    print(
        f"median_prior_ratio_ess_fraction={gate['median_prior_ratio_ess_fraction']:.6f}"
    )
    print(
        "fraction_ratio_ess_ge_0p2="
        f"{gate['fraction_objects_prior_ratio_ess_ge_0p2']:.6f}"
    )
    for seed, value in gate["seed_mean_log_evidence_delta"].items():
        print(f"{seed}_mean_logz_delta={value:.6f}")
    print(f"NEXT_ACTION={payload['next_action']}")


if __name__ == "__main__":
    main()
