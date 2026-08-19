#!/usr/bin/env python3
"""Report the independent prior confirmation and fast-encoder recovery chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    smc = _read(root / "pilot_selection/selection_summary.json")
    confirmation = _read(root / "prior_confirmation/prior_confirmation_summary.json")
    refresh = _read(root / "proposal_refresh_k2048/refresh_validation_summary.json")

    if smc is None:
        print("candidate_prior_smc=PENDING")
        print("NEXT_ACTION=WAIT_FOR_CANDIDATE_PRIOR_SMC")
        return
    print(f"candidate_prior_smc={smc.get('selection_status', 'UNKNOWN')}")
    print(f"selected_variant={smc.get('selected_variant')}")
    candidate = next(
        (
            item
            for item in smc.get("candidates", [])
            if item.get("variant") == smc.get("selected_variant")
        ),
        None,
    )
    if candidate:
        print(f"candidate_smc_support={candidate.get('support_pass')}")
        print(
            "candidate_smc_seed_logz_delta="
            f"{candidate.get('median_abs_logz_seed_delta'):.6f}"
        )
    if smc.get("selection_status") != "PASS":
        print("NEXT_ACTION=STOP_CANDIDATE_PRIOR_SMC_FAILED")
        return

    if confirmation is None:
        failed = root / "prior_confirmation/FAILED"
        print(
            "prior_confirmation=FAILED"
            if failed.exists()
            else "prior_confirmation=PENDING"
        )
        print(
            "NEXT_ACTION=STOP_UPDATED_PRIOR_CONFIRMATION_FAILED"
            if failed.exists()
            else "NEXT_ACTION=WAIT_FOR_UPDATED_PRIOR_CONFIRMATION"
        )
        return
    gate = confirmation["validation_gate"]
    print(f"prior_confirmation={confirmation['confirmation_status']}")
    print(f"fresh_confirmation_objects={confirmation['n_objects']}")
    print(f"mean_fresh_logz_delta={gate['mean_log_evidence_delta']:.6f}")
    print(
        "median_reverse_ratio_ess_fraction="
        f"{gate['median_prior_ratio_ess_fraction']:.6f}"
    )
    for seed, value in gate["seed_mean_log_evidence_delta"].items():
        print(f"{seed}_fresh_mean_logz_delta={value:.6f}")
    if confirmation["confirmation_status"] != "PASS":
        print("NEXT_ACTION=STOP_UPDATED_PRIOR_CONFIRMATION_FAILED")
        return

    if refresh is None:
        print("encoder_refresh=PENDING")
        print("ordinary_is=PENDING")
        print("NEXT_ACTION=WAIT_FOR_FAST_ENCODER_RECOVERY")
        return
    print(f"encoder_refresh={refresh['encoder_refresh_gate']}")
    print(f"ordinary_is={refresh['ordinary_importance_support_gate']}")
    metrics = refresh["candidate_metrics"]
    print(f"median_raw_ess_fraction={metrics['median_raw_ess_fraction']:.6f}")
    print(f"fraction_pareto_k_gt_0p7={metrics['fraction_pareto_k_gt_0p7']:.6f}")
    if refresh["encoder_refresh_gate"] != "PASS":
        print("NEXT_ACTION=STOP_ENCODER_REFRESH_FAILED")
    elif refresh["ordinary_importance_support_gate"] != "PASS":
        print("NEXT_ACTION=STOP_FAST_ENCODER_SUPPORT_FAILED")
    else:
        print(
            "NEXT_ACTION=EVALUATE_FAST_POSTERIOR_CALIBRATION_ON_FROZEN_SPECTROSCOPIC_COHORT"
        )


if __name__ == "__main__":
    main()
