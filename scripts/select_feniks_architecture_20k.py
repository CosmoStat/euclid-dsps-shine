#!/usr/bin/env python3
"""Aggregate the two-seed FENIKS architecture battle without using truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LABELS = (
    "current_mlp_realnvp",
    "set_transformer_realnvp",
    "set_transformer_autoregressive_spline",
)


def select(root: Path, seeds: tuple[int, ...]) -> dict[str, object]:
    candidates = []
    for label in LABELS:
        runs = []
        for seed in seeds:
            path = root / label / f"seed_{seed}" / "candidate_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        ess = np.asarray([run["median_raw_ess_fraction"] for run in runs])
        bad_k = np.asarray([run["fraction_pareto_k_gt_0p7"] for run in runs])
        objective = np.asarray([run["best_validation_objective"] for run in runs])
        candidates.append(
            {
                "candidate": label,
                "seeds": list(seeds),
                "all_seed_support_pass": all(
                    run["ordinary_iw_support"] == "PASS" for run in runs
                ),
                "median_seed_ess_fraction": float(np.median(ess)),
                "worst_seed_bad_pareto_fraction": float(np.max(bad_k)),
                "mean_best_validation_objective": float(np.mean(objective)),
                "runs": runs,
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            bool(item["all_seed_support_pass"]),
            float(item["median_seed_ess_fraction"]),
            -float(item["worst_seed_bad_pareto_fraction"]),
            -float(item["mean_best_validation_objective"]),
        ),
        reverse=True,
    )
    promoted = [item for item in ranked if item["all_seed_support_pass"]]
    payload = {
        "status": "complete",
        "selection_status": "PASS" if promoted else "DIAGNOSTIC_ONLY",
        "selected_architecture": promoted[0]["candidate"] if promoted else None,
        "diagnostic_leader": ranked[0]["candidate"],
        "ready_for_selected_feniks_adaptation": bool(promoted),
        "selection_rule": (
            "require ESS>=0.05 and bad Pareto-k fraction<=0.20 in both seeds; "
            "rank eligible candidates by median seed ESS, then Pareto tail and "
            "held-out training objective"
        ),
        "truth_used_for_training_or_selection": False,
        "candidates": candidates,
        "next_action": (
            "ADAPT_WINNER_TO_OBSERVED_R25_SELECTION"
            if promoted
            else "DO_NOT_PROMOTE_REVIEW_ARCHITECTURE_DIAGNOSTICS"
        ),
    }
    (root / "architecture_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "DONE").touch()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", default="260820,260821")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    print(json.dumps(select(args.root, seeds), indent=2), flush=True)


if __name__ == "__main__":
    main()
