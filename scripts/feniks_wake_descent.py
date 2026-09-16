"""Prepare a versioned descent-controlled comparison from the completed pilot."""

import argparse
from pathlib import Path

from scripts.feniks_objective_night import read, write
from scripts.feniks_wake_forensics import prepare as prepare_reference


def prepare(pilot, root, holdout=False):
    prepare_reference(pilot, root)
    manifest = read(root / "RUN_MANIFEST.json")
    manifest["wake_descent_reference"] = manifest.pop("wake_forensic_reference")
    manifest.pop("forensic_parameter_atol")
    manifest["wake_backtracking"] = dict(armijo=1e-4, trials=12)
    manifest["adaptation_contract"] = "wake_armijo_v1"
    if holdout:
        manifest["wake_holdout"] = dict(
            version="independent_fixed_mixture_v1",
            draws=256,
            replicates=2,
            seed_tag=918273,
            minimum_ess=16,
            maximum_weight=0.2,
            used_for_acceptance=False,
        )
    manifest["interpretation"] = (
        "Same pilot seeds and budgets; wake Armijo descent control, original reverse control. Independent prescribed evaluations; no promotion."
    )
    write(root / "RUN_MANIFEST.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    prepare(args.pilot, args.root, holdout=args.holdout)
