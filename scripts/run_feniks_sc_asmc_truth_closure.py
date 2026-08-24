#!/usr/bin/env python3
"""Run dense-draw FENIKS truth closure after SC-ASMC-EM is frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_dsps.amortized.sc_asmc_postfreeze import (  # noqa: E402
    run_sc_asmc_truth_closure,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="No-truth training config."
    )
    parser.add_argument("--truth-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-object", type=int, default=128)
    parser.add_argument("--num-mira-regions", type=int, default=100)
    parser.add_argument("--num-bootstrap", type=int, default=1000)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--seed", type=int, default=260824)
    args = parser.parse_args()
    result = run_sc_asmc_truth_closure(
        training_config_path=args.config,
        truth_config_path=args.truth_config,
        run_root=args.run_root,
        out_dir=args.out,
        samples_per_object=args.samples_per_object,
        num_mira_regions=args.num_mira_regions,
        num_bootstrap=args.num_bootstrap,
        evaluation_limit=args.evaluation_limit,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
