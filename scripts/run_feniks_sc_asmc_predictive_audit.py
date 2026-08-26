#!/usr/bin/env python3
"""Run the full-catalogue post-freeze FENIKS photometric audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_dsps.amortized.sc_asmc_predictive_audit import (  # noqa: E402
    run_full_catalogue_predictive_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--truth-config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--posterior-draws", type=int, default=16)
    parser.add_argument("--decoder-pairs-per-batch", type=int, default=128)
    args = parser.parse_args()
    result = run_full_catalogue_predictive_audit(
        training_config_path=args.config,
        truth_config_path=args.truth_config,
        run_root=args.run_root,
        closure_root=args.closure_root,
        out_dir=args.out,
        posterior_draws=args.posterior_draws,
        decoder_pairs_per_batch=args.decoder_pairs_per_batch,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
