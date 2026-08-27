#!/usr/bin/env python3
"""Train truth-free selection-corrected defensive RWS on FENIKS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from euclid_dsps.amortized.sc_drws_trainer import train_feniks_sc_drws
from euclid_dsps.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-indices-file", type=Path, required=True)
    parser.add_argument("--validation-indices-file", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path, required=True)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-full-dataset", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["amortized"]["training"]["seed"] = int(args.seed)
        config["amortized"]["objective"]["selection_correction"]["seed"] = int(
            args.seed
        )
    receipt = train_feniks_sc_drws(
        config,
        out_dir=args.out,
        train_indices_file=args.train_indices_file,
        validation_indices_file=args.validation_indices_file,
        manifest_file=args.manifest_file,
        resume_state=args.resume_state,
        smoke=args.smoke,
        require_full_dataset=args.require_full_dataset,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
