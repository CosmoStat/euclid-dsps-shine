#!/usr/bin/env python3
"""Validate the immutable four-H100 gate before SC-ASMC-EM production."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_dsps.amortized.sc_asmc_validate import (  # noqa: E402
    validate_production_smoke_gate,
)
from euclid_dsps.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["catalog_path"] = str(args.catalog.resolve())
    receipt = validate_production_smoke_gate(
        args.smoke_root,
        config,
        args.catalog,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
