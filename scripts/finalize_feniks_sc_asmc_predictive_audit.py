#!/usr/bin/env python3
"""Finalize a decoded FENIKS predictive audit without rerunning DSPS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_dsps.amortized.sc_asmc_predictive_audit import (  # noqa: E402
    finalize_existing_predictive_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_existing_predictive_audit(
        out_dir=args.audit_root,
        closure_root=args.closure_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
