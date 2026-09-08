#!/usr/bin/env python3
"""Check hashes and replay point-4 branch stencils on CPU."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_dsps.amortized.redshift_precision import analyze


def summarize(root):
    root = Path(root)
    final = json.loads((root / "FINAL.json").read_text())
    if final["status"] != "REDSHIFT_PRECISION_COMPLETE":
        raise ValueError("redshift precision audit incomplete")
    for name, item in final["artifacts"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"changed artifact: {name}")
    stored = json.loads((root / "REDSHIFT_PRECISION.json").read_text())
    snapshot = json.loads((root / "REDSHIFT_PRECISION_SNAPSHOT.json").read_text())
    replay, _ = analyze(snapshot)
    if replay["branches"] != stored["branches"]:
        raise ValueError("CPU replay differs from saved checks")
    print(stored["status"], "CPU replay=PASS")
    for b in stored["branches"]:
        c = next(c for c in b["checks"] if c["component"] == "lsst_z")
        print(
            b["name"],
            "lsst_z",
            c["status"],
            "AD",
            c["ad"],
            "FD",
            c["fd"],
            "all_checks",
            b["all_checks_passed"],
            "center_delta_sigma",
            b["max_center_delta_sigma"],
            "dtypes",
            b["floating_dtypes"],
        )
        print(
            "  unresolved:",
            [
                (c["component"], c["status"])
                for c in b["checks"]
                if c["status"] != "PASS"
            ],
        )
    print(
        "branch-sum minus full AD (sigma/x):", stored["chain_minus_full_ad_sigma_per_x"]
    )
    print(stored["precision_scope"])
    print("No production decoder, bank, NPE or population promotion.")
    return stored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
