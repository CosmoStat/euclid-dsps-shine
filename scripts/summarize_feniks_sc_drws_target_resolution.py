#!/usr/bin/env python3
"""Verify and replay the residual audit without cluster checkpoints or SSPs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euclid_dsps.amortized.target_resolution import analyze


def summarize(root):
    root = Path(root)
    final = json.loads((root / "FINAL.json").read_text())
    if final["status"] != "TARGET_RESOLUTION_COMPLETE":
        raise ValueError("target resolution audit is incomplete")
    for name, item in final["artifacts"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"changed artifact: {name}")
    stored = json.loads((root / "TARGET_RESOLUTION.json").read_text())
    snapshot = json.loads((root / "TARGET_RESOLUTION_SNAPSHOT.json").read_text())
    replay, _ = analyze(snapshot)
    if replay["checks"] != stored["checks"]:
        raise ValueError("CPU replay differs from saved checks")
    print(stored["status"], "CPU replay=PASS")
    print("All unresolved checks resolved:", stored["unresolved_checks_resolved"])
    for c in stored["checks"]:
        print(
            "point",
            c["point_index"],
            c["coordinate"],
            c["component"],
            c["status"],
            "AD=",
            c["ad"],
            "FD=",
            c["fd"],
            "step_index=",
            c["selected_step_index"],
        )
    print("next:", stored["next_stage"])
    print("No posterior, NPE or population promotion; previous reports unchanged.")
    return stored


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
