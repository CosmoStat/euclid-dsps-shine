#!/usr/bin/env python3
"""Read full-decoder qualification with final artifact integrity checks."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def summarize(root):
    root = Path(root)
    final = json.loads((root / "FINAL.json").read_text())
    if final["status"] != "FULL_DECODER_QUALIFICATION_COMPLETE":
        raise ValueError("full decoder qualification not complete")
    for name, item in final["artifacts"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"changed artifact: {name}")
    result = json.loads((root / "FULL_DECODER_QUALIFICATION.json").read_text())
    print(result["status"], "candidate=", result["candidate_numerical_checks"])
    print(
        "next=",
        result["next_stage"],
        "prior unchanged=",
        result["prior_bitwise_unchanged"],
    )
    print("variant point checks max_delta_flux/sigma steady_forward_s")
    for case in result["cases"]:
        print(
            case["variant"],
            case["point_index"],
            case["numerical_checks"],
            case["max_abs_forward_delta_sigma"],
            case["steady_forward_seconds"],
        )
        if case["variant"] == "merged":
            bad = Counter(
                c["coordinate"] + ":" + c["status"]
                for c in case["checks"]
                if c["status"] != "PASS" and c["component"] != "canonical_loglike"
            )
            print("  unresolved/failing components by coordinate:", dict(bad))
            print(
                "  failing gradient identities:",
                [
                    c
                    for c in case["canonical_centered_gradient"]
                    if not c["passed"] or not c["reverse_passed"]
                ],
            )
            print(
                "  dtypes:",
                case["floating_trace_dtypes"],
                "memory:",
                case["device_memory"],
            )
    print(
        "Catalogue simulator compatibility NOT VERIFIED; no NPE/population promotion."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
