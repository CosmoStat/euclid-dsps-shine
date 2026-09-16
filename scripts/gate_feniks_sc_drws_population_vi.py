#!/usr/bin/env python3
"""Fail-closed gate before the observed-catalogue population VI M-step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from euclid_dsps.amortized.population_vi import require_population_vi_gate
from euclid_dsps.io import write_json


def gate(*, validation: Path, out: Path) -> dict:
    receipt = json.loads(validation.read_text(encoding="utf-8"))
    try:
        payload = require_population_vi_gate(receipt)
    except ValueError as exc:
        payload = {
            "status": "BLOCKED",
            "truth_used": False,
            "reason": str(exc),
            "validation_receipt": str(validation.resolve()),
            "scientific_promotion": False,
        }
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "POPULATION_VI_BLOCKED.json", payload)
        return payload
    payload["validation_receipt"] = str(validation.resolve())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "POPULATION_VI_READY.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = gate(**vars(args))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
