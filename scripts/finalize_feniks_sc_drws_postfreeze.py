#!/usr/bin/env python3
"""Finalize the independent SC-DRWS post-freeze evaluation chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def finalize(
    *, recovery_root: Path, closure_root: Path, inference_root: Path
) -> dict[str, Any]:
    closure_path = closure_root / "POSTFREEZE_COMPLETE.json"
    inference_path = inference_root / "inference_receipt.json"
    closure = _read(closure_path)
    inference = _read(inference_path)
    if closure.get("status") != "DIAGNOSTIC_COMPLETE":
        raise ValueError("post-freeze closure is incomplete")
    if inference.get("status") not in {"PASS", "DIAGNOSTIC_COMPLETE"}:
        raise ValueError("four-shard catalogue inference is incomplete")
    if closure.get("truth_used_for_training_or_checkpoint_selection") is not False:
        raise ValueError("closure receipt violates the no-truth training contract")
    if inference.get("truth_used") is not False:
        raise ValueError("catalogue inference unexpectedly used truth")
    diagnostic = bool(inference.get("diagnostic_only"))
    payload = {
        "status": "DIAGNOSTIC_COMPLETE" if diagnostic else "PASS",
        "workflow": "SC-DRWS post-freeze posterior and population evaluation",
        "scientific_promotion": not diagnostic,
        "truth_used_for_training_or_checkpoint_selection": False,
        "posterior_catalogue_inference": inference,
        "truth_closure": closure,
        "artifacts": {
            "catalogue_inference_receipt": _record(inference_path),
            "postfreeze_closure_receipt": _record(closure_path),
        },
    }
    output = recovery_root / "SC_DRWS_POSTFREEZE_RECEIPT.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    args = parser.parse_args()
    finalize(**vars(args))


if __name__ == "__main__":
    main()
