#!/usr/bin/env python3
"""Finalize catalogue-wide calibration from frozen epoch-160 posterior banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXPECTED = {
    ("common32", "mira"): (32, ["raw_q", "raw_iw", "ema_q", "ema_iw"]),
    ("common32", "tarp"): (32, ["raw_q", "raw_iw", "ema_q", "ema_iw"]),
    ("q256", "mira"): (256, ["raw_q", "ema_q"]),
    ("q256", "tarp"): (256, ["raw_q", "ema_q"]),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def finalize(*, source_evaluation_root: Path, root: Path) -> dict[str, Any]:
    completion = root / "CATALOGUE_CALIBRATION_COMPLETE.json"
    if completion.is_file():
        return _read_json(completion)

    source_receipt_path = source_evaluation_root / "EPOCH160_EVALUATION_COMPLETE.json"
    source_receipt = _read_json(source_receipt_path)
    if (
        source_receipt.get("status") != "DIAGNOSTIC_COMPLETE"
        or int(source_receipt.get("epoch", -1)) != 160
        or source_receipt.get("training_frozen_before_truth") is not True
        or source_receipt.get("truth_used_for_training_or_checkpoint_selection")
        is not False
    ):
        raise ValueError("invalid source epoch-160 evaluation receipt")
    expected_objects = int(source_receipt["catalogue_objects"])

    summaries: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    object_counts = set()
    for (mode, diagnostic), (draws, models) in EXPECTED.items():
        out = root / mode / diagnostic
        if not (out / "DONE").is_file():
            raise FileNotFoundError(out / "DONE")
        summary_path = out / f"{diagnostic}_summary.json"
        summary = _read_json(summary_path)
        if (
            summary.get("status") != "complete"
            or int(summary.get("num_posterior_samples", -1)) != draws
            or summary.get("models") != models
        ):
            raise ValueError(f"invalid {mode}/{diagnostic} summary")
        objects = int(summary.get("num_objects", -1))
        if objects <= 0 or objects > expected_objects:
            raise ValueError(f"invalid object count for {mode}/{diagnostic}: {objects}")
        object_counts.add(objects)
        key = f"{mode}_{diagnostic}"
        summaries[key] = summary
        artifacts[key] = _file_record(summary_path)
        plot_name = "mira_scores.png" if diagnostic == "mira" else "tarp_coverage.png"
        artifacts[f"{key}_plot"] = _file_record(out / plot_name)
    if len(object_counts) != 1:
        raise ValueError(
            f"calibration tasks used different object counts: {object_counts}"
        )

    support = {
        variant: _read_json(
            source_evaluation_root / "heldout" / f"{variant}_support_summary.json"
        )
        for variant in ("raw", "ema")
    }
    iw_effective_samples = {
        variant: float(values["median_raw_ess"]) for variant, values in support.items()
    }
    payload = {
        "status": "DIAGNOSTIC_COMPLETE",
        "epoch": 160,
        "scientific_promotion": False,
        "training_frozen_before_truth": True,
        "truth_used_for_training_or_checkpoint_selection": False,
        "cohort": "all observed-selected rows in the independent test catalogue",
        "catalogue_objects_expected": expected_objects,
        "catalogue_objects_evaluated": object_counts.pop(),
        "evaluations": {
            "common32": {
                "models": ["raw_q", "raw_iw", "ema_q", "ema_iw"],
                "draws_per_object": 32,
                "shared_random_regions_or_references": True,
                "purpose": "matched nominal-budget q versus ordinary-IW comparison",
            },
            "q256": {
                "models": ["raw_q", "ema_q"],
                "draws_per_object": 256,
                "shared_random_regions_or_references": True,
                "purpose": "higher-resolution proposal calibration",
            },
        },
        "iw_support_warning": {
            "heldout_k": 1024,
            "median_effective_samples": iw_effective_samples,
            "interpretation": (
                "The 32 ordinary-IW resamples are nominal replicates, not 32 "
                "independent effective posterior draws. Interpret IW calibration "
                "with the K=1024 support diagnostics."
            ),
        },
        "contracts": {
            "parent_prior": "p_eta(theta | C0) from the learned prior flow",
            "selected_prior": "beta(theta) p_eta(theta | C0) / alpha_eta",
            "posterior_aggregate": (
                "object-equal mixture of dense per-object posterior draws; descriptive "
                "and not relabeled as either population prior"
            ),
            "truth_role": "post-freeze synthetic closure diagnostics only",
        },
        "source_epoch160_receipt": _file_record(source_receipt_path),
        "source_support": support,
        "summaries": summaries,
        "artifacts": artifacts,
    }
    temporary = completion.with_name(f".{completion.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, completion)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evaluation-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        source_evaluation_root=args.source_evaluation_root,
        root=args.root,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
