#!/usr/bin/env python3
"""Audit reproducibility of the published 12,014-object spectroscopic cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PUBLISHED_SOURCE_COUNTS = {
    "zCOSMOS": 6_567,
    "DEIMOS": 2_881,
    "C3R2": 1_849,
    "MUSE": 337,
    "FMOS": 303,
    "VVDS": 56,
    "VUDS": 21,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--popcosmos", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--prepared-full", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-published", type=int, default=12_014)
    parser.add_argument("--expected-xray", type=int, default=501)
    parser.add_argument("--expected-fallback", type=int, default=1_395)
    parser.add_argument("--require-exact-numeric", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def published_cohort(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the exact public ID/source contract used for the paper cohort."""
    cohort = summary.loc[
        (summary["MAGCUT_r"] == "Y") & (summary["z_SPEC"] == "Y")
    ].copy()
    cohort["INDEX_COSMOS"] = pd.to_numeric(
        cohort["INDEX_COSMOS"], errors="raise"
    ).astype(np.int64)
    if cohort["INDEX_COSMOS"].duplicated().any():
        raise ValueError("Published spectroscopic cohort has duplicate INDEX_COSMOS")
    return cohort.sort_values("INDEX_COSMOS").reset_index(drop=True)


def _numeric_truth_ids(path: Path) -> np.ndarray:
    columns = set(pq.ParquetFile(path).schema.names)
    if "object_id" not in columns or "redshift_true" not in columns:
        return np.empty(0, dtype=np.int64)
    frame = pd.read_parquet(path, columns=["object_id", "redshift_true"])
    truth = pd.to_numeric(frame["redshift_true"], errors="coerce").to_numpy(float)
    valid = np.isfinite(truth) & (truth >= 0.0)
    return frame.loc[valid, "object_id"].to_numpy(np.int64)


def audit_cohort(
    summary: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    expected_published: int,
    expected_xray: int,
    expected_fallback: int,
    prepared_truth_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cohort = published_cohort(summary)
    source_counts = cohort["z_SPECSOURCE"].value_counts().sort_index()
    source_rows = pd.DataFrame(
        [
            {
                "source": source,
                "observed": int(source_counts.get(source, 0)),
                "published": int(expected),
                "matches_paper": int(source_counts.get(source, 0)) == int(expected),
            }
            for source, expected in PUBLISHED_SOURCE_COUNTS.items()
        ]
    )
    n_xray = int((cohort["XRAY"] == "Y").sum())
    ids_exact = (
        len(cohort) == int(expected_published)
        and n_xray == int(expected_xray)
        and bool(source_rows["matches_paper"].all())
    )

    evaluation_truth_ids = _truth_ids_from_frame(evaluation)
    if len(evaluation_truth_ids) != int(expected_fallback):
        raise RuntimeError(
            f"Expected {expected_fallback} fallback objects, got {len(evaluation_truth_ids)}"
        )
    published_ids = cohort["INDEX_COSMOS"].to_numpy(np.int64)
    evaluation_in_published = np.intersect1d(
        evaluation_truth_ids, published_ids, assume_unique=False
    )
    prepared_truth_ids = (
        np.asarray(prepared_truth_ids, dtype=np.int64)
        if prepared_truth_ids is not None
        else np.empty(0, dtype=np.int64)
    )
    prepared_in_published = np.intersect1d(
        prepared_truth_ids, published_ids, assume_unique=False
    )
    exact_numeric_ready = len(prepared_in_published) == int(expected_published)
    payload = {
        "status": "complete",
        "published_cohort": {
            "selection": "MAGCUT_r == Y and z_SPEC == Y",
            "expected_rows": int(expected_published),
            "observed_rows": int(len(cohort)),
            "expected_xray_rows": int(expected_xray),
            "observed_xray_rows": n_xray,
            "id_and_source_contract_reproduced": bool(ids_exact),
            "numeric_redshift_column_in_public_summary": False,
            "z_SPEC_semantics": "Y/N availability flag; not a numeric redshift",
        },
        "numeric_truth": {
            "prepared_catalog_numeric_rows": int(len(prepared_truth_ids)),
            "prepared_overlap_with_published_ids": int(len(prepared_in_published)),
            "exact_12014_numeric_truth_ready": bool(exact_numeric_ready),
        },
        "fallback_benchmark": {
            "contract": "same 5,000 evaluation objects with public DR1.1 spectroscopy",
            "numeric_truth_rows": int(len(evaluation_truth_ids)),
            "overlap_with_published_id_cohort": int(len(evaluation_in_published)),
            "expected_rows": int(expected_fallback),
            "selected": not exact_numeric_ready,
        },
        "decision": (
            "exact_12014_numeric"
            if exact_numeric_ready
            else f"public_dr1p1_paired_{len(evaluation_truth_ids)}"
        ),
    }
    return cohort, source_rows, payload


def _truth_ids_from_frame(frame: pd.DataFrame) -> np.ndarray:
    required = {"object_id", "redshift_true"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Evaluation table lacks {sorted(required - set(frame.columns))}")
    truth = pd.to_numeric(frame["redshift_true"], errors="coerce").to_numpy(float)
    valid = np.isfinite(truth) & (truth >= 0.0)
    ids = frame.loc[valid, "object_id"].to_numpy(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Evaluation numeric-truth object IDs are not unique")
    return ids


def main() -> None:
    args = parse_args()
    for path in (args.popcosmos, args.evaluation):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = pd.read_csv(
        args.popcosmos,
        sep=r"\s+",
        usecols=[
            "INDEX_COSMOS",
            "MAGCUT_r",
            "XRAY",
            "z_SPEC",
            "z_SPECSOURCE",
        ],
        low_memory=False,
    )
    evaluation = pd.read_parquet(
        args.evaluation, columns=["object_id", "redshift_true"]
    )
    prepared_truth_ids = None
    if args.prepared_full is not None:
        if not args.prepared_full.is_file():
            raise FileNotFoundError(args.prepared_full)
        prepared_truth_ids = _numeric_truth_ids(args.prepared_full)
    cohort, source_rows, payload = audit_cohort(
        summary,
        evaluation,
        expected_published=args.expected_published,
        expected_xray=args.expected_xray,
        expected_fallback=args.expected_fallback,
        prepared_truth_ids=prepared_truth_ids,
    )
    payload["inputs"] = {
        "popcosmos": {"path": str(args.popcosmos), "sha256": _sha256(args.popcosmos)},
        "evaluation": {"path": str(args.evaluation), "sha256": _sha256(args.evaluation)},
    }
    if args.prepared_full is not None:
        payload["inputs"]["prepared_full"] = {
            "path": str(args.prepared_full),
            "sha256": _sha256(args.prepared_full),
        }

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out / "published_specz_cohort_ids.parquet", index=False)
    source_rows.to_csv(out / "published_specz_source_counts.csv", index=False)
    (out / "spectroscopy_cohort_audit.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (out / "DONE").touch()
    print(
        "[cosmos-specz-audit] "
        f"published_ids={len(cohort)} decision={payload['decision']} -> {out}"
    )
    if args.require_exact_numeric and not payload["numeric_truth"][
        "exact_12014_numeric_truth_ready"
    ]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
