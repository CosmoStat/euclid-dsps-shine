"""Standalone OpenUniverse utility CLI.

This module intentionally stays separate from ``euclid_dsps.cli`` so the
OpenUniverse truth/SED tools can evolve without perturbing the legacy FS2 CLI.
Run with ``python -m euclid_dsps.openuniverse.cli ...``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.io import ensure_dir

from .arrays import load_openuniverse_photometry_arrays
from .diagnostics import compute_photoz_metrics, compute_prior_overlap_metrics
from .inventory import (
    inventory_openuniverse_truth_fields,
    write_basic_truth_artifacts,
)
from .schema import OU_LSST_ROMAN_14_BANDS


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "inventory-truth":
        _run_inventory_truth(args)
        return
    if args.command == "extract-truth":
        _run_extract_truth(args)
        return
    if args.command == "photoz-metrics":
        _run_photoz_metrics(args)
        return
    if args.command == "prior-overlap":
        _run_prior_overlap(args)
        return
    if args.command == "feature-stats":
        _run_feature_stats(args)
        return
    parser.error("No OpenUniverse subcommand was provided")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m euclid_dsps.openuniverse.cli",
        description="OpenUniverse truth, SED, and diagnostic utility commands.",
    )
    sub = parser.add_subparsers(dest="command")

    inventory = sub.add_parser(
        "inventory-truth",
        help="Inventory OpenUniverse truth-like fields and optional SED files.",
    )
    inventory.add_argument("--input", type=Path, help="Prepared OpenUniverse parquet")
    inventory.add_argument("--input-root", type=Path, help="Raw OpenUniverse root")
    inventory.add_argument("--hpix", nargs="*", type=int, default=[])
    inventory.add_argument("--sed", action="store_true", help="Inspect SED HDF5 files")
    inventory.add_argument("--sed-sample-limit", type=int, default=5)
    inventory.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reports/openuniverse_truth_inventory"),
    )

    extract = sub.add_parser(
        "extract-truth",
        help="Extract direct public OpenUniverse truth columns from a prepared parquet.",
    )
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--schema-out", type=Path)
    extract.add_argument("--basic-only", action="store_true", default=True)
    extract.add_argument(
        "--extended-diffsky",
        action="store_true",
        help="Reserved flag; full Diffsky latent export is not wired yet.",
    )

    photoz = sub.add_parser(
        "photoz-metrics",
        help="Compute photo-z metrics from a toy or real posterior sample table.",
    )
    photoz.add_argument("--samples", type=Path, required=True)
    photoz.add_argument("--truth", type=Path, required=True)
    photoz.add_argument("--truth-column", default="redshift_truth")
    photoz.add_argument(
        "--sample-prefix",
        default="z_sample_",
        help="Prefix for wide posterior sample columns.",
    )
    photoz.add_argument("--out", type=Path, required=True)

    overlap = sub.add_parser(
        "prior-overlap",
        help="Compute 1D truth/posterior/prior overlap metrics from parquet columns.",
    )
    overlap.add_argument("--truth", type=Path, required=True)
    overlap.add_argument("--posterior", type=Path, required=True)
    overlap.add_argument("--prior", type=Path, required=True)
    overlap.add_argument("--truth-column", required=True)
    overlap.add_argument("--posterior-column", required=True)
    overlap.add_argument("--prior-column", required=True)
    overlap.add_argument("--name", default="parameter")
    overlap.add_argument("--out", type=Path, required=True)

    features = sub.add_parser(
        "feature-stats",
        help="Compute amortized encoder feature stats for a prepared OU parquet.",
    )
    features.add_argument("--input", type=Path, required=True)
    features.add_argument("--out", type=Path, required=True)
    features.add_argument("--limit", type=int)
    features.add_argument("--flux-transform", default="asinh")
    return parser


def _run_inventory_truth(args: argparse.Namespace) -> None:
    payload = inventory_openuniverse_truth_fields(
        output_dir=args.out,
        processed_path=args.input,
        input_root=args.input_root,
        hpix_ids=args.hpix,
        include_sed=bool(args.sed),
        sed_sample_limit=int(args.sed_sample_limit),
    )
    print(
        "[openuniverse] inventory -> "
        f"{args.out / 'openuniverse_truth_inventory.md'} "
        f"({len(payload['physical_truth_summary'])} truth rows)"
    )


def _run_extract_truth(args: argparse.Namespace) -> None:
    if args.extended_diffsky:
        raise NotImplementedError(
            "Extended Diffsky latent truth export is not wired yet. Use "
            "`extract-truth --basic-only` for direct public OpenUniverse truths."
        )
    payload = write_basic_truth_artifacts(
        input_path=args.input,
        output_path=args.out,
        schema_path=args.schema_out,
    )
    print(
        "[openuniverse] basic truth -> "
        f"{args.out} ({payload['number_of_rows']} rows)"
    )
    if args.schema_out is not None:
        print(f"[openuniverse] schema -> {args.schema_out}")


def _run_photoz_metrics(args: argparse.Namespace) -> None:
    sample_frame = pd.read_parquet(args.samples)
    truth_frame = pd.read_parquet(args.truth)
    z_samples = _wide_samples_to_array(sample_frame, str(args.sample_prefix))
    if args.truth_column not in truth_frame:
        raise ValueError(f"Missing truth column {args.truth_column!r}")
    metrics = compute_photoz_metrics(
        z_samples,
        truth_frame[args.truth_column].to_numpy(dtype=float),
    )
    out = Path(args.out)
    ensure_dir(out.parent)
    pd.DataFrame([metrics]).to_csv(out, index=False)
    print(f"[openuniverse] photo-z metrics -> {out}")


def _run_prior_overlap(args: argparse.Namespace) -> None:
    truth = pd.read_parquet(args.truth)
    posterior = pd.read_parquet(args.posterior)
    prior = pd.read_parquet(args.prior)
    for frame, column, role in (
        (truth, args.truth_column, "truth"),
        (posterior, args.posterior_column, "posterior"),
        (prior, args.prior_column, "prior"),
    ):
        if column not in frame:
            raise ValueError(f"Missing {role} column {column!r}")
    metrics = compute_prior_overlap_metrics(
        truth[args.truth_column].to_numpy(dtype=float),
        posterior[args.posterior_column].to_numpy(dtype=float),
        prior[args.prior_column].to_numpy(dtype=float),
        name=str(args.name),
    )
    out = Path(args.out)
    ensure_dir(out.parent)
    if out.suffix.lower() == ".json":
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    else:
        pd.DataFrame([metrics]).to_csv(out, index=False)
    print(f"[openuniverse] prior overlap metrics -> {out}")


def _run_feature_stats(args: argparse.Namespace) -> None:
    from euclid_dsps.amortized.features import (
        compute_feature_stats,
        make_encoder_features,
        write_feature_stats,
    )

    arrays = load_openuniverse_photometry_arrays(
        args.input,
        band_names=OU_LSST_ROMAN_14_BANDS,
        limit=args.limit,
    )
    stats = compute_feature_stats(
        arrays.flux,
        arrays.flux_err,
        arrays.mask,
        band_names=arrays.band_names,
        flux_transform=str(args.flux_transform),
    )
    out = Path(args.out)
    write_feature_stats(out, stats)
    # Materialize a tiny preview so shape errors are caught before training.
    preview_rows = min(len(arrays.flux), 8)
    preview = make_encoder_features(
        arrays.flux[:preview_rows],
        arrays.flux_err[:preview_rows],
        stats,
    )
    summary = {
        "input": str(args.input),
        "output": str(out),
        "n_objects": int(arrays.flux.shape[0]),
        "n_bands": int(arrays.flux.shape[1]),
        "feature_dim": int(preview.shape[-1]),
        "band_names": list(arrays.band_names),
        "flux_transform": str(stats.flux_transform),
        "truth_columns": [] if arrays.truth is None else sorted(arrays.truth),
    }
    summary_path = out.with_suffix(".summary.json")
    ensure_dir(summary_path.parent)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[openuniverse] feature stats -> "
        f"{out} (n_bands={summary['n_bands']}, feature_dim={summary['feature_dim']})"
    )


def _wide_samples_to_array(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    sample_columns = [column for column in frame.columns if str(column).startswith(prefix)]
    if sample_columns:
        values = frame[sample_columns].to_numpy(dtype=float)
        return values.T
    if "object_index" in frame and "sample_index" in frame and "redshift" in frame:
        pivot = frame.pivot(index="sample_index", columns="object_index", values="redshift")
        return pivot.sort_index(axis=0).sort_index(axis=1).to_numpy(dtype=float)
    raise ValueError(
        "Could not infer redshift posterior samples. Expected wide columns with "
        f"prefix {prefix!r}, or long columns object_index/sample_index/redshift."
    )


if __name__ == "__main__":
    main()
