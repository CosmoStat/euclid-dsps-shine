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
from .filter_curves import parse_filter_path_overrides
from .flux_closure import run_sed_flux_closure, write_sed_flux_closure_outputs
from .inventory import (
    inventory_openuniverse_truth_fields,
    write_basic_truth_artifacts,
)
from .schema import OU_LSST_ROMAN_14_BANDS
from .truth_merge import merge_external_truth_table


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
    if args.command == "sed-flux-closure":
        _run_sed_flux_closure(args)
        return
    if args.command == "merge-external-truth":
        _run_merge_external_truth(args)
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

    closure = sub.add_parser(
        "sed-flux-closure",
        help="Compare OU SED-integrated photon rates to prepared truth fluxes.",
    )
    closure.add_argument("--catalog", type=Path, required=True)
    closure.add_argument("--sed", type=Path, required=True)
    closure.add_argument("--out", type=Path, required=True)
    closure.add_argument("--bands", nargs="*", default=list(OU_LSST_ROMAN_14_BANDS))
    closure.add_argument("--filter-root", type=Path, default=Path("filters"))
    closure.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Exact filter override of the form band=/path/to/filter.dat",
    )
    closure.add_argument(
        "--allow-approx-filters",
        action="store_true",
        help="Allow coarse top-hat Roman filters for smoke tests only.",
    )
    closure.add_argument("--limit", type=int)
    closure.add_argument(
        "--sed-fnu-unit",
        default="native",
        choices=["native", "fnu_cgs", "jy", "microjy", "nanojy"],
    )
    closure.add_argument("--sed-fnu-scale", type=float, default=1.0)
    closure.add_argument(
        "--sed-wavelength-frame",
        default="rest",
        choices=["rest", "observer"],
        help="Interpret HDF5 wave_list as rest-frame or observer-frame wavelengths.",
    )
    closure.add_argument("--no-calibrate", action="store_true")

    merge_truth = sub.add_parser(
        "merge-external-truth",
        help="Merge a real externally exported Diffsky/Diffstar truth table.",
    )
    merge_truth.add_argument("--input", type=Path, required=True)
    merge_truth.add_argument("--truth", type=Path, required=True)
    merge_truth.add_argument("--out", type=Path, required=True)
    merge_truth.add_argument("--schema-out", type=Path)
    merge_truth.add_argument("--id-column", default="galaxy_id")
    merge_truth.add_argument(
        "--truth-column",
        action="append",
        default=[],
        help="Column to merge; repeatable. Defaults to all non-id columns.",
    )
    merge_truth.add_argument("--prefix", default="generated_truth_")
    merge_truth.add_argument(
        "--truth-level",
        default="generated_truth",
        choices=["truth", "generated_truth", "proxy"],
    )
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


def _run_sed_flux_closure(args: argparse.Namespace) -> None:
    filter_paths = parse_filter_path_overrides(args.filter)
    result = run_sed_flux_closure(
        catalog_path=args.catalog,
        sed_path=args.sed,
        band_names=tuple(args.bands),
        filter_root=args.filter_root,
        filter_paths=filter_paths,
        allow_approx_filters=bool(args.allow_approx_filters),
        limit=args.limit,
        sed_fnu_unit=args.sed_fnu_unit,
        sed_fnu_scale=float(args.sed_fnu_scale),
        sed_wavelength_frame=args.sed_wavelength_frame,
        calibrate=not bool(args.no_calibrate),
    )
    paths = write_sed_flux_closure_outputs(result, args.out)
    print(
        "[openuniverse] SED flux closure -> "
        f"{paths['metrics']} ({result.summary['n_objects']} objects, "
        f"{len(result.summary['bands'])} bands)"
    )
    if result.summary["uses_approximate_filters"]:
        print(
            "[openuniverse] warning: approximate top-hat filters were used; "
            "do not treat Roman closure metrics as science-grade."
        )


def _run_merge_external_truth(args: argparse.Namespace) -> None:
    payload = merge_external_truth_table(
        input_path=args.input,
        truth_path=args.truth,
        output_path=args.out,
        schema_path=args.schema_out,
        id_column=str(args.id_column),
        truth_columns=tuple(args.truth_column) if args.truth_column else None,
        prefix=str(args.prefix),
        truth_level=args.truth_level,
    )
    print(
        "[openuniverse] merged external truth -> "
        f"{args.out} ({payload['n_output_rows']} rows, "
        f"{len(payload['exported_columns'])} truth columns)"
    )
    if args.schema_out is not None:
        print(f"[openuniverse] external truth schema -> {args.schema_out}")


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
