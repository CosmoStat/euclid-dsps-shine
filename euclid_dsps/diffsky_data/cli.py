"""CLI helpers for Diffsky/OpenCosmo dataset investigation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .diagnostics import write_dataset_diagnostics
from .download import download_candidate_subset
from .inventory import inventory_local_hdf5, inventory_remote_listing
from .prepare import build_diffsky_photometric_dataset
from .remote_listing import crawl_remote_tree, write_remote_listing
from .urls import HLTDS_COSMOS_20260414
from .validation import validate_for_prior_learning, write_validation_report


def add_diffsky_subcommands(sub: argparse._SubParsersAction) -> None:
    listing = sub.add_parser("diffsky-list-remote", help="List remote Diffsky/OpenCosmo files.")
    listing.add_argument("--url", default=HLTDS_COSMOS_20260414)
    listing.add_argument("--max-depth", type=int, default=0)
    listing.add_argument("--out", default="outputs/diffsky_remote_listing.json")

    remote_inv = sub.add_parser("diffsky-inventory-remote", help="Rank remote Diffsky candidate files.")
    remote_inv.add_argument("--listing", required=True)
    remote_inv.add_argument("--out", default="outputs/diffsky_candidate_files.csv")

    download = sub.add_parser("diffsky-download-subset", help="Download a bounded Diffsky subset.")
    download.add_argument("--listing", required=True)
    download.add_argument("--out-dir", required=True)
    download.add_argument("--max-files", type=int, default=5)
    download.add_argument("--max-total-gb", type=float, default=5.0)
    download.add_argument("--include", action="append", default=[])
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--yes", action="store_true")

    local_inv = sub.add_parser("diffsky-inventory-local", help="Inspect local Diffsky HDF5 files.")
    local_inv.add_argument("--root", required=True)
    local_inv.add_argument("--out", default="outputs/diffsky_local_inventory.json")

    prepare = sub.add_parser("diffsky-prepare-dataset", help="Build normalized Diffsky photometry+truth parquet.")
    prepare.add_argument("--raw-root", required=True)
    prepare.add_argument("--inventory")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--max-objects", type=int)
    prepare.add_argument("--snr", type=float, default=50.0)

    diagnostics = sub.add_parser("diffsky-dataset-diagnostics", help="Write diagnostics for a prepared Diffsky parquet.")
    diagnostics.add_argument("--dataset", required=True)
    diagnostics.add_argument("--manifest")
    diagnostics.add_argument("--out", default="outputs/reports/diffsky_dataset")

    validate = sub.add_parser("diffsky-validate-dataset", help="Validate prepared Diffsky dataset for prior learning.")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--manifest")
    validate.add_argument("--out", default="outputs/reports/diffsky_dataset/prior_learning_validation_report.md")

    fit_report = sub.add_parser("diffsky-fit-report", help="Regenerate Diffsky MAP fit diagnostics from a run.")
    fit_report.add_argument("--run", type=Path, required=True)
    fit_report.add_argument("--config", type=Path)
    fit_report.add_argument("--label", default="batch_fit")
    fit_report.add_argument("--reporting-level", choices=("light", "full"), default="light")


def run_diffsky_command(args: argparse.Namespace) -> None:
    if args.command == "diffsky-list-remote":
        files = crawl_remote_tree(args.url, max_depth=int(args.max_depth))
        path = write_remote_listing(files, args.out)
        print(f"[diffsky] listed {len(files)} files -> {path}")
    elif args.command == "diffsky-inventory-remote":
        frame = inventory_remote_listing(args.listing, args.out)
        print(f"[diffsky] ranked {len(frame)} files -> {args.out}")
    elif args.command == "diffsky-download-subset":
        if not args.yes:
            raise SystemExit("Pass --yes to download files.")
        patterns = tuple(args.include or ["diffsky_gals", "param", "transmission", "yaml"])
        paths = download_candidate_subset(
            Path(args.listing),
            Path(args.out_dir),
            max_files=int(args.max_files),
            max_total_gb=float(args.max_total_gb),
            include_patterns=patterns,
            overwrite=bool(args.overwrite),
        )
        print(f"[diffsky] downloaded {len(paths)} files -> {args.out_dir}")
    elif args.command == "diffsky-inventory-local":
        inventory_local_hdf5(args.root, args.out)
        print(f"[diffsky] local inventory -> {args.out}")
    elif args.command == "diffsky-prepare-dataset":
        report = build_diffsky_photometric_dataset(
            raw_root=Path(args.raw_root),
            inventory_path=Path(args.inventory) if args.inventory else None,
            output_path=Path(args.out),
            max_objects=args.max_objects,
            snr=float(args.snr),
        )
        print(f"[diffsky] dataset -> {report.output_path}")
        print(f"[diffsky] readiness -> {report.readiness}")
    elif args.command == "diffsky-dataset-diagnostics":
        outputs = write_dataset_diagnostics(args.dataset, args.out)
        print(f"[diffsky] diagnostics -> {args.out} ({len(outputs)} files)")
    elif args.command == "diffsky-validate-dataset":
        report = validate_for_prior_learning(args.dataset, args.manifest)
        path = write_validation_report(report, args.out)
        print(f"[diffsky] validation {report['readiness']} -> {path}")
    elif args.command == "diffsky-fit-report":
        path = write_diffsky_fit_report(
            run_dir=Path(args.run),
            config_path=Path(args.config) if args.config else None,
            label=str(args.label),
            reporting_level=str(args.reporting_level),
        )
        print(f"[diffsky] fit report -> {path}")
    else:
        raise ValueError(f"Unsupported Diffsky command: {args.command}")


def write_diffsky_fit_report(
    *,
    run_dir: Path,
    config_path: Path | None,
    label: str,
    reporting_level: str,
) -> Path:
    """Regenerate aggregate tables and write a compact Diffsky recovery report."""
    from euclid_dsps.config import load_config
    from euclid_dsps.reporting import (
        write_batch_outputs,
        write_fit_diagnostic_outputs,
        write_trace_truth_outputs,
    )

    config = load_config(config_path) if config_path else _load_run_config(run_dir)
    fits = _read_run_table(run_dir, f"{label}_results")
    comparison = _read_run_table(run_dir, f"{label}_photometry_comparison")
    write_batch_outputs(
        comparison,
        run_dir,
        label=label,
        reporting_level=reporting_level,
        config=config,
    )
    write_fit_diagnostic_outputs(fits, comparison, config, run_dir, label=label)
    trace_path = _existing_table(run_dir, f"{label}_trace")
    if trace_path is not None:
        trace = _read_table(trace_path)
        write_trace_truth_outputs(
            trace,
            run_dir,
            label=label,
            make_plots=reporting_level == "full",
        )
    report_path = run_dir / f"{label}_diffsky_report.md"
    _write_fit_report_markdown(run_dir, label, report_path)
    return report_path


def _load_run_config(run_dir: Path) -> dict:
    normalized = run_dir / "normalized_config.json"
    if normalized.exists():
        return json.loads(normalized.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No --config was provided and {normalized} does not exist."
    )


def _read_run_table(run_dir: Path, stem: str) -> pd.DataFrame:
    path = _existing_table(run_dir, stem)
    if path is None:
        raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv in {run_dir}")
    return _read_table(path)


def _existing_table(run_dir: Path, stem: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        path = run_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_fit_report_markdown(run_dir: Path, label: str, path: Path) -> None:
    summary_path = run_dir / f"{label}_summary.json"
    truth_path = run_dir / f"{label}_truth_metrics.csv"
    band_path = run_dir / f"{label}_summary_by_band.csv"
    objective_path = run_dir / f"{label}_objective_components.csv"
    lines = [f"# Diffsky MAP Fit Report: `{run_dir.name}`", ""]
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.extend(["## Summary", ""])
        for key in sorted(summary):
            lines.append(f"- `{key}`: {summary[key]}")
        lines.append("")
    if truth_path.exists():
        truth = pd.read_csv(truth_path)
        lines.extend(["## Truth Recovery", "", _frame_to_markdown(truth), ""])
    if band_path.exists():
        by_band = pd.read_csv(band_path)
        cols = [
            col
            for col in (
                "band",
                "n",
                "median_residual_mag",
                "mean_residual_mag",
                "rms_residual_mag",
                "median_flux_ratio",
                "mean_photometric_objective_contribution",
            )
            if col in by_band
        ]
        lines.extend(["## Band Residuals", "", _frame_to_markdown(by_band[cols]), ""])
    if objective_path.exists():
        objective = pd.read_csv(objective_path)
        lines.extend(["## Objective", "", _frame_to_markdown(objective), ""])
    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `truth_kind=generated_truth` means the column is a Diffsky/Diffstar latent exported by the HLTDS mock.",
            "- `log10_stellar_metallicity` is fitted as a nuisance parameter in the forward model only when present in the config; no HLTDS metallicity truth was found in these shards.",
            "- Basic config fixes Diffmah and two Diffstar shape parameters to reduce degeneracy; use the extended config only as a stress test.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _frame_to_markdown(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)
