#!/usr/bin/env python
"""Finalize a batch MAP run from per-chunk checkpoints."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate batch fit checkpoints and regenerate aggregate reports. "
            "This is useful when a long run completed most chunks but crashed "
            "before the final write step."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Config used by the fit run.")
    parser.add_argument("--run", required=True, help="Run directory containing _chunks.")
    parser.add_argument(
        "--expected-limit",
        type=int,
        default=None,
        help="Expected contiguous catalog limit, used only to report missing rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Original batch size, recorded in the completion summary.",
    )
    parser.add_argument(
        "--reporting-level",
        choices=["full", "light", "none"],
        default=None,
        help="Override config reporting.level while regenerating plots/tables.",
    )
    parser.add_argument(
        "--jax-platforms",
        default="cpu",
        help="Platform forced before importing reporting modules; CPU is enough here.",
    )
    parser.add_argument(
        "--include-nebular",
        action="store_true",
        help="Load the DSPS context and regenerate nebular diagnostics too.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jax_platforms:
        os.environ["JAX_PLATFORMS"] = args.jax_platforms
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from euclid_dsps.config import load_config
    from euclid_dsps.filters import load_filters
    from euclid_dsps.io import ensure_dir, write_dataframe_outputs, write_json
    from euclid_dsps.model import load_context
    from euclid_dsps.nebular import write_nebular_diagnostic_outputs
    from euclid_dsps.reporting import (
        write_batch_outputs,
        write_fit_diagnostic_outputs,
        write_trace_truth_outputs,
    )

    run_dir = ensure_dir(args.run)
    chunk_dir = run_dir / "_chunks"
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    config = load_config(args.config)
    if args.reporting_level is not None:
        config["reporting"]["level"] = args.reporting_level
    reporting_level = str(config.get("reporting", {}).get("level", "full"))

    fits, fit_sources = _read_checkpoint_frames(chunk_dir, "batch_fit_results")
    comparison, comparison_sources = _read_checkpoint_frames(
        chunk_dir, "batch_fit_photometry_comparison"
    )
    trace, trace_sources = _read_checkpoint_frames(chunk_dir, "batch_fit_trace")

    if fits.empty:
        raise RuntimeError(f"No batch_fit_results checkpoints found in {chunk_dir}")
    if comparison.empty:
        raise RuntimeError(
            f"No batch_fit_photometry_comparison checkpoints found in {chunk_dir}"
        )

    fits = _sort_if_present(fits, ["row_index", "chunk_index"])
    comparison = _sort_if_present(comparison, ["row_index", "band"])
    trace = _sort_if_present(trace, ["chunk_index", "iteration"])

    write_dataframe_outputs(fits, run_dir, "batch_fit_results", config)
    write_dataframe_outputs(
        comparison, run_dir, "batch_fit_photometry_comparison", config
    )
    if not trace.empty:
        write_dataframe_outputs(trace, run_dir, "batch_fit_trace", config)

    write_batch_outputs(
        comparison,
        run_dir,
        label="batch_fit",
        reporting_level=reporting_level,
        config=config,
    )
    write_fit_diagnostic_outputs(fits, comparison, config, run_dir, label="batch_fit")
    if not trace.empty:
        write_trace_truth_outputs(
            trace,
            run_dir,
            label="batch_fit",
            make_plots=reporting_level == "full",
        )

    if args.include_nebular:
        filters = load_filters(config["bands"])
        context = load_context(
            config["ssp_path"],
            filters,
            n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
            cosmos_config=config.get("cosmos_sed"),
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=config.get("model"),
        )
        write_nebular_diagnostic_outputs(
            context,
            fits,
            run_dir,
            label="batch_fit",
            make_plots=reporting_level == "full",
        )

    missing_rows = _missing_contiguous_rows(fits, args.expected_limit)
    completion = {
        "source": "scripts/finalize_fit_from_chunks.py",
        "config": str(args.config),
        "run_dir": str(run_dir),
        "chunk_dir": str(chunk_dir),
        "expected_limit": args.expected_limit,
        "batch_size": args.batch_size,
        "reporting_level": reporting_level,
        "n_fit_chunks": len(fit_sources),
        "n_comparison_chunks": len(comparison_sources),
        "n_trace_chunks": len(trace_sources),
        "n_fit_rows": int(len(fits)),
        "n_comparison_rows": int(len(comparison)),
        "n_trace_rows": int(len(trace)),
        "row_index_min": _json_int(fits["row_index"].min()),
        "row_index_max": _json_int(fits["row_index"].max()),
        "n_missing_rows": len(missing_rows),
        "missing_rows_preview": missing_rows[:100],
        "complete_for_expected_limit": (
            args.expected_limit is None or len(missing_rows) == 0
        ),
        "note": (
            "Aggregate outputs were regenerated from existing chunk checkpoints; "
            "no new MAP optimization was run."
        ),
    }
    write_json(run_dir / "batch_fit_completion_summary.json", completion)
    write_json(
        run_dir / "batch_fit_run_config.json",
        {
            "rows_written": int(len(comparison)),
            "fit_rows_written": int(len(fits)),
            "limit": args.expected_limit,
            "batch_size": args.batch_size,
            "row_indices_file": None,
            "source": "scripts/finalize_fit_from_chunks.py",
            "complete_for_expected_limit": completion["complete_for_expected_limit"],
        },
    )
    if missing_rows:
        pd.DataFrame({"row_index": missing_rows}).to_csv(
            run_dir / "batch_fit_missing_rows.csv", index=False, header=False
        )
    write_json(run_dir / "normalized_config.json", config)
    print(
        "finalized "
        f"{len(fits)} fit rows from {len(fit_sources)} chunks; "
        f"missing_rows={len(missing_rows)}"
    )
    return 0


def _read_checkpoint_frames(chunk_dir: Path, stem: str) -> tuple[pd.DataFrame, list[str]]:
    parquet_paths = sorted(chunk_dir.glob(f"{stem}_chunk_*.parquet"))
    csv_paths = sorted(chunk_dir.glob(f"{stem}_chunk_*.csv"))
    paths = parquet_paths if parquet_paths else csv_paths
    if not paths:
        return pd.DataFrame(), []
    frames = []
    for path in paths:
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), [path.name for path in paths]


def _sort_if_present(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [column for column in columns if column in frame]
    if not present:
        return frame.reset_index(drop=True)
    return frame.sort_values(present).reset_index(drop=True)


def _missing_contiguous_rows(fits: pd.DataFrame, expected_limit: int | None) -> list[int]:
    if expected_limit is None or "row_index" not in fits:
        return []
    observed = set(int(value) for value in fits["row_index"].dropna().astype(int))
    return [index for index in range(expected_limit) if index not in observed]


def _json_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


if __name__ == "__main__":
    raise SystemExit(main())
