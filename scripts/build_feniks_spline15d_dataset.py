#!/usr/bin/env python3
"""Project an existing Diffsky dataset to the production spline-SFH 15D truth."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import jax
import pandas as pd
import yaml

from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.prior_learning.spline15d import (
    PHYSICAL_PARAMETER_NAMES,
    SFH_CONTRAST_NAMES,
    SPLINE15D_PARAMETER_NAMES,
    dequantize_spline_contrast_atoms,
    project_diffsky_frame_to_spline15d,
    validate_normalized_log_time_nodes,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--splits", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = dict(config.get("spline15d_postprocess", {}) or {})
    source = Path(args.source_dir or cfg["source_dataset_dir"])
    out = ensure_dir(args.out or cfg["output_dataset_dir"])
    splits = tuple(args.splits or cfg.get("splits", ("train", "validation", "test")))
    nodes = validate_normalized_log_time_nodes(cfg["normalized_log_time_nodes"])
    n_sfh_bins = int(cfg.get("n_sfh_bins", 80))
    batch_size = int(args.batch_size or cfg.get("batch_size", 2048))
    half_width = float(cfg.get("dequantization_half_width_dex", 1.0e-4))
    seed = int(cfg.get("seed", 260715))
    grouped_audit_path = source / "grouped_split_audit.json"
    grouped_audit = None
    if grouped_audit_path.exists():
        grouped_audit = json.loads(grouped_audit_path.read_text(encoding="utf-8"))
        overlaps = grouped_audit.get("effective_proposal_overlap", {})
        if any(int(value) != 0 for value in overlaps.values()):
            raise ValueError("Grouped source audit reports effective proposal leakage")
    elif bool(cfg.get("require_grouped_source_audit", False)):
        raise FileNotFoundError(
            f"Missing required grouped source audit: {grouped_audit_path}"
        )
    split_records = {}

    for split_index, split in enumerate(splits):
        source_path = source / f"{split}.parquet"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        paths = {
            "flow": out / f"{split}.parquet",
            "exact": out / f"{split}_exact.parquet",
            "nodes": out / f"{split}_spline_nodes.parquet",
        }
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                "Refusing to overwrite existing outputs; pass --overwrite: "
                + ", ".join(existing)
            )
        frame = pd.read_parquet(source_path)
        source_rows = len(frame)
        if args.limit is not None:
            frame = frame.head(max(int(args.limit), 0))
        print(f"[spline15d] projecting {split}: rows={len(frame)}", flush=True)
        exact, node_audit = project_diffsky_frame_to_spline15d(
            frame,
            normalized_log_time_nodes=nodes,
            n_sfh_bins=n_sfh_bins,
            batch_size=batch_size,
        )
        flow, atom_counts = dequantize_spline_contrast_atoms(
            exact,
            half_width_dex=half_width,
            seed=seed + split_index,
        )
        exact.to_parquet(paths["exact"], index=False)
        flow.to_parquet(paths["flow"], index=False)
        node_audit.to_parquet(paths["nodes"], index=False)
        split_records[split] = {
            "source": str(source_path),
            "source_rows": int(source_rows),
            "output_rows": int(len(flow)),
            "split_seed": int(seed + split_index),
            "exact_zero_atom_counts": atom_counts,
            "remaining_exact_zero_counts": {
                name: int((flow[name] == 0.0).sum()) for name in SFH_CONTRAST_NAMES
            },
            "outputs": {key: str(value) for key, value in paths.items()},
        }

    write_json(
        out / "spline15d_contract.json",
        {
            "version": 2,
            "source_is_pre_generated_diffsky": True,
            "source_dataset_dir": str(source),
            "output_dataset_dir": str(out),
            "parameter_count": len(SPLINE15D_PARAMETER_NAMES),
            "parameter_names": SPLINE15D_PARAMETER_NAMES,
            "physical_parameter_names": PHYSICAL_PARAMETER_NAMES,
            "sfh_parameterization": {
                "type": "jax_cosmo_not_a_knot_cubic_log_sfr_contrasts",
                "implementation": (
                    "jax_cosmo.scipy.interpolate.InterpolatedUnivariateSpline"
                ),
                "degree": 3,
                "endpoints": "not-a-knot",
                "node_count": len(nodes),
                "contrast_count": len(SFH_CONTRAST_NAMES),
                "normalized_log_time_nodes": nodes,
                "amplitude_source": "log10_stellar_mass",
                "n_native_sfh_bins": n_sfh_bins,
            },
            "flow_target": {
                "files": "<split>.parquet",
                "dequantization": "uniform_exact_zero_atoms_only",
                "half_width_dex": half_width,
                "exact_truth_files": "<split>_exact.parquet",
            },
            "seed": seed,
            "config": str(args.config),
            "grouped_source_audit": grouped_audit,
            "runtime": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "jax_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            },
            "splits": split_records,
        },
    )
    print(f"[spline15d] wrote dataset contract to {out}", flush=True)


if __name__ == "__main__":
    main()
