#!/usr/bin/env python3
"""Summarize and gate the one-chain adjusted-MCLMC adaptation probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--label", default="mclmc_adaptation_probe_t10_compat")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    galaxies = sorted(
        path for path in (args.root / "galaxies").iterdir() if path.is_dir()
    )
    if not galaxies:
        raise FileNotFoundError(f"No galaxies found in {args.root}")
    probe = galaxies[0] / args.label / "chain_00"
    manifest_path = probe / "chain_manifest.json"
    if not (probe / "DONE").exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete MCLMC adaptation probe: {probe}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    info_paths = sorted((probe / "chunks").glob("part_*_info.parquet"))
    if not info_paths:
        raise FileNotFoundError(f"No MCLMC transition diagnostics in {probe}")
    info = pd.concat(
        [pd.read_parquet(path) for path in info_paths], ignore_index=True
    )
    acceptance = info["acceptance_rate"].to_numpy(dtype=np.float64)
    energy = info["energy"].to_numpy(dtype=np.float64)
    divergent = info["is_divergent"].astype(bool).to_numpy()

    checks = {
        "blackjax_adaptation": (
            manifest.get("adaptation_mode") == "blackjax_three_phase"
        ),
        "positive_tuning_work": (
            int(manifest.get("tune_steps", 0)) > 0
            and int(manifest.get("actual_tuning_integrator_steps", 0)) > 0
        ),
        "finite_geometry": bool(
            np.isfinite(float(manifest.get("step_size", np.nan)))
            and float(manifest.get("step_size", 0.0)) > 1.0e-8
            and np.isfinite(float(manifest.get("L", np.nan)))
            and float(manifest.get("L", 0.0)) > 0.0
        ),
        "finite_transitions": bool(
            acceptance.size > 0
            and np.isfinite(acceptance).all()
            and np.isfinite(energy).all()
        ),
        "acceptance_sanity": bool(
            acceptance.size > 0 and float(np.mean(acceptance)) >= 0.05
        ),
        "no_divergences": bool(not divergent.any()),
    }
    passed = bool(all(checks.values()))
    summary = {
        "status": "passed" if passed else "failed",
        "root": str(args.root),
        "galaxy": galaxies[0].name,
        "probe_label": args.label,
        "checks": checks,
        "tune_steps": int(manifest["tune_steps"]),
        "actual_tuning_integrator_steps": int(
            manifest["actual_tuning_integrator_steps"]
        ),
        "step_size": float(manifest["step_size"]),
        "L": float(manifest["L"]),
        "integration_steps_per_transition": int(
            manifest["integration_steps_per_transition"]
        ),
        "stored_samples": int(manifest["stored_samples"]),
        "mean_acceptance": float(np.mean(acceptance)),
        "median_acceptance": float(np.median(acceptance)),
        "divergences": int(divergent.sum()),
        "energy_std": float(np.std(energy)),
        "warmup_elapsed_s": float(manifest["warmup_elapsed_s"]),
        "total_elapsed_s": float(manifest["total_elapsed_s"]),
    }
    out = args.out or (galaxies[0] / args.label / "probe_summary.json")
    out.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("MCLMC adaptation probe failed; do not submit the pilot")


if __name__ == "__main__":
    main()
