#!/usr/bin/env python3
"""Select the adjusted-MCLMC production settings from the fixed pilot grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.exact_posterior import combine_chain_diagnostics

CONFIGS = (
    ("adj_official_t1", "mclmc", (0.1, 0.1, 0.1), 1, None),
    ("adj_jaxfli_t1", "mclmc", (0.4, 0.4, 0.2), 1, None),
    ("unadj_jaxfli_e1e3", "mclmc_unadjusted", (0.4, 0.4, 0.2), 1, 1.0e-3),
    ("unadj_jaxfli_e5e4", "mclmc_unadjusted", (0.4, 0.4, 0.2), 1, 5.0e-4),
    ("adj_jaxfli_t4", "mclmc", (0.4, 0.4, 0.2), 4, None),
    ("adj_jaxfli_t8", "mclmc", (0.4, 0.4, 0.2), 8, None),
    ("adj_jaxfli_t16", "mclmc", (0.4, 0.4, 0.2), 16, None),
)


def _load_x(directories: list[Path]) -> np.ndarray:
    frames = []
    for directory in directories:
        chunks = sorted(
            path
            for path in (directory / "chunks").glob("part_*.parquet")
            if not path.name.endswith("_info.parquet")
        )
        if not chunks:
            raise FileNotFoundError(f"No sample chunks found in {directory}")
        for chunk in chunks:
            frame = pd.read_parquet(chunk)
            columns = [column for column in frame if column.startswith("x_")]
            frames.append(frame[columns].to_numpy(dtype=float))
    return np.concatenate(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--galaxy-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-galaxy-dir")
    args = parser.parse_args()
    galaxy = args.root / "galaxies" / args.galaxy_dir
    prepare = json.loads((galaxy / "prepare_manifest.json").read_text())
    names = tuple(prepare["latent_spec"]["names"])
    if args.validate_galaxy_dir:
        validate = args.root / "galaxies" / args.validate_galaxy_dir
        validate_prepare = json.loads(
            (validate / "prepare_manifest.json").read_text(encoding="utf-8")
        )
        validate_names = tuple(validate_prepare["latent_spec"]["names"])
        directories = [
            validate / "mclmc" / f"chain_{chain:02d}"
            for chain in range(2)
        ]
        diagnostics, summary = combine_chain_diagnostics(
            directories, parameter_names=validate_names
        )
        diagnostics.to_parquet(
            validate / "mclmc" / "pilot_diagnostics.parquet",
            index=False,
        )
        selection = json.loads(args.out.read_text(encoding="utf-8"))
        selection["validation_galaxy"] = args.validate_galaxy_dir
        selection["validation_max_rhat"] = summary["max_rhat"]
        selection["validation_min_bulk_ess"] = summary["min_bulk_ess"]
        selection["validation_passes_rhat_1_10"] = bool(
            summary["max_rhat"] <= 1.10
        )
        args.out.write_text(
            json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not selection["validation_passes_rhat_1_10"]:
            raise SystemExit(
                "Selected MCLMC pilot failed the second-galaxy R-hat <= 1.10 gate"
            )
        print(json.dumps(selection, indent=2))
        return
    nuts_directories = [
        galaxy / "nuts" / f"chain_{chain:02d}" for chain in range(4)
    ]
    if not all((directory / "DONE").exists() for directory in nuts_directories):
        raise FileNotFoundError("The four-chain NUTS pilot reference is incomplete")
    nuts_x = _load_x(nuts_directories)
    nuts_mean = np.mean(nuts_x, axis=0)
    nuts_scale = np.maximum(np.std(nuts_x, axis=0, ddof=1), 1.0e-6)
    rows = []
    for label, sampler, frac, thinning, energy in CONFIGS:
        directories = [galaxy / label / f"chain_{chain:02d}" for chain in range(2)]
        if not all((directory / "DONE").exists() for directory in directories):
            raise FileNotFoundError(f"Pilot configuration is incomplete: {label}")
        diagnostics, summary = combine_chain_diagnostics(
            directories, parameter_names=names
        )
        diagnostics.to_parquet(galaxy / label / "diagnostics.parquet", index=False)
        elapsed = 0.0
        integrator_steps = 0
        for directory in directories:
            manifest = json.loads(
                (directory / "chain_manifest.json").read_text(encoding="utf-8")
            )
            elapsed += float(manifest["total_elapsed_s"])
            integrator_steps += int(
                manifest.get(
                    "integrator_steps_after_warmup",
                    manifest.get("kernel_transitions", 0),
                )
            )
        candidate_x = _load_x(directories)
        standardized_mean_distance = float(
            np.sqrt(
                np.mean(
                    ((np.mean(candidate_x, axis=0) - nuts_mean) / nuts_scale) ** 2
                )
            )
        )
        rows.append(
            {
                "label": label,
                "sampler": sampler,
                "frac_tune": list(frac),
                "thinning": thinning,
                "desired_energy_var": energy,
                "max_rhat": summary["max_rhat"],
                "min_bulk_ess": summary["min_bulk_ess"],
                "min_tail_ess": summary["min_tail_ess"],
                "elapsed_s": elapsed,
                "integrator_steps": integrator_steps,
                "nuts_standardized_mean_distance": standardized_mean_distance,
                "bulk_ess_per_1000_integrator_steps": (
                    1000.0 * summary["min_bulk_ess"] / max(integrator_steps, 1)
                ),
            }
        )
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out.with_suffix(".parquet"), index=False)
    table.to_csv(args.out.with_suffix(".csv"), index=False)
    adjusted = table.loc[table["sampler"] == "mclmc"].copy()
    converged = adjusted.loc[adjusted["max_rhat"] <= 1.05]
    candidates = converged if not converged.empty else adjusted
    chosen = candidates.sort_values(
        [
            "nuts_standardized_mean_distance",
            "bulk_ess_per_1000_integrator_steps",
            "min_bulk_ess",
        ],
        ascending=[True, False, False],
    ).iloc[0]
    payload = {
        "selected_label": str(chosen["label"]),
        "frac_tune": list(chosen["frac_tune"]),
        "thinning": int(chosen["thinning"]),
        "pilot_max_rhat": float(chosen["max_rhat"]),
        "pilot_min_bulk_ess": float(chosen["min_bulk_ess"]),
        "pilot_nuts_standardized_mean_distance": float(
            chosen["nuts_standardized_mean_distance"]
        ),
        "selection_gate": (
            "adjusted_only_then_rhat_le_1.05_then_min_nuts_mean_distance"
            "_then_max_bulk_ess_per_1000_integrator_steps"
        ),
        "unadjusted_configs_are_diagnostics_only": True,
    }
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
