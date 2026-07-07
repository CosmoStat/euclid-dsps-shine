#!/usr/bin/env python3
"""Check that optimized DSPS magnitude paths preserve forward photometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.config import load_config
from euclid_dsps.jax_runtime import apply_jax_runtime_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/popcosmos_binned_compressed.yaml",
        help="Config to evaluate.",
    )
    parser.add_argument("--n", type=int, default=16, help="Number of random points.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--out",
        default="outputs/benchmarks/photometry_equivalence",
        help="Output directory.",
    )
    parser.add_argument(
        "--atol-mag",
        type=float,
        default=5.0e-5,
        help="Failure threshold on max |delta_mag|.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    apply_jax_runtime_env(config.get("runtime"))

    from euclid_dsps.filters import load_filters
    from euclid_dsps.model import (
        load_context,
        run_dsps_model_jax,
        run_dsps_model_mags_jax,
    )

    filters = load_filters(config["bands"])
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    rng = np.random.default_rng(args.seed)
    points = sample_parameter_points(config, args.n, rng)
    rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for point_index, params in enumerate(points):
        full = np.asarray(run_dsps_model_jax(context, params).model_mags, dtype=float)
        fast = np.asarray(run_dsps_model_mags_jax(context, params), dtype=float)
        delta = fast - full
        max_abs_delta = max(max_abs_delta, float(np.nanmax(np.abs(delta))))
        for band_index, band in enumerate(config["bands"]):
            rows.append(
                {
                    "point_index": point_index,
                    "band": str(band["name"]),
                    "full_mag": float(full[band_index]),
                    "fast_mag": float(fast[band_index]),
                    "delta_mag_fast_minus_full": float(delta[band_index]),
                    **{f"param_{key}": float(value) for key, value in params.items()},
                }
            )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "photometry_equivalence_points.csv", index=False)
    summary = {
        "config": str(args.config),
        "n_points": int(args.n),
        "n_bands": int(len(config["bands"])),
        "max_abs_delta_mag": float(max_abs_delta),
        "median_abs_delta_mag": float(
            frame["delta_mag_fast_minus_full"].abs().median()
        ),
        "p95_abs_delta_mag": float(
            frame["delta_mag_fast_minus_full"].abs().quantile(0.95)
        ),
        "atol_mag": float(args.atol_mag),
        "passed": bool(np.isfinite(max_abs_delta) and max_abs_delta <= args.atol_mag),
    }
    (out / "photometry_equivalence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def sample_parameter_points(
    config: dict[str, Any],
    n_points: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    fixed = dict(config.get("model", {}).get("fixed_parameters", {}) or {})
    free = dict(config.get("fit", {}).get("free_parameters", {}) or {})
    points: list[dict[str, float]] = []
    for _ in range(n_points):
        params = {key: float(value) for key, value in fixed.items()}
        for name, spec in free.items():
            bounds = spec.get("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                params[name] = float(spec.get("initial", 0.0))
                continue
            low, high = float(bounds[0]), float(bounds[1])
            params[name] = float(rng.uniform(low, high))
        if {
            "log10_gas_metallicity",
            "log10_stellar_metallicity",
        }.issubset(params):
            params["log10_gas_metallicity"] = max(
                params["log10_gas_metallicity"],
                params["log10_stellar_metallicity"],
            )
        points.append(params)
    return points


if __name__ == "__main__":
    raise SystemExit(main())
