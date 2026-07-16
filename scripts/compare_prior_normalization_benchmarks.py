#!/usr/bin/env python3
"""Compare the four FENIKS normalization/flow benchmark runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_RUNS = (
    "feniks_norm_hybrid_realnvp",
    "feniks_norm_hybrid_rqspline",
    "feniks_norm_dirac_realnvp",
    "feniks_norm_dirac_rqspline",
)


def _to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_name in args.runs:
        run = args.runs_root / run_name
        benchmark = json.loads(
            (run / "benchmark_summary.json").read_text(encoding="utf-8")
        )
        diagnostic = json.loads(
            (run / "supervised_prior_summary.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "run": run_name,
                "normalization": benchmark["normalization_version"],
                "flow": benchmark["flow_type"],
                "test_nll_same_measure_only": benchmark["negative_mean_log_prob"][
                    "test"
                ],
                "median_ks_physical": diagnostic["median_ks_distance"],
                "median_wasserstein_physical": diagnostic[
                    "median_wasserstein_distance"
                ],
                "correlation_frobenius_error": diagnostic[
                    "correlation_frobenius_error"
                ],
                "sliced_wasserstein_physical": diagnostic[
                    "sliced_wasserstein_distance"
                ],
                "energy_distance_physical": diagnostic["energy_distance"],
                "quality_gate": diagnostic["prior_quality_gate_status"],
                "truth_atom_fraction": benchmark["atom_fraction_test"],
                "sample_atom_fraction": benchmark["atom_fraction_sampled"],
                "exact_atom_sample_rows": benchmark["exact_atom_sample_rows"],
                "elapsed_time_s": benchmark["elapsed_time_s"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["normalization", "median_ks_physical", "flow"]
    )
    frame.to_csv(args.out / "normalization_flow_comparison.csv", index=False)
    lines = [
        "# FENIKS normalization and flow comparison",
        "",
        "Physical-space distances compare all four runs. Test NLL is only comparable "
        "between RealNVP and RQ spline inside the same normalization version: the "
        "hybrid model and the one-vector model use different reference measures.",
        "",
        _to_markdown(frame),
        "",
    ]
    (args.out / "normalization_flow_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
