#!/usr/bin/env python3
"""Select one population-flow architecture using only frozen validation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from euclid_dsps.amortized.population_projection import (
    require_projection_runtime_commit,
)
from euclid_dsps.amortized.population_projection_benchmark import (
    BASELINE_NAME,
    select_truth_free_candidate,
)
from euclid_dsps.amortized.population_vem import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = _read_json(root / "RUN_MANIFEST.json")
    repo = Path(__file__).resolve().parents[1]
    runtime_provenance = require_projection_runtime_commit(
        root, manifest, repo, stage="evaluation"
    )
    winner_path = root / "TRUTH_FREE_ARCHITECTURE_WINNER.json"
    fit_path = root / "PROJECTION_FIT_COMPLETE.json"
    if winner_path.is_file() and fit_path.is_file():
        print(winner_path.read_text(encoding="utf-8"), flush=True)
        return

    benchmark = manifest["architecture_benchmark"]
    names = [item["name"] for item in benchmark["trained_candidates"]]
    names.append(BASELINE_NAME)
    records = []
    evidence = []
    for name in names:
        path = root / "candidates" / name / "TRUTH_FREE_EVALUATION.json"
        record = _read_json(path)
        for artifact, artifact_path in record["artifacts"].items():
            if sha256_file(artifact_path) != record["artifact_sha256"][artifact]:
                raise ValueError(f"candidate artifact changed: {artifact_path}")
        records.append(record)
        evidence.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    winner = select_truth_free_candidate(records)

    rows = []
    for record in records:
        rows.append(
            {
                "candidate": record["candidate"],
                "label": record["label"],
                "primary_score": record["primary_score"],
                "mean_core_5d_cdf_supremum": record[
                    "secondary_mean_core_5d_cdf_supremum"
                ],
                "fit_validation_weighted_nll_mean": record[
                    "fit_validation_weighted_nll_mean"
                ],
                "passes_truth_free_gates": record[
                    "passes_all_truth_free_distribution_gates"
                ],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["primary_score", "mean_core_5d_cdf_supremum", "candidate"]
    )
    frame.to_csv(root / "architecture_scores.csv", index=False)
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    colors = [
        "#0072B2" if name == winner["candidate"] else "#999999"
        for name in frame["candidate"]
    ]
    axis.bar(frame["label"], frame["primary_score"], color=colors)
    axis.axhline(
        1.0, color="#D55E00", linestyle="--", linewidth=1.4, label="all gates pass"
    )
    axis.set(
        ylabel="Worst normalized truth-free CDF gate",
        title="Population-flow architecture selection",
    )
    axis.tick_params(axis="x", rotation=18)
    axis.legend(frameon=False)
    plot_path = root / "architecture_scores.png"
    figure.savefig(plot_path, dpi=220)
    figure.savefig(plot_path.with_suffix(".pdf"))
    plt.close(figure)

    selected = dict(winner["fit"]["selected"])
    parent = dict(winner["fit"]["parent"])
    compatible_fit = {
        "status": "COMPLETE",
        "stage": "truth_free_population_projection_architecture_selection",
        "selected": selected,
        "parent": parent,
        "truth_used": False,
        "point_estimates_used": False,
        "dsps_calls_inside_optimizer": 0,
        "checkpoint_selection": "truth-free validation distributions only",
        "winner": winner["candidate"],
        "runtime_provenance": runtime_provenance,
    }
    _write_json(fit_path, compatible_fit)
    receipt = {
        "status": "WINNER_SELECTED",
        "winner": winner["candidate"],
        "winner_label": winner["label"],
        "winner_primary_score": winner["primary_score"],
        "winner_passes_all_truth_free_distribution_gates": winner[
            "passes_all_truth_free_distribution_gates"
        ],
        "winner_metrics": winner["comparisons"],
        "selection_order": [item["candidate"] for item in records],
        "candidate_records": evidence,
        "truth_used": False,
        "point_estimates_used": False,
        "sfh_used_for_architecture_selection": False,
        "redshift_median_gate_used": False,
        "posterior_calibration_used_for_selection": False,
        "closure_authorized": True,
        "scientific_promotion": False,
        "runtime_provenance": runtime_provenance,
        "artifacts": {
            "scores": str((root / "architecture_scores.csv").resolve()),
            "score_plot": str(plot_path.resolve()),
            "selected_checkpoint": selected["checkpoint"],
            "parent_checkpoint": parent["checkpoint"],
        },
    }
    _write_json(winner_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
