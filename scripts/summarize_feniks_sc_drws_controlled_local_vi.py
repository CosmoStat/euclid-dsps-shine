"""Read saved local trajectories without decoder calls or checkpoint selection."""

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(root: Path) -> pd.DataFrame:
    rows = []
    for case in sorted((root / "cases").glob("*")):
        for path in sorted(case.rglob("SUMMARY.json")):
            relative = path.relative_to(case)
            if relative.parts[0] == "amortized":
                regime, start, step = "amortized", -1, 0
            else:
                metadata = json.loads(
                    (case / relative.parts[0] / "REGIME.json").read_text()
                )
                regime, start = metadata["name"], metadata["initialization"]
                step = (
                    int(relative.parts[1][5:])
                    if len(relative.parts) == 3
                    else json.loads((root / "RUN_MANIFEST.json").read_text())["steps"]
                )
            summary = json.loads(path.read_text())
            support = summary
            rows.append(
                dict(
                    case=case.name,
                    regime=regime,
                    start=start,
                    step=step,
                    ess_fraction=support["raw_ess"]["fraction_median"],
                    max_weight=support["maximum_raw_weight"]["p90"],
                    bad_k=support["pareto_k"]["gt_0p7_or_nonfinite_fraction"],
                    nonfinite_k=support["pareto_k"]["nonfinite_fraction"],
                    negative_elbo=summary["negative_elbo"],
                    evidence_delta=summary["replicate_abs_log_evidence_delta"],
                    residual_rms=summary["residual_rms"],
                    min_latent_std=min(summary["latent_std"]),
                    max_latent_std=max(summary["latent_std"]),
                    **summary.get("density_means", {}),
                )
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    frame = collect(args.root)
    if frame.empty:
        print("No completed evaluations yet")
        return
    print(frame.sort_values(["case", "regime", "start", "step"]).to_string(index=False))
    print(
        "Descriptive development trajectories only; no checkpoint selection or promotion."
    )


if __name__ == "__main__":
    main()
