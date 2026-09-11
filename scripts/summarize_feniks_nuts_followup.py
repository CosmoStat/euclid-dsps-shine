"""Summarize a frozen dense-mass NUTS probe or long run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(root: Path) -> pd.DataFrame:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    rows = []
    missing = []
    for task in manifest["nuts_tasks"]:
        out = root / "nuts" / task["case"] / task["variant"]
        final_path = out / "FINAL.json"
        if not final_path.exists():
            missing.append(str(out.relative_to(root)))
            continue
        final = json.loads(final_path.read_text())
        diagnostics = pd.read_csv(out / "diagnostics.csv")
        chain_manifests = [
            json.loads((out / f"chain_{i}" / "chain_manifest.json").read_text())
            for i in range(manifest["chains"])
        ]
        draws = manifest["chains"] * sum(manifest["chunks"])
        rows.append(
            {
                "case": task["case"],
                "variant": task["variant"],
                "depth": task["max_num_doublings"],
                "mass_matrix": final["mass_matrix"],
                "draws": draws,
                "limit_hit_fraction": final["integration_limit_hits"] / draws,
                "divergence_fraction": final["divergences"] / draws,
                "max_rhat": diagnostics["rhat"].replace([np.inf, -np.inf], np.nan).max(),
                "min_bulk_ess": diagnostics["bulk_ess"].min(),
                "min_tail_ess": diagnostics["tail_ess"].min(),
                "warmup_seconds": max(row["warmup_elapsed_s"] for row in chain_manifests),
                "total_seconds": max(row["total_elapsed_s"] for row in chain_manifests),
                "diagnostics_pass": final["diagnostics_pass"],
            }
        )
    if missing:
        raise RuntimeError("incomplete NUTS tasks: " + ", ".join(missing))
    frame = pd.DataFrame(rows).sort_values(["case", "depth"])
    frame.to_csv(root / "nuts_followup_summary.csv", index=False)
    return frame


def plot_depth_comparison(frame: pd.DataFrame, root: Path) -> None:
    if frame["depth"].nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for case, group in frame.groupby("case"):
        ordered = group.sort_values("depth")
        axes[0].plot(ordered["depth"], ordered["limit_hit_fraction"], "o-", label=case)
        axes[1].plot(ordered["depth"], ordered["divergence_fraction"], "o-", label=case)
    axes[0].set_ylabel("Fraction reaching trajectory limit")
    axes[1].set_ylabel("Divergence fraction")
    for ax in axes:
        ax.set_xlabel("Maximum NUTS doubling depth")
        ax.set_xticks(sorted(frame["depth"].unique()))
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "depth_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    frame = summarize(args.root.resolve())
    plot_depth_comparison(frame, args.root.resolve())
    print(frame.to_string(index=False))
    print("\nCompletion is not convergence. Use diagnostics_pass and inspect traces.")


if __name__ == "__main__":
    main()
