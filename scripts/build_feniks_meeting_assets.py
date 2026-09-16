"""Publish historical meeting assets with source hashes; never infer run status."""

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo / "outputs/feniks_epoch160_complete"
    out = repo / "docs/source/_static/feniks_debug"
    out.mkdir(parents=True, exist_ok=True)
    provenance = {}

    def record(path):
        provenance[str(path.relative_to(repo))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    figures = source / "aggregated_figures_v2"
    manifest = json.loads((figures / "figure_manifest.json").read_text())
    for name in (
        "population_selected_marginals",
        "individual_posteriors_physical5d",
        "heldout_importance_support",
    ):
        for suffix in ("png", "pdf"):
            path = figures / f"{name}.{suffix}"
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != manifest["artifacts"][path.name]["sha256"]
            ):
                raise ValueError(f"Historical asset hash mismatch: {path}")
            shutil.copyfile(path, out / f"epoch160_{path.name}")
            record(path)
    for path, name in (
        (figures / "figure_manifest.json", "epoch160_figure_manifest.json"),
        (source / "CHECKPOINT_FROZEN.json", "epoch160_checkpoint.json"),
        (source / "heldout/mira/mira_manifest.json", "epoch160_mira_manifest.json"),
        (source / "heldout/mira/mira_scores.csv", "epoch160_mira_scores.csv"),
    ):
        shutil.copyfile(path, out / name)
        record(path)

    scores = pd.read_csv(source / "heldout/mira/mira_scores.csv")
    groups = ["full_15d", "physical_5d", "sfh_contrasts_10d", "marginal_z_obs"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), layout="constrained", sharey=True)
    for ax, group in zip(axes, groups, strict=True):
        part = scores[scores.group.eq(group)]
        y = np.arange(len(part))
        ax.errorbar(
            part.score,
            y,
            xerr=np.vstack(
                (part.score - part.bootstrap_q025, part.bootstrap_q975 - part.score)
            ),
            fmt="o",
            color="#177e89",
            capsize=3,
        )
        ax.axvline(2 / 3, color="#bd3654", ls="--")
        ax.set(
            yticks=y,
            yticklabels=part.model,
            xlabel="MIRA score",
            title=group.replace("_", " "),
        )
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Historical epoch 160: 512 held-out objects, 128 draws, 100 regions\nIntervals: object + region bootstrap; dashed line: calibrated reference 2/3"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"epoch160_mira.{suffix}", dpi=170)
    plt.close(fig)

    path = source / "population/individual_panels/raw_q.parquet"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != manifest["sources"]["individual_raw_q"]["sha256"]
    ):
        raise ValueError("Historical joint-draw source hash mismatch")
    record(path)
    frame = pd.read_parquet(path)
    # Smallest archived row ID is deterministic and independent of fit/ESS/truth.
    row = int(frame.row_index.min())
    frame = frame[frame.row_index.eq(row)]
    columns = [
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
    ]
    labels = ["z", "log10 mass", "log10 Z", "dust Av", "dust slope"]
    values = frame[columns].to_numpy()
    if len(frame) < 2 or not np.isfinite(values).all():
        raise ValueError("Corner requires multiple finite joint draws")
    fig, axes = plt.subplots(5, 5, figsize=(11, 11), layout="constrained")
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
            elif i == j:
                ax.hist(
                    values[:, i], bins=20, density=True, color="#177e89", alpha=0.75
                )
            else:
                ax.scatter(
                    values[:, j],
                    values[:, i],
                    s=4,
                    alpha=0.25,
                    color="#177e89",
                    rasterized=True,
                )
            if i == 4:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel("density" if i == 0 else labels[i])
            ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Historical epoch 160 / raw q / row {row}: {len(frame)} joint draws\nOne object's proposal, NOT the current pilot or an importance-corrected posterior"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"epoch160_corner.{suffix}", dpi=160)
    plt.close(fig)
    (out / "meeting_asset_provenance.json").write_text(
        json.dumps(
            {
                "role": "historical_epoch160_not_current_pilot",
                "corner": {
                    "row_index": row,
                    "draws": len(frame),
                    "columns": columns,
                    "weights": "uniform direct q draws",
                },
                "source_sha256": provenance,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
