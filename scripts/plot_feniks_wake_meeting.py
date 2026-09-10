"""Meeting figures: operator transcription of job 1965476, 10 September 2026.

CSV values are rounded terminal output, not independently downloaded artifacts.
Only the first accepted update in each of nine changed trajectories is shown.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "docs/wake_forensic_evidence.csv")
    out = root / "docs/source/_static/feniks_debug"
    labels = df.case + "/" + df.start.astype(str)
    y = np.arange(len(df))
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    delta = df.loss_after - df.loss_before
    ax.barh(y, delta, color=np.where(delta > 0, "#bd3654", "#168575"))
    ax.set(
        yticks=y,
        yticklabels=labels,
        xlabel="Wake loss after - before (fixed batch)",
        title="7/9 first accepted updates increase their own loss",
    )
    ax.axvline(0, color="black", lw=1)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(out / "wake_loss.png", dpi=180)
    fig.savefig(out / "wake_loss.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ("rms", "ess"),
        (
            "RMS / no-update RMS (lower is better)",
            "ESS fraction / reference (higher is better)",
        ),
        strict=True,
    ):
        ratios = np.column_stack(
            [df[f"{metric}_{s}"] / df[f"{metric}_zero"] for s in ("p01", "p10", "full")]
        )
        im = ax.imshow(
            np.log2(ratios),
            cmap="RdBu_r" if metric == "rms" else "RdBu",
            vmin=-3,
            vmax=3,
            aspect="auto",
        )
        ax.set(
            yticks=y,
            yticklabels=labels,
            xticks=[0, 1, 2],
            xticklabels=["0.01", "0.1", "1"],
            xlabel="Update scale",
            title=title,
        )
        for i in range(len(df)):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{ratios[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(np.log2(ratios[i, j])) > 1.8 else "black",
                )
        fig.colorbar(im, ax=ax, label="log2 ratio", shrink=0.7)
    fig.suptitle(
        "K4096 evaluation: fixed scales, common noise; nine first updates"
    )
    fig.savefig(out / "wake_scales.png", dpi=180)
    fig.savefig(out / "wake_scales.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    h = np.array([1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
    for ax, name, ad, fd in (
        (
            axes[0],
            "observed_000 / 0",
            -5.815573,
            [-6.069105, -5.879165, -5.831484, -5.819551, -5.816567, -5.815821],
        ),
        (
            axes[1],
            "observed_006 / 0",
            -3.155938,
            [-3.041251, -3.127305, -3.148782, -3.154149, -3.155490, -3.155826],
        ),
    ):
        ax.plot(h, fd, "o-", color="#168575", label="Centered finite difference")
        ax.axhline(ad, color="#bd3654", ls="--", label="AD: gradient . displacement")
        ax.set(
            xscale="log",
            xlabel="h around the initial point",
            ylabel="Directional derivative",
            title=name,
        )
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Local derivatives agree; this does not guarantee full-step descent"
    )
    fig.savefig(out / "wake_derivatives.png", dpi=180)
    fig.savefig(out / "wake_derivatives.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
