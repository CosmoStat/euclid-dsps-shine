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
        xlabel="Perte wake apres - avant (lot fige)",
        title="7/9 premiers pas acceptes augmentent leur propre perte",
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
            "RMS / RMS sans mise a jour (bas = mieux)",
            "Fraction ESS / reference (haut = mieux)",
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
            xlabel="Amplitude du pas",
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
        fig.colorbar(im, ax=ax, label="log2 du ratio", shrink=0.7)
    fig.suptitle(
        "Evaluations K4096 : amplitudes prescrites, bruit commun; 9 premiers pas"
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
        ax.plot(h, fd, "o-", color="#168575", label="Difference finie centree")
        ax.axhline(ad, color="#bd3654", ls="--", label="AD : gradient . deplacement")
        ax.set(
            xscale="log",
            xlabel="h autour du point initial",
            ylabel="Derivee directionnelle",
            title=name,
        )
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)
    fig.suptitle(
        "Le gradient local concorde; cela ne garantit pas la descente au pas complet"
    )
    fig.savefig(out / "wake_derivatives.png", dpi=180)
    fig.savefig(out / "wake_derivatives.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
