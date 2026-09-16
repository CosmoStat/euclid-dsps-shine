"""Build reproducible documentation figures from attributed pasted evidence."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "docs/controlled_vi_evidence.json").read_text())
    out = root / "docs/source/_static/feniks_debug"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for row, (name, values) in enumerate(data["examples"].items()):
        for col, metric in enumerate(("residual_rms", "ess_fraction")):
            ax = axes[row, col]
            ax.plot(
                data["steps"], values[metric], "o-", color=("#147d92", "#b34364")[col]
            )
            ax.set(title=name, xlabel="Optimization step", ylabel=metric)
            ax.grid(alpha=0.2)
            if col == 1:
                ax.axhline(1 / 256, color="black", ls=":", label="ESS = 1 / K256")
                ax.legend()
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    matrix = np.concatenate(
        [np.asarray(v) for v in data["observed_final_ess_fraction"].values()], axis=1
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    im = ax.imshow(matrix * 256, cmap="viridis", vmin=1, vmax=12, aspect="auto")
    ax.set(
        xticks=range(6),
        xticklabels=[
            f"{r}\nstart {s}"
            for r in data["observed_final_ess_fraction"]
            for s in range(2)
        ],
        yticks=range(8),
        yticklabels=[f"observed_{i:03d}" for i in range(8)],
        title="Final raw ESS out of 256 draws; all 48 have bad Pareto-k",
    )
    for i in range(8):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{matrix[i, j] * 256:.1f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] * 256 < 7 else "black",
            )
    fig.colorbar(im, ax=ax, label="Raw ESS (not a certification)")
    fig.savefig(out / "final_support.png", dpi=160)
    plt.close(fig)
    x = np.linspace(-6, 6, 1200)

    def normal(mean, std):
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(
        x,
        0.7 * normal(-1, 0.7) + 0.3 * normal(2, 0.6),
        color="black",
        label="Illustrative target",
    )
    ax.plot(x, normal(-1, 0.45), color="#b34364", label="Narrow local proposal")
    ax.plot(
        x,
        0.5 * normal(-1, 0.9) + 0.5 * normal(0, 2),
        color="#147d92",
        label="Broadened + anchor mixture",
    )
    ax.set(
        title="Concept only: better local fit can still miss probability mass",
        xlabel="One illustrative parameter (not a galaxy result)",
        ylabel="Normalized density",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(out / "principle.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
