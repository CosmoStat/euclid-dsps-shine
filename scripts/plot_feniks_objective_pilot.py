"""Plot attributed replay readback and the proposed objective-pilot contract."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def replay_figure(out: Path) -> None:
    # Transcribed from operator job 1959175, not from remotely fetched artifacts.
    examples = {
        "observed_002": {
            "ESS/K": [[0.003904, 0.003303], [0.042724, 0.001011], [0.035775, 0.006330]],
            "Residual RMS": [
                [4.086969, 4.090124],
                [1.815766, 1.809893],
                [2.331290, 2.418422],
            ],
        },
        "simulated_003": {
            "ESS/K": [[0.112768, 0.093749], [0.007318, 0.009692], [0.005025, 0.019412]],
            "Residual RMS": [
                [2.902106, 2.691419],
                [0.880715, 0.887991],
                [0.860482, 0.865114],
            ],
        },
    }
    colors = ("#626873", "#147d92", "#b34364")
    labels = ("Frozen amortized C", "Local start 0", "Local start 1")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, (case, metrics) in enumerate(examples.items()):
        for col, (metric, values) in enumerate(metrics.items()):
            ax = axes[row, col]
            for value, color, label in zip(values, colors, labels, strict=True):
                ax.plot([0, 1], value, "o-", color=color, label=label, linewidth=2)
            ax.set(
                title=case,
                ylabel=metric,
                xticks=[0, 1],
                xticklabels=["K1024", "K4096 (independent draws)"],
                xlim=(-0.12, 1.12),
            )
            if col == 0:
                ax.set_yscale("log")
                ax.plot([0, 1], 1 / np.asarray([1024, 4096]), "k:", label="ESS = 1")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle(
        "Frozen checkpoints: stable flux residuals, unstable importance weights",
        fontsize=14,
    )
    fig.savefig(out / "replay_examples.png", dpi=170)
    plt.close(fig)


def protocol_figure(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")

    def label(x, y, heading, body, color):
        ax.text(
            x,
            y,
            heading,
            ha="center",
            va="center",
            weight="bold",
            fontsize=12,
            color=color,
        )
        ax.text(x, y - 0.45, body, ha="center", va="top", fontsize=10, linespacing=1.6)

    def arrow(start, end, color="#626873"):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
        )

    label(
        2,
        5.6,
        "1. Fixed source",
        "16 development cases\n2 final step-512 starts\nPrior, decoder, context frozen",
        "#34383e",
    )
    label(
        6.2,
        5.6,
        "2. Full VI gradient audit",
        "Same random noise for AD / FD\nMean, scale, flow-layer directions\nNative arithmetic is the gate",
        "#34383e",
    )
    arrow((3.8, 4.7), (4.4, 4.7))
    arrow((8.2, 4.7), (9.0, 4.7))
    label(
        10.4,
        5.6,
        "FAIL / INCONCLUSIVE",
        "Stop before any adaptation\nParameter64 is diagnostic only\nNo override from a second path",
        "#b34364",
    )
    ax.text(
        6.2, 3.7, "All native checks PASS", ha="center", weight="bold", color="#147d92"
    )
    arrow((6.2, 4.0), (6.2, 3.9), "#147d92")
    arrow((5.8, 3.45), (3.2, 3.0), "#147d92")
    arrow((6.6, 3.45), (9.2, 3.0), "#147d92")
    label(
        3.2,
        2.8,
        "3A. Reverse-KL control",
        "128 updates x 32 fresh draws\nPathwise gradient through decoder\nSame final-512 initialization",
        "#147d92",
    )
    label(
        9.2,
        2.8,
        "3B. Guarded wake candidate",
        "16 attempts x 256 mixture draws\nExact mixture weights; stopped draws / weights\nESS >= 16 and maximum weight <= 0.20",
        "#b34364",
    )
    arrow((3.2, 1.45), (5.3, 0.95))
    arrow((9.2, 1.45), (7.1, 0.95))
    ax.text(
        6.2,
        0.65,
        "4. Independent direct-draw evaluation",
        ha="center",
        weight="bold",
        fontsize=12,
    )
    ax.text(
        6.2,
        0.17,
        "Same decoder-sample budget, not equal compute. No winner selection or promotion.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(out / "objective_protocol.png", dpi=170)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "docs/source/_static/feniks_debug"
    out.mkdir(parents=True, exist_ok=True)
    replay_figure(out)
    protocol_figure(out)


if __name__ == "__main__":
    main()
