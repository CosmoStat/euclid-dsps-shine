"""Plot operator-transcribed job 1960443 stencils; not a new measurement."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    steps = 0.04 / 2 ** np.arange(8)
    fd = [
        [
            -1.745213,
            -2.004266,
            -2.559748,
            -3.688096,
            -4.071328,
            -4.056702,
            -3.980167,
            -4.002264,
        ],
        [
            -6.172213,
            -5.919832,
            -5.363936,
            -4.220589,
            -3.974516,
            -3.968873,
            -3.976938,
            -4.045865,
        ],
    ]
    ad = [-4.057858, -3.978338]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.semilogx(steps, fd[i], "o-", color="#147d92", label="Native FD")
        ax.axhline(ad[i], color="#b34364", linestyle="--", label="Native AD")
        ax.set(
            xlabel="Perturbation h (smaller to the right)",
            ylabel="Derivative of negative log likelihood",
            title=f"log_std direction {i}: {'INCONCLUSIVE' if i == 0 else 'PASS'}",
        )
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        "simulated_004 / start 1 | operator logs, job 1960443\nParameter64 overlaps native; transport64 has not been measured"
    )
    output = (
        Path(__file__).resolve().parents[1]
        / "docs/source/_static/feniks_debug/transport_resolution.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
