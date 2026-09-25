"""Read-only analysis of the small rsync bundle; never substitutes missing draws."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, sha, write


def verify(root):
    manifest = read(root / "MANIFEST.json")
    digest = sha(root / "MANIFEST.json")
    present, missing = [], []
    listings = [(root, manifest["frozen_files"])]
    for name in ("physical", "sfh_conditional", "sfh_zeros", "report"):
        receipt = read(root / name / "FINAL.json")
        if receipt["status"] != "COMPLETE" or receipt["contract"] != digest:
            raise ValueError(f"Invalid completion contract: {name}")
        listings.append((root / name, receipt["artifacts"]))
    for directory, artifacts in listings:
        for name, checksum in artifacts.items():
            p = directory / name
            if not p.exists():
                missing.append(str(p.relative_to(root)))
            elif sha(p) != checksum:
                raise ValueError(f"Artifact hash mismatch: {p}")
            else:
                present.append(str(p.relative_to(root)))
    return dict(verified=present, excluded_or_missing=missing)


def convergence(history):
    recent = history.tail(20)
    best = history.loc[history.validation_nll.idxmin()]
    return dict(
        epochs=int(history.epoch.iloc[-1]),
        best_epoch=int(best.epoch),
        best_nll=float(best.validation_nll),
        last_nll=float(history.validation_nll.iloc[-1]),
        best_gain_last_20=float(
            history.best_nll.iloc[max(0, len(history) - 21)] - history.best_nll.iloc[-1]
        ),
        recent_validation_std=float(recent.validation_nll.std()),
        recent_validation_min=float(recent.validation_nll.min()),
        recent_validation_max=float(recent.validation_nll.max()),
        recent_train_validation_gap_median=float(
            np.median(recent.validation_nll - recent.train_nll)
        ),
        final_learning_rate=float(history.learning_rate.iloc[-1]),
    )


def run(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    integrity = verify(root)
    report = root / "report"
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    histories = {
        s: pd.read_json(root / s / "training.jsonl", lines=True)
        for s in ("physical", "sfh_conditional")
    }
    summaries = {s: convergence(h) for s, h in histories.items()}
    marginal = pd.read_csv(report / "marginals.csv")
    joint = pd.read_csv(report / "joint.csv")
    zero = pd.read_csv(report / "sfh_zero_neighborhoods.csv")
    truth = pd.read_csv(report / "truth_spearman.csv", index_col=0)
    flow = pd.read_csv(report / "flow_spearman.csv", index_col=0)
    delta = flow - truth
    errors = []
    for i in range(15):
        for j in range(i):
            errors.append(
                dict(
                    parameter_a=truth.columns[i],
                    parameter_b=truth.columns[j],
                    truth=float(truth.iloc[i, j]),
                    flow=float(flow.iloc[i, j]),
                    absolute_error=float(abs(delta.iloc[i, j])),
                )
            )
    pd.DataFrame(errors).sort_values("absolute_error", ascending=False).to_csv(
        out / "correlation_errors.csv", index=False
    )
    pd.DataFrame(summaries).T.to_csv(out / "convergence.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, (name, h) in enumerate(histories.items()):
        for row, start in ((0, 1), (1, 91)):
            part = h[h.epoch >= start]
            for field, color in (
                ("train_nll", "#3285ad"),
                ("validation_nll", "#c06428"),
                ("best_nll", "#238960"),
            ):
                axes[row, col].plot(part.epoch, part[field], label=field, color=color)
            axes[row, col].axvline(
                summaries[name]["best_epoch"], color="#444444", ls=":", lw=1
            )
            axes[row, col].set_title(
                name + (" | epochs 91-120" if row else " | full trajectory")
            )
            axes[row, col].set_xlabel("Epoch")
            axes[row, col].set_ylabel("NLL; lower is better")
            axes[row, col].legend(fontsize=8)
            axes[row, col].grid(alpha=0.15)
    fig.suptitle("Fixed validation set: fluctuations are not validation resampling")
    fig.tight_layout()
    fig.savefig(out / "convergence_zoom.png", dpi=150)
    plt.close(fig)
    labels = ["z", "logM", "logZ", "Av", "delta"] + [
        f"SFH{i:02d}" for i in range(1, 11)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, array, title, limit in zip(
        axes,
        (truth, flow, delta),
        ("Held-out parent truth", "Structured flow", "Flow minus truth"),
        (1, 1, 0.1),
        strict=True,
    ):
        im = ax.imshow(array, vmin=-limit, vmax=limit, cmap="RdBu_r")
        ax.set_title(title)
        ax.set_xticks(range(15), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(15), labels, fontsize=8)
        ax.axhline(4.5, color="black", lw=0.8)
        ax.axvline(4.5, color="black", lw=0.8)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(
        "Spearman correlations: useful dependence check, not full joint validation"
    )
    fig.tight_layout()
    fig.savefig(out / "dependence.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for k, group in enumerate(("physical", "sfh")):
        m = marginal[marginal.group == group]
        names = m[m.space == "theta"].parameter.tolist()
        idx = np.arange(len(names))
        for shift, space, color in (
            (-0.18, "theta", "#a43d64"),
            (0.18, "x", "#3285ad"),
        ):
            axes[k].bar(
                idx + shift,
                m[m.space == space]
                .set_index("parameter")
                .loc[names, "w1_over_truth_iqr"],
                width=0.36,
                color=color,
                label=space,
            )
        axes[k].set_xticks(idx, labels[:5] if k == 0 else labels[5:], rotation=45)
        axes[k].set_ylabel("Marginal W1 / held-out IQR")
        axes[k].set_title(group + " representation error")
        axes[k].legend()
    idx = np.arange(10)
    axes[2].bar(
        idx - 0.18,
        100 * zero.truth_within_001_iqr,
        width=0.36,
        label="Truth",
        color="#333333",
    )
    axes[2].bar(
        idx + 0.18,
        100 * zero.flow_within_001_iqr,
        width=0.36,
        label="Flow",
        color="#a43d64",
    )
    axes[2].set_xticks(idx, labels[5:], rotation=45)
    axes[2].set_ylabel("Samples within 0.01 IQR of zero (%)")
    axes[2].set_title("SFH: missing mass near zero")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "errors_and_zeros.png", dpi=150)
    plt.close(fig)
    summary = dict(
        integrity=integrity,
        convergence=summaries,
        transform_max_roundtrip_over_iqr=float(
            pd.read_csv(root / "transform_checks.csv").max_roundtrip_over_iqr.max()
        ),
        joint=joint.to_dict(orient="records"),
        maximum_physical_correlation_error=float(
            np.nanmax(abs(delta.to_numpy()[:5, :5]))
        ),
        maximum_cross_correlation_error=float(np.nanmax(abs(delta.to_numpy()[:5, 5:]))),
        maximum_sfh_correlation_error=float(np.nanmax(abs(delta.to_numpy()[5:, 5:]))),
        zero_floor_attribution_valid=False,
        zero_floor_issue="Audit threshold -30 misses upstream SFR floor 1e-14; replay agreement unaffected",
        tail_draw_counts_available=(report / "draws.npz").exists(),
        production_ready=False,
    )
    write(out / "SUMMARY.json", summary)
    print(out)
    print(pd.DataFrame(summaries).T.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root.resolve())
