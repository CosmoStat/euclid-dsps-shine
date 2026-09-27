"""Read small synchronized parent results; never require simulation banks or q."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, sha, write


def verify_available(root: Path) -> dict:
    digest = sha(root / "MANIFEST.json")
    verified, unavailable = [], []
    for stage in ("reference", "qualification", "population", "report"):
        receipt = read(root / stage / "FINAL.json")
        if receipt["contract"] != digest or receipt["status"] != "COMPLETE":
            raise ValueError(f"Incompatible receipt: {stage}")
        for name, expected in receipt["artifacts"].items():
            path = (root / stage / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Artifact escapes run: {name}")
            relative = str(path.relative_to(root.resolve()))
            if not path.is_file():
                unavailable.append(relative)
            elif sha(path) != expected:
                raise ValueError(f"Artifact changed: {relative}")
            else:
                verified.append(relative)
    return dict(contract=digest, verified=verified, unavailable=unavailable)


def weight_checks(parent: dict) -> dict:
    u, v, alpha = (np.asarray(parent[key], dtype=float) for key in ("u", "v", "alpha"))
    if (
        u.ndim != 1
        or not len(u)
        or u.shape != v.shape
        or u.shape != alpha.shape
        or not all(np.isfinite(values).all() for values in (u, v, alpha))
        or np.any(u < 0)
        or np.any(v < 0)
        or np.any(alpha <= 0)
        or np.any(alpha > 1)
    ):
        raise ValueError("Invalid parent/selected weights or selection probabilities")
    reconstructed = v / alpha
    reconstructed /= reconstructed.sum()
    if not np.isclose(u.sum(), 1) or not np.isclose(v.sum(), 1):
        raise ValueError("Weights are not normalized")
    if not np.allclose(u, reconstructed, rtol=1e-9, atol=1e-12):
        raise ValueError("Parent weights disagree with selection correction")
    if not np.isclose(u @ alpha, parent["alpha_parent"], rtol=1e-9, atol=1e-12):
        raise ValueError("Parent alpha disagrees with component mixture")
    return dict(
        parent_sum=float(u.sum()),
        selected_sum=float(v.sum()),
        reconstruction_max_error=float(abs(u - reconstructed).max()),
        alpha_from_components=float(u @ alpha),
        alpha_min=float(alpha.min()),
        alpha_max=float(alpha.max()),
        effective_parent_components=float(1 / (u @ u)),
        maximum_parent_weight=float(u.max()),
        top_three_parent_mass=float(np.sort(u)[-3:].sum()),
    )


def run(root: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    provenance = verify_available(root)
    checks = weight_checks(read(root / "population/parent.json"))
    decision = read(root / "report/DECISION.json")
    audit = read(root / "population/classifier_audit.json")
    joint = pd.read_csv(root / "report/joint.csv")
    marginals = pd.read_csv(root / "report/marginals.csv")
    predictive = pd.read_csv(root / "report/observable_predictive_comparison.csv")
    fits = pd.read_csv(root / "population/regularization.csv")
    history = pd.read_json(root / "population/training.jsonl", lines=True)
    history = history.drop_duplicates("epoch", keep="last").sort_values("epoch")
    best, last = history.loc[history.validation_nll.idxmin()], history.iloc[-1]
    summary = dict(
        provenance=provenance,
        normalization=checks,
        classifier=dict(
            best_epoch=int(best.epoch),
            best_validation_nll=float(best.validation_nll),
            last_epoch=int(last.epoch),
            last_train_nll=float(last.train_nll),
            last_validation_nll=float(last.validation_nll),
            **audit,
        ),
        largest_kkt_gap=float(fits.kkt_gap.max()),
        selected_penalty=float(fits.loc[fits.selected, "strength"].item()),
        joint=joint.to_dict("records"),
        decision=decision,
        ready_for_production=False,
        limitations="Only available small artifacts verified; no bank, model or joint-flux readback.",
    )
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    write(out / "SUMMARY.json", summary)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    short = ["Redshift", "log mass", "log metallicity", "Dust Av", "Dust slope"]
    ph = marginals[marginals.group.eq("physical")]
    for population, values in ph.groupby("population", sort=False):
        axes[0, 0].plot(short, values.w1_over_truth_iqr, "o-", label=population)
    axes[0, 0].axhline(0.1, color="black", ls="--", label="marginal gate")
    axes[0, 0].set(
        title="Physical marginals: dust is the largest error", ylabel="W1 / truth IQR"
    )
    axes[0, 0].legend(fontsize=7)
    labels = ["parent", "selected", "capacity_parent", "capacity_selected"]
    values = joint[joint.group.eq("physical")].set_index("population").loc[labels]
    axes[0, 1].barh(
        labels,
        values.sliced_wasserstein,
        color=["#277da8", "#59a5c6", "#418452", "#83b087"],
    )
    axes[0, 1].axvline(0.05, color="black", ls="--")
    axes[0, 1].set(
        title="Joint physical discrepancy", xlabel="Sliced Wasserstein / truth IQR"
    )
    for key in ("train_nll", "validation_nll", "best_nll"):
        axes[1, 0].plot(history.epoch, history[key], label=key)
    axes[1, 0].axvline(best.epoch, color="black", ls=":")
    axes[1, 0].set(
        title=f"Classifier: best validation at epoch {int(best.epoch)}",
        xlabel="Epoch",
        ylabel="NLL",
    )
    axes[1, 0].legend(fontsize=8)
    for label, values in predictive.groupby("model", sort=False):
        axes[1, 1].plot(
            range(len(values)),
            100 * values.predicted_mass_above_truth_q999,
            "o-",
            label=label,
        )
    values = predictive[predictive.model.eq("learned_parent")]
    axes[1, 1].plot(
        range(len(values)), 100 * values.truth_mass_above_q999, "k:", label="target"
    )
    axes[1, 1].axhline(0.5, color="black", ls="--", label="tail-count gate")
    axes[1, 1].set_xticks(
        range(len(values)), values.band, rotation=65, ha="right", fontsize=7
    )
    axes[1, 1].set(
        title="Bright-tail frequency (not tail amplitude)",
        ylabel="Percent above target q99.9",
    )
    axes[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "diagnosis_summary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for population, values in marginals[marginals.group.eq("sfh")].groupby(
        "population", sort=False
    ):
        axes[0].plot(range(1, 11), values.w1_over_truth_iqr, "o-", label=population)
    axes[0].set(
        title="SFH population discrepancies remain",
        xlabel="SFH contrast",
        ylabel="W1 / truth IQR",
    )
    axes[0].legend(fontsize=7)
    for label, values in predictive.groupby("model", sort=False):
        axes[1].plot(range(len(values)), values.raw_w1_over_iqr, "o-", label=label)
        axes[1].plot(
            range(len(values)),
            values.upper_capped_diagnostic_w1_over_iqr,
            ":",
            label=f"{label}: diagnostic upper cap",
        )
    axes[1].set_xticks(
        range(len(values)), values.band, rotation=65, ha="right", fontsize=7
    )
    axes[1].set(
        title="Flux tail amplitude: not gated by PASS",
        ylabel="Flux W1 / max(IQR, 1e-30)",
        yscale="log",
    )
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "sfh_and_flux_tails.png", dpi=160)
    plt.close(fig)
    print(
        f"Verified {len(provenance['verified'])} artifacts; {len(provenance['unavailable'])} unavailable."
    )
    print(out / "diagnosis_summary.png")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    run(parser.parse_args().root.resolve())
