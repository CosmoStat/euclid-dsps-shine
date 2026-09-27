"""Read downloaded small results without requiring banks or checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, sha, write


def verify_available(root):
    digest = sha(root / "MANIFEST.json")
    present, missing = [], []
    for stage in ("reference", "population", "oracle", "posterior", "report"):
        receipt = read(root / stage / "FINAL.json")
        if receipt["contract"] != digest or receipt["status"] != "COMPLETE":
            raise ValueError(f"Incompatible receipt: {stage}")
        for name, expected in receipt["artifacts"].items():
            path = root / stage / name
            if not path.is_file():
                missing.append(f"{stage}/{name}")
            elif sha(path) != expected:
                raise ValueError(f"Artifact changed: {path}")
            else:
                present.append(f"{stage}/{name}")
    return dict(contract=digest, verified=present, unavailable=missing)


def run(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    provenance = verify_available(root)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    parent = read(root / "population/parent.json")
    u, v, alpha = (np.asarray(parent[k]) for k in ("u", "v", "alpha"))
    reconstructed = v / alpha
    reconstructed /= reconstructed.sum()
    if not np.allclose(u.sum(), 1) or not np.allclose(v.sum(), 1):
        raise ValueError("Non-normalized weights")
    closure = pd.read_csv(root / "report/population_joint.csv")
    cal = pd.read_csv(root / "report/calibration_comparison.csv")
    physical = cal[cal.group.eq("physical")].copy()
    own = pd.read_csv(root / "report/in_model_calibration.csv")
    own = own[own.group.eq("physical")].assign(model="posterior_in_model")
    physical = pd.concat([physical, own], ignore_index=True)
    physical.to_csv(out / "physical_calibration.csv", index=False)
    histories, rows = {}, []
    for stage in ("population", "oracle", "posterior"):
        h = pd.read_json(root / stage / "training.jsonl", lines=True)
        h = h.drop_duplicates("epoch", keep="last").sort_values("epoch")
        histories[stage] = h
        best, last = h.loc[h.validation_nll.idxmin()], h.iloc[-1]
        rows.append(
            dict(
                stage=stage,
                best_epoch=int(best.epoch),
                best_validation_nll=float(best.validation_nll),
                last_epoch=int(last.epoch),
                last_train_nll=float(last.train_nll),
                last_validation_nll=float(last.validation_nll),
                stop_reason=read(root / stage / "STOP.json")["reason"],
            )
        )
    pd.DataFrame(rows).to_csv(out / "checkpoint_summary.csv", index=False)
    summary = dict(
        provenance=provenance,
        checkpoints=rows,
        parent_weight_sum=float(u.sum()),
        selected_weight_sum=float(v.sum()),
        selection_reconstruction_max_error=float(np.max(np.abs(u - reconstructed))),
        effective_parent_components=float(1 / (u @ u)),
        top_three_parent_mass=float(np.sort(u)[-3:].sum()),
        physical_coverage=physical.groupby("model")[["coverage_68", "coverage_95"]]
        .mean()
        .to_dict("index"),
        ready_for_production=False,
        unavailable_note="Missing dense draws/checkpoints/banks were not independently verified.",
    )
    write(out / "SUMMARY.json", summary)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    names = physical.parameter.drop_duplicates().tolist()
    short = ["Redshift", "log mass", "log metallicity", "Av", "Dust slope"]
    for model, values in physical.groupby("model", sort=False):
        vals = values.set_index("parameter").loc[names]
        axes[0, 0].plot(short, vals.coverage_68, "o-", label=model)
    axes[0, 0].axhline(0.68, ls="--", c="black")
    axes[0, 0].set(title="68% physical marginal coverage", ylim=(0.5, 1))
    axes[0, 0].legend(fontsize=8)
    ph = closure.loc[closure.group.eq("physical")]
    axes[0, 1].barh(ph.comparison, ph.sliced_wasserstein, color="#328a84")
    axes[0, 1].set(xlabel="Physical sliced-Wasserstein / truth IQR", xscale="log")
    h = histories["posterior"]
    for key in ("train_nll", "validation_nll", "best_nll"):
        axes[1, 0].plot(h.epoch, h[key], label=key)
    axes[1, 0].axvline(rows[-1]["best_epoch"], color="black", ls=":")
    axes[1, 0].set(
        title="Final posterior: best checkpoint, then overfitting", xlabel="Epoch"
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(np.arange(len(u)), np.sort(u)[::-1], color="#d8863c")
    axes[1, 1].set(
        title="Parent weights (sorted); concentration is not proof of a bug",
        xlabel="Component rank",
        ylabel="Parent probability",
    )
    fig.tight_layout()
    fig.savefig(out / "diagnosis_summary.png", dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summary = run(args.root.resolve())
    print(f"Verified {len(summary['provenance']['verified'])} available artifacts.")
    print(args.root / "analysis/diagnosis_summary.png")


if __name__ == "__main__":
    main()
