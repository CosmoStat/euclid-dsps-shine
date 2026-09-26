"""Compare a truth-supervised control with genuinely decoupled population NPE."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, write
from scripts.feniks_coherent_parent import complete, finish


def run(root):
    import matplotlib

    from euclid_dsps.amortized.coherent_coordinates import to_theta
    from euclid_dsps.amortized.native_reference import sample_basis, supervised_weights
    from scripts.feniks_coherent_inference import (
        NAMES,
        load_bank,
        require_reference,
        settings,
    )
    from scripts.report_feniks_forward_population import population_metrics

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m, cfg, digest = settings(root)
    out = root / "report"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    missing = [
        stage
        for stage in ("reference", "oracle", "population", "posterior")
        if not complete(root / stage, digest)
    ]
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        (root / "ROADMAP_STATUS.md").write_text(
            "# Coherent inference\n\nIncomplete: "
            + ", ".join(missing)
            + "\nNo scientific validation.\n"
        )
        return
    basis, spec, _ = require_reference(root, digest)
    source = Path(m["source"])
    parent = read(root / "population/parent.json")
    u = np.asarray(parent["u"])
    rng = np.random.default_rng(cfg["seed"] + 333)
    n = cfg["evaluation"]["population_draws"]
    labels = rng.choice(len(u), n, p=u)
    learned = np.asarray(to_theta(sample_basis(basis, labels, cfg["seed"] + 444), spec))
    ref = np.asarray(
        to_theta(
            sample_basis(basis, rng.integers(len(u), size=n), cfg["seed"] + 444), spec
        )
    )
    truth_parent = pd.read_parquet(source / "dataset/parent/test.parquet")
    truth_selected = truth_parent.loc[truth_parent.selected_r29, NAMES].to_numpy()
    data = load_bank(root, cfg, digest)
    audit = data["selected"] & (data["role"] == 4)
    indices = np.flatnonzero(audit)
    weights = supervised_weights(data["component"][audit], u, np.ones(len(u)) / len(u))
    chosen = rng.choice(indices, n, p=weights / weights.sum())
    selected = data["theta"][chosen]
    joint, marginals = [], []
    series = [
        ("reference_parent", ref, truth_parent[NAMES].to_numpy()),
        ("learned_parent", learned, truth_parent[NAMES].to_numpy()),
        ("learned_selected", selected, truth_selected),
    ]
    # Empirical finite-catalogue baseline is reported, never subtracted from SW.
    validation = pd.read_parquet(
        source / "dataset/parent/validation.parquet", columns=NAMES
    ).to_numpy()
    series.append(
        ("validation_vs_test_parent", validation, truth_parent[NAMES].to_numpy())
    )
    for name, pred, truth in series:
        one, multi = population_metrics(pred, truth, NAMES, cfg["seed"])
        marginals.append(one.assign(comparison=name))
        joint.append(multi.assign(comparison=name))
    pd.concat(joint).to_csv(out / "population_joint.csv", index=False)
    pd.concat(marginals).to_csv(out / "population_marginals.csv", index=False)
    np.savez(
        out / "population_draws.npz", reference=ref, parent=learned, selected=selected
    )
    from scripts.feniks_avi_overnight import _corner

    _corner(
        out / "parent_physical_corner.png",
        {"Learned parent": learned[:, :5], "Reference": ref[:, :5]},
        truth_parent[NAMES[:5]].to_numpy(),
        NAMES[:5],
    )
    cal = pd.concat(
        [
            pd.read_csv(root / stage / "calibration.csv").assign(model=stage)
            for stage in ("oracle", "posterior")
        ]
    )
    cal.to_csv(out / "calibration_comparison.csv", index=False)
    in_model = pd.read_csv(root / "posterior/in_model_calibration.csv")
    in_model.to_csv(out / "in_model_calibration.csv", index=False)
    fig, axes = plt.subplots(3, 5, figsize=(17, 9))
    for j, ax in enumerate(axes.flat):
        combined = np.r_[truth_parent[NAMES[j]].to_numpy(), learned[:, j], ref[:, j]]
        lo, hi = np.quantile(combined, [0.001, 0.999])
        edges = np.linspace(lo, hi if hi > lo else lo + 1, 65)
        for label, values in (
            ("True parent", truth_parent[NAMES[j]].to_numpy()),
            ("Reference", ref[:, j]),
            ("Learned parent", learned[:, j]),
            ("Learned selected", selected[:, j]),
        ):
            # Whole-distribution normalization: displayed tails are not renormalized.
            density = np.histogram(values, edges)[0] / (len(values) * np.diff(edges))
            ax.stairs(density, edges, label=label)
        ax.set_title(NAMES[j], fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "parent_selected_reference_15d.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for level, ax in zip((68, 95), axes, strict=True):
        for stage in ("oracle", "posterior"):
            values = cal.loc[cal.model.eq(stage)].set_index("parameter").loc[NAMES]
            ax.plot(np.arange(15), values[f"coverage_{level}"], "o-", label=stage)
        ax.axhline(level / 100, color="black", ls="--")
        ax.plot(
            np.arange(15),
            in_model.set_index("parameter").loc[NAMES][f"coverage_{level}"],
            "s--",
            label="posterior on learned-parent simulations",
        )
        ax.axvline(4.5, color="gray")
        ax.set_ylabel(f"{level}% coverage")
        ax.set_xticks(np.arange(15), NAMES, rotation=30, ha="right", fontsize=8)
        ax.legend()
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "coverage_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    for stage in ("oracle", "posterior"):
        with np.load(root / stage / "draws.npz") as f:
            for j, ax in enumerate(axes.flat):
                ax.hist(
                    f["ranks"][:, j],
                    bins=np.linspace(0, 1, 11),
                    density=True,
                    histtype="step",
                    label=stage,
                )
                ax.set_title(NAMES[j], fontsize=8)
    for ax in axes.flat:
        ax.axhline(1, color="black", ls="--", lw=0.5)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "pit_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    convergence = {}
    for stage, ax in zip(("population", "oracle", "posterior"), axes, strict=True):
        history = pd.read_json(root / stage / "training.jsonl", lines=True)
        for name in ("train_nll", "validation_nll", "best_nll"):
            ax.plot(history.epoch, history[name], label=name)
        convergence[stage] = read(root / stage / "STOP.json")
        ax.set_title(stage)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=150)
    plt.close(fig)
    bands = [b["name"] for b in read(source / "decoder.json")["bands"]]
    from scipy.stats import wasserstein_distance

    predictive = []
    for j, band in enumerate(bands):
        actual = truth_parent.loc[truth_parent.selected_r29, f"flux_{band}"].to_numpy()
        scale = max(float(np.subtract(*np.quantile(actual, [0.75, 0.25]))), 1e-30)
        predictive.append(
            dict(
                band=band,
                w1_over_iqr=float(
                    wasserstein_distance(data["flux"][chosen, j], actual) / scale
                ),
            )
        )
    pd.DataFrame(predictive).to_csv(out / "observable_predictive.csv", index=False)
    alpha = dict(
        learned=parent["alpha_parent"], test=float(truth_parent.selected_r29.mean())
    )
    text = """# Coherent inference benchmark

1. `training.png`: checkpoint NLL and STOP.json plateau flags; budget completion is not convergence.
2. `parent_selected_reference_15d.png`: true parent, independent reference, learned parent, learned selected.
3. `population_joint.csv`: physical and SFH sliced-Wasserstein, with validation/test sampling baseline.
4. `coverage_comparison.png`, `pit_comparison.png`: identical held-out galaxies, direct supervised control versus learned-parent NPE.
5. `observable_predictive.csv`: predicted selected photometry compared with held-out catalogue.
6. `in_model_calibration.csv`: final q on the reserved, importance-resampled learned-parent bank.
   If this fails, amortization is not calibrated even under its own training model.
   If this passes but target-test calibration fails, investigate the learned parent/reference family.
   Unique row counts are recorded; finite-bank resampling is not new independent simulation.

The oracle uses target training labels and is NOT blind population recovery.
The population branch uses observed photometry, native reference simulations and explicit selection correction only.
The 15D reference is a smoothed finite native family. Its SFH conditional/support remain assumptions.
Posterior training reuses a finite importance-weighted simulator bank, never q-generated targets.
Stored SFH atoms are untouched; continuous flows may approximate their masses imperfectly.
This report is benchmark evidence, not automatic production promotion or validation on a real survey.
"""
    (out / "REPORT.md").write_text(text)
    (root / "ROADMAP_STATUS.md").write_text(
        "# Roadmap\n\n- Coherent dataset: verified unchanged.\n"
        "- Independent reference / selection-corrected population: execution complete, inspect closure.\n"
        "- Supervised 15D posterior control and learned-parent branch: execution complete.\n"
        "- Classifier/posterior convergence: " + json_status(convergence) + "\n"
        "- Real catalogue and paper-level population recovery: NOT VALIDATED.\n"
        "- Next decision: separate reference-family error from amortization error using this comparison.\n"
    )
    (out / "BLOCKED.json").unlink(missing_ok=True)
    finish(
        out,
        sorted(p for p in out.iterdir() if p.is_file()),
        digest,
        alpha=alpha,
        convergence=convergence,
        population_uses_q=False,
        dimensions=15,
        ready_for_production=False,
    )


def json_status(convergence):
    return "; ".join(
        f"{k}: {v['reason']} at {v['epoch']}" for k, v in convergence.items()
    )
