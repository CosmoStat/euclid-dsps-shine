"""One bounded CPU audit of saved reference capacity and observable tails.

No DSPS, classifier, q sampling, neural training or production prior mutation.
Capacity fitting uses parent TRAIN truth; held-out data only evaluate the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from euclid_dsps.amortized.coherent_coordinates import to_theta, to_x
from euclid_dsps.amortized.native_reference import sample_basis, supervised_weights
from euclid_dsps.amortized.reference_capacity import (
    cdf_design,
    empirical_features,
    fit_cdf_weights,
    observable_tail_metrics,
)
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_inference import NAMES, require_reference, settings
from scripts.feniks_coherent_parent import complete, finish
from scripts.report_feniks_forward_population import population_metrics


def capacity(root, out, m, cfg, digest, contract, seed, draws):
    out.mkdir(exist_ok=True)
    if complete(out, contract):
        return
    basis, spec, _ = require_reference(root, digest)
    source = Path(m["source"])
    write(out / "PROGRESS.json", dict(stage="analytic_component_cdfs"))
    train = pd.read_parquet(
        source / "dataset/parent/train.parquet", columns=NAMES
    ).to_numpy()
    f, y, directions, grid = cdf_design(
        basis, np.asarray(to_x(train, spec))[:, :5], seed=seed
    )
    u, diagnostics = fit_cdf_weights(f, y)
    learned = np.asarray(read(root / "population/parent.json")["u"])
    write(out / "truth_diagnostic_weights.json", dict(u=u.tolist(), **diagnostics))
    np.savez(
        out / "cdf_features.npz",
        components=f,
        train=y,
        directions=directions,
        thresholds=grid,
    )
    # Same fixed directions/thresholds for every split. No validation/test tuning.
    validation = pd.read_parquet(
        source / "dataset/parent/validation.parquet", columns=NAMES
    ).to_numpy()
    test = pd.read_parquet(
        source / "dataset/parent/test.parquet", columns=NAMES
    ).to_numpy()
    cdf_rows = []
    for split, theta in (("train", train), ("validation", validation), ("test", test)):
        truth = empirical_features(
            np.asarray(to_x(theta, spec))[:, :5], directions, grid
        )
        for label, weights in (
            ("learned_photometric", learned),
            ("truth_cdf_capacity", u),
        ):
            residual = f @ weights - truth
            cdf_rows.append(
                dict(
                    split=split,
                    model=label,
                    max_cdf_error=float(abs(residual).max()),
                    mean_cdf_error=float(abs(residual).mean()),
                )
            )
    baseline = empirical_features(
        np.asarray(to_x(validation, spec))[:, :5], directions, grid
    )
    truth = empirical_features(np.asarray(to_x(test, spec))[:, :5], directions, grid)
    cdf_rows.append(
        dict(
            split="test",
            model="validation_vs_test",
            max_cdf_error=float(abs(baseline - truth).max()),
            mean_cdf_error=float(abs(baseline - truth).mean()),
        )
    )
    pd.DataFrame(cdf_rows).to_csv(out / "cdf_errors.csv", index=False)
    write(out / "PROGRESS.json", dict(stage="heldout_distributions"))
    rng = np.random.default_rng(seed)
    predictions = {}
    for label, weights in (("learned_photometric", learned), ("truth_cdf_capacity", u)):
        labels = rng.choice(len(u), draws, p=weights)
        predictions[label] = np.asarray(
            to_theta(sample_basis(basis, labels, seed + 1), spec)
        )
    marginals, joints = [], []
    for label, values in [*predictions.items(), ("validation_vs_test", validation)]:
        one, joint = population_metrics(values, test, NAMES, cfg["seed"])
        marginals.append(one.assign(model=label))
        joints.append(joint.assign(model=label))
    pd.concat(marginals).to_csv(out / "marginals.csv", index=False)
    pd.concat(joints).to_csv(out / "joint.csv", index=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for j, ax in enumerate(axes):
        lo, hi = np.quantile(test[:, j], [0.001, 0.999])
        edges = np.linspace(lo, hi, 65)
        for label, values in [("True test parent", test), *predictions.items()]:
            density = np.histogram(values[:, j], edges)[0] / (
                len(values) * np.diff(edges)
            )
            ax.stairs(density, edges, label=label)
        ax.set_title(NAMES[j], fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "TRAIN-truth capacity diagnostic, not a deployable population estimate"
    )
    fig.tight_layout()
    fig.savefig(out / "capacity_physical.png", dpi=160)
    plt.close(fig)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        contract,
        **diagnostics,
        draws=draws,
        physical_features=len(y),
        limitation="Finite projected CDF objective, not minimum SW or full joint capacity proof",
    )


def selected_audit_bank(root, cfg, digest):
    """Stream reserved rows only; do not load the million-row training bank."""
    flux, labels = [], []
    for task in range(cfg["bank"]["shards"]):
        shard = root / "banks" / f"shard_{task:03d}"
        if not complete(shard, digest):
            raise ValueError(f"Incomplete shard: {shard}")
        for start in range(
            0, cfg["bank"]["rows_per_shard"], cfg["bank"]["checkpoint_rows"]
        ):
            block = shard / f"block_{start:07d}"
            if not complete(block, digest):
                raise ValueError(f"Incomplete block: {block}")
            with np.load(block / "bank.npz") as data:
                mask = data["selected"] & (data["role"] == 4)
                flux.append(data["flux"][mask])
                labels.append(data["component"][mask])
    return np.concatenate(flux), np.concatenate(labels)


def tails(root, out, m, cfg, digest, contract):
    out.mkdir(exist_ok=True)
    if complete(out, contract):
        return
    write(out / "PROGRESS.json", dict(stage="reading_reserved_bank"))
    flux, labels = selected_audit_bank(root, cfg, digest)
    source = Path(m["source"])
    bands = [b["name"] for b in read(source / "decoder.json")["bands"]]
    target = pd.read_parquet(
        source / "dataset/selected_r29/test.parquet",
        columns=[f"flux_{b}" for b in bands],
    )
    u = np.asarray(read(root / "population/parent.json")["u"])
    weights = supervised_weights(labels, u, np.ones(len(u)) / len(u))
    weights /= weights.sum()
    metrics, components = [], []
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        int(np.ceil(len(bands) / 3)),
        3,
        figsize=(14, 3 * int(np.ceil(len(bands) / 3))),
        squeeze=False,
    )
    for j, band in enumerate(bands):
        t, p = target[f"flux_{band}"].to_numpy(), flux[:, j]
        row = observable_tail_metrics(p, t, weights)
        metrics.append(dict(band=band, **row))
        for component in range(len(u)):
            mask = labels == component
            components.append(
                dict(
                    band=band,
                    component=component,
                    selected_mass=float(weights[mask].sum()),
                    upper_tail_mass=float(
                        weights[mask & (p > row["truth_q999"])].sum()
                    ),
                    upper_excess_over_iqr=float(
                        weights[mask]
                        @ np.maximum(p[mask] - row["truth_q999"], 0)
                        / row["truth_flux_iqr"]
                    ),
                )
            )
        ax = axes.flat[j]
        for label, values, w in (
            ("Target test", t, np.ones(len(t)) / len(t)),
            ("Learned parent predictive", p, weights),
        ):
            order = np.argsort(values)
            ax.plot(
                np.arcsinh(values[order] / row["truth_flux_iqr"]),
                np.cumsum(w[order]),
                label=label,
            )
        ax.set(title=band, xlabel="asinh(flux / target IQR)", ylabel="CDF", ylim=(0, 1))
    for ax in list(axes.flat)[len(bands) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "observable_cdfs.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(out / "observable_tails.csv", index=False)
    pd.DataFrame(components).to_csv(
        out / "component_tail_contributions.csv", index=False
    )
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        contract,
        independent_reference_rows=len(flux),
        importance_effective_rows=float(1 / (weights @ weights)),
        resampling_used=False,
        clipping_used_for_inference=False,
        limitation="Upper-capped W1 is diagnostic only; full raw and asinh metrics retain all rows",
    )


def run(root, out, seed=20260927, draws=20000):
    if root == out or root in out.parents or out in root.parents or draws < 100:
        raise ValueError("Use a separate audit root and at least 100 draws")
    m, cfg, digest = settings(root)
    parent_path = root / "population/parent.json"
    receipt = read(root / "population/FINAL.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["contract"] != digest
        or sha(parent_path) != receipt["artifacts"]["parent.json"]
    ):
        raise ValueError("Invalid saved parent")
    # Bind reuse to immutable source data, reference basis, parent, parameters and code.
    manifest = dict(
        source=str(root),
        source_contract=digest,
        parent_sha256=sha(parent_path),
        reference_receipt_sha256=sha(root / "reference/FINAL.json"),
        code_sha256=(out / "CODE_SHA256").read_text().strip()
        if (out / "CODE_SHA256").exists()
        else sha(Path(__file__)),
        helper_sha256=sha(
            Path(__file__).parents[1] / "euclid_dsps/amortized/reference_capacity.py"
        ),
        seed=seed,
        draws=draws,
        random_directions=16,
        thresholds=25,
        truth_role="TRAIN-only diagnostic oracle, never production",
        new_simulations=0,
        neural_training=0,
    )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "MANIFEST.json").exists() and read(out / "MANIFEST.json") != manifest:
        raise ValueError("Audit inputs changed; use a new output root")
    write(out / "MANIFEST.json", manifest)
    contract = sha(out / "MANIFEST.json")
    capacity(root, out / "capacity", m, cfg, digest, contract, seed, draws)
    tails(root, out / "tails", m, cfg, digest, contract)
    report = out / "report"
    report.mkdir(exist_ok=True)
    (report / "REPORT.md").write_text(
        "# Coherent reference diagnosis\n\n"
        "1. capacity/cdf_errors.csv: exact projected component CDFs, fitted ONLY on TRAIN.\n"
        "2. capacity/joint.csv and capacity_physical.png: 15D draws, independent test evaluation.\n"
        "3. tails/observable_tails.csv and observable_cdfs.png: exact weighted reserved-bank flux distributions.\n"
        "4. tails/component_tail_contributions.csv: components responsible for bright excess.\n\n"
        "A large optimized TRAIN CDF residual establishes a restriction of this fixed family on those features. "
        "A small residual does not prove full 15D capacity. If the truth-fit family recovers test marginals "
        "but the photometric fit does not, investigate ratios/inversion/identifiability before changing the family.\n"
        "A strong raw-vs-upper-capped W1 difference indicates upper-tail sensitivity, not permission to clip data.\n"
        "No new model has been validated for production. Joint/conditional posterior coverage remains a separate gate.\n"
    )
    finish(
        report,
        [report / "REPORT.md", out / "capacity/FINAL.json", out / "tails/FINAL.json"],
        contract,
        ready_for_production=False,
        production_prior_modified=False,
    )
    print(f"Audit complete: {report}")


def main():
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20000)
    args = parser.parse_args()
    run(args.source.resolve(), args.root.resolve(), draws=args.draws)


if __name__ == "__main__":
    main()
