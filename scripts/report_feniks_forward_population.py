"""Blind forward-population closure and direct, unweighted 15D NPE calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kstest, wasserstein_distance

from euclid_dsps.amortized.forward_population import PHYSICAL
from scripts.feniks_avi_experiments import read, sha, write


def calibration(draws, truth, names):
    """draws [object, draw, coordinate]; never replace a distribution by its median."""
    if draws.ndim != 3 or draws.shape[0] != len(truth) or draws.shape[2] != len(names):
        raise ValueError("calibration shapes disagree")
    q = np.quantile(draws, [0.025, 0.16, 0.5, 0.84, 0.975], axis=1)
    ranks = (np.sum(draws < truth[:, None], axis=1) + 0.5) / (draws.shape[1] + 1)
    rows = []
    for j, name in enumerate(names):
        rows.append(
            dict(
                parameter=name,
                group="physical" if name in PHYSICAL else "sfh",
                coverage_68=float(
                    np.mean((truth[:, j] >= q[1, :, j]) & (truth[:, j] <= q[3, :, j]))
                ),
                coverage_95=float(
                    np.mean((truth[:, j] >= q[0, :, j]) & (truth[:, j] <= q[4, :, j]))
                ),
                median_bias=float(np.median(q[2, :, j] - truth[:, j])),
                median_width_68=float(np.median(q[3, :, j] - q[1, :, j])),
                median_width_95=float(np.median(q[4, :, j] - q[0, :, j])),
                pit_ks=float(kstest(ranks[:, j], "uniform").statistic),
                objects=len(truth),
            )
        )
    return pd.DataFrame(rows), ranks, q[2]


def population_metrics(predicted, truth, names, seed=0):
    scale = np.maximum(
        np.quantile(truth, 0.75, axis=0) - np.quantile(truth, 0.25, axis=0), 1e-8
    )
    rows = [
        dict(
            parameter=name,
            group="physical" if name in PHYSICAL else "sfh",
            w1_over_truth_iqr=wasserstein_distance(predicted[:, j], truth[:, j])
            / scale[j],
        )
        for j, name in enumerate(names)
    ]
    rng = np.random.default_rng(seed)
    joint = []
    for group in ("physical", "sfh"):
        indices = [
            i
            for i, name in enumerate(names)
            if (name in PHYSICAL) == (group == "physical")
        ]
        directions = rng.normal(size=(64, len(indices)))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        p, t = (
            predicted[:, indices] / scale[indices],
            truth[:, indices] / scale[indices],
        )
        distances = [
            wasserstein_distance(p @ direction, t @ direction)
            for direction in directions
        ]
        joint.append(
            dict(
                group=group, sliced_wasserstein=float(np.mean(distances)), directions=64
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(joint)


def diagnostic_plots(directory, draws, truth, names, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame, ranks, med = calibration(draws, truth, names)
    frame.to_csv(directory / f"{label}_calibration.csv", index=False)
    for group in ("physical", "sfh"):
        indices = [
            i
            for i, name in enumerate(names)
            if (name in PHYSICAL) == (group == "physical")
        ]
        fig, axes = plt.subplots(
            2, len(indices), figsize=(3 * len(indices), 5), squeeze=False
        )
        for col, j in enumerate(indices):
            axes[0, col].hist(
                ranks[:, j], bins=np.linspace(0, 1, 11), density=True, color="#277c8e"
            )
            axes[0, col].axhline(1, color="black", ls="--")
            axes[0, col].set_title(names[j], fontsize=8)
            axes[1, col].scatter(
                truth[:, j], med[:, j], s=3, alpha=0.3, color="#ae4a56"
            )
            lo, hi = np.quantile(truth[:, j], [0.005, 0.995])
            axes[1, col].plot([lo, hi], [lo, hi], color="black", lw=1)
            axes[1, col].set_xlabel("Truth")
            axes[1, col].set_ylabel("Posterior median")
        fig.tight_layout()
        fig.savefig(directory / f"{label}_{group}_pit_truth.png", dpi=140)
        plt.close(fig)
    return frame


def report(root):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.forward_population_runtime import (
        load_forward_runtime,
        posterior_template,
    )
    from euclid_dsps.amortized.latent import x_to_theta
    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from scripts.feniks_avi_overnight import _corner, _marginals, _representative_rows
    from scripts.feniks_forward_population import (
        attach_frozen_parent,
        basis_for,
        contract,
        load_banks,
    )

    m, cfg = contract(root)
    out = root / "report"
    basis = basis_for(root)
    names = basis.names
    physical = [names.index(n) for n in PHYSICAL]
    parent = read(root / "population/parent.json")
    receipt = read(root / "posterior/FINAL.json")
    if receipt["parent_sha256"] != sha(root / "population/parent.json") or receipt[
        "checkpoint_sha256"
    ] != sha(root / "posterior/best.eqx"):
        raise ValueError("frozen parent/posterior integrity check failed")
    model, rt, _, config = load_forward_runtime(Path(m["source"]), out / "runtime", cfg)
    model = attach_frozen_parent(root, model)
    candidate = posterior_template(model, rt, config, cfg)
    candidate = eqx.tree_deserialise_leaves(root / "posterior/best.eqx", candidate)
    key = jax.random.PRNGKey(cfg["seed"] + 30000)

    @eqx.filter_jit
    def infer(net, features, k):
        x = sample_independent_mixture(model, net, k, features, cfg["report_draws"]).x
        return jnp.swapaxes(x_to_theta(x, rt.latent_spec), 0, 1)

    def posterior_draws(features):
        chunks = []
        for start in range(0, len(features), 16):
            idx = np.minimum(np.arange(start, start + 16), len(features) - 1)
            draws = np.asarray(
                infer(
                    candidate,
                    jnp.asarray(features[idx]),
                    jax.random.fold_in(key, start),
                )
            )
            chunks.append(draws[: min(16, len(features) - start)])
        return np.concatenate(chunks)

    bank = load_banks(
        root,
        "posterior_bank",
        [cfg["posterior_shards"]],
        ("theta", "features", "selected", "flux", "errors"),
    )
    # Second half is untouched by validation NLL / checkpoint selection.
    selected = np.flatnonzero(
        bank["selected"]
        & (np.arange(len(bank["selected"])) >= len(bank["selected"]) // 2)
    )[: cfg["report_objects"]]
    if len(selected) < 128:
        raise ValueError("insufficient independent posterior test cohort")
    simulated = posterior_draws(bank["features"][selected])
    np.savez(
        out / "simulation_posterior_15d.npz",
        draws=simulated,
        truth=bank["theta"][selected],
        names=names,
    )
    sim_cal = diagnostic_plots(
        out, simulated, bank["theta"][selected], names, "simulation"
    )
    sim_truth = pd.DataFrame(bank["theta"][selected], columns=names)
    sim_truth.insert(0, "row_index", np.arange(len(sim_truth)))
    for k, row in enumerate(_representative_rows(sim_truth)):
        _corner(
            out / f"simulation_individual_{k:02d}_physical_corner.png",
            {"Posterior": simulated[row][:, physical]},
            bank["theta"][selected[row], physical],
            PHYSICAL,
        )
        _marginals(
            out / f"simulation_individual_{k:02d}_15d.png",
            {"Posterior": simulated[row]},
            bank["theta"][selected[row]],
            names,
        )
    arrays = rt.validation_arrays
    features = np.asarray(
        make_encoder_features(
            arrays.flux, arrays.flux_err, rt.feature_stats, arrays.mask
        )
    )
    count = min(cfg["report_objects"], len(features))
    observed = posterior_draws(features[:count])
    np.savez(
        out / "observed_posterior_15d.npz",
        draws=observed,
        row_index=arrays.row_index[:count],
        names=names,
    )
    # Observed truths are loaded only here, after all training and population fitting.
    source_manifest = read(Path(m["source"]) / "MANIFEST.json")
    catalog = Path(source_manifest["validation_catalog"])
    import pyarrow.parquet as pq

    truth_available = set(names).issubset(pq.read_schema(catalog).names)
    if truth_available:
        truth = (
            pd.read_parquet(catalog, columns=list(names))
            .iloc[arrays.row_index[:count]]
            .to_numpy()
        )
        diagnostic_plots(out, observed, truth, names, "observed_blind")
        frame = pd.DataFrame(truth, columns=names)
        frame.insert(0, "row_index", np.arange(len(frame)))
        for k, row in enumerate(_representative_rows(frame)):
            _corner(
                out / f"individual_{k:02d}_physical_corner.png",
                {"Posterior": observed[row][:, physical]},
                truth[row, physical],
                PHYSICAL,
            )
            _marginals(
                out / f"individual_{k:02d}_15d.png",
                {"Posterior": observed[row]},
                truth[row],
                names,
            )
    rng = np.random.default_rng(cfg["seed"] + 40000)
    size = cfg["population_report_draws"]
    parent_x, _ = basis.sample(rng, size, weights=parent["u"])
    parent_draws = np.asarray(x_to_theta(jnp.asarray(parent_x), rt.latent_spec))
    # Use the fresh, independent evaluation bank, never classifier training
    # draws, to display the selected predictive population.
    pool = np.flatnonzero(
        bank["selected"]
        & (np.arange(len(bank["selected"])) >= len(bank["selected"]) // 2)
    )
    selected_draws = bank["theta"][rng.choice(pool, size)]
    pd.DataFrame(parent_draws, columns=names).to_parquet(
        out / "learned_parent_15d.parquet", index=False
    )
    pd.DataFrame(selected_draws, columns=names).to_parquet(
        out / "learned_selected_15d.parquet", index=False
    )
    true_pop = {}
    metric_rows = []
    for population, predicted in (
        ("parent", parent_draws),
        ("selected", selected_draws),
    ):
        path = m[f"blind_truth_{population}"]
        if not path:
            continue
        tf = pd.read_parquet(path)
        tw = (
            np.asarray(tf["population_weight"])
            if "population_weight" in tf
            else np.ones(len(tf))
        )
        if not np.isfinite(tw).all() or np.any(tw < 0) or tw.sum() <= 0:
            raise ValueError("invalid blind truth population weights")
        true_pop[population] = tf[list(names)].to_numpy()[
            rng.choice(len(tf), size, p=tw / tw.sum())
        ]
        metrics, joint = population_metrics(predicted, true_pop[population], names)
        metrics["population"] = population
        metric_rows.append(metrics)
        metrics.to_csv(out / f"{population}_marginal_closure.csv", index=False)
        joint.to_csv(out / f"{population}_joint_closure.csv", index=False)
        _corner(
            out / f"{population}_vs_truth_corner.png",
            {f"Learned {population}": predicted[:, physical]},
            true_pop[population][:, physical],
            PHYSICAL,
        )
    series = {
        "Learned parent": parent_draws[:, physical],
        "Learned selected": selected_draws[:, physical],
    }
    series.update({f"True {name}": x[:, physical] for name, x in true_pop.items()})
    _corner(out / "population_physical_corner.png", series, None, PHYSICAL)
    _marginals(out / "population_physical.png", series, None, PHYSICAL)
    full_series = {"Learned parent": parent_draws, "Learned selected": selected_draws}
    full_series.update({f"True {name}": x for name, x in true_pop.items()})
    _marginals(out / "population_15d.png", full_series, None, names)
    # Equal weight per observed object; retain joint posterior draws, not medians.
    aggregate = observed.reshape(-1, len(names))
    aggregate = aggregate[
        rng.choice(len(aggregate), min(size, len(aggregate)), replace=False)
    ]
    selected_series = {
        "Selected prior": selected_draws[:, physical],
        "Posterior aggregate": aggregate[:, physical],
    }
    if "selected" in true_pop:
        selected_series["True selected"] = true_pop["selected"][:, physical]
        ag, ag_joint = population_metrics(aggregate, true_pop["selected"], names)
        ag.to_csv(out / "aggregate_selected_closure.csv", index=False)
        ag_joint.to_csv(out / "aggregate_selected_joint.csv", index=False)
    _marginals(out / "selected_aggregate_physical.png", selected_series, None, PHYSICAL)
    if metric_rows:
        pd.concat(metric_rows).to_csv(out / "population_closure.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(parent["v"], label="Selected v")
    axes[0].plot(parent["u"], label="Parent u")
    axes[0].legend()
    axes[0].set_xlabel("Joint component")
    axes[1].plot(parent["alpha"], color="#ae4a56")
    axes[1].set_ylabel("Selection efficiency alpha_j")
    fig.tight_layout()
    fig.savefig(out / "selection_weights.png", dpi=150)
    plt.close(fig)
    # Observable-space posterior-population predictive, compared with held-out observations.
    from scipy.stats import ks_2samp

    predictive = bank["features"][selected]
    feature_rows = []
    for j in range(features.shape[1]):
        s = max(np.subtract(*np.quantile(features[:, j], [0.75, 0.25])), 1e-8)
        feature_rows.append(
            dict(
                feature=j,
                w1_over_observed_iqr=wasserstein_distance(
                    predictive[:, j], features[:, j]
                )
                / s,
                ks=ks_2samp(predictive[:, j], features[:, j]).statistic,
            )
        )
    pd.DataFrame(feature_rows).to_csv(out / "observable_predictive.csv", index=False)
    fig, axes = plt.subplots(3, 6, figsize=(18, 8))
    for j, ax in enumerate(axes.flat):
        if j >= len(rt.feature_stats.band_names):
            break
        lo, hi = np.quantile(np.r_[predictive[:, j], features[:, j]], [0.005, 0.995])
        bins = np.linspace(lo, hi, 40)
        ax.hist(
            features[:, j], bins=bins, density=True, histtype="step", label="Observed"
        )
        ax.hist(
            predictive[:, j],
            bins=bins,
            density=True,
            histtype="step",
            label="Parent -> selected",
        )
        ax.set_title(rt.feature_stats.band_names[j], fontsize=9)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "observable_predictive.png", dpi=140)
    plt.close(fig)
    baseline_note = "No baseline path supplied."
    if m["baseline"]:
        baseline = Path(m["baseline"])
        paths = sorted(
            baseline.glob("trajectories/*/inference/report/support_by_cycle.csv")
        )
        frames = []
        for path in paths:
            f = pd.read_csv(path)
            f["track"] = path.parents[2].name
            frames.append(f)
        if frames:
            pd.concat(frames).to_csv(
                out / "old_sbeb_support_calibration.csv", index=False
            )
            old = pd.concat(frames)
            fig, ax = plt.subplots(figsize=(9, 4))
            for track in ("warm_iw_r29", "warm_raw_r29", "warm_iw_r27"):
                f = old[old.track == track]
                ax.plot(f.cycle, f.mean_physical_coverage_68, "o-", label=track)
            new = (
                pd.read_csv(out / "observed_blind_calibration.csv")
                if truth_available
                else sim_cal
            )
            ax.axhline(
                new[new.group == "physical"].coverage_68.mean(),
                color="#277c8e",
                label="New direct posterior",
            )
            ax.axhline(0.68, color="black", ls="--", label="Nominal")
            ax.legend(fontsize=8)
            ax.set_ylabel("Mean physical 68% coverage")
            ax.set_xlabel("Old SBEB cycle")
            fig.tight_layout()
            fig.savefig(out / "baseline_coverage_comparison.png", dpi=150)
            plt.close(fig)
            baseline_note = "Baseline table copied; cohorts/targets may differ. Comparison is descriptive, not paired."
        closure_frames = []
        for path in sorted(
            baseline.glob("trajectories/*/inference/report/fixed_point_summary.csv")
        ):
            frame = pd.read_csv(path)
            # Never average physical and SFH groups under a physical label.
            frame = frame[frame.group == "physical"].copy()
            frame["track"] = path.parents[2].name
            closure_frames.append(frame)
        if closure_frames:
            old_closure = pd.concat(closure_frames)
            old_closure.to_csv(out / "old_sbeb_physical_population.csv", index=False)
            fig, ax = plt.subplots(figsize=(9, 4))
            for track in ("warm_iw_r29", "warm_raw_r29", "warm_iw_r27"):
                f = old_closure[
                    (old_closure.track == track)
                    & (old_closure.comparison == "prior_parent_vs_truth_parent")
                ]
                ax.plot(f.cycle, f.median_wasserstein_over_iqr, "o-", label=track)
            if (out / "parent_marginal_closure.csv").exists():
                new = pd.read_csv(out / "parent_marginal_closure.csv")
                ax.axhline(
                    new[new.group == "physical"].w1_over_truth_iqr.median(),
                    color="#277c8e",
                    label="New parent",
                )
            ax.set_ylabel("Median physical W1 / truth IQR")
            ax.set_xlabel("Old SBEB cycle")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out / "baseline_parent_comparison.png", dpi=150)
            plt.close(fig)
    text = f"""# Decoupled forward population report

Population fit uses no q samples; the frozen parent is a normalized 15D mixture.
Selected v and parent u are in ../population/component_weights.csv.
Parent alpha: {parent['alpha_parent']:.6f}; KKT gap: {parent['kkt_gap']:.3g}.

## Read first

- population_physical.png / population_physical_corner.png: parent and selected closure.
- population_15d.png: nuisance dimensions retained, including their misspecification.
- selected_aggregate_physical.png: selected prior versus aggregate of full posteriors.
- selection_weights.png: explicit v, u and selection efficiencies.
- simulation_*: independent forward-test calibration under the fitted parent.
- observed_blind_*: blind catalogue coverage, including possible model misspecification.
- individual_*: full-distribution 5D corners and 15D marginals.
- observable_predictive.png: observable-space population predictive check.
- *_joint_closure.csv: joint physical and nuisance sliced Wasserstein metrics.

{baseline_note}

Calibration under the fitted model alone cannot prove recovery of the real parent.
The fixed reference SFH conditional can be misspecified. Low-alpha parent mass
is support-constrained; active constraints invalidate any unconstrained recovery claim.
No NUTS, SMC, RWS or post-hoc uncertainty inflation is used.
"""
    (out / "REPORT.md").write_text(text)
    write(
        out / "FINAL.json",
        dict(
            status="FORWARD_POPULATION_REPORT_COMPLETE",
            scientific_promotion=False,
            posterior_dimensions=15,
            population_uses_q=False,
            truth_used_for_training=False,
            artifacts={
                str(p.relative_to(out)): sha(p)
                for p in out.rglob("*")
                if p.is_file() and p.name != "FINAL.json"
            },
        ),
    )
