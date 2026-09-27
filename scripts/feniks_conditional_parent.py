"""One blind parent experiment: nested SFH flexibility, recycled DSPS bank.

Target latent truth is read only in report(). Preparation uses the frozen
independent reference; population() uses catalogue flux/error/mask columns only.
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

from euclid_dsps.amortized.coherent_coordinates import to_theta
from euclid_dsps.amortized.conditional_reference import (
    lift_parent,
    reserved_row_weights,
    split_sfh_basis,
    tied_selected_ratios,
)
from euclid_dsps.amortized.forward_population import (
    parent_from_selected,
    selection_efficiencies,
)
from euclid_dsps.amortized.native_reference import choose_penalty, sample_basis
from scripts import feniks_coherent_inference as ci
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish


def initialize(source: Path, root: Path, config: Path) -> None:
    if root.exists() or source in root.parents or root in source.parents:
        raise ValueError("Use a new root outside the immutable baseline")
    m, old, digest = ci.settings(source)
    if m.get("population_uses_q") or m.get("population_uses_target_truth"):
        raise ValueError("A blind, q-independent baseline is required")
    for stage in ("reference", "population", "report"):
        if not complete(source / stage, digest):
            raise ValueError(f"Completed source {stage} required")
    source_parent = read(source / "population/parent.json")
    u, v, alpha = [np.asarray(source_parent[k], float) for k in ("u", "v", "alpha")]
    if (
        any(
            a.shape != (old["reference"]["components"],) or not np.isfinite(a).all()
            for a in (u, v, alpha)
        )
        or np.any(u < 0)
        or np.any(v < 0)
        or np.any(alpha <= 0)
        or np.any(alpha > 1)
        or not np.isclose(u.sum(), 1)
        or not np.isclose(v.sum(), 1)
    ):
        raise ValueError("Invalid baseline parent normalization or support")
    np.testing.assert_allclose(v, u * alpha / (u @ alpha), rtol=1e-8, atol=1e-10)
    override = yaml.safe_load(config.read_text())
    if not 0 < override["probability_floor"] < 0.5:
        raise ValueError("Invalid soft partition floor")
    if any(not isinstance(v, int) or v < 1 for v in override["resources"].values()):
        raise ValueError("Positive integer resources required")
    cfg = copy.deepcopy(old)
    for key in ("capacity", "candidate", "qualification_contracts", "posterior"):
        cfg.pop(key, None)
    cfg.update(
        seed=override["seed"],
        probability_floor=override["probability_floor"],
        resources=override["resources"],
        reference=dict(
            components=2 * old["reference"]["components"],
            family="nested_sfh_split_of_frozen_joint_kernels",
        ),
    )
    cfg["classifier"].update(override["classifier"])
    if not cfg["classifier"]["fixed_validation"] or cfg["classifier"]["epochs"] < 1:
        raise ValueError("Bounded classifier with fixed validation required")
    files = dict(m["source_files"])
    paths = [source / n for n in ("MANIFEST.json", "experiment.yaml")]
    for stage in ("reference", "population", "report"):
        paths.append(source / stage / "FINAL.json")
    paths += [
        source / "reference" / n
        for n in ("basis.npz", "coordinates.json", "feature_stats.json")
    ]
    paths += [source / "population/parent.json", source / "report/DECISION.json"]
    for task in range(old["bank"]["shards"]):
        shard = source / "banks" / f"shard_{task:03d}"
        if not complete(shard, digest):
            raise ValueError(f"Incomplete source bank {task}")
        paths.append(shard / "FINAL.json")
        paths += sorted(shard.glob("block_*/FINAL.json"))
    files.update({str(p): sha(p) for p in paths})
    root.mkdir(parents=True)
    for name in ("logs", "reference", "banks", "population", "tied", "report"):
        (root / name).mkdir()
    (root / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write(
        root / "MANIFEST.json",
        dict(
            source=m["source"],
            baseline_parent_run=str(source),
            source_contract=digest,
            source_settings=old,
            settings=cfg,
            selection=m["selection"],
            source_files=files,
            frozen_files={"experiment.yaml": sha(root / "experiment.yaml")},
            population_uses_q=False,
            population_uses_target_truth=False,
            reference_design_used_development_truth=m.get(
                "reference_design_used_development_truth", True
            ),
            inherited_failures=read(source / "report/DECISION.json"),
            new_dsps_simulations=0,
            no_posterior_training=True,
            production_promotion=False,
        ),
    )
    print(
        yaml.safe_dump(
            dict(
                experiment="one nested SFH refinement plus tied control using the same new classifier",
                components=cfg["reference"]["components"],
                construction="two soft reference-SFH-PC1 gates inside each original component",
                nuisance="all ten SFH coordinates and original joint 15D kernels retained",
                selection="rejected and selected rows retained; new binomial alpha per subcomponent",
                reused_simulations=old["bank"]["shards"]
                * old["bank"]["rows_per_shard"],
                new_dsps_simulations=0,
                classifier=cfg["classifier"],
                posterior="not trained here",
                fit_inputs="catalogue flux/error/mask only; no catalogue latent truth or q",
                resources=cfg["resources"],
                outputs="split basis, replay receipts, classifier, expanded/tied v/u/alpha, plots, roadmap",
            ),
            sort_keys=False,
        )
    )


def prepare(root: Path) -> None:
    m, cfg, digest = ci.settings(root)
    out = root / "reference"
    if complete(out, digest):
        return
    source = Path(m["baseline_parent_run"])
    old, _, _ = ci.require_reference(source, m["source_contract"])
    new, gates, info = split_sfh_basis(old, cfg["probability_floor"])
    np.savez(out / "basis.npz", **new)
    np.savez(out / "gates.npz", probabilities=gates)
    write(out / "split.json", info)
    for name in ("coordinates.json", "feature_stats.json"):
        shutil.copy2(source / "reference" / name, out / name)
    finish(
        out,
        [
            out / n
            for n in (
                "basis.npz",
                "gates.npz",
                "split.json",
                "coordinates.json",
                "feature_stats.json",
            )
        ],
        digest,
        dimensions=15,
        components=new["conditional"].shape[1],
        target_truth_used=False,
        exact_old_family_embedded=True,
        strict_capacity_qualification="not_repeated_not_promoted",
    )


def relabel(root: Path, task: int) -> None:
    """Recover generating anchor with the ORIGINAL RNG; require exact replay."""
    m, cfg, digest = ci.settings(root)
    if not complete(root / "reference", digest):
        raise ValueError("Split reference incomplete")
    source, old_cfg = Path(m["baseline_parent_run"]), m["source_settings"]
    if not 0 <= task < cfg["bank"]["shards"]:
        raise ValueError("Invalid shard")
    old, _, _ = ci.require_reference(source, m["source_contract"])
    with np.load(root / "reference/gates.npz") as f:
        gates = f["probabilities"]
    out = root / "banks" / f"shard_{task:03d}"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    files = []
    for start in range(
        0, cfg["bank"]["rows_per_shard"], cfg["bank"]["checkpoint_rows"]
    ):
        original = source / "banks" / out.name / f"block_{start:07d}"
        block = out / original.name
        block.mkdir(exist_ok=True)
        if not complete(block, digest):
            if not complete(original, m["source_contract"]):
                raise ValueError(f"Incomplete/corrupt original block: {original}")
            with np.load(original / "bank.npz") as f:
                data = dict(f)
            labels = data["component"]
            expected = np.arange(len(labels)) % old_cfg["reference"]["components"]
            if not np.array_equal(labels, expected):
                raise ValueError(
                    "Source bank is not the declared balanced component sampler"
                )
            seed = old_cfg["seed"] + task * 10000000 + start + 1000
            replay, anchor, _ = sample_basis(old, labels, seed, return_aux=True)
            if not np.array_equal(replay, data["x"]):
                raise ValueError(
                    "Anchor replay mismatch: refuse approximate relabeling or DSPS rerun"
                )
            rng = np.random.default_rng(
                cfg["seed"] + 19000000 + task * 10000000 + start
            )
            high = rng.random(len(labels)) < gates[anchor, labels, 1]
            data["component"] = 2 * labels + high.astype(int)
            data["source_component"] = labels
            data["anchor_index"] = anchor
            np.savez(block / "bank.npz", **data)
            finish(
                block,
                [block / "bank.npz"],
                digest,
                rows=len(labels),
                exact_anchor_replay=True,
                new_dsps_simulations=0,
                original_sha256=sha(original / "bank.npz"),
            )
        files.append(block / "FINAL.json")
        write(
            out / "PROGRESS.json",
            dict(
                done=min(
                    start + cfg["bank"]["checkpoint_rows"],
                    cfg["bank"]["rows_per_shard"],
                ),
                total=cfg["bank"]["rows_per_shard"],
            ),
        )
    finish(out, files, digest, exact_anchor_replay=True, new_dsps_simulations=0)


def population(root: Path) -> None:
    import equinox as eqx

    from euclid_dsps.amortized.population_regularization import fit_selected_weights_kl
    from scripts.feniks_forward_population import classifier_template, classify
    from scripts.feniks_ratio_followup import apply_logit_offsets

    _, cfg, digest = ci.settings(root)
    out = root / "tied"
    if complete(root / "population", digest) and complete(out, digest):
        return
    if not complete(root / "population", digest):
        # Check the nested control's support before spending time on training.
        labels, selected, roles = [], [], []
        for task in range(cfg["bank"]["shards"]):
            shard = root / "banks" / f"shard_{task:03d}"
            if not complete(shard, digest):
                raise ValueError(f"Incomplete relabel shard: {task}")
            for block in sorted(shard.glob("block_*")):
                if not complete(block, digest):
                    raise ValueError(f"Incomplete relabel block: {block}")
                with np.load(block / "bank.npz") as f:
                    labels.append(f["component"])
                    selected.append(f["selected"])
                    roles.append(f["role"])
        labels, selected, roles = map(np.concatenate, (labels, selected, roles))
        efficiency = selection_efficiencies(
            labels,
            selected,
            cfg["reference"]["components"],
            min_selected=cfg["population"]["min_selected"],
            min_alpha=cfg["population"]["min_alpha"],
        )
        counts = np.array(
            [
                np.bincount(
                    labels[selected & (roles == r)],
                    minlength=cfg["reference"]["components"],
                )
                for r in range(5)
            ]
        )
        valid = bool(efficiency["eligible"].all() and (counts > 0).all())
        write(
            root / "population/split_support.json",
            dict(
                passed=valid,
                counts=counts.tolist(),
                eligible=efficiency["eligible"].tolist(),
                checked_before_training=True,
            ),
        )
        del labels, selected, roles
        if not valid:
            write(
                root / "population/BLOCKED.json",
                dict(reason="split_support_before_training"),
            )
            raise ValueError(
                "Split support insufficient for the declared nested comparison"
            )
    # No balanced-parent assumption here: selected frequencies and alpha use
    # actual class counts. ci.population reads target observation columns ONLY.
    ci.population(root)
    _, _, stats = ci.require_reference(root, digest)
    parent = read(root / "population/parent.json")
    z = np.array(read(root / "reference/split.json")["z"])
    net_cfg = dict(cfg["classifier"], seed=cfg["seed"] + 1)
    logs = []
    model = None
    for split in ("train", "validation"):
        frame, bands = ci.observed(root, split)
        f = ci.features(frame, bands, stats)
        if model is None:
            model = eqx.tree_deserialise_leaves(
                root / "population/best.eqx",
                classifier_template(
                    f.shape[1], cfg["reference"]["components"], net_cfg
                ),
            )
        logs.append(
            apply_logit_offsets(classify(model, f), np.array(parent["logit_offsets"]))
        )
    transformed = [
        tied_selected_ratios(
            logc, parent["classifier_reference_frequencies"], parent["alpha"], z
        )
        for logc in logs
    ]
    _, c, alpha = transformed[0]
    eligible = (
        pd.read_csv(root / "population/weights.csv")
        .eligible.to_numpy(bool)
        .reshape(-1, 2)
        .all(axis=1)
    )
    if not eligible.all():
        raise ValueError(
            "Weak subcomponent support: cannot claim the same feasible tied control"
        )
    candidates, heldout, rows = [], [], []
    for strength in cfg["population"]["penalties"]:
        write(out / "PROGRESS.json", dict(stage="tied_solver", strength=strength))
        v, diagnostics = fit_selected_weights_kl(
            transformed[0][0],
            c,
            strength=strength,
            alpha=alpha,
            eligible=eligible,
            weak_parent_mass=cfg["population"]["weak_parent_mass"],
        )
        candidates.append(v)
        heldout.append(logsumexp(transformed[1][0] - np.log(c) + np.log(v), axis=1))
        rows.append(diagnostics)
    index, accepted, degradation, se = choose_penalty(
        heldout, cfg["population"]["penalties"]
    )
    for i, row in enumerate(rows):
        row.update(
            selected=i == index,
            heldout_one_se=bool(accepted[i]),
            heldout_degradation=float(degradation[i]),
            paired_se=float(se[i]),
        )
    pd.DataFrame(rows).to_csv(out / "regularization.csv", index=False)
    v = candidates[index]
    u = parent_from_selected(v, alpha)
    expanded_score = logsumexp(
        logs[1]
        - np.log(parent["classifier_reference_frequencies"])
        + np.log(parent["v"]),
        axis=1,
    )
    delta = expanded_score - heldout[index]
    write(
        out / "parent.json",
        dict(
            u=u.tolist(),
            v=v.tolist(),
            alpha=alpha.tolist(),
            expanded_u=lift_parent(u, z).tolist(),
            selected_penalty=cfg["population"]["penalties"][index],
            alpha_parent=float(u @ alpha),
            population_uses_target_truth=False,
            population_uses_q=False,
            same_classifier_control=True,
        ),
    )
    write(
        out / "comparison.json",
        dict(
            validation_loglik_gain_expanded_minus_tied=float(delta.mean()),
            paired_standard_error=float(delta.std(ddof=1) / np.sqrt(len(delta))),
            validation_rows=len(delta),
            same_reference_density=True,
            heldout_used_for_penalty_selection=True,
            independent_final_test=False,
            target_truth_used=False,
        ),
    )
    finish(
        out,
        [out / n for n in ("parent.json", "regularization.csv", "comparison.json")],
        digest,
        same_classifier_control=True,
        target_truth_used=False,
    )


def report(root: Path) -> None:
    from scipy.stats import wasserstein_distance

    from euclid_dsps.amortized.reference_capacity import observable_tail_metrics
    from scripts.feniks_reference_to_parent import (
        physical_sw,
        population_plot,
        weighted_ks,
    )
    from scripts.report_feniks_forward_population import population_metrics

    m, cfg, digest = ci.settings(root)
    out = root / "report"
    if complete(out, digest):
        return
    missing = [
        stage
        for stage in ("reference", "population", "tied")
        if not complete(root / stage, digest)
    ]
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        return
    basis, spec, _ = ci.require_reference(root, digest)
    data = ci.load_bank(root, cfg, digest)
    parent = read(root / "population/parent.json")
    tied = read(root / "tied/parent.json")
    z = np.array(read(root / "reference/split.json")["z"])
    old_parent = read(Path(m["baseline_parent_run"]) / "population/parent.json")
    models = dict(
        baseline=lift_parent(old_parent["u"], z),
        tied=np.array(tied["expanded_u"]),
        expanded=np.array(parent["u"]),
    )
    source = Path(m["source"])
    truth = pd.read_parquet(source / "dataset/parent/validation.parquet")
    selected_truth = pd.read_parquet(source / "dataset/selected_r29/validation.parquet")
    bands = [b["name"] for b in read(source / "decoder.json")["bands"]]
    n = cfg["evaluation"]["population_draws"]
    marginal, joint, predictive, measures, observable_joint = [], [], [], [], []
    parent_draws, selected_draws = {}, {}
    for name, weights in models.items():
        rng = np.random.default_rng(cfg["seed"] + 800)
        labels = rng.choice(len(weights), n, p=weights)
        draws = np.asarray(
            to_theta(sample_basis(basis, labels, cfg["seed"] + 801), spec)
        )
        mask, w = reserved_row_weights(
            data["component"], data["selected"], data["role"], weights
        )
        selected = data["theta"][rng.choice(np.flatnonzero(mask), n, p=w)]
        parent_draws[name], selected_draws[name] = draws, selected
        measures.append(
            dict(
                model=name,
                selected_effective_rows=float(1 / (w @ w)),
                alpha=float(weights @ np.array(parent["alpha"])),
            )
        )
        for population_name, values, actual in (
            ("parent", draws, truth),
            ("selected", selected, selected_truth),
        ):
            one, multi = population_metrics(
                values, actual[ci.NAMES].to_numpy(), ci.NAMES, cfg["seed"] + 802
            )
            marginal.append(one.assign(model=name, population=population_name))
            joint.append(multi.assign(model=name, population=population_name))
            actual_theta = actual[ci.NAMES].to_numpy()
            scale = np.maximum(
                np.subtract(*np.quantile(actual_theta, [0.75, 0.25], axis=0)), 1e-8
            )
            directions = np.random.default_rng(cfg["seed"] + 803).normal(size=(64, 15))
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            distance = np.mean(
                [
                    wasserstein_distance(values / scale @ d, actual_theta / scale @ d)
                    for d in directions
                ]
            )
            joint.append(
                pd.DataFrame(
                    [
                        dict(
                            model=name,
                            population=population_name,
                            group="joint_15d",
                            sliced_wasserstein=float(distance),
                            directions=64,
                        )
                    ]
                )
            )
        target_flux = selected_truth[[f"flux_{b}" for b in bands]].to_numpy()
        scale = np.maximum(
            np.subtract(*np.quantile(target_flux, [0.75, 0.25], axis=0)), 1e-30
        )
        p, t = np.arcsinh(data["flux"][mask] / scale), np.arcsinh(target_flux / scale)
        width = np.maximum(np.subtract(*np.quantile(t, [0.75, 0.25], axis=0)), 1e-8)
        directions = np.random.default_rng(cfg["seed"] + 804).normal(
            size=(32, len(bands))
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        distance = np.mean(
            [
                wasserstein_distance(p / width @ d, t / width @ d, u_weights=w)
                for d in directions
            ]
        )
        observable_joint.append(
            dict(
                model=name,
                sliced_wasserstein=float(distance),
                directions=32,
                space="asinh_flux_standardized_by_target_iqr",
            )
        )
        for j, band in enumerate(bands):
            values, actual = (
                data["flux"][mask, j],
                selected_truth[f"flux_{band}"].to_numpy(),
            )
            predictive.append(
                dict(
                    model=name,
                    band=band,
                    ks=weighted_ks(values, actual, w),
                    **observable_tail_metrics(values, actual, w),
                )
            )
    one, multi, pred = pd.concat(marginal), pd.concat(joint), pd.DataFrame(predictive)
    one.to_csv(out / "marginals.csv", index=False)
    multi.to_csv(out / "joint.csv", index=False)
    pred.to_csv(out / "observable_predictive.csv", index=False)
    pd.DataFrame(observable_joint).to_csv(out / "observable_joint.csv", index=False)
    pd.DataFrame(measures).to_csv(out / "measures.csv", index=False)
    population_plot(
        out / "parent_physical.png", truth[ci.NAMES].to_numpy(), parent_draws
    )
    population_plot(
        out / "selected_physical.png",
        selected_truth[ci.NAMES].to_numpy(),
        selected_draws,
    )
    plot_diagnostics(root, one, pred, parent, z)
    plot_correlations(out, truth[ci.NAMES].to_numpy(), parent_draws)
    limits = cfg["parent_contracts"]

    def metric(pop):
        return physical_sw(
            multi.loc[multi.model.eq("expanded") & multi.population.eq(pop)]
        )

    physical = one.loc[one.model.eq("expanded") & one.group.eq("physical")]
    core = physical.loc[physical.parameter.isin(ci.NAMES[:3])]
    audit = read(root / "population/classifier_audit.json")
    stopping = read(root / "population/STOP.json")
    alpha_truth = len(selected_truth) / len(truth)
    pred_new = pred.loc[pred.model.eq("expanded")]
    gates = dict(
        parent_physical_sw=metric("parent") <= limits["maximum_physical_sw"],
        selected_physical_sw=metric("selected") <= limits["maximum_physical_sw"],
        physical_marginals=bool(
            physical.w1_over_truth_iqr.max() <= limits["maximum_physical_marginal_w1"]
        ),
        core_marginals=bool(
            core.w1_over_truth_iqr.max() <= limits["maximum_physical_marginal_w1"]
        ),
        parent_alpha=abs(parent["alpha_parent"] - alpha_truth)
        <= limits["maximum_alpha_error"],
        observable_cdfs=bool(pred_new.ks.max() <= limits["maximum_observable_ks"]),
        observable_tails=bool(
            pred_new.predicted_mass_above_truth_q999.max()
            <= limits["maximum_tail_probability"]
        ),
        classifier_ratio_moment=audit["independent_ratio_moment_median_error"]
        <= limits["maximum_ratio_moment_error"],
        classifier_gain=audit["audit_null_nll"] - audit["audit_nll"]
        >= limits["minimum_classifier_gain"],
        classifier_validation_plateau=bool(stopping["plateau"]),
    )
    comparison = read(root / "tied/comparison.json")
    decision = dict(
        gates=gates,
        limits=limits,
        parent_physical_sw=metric("parent"),
        selected_physical_sw=metric("selected"),
        comparison=comparison,
        parent_scope="unselected",
        population_uses_target_truth=False,
        population_uses_q=False,
        new_dsps_simulations=0,
        inherited_failures=m["inherited_failures"],
        sfh_conditional_validated=False,
        ready_for_production=False,
        validation_used_for_development=True,
        classifier_stop=stopping,
        next="compare_expanded_vs_tied_parent_and_parallel_posterior_calibration",
    )
    write(out / "DECISION.json", decision)
    text = "# Conditional parent experiment\n\n" + "\n".join(
        f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in gates.items()
    )
    text += (
        "\n\nNo target latent truth in fitting. Report truth is diagnostic only. "
        "Same classifier for expanded/tied comparison; old family is exactly nested. "
        "No kernels/support changed; only one reference SFH axis can be reweighted. "
        "This cannot repair arbitrary missing joint support or certify an unrestricted parent. "
        "Inherited reference failures remain. No production promotion.\n"
    )
    (out / "REPORT.md").write_text(text)
    (root / "ROADMAP_STATUS.md").write_text(text)
    (out / "BLOCKED.json").unlink(missing_ok=True)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        digest,
        **decision,
    )


def plot_diagnostics(root, marginals, predictive, parent, z):
    import matplotlib.pyplot as plt

    out = root / "report"
    history = pd.read_json(root / "population/training.jsonl", lines=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for column in ("train_nll", "validation_nll", "best_nll"):
        axes[0].plot(history.epoch, history[column], label=column)
    axes[0].set(xlabel="Classifier epoch", ylabel="NLL")
    axes[0].legend()
    u = np.array(parent["u"]).reshape(-1, 2)
    high = u[:, 1] / u.sum(axis=1)
    axes[1].scatter(z[:, 1], high, c=u.sum(axis=1), cmap="viridis", s=20)
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set(
        xlabel="Reference positive SFH-PC1 branch fraction",
        ylabel="Blind learned fraction",
    )
    fig.tight_layout()
    fig.savefig(out / "classifier_and_sfh_weights.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    for name in ("baseline", "tied", "expanded"):
        rows = marginals.loc[
            marginals.model.eq(name) & marginals.population.eq("parent")
        ]
        axes[0].plot(
            np.arange(len(rows)), rows.w1_over_truth_iqr, marker=".", label=name
        )
        rows = predictive.loc[predictive.model.eq(name)]
        axes[1].plot(np.arange(len(rows)), rows.ks, marker=".", label=name)
    axes[0].set_xticks(range(15), ci.NAMES, rotation=70, fontsize=7)
    axes[1].set_xticks(range(len(rows)), rows.band, rotation=70, fontsize=7)
    axes[0].set_ylabel("Parent W1 / truth IQR (all 15D)")
    axes[1].set_ylabel("Selected flux CDF KS")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


def plot_correlations(out, truth, draws):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    rows = []
    for ax, (name, values) in zip(
        axes, {"target": truth, **draws}.items(), strict=True
    ):
        correlation = pd.DataFrame(values).corr(method="spearman").to_numpy()[:5, 5:]
        picture = ax.imshow(
            correlation, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto"
        )
        ax.set_title(name)
        ax.set_yticks(range(5), ci.NAMES[:5], fontsize=6)
        ax.set_xticks(range(10), range(1, 11), fontsize=7)
        ax.set_xlabel("SFH contrast")
        for j in range(5):
            for k in range(10):
                rows.append(
                    dict(
                        model=name,
                        physical=ci.NAMES[j],
                        sfh=ci.NAMES[5 + k],
                        spearman=correlation[j, k],
                    )
                )
    fig.colorbar(picture, ax=axes, label="Spearman correlation")
    fig.savefig(out / "physical_sfh_correlations.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "physical_sfh_correlations.csv", index=False)


def schedule(root: Path) -> None:
    _, cfg, digest = ci.settings(root)
    for key, value in cfg["resources"].items():
        print(f"{key.upper()}={value}")
    print(f"NEED_PREPARE={int(not complete(root / 'reference', digest))}")
    print(
        f"NEED_POPULATION={int(not all(complete(root / s, digest) for s in ('population', 'tied')))}"
    )
    print(f"NEED_REPORT={int(not complete(root / 'report', digest))}")
    print(
        "MISSING_BANKS="
        + ",".join(
            str(i)
            for i in range(cfg["bank"]["shards"])
            if not complete(root / "banks" / f"shard_{i:03d}", digest)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["init", "prepare", "relabel", "population", "report", "schedule"],
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "init":
        initialize(args.source.resolve(), root, args.config.resolve())
    elif args.mode == "relabel":
        relabel(root, args.task)
    else:
        globals()[args.mode](root)


if __name__ == "__main__":
    main()
