"""Frozen-classifier in-family recovery on disjoint saved reference bank roles.

No simulator, training, q draws, new target catalogue or production prior writes.
The known mixtures are diagnostics only, never replacement fitted populations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.forward_population import parent_from_selected, simplex
from euclid_dsps.amortized.population_regularization import fit_selected_weights_kl
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish
from scripts.feniks_coherent_parent import settings as base_settings

CASES = ("learned_parent", "capacity_control")


def settings(root):
    m, cfg, digest = base_settings(root)
    for path, expected in m["source_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Immutable diagnostic source changed: {path}")
    return m, cfg, digest


def initialize(source, root, config):
    from scripts import feniks_coherent_inference as ci

    if root.exists() or source in root.parents or root in source.parents:
        raise ValueError("Use a new diagnostic root outside the source run")
    m, original, contract = ci.settings(source)
    for stage in ("population", "qualification", "report"):
        if not complete(source / stage, contract):
            raise ValueError(f"Completed source {stage} required")
    cfg = yaml.safe_load(config.read_text())
    if tuple(cfg["cases"]) != CASES or (cfg["fit_role"], cfg["evaluation_role"]) != (
        4,
        3,
    ):
        raise ValueError(
            "Use the two declared mixtures and disjoint reserved roles 4/3"
        )
    for name in (
        "evaluation_directions",
        "ratio_blocks",
        "ratio_resamples",
        "classifier_batch_size",
        "minimum_effective_rows",
    ):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f"Positive integer required: {name}")
    if cfg["ratio_blocks"] < 2 or cfg["ratio_resamples"] < 2:
        raise ValueError("Multiple blocks/resamples required")
    if any(not isinstance(v, int) or v < 1 for v in cfg["resources"].values()):
        raise ValueError("Positive integer resources required")
    parent = read(source / "population/parent.json")
    known = dict(
        learned_parent=parent["u"],
        capacity_control=read(source / "qualification/diagnostic_weights.json")["u"],
    )
    k = original["reference"]["components"]
    efficiencies = pd.read_csv(source / "population/weights.csv")
    if not np.array_equal(efficiencies.component.to_numpy(), np.arange(k)):
        raise ValueError("Source efficiency rows must match component order")
    eligible = efficiencies.eligible.to_numpy(dtype=bool)
    for values in known.values():
        u = np.asarray(values, float)
        if (
            len(u) != k
            or not np.isfinite(u).all()
            or np.any(u < 0)
            or not np.isclose(u.sum(), 1)
        ):
            raise ValueError("Invalid known parent mixture")
        if u[~eligible].sum() > original["population"]["weak_parent_mass"] + 1e-10:
            raise ValueError(
                "Known mixture violates the frozen inversion support constraint"
            )
    dataset = Path(m["source"])
    names = [source / "MANIFEST.json", source / "experiment.yaml"]
    for stage in ("population", "qualification", "report"):
        names.append(source / stage / "FINAL.json")
    names += [
        source / "population" / p
        for p in ("best.eqx", "parent.json", "weights.csv", "STOP.json")
    ]
    names += [
        source / "qualification/diagnostic_weights.json",
        dataset / "decoder.json",
        dataset / "dataset/selected_r29/validation.parquet",
    ]
    for task in range(original["bank"]["shards"]):
        shard = source / "banks" / f"shard_{task:03d}"
        if not complete(shard, contract):
            raise ValueError(f"Missing source bank shard {task}")
        names.append(shard / "FINAL.json")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "experiment.yaml").write_text(config.read_text())
    write(
        root / "MANIFEST.json",
        dict(
            source=str(source),
            dataset=str(dataset),
            source_contract=contract,
            settings=cfg,
            source_settings=original,
            known_weights=known,
            selected_penalty=parent["selected_penalty"],
            source_files={str(p): sha(p) for p in names},
            frozen_files={"experiment.yaml": sha(root / "experiment.yaml")},
            no_training=True,
            no_dsps=True,
            population_uses_q=False,
            diagnostic_only=True,
            production_prior_modified=False,
        ),
    )
    print(
        yaml.safe_dump(
            dict(
                cases=list(CASES),
                classifier="frozen best checkpoint + frozen offsets",
                fit="stratified importance-weighted selected bank role 4",
                evaluation="disjoint role 3, full 15D weighted distributions",
                penalty="fixed at the source-selected value",
                new_simulations=0,
                new_training=0,
                gpus=0,
                resources=cfg["resources"],
            )
        )
    )


def row_weights(labels, parent, parent_counts):
    """Empirical g_j is normalized BEFORE selection: w_i = u_j / n_parent,j.

    Applying the selected mask and then normalizing yields v_j proportional to
    u_j alpha_j. Dividing by selected counts here would wrongly remove selection.
    """
    labels = np.asarray(labels, int)
    u, counts = np.asarray(parent, float), np.asarray(parent_counts, float)
    if u.shape != counts.shape or np.any(counts <= 0) or np.any(u < 0):
        raise ValueError("Positive per-component parent counts and valid u required")
    return simplex((u / counts)[labels])


def ratio_uncertainty(logc, c, labels, *, blocks, resamples, seed):
    """Cheap row-block bootstrap of moments only, not any population refit.

    Intervals condition on the frozen calibration/checkpoint. They do not
    include model-training or calibration-set uncertainty.
    """
    p = np.exp(logc)
    c = np.asarray(c, float)
    n = len(p)
    if n < 2 or np.any(c <= 0) or not np.isfinite(p).all():
        raise ValueError("Finite independent classifier probabilities required")
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(n), min(blocks, n))
    sums = np.array([p[ids].sum(axis=0) for ids in groups])
    counts = np.array([len(ids) for ids in groups])
    mean = p.mean(axis=0)
    draws, noise = [], []
    for _ in range(resamples):
        take = rng.integers(len(groups), size=len(groups))
        estimate = sums[take].sum(axis=0) / counts[take].sum()
        draws.append(np.median(abs(estimate / c - 1)))
        noise.append(np.median(abs((estimate - mean) / c)))
    return dict(
        observations=n,
        blocks=len(groups),
        resamples=resamples,
        median_relative_error=float(np.median(abs(mean / c - 1))),
        median_error_interval95=np.quantile(draws, [0.025, 0.975]).tolist(),
        centered_sampling_noise_median_q95=float(np.quantile(noise, 0.95)),
        label_frequency_relative_error=float(
            np.median(abs(np.bincount(labels, minlength=len(c)) / n / c - 1))
        ),
        conditional_on_frozen_calibration=True,
        full_ratio_validation=False,
    )


def cache(root):
    import equinox as eqx

    from scripts.feniks_forward_population import classifier_template, classify
    from scripts.feniks_ratio_followup import apply_logit_offsets

    m, cfg, digest = settings(root)
    out = root / "cache"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    source, original = Path(m["source"]), m["source_settings"]
    bank, k = original["bank"], original["reference"]["components"]
    parent = read(source / "population/parent.json")
    roles = (cfg["fit_role"], cfg["evaluation_role"])
    arrays = {role: {} for role in roles}
    used_receipts = {}
    blocks = bank["shards"] * int(
        np.ceil(bank["rows_per_shard"] / bank["checkpoint_rows"])
    )
    done = 0
    for task in range(bank["shards"]):
        for start in range(0, bank["rows_per_shard"], bank["checkpoint_rows"]):
            folder = source / "banks" / f"shard_{task:03d}" / f"block_{start:07d}"
            if not complete(folder, m["source_contract"]):
                raise ValueError(f"Incomplete bank block: {folder}")
            used_receipts[str(folder / "FINAL.json")] = sha(folder / "FINAL.json")
            with np.load(folder / "bank.npz") as data:
                ids = (
                    task * bank["rows_per_shard"]
                    + start
                    + np.arange(len(data["component"]))
                )
                for role in roles:
                    mask = data["role"] == role
                    for name in ("component", "selected", "features", "theta", "flux"):
                        arrays[role].setdefault(name, []).append(data[name][mask])
                    arrays[role].setdefault("row_id", []).append(ids[mask])
            done += 1
            write(
                out / "PROGRESS.json",
                dict(stage="reading_reserved_banks", done=done, total=blocks),
            )
    data = {
        role: {key: np.concatenate(parts) for key, parts in values.items()}
        for role, values in arrays.items()
    }
    del arrays
    fit, evaluation = (data[role] for role in roles)
    if len(np.intersect1d(fit["row_id"], evaluation["row_id"])):
        raise ValueError("Fit/evaluation bank leakage")
    net_cfg = dict(original["classifier"], seed=original["seed"] + 1)
    model = eqx.tree_deserialise_leaves(
        source / "population/best.eqx",
        classifier_template(fit["features"].shape[1], k, net_cfg),
    )
    counts = {}
    for role in roles:
        d = data[role]
        selected = d["selected"].astype(bool)
        total = np.bincount(d["component"], minlength=k)
        accepted = np.bincount(d["component"][selected], minlength=k)
        if np.any(total == 0) or np.any(accepted == 0):
            raise ValueError("Missing components in independent reserved roles")
        write(
            out / "PROGRESS.json",
            dict(stage=f"frozen_classifier_role_{role}", rows=int(selected.sum())),
        )
        logc = apply_logit_offsets(
            classify(model, d["features"][selected], cfg["classifier_batch_size"]),
            parent["logit_offsets"],
        )
        if not np.isfinite(logc).all() or not np.allclose(np.exp(logc).sum(axis=1), 1):
            raise ValueError("Invalid frozen classifier probabilities")
        counts[str(role)] = dict(parent=total.tolist(), selected=accepted.tolist())
        if role == cfg["fit_role"]:
            np.savez(
                out / "fit.npz",
                logc=logc,
                labels=d["component"][selected],
                row_id=d["row_id"][selected],
                parent_counts=total,
            )
        else:
            moment = ratio_uncertainty(
                logc,
                parent["classifier_reference_frequencies"],
                d["component"][selected],
                blocks=cfg["ratio_blocks"],
                resamples=cfg["ratio_resamples"],
                seed=cfg["seed"],
            )
            write(out / "ratio_moment.json", moment)
            np.savez(
                out / "evaluation.npz",
                theta=d["theta"],
                flux=d["flux"],
                labels=d["component"],
                selected=selected,
                row_id=d["row_id"],
                parent_counts=total,
            )
    write(out / "bank_receipts.json", used_receipts)
    write(out / "counts.json", counts)
    finish(
        out,
        [
            out / n
            for n in (
                "fit.npz",
                "evaluation.npz",
                "ratio_moment.json",
                "bank_receipts.json",
                "counts.json",
            )
        ],
        digest,
        fit_role=roles[0],
        evaluation_role=roles[1],
        disjoint_rows=True,
        classifier_sha256=sha(source / "population/best.eqx"),
        no_training=True,
        no_dsps=True,
    )


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    x, w = np.asarray(values)[order], simplex(weights)[order]
    positive = w > 0
    return np.interp(
        probabilities, np.cumsum(w[positive]) - w[positive] / 2, x[positive]
    )


def common_distances(theta, known, candidates, *, directions, seed):
    """Weighted dense distributions on independent common bank support, not medians."""
    from scripts.feniks_coherent_inference import NAMES

    theta, known = np.asarray(theta, float), simplex(known)
    candidates = {name: simplex(w) for name, w in candidates.items()}
    if theta.shape != (len(known), 15) or not np.isfinite(theta).all():
        raise ValueError("Finite full-15D evaluation bank required")
    scales = np.array(
        [
            max(
                np.diff(weighted_quantile(theta[:, j], known, [0.25, 0.75])).item(),
                1e-8,
            )
            for j in range(15)
        ]
    )

    def distance(values, w):
        order = np.argsort(values)
        return float(
            np.dot(abs(np.cumsum((w - known)[order])[:-1]), np.diff(values[order]))
        )

    marginals = [
        dict(
            model=name,
            parameter=NAMES[j],
            group="physical" if j < 5 else "sfh",
            w1_over_truth_iqr=distance(theta[:, j], w) / scales[j],
        )
        for name, w in candidates.items()
        for j in range(15)
    ]
    rng = np.random.default_rng(seed)
    joints = []
    for group, indices in (("physical", np.arange(5)), ("sfh", np.arange(5, 15))):
        vectors = rng.normal(size=(directions, len(indices)))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        results = {name: [] for name in candidates}
        for vector in vectors:
            values = theta[:, indices] / scales[indices] @ vector
            for name, w in candidates.items():
                results[name].append(distance(values, w))
        for name, values in results.items():
            joints.append(
                dict(
                    model=name,
                    group=group,
                    sliced_wasserstein=float(np.mean(values)),
                    directions=directions,
                )
            )
    return pd.DataFrame(marginals), pd.DataFrame(joints)


def fit_controls(logc, labels, parent_counts, known_u, parent, eligible, config):
    alpha, c = (
        np.asarray(parent["alpha"]),
        np.asarray(parent["classifier_reference_frequencies"]),
    )
    ow = row_weights(labels, known_u, parent_counts)
    v_label = np.bincount(labels, weights=ow, minlength=len(c))
    controls = dict(label_unpenalized=parent_from_selected(v_label, alpha))
    diagnostics = {}
    for name, logs, weights in (
        ("classifier", logc, ow),
        ("label_penalized", np.where(np.eye(len(c), dtype=bool), 0.0, -700.0), v_label),
    ):
        v, diag = fit_selected_weights_kl(
            logs,
            c,
            strength=parent["selected_penalty"],
            alpha=alpha,
            eligible=eligible,
            weak_parent_mass=config["weak_parent_mass"],
            observation_weights=weights,
        )
        controls[name] = parent_from_selected(v, alpha)
        diagnostics[name] = dict(selected_v=v.tolist(), **diag)
    return controls, diagnostics, float(1 / (ow @ ow))


def tails(root, out, data, mixtures, cfg, m):
    """Attribute tail frequency and amplitude; no clipping or causal SFH claims."""
    from scipy.stats import wasserstein_distance

    from euclid_dsps.amortized.reference_capacity import observable_tail_metrics
    from scripts.feniks_coherent_inference import NAMES
    from scripts.feniks_reference_to_parent import weighted_ks

    dataset = Path(m["dataset"])
    bands = [b["name"] for b in read(dataset / "decoder.json")["bands"]]
    truth = pd.read_parquet(
        dataset / "dataset/selected_r29/validation.parquet",
        columns=[f"flux_{b}" for b in bands],
    )
    selected = data["selected"]
    labels, flux, theta = (
        data["labels"][selected],
        data["flux"][selected],
        data["theta"][selected],
    )
    rows, components, populations, joint = [], [], [], []
    target = truth.to_numpy()
    if not np.isfinite(target).all() or not np.isfinite(flux).all():
        raise ValueError("This coherent photometric diagnostic requires finite fluxes")
    scale = np.maximum(np.subtract(*np.quantile(target, [0.75, 0.25], axis=0)), 1e-30)
    target_asinh, predicted_asinh = np.arcsinh(target / scale), np.arcsinh(flux / scale)
    width = np.maximum(
        np.subtract(*np.quantile(target_asinh, [0.75, 0.25], axis=0)), 1e-8
    )
    directions = np.random.default_rng(cfg["seed"] + 20).normal(size=(32, len(bands)))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for model, u in mixtures.items():
        w = row_weights(labels, u, data["parent_counts"])
        distances = [
            wasserstein_distance(
                predicted_asinh / width @ direction,
                target_asinh / width @ direction,
                u_weights=w,
            )
            for direction in directions
        ]
        joint.append(
            dict(
                model=model,
                space="asinh_flux_standardized_by_target_iqr",
                sliced_wasserstein=float(np.mean(distances)),
                directions=len(directions),
            )
        )
        for j, band in enumerate(bands):
            t, p = truth[f"flux_{band}"].to_numpy(), flux[:, j]
            result = observable_tail_metrics(p, t, w)
            rows.append(dict(model=model, band=band, ks=weighted_ks(p, t, w), **result))
            if band != "lsst_r":
                continue
            is_tail = p > result["truth_q999"]
            tail_mass = np.bincount(labels, weights=w * is_tail, minlength=len(u))
            excess = np.bincount(
                labels,
                weights=w
                * np.maximum(p - result["truth_q999"], 0)
                / result["truth_flux_iqr"],
                minlength=len(u),
            )
            mass = np.bincount(labels, weights=w, minlength=len(u))
            for k in range(len(u)):
                components.append(
                    dict(
                        model=model,
                        component=k,
                        parent_weight=float(u[k]),
                        selected_mass=mass[k],
                        tail_mass=tail_mass[k],
                        upper_excess_over_scale=excess[k],
                    )
                )
            for name, mask in (("tail", is_tail), ("other", ~is_tail)):
                total = w[mask].sum()
                for j, parameter in enumerate(NAMES):
                    populations.append(
                        dict(
                            model=model,
                            cohort=name,
                            parameter=parameter,
                            probability=float(total),
                            weighted_mean=float(w[mask] @ theta[mask, j] / total)
                            if total > 0
                            else None,
                        )
                    )
    pd.DataFrame(rows).to_csv(out / "observable_predictive.csv", index=False)
    pd.DataFrame(components).to_csv(out / "tail_components.csv", index=False)
    pd.DataFrame(populations).to_csv(out / "tail_physical_sfh.csv", index=False)
    pd.DataFrame(joint).to_csv(out / "observable_joint.csv", index=False)


def run_case(root, task):
    m, cfg, digest = settings(root)
    if not 0 <= task < len(CASES) or not complete(root / "cache", digest):
        raise ValueError("Valid case and completed cache required")
    name = CASES[task]
    out = root / name
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    source = Path(m["source"])
    parent = read(source / "population/parent.json")
    eligible = pd.read_csv(source / "population/weights.csv").eligible.to_numpy(bool)
    known_u = simplex(m["known_weights"][name])
    fit_out = out / "fit"
    fit_out.mkdir(exist_ok=True)
    if not complete(fit_out, digest):
        write(out / "PROGRESS.json", dict(stage="frozen_ratio_inversion"))
        with np.load(root / "cache/fit.npz") as data:
            controls, diagnostics, effective = fit_controls(
                data["logc"],
                data["labels"],
                data["parent_counts"],
                known_u,
                parent,
                eligible,
                m["source_settings"]["population"],
            )
        write(
            fit_out / "weights.json",
            dict(
                known_u=known_u.tolist(),
                controls={k: v.tolist() for k, v in controls.items()},
                diagnostics=diagnostics,
                effective_selected_fit_rows=effective,
            ),
        )
        finish(fit_out, [fit_out / "weights.json"], digest)
    fitted = read(fit_out / "weights.json")
    controls = {key: np.asarray(values) for key, values in fitted["controls"].items()}
    write(out / "PROGRESS.json", dict(stage="independent_density_evaluation"))
    with np.load(root / "cache/evaluation.npz") as handle:
        data = dict(handle)
    marginals, joint, effective = [], [], {}
    for population, mask in (
        ("parent", np.ones(len(data["labels"]), dtype=bool)),
        ("selected", data["selected"]),
    ):
        true = row_weights(data["labels"][mask], known_u, data["parent_counts"])
        predictions = {
            label: row_weights(data["labels"][mask], u, data["parent_counts"])
            for label, u in controls.items()
        }
        one, many = common_distances(
            data["theta"][mask],
            true,
            predictions,
            directions=cfg["evaluation_directions"],
            seed=cfg["seed"],
        )
        marginals.append(one.assign(population=population))
        joint.append(many.assign(population=population))
        effective[population] = float(1 / (true @ true))
    one, many = pd.concat(marginals), pd.concat(joint)
    one.to_csv(out / "marginals.csv", index=False)
    many.to_csv(out / "joint.csv", index=False)
    write(out / "PROGRESS.json", dict(stage="tail_attribution"))
    tails(
        root, out, data, {"known": known_u, "recovered": controls["classifier"]}, cfg, m
    )
    limits = m["source_settings"]["parent_contracts"]
    gates = {}
    for model in ("classifier", "label_unpenalized", "label_penalized"):
        scores = many[many.model.eq(model) & many.group.eq("physical")]
        errors = one[one.model.eq(model) & one.group.eq("physical")]
        gates[model] = bool(
            scores.sliced_wasserstein.max() <= limits["maximum_physical_sw"]
            and errors.w1_over_truth_iqr.max() <= limits["maximum_physical_marginal_w1"]
        )
    gates["effective_rows"] = (
        min(fitted["effective_selected_fit_rows"], *effective.values())
        >= cfg["minimum_effective_rows"]
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    all_w = {
        "known": row_weights(data["labels"], known_u, data["parent_counts"]),
        **{
            label: row_weights(data["labels"], u, data["parent_counts"])
            for label, u in controls.items()
        },
    }
    for j, ax in enumerate(axes):
        lo, hi = weighted_quantile(data["theta"][:, j], all_w["known"], [0.001, 0.999])
        edges = np.linspace(lo, hi, 65)
        for label in ("known", "classifier", "label_penalized"):
            density = np.histogram(data["theta"][:, j], edges, weights=all_w[label])[
                0
            ] / np.diff(edges)
            ax.stairs(density, edges, label=label)
        ax.set_title(
            ("Redshift", "log mass", "log metallicity", "Dust Av", "Dust slope")[j]
        )
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{name}: known/recovered parent, independent bank role 3")
    fig.tight_layout()
    fig.savefig(out / "parent_physical.png", dpi=150)
    plt.close(fig)
    result = dict(
        case=name,
        gates=gates,
        effective_fit_rows=fitted["effective_selected_fit_rows"],
        effective_evaluation_rows=effective,
        classifier_parent_sw=float(
            many.loc[
                many.model.eq("classifier")
                & many.group.eq("physical")
                & many.population.eq("parent"),
                "sliced_wasserstein",
            ].item()
        ),
        penalty=parent["selected_penalty"],
        no_training=True,
        no_dsps=True,
        diagnostic_only=True,
        ready_for_production=False,
    )
    finish(
        out,
        [
            p
            for p in out.iterdir()
            if p.is_file()
            and p.name not in ("FINAL.json", "PROGRESS.json", "FAILED.json")
        ]
        + [fit_out / "FINAL.json", fit_out / "weights.json"],
        digest,
        **result,
    )


def report(root):
    m, cfg, digest = settings(root)
    out = root / "report"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    missing = [name for name in ("cache", *CASES) if not complete(root / name, digest)]
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        return
    records = [read(root / name / "FINAL.json") for name in CASES]
    moment = read(root / "cache/ratio_moment.json")
    if not all(
        r["gates"]["effective_rows"] and r["gates"]["label_unpenalized"]
        for r in records
    ):
        action = "inspect_finite_bank_and_selection_control_before_ratio_claims"
    elif not all(r["gates"]["label_penalized"] for r in records):
        action = "inspect_penalty_bias_before_new_simulations"
    elif not all(r["gates"]["classifier"] for r in records):
        action = "inspect_ratio_inversion_and_identifiability_before_new_simulations"
    else:
        action = "in_family_recovery_passes_inspect_joint_reference_target_mismatch"
    result = dict(
        cases=records,
        ratio_moment=moment,
        next=action,
        no_training=True,
        no_dsps=True,
        production_prior_modified=False,
        ready_for_production=False,
    )
    write(out / "DECISION.json", result)
    text = "# Frozen-classifier recovery check\n\n" + f"Next: {action}\n\n"
    for record in records:
        text += f"- {record['case']}: physical parent SW {record['classifier_parent_sw']:.5f}; gates {record['gates']}\n"
    text += "\nKnown mixtures and disjoint bank roles, not recovery of the coherent target. Fixed source penalty and alpha; label controls separate sampling/penalty effects. No q, training, new DSPS or production promotion.\n\n"
    text += "Ratio intervals condition on frozen calibration. Passing a moment is not full conditional-ratio validation. Density metrics share independent evaluation support and do not test exact weight identifiability.\n\n"
    text += "Look at each case parent_physical.png, marginals.csv and joint.csv; tail_components.csv ranks r-band frequency/excess contributions, tail_physical_sfh.csv is descriptive not causal.\n"
    (out / "REPORT.md").write_text(text)
    (root / "ROADMAP_STATUS.md").write_text(text)
    (out / "BLOCKED.json").unlink(missing_ok=True)
    finish(
        out,
        [out / "DECISION.json", out / "REPORT.md"],
        digest,
        next=action,
        ready_for_production=False,
    )


def schedule(root):
    _, cfg, digest = settings(root)
    for key, value in cfg["resources"].items():
        print(f"{key.upper()}={value}")
    print(f"NEED_CACHE={int(not complete(root / 'cache', digest))}")
    print(
        "MISSING_CASES="
        + ",".join(
            str(i) for i, name in enumerate(CASES) if not complete(root / name, digest)
        )
    )
    print(f"NEED_REPORT={int(not complete(root / 'report', digest))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "cache", "case", "report", "schedule"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "init":
        initialize(args.source.resolve(), root, args.config.resolve())
        return
    stage = (
        CASES[args.task]
        if args.mode == "case" and 0 <= args.task < len(CASES)
        else args.mode
    )
    if args.mode != "schedule":
        (root / stage / "FAILED.json").unlink(missing_ok=True)
    try:
        if args.mode == "case":
            run_case(root, args.task)
        else:
            globals()[args.mode](root)
        (root / stage / "FAILED.json").unlink(missing_ok=True)
    except Exception as exc:
        if args.mode != "schedule":
            (root / stage).mkdir(exist_ok=True)
            write(
                root / stage / "FAILED.json",
                dict(error=type(exc).__name__, message=str(exc)),
            )
        raise


if __name__ == "__main__":
    main()
