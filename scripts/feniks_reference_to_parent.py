"""Fixed-basis objective qualification followed by one blind parent benchmark."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.coherent_coordinates import to_theta, to_x
from euclid_dsps.amortized.native_reference import sample_basis
from euclid_dsps.amortized.physical_capacity import (
    fit_physical_weights,
    physical_cdf_design,
)
from euclid_dsps.amortized.reference_capacity import (
    cdf_design,
    empirical_features,
    fit_cdf_weights,
    observable_tail_metrics,
)
from euclid_dsps.amortized.reference_redesign import mass_moment, qualification
from scripts import feniks_coherent_inference as ci
from scripts import feniks_reference_redesign as redesign
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish
from scripts.report_feniks_forward_population import population_metrics


def exploratory_cdf_admission(record: dict, cfg: dict) -> dict:
    """Post-hoc compute admission, NOT a relaxed scientific qualification."""
    rows = [r for r in record["results"] if r["objective"] == "physical_cdf"]
    if record["passed"] is not False or len(rows) != 1:
        raise ValueError(
            "Exploration requires one completed, failed physical_cdf result"
        )
    row = rows[0]
    values = (
        [
            row[k]
            for k in (
                "validation_cdf_max",
                "physical_sw",
                "physical_marginal_max",
                "log10_mean_mass",
            )
        ]
        + list(record["baseline"].values())
        + list(cfg["qualification_contracts"].values())
    )
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite capacity metrics cannot be admitted")
    gates, limits = qualification(
        row, record["baseline"], cfg["qualification_contracts"]
    )
    if gates != row["gates"] or limits != row["limits"] or row["passed"] is not False:
        raise ValueError("Inconsistent saved qualification certificate")
    failed = [k for k, passed in gates.items() if not passed]
    excess = row["validation_cdf_max"] - limits["cdf_limit"]
    # Deliberately narrow, disclosed after seeing the failed result; no statistical
    # equivalence claim. Other failed gates must still stop expensive downstream work.
    if (
        failed != ["validation_cdf"]
        or not (0 < excess <= 0.002)
        or row["validation_cdf_max"] > 0.04
    ):
        raise ValueError("Only a small CDF-only failure is eligible for exploration")
    return dict(
        mode="explicit_cdf_only_exploration",
        strict_qualification_passed=False,
        gates=gates,
        limits=limits,
        validation_cdf_max=row["validation_cdf_max"],
        cdf_excess=excess,
        maximum_admitted_cdf_excess=0.002,
        maximum_admitted_cdf=0.04,
        post_hoc_execution_policy=True,
        production_promotion=False,
    )


def initialize_exploratory(source: Path, root: Path) -> None:
    """Import immutable completed geometry/results; never import fitted parent u."""
    if root.exists() or root in source.parents or source in root.parents:
        raise ValueError("Use a new root outside completed runs")
    m, cfg, digest = ci.settings(source)
    if m.get("exploratory_admission") or not m.get("no_posterior_training"):
        raise ValueError("Expected an original reference-to-parent qualification run")
    if set(m["frozen_files"]) != {"experiment.yaml"}:
        raise ValueError("Unsupported additional frozen configuration files")
    if not complete(source / "qualification", digest):
        raise ValueError("Completed qualification required")
    ci.require_reference(source, digest)
    q = read(source / "qualification/FINAL.json")
    admission = exploratory_cdf_admission(q, cfg)
    artifacts = {}
    for stage in ("reference", "qualification"):
        receipt = read(source / stage / "FINAL.json")
        for name in receipt["artifacts"]:
            path = Path(stage) / name
            if Path(name).is_absolute() or ".." in path.parts:
                raise ValueError("Only stage-local imported artifacts are supported")
            artifacts[str(path)] = sha(source / path)
        artifacts[f"{stage}/FINAL.json"] = sha(source / stage / "FINAL.json")
    for name in ("basis.npz", "coordinates.json", "feature_stats.json"):
        if f"reference/{name}" not in artifacts:
            raise ValueError("Missing reference artifact in receipt")
    if "qualification/diagnostic_weights.json" not in artifacts:
        raise ValueError("Missing capacity control weights in receipt")
    m = copy.deepcopy(m)
    m.update(
        exploratory_admission=admission,
        qualification_source=str(source),
        qualification_source_sha256=sha(source / "qualification/FINAL.json"),
    )
    m["source_files"].update({str(source / p): h for p, h in artifacts.items()})
    m["source_files"][str(source / "MANIFEST.json")] = digest
    root.mkdir(parents=True)
    for name in ("logs", "reference", "banks", "population", "qualification", "report"):
        (root / name).mkdir()
    shutil.copy2(source / "experiment.yaml", root / "experiment.yaml")
    write(root / "MANIFEST.json", m)
    new_digest = sha(root / "MANIFEST.json")
    for stage in ("reference", "qualification"):
        record = read(source / stage / "FINAL.json")
        files = []
        for name in record["artifacts"]:
            dest = root / stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / stage / name, dest)
            files.append(dest)
        original = root / stage / "SOURCE_FINAL.json"
        shutil.copy2(source / stage / "FINAL.json", original)
        values = {
            k: v
            for k, v in record.items()
            if k not in ("artifacts", "status", "contract")
        }
        finish(root / stage, files + [original], new_digest, **values)
    print(
        yaml.safe_dump(
            dict(
                admission=admission,
                qualification_recomputed=False,
                original_run_unchanged=str(source),
                components=cfg["reference"]["components"],
                basis="unchanged local joint 15D anchor mixture",
                nuisance="all 10 SFH coordinates sampled with native associations and kernels",
                selection=m["selection"],
                efficiencies="binomial counts including rejected parents",
                new_reference_simulations=cfg["bank"]["shards"]
                * cfg["bank"]["rows_per_shard"],
                classifier=cfg["classifier"],
                resources=cfg["resources"],
                posterior="not trained",
                truth_weights="report-only capacity control; never used in population fitting",
            ),
            sort_keys=False,
        )
    )


def initialize(source: Path, root: Path, config: Path) -> None:
    if root.exists() or root in source.parents or source in root.parents:
        raise ValueError("Use a new root outside completed runs")
    rcfg, old, manifest, original, _, rdigest = redesign.settings(source)
    if (
        not complete(source / "report", rdigest)
        or not complete(source / "replay", rdigest)
        or not read(source / "replay/FINAL.json")["replay_identity_pass"]
    ):
        raise ValueError("Completed redesign and successful saved-bank replay required")
    overrides = yaml.safe_load(config.read_text())
    if overrides["candidate"] != "local_256":
        raise ValueError("This controlled continuation keeps local_256 fixed")
    basis, _ = redesign.load_model(source, overrides["candidate"], rdigest)
    cfg = copy.deepcopy(original)
    for key, value in overrides.items():
        if isinstance(value, dict) and key != "resources":
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    # The copied local geometry, not obsolete global kernel parameters, is authoritative.
    cfg["reference"] = dict(
        components=basis["conditional"].shape[1],
        anchors=len(basis["anchors"]),
        family="saved_local_joint_anchor_mixture",
        geometry=str(source / cfg["candidate"] / "model"),
    )
    cfg.pop("posterior", None)
    cfg["evaluation"] = dict(population_draws=cfg["evaluation"]["population_draws"])
    cfg["qualification_contracts"] = rcfg["contracts"]
    for section in ("bank", "resources"):
        if any(not isinstance(v, int) or v < 1 for v in cfg[section].values()):
            raise ValueError(f"Positive integer {section} required")
    k = cfg["reference"]["components"]
    if cfg["bank"]["rows_per_shard"] % k or cfg["bank"]["checkpoint_rows"] % k:
        raise ValueError("Balanced parent component bank required")
    if not cfg["classifier"]["fixed_validation"] or cfg["classifier"]["epochs"] < 1:
        raise ValueError("Bounded classifier with fixed validation required")
    if cfg["evaluation"]["population_draws"] < 2 or cfg["bank"]["shards"] < 2:
        raise ValueError("Independent banks and population evaluation required")
    cap = cfg["capacity"]
    if any(not np.isfinite(v) or v <= 0 for v in cap.values()) or cap["grid_size"] < 3:
        raise ValueError("Invalid capacity budget")
    for key in ("draws_per_component", "directions", "grid_size", "evaluation_draws"):
        if not isinstance(cap[key], int):
            raise ValueError("Integer capacity sizes required")
    if any(not np.isfinite(v) or v <= 0 for v in cfg["parent_contracts"].values()):
        raise ValueError("Invalid parent criteria")
    model = source / cfg["candidate"] / "model"
    files = dict(manifest["source_files"])
    paths = [
        source / "MANIFEST.json",
        source / "report/FINAL.json",
        source / "replay/FINAL.json",
        old / "MANIFEST.json",
        old / "reference/feature_stats.json",
    ]
    paths += [model / n for n in ("FINAL.json", "basis.npz", "coordinates.json")]
    files.update({str(p): sha(p) for p in paths})
    root.mkdir(parents=True)
    for name in ("logs", "reference", "banks", "population", "qualification", "report"):
        (root / name).mkdir()
    (root / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    m = copy.deepcopy(manifest)
    m.update(
        settings=cfg,
        source_files=files,
        frozen_files={"experiment.yaml": sha(root / "experiment.yaml")},
        redesign=str(source),
        baseline_inference=str(old),
        qualification_uses_train_truth=True,
        population_uses_target_truth=False,
        no_posterior_training=True,
        production_promotion=False,
        reference_design_used_development_truth=True,
    )
    write(root / "MANIFEST.json", m)
    print(
        yaml.safe_dump(
            dict(
                candidate=cfg["candidate"],
                components=k,
                dimensions=15,
                basis="unchanged saved local joint-anchor kernels; no diagnostic u handoff",
                nuisance="native joint 10D SFH associations plus stochastic kernels",
                selection=manifest["selection"],
                efficiencies="binomial counts over all simulated parents, including rejected",
                capacity_draws=k * cap["draws_per_component"],
                new_reference_simulations=cfg["bank"]["shards"]
                * cfg["bank"]["rows_per_shard"],
                classifier=cfg["classifier"],
                resources=overrides["resources"],
                posterior="not trained until blind parent validated",
                target_catalogue="unchanged",
                truth_weights="never used in population fitting",
                outputs="capacity qualification, banks, classifier, v/u/alpha, parent/selected/observable report, roadmap",
            ),
            sort_keys=False,
        )
    )


def physical_sw(joint: pd.DataFrame) -> float:
    return float(joint.loc[joint.group.eq("physical"), "sliced_wasserstein"].iloc[0])


def qualify(root: Path) -> None:
    m, cfg, digest = ci.settings(root)
    out = root / "qualification"
    if complete(out, digest):
        if not read(out / "FINAL.json")["passed"]:
            raise SystemExit(2)
        ci.require_reference(root, digest)
        return
    source = Path(m["redesign"])
    basis, spec = redesign.load_model(
        source, cfg["candidate"], sha(source / "MANIFEST.json")
    )
    target = Path(m["source"])
    train = pd.read_parquet(
        target / "dataset/parent/train.parquet", columns=ci.NAMES
    ).to_numpy()
    validation = pd.read_parquet(
        target / "dataset/parent/validation.parquet", columns=ci.NAMES
    ).to_numpy()
    rcfg = read(source / "MANIFEST.json")["settings"]
    f, y, dirs, grid = cdf_design(
        basis,
        np.asarray(to_x(train, spec))[:, :5],
        rcfg["evaluation"]["random_directions"],
        rcfg["evaluation"]["thresholds"],
        rcfg["seed"],
    )
    minimax, info = fit_cdf_weights(f, y)
    cap = cfg["capacity"]
    k = basis["conditional"].shape[1]
    cache = out / "physical_cdf.npz"
    if not complete(out / "features", digest):
        parts = []
        for j in range(k):
            write(out / "PROGRESS.json", dict(stage="component_cdfs", done=j, total=k))
            x = sample_basis(
                basis, np.full(cap["draws_per_component"], j), cfg["seed"] + j
            )
            parts.append(np.asarray(to_theta(x, spec))[:, :5])
        pf, py, pq = physical_cdf_design(
            np.asarray(parts),
            train[:, :5],
            cap["directions"],
            cap["grid_size"],
            cfg["seed"],
        )
        np.savez(cache, features=pf, target=py, quadrature=pq)
        (out / "features").mkdir(exist_ok=True)
        finish(out / "features", [cache], digest)
    with np.load(cache) as values:
        pf, py, pq = [values[n] for n in ("features", "target", "quadrature")]
    write(out / "PROGRESS.json", dict(stage="physical_cdf_solver"))
    u, solver = fit_physical_weights(
        pf,
        py,
        pq,
        f,
        y,
        info["train_cdf_max_error"] + cap["original_cdf_slack"],
        cap["maximum_solver_seconds"],
    )
    v = empirical_features(np.asarray(to_x(validation, spec))[:, :5], dirs, grid)
    rng = np.random.default_rng(cfg["seed"] + 501)
    empirical = train[
        rng.choice(len(train), min(len(train), len(validation)), replace=False)
    ]
    _, baseline_joint = population_metrics(
        empirical, validation, ci.NAMES, cfg["seed"] + 500
    )
    baseline = dict(
        cdf_max=float(
            abs(
                empirical_features(np.asarray(to_x(empirical, spec))[:, :5], dirs, grid)
                - v
            ).max()
        ),
        physical_sw=physical_sw(baseline_joint),
    )
    rows = []
    samples = {}
    tables = []
    for label, weights in (("old_minimax", minimax), ("physical_cdf", u)):
        labels = np.random.default_rng(cfg["seed"] + 502).choice(
            k, cap["evaluation_draws"], p=weights
        )
        draws = np.asarray(
            to_theta(sample_basis(basis, labels, cfg["seed"] + 503), spec)
        )
        one, joint = population_metrics(draws, validation, ci.NAMES, cfg["seed"] + 500)
        metrics = dict(
            validation_cdf_max=float(abs(f @ weights - v).max()),
            physical_sw=physical_sw(joint),
            physical_marginal_max=float(
                one.loc[one.group.eq("physical"), "w1_over_truth_iqr"].max()
            ),
            **mass_moment(basis, spec, weights),
        )
        gates, limits = qualification(metrics, baseline, cfg["qualification_contracts"])
        rows.append(
            dict(
                objective=label,
                **metrics,
                gates=gates,
                limits=limits,
                passed=all(gates.values()),
            )
        )
        tables.append(one.assign(objective=label))
        samples[label] = draws
    pd.concat(tables).to_csv(out / "marginals.csv", index=False)
    pd.DataFrame(
        [{k: v for k, v in row.items() if k not in ("gates", "limits")} for row in rows]
    ).to_csv(out / "comparison.csv", index=False)
    write(
        out / "diagnostic_weights.json",
        dict(u=u.tolist(), production_prior=False, target_truth_used=True),
    )
    write(out / "solver.json", solver)
    population_plot(out / "physical_objective.png", validation, samples)
    # Only the geometry moves forward. Never copy diagnostic u into population/.
    ref = root / "reference"
    for name in ("basis.npz", "coordinates.json"):
        shutil.copy2(source / cfg["candidate"] / "model" / name, ref / name)
    shutil.copy2(
        Path(m["baseline_inference"]) / "reference/feature_stats.json",
        ref / "feature_stats.json",
    )
    passed = rows[1]["passed"]
    finish(
        ref,
        [ref / n for n in ("basis.npz", "coordinates.json", "feature_stats.json")],
        digest,
        target_truth_used_for_basis=False,
        diagnostic_weights_excluded=True,
    )
    finish(
        out,
        [
            out / n
            for n in (
                "marginals.csv",
                "comparison.csv",
                "diagnostic_weights.json",
                "solver.json",
                "physical_objective.png",
            )
        ],
        digest,
        passed=passed,
        results=rows,
        baseline=baseline,
        truth_diagnostic_only=True,
        production_weights_initialized="balanced reference; learn from photometry only",
    )
    if not passed:
        raise SystemExit(2)


def population_plot(path: Path, truth: np.ndarray, series: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for j, ax in enumerate(axes):
        edges = np.linspace(*np.quantile(truth[:, j], [0.001, 0.999]), 65)
        outside = []
        for name, values in {"Validation truth": truth, **series}.items():
            outside.append(
                f"{name}: {np.mean((values[:, j] < edges[0]) | (values[:, j] > edges[-1])):.1%}"
            )
            ax.stairs(
                np.histogram(values[:, j], edges)[0] / (len(values) * np.diff(edges)),
                edges,
                label=name,
            )
        ax.set_title(ci.NAMES[j], fontsize=9)
        ax.set_xlabel("Off-axis mass\n" + "\n".join(outside), fontsize=6)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def population_admitted(root: Path, m: dict, cfg: dict, digest: str) -> bool:
    if not complete(root / "qualification", digest):
        return False
    record = read(root / "qualification/FINAL.json")
    if m.get("exploratory_admission"):
        expected = exploratory_cdf_admission(record, cfg)
        if expected != m["exploratory_admission"]:
            raise ValueError("Exploratory admission does not match qualification")
        if (
            sha(root / "qualification/SOURCE_FINAL.json")
            != m["qualification_source_sha256"]
        ):
            raise ValueError("Imported qualification provenance changed")
        return True
    return record["passed"] is True


def require_population_admission(root: Path) -> tuple:
    m, cfg, digest = ci.settings(root)
    if not population_admitted(root, m, cfg, digest):
        raise ValueError(
            "Reference capacity not qualified or explicitly admitted; no downstream work"
        )
    return m, cfg, digest


def bank(root: Path, task: int) -> None:
    require_population_admission(root)
    ci.bank(root, task)


def population(root: Path) -> None:
    _, cfg, digest = require_population_admission(root)
    data = ci.load_bank(root, cfg, digest)
    k = cfg["reference"]["components"]
    counts = np.array(
        [
            np.bincount(
                data["component"][(data["role"] == r) & data["selected"]], minlength=k
            )
            for r in range(5)
        ]
    )
    write(
        root / "population/selected_support.json",
        dict(
            selected_counts_by_role=counts.tolist(),
            all_roles_covered=bool(np.all(counts > 0)),
            no_alpha_floor=True,
        ),
    )
    if np.any(counts == 0):
        write(
            root / "population/BLOCKED.json",
            dict(
                reason="unobserved_component_in_selected_bank_role",
                zero_role_components=np.argwhere(counts == 0).tolist(),
            ),
        )
        raise ValueError("Unidentified selected support; inspect selected_support.json")
    (root / "population/BLOCKED.json").unlink(missing_ok=True)
    del data
    ci.population(root)


def weighted_ks(predicted: np.ndarray, truth: np.ndarray, weights: np.ndarray) -> float:
    predicted, truth, weights = [
        np.asarray(v, float) for v in (predicted, truth, weights)
    ]
    if (
        predicted.ndim != 1
        or truth.ndim != 1
        or weights.shape != predicted.shape
        or not len(truth)
        or not all(np.isfinite(v).all() for v in (predicted, truth, weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError(
            "Finite scalar observations and positive weight measure required"
        )
    order = np.argsort(predicted)
    p = predicted[order]
    w = weights[order] / weights.sum()
    t = np.sort(truth)
    grid = np.unique(np.r_[p, t])
    cumulative = np.r_[0, np.cumsum(w)]
    return float(
        abs(
            cumulative[np.searchsorted(p, grid, side="right")]
            - np.searchsorted(t, grid, side="right") / len(t)
        ).max()
    )


def report(root: Path) -> None:
    m, cfg, digest = ci.settings(root)
    out = root / "report"
    if complete(out, digest):
        return
    missing = [
        s for s in ("qualification", "population") if not complete(root / s, digest)
    ]
    if complete(root / "qualification", digest) and not population_admitted(
        root, m, cfg, digest
    ):
        missing.append("qualification_failed")
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        (root / "ROADMAP_STATUS.md").write_text(
            "# Reference to parent\n\nBlocked: "
            + ", ".join(missing)
            + "\nNo posterior training or production approval.\n"
        )
        return
    basis, spec, _ = ci.require_reference(root, digest)
    parent = read(root / "population/parent.json")
    u = np.array(parent["u"])
    data = ci.load_bank(root, cfg, digest)
    mask = data["selected"] & (data["role"] == 4)
    indices = np.flatnonzero(mask)
    weights = u[data["component"][mask]]
    weights /= weights.sum()
    rng = np.random.default_rng(cfg["seed"] + 700)
    n = cfg["evaluation"]["population_draws"]
    labels = rng.choice(len(u), n, p=u)
    learned = np.asarray(to_theta(sample_basis(basis, labels, cfg["seed"] + 701), spec))
    selected = data["theta"][rng.choice(indices, n, p=weights)]
    source = Path(m["source"])
    truth = pd.read_parquet(source / "dataset/parent/validation.parquet")
    target_selected = pd.read_parquet(
        source / "dataset/selected_r29/validation.parquet"
    )
    # This truth-assisted control is evaluated only after blind fitting, on the
    # same reserved bank. It never supplies targets or initialization to the fit.
    oracle_u = np.asarray(
        read(root / "qualification/diagnostic_weights.json")["u"], float
    )
    if (
        oracle_u.shape != u.shape
        or not np.isfinite(oracle_u).all()
        or np.any(oracle_u < 0)
        or not np.isclose(oracle_u.sum(), 1)
    ):
        raise ValueError("Invalid capacity control distribution")
    oracle_weights = oracle_u[data["component"][mask]]
    if oracle_weights.sum() <= 0:
        raise ValueError("No reserved selected support for capacity control")
    oracle_weights /= oracle_weights.sum()
    control_rng = np.random.default_rng(cfg["seed"] + 700)
    oracle = np.asarray(
        to_theta(
            sample_basis(
                basis, control_rng.choice(len(u), n, p=oracle_u), cfg["seed"] + 701
            ),
            spec,
        )
    )
    oracle_selected = data["theta"][control_rng.choice(indices, n, p=oracle_weights)]
    marginals = []
    joint = []
    for name, values, actual in (
        ("parent", learned, truth),
        ("selected", selected, target_selected),
        ("capacity_parent", oracle, truth),
        ("capacity_selected", oracle_selected, target_selected),
    ):
        one, multi = population_metrics(
            values, actual[ci.NAMES].to_numpy(), ci.NAMES, cfg["seed"] + 702
        )
        marginals.append(one.assign(population=name))
        joint.append(multi.assign(population=name))
    one = pd.concat(marginals)
    multi = pd.concat(joint)
    one.to_csv(out / "marginals.csv", index=False)
    multi.to_csv(out / "joint.csv", index=False)
    population_plot(
        out / "parent_physical.png",
        truth[ci.NAMES].to_numpy(),
        dict(learned_parent=learned, truth_assisted_capacity=oracle),
    )
    population_plot(
        out / "selected_physical.png",
        target_selected[ci.NAMES].to_numpy(),
        dict(learned_selected=selected, truth_assisted_capacity=oracle_selected),
    )
    bands = [b["name"] for b in read(source / "decoder.json")["bands"]]
    predictive = []
    for label, measure in (
        ("learned_parent", weights),
        ("truth_assisted_capacity", oracle_weights),
    ):
        for j, band in enumerate(bands):
            p = data["flux"][mask, j]
            t = target_selected[f"flux_{band}"].to_numpy()
            predictive.append(
                dict(
                    model=label,
                    band=band,
                    ks=weighted_ks(p, t, measure),
                    **observable_tail_metrics(p, t, measure),
                )
            )
    comparison = pd.DataFrame(predictive)
    comparison.to_csv(out / "observable_predictive_comparison.csv", index=False)
    pred = comparison.loc[comparison.model.eq("learned_parent")].copy()
    pred.drop(columns="model").to_csv(out / "observable_predictive.csv", index=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for offset, (label, group) in zip(
        (-0.2, 0.2), comparison.groupby("model", sort=False), strict=True
    ):
        positions = np.arange(len(group)) + offset
        axes[0].bar(positions, group.ks, width=0.4, label=label)
        axes[1].bar(
            positions, group.predicted_mass_above_truth_q999, width=0.4, label=label
        )
    axes[0].legend(fontsize=7)
    axes[0].axhline(
        cfg["parent_contracts"]["maximum_observable_ks"], color="black", ls="--"
    )
    axes[0].set_ylabel("Weighted flux CDF KS")
    axes[1].axhline(
        cfg["parent_contracts"]["maximum_tail_probability"], color="black", ls="--"
    )
    axes[1].set_ylabel("Predicted mass above validation q99.9")
    for ax in axes:
        ax.set_xticks(range(len(pred)), pred.band, rotation=55, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "observable_predictive.png", dpi=150)
    plt.close(fig)
    limits = cfg["parent_contracts"]
    classifier = read(root / "population/classifier_audit.json")
    stopping = read(root / "population/STOP.json")
    alpha_actual = len(target_selected) / len(truth)
    alpha_error = abs(parent["alpha_parent"] - alpha_actual)
    gates = dict(
        parent_physical_sw=physical_sw(joint[0]) <= limits["maximum_physical_sw"],
        selected_physical_sw=physical_sw(joint[1]) <= limits["maximum_physical_sw"],
        physical_marginals=bool(
            one.loc[
                one.group.eq("physical") & one.population.isin(["parent", "selected"]),
                "w1_over_truth_iqr",
            ].max()
            <= limits["maximum_physical_marginal_w1"]
        ),
        parent_alpha=alpha_error <= limits["maximum_alpha_error"],
        observable_cdfs=bool(pred.ks.max() <= limits["maximum_observable_ks"]),
        observable_tails=bool(
            pred.predicted_mass_above_truth_q999.max()
            <= limits["maximum_tail_probability"]
        ),
        classifier_gain=classifier["audit_null_nll"] - classifier["audit_nll"]
        >= limits["minimum_classifier_gain"],
        classifier_ratio_moment=classifier["independent_ratio_moment_median_error"]
        <= limits["maximum_ratio_moment_error"],
    )
    decision = dict(
        gates=gates,
        limits=limits,
        ready_for_fresh_posterior_benchmark=all(gates.values()),
        ready_for_production=False,
        strict_qualification_passed=read(root / "qualification/FINAL.json")["passed"],
        exploratory_continuation=bool(m.get("exploratory_admission")),
        exploratory_admission=m.get("exploratory_admission"),
        population_uses_target_truth=False,
        population_uses_q=False,
        alpha_learned=parent["alpha_parent"],
        alpha_validation=alpha_actual,
        alpha_absolute_error=alpha_error,
        reserved_effective_rows=float(1 / (weights @ weights)),
        next="fresh_parent_simulations_and_15d_posterior"
        if all(gates.values())
        else "inspect_parent_and_observation_closure",
        sfh_conditional_validated=False,
        validation_used_for_development=True,
    )
    control_pred = comparison.loc[comparison.model.eq("truth_assisted_capacity")]
    decision["capacity_control"] = dict(
        truth_assisted=True,
        used_for_population_fitting=False,
        same_reserved_bank=True,
        parent_physical_sw=physical_sw(joint[2]),
        selected_physical_sw=physical_sw(joint[3]),
        alpha_parent=float(oracle_u @ np.asarray(parent["alpha"])),
        alpha_absolute_error=float(
            abs(oracle_u @ np.asarray(parent["alpha"]) - alpha_actual)
        ),
        maximum_observable_ks=float(control_pred.ks.max()),
        maximum_tail_probability=float(
            control_pred.predicted_mass_above_truth_q999.max()
        ),
        reserved_effective_rows=float(1 / (oracle_weights @ oracle_weights)),
        role="diagnostic comparison, not an independent model or likelihood oracle",
    )
    decision["classifier_stop"] = stopping
    decision["optimization_converged"] = bool(stopping["plateau"])
    write(out / "DECISION.json", decision)
    text = "# Reference to parent benchmark\n\n" + "\n".join(
        f"- {key}: {'PASS' if value else 'FAIL'}" for key, value in gates.items()
    )
    if decision["exploratory_continuation"]:
        text += "\n\nCapacity CDF remains FAIL. Explicit exploratory execution only; original thresholds unchanged."
    text += "\n\nPlots compare the blind learned parent with a truth-assisted capacity control on the same reserved bank. Control weights never enter fitting."
    text += (
        "\n\nNext: "
        + decision["next"]
        + "\n\nNo truth weights used in blind fitting. SFH population errors remain reported, not erased. Posterior not trained. No production promotion. A new independent final evaluation set is still required.\n"
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


def schedule(root: Path) -> None:
    m, cfg, digest = ci.settings(root)
    if complete(root / "qualification", digest) and not population_admitted(
        root, m, cfg, digest
    ):
        raise ValueError(
            "Completed qualification failed; inspect results, do not resubmit unchanged"
        )
    for key, value in cfg["resources"].items():
        print(f"{key.upper()}={int(value)}")
    for name in ("qualification", "population", "report"):
        print(f"NEED_{name.upper()}={int(not complete(root / name, digest))}")
    print(
        "MISSING_BANKS="
        + ",".join(
            str(i)
            for i in range(cfg["bank"]["shards"])
            if not complete(root / "banks" / f"shard_{i:03d}", digest)
        )
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        choices=(
            "init",
            "init-exploratory",
            "qualify",
            "bank",
            "population",
            "report",
            "schedule",
        ),
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--redesign", type=Path)
    p.add_argument("--source-run", type=Path)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/feniks_reference_to_parent.yaml"),
    )
    p.add_argument("--task", type=int, default=0)
    args = p.parse_args()
    root = args.root.resolve()
    if args.mode == "init":
        if args.redesign is None:
            p.error("init requires --redesign")
        initialize(args.redesign.resolve(), root, args.config.resolve())
    elif args.mode == "init-exploratory":
        if args.source_run is None:
            p.error("init-exploratory requires --source-run")
        initialize_exploratory(args.source_run.resolve(), root)
    elif args.mode == "bank":
        bank(root, args.task)
    else:
        globals()[args.mode](root)


if __name__ == "__main__":
    main()
