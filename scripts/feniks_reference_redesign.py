"""Bounded reference qualification: CPU capacity cells and a paired GPU replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.coherent_coordinates import fit_coordinates, to_theta, to_x
from euclid_dsps.amortized.native_reference import kernel_bandwidths, sample_basis
from euclid_dsps.amortized.reference_capacity import (
    cdf_design,
    empirical_features,
    fit_cdf_weights,
)
from euclid_dsps.amortized.reference_redesign import (
    apply_saved_noise,
    local_basis,
    mass_moment,
    qualification,
    replay_normals,
)
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_inference import NAMES, require_reference
from scripts.feniks_coherent_inference import settings as source_settings
from scripts.feniks_coherent_parent import complete, finish, photometer
from scripts.report_feniks_forward_population import population_metrics

CANDIDATES = ("legacy", "affine_mass", "local_128", "local_256")


def initialize(source: Path, root: Path, config: Path) -> None:
    if (
        root.exists()
        or source == root
        or source in root.parents
        or root in source.parents
    ):
        raise ValueError("Use a new root outside the source")
    cfg = yaml.safe_load(config.read_text())
    if tuple(cfg["candidates"]) != CANDIDATES:
        raise ValueError("Expected the four declared ablations")
    for section in ("evaluation", "resources"):
        if any(not isinstance(v, int) or v <= 0 for v in cfg[section].values()):
            raise ValueError(f"Positive integers required: {section}")
    if (
        cfg["replay"]["extremes"] < 1
        or cfg["replay"]["controls"] < 1
        or cfg["replay"]["extremes"] + cfg["replay"]["controls"] > 128
    ):
        raise ValueError("Require 1..128 total replay objects")
    if any(not np.isfinite(v) or v <= 0 for v in cfg["contracts"].values()):
        raise ValueError("Positive finite qualification thresholds required")
    local = cfg["local"]
    if (
        any(not np.isfinite(v) or v <= 0 for v in local.values())
        or not 2 <= local["gate_neighbors"] < 128
        or local["kernel_neighbors"] < 2
        or local["minimum_bandwidth"] > local["maximum_bandwidth"]
    ):
        raise ValueError("Invalid local basis configuration")
    for key in ("gate_neighbors", "kernel_neighbors"):
        if not isinstance(local[key], int):
            raise ValueError("Integer neighbor counts required")
    for key in ("extremes", "controls", "decoder_batch_size"):
        if not isinstance(cfg["replay"][key], int) or cfg["replay"][key] < 1:
            raise ValueError("Positive integer replay sizes required")
    if (
        not np.isfinite(cfg["replay"]["maximum_noise_scaled_replay_error"])
        or cfg["replay"]["maximum_noise_scaled_replay_error"] <= 0
        or cfg["evaluation"]["thresholds"] < 3
    ):
        raise ValueError("Invalid replay tolerance or CDF grid")
    source_settings(source)
    record = read(source / "report/FINAL.json")
    digest = sha(source / "MANIFEST.json")
    if record["contract"] != digest or record["status"] != "COMPLETE":
        raise ValueError("Source inference report not complete")
    paths = [
        source / n
        for n in (
            "MANIFEST.json",
            "reference/FINAL.json",
            "population/FINAL.json",
            "population/parent.json",
        )
    ]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "experiment.yaml").write_text(config.read_text())
    write(
        root / "MANIFEST.json",
        dict(
            source=str(source),
            settings=cfg,
            frozen_files={"experiment.yaml": sha(root / "experiment.yaml")},
            source_files={str(p): sha(p) for p in paths},
            neural_training=0,
            new_catalogue_rows=0,
            maximum_forward_rows=4
            * (cfg["replay"]["extremes"] + cfg["replay"]["controls"]),
            truth_role="TRAIN-only capacity weights; validation-only design qualification",
            ready_for_production=False,
        ),
    )
    print(
        yaml.safe_dump(
            dict(
                candidates=cfg["candidates"],
                nuisance="All 15D anchors/kernels; SFH remains stochastic",
                forward_rows=4
                * (cfg["replay"]["extremes"] + cfg["replay"]["controls"]),
                resources=cfg["resources"],
                training=0,
                test_truth_used=False,
            ),
            sort_keys=False,
        )
    )


def settings(root: Path) -> tuple:
    m = read(root / "MANIFEST.json")
    for name, digest in m["frozen_files"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Prepared configuration changed: {name}")
    for path, digest in m["source_files"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Immutable source changed: {path}")
    source = Path(m["source"])
    original, scfg, sdigest = source_settings(source)
    return m["settings"], source, original, scfg, sdigest, sha(root / "MANIFEST.json")


def load_model(root: Path, name: str, contract: str) -> tuple[dict, dict]:
    directory = root / name / "model"
    if not complete(directory, contract):
        raise ValueError(f"Incomplete model: {name}")
    with np.load(directory / "basis.npz") as f:
        basis = dict(f)
    return basis, read(directory / "coordinates.json")


def prepare_replay(
    source: Path,
    original: dict,
    scfg: dict,
    sdigest: str,
    basis: dict,
    cfg: dict,
    out: Path,
) -> None:
    """Select reserved influential rows, then recover exact anchors and RNG draws."""
    data_root = Path(original["source"])
    bands = read(data_root / "decoder.json")["bands"]
    r = [b["name"] for b in bands].index(original["selection"]["band"])
    band = bands[r]["name"]
    target = pd.read_parquet(
        data_root / "dataset/selected_r29/validation.parquet", columns=[f"flux_{band}"]
    )
    threshold = float(target[f"flux_{band}"].quantile(0.999))
    u = np.asarray(read(source / "population/parent.json")["u"])
    values = {
        k: [] for k in ("x", "theta", "flux", "component", "task", "start", "row")
    }
    for task in range(scfg["bank"]["shards"]):
        shard = source / "banks" / f"shard_{task:03d}"
        if not complete(shard, sdigest):
            raise ValueError(f"Incomplete source bank: {shard}")
        for start in range(
            0, scfg["bank"]["rows_per_shard"], scfg["bank"]["checkpoint_rows"]
        ):
            block = shard / f"block_{start:07d}"
            if not complete(block, sdigest):
                raise ValueError(f"Incomplete source block: {block}")
            with np.load(block / "bank.npz") as f:
                rows = np.flatnonzero(f["selected"] & (f["role"] == 4))
                for key in ("x", "theta", "flux", "component"):
                    values[key].append(f[key][rows])
                values["row"].append(rows)
                values["task"].append(np.full(len(rows), task))
                values["start"].append(np.full(len(rows), start))
    values = {key: np.concatenate(v) for key, v in values.items()}
    score = u[values["component"]] * np.maximum(values["flux"][:, r] - threshold, 0)
    ne, nc = cfg["replay"]["extremes"], cfg["replay"]["controls"]
    if len(score) < ne + nc:
        raise ValueError("Insufficient reserved rows")
    extremes = np.argsort(-score, kind="stable")[:ne]
    available = np.setdiff1d(np.arange(len(score)), extremes)
    controls = np.random.default_rng(cfg["seed"]).choice(available, nc, replace=False)
    chosen = np.r_[extremes, controls]
    selected = {key: v[chosen] for key, v in values.items()}
    selected["cohort"] = np.array(["upper_excess"] * ne + ["reserved_control"] * nc)
    selected["anchor"] = np.zeros(len(chosen), dtype=int)
    selected["epsilon"] = np.zeros((len(chosen), 15))
    selected["noise"] = np.zeros((len(chosen), len(bands)))
    max_error = 0.0
    pairs = np.unique(np.c_[selected["task"], selected["start"]], axis=0)
    for task, start in pairs:
        subset = np.flatnonzero(
            (selected["task"] == task) & (selected["start"] == start)
        )
        n = min(scfg["bank"]["checkpoint_rows"], scfg["bank"]["rows_per_shard"] - start)
        seed = scfg["seed"] + int(task) * 10000000 + int(start) + 1000
        generated, anchors, noise = sample_basis(
            basis, np.arange(n) % len(u), seed, return_aux=True
        )
        rows = selected["row"][subset]
        max_error = max(
            max_error, float(np.max(np.abs(generated[rows] - selected["x"][subset])))
        )
        selected["anchor"][subset] = anchors[rows]
        selected["epsilon"][subset] = noise[rows]
        selected["noise"][subset] = replay_normals(n, rows, len(bands), seed + 1)
    if max_error > 1e-12:
        raise ValueError(f"Cannot replay original bank RNG: {max_error}")
    np.savez(out / "replay_inputs.npz", **selected)
    write(
        out / "replay_selection.json",
        dict(
            objects=len(chosen),
            threshold=threshold,
            threshold_source="selected validation q99.9",
            band=band,
            anchor_replay_max_error=max_error,
            cohorts=dict(extremes=ne, controls=nc),
        ),
    )


def prepare(root: Path) -> None:
    cfg, source, original, scfg, sdigest, contract = settings(root)
    out = root / "prepare"
    out.mkdir(exist_ok=True)
    if complete(out, contract):
        return
    old, old_spec, _ = require_reference(source, sdigest)
    theta = np.asarray(to_theta(old["anchors"], old_spec))
    positive = [
        n
        for n, flag in zip(NAMES, old_spec.get("positive", [False] * 15), strict=True)
        if flag
    ]
    spec = fit_coordinates(
        theta,
        NAMES,
        np.asarray(old_spec["lower"]),
        np.asarray(old_spec["upper"]),
        positive=positive,
        affine=["log10_stellar_mass"],
    )
    spec.update(
        fitted_on="independent_unweighted_native_reference",
        role="candidate_reference_only",
    )
    x = np.asarray(to_x(theta, spec))
    paths = []
    for name in CANDIDATES:
        write(out / "PROGRESS.json", dict(stage="basis", candidate=name))
        directory = root / name / "model"
        directory.mkdir(parents=True, exist_ok=True)
        if not complete(directory, contract):
            if name == "legacy":
                basis, coord = old, old_spec
            elif name == "affine_mass":
                basis, coord = (
                    dict(
                        anchors=x,
                        conditional=old["conditional"],
                        bandwidth=old["bandwidth"],
                    ),
                    spec,
                )
            else:
                basis, coord = (
                    local_basis(x, int(name.split("_")[1]), cfg["local"], cfg["seed"]),
                    spec,
                )
            np.savez(directory / "basis.npz", **basis)
            write(directory / "coordinates.json", coord)
            finish(
                directory,
                [directory / "basis.npz", directory / "coordinates.json"],
                contract,
                components=basis["conditional"].shape[1],
                dimensions=15,
                target_truth_used=False,
                **mass_moment(
                    basis,
                    coord,
                    np.ones(basis["conditional"].shape[1])
                    / basis["conditional"].shape[1],
                ),
            )
        paths.append(directory / "FINAL.json")
    write(out / "PROGRESS.json", dict(stage="reserved_replay_rows"))
    prepare_replay(source, original, scfg, sdigest, old, cfg, out)
    finish(
        out,
        paths + [out / "replay_inputs.npz", out / "replay_selection.json"],
        contract,
    )


def capacity(root: Path, task: int) -> None:
    cfg, _, original, _, _, contract = settings(root)
    if not 0 <= task < len(CANDIDATES):
        raise ValueError("Invalid capacity task")
    name = CANDIDATES[task]
    out = root / name / "capacity"
    out.mkdir(exist_ok=True)
    if complete(out, contract):
        return
    basis, spec = load_model(root, name, contract)
    source = Path(original["source"])
    train = pd.read_parquet(
        source / "dataset/parent/train.parquet", columns=NAMES
    ).to_numpy()
    validation = pd.read_parquet(
        source / "dataset/parent/validation.parquet", columns=NAMES
    ).to_numpy()
    write(out / "PROGRESS.json", dict(stage="analytic_cdf_fit"))
    x = np.asarray(to_x(train, spec))
    f, y, directions, grid = cdf_design(
        basis,
        x[:, :5],
        cfg["evaluation"]["random_directions"],
        cfg["evaluation"]["thresholds"],
        cfg["seed"],
    )
    u, optimizer = fit_cdf_weights(f, y)
    v = empirical_features(np.asarray(to_x(validation, spec))[:, :5], directions, grid)
    n = cfg["evaluation"]["population_draws"]
    rng = np.random.default_rng(cfg["seed"])
    sampled = np.asarray(
        to_theta(sample_basis(basis, rng.choice(len(u), n, p=u), cfg["seed"] + 1), spec)
    )
    write(out / "PROGRESS.json", dict(stage="validation_closure"))
    one, joint = population_metrics(sampled, validation, NAMES, cfg["seed"])
    reference = train[
        rng.choice(len(train), min(len(train), len(validation)), replace=False)
    ]
    baseline_cdf = empirical_features(
        np.asarray(to_x(reference, spec))[:, :5], directions, grid
    )
    _, baseline_joint = population_metrics(reference, validation, NAMES, cfg["seed"])

    def physical_sw(table):
        return float(
            table.loc[table.group.eq("physical"), "sliced_wasserstein"].iloc[0]
        )

    baseline = dict(
        cdf_max=float(np.max(np.abs(baseline_cdf - v))),
        physical_sw=physical_sw(baseline_joint),
    )
    metrics = dict(
        validation_cdf_max=float(np.max(np.abs(f @ u - v))),
        physical_sw=physical_sw(joint),
        physical_marginal_max=float(
            one.loc[one.group.eq("physical"), "w1_over_truth_iqr"].max()
        ),
        **mass_moment(basis, spec, u),
    )
    gates, limits = qualification(metrics, baseline, cfg["contracts"])
    write(
        out / "truth_diagnostic_weights.json",
        dict(u=u.tolist(), truth_used=True, production_prior=False),
    )
    one.to_csv(out / "marginals.csv", index=False)
    pd.concat(
        [
            joint.assign(comparison="oracle_vs_validation"),
            baseline_joint.assign(comparison="train_subsample_vs_validation"),
        ]
    ).to_csv(out / "joint.csv", index=False)
    residuals = []
    count = len(grid[0])
    for k in range(len(directions)):
        sl = slice(k * count, (k + 1) * count)
        residuals.append(
            dict(
                direction=k,
                label=NAMES[k] if k < 5 else f"joint_{k - 5}",
                train_max=float(abs(f[sl] @ u - y[sl]).max()),
                validation_max=float(abs(f[sl] @ u - v[sl]).max()),
                empirical_max=float(abs(baseline_cdf[sl] - v[sl]).max()),
            )
        )
    pd.DataFrame(residuals).to_csv(out / "cdf_by_direction.csv", index=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    for k, ax in enumerate(axes):
        edges = np.linspace(*np.quantile(validation[:, k], [0.001, 0.999]), 65)
        for label, data in (("Validation truth", validation), (name, sampled)):
            ax.stairs(
                np.histogram(data[:, k], edges)[0] / (len(data) * np.diff(edges)),
                edges,
                label=label,
            )
        ax.set_title(NAMES[k], fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "TRAIN-truth capacity fit; validation only; NOT blind population recovery"
    )
    fig.tight_layout()
    fig.savefig(out / "physical.png", dpi=150)
    plt.close(fig)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        contract,
        candidate=name,
        components=len(u),
        **metrics,
        baseline=baseline,
        limits=limits,
        gates=gates,
        capacity_qualified=all(gates.values()),
        optimizer=optimizer,
        test_used=False,
        coordinate_note="Legacy mixed projections use legacy mass coordinates; SW uses common physical space",
        production_promotion=False,
    )


def replay(root: Path) -> None:
    cfg, _, original, _, _, contract = settings(root)
    out = root / "replay"
    out.mkdir(exist_ok=True)
    if complete(out, contract):
        return
    if not complete(root / "prepare", contract):
        raise ValueError("Preparation incomplete")
    with np.load(root / "prepare/replay_inputs.npz") as f:
        selected = dict(f)
    source = Path(original["source"])
    decoder, noise = read(source / "decoder.json"), read(source / "noise.json")
    predict = photometer(decoder, cfg["replay"]["decoder_batch_size"])
    old, old_spec = load_model(root, "legacy", contract)
    ids, eps = selected["anchor"], selected["epsilon"]
    cases = dict(
        anchor=np.asarray(to_theta(old["anchors"][ids], old_spec)),
        legacy=selected["theta"],
    )
    for name in ("affine_mass", "local_128"):
        basis, spec = load_model(root, name, contract)
        cases[name] = np.asarray(
            to_theta(basis["anchors"][ids] + kernel_bandwidths(basis)[ids] * eps, spec)
        )
    # Both local resolutions share exactly the same kernels; only gates differ.
    broad, _ = load_model(root, "local_256", contract)
    local, _ = load_model(root, "local_128", contract)
    if not np.array_equal(broad["anchors"], local["anchors"]) or not np.array_equal(
        kernel_bandwidths(broad), kernel_bandwidths(local)
    ):
        raise ValueError("Local paired replay no longer represents both resolutions")
    tables, theta_tables, fluxes = [], [], {}
    saved_error = None
    for name, theta in cases.items():
        write(
            out / "PROGRESS.json", dict(stage="forward", case=name, objects=len(theta))
        )
        flux = predict(theta)
        if not np.isfinite(flux).all() or np.any(flux < 0):
            raise ValueError(f"Invalid paired forward output: {name}")
        observed, errors = apply_saved_noise(
            flux, decoder["bands"], noise, selected["noise"]
        )
        fluxes[name] = flux
        if name == "legacy":
            saved_error = float(np.max(np.abs(observed - selected["flux"]) / errors))
        theta_frame = pd.DataFrame(theta, columns=NAMES)
        theta_frame["case"], theta_frame["cohort"], theta_frame["pair"] = (
            name,
            selected["cohort"],
            np.arange(len(theta)),
        )
        theta_frame["original_component"], theta_frame["anchor"] = (
            selected["component"],
            ids,
        )
        theta_tables.append(theta_frame)
        for j, band in enumerate(decoder["bands"]):
            tables.append(
                pd.DataFrame(
                    dict(
                        pair=np.arange(len(theta)),
                        cohort=selected["cohort"],
                        case=name,
                        band=band["name"],
                        noiseless_flux=flux[:, j],
                        paired_noisy_flux=observed[:, j],
                        saved_original_flux=selected["flux"][:, j],
                        paired_error=errors[:, j],
                    )
                )
            )
    pd.concat(tables).to_csv(out / "paired_photometry.csv", index=False)
    pd.concat(theta_tables).to_csv(out / "paired_theta.csv", index=False)
    r = [b["name"] for b in decoder["bands"]].index(original["selection"]["band"])
    threshold = read(root / "prepare/replay_selection.json")["threshold"]
    rows = []
    for cohort in np.unique(selected["cohort"]):
        mask = selected["cohort"] == cohort
        for case, flux in fluxes.items():
            rows.append(
                dict(
                    cohort=cohort,
                    case=case,
                    objects=int(mask.sum()),
                    r_above_validation_q999=int(np.sum(flux[mask, r] > threshold)),
                    r_flux_median=float(np.median(flux[mask, r])),
                    r_flux_max=float(flux[mask, r].max()),
                )
            )
    pd.DataFrame(rows).to_csv(out / "replay_summary.csv", index=False)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cohort in zip(axes, ("upper_excess", "reserved_control"), strict=True):
        mask = selected["cohort"] == cohort
        for case in ("legacy", "affine_mass", "local_128"):
            ax.scatter(
                fluxes["anchor"][mask, r], fluxes[case][mask, r], label=case, s=16
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Anchor noiseless r flux",
            ylabel="Perturbed noiseless r flux",
            title=cohort,
        )
        lo = max(min(fluxes[n][mask, r].min() for n in fluxes), 1e-40)
        hi = max(fluxes[n][mask, r].max() for n in fluxes)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "paired_bright_replay.png", dpi=150)
    plt.close(fig)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        contract,
        objects=len(ids),
        forward_rows=4 * len(ids),
        maximum_noise_scaled_replay_error=saved_error,
        replay_identity_pass=saved_error
        <= cfg["replay"]["maximum_noise_scaled_replay_error"],
        limitation="Targeted pairs, not a population predictive sample or alpha estimate",
    )


def report(root: Path) -> None:
    _, _, _, _, _, contract = settings(root)
    out = root / "report"
    out.mkdir(exist_ok=True)
    stages = [root / n / "capacity" for n in CANDIDATES] + [root / "replay"]
    missing = [str(p.relative_to(root)) for p in stages if not complete(p, contract)]
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        return
    rows = [read(p / "FINAL.json") for p in stages[:-1]]
    replay_result = read(root / "replay/FINAL.json")
    eligible = [
        r for r in rows if r["capacity_qualified"] and r["candidate"] != "legacy"
    ]
    best = (
        min(eligible, key=lambda r: (r["components"], r["validation_cdf_max"]))
        if eligible
        else None
    )
    ready = best is not None and replay_result["replay_identity_pass"]
    next_action = (
        "new_reference_bank_then_blind_parent_benchmark"
        if ready
        else "revise_reference_capacity"
        if best is None
        else "repair_forward_replay_contract"
    )
    pd.DataFrame(
        [
            {k: r[k] for k in ("candidate", "components")}
            | {"train_cdf_max_error": r["optimizer"]["train_cdf_max_error"]}
            | {
                k: r[k]
                for k in (
                    "validation_cdf_max",
                    "physical_sw",
                    "physical_marginal_max",
                    "capacity_qualified",
                    "finite_linear_mass_moment",
                )
            }
            for r in rows
        ]
    ).to_csv(out / "comparison.csv", index=False)
    decision = dict(
        selected_candidate=best["candidate"] if best else None,
        next=next_action,
        ready_for_reference_bank_benchmark=ready,
        ready_for_production=False,
        blind_parent_recovered=False,
        targeted_replay_is_not_predictive_closure=True,
        paired_tail_review_required=True,
        population_tail_remediation_validated=False,
        replay_identity_pass=replay_result["replay_identity_pass"],
    )
    if ready:
        model = root / best["candidate"] / "model"
        write(
            out / "REFERENCE_CANDIDATE.json",
            dict(
                candidate=best["candidate"],
                basis=str(model / "basis.npz"),
                coordinates=str(model / "coordinates.json"),
                artifacts={
                    str(p): sha(p)
                    for p in (model / "basis.npz", model / "coordinates.json")
                },
                initialization="uniform component weights; NEVER diagnostic truth weights",
                diagnostic_weights_forbidden=True,
                target_truth_used_for_basis=False,
                architecture_qualified_with_development_truth=True,
                require_new_classifier_and_selection_efficiencies=True,
            ),
        )
    write(out / "DECISION.json", decision)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, limit, label in (
        (axes[0], "validation_cdf_max", "cdf_limit", "Maximum validation CDF residual"),
        (axes[1], "physical_sw", "physical_sw_limit", "Physical sliced-Wasserstein"),
    ):
        ax.bar(
            range(4),
            [r[metric] for r in rows],
            color=["#777777", "#287f8e", "#bd6150", "#538b49"],
        )
        ax.scatter(
            range(4),
            [r["limits"][limit] for r in rows],
            marker="_",
            s=250,
            color="black",
            label="Qualification limit",
        )
        ax.set_xticks(range(4), [r["candidate"] for r in rows], rotation=15)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Truth-supervised capacity only; common physical SW, coordinate-dependent joint CDF"
    )
    fig.tight_layout()
    fig.savefig(out / "reference_capacity_comparison.png", dpi=150)
    plt.close(fig)
    text = "# Reference redesign qualification\n\n"
    text += f"Next: {next_action}\nSelected: {decision['selected_candidate']}\n\n"
    text += "Capacity is a TRAIN-truth diagnostic with validation-only selection, not blind recovery.\n"
    text += "Legacy projected features use a different mass coordinate; compare physical SW and axis residuals.\n"
    text += "Local 128 vs 256 compares resolution; affine vs local also changes gates and physical kernels.\n"
    text += "Paired replay separates original anchors and perturbations, not the full predictive population.\n"
    text += "SFH metrics remain explicit, not a requirement of sharp individual SFH posteriors.\n"
    text += "A passing candidate permits a new-bank benchmark only. Parent, posterior and paper gates remain open.\n"
    (out / "REPORT.md").write_text(text)
    (root / "ROADMAP_STATUS.md").write_text(text)
    (out / "BLOCKED.json").unlink(missing_ok=True)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        contract,
        **decision,
    )


def schedule(root: Path) -> None:
    """Validated shell assignments for missing work, never trust file existence alone."""
    cfg, _, _, _, _, contract = settings(root)
    for key, value in cfg["resources"].items():
        print(f"{key.upper()}={int(value)}")
    print(f"NEED_PREPARE={int(not complete(root / 'prepare', contract))}")
    missing = [
        str(i)
        for i, name in enumerate(CANDIDATES)
        if not complete(root / name / "capacity", contract)
    ]
    print(f"MISSING_CELLS={','.join(missing)}")
    print(f"NEED_REPLAY={int(not complete(root / 'replay', contract))}")
    print(f"NEED_REPORT={int(not complete(root / 'report', contract))}")


def main():
    import matplotlib

    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("init", "prepare", "capacity", "replay", "report", "schedule")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/feniks_reference_redesign.yaml"),
    )
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "init":
        initialize(args.source.resolve(), root, args.config.resolve())
    elif args.mode == "capacity":
        capacity(root, args.task)
    else:
        globals()[args.mode](root)


if __name__ == "__main__":
    main()
