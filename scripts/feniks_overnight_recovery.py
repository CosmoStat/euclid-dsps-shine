"""Recover parent reporting and continue a frozen-parent posterior, no new DSPS.

The new root owns every mutable output. Source banks/models stay read-only.
Numerical replay/transport failures block training; extreme tails stay visible
and do NOT become a scientific PASS merely because optimization is permitted.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts import feniks_coherent_inference as ci
from scripts import feniks_conditional_parent as cp
from scripts import feniks_parent_to_posterior as ptp
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish


def settings(root):
    return ci.settings(root)


def initialize(conditional, posterior, root, config, reuse_parent=None):
    if root.exists() or any(
        s in root.parents or root in s.parents for s in (conditional, posterior)
    ):
        raise ValueError("Use a new separate recovery root")
    cm, cc, cd = cp.ci.settings(conditional)
    pm, pc, pdigest = ptp.settings(posterior)
    if Path(cm["baseline_parent_run"]).resolve() != Path(pm["parent_run"]).resolve():
        raise ValueError("Branches must share the same baseline parent")
    for source, digest, stages in (
        (conditional, cd, ("reference", "population")),
        (posterior, pdigest, ("reference", "posterior", "evaluation", "report")),
    ):
        for stage in stages:
            if not complete(source / stage, digest):
                raise ValueError(f"Completed source stage required: {source / stage}")
    cfg = yaml.safe_load(config.read_text())
    c = cfg["continuation"]
    for value in [
        c[k]
        for k in ("epochs", "block_epochs", "validation_objects", "validation_draws")
    ] + list(cfg["resources"].values()):
        if type(value) is not int or value < 1:
            raise ValueError("Positive integer budgets required")
    if c["epochs"] > 100 or c["epochs"] % c["block_epochs"]:
        raise ValueError("At most 100 additional epochs, in complete evaluation blocks")
    if not 0 < c["minimum_lr"] <= c["learning_rate"] < pc["posterior"]["learning_rate"]:
        raise ValueError("Declare a reduced positive learning rate")
    if not 0 < c["lr_decay_factor"] <= 1 or c["lr_decay_every"] < 1:
        raise ValueError("Invalid continuation learning-rate schedule")
    a = cfg["audit"]
    if (
        type(a["extreme_draws"]) is not int
        or a["extreme_draws"] < 1
        or any(
            not np.isfinite(a[k]) or a[k] <= 0
            for k in ("maximum_replay_latent_error", "transport_tolerance", "far_iqr")
        )
    ):
        raise ValueError("Invalid numerical audit thresholds")
    pinned = {}
    for source in (conditional, posterior):
        for name in ("MANIFEST.json", "experiment.yaml", "reference/FINAL.json"):
            pinned[str(source / name)] = sha(source / name)
    for source, names in (
        (
            conditional,
            (
                "population/FINAL.json",
                "population/parent.json",
                "population/best.eqx",
                "reference/split.json",
            ),
        ),
        (
            posterior,
            (
                "posterior/FINAL.json",
                "posterior/best.eqx",
                "posterior/STOP.json",
                "frozen_parent.json",
                "report/DECISION.json",
            ),
        ),
    ):
        for name in names:
            pinned[str(source / name)] = sha(source / name)
    # Pin small receipt inventories. Large bank contents are verified when read.
    for source in (conditional, posterior):
        for p in (source / "banks").glob("shard_*/FINAL.json"):
            pinned[str(p)] = sha(p)
    if reuse_parent is not None:
        previous, _, previous_digest = settings(reuse_parent)
        if (
            Path(previous["source_conditional"]).resolve() != conditional.resolve()
            or Path(previous["source_posterior"]).resolve() != posterior.resolve()
            or not complete(reuse_parent / "parent", previous_digest)
        ):
            raise ValueError("Parent reuse requires a completed matching recovery")
        if (
            root == reuse_parent
            or root in reuse_parent.parents
            or reuse_parent in root.parents
        ):
            raise ValueError("Use a separate sibling recovery root")
        for name in ("tied", "report"):
            directory = reuse_parent / "parent" / name
            if not complete(directory, cd):
                raise ValueError(f"Incomplete recovered parent stage: {directory}")
            final = directory / "FINAL.json"
            pinned[str(final)] = sha(final)
            for relative, digest in read(final)["artifacts"].items():
                pinned[str(directory / relative)] = digest
        pinned[str(reuse_parent / "MANIFEST.json")] = previous_digest
    root.mkdir(parents=True)
    for name in (
        "logs",
        "parent",
        "audit",
        "posterior",
        "evaluation",
        "report",
        "milestones",
    ):
        (root / name).mkdir()
    shutil.copyfile(config, root / "experiment.yaml")
    # Identical scientific contract for reused conditional artifacts. New code
    # provenance belongs to the outer recovery manifest and frozen snapshot.
    for name in ("MANIFEST.json", "experiment.yaml"):
        shutil.copyfile(conditional / name, root / "parent" / name)
    for name in ("reference", "population", "banks"):
        (root / "parent" / name).symlink_to(
            conditional / name, target_is_directory=True
        )
    for name in ("tied", "report"):
        if reuse_parent is None:
            (root / "parent" / name).mkdir()
        else:
            # These are small fits/reports, not banks. Real files remain usable
            # after lightweight rsync, unlike absolute links to the old run.
            shutil.copytree(reuse_parent / "parent" / name, root / "parent" / name)
    frozen = ["experiment.yaml", "parent/MANIFEST.json", "parent/experiment.yaml"]
    if reuse_parent is not None:
        for name in ("tied", "report"):
            frozen.extend(
                str(p.relative_to(root))
                for p in (root / "parent" / name).rglob("*")
                if p.is_file()
            )
    write(
        root / "MANIFEST.json",
        dict(
            source_conditional=str(conditional),
            source_posterior=str(posterior),
            settings=cfg,
            source_files=pinned,
            frozen_files={name: sha(root / name) for name in frozen},
            new_dsps_simulations=0,
            classifier_retrained=False,
            population_uses_q=False,
            population_uses_target_truth=False,
            production_promotion=False,
            posterior_start="source best checkpoint, explicit optimizer reset",
            reused_parent_recovery=None if reuse_parent is None else str(reuse_parent),
            numerical_replay_contract="float64_theta_replay_v2",
        ),
    )
    if reuse_parent is not None:
        finish(
            root / "parent",
            [root / "parent" / n / "FINAL.json" for n in ("tied", "report")],
            sha(root / "MANIFEST.json"),
            reused_parent_recovery=str(reuse_parent),
            classifier_retrained=False,
            new_dsps_simulations=0,
        )
    print(
        yaml.safe_dump(
            dict(
                parent=(
                    "completed parent report reused; no parent job"
                    if reuse_parent is not None
                    else "saved expanded fit + classifier; only tied control and report recomputed"
                ),
                posterior="same 15D two-expert spline, same frozen parent and bank",
                initial_checkpoint=str(posterior / "posterior/best.eqx"),
                additional_epochs=c["epochs"],
                learning_rate=c["learning_rate"],
                new_dsps_simulations=0,
                existing_parent_simulations=pc["bank"]["shards"]
                * pc["bank"]["rows_per_shard"],
                audit="saved tails + extreme-draw replay + independent expert transport",
                training_gate="numerical consistency only; known tail and calibration FAILs retained",
                resources=cfg["resources"],
                artifacts="recovered parent comparison, tail CSVs/expert attribution, milestone calibration, continued flow, paired final report",
            )
        )
    )


def recover_parent(root):
    m, _, digest = settings(root)
    target = root / "parent"
    if complete(target, digest):
        return
    source = Path(m["source_conditional"])
    _, _, old = cp.ci.settings(source)
    if not complete(source / "population", old):
        raise ValueError("Refuse classifier retraining in recovery")
    cp.population(target)
    cp.report(target)
    if not complete(target / "tied", old) or not complete(target / "report", old):
        raise ValueError("Recovered parent report incomplete")
    finish(
        target,
        [target / "tied/FINAL.json", target / "report/FINAL.json"],
        digest,
        classifier_retrained=False,
        new_dsps_simulations=0,
    )


def cohort_features(source, cfg, digest, label, positions):
    if label == "in_model":
        bank = ptp.selected_rows(source, cfg, digest, 2)
        order = np.argsort(bank["row_id"])
        idx = order[np.searchsorted(bank["row_id"][order], positions)]
        np.testing.assert_array_equal(bank["row_id"][idx], positions)
        return bank["features"][idx], bank["theta"][idx]
    _, _, stats = ci.require_reference(source, digest)
    frame, bands = ci.observed(source, "test")
    truth = pd.read_parquet(
        Path(read(source / "MANIFEST.json")["source"])
        / "dataset/selected_r29/test.parquet",
        columns=ci.NAMES,
    ).to_numpy()
    return ci.features(frame.iloc[positions], bands, stats), truth[positions]


def describe_draws(directory, draws, truth, spec, cfg, latent_x=None):
    from euclid_dsps.amortized.posterior_tail_audit import tail_table

    tables = []
    for space, values, target in (
        ("theta", draws, truth),
        (
            "latent_x",
            (
                np.asarray(ci.to_x(draws.reshape(-1, 15), spec)).reshape(draws.shape)
                if latent_x is None
                else latent_x
            ),
            np.asarray(ci.to_x(truth, spec)),
        ),
    ):
        tables.append(
            tail_table(
                values, target, ci.NAMES, far_iqr=cfg["audit"]["far_iqr"]
            ).assign(space=space)
        )
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(directory / "tails.csv", index=False)
    return result


def replay_extremes(
    model,
    features,
    draws,
    truth,
    spec,
    seed,
    cfg,
    *,
    saved_x=None,
    coordinate_path=None,
):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.posterior import posterior_encoder_state
    from euclid_dsps.amortized.posterior_tail_audit import (
        extreme_positions,
        replay_coordinates,
    )
    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from euclid_dsps.amortized.structured_population import (
        DensityModel,
        transport_audit,
    )

    ids = extreme_positions(draws, truth, cfg["audit"]["extreme_draws"])
    # Keep the numerical replay graph identical to ci.evaluate_posterior: that
    # function materializes only `.x`. Returning the full diagnostic structure
    # can give slightly different tail-rounding in JAX for extreme coordinates.
    sample_x = eqx.filter_jit(
        lambda net, f, k: (
            sample_independent_mixture(
                DensityModel(net.experts[0]), net, k, f, draws.shape[1]
            ).x
        )
    )
    sample_full = eqx.filter_jit(
        lambda net, f, k: sample_independent_mixture(
            DensityModel(net.experts[0]), net, k, f, draws.shape[1]
        )
    )
    rows, contexts = [], []
    extreme_x, physical, stored_x, graph_deltas = [], [], [], []
    for start in sorted(set((ids[:, 0] // 16 * 16).tolist())):
        current = features[start : start + 16]
        padded = np.pad(current, ((0, 16 - len(current)), (0, 0)), mode="edge")
        key = jax.random.PRNGKey(seed + start)
        replay_x = np.asarray(sample_x(model, jnp.asarray(padded), key))
        replay = sample_full(model, jnp.asarray(padded), key)
        replay_full_x = np.asarray(replay.x)
        for i, d in ids[ids[:, 0] // 16 * 16 == start]:
            artifact_x = (
                np.asarray(saved_x[i, d])
                if saved_x is not None
                else np.asarray(ci.to_x(draws[i, d][None], spec))[0]
            )
            rx = replay_x[d, i - start]
            full_rx = replay_full_x[d, i - start]
            error = float(np.max(abs(rx - artifact_x)))
            full_error = float(np.max(abs(full_rx - artifact_x)))
            graph_delta = float(np.max(abs(full_rx - rx)))
            expert = int(replay.component[d, i - start])
            context = jnp.asarray(features[i : i + 1])
            state = posterior_encoder_state(
                DensityModel(model.experts[expert]), context
            )
            rows.append(
                dict(
                    object_position=int(i),
                    draw=int(d),
                    expert=expert,
                    gate_probability=float(
                        jax.nn.softmax(model.logits(context), axis=-1)[0, expert]
                    ),
                    maximum_base_std=float(jnp.exp(state.log_std).max()),
                    replay_latent_error=error,
                    full_graph_replay_latent_error=full_error,
                    xonly_full_graph_delta=graph_delta,
                    maximum_abs_latent=float(abs(artifact_x).max()),
                    sfh05=float(draws[i, d, 9]),
                )
            )
            contexts.append(features[i])
            extreme_x.append(rx)
            physical.append(draws[i, d])
            graph_deltas.append(graph_delta)
            if saved_x is not None:
                stored_x.append(saved_x[i, d])
    coordinates, replay_check = replay_coordinates(
        np.asarray(extreme_x),
        np.asarray(physical),
        spec,
        tolerance=cfg["audit"]["maximum_replay_latent_error"],
        saved_x=None if saved_x is None else np.asarray(stored_x),
    )
    attribution = pd.DataFrame(rows)
    coordinates["object_position"] = attribution.object_position.to_numpy()[
        coordinates.extreme_position
    ]
    coordinates["draw"] = attribution.draw.to_numpy()[coordinates.extreme_position]
    if coordinate_path is not None:
        coordinates.to_csv(coordinate_path, index=False)
    audit = transport_audit(
        model,
        jnp.asarray(contexts),
        jax.random.PRNGKey(seed + 10000),
        tolerance=cfg["audit"]["transport_tolerance"],
        values=jnp.asarray(extreme_x),
    )
    ok = bool(replay_check["passed"] and audit["passed"])
    nonfinite = []
    for i, row in enumerate(audit["experts"]):
        for name, value in row.items():
            if not np.isfinite(value):
                nonfinite.append(f"expert_{i}.{name}")
                row[name] = None
    if replay_check["maximum_replay_latent_error"] is None:
        nonfinite.append("maximum_replay_latent_error")
    graph_delta = max(graph_deltas) if graph_deltas else None
    return attribution, dict(
        **{k: v for k, v in replay_check.items() if k != "passed"},
        passed=ok and not nonfinite,
        nonfinite_metrics=nonfinite,
        transport=audit,
        replay_graph="evaluate_posterior_x_only_v1",
        xonly_full_graph_max_delta=graph_delta,
    )


def audit(root):
    import equinox as eqx

    m, cfg, digest = settings(root)
    out = root / "audit"
    if complete(out, digest):
        if not read(out / "DECISION.json")["safe_to_optimize"]:
            raise ValueError("Saved numerical audit blocks continuation")
        return
    source = Path(m["source_posterior"])
    _, pc, old = ptp.settings(source)
    _, spec, _ = ci.require_reference(source, old)
    checks = {}
    for label, offset in (("in_model", 20), ("coherent_target", 30)):
        directory = out / label
        directory.mkdir(exist_ok=True)
        write(out / "PROGRESS.json", dict(stage=label))
        receipt = source / "evaluation" / label
        if not complete(receipt, old):
            raise ValueError("Completed saved evaluation required")
        with np.load(receipt / "draws.npz") as saved:
            draws, truth, positions = (
                saved[k] for k in ("draws", "truth", "evaluation_positions")
            )
            latent_x = saved["latent_x"] if "latent_x" in saved else None
        features, expected_truth = cohort_features(source, pc, old, label, positions)
        np.testing.assert_allclose(truth, expected_truth, rtol=0, atol=1e-12)
        describe_draws(directory, draws, truth, spec, cfg, latent_x)
        model = eqx.tree_deserialise_leaves(
            source / "posterior/best.eqx", ptp.template(pc, features.shape[1])
        )
        attribution, checks[label] = replay_extremes(
            model,
            features,
            draws,
            truth,
            spec,
            pc["seed"] + offset,
            cfg,
            saved_x=latent_x,
            coordinate_path=directory / "replay_coordinates.csv",
        )
        attribution["row_id"] = positions[attribution.object_position.to_numpy()]
        attribution.to_csv(directory / "extreme_draws.csv", index=False)
        write(directory / "numerics.json", checks[label])
    decision = dict(
        safe_to_optimize=all(v["passed"] for v in checks.values()),
        checks=checks,
        scientific_pass=False,
        meaning="Numerical consistency gate only; tail quality is NOT certified",
        no_clipping=True,
        no_dimensions_removed=True,
    )
    write(out / "DECISION.json", decision)
    finish(
        out,
        sorted(out.glob("*/*.csv"))
        + sorted(out.glob("*/*.json"))
        + [out / "DECISION.json"],
        digest,
    )
    if not decision["safe_to_optimize"]:
        raise ValueError("Numerical replay/transport failed: continuation blocked")


def evaluate_model(source, pc, model, directory, spec, stats, seed_offset, data=None):
    directory.mkdir(exist_ok=True, parents=True)
    ci.evaluate_posterior(
        source,
        model,
        directory,
        dict(pc, seed=pc["seed"] + seed_offset),
        spec,
        stats,
        bank_evaluation=data,
    )
    prefix = "in_model_" if data is not None else ""
    return directory / f"{prefix}draws.npz", directory / f"{prefix}calibration.csv"


def train(root):
    import equinox as eqx

    from euclid_dsps.amortized.structured_population import factor_log_prob
    from scripts.feniks_forward_population import supervised_fit

    m, cfg, digest = settings(root)
    out = root / "posterior"
    if complete(out, digest):
        return
    if (
        not complete(root / "audit", digest)
        or not read(root / "audit/DECISION.json")["safe_to_optimize"]
    ):
        raise ValueError("Passed numerical audit required before training")
    source = Path(m["source_posterior"])
    _, pc, old = ptp.settings(source)
    _, spec, stats = ci.require_reference(source, old)
    c = cfg["continuation"]
    training, validation = (ptp.selected_rows(source, pc, old, role) for role in (0, 1))
    if np.intersect1d(training["row_id"], validation["row_id"]).size:
        raise ValueError("Training/validation identity overlap")
    if len(validation["x"]) < c["validation_objects"]:
        raise ValueError("Insufficient distinct validation objects")
    ids = np.random.default_rng(pc["seed"] + 810).choice(
        len(validation["x"]), c["validation_objects"], replace=False
    )
    probe = (
        validation["features"][ids],
        validation["theta"][ids],
        validation["row_id"][ids],
    )
    net_cfg = dict(
        pc["posterior"],
        **{
            k: c[k]
            for k in (
                "learning_rate",
                "minimum_lr",
                "lr_decay_every",
                "lr_decay_factor",
            )
        },
        seed=pc["seed"] + 10,
        initial_checkpoint=str(source / "posterior/best.eqx"),
    )
    model = ptp.template(pc, training["features"].shape[1])
    write(
        out / "training_measure.json",
        dict(
            source_bank=str(source / "banks"),
            training_rows=len(training["x"]),
            validation_rows=len(validation["x"]),
            weights="unit",
            dimensions=15,
            targets="saved simulator x, never q draws or catalogue truth",
            initial_checkpoint=net_cfg["initial_checkpoint"],
            optimizer_reset=True,
            source_best_sha256=sha(source / "posterior/best.eqx"),
            parent_updated=False,
        ),
    )
    # Milestones are restartable even if interrupted after training but before evaluation.
    for epoch in range(c["block_epochs"], c["epochs"] + 1, c["block_epochs"]):
        checkpoint = root / "milestones" / f"epoch_{epoch:03d}"
        checkpoint.mkdir(exist_ok=True)
        if complete(checkpoint, digest):
            stop = read(checkpoint / "FINAL.json").get("stop")
            if stop:
                write(out / "STOP.json", stop)
                break
            continue
        if (out / "STOP.json").exists():
            break
        write(out / "STAGE.json", dict(stage="training", target_additional_epoch=epoch))
        model = supervised_fit(
            model,
            lambda net, f, x: -factor_log_prob(net, f, x),
            training["features"],
            training["x"],
            (validation["features"], validation["x"]),
            dict(net_cfg, epochs=epoch),
            out,
        )
        # Snapshot best-by-validation NLL, never choose on coherent-target truths.
        shutil.copyfile(out / "best.eqx", checkpoint / "best.eqx")
        write(
            out / "STAGE.json", dict(stage="validation", target_additional_epoch=epoch)
        )
        probe_cfg = dict(
            pc,
            evaluation=dict(
                pc["evaluation"],
                objects=c["validation_objects"],
                draws_per_object=c["validation_draws"],
            ),
        )
        draws_path, cal_path = evaluate_model(
            source, probe_cfg, model, checkpoint, spec, stats, 811, probe
        )
        with np.load(draws_path) as f:
            describe_draws(
                checkpoint, f["draws"], f["truth"], spec, cfg, f.get("latent_x")
            )
        hist = pd.read_json(out / "training.jsonl", lines=True).drop_duplicates(
            "epoch", keep="last"
        )
        prior = [
            read(p)["best_nll"]
            for p in sorted((root / "milestones").glob("epoch_*/FINAL.json"))
        ]
        values = [*prior, float(hist.best_nll.iloc[-1])]
        gains = -np.diff(values)[-3:]
        plateau = len(gains) == 3 and bool(
            np.all(gains < pc["stopping"]["minimum_gain"])
        )
        stop = (
            dict(
                epoch=epoch,
                best_nll=values[-1],
                plateau=plateau,
                reason="validation_plateau" if plateau else "maximum_epoch",
                recent_best_gains=gains.tolist(),
            )
            if plateau or epoch == c["epochs"]
            else None
        )
        finish(
            checkpoint,
            [checkpoint / "best.eqx", draws_path, cal_path, checkpoint / "tails.csv"],
            digest,
            epoch=epoch,
            best_nll=values[-1],
            scientific_pass=False,
            stop=stop,
        )
        if stop:
            write(out / "STOP.json", stop)
    if not (out / "STOP.json").exists():
        # Recovery after the last milestone receipt but before STOP was committed.
        last = read(root / "milestones" / f"epoch_{c['epochs']:03d}" / "FINAL.json")
        write(
            out / "STOP.json",
            dict(
                epoch=last["epoch"],
                best_nll=last["best_nll"],
                plateau=False,
                reason="maximum_epoch",
            ),
        )
    model = eqx.tree_deserialise_leaves(out / "best.eqx", model)
    finish(
        out,
        [out / n for n in ("best.eqx", "STOP.json", "training_measure.json")],
        digest,
        new_dsps_simulations=0,
        parent_updated=False,
    )


def evaluate(root):
    import equinox as eqx

    m, cfg, digest = settings(root)
    out = root / "evaluation"
    if complete(out, digest):
        return
    if not complete(root / "posterior", digest):
        raise ValueError("Completed continuation required")
    source = Path(m["source_posterior"])
    _, pc, old = ptp.settings(source)
    _, spec, stats = ci.require_reference(source, old)
    for label, offset in (("in_model", 20), ("coherent_target", 30)):
        directory = out / label
        directory.mkdir(exist_ok=True)
        if complete(directory, digest):
            continue
        with np.load(source / "evaluation" / label / "draws.npz") as f:
            positions = f["evaluation_positions"]
        features, truth = cohort_features(source, pc, old, label, positions)
        model = eqx.tree_deserialise_leaves(
            root / "posterior/best.eqx", ptp.template(pc, features.shape[1])
        )
        data = (features, truth, positions) if label == "in_model" else None
        draws_path, cal_path = evaluate_model(
            source, pc, model, directory, spec, stats, offset, data
        )
        with np.load(draws_path) as f:
            np.testing.assert_array_equal(f["evaluation_positions"], positions)
            np.testing.assert_allclose(f["truth"], truth, rtol=0, atol=1e-12)
            describe_draws(
                directory, f["draws"], f["truth"], spec, cfg, f.get("latent_x")
            )
        finish(directory, [draws_path, cal_path, directory / "tails.csv"], digest)
    finish(
        out,
        [out / label / "FINAL.json" for label in ("in_model", "coherent_target")],
        digest,
    )


def report(root):
    import matplotlib.pyplot as plt

    m, cfg, digest = settings(root)
    out = root / "report"
    if complete(out, digest):
        return
    missing = [
        s
        for s in ("parent", "audit", "posterior", "evaluation")
        if not complete(root / s, digest)
    ]
    if (
        complete(root / "audit", digest)
        and not read(root / "audit/DECISION.json")["safe_to_optimize"]
    ):
        missing.append("numerical_audit_failed")
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        (root / "ROADMAP_STATUS.md").write_text(
            "# Recovery blocked\n\nMissing: "
            + ", ".join(missing)
            + "\nNo production approval.\n"
        )
        return
    source = Path(m["source_posterior"])
    frames, tails = [], []
    for label in ("in_model", "coherent_target"):
        for version, directory in (
            ("baseline", source / "evaluation" / label),
            ("continued", root / "evaluation" / label),
        ):
            cal = directory / (
                "in_model_calibration.csv"
                if version == "continued" and label == "in_model"
                else "calibration.csv"
            )
            frames.append(pd.read_csv(cal).assign(cohort=label, version=version))
        for version, directory in (
            ("baseline", root / "audit" / label),
            ("continued", root / "evaluation" / label),
        ):
            tails.append(
                pd.read_csv(directory / "tails.csv").assign(
                    cohort=label, version=version
                )
            )
    table, tail = pd.concat(frames), pd.concat(tails)
    table.to_csv(out / "calibration_comparison.csv", index=False)
    tail.to_csv(out / "tail_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    for row, label in enumerate(("in_model", "coherent_target")):
        for col, key in enumerate(("coverage_68", "coverage_95", "pit_ks")):
            ax = axes[row, col]
            for version, color in (("baseline", "#b95143"), ("continued", "#167d91")):
                t = table.loc[table.cohort.eq(label) & table.version.eq(version)]
                ax.plot(range(15), t[key], ".-", label=version, color=color)
            ax.set(title=f"{label}: {key}", xticks=range(15))
            ax.set_xticklabels(ci.NAMES, rotation=80, fontsize=6)
            ax.axhline(
                {"coverage_68": 0.68, "coverage_95": 0.95, "pit_ks": 0.05}[key],
                color="black",
                ls="--",
            )
            ax.legend()
    fig.savefig(out / "calibration_comparison.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for ax, label in zip(axes, ("in_model", "coherent_target"), strict=True):
        for version in ("baseline", "continued"):
            t = tail.loc[
                tail.cohort.eq(label)
                & tail.version.eq(version)
                & tail.space.eq("theta")
            ]
            ax.semilogy(range(15), t.w1_over_truth_iqr, ".-", label=version)
        ax.set(title=f"{label}: full-support W1 / IQR", xticks=range(15))
        ax.set_xticklabels(ci.NAMES, rotation=80, fontsize=6)
        ax.legend()
    fig.savefig(out / "tails.png", dpi=140)
    plt.close(fig)
    history = pd.read_json(root / "posterior/training.jsonl", lines=True)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for key in ("train_nll", "validation_nll", "best_nll"):
        ax.plot(history.epoch, history[key], label=key)
    ax.axhline(
        read(source / "posterior/STOP.json")["best_nll"],
        ls="--",
        color="black",
        label="baseline best",
    )
    ax.set(
        xlabel="Additional epochs from baseline best (optimizer reset)", ylabel="NLL"
    )
    ax.legend()
    fig.savefig(out / "training.png", dpi=140)
    plt.close(fig)
    _, pc, _ = ptp.settings(source)
    flags = {}
    for label in ("in_model", "coherent_target"):
        t = table.loc[table.cohort.eq(label) & table.version.eq("continued")]
        flags[label] = dict(
            coverage=bool(
                np.all(
                    abs(t.coverage_68 - 0.68)
                    <= pc["evaluation"]["maximum_coverage_error"]
                )
                and np.all(
                    abs(t.coverage_95 - 0.95)
                    <= pc["evaluation"]["maximum_coverage_error"]
                )
            ),
            pit=bool(t.pit_ks.max() <= pc["evaluation"]["maximum_pit_ks"]),
        )
    decision = dict(
        calibration=flags,
        parent=read(root / "parent/report/DECISION.json"),
        stop=read(root / "posterior/STOP.json"),
        ready_for_production=False,
        tail_quality_automatically_approved=False,
        target_test_reused_for_development=True,
        next="inspect_parent_comparison_and_posterior_calibration_with_full_tails",
    )
    write(out / "DECISION.json", decision)
    text = (
        "# Targeted overnight recovery\n\n"
        "- Parent comparison recovered, no new DSPS or classifier training.\n"
        "- Continued posterior still belongs to the old frozen parent.\n"
        "- Parent fitting remains blind; simulation truth only trains q.\n"
        "- Numerical audit permits optimization, not approval of tail quality.\n"
        "- Compare parent/report, calibration_comparison.png, tails.png and training.png.\n"
        "- No production promotion; no independent final paper test.\n"
    )
    (out / "REPORT.md").write_text(text)
    (root / "ROADMAP_STATUS.md").write_text(text)
    (out / "BLOCKED.json").unlink(missing_ok=True)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        digest,
    )


def schedule(root):
    _, cfg, digest = settings(root)
    for key, value in cfg["resources"].items():
        print(f"{key.upper()}={value}")
    for name in ("parent", "audit", "posterior", "evaluation", "report"):
        print(f"NEED_{name.upper()}={int(not complete(root / name, digest))}")
    if (
        complete(root / "audit", digest)
        and not read(root / "audit/DECISION.json")["safe_to_optimize"]
    ):
        raise ValueError("Numerical audit failed; no automatic resubmission")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("init", "parent", "audit", "train", "report", "schedule")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--conditional", type=Path)
    parser.add_argument("--posterior", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reuse-parent", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "init":
        initialize(
            args.conditional.resolve(),
            args.posterior.resolve(),
            root,
            args.config.resolve(),
            None if args.reuse_parent is None else args.reuse_parent.resolve(),
        )
        return
    if args.mode == "schedule":
        schedule(root)
        return
    stage = "posterior" if args.mode == "train" else args.mode
    (root / stage / "FAILED.json").unlink(missing_ok=True)
    try:
        if args.mode == "parent":
            recover_parent(root)
        elif args.mode == "train":
            train(root)
            stage = "evaluation"
            (root / stage / "FAILED.json").unlink(missing_ok=True)
            evaluate(root)
        else:
            globals()[args.mode](root)
    except Exception as error:
        write(
            root / stage / "FAILED.json",
            dict(error_type=type(error).__name__, message=str(error)),
        )
        raise


if __name__ == "__main__":
    main()
