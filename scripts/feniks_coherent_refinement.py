"""Bounded physical-factor continuation and independent read-only SFH audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.coherent_coordinates import to_theta, validate_theta
from euclid_dsps.amortized.distribution_diagnostics import physical_sw, tail_table
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish
from scripts.feniks_coherent_parent import settings as base_settings
from scripts.feniks_coherent_representation import FACTORS, projection_zero_table


def settings(root):
    m, cfg, digest = base_settings(root)
    for path, expected in m["source_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Immutable source changed: {path}")
    return m, cfg, digest


def prepare(source, root, config):
    if root.exists() or source == root or source in root.parents:
        raise ValueError("Use a new root outside the immutable representation run")
    _, old, contract = base_settings(source)
    for stage in (*FACTORS, "sfh_zeros", "report"):
        if not complete(source / stage, contract):
            raise ValueError(f"Incomplete source: {stage}")
    cfg = yaml.safe_load(config.read_text())
    steps = cfg["milestones"]
    if (
        not steps
        or steps[0] != 0
        or steps[-1] != cfg["epochs"]
        or steps != sorted(set(steps))
        or any(not isinstance(x, int) or x < 0 for x in steps)
    ):
        raise ValueError("Milestones must increase from zero to the epoch budget")
    if not np.isfinite(cfg["learning_rate"]) or not 0 < cfg["learning_rate"] < 1:
        raise ValueError("Invalid learning rate")
    for value in (cfg["draws"], cfg["epochs"], *cfg["resources"].values()):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Positive integer budgets required")
    if old["flow"].get("fixed_validation") is not True:
        raise ValueError("Source must use fixed validation")
    paths = [
        source / n
        for n in (
            "MANIFEST.json",
            "coordinates.json",
            "projection.json",
            "cache/train.npz",
            "cache/validation.npz",
            "cache/test.npz",
            "cache/projection_train.parquet",
            "report/draws.npz",
        )
    ]
    paths += [
        source / s / n
        for s in FACTORS
        for n in ("best.eqx", "FINAL.json", "training.jsonl")
    ]
    source_files = {str(p): sha(p) for p in paths}
    history = pd.read_json(source / "physical/training.jsonl", lines=True)
    best_epoch = int(history.loc[history.validation_nll.idxmin(), "epoch"])
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "experiment.yaml").write_text(config.read_text())
    write(
        root / "MANIFEST.json",
        dict(
            settings=cfg,
            source=str(source),
            source_settings=old,
            source_files=source_files,
            source_physical_best_epoch=best_epoch,
            frozen_files={"experiment.yaml": sha(root / "experiment.yaml")},
            truth_used_for_training=True,
            population_uses_q=False,
            conditional_frozen=True,
            dataset_modified=False,
            production_prior_modified=False,
        ),
    )
    print(
        f"Physical factor: best source epoch {best_epoch}, +{cfg['epochs']} epochs, "
        f"LR={cfg['learning_rate']}; Adam state reset at initial best checkpoint."
    )
    print("Frozen conditional SFH. Parallel CPU tails/zero-precision audit.")
    print(
        "No new photometry; no blind parent or posterior training. No production promotion."
    )


def load_factor(source, index, checkpoint=None):
    import equinox as eqx

    from euclid_dsps.amortized.structured_population import factor_template

    cfg = read(source / "MANIFEST.json")["settings"]
    template = factor_template(
        cfg["flow"],
        5 if index == 0 else 10,
        1 if index == 0 else 5,
        cfg["seed"] + index,
    )
    return eqx.tree_deserialise_leaves(
        checkpoint or source / FACTORS[index] / "best.eqx", template
    )


def physical_draws(model, count, seed):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import factor_sample

    sample = eqx.filter_jit(lambda m, k: factor_sample(m, jnp.zeros((512, 1)), k))
    return np.concatenate(
        [
            np.asarray(sample(model, jax.random.PRNGKey(seed + i)))
            for i in range(int(np.ceil(count / 512)))
        ]
    )[:count]


def physical_theta(x, transform):
    return np.asarray(to_theta(np.pad(x, ((0, 0), (0, 10))), transform))[:, :5]


def checkpoint_metrics(model, source, cfg, out, epoch):
    import equinox as eqx

    transform = read(source / "coordinates.json")
    x = physical_draws(model, cfg["draws"], cfg["draw_seed"])
    theta = physical_theta(x, transform)
    with np.load(source / "cache/validation.npz") as f:
        vx, vt = f["x"][:, :5], f["theta"][:, :5]
    out.mkdir(exist_ok=True, parents=True)
    eqx.tree_serialise_leaves(out / "best.eqx", model)
    np.savez(out / "draws.npz", x=x, theta=theta)
    tails = tail_table(theta, vt, transform["names"][:5])
    tails.to_csv(out / "validation_tails.csv", index=False)
    return dict(
        epoch=epoch,
        validation_physical_sw=physical_sw(theta, vt, cfg["draw_seed"]),
        validation_x_sw=physical_sw(x, vx, cfg["draw_seed"]),
        max_marginal_w1=float(tails.w1_over_iqr.max()),
        checkpoint_sha256=sha(out / "best.eqx"),
        checkpoint_selection="best validation NLL so far, not SW or test",
        evaluation_split="validation",
        common_random_seed=cfg["draw_seed"],
    )


def fit(root):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import (
        factor_log_prob,
        transport_audit,
    )
    from scripts.feniks_forward_population import supervised_fit

    m, cfg, digest = settings(root)
    source = Path(m["source"])
    out = root / "physical"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    with np.load(source / "cache/train.npz") as f:
        train = f["x"][:, :5]
    with np.load(source / "cache/validation.npz") as f:
        val = f["x"][:, :5]
    features = np.zeros((len(train), 1))
    validation = (np.zeros((len(val), 1)), val)
    training = dict(
        m["source_settings"]["flow"],
        seed=m["source_settings"]["seed"],
        learning_rate=cfg["learning_rate"],
        initial_checkpoint=str(source / "physical/best.eqx"),
    )
    # Constant low LR; source schedule must not silently override it.
    for key in ("lr_decay_every", "lr_decay_factor", "minimum_lr"):
        training.pop(key, None)
    for epoch in cfg["milestones"]:
        step = out / f"milestone_{epoch:03d}"
        if complete(step, digest):
            continue
        saved = (
            read(out / "RESUME.json")["epoch"] if (out / "RESUME.json").exists() else 0
        )
        if saved > epoch:
            raise ValueError(
                "Missing past milestone; do not label a later checkpoint as earlier"
            )
        model = load_factor(source, 0)
        if epoch:
            model = supervised_fit(
                model,
                lambda net, c, t: -factor_log_prob(net, c, t),
                features,
                train,
                validation,
                dict(training, epochs=epoch),
                out,
            )
        row = checkpoint_metrics(model, source, cfg, step, epoch)
        row["best_nll"] = (
            read(out / "RESUME.json")["best_nll"]
            if epoch
            else float(
                pd.read_json(
                    source / "physical/training.jsonl", lines=True
                ).validation_nll.min()
            )
        )
        finish(
            step,
            [step / n for n in ("best.eqx", "draws.npz", "validation_tails.csv")],
            digest,
            **row,
        )
        write(out / "DISTRIBUTION_PROGRESS.json", row)
    model = load_factor(source, 0, out / "best.eqx")
    audit = transport_audit(
        model,
        jnp.asarray(validation[0][:8]),
        jax.random.PRNGKey(731),
        values=jnp.asarray(val[:8]),
    )
    write(out / "transport.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Physical flow inverse/Jacobian audit failed")
    files = [out / n for n in ("best.eqx", "training.jsonl", "transport.json")]
    files += [out / f"milestone_{e:03d}/FINAL.json" for e in cfg["milestones"]]
    finish(
        out,
        files,
        digest,
        epoch=cfg["epochs"],
        conditional_frozen=True,
        training_budget_exhausted=True,
        converged_not_certified=True,
    )


def precision_table(original, p32, k32, p64, k64):
    table = projection_zero_table(original, p32, k32)
    rows = []
    for name in table.parameter:
        a, b = p32[name].to_numpy(), p64[name].to_numpy()
        zero = a == 0
        rows.append(
            dict(
                parameter=name,
                zero32_count=int(zero.sum()),
                zero64_count=int(np.sum(b == 0)),
                zero32_nonzero64_count=int(np.sum(zero & (b != 0))),
                zero32_zero64_count=int(np.sum(zero & (b == 0))),
                max_contrast_difference=float(np.max(abs(a - b))),
                max_abs64_at_zero32=float(np.max(abs(b[zero]))) if zero.any() else 0.0,
            )
        )
    floor64 = projection_zero_table(p64, p64, k64)[
        ["parameter", "equal_floor_knots_count"]
    ]
    return table.merge(pd.DataFrame(rows), on="parameter").merge(
        floor64.rename(columns={"equal_floor_knots_count": "floor64_count"}),
        on="parameter",
    )


def audit(root):
    from euclid_dsps.prior_learning.spline15d import project_diffsky_frame_to_spline15d

    m, _, digest = settings(root)
    out = root / "audit"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    source = Path(m["source"])
    names = read(source / "coordinates.json")["names"]
    tables = []
    with (
        np.load(source / "report/draws.npz") as pred,
        np.load(source / "cache/test.npz") as truth,
    ):
        for space in ("x", "theta"):
            tables.append(
                tail_table(pred[space], truth[space], names).assign(space=space)
            )
    pd.concat(tables).to_csv(out / "saved_draw_tail_bias.csv", index=False)
    original = pd.read_parquet(source / "cache/projection_train.parquet")
    kw = dict(n_sfh_bins=read(source / "projection.json")["n_sfh_bins"], batch_size=256)
    p32, k32 = project_diffsky_frame_to_spline15d(original, **kw)
    p64, k64 = project_diffsky_frame_to_spline15d(original, **kw, precision="float64")
    table = precision_table(original, p32, k32, p64, k64)
    table.to_csv(out / "sfh_precision.csv", index=False)
    replay = bool(table.max_replay_error.max() <= 1e-5)
    finish(
        out,
        [out / "saved_draw_tail_bias.csv", out / "sfh_precision.csv"],
        digest,
        replay_pass=replay,
        projection_objects=len(original),
        dataset_modified=False,
        sfh_retrained=False,
        dequantization_applied=False,
        conclusion="Disappearing zeros are precision-sensitive; persistent zeros are NOT proof of physical atoms",
    )


def plots(out, comparisons, truth, names, history, trajectory, tail_rows, zero_table):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for k, name in enumerate(names[:5]):
        for row, limits in enumerate(((0.005, 0.995), (0.0, 1.0))):
            lo, hi = np.quantile(truth[:, k], limits)
            if row == 1:
                lo = min(lo, *(x[:, k].min() for x in comparisons.values()))
                hi = max(hi, *(x[:, k].max() for x in comparisons.values()))
            edges = np.linspace(lo, hi, 70)
            for label, values in dict(truth=truth, **comparisons).items():
                # counts/N/bin-width retain the probability outside central panels.
                counts, _ = np.histogram(values[:, k], edges)
                axes[row, k].stairs(
                    counts / len(values) / np.diff(edges), edges, label=label
                )
            axes[row, k].set_title(name, fontsize=9)
            axes[row, k].set_xlabel(
                "theta; central 99%" if row == 0 else "theta; full range"
            )
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "physical_before_after.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for field in ("validation_nll", "best_nll"):
        axes[0].plot(history.epoch, history[field], label=field)
    axes[0].legend()
    axes[0].set_xlabel("Additional epoch")
    axes[1].plot(trajectory.epoch, trajectory.validation_physical_sw, "o-")
    axes[1].set_ylabel("Validation physical SW; best-NLL checkpoint")
    axes[1].set_xlabel("Additional epoch")
    fig.tight_layout()
    fig.savefig(out / "continuation.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    theta = tail_rows[tail_rows.space == "theta"]
    bottom = np.zeros(len(theta))
    for field in ("lower_tail_w1", "central_w1", "upper_tail_w1"):
        axes[0].bar(np.arange(len(theta)), theta[field], bottom=bottom, label=field)
        bottom += theta[field].to_numpy()
    axes[0].set_xticks(
        np.arange(len(theta)), theta.parameter, rotation=60, ha="right", fontsize=8
    )
    axes[0].set_ylabel("Saved flow W1 / truth IQR")
    axes[0].legend()
    for i, field in enumerate(
        ("zero32_nonzero64_count", "zero32_zero64_count", "equal_floor_knots_count")
    ):
        axes[1].bar(
            np.arange(len(zero_table)) + 0.25 * i,
            zero_table[field],
            width=0.25,
            label=field,
        )
    axes[1].set_xticks(
        np.arange(len(zero_table)),
        zero_table.parameter,
        rotation=35,
        ha="right",
        fontsize=8,
    )
    axes[1].legend(fontsize=8)
    axes[1].set_ylabel("Train replay objects")
    fig.tight_layout()
    fig.savefig(out / "tails_and_zeros.png", dpi=140)
    plt.close(fig)


def report(root):
    m, cfg, digest = settings(root)
    out = root / "report"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    missing = [s for s in ("audit", "physical") if not complete(root / s, digest)]
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        raise RuntimeError(f"Incomplete stages: {missing}; resume missing work only")
    for epoch in cfg["milestones"]:
        if not complete(root / f"physical/milestone_{epoch:03d}", digest):
            raise ValueError(f"Missing distribution milestone: {epoch}")
    source = Path(m["source"])
    transform = read(source / "coordinates.json")
    names = transform["names"]
    with np.load(source / "cache/test.npz") as f:
        tx, tt = f["x"], f["theta"]
    with np.load(source / "cache/validation.npz") as f:
        vx, vt = f["x"], f["theta"]
    rows, tails, comparisons = [], [], {}
    for epoch, label in ((0, "before"), (cfg["epochs"], "after")):
        with np.load(root / f"physical/milestone_{epoch:03d}/draws.npz") as f:
            x, theta = f["x"], f["theta"]
        comparisons[label] = theta
        for space, a, b in (("theta", theta, tt[:, :5]), ("x", x, tx[:, :5])):
            rows.append(
                dict(
                    comparison=label,
                    space=space,
                    physical_sw=physical_sw(a, b, cfg["draw_seed"]),
                )
            )
            tails.append(
                tail_table(a, b, names[:5]).assign(space=space, comparison=label)
            )
    for space, a, b in (("theta", vt, tt), ("x", vx, tx)):
        rows.append(
            dict(
                comparison="validation_vs_test",
                space=space,
                physical_sw=physical_sw(a[:, :5], b[:, :5], cfg["draw_seed"]),
            )
        )
    pd.DataFrame(rows).to_csv(out / "physical_joint.csv", index=False)
    pd.concat(tails).to_csv(out / "physical_tail_bias.csv", index=False)
    trajectory = pd.DataFrame(
        [
            read(root / f"physical/milestone_{e:03d}/FINAL.json")
            for e in cfg["milestones"]
        ]
    ).drop(columns=["artifacts"])
    trajectory.to_csv(out / "validation_trajectory.csv", index=False)
    # Joint 15D draws: new p(a), unchanged p(b|a). This is NOT a posterior.
    import equinox as eqx
    import jax

    from euclid_dsps.amortized.structured_population import StructuredPopulation
    from scripts.report_feniks_forward_population import population_metrics

    model = StructuredPopulation(
        load_factor(source, 0, root / "physical/best.eqx"),
        load_factor(source, 1),
        tuple(names),
    )
    sample = eqx.filter_jit(lambda net, key: net.sample(key, 512))
    dx = np.concatenate(
        [
            np.asarray(sample(model, jax.random.PRNGKey(cfg["draw_seed"] + i)))
            for i in range(int(np.ceil(cfg["draws"] / 512)))
        ]
    )[: cfg["draws"]]
    dt = np.asarray(to_theta(dx, transform))
    validate_theta(dt, transform)
    np.savez(out / "joint_draws.npz", x=dx, theta=dt)
    marginal, joint = population_metrics(dt, tt, names, seed=cfg["draw_seed"])
    marginal.to_csv(out / "joint15_marginals.csv", index=False)
    joint.to_csv(out / "joint15_metrics.csv", index=False)
    from scripts.feniks_coherent_representation import plots as joint_plots

    history = pd.read_json(root / "physical/training.jsonl", lines=True)
    joint_plots(
        out,
        tt,
        tx,
        dt,
        dx,
        names,
        dict(
            physical=history,
            sfh_conditional=pd.read_json(
                source / "sfh_conditional/training.jsonl", lines=True
            ),
        ),
    )
    plots(
        out,
        comparisons,
        tt,
        names,
        history,
        trajectory,
        pd.read_csv(root / "audit/saved_draw_tail_bias.csv"),
        pd.read_csv(root / "audit/sfh_precision.csv"),
    )
    (out / "REPORT.md").write_text(
        "# Bounded coherent representation follow-up\n\n"
        "Read physical_before_after.png, continuation.png, tails_and_zeros.png, "
        "physical_joint.csv and ../audit/sfh_precision.csv. Corners and joint15 "
        "metrics use the new physical factor with frozen conditional SFH.\n\n"
        "Validation NLL chooses checkpoints. Fixed-seed validation draws monitor "
        "distribution drift. Test is evaluated only in this report, not used for "
        "optimization; it was already inspected in the preceding experiment. "
        "Validation/test disagreement is an empirical comparator, not a universal floor.\n\n"
        "W1 tail contributions partition the quantile integral exactly; central "
        "plots retain full-sample normalization. Float64 replay does not replace "
        "the stored targets. Persistent exact zeros do not establish physical atoms.\n\n"
        "Budget exhaustion does not certify convergence. This remains a truth-trained "
        "capacity oracle: no blind population recovery or posterior calibration. "
        "Review SFH tail/observable impact before freezing an independent reference.\n"
    )
    (out / "BLOCKED.json").unlink(missing_ok=True)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        digest,
        truth_used_for_training=True,
        conditional_frozen=True,
        dataset_modified=False,
        ready_for_production=False,
        posterior_trained=False,
        zero_replay_pass=read(root / "audit/FINAL.json")["replay_pass"],
        physical_sw_before=rows[0]["physical_sw"],
        physical_sw_after=rows[2]["physical_sw"],
        next_action="review_physical_convergence_and_SFH_target_before_blind_population_run",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "fit", "audit", "report"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a.source.resolve(), a.root.resolve(), a.config.resolve())
    else:
        {"fit": fit, "audit": audit, "report": report}[a.mode](a.root.resolve())


if __name__ == "__main__":
    main()
