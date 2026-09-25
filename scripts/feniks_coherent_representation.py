"""Support-corrected structured parent-density oracle on the coherent dataset.

This pipeline uses training truth. It must never be advertised as observed-only
population recovery or be substituted silently for a blind reference prior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.coherent_coordinates import (
    fit_coordinates,
    log_abs_det_dtheta_dx,
    to_theta,
    to_x,
    validate_theta,
)
from scripts.audit_feniks_coherent_parent import load_spec, verify_catalogues
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import SPLITS, complete, finish, settings

FACTORS = ("physical", "sfh_conditional")


def prepare(source, old_spec, root, config):
    if root.exists() or source == root or source in root.parents:
        raise ValueError("Use a new root outside the immutable coherent dataset")
    cfg = yaml.safe_load(config.read_text())
    if cfg["flow"]["fixed_validation"] is not True:
        raise ValueError("A fixed validation set is required")
    for value in (cfg["draws"], cfg["projection_objects"], *cfg["resources"].values()):
        if not isinstance(value, int) or value < 1:
            raise ValueError("Resource and sample counts must be positive integers")
    train, counts, _ = verify_catalogues(source)
    old = load_spec(old_spec)
    names = list(old.names)
    transform = fit_coordinates(
        train[names].to_numpy(), names, np.asarray(old.lower), np.asarray(old.upper)
    )
    # Fail closed on held-out support; never extend/refit the transform using it.
    arrays, rows, files = {}, [], {}
    for split in SPLITS:
        path = source / "dataset/parent" / f"{split}.parquet"
        before = sha(path)
        frame = train if split == "train" else pd.read_parquet(path)
        theta = frame[names].to_numpy(np.float64)
        validate_theta(theta, transform)
        x = np.asarray(to_x(theta, transform))
        restored = np.asarray(to_theta(x, transform))
        scale = np.maximum(np.subtract(*np.percentile(theta, [75, 25], axis=0)), 1e-8)
        errors = np.max(abs(restored - theta), axis=0) / scale
        if (
            not np.isfinite(x).all()
            or not np.isfinite(errors).all()
            or errors.max() > cfg["roundtrip_tolerance"]
        ):
            raise ValueError(f"Coordinate roundtrip failed: {split}")
        arrays[split] = dict(x=x, theta=theta)
        rows.extend(
            dict(
                split=split,
                parameter=name,
                max_roundtrip_over_iqr=float(errors[k]),
                zero_fraction=float(np.mean(theta[:, k] == 0)),
            )
            for k, name in enumerate(names)
        )
        if sha(path) != before:
            raise ValueError(f"Dataset changed while preparing: {path}")
        files[str(path)] = before
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "cache").mkdir()
    (root / "experiment.yaml").write_text(config.read_text())
    write(root / "coordinates.json", transform)
    write(
        root / "projection.json",
        dict(
            n_sfh_bins=read(source / "decoder.json")["model"]["n_sfh_bins"],
            source_decoder_sha256=sha(source / "decoder.json"),
        ),
    )
    for split, data in arrays.items():
        np.savez(root / "cache" / f"{split}.npz", **data)
    # Projection replay is train-only and small; native columns retained intact.
    train.sample(
        n=min(len(train), cfg["projection_objects"]), random_state=cfg["seed"]
    ).to_parquet(root / "cache/projection_train.parquet", index=False)
    pd.DataFrame(rows).to_csv(root / "transform_checks.csv", index=False)
    frozen = [
        root / "experiment.yaml",
        root / "coordinates.json",
        root / "projection.json",
        root / "transform_checks.csv",
        *(root / "cache").iterdir(),
    ]
    write(
        root / "MANIFEST.json",
        dict(
            settings=cfg,
            source=str(source),
            source_manifest_sha256=sha(source / "MANIFEST.json"),
            source_files=files,
            splits=counts,
            old_spec_path=str(old_spec),
            old_spec_sha256=sha(old_spec),
            frozen_files={str(p.relative_to(root)): sha(p) for p in frozen},
            truth_role="direct density capacity oracle; NOT blind population recovery",
            dataset_modified=False,
            resimulated_photometry=False,
            population_uses_q=False,
        ),
    )
    print(
        "Coordinate checks PASS on all splits; no rows clipped, rejected or dequantized."
    )
    print(
        "120 epochs maximum per factor by default; completed epochs resume on timeout."
    )
    print("Two parallel H100 tasks: p(physical 5D), p(SFH 10D | physical 5D).")
    print("Truth-trained oracle only; no production prior or posterior is changed.")


def factor_data(data, task):
    return (
        (np.zeros((len(data), 1)), data[:, :5])
        if task == 0
        else (data[:, :5], data[:, 5:])
    )


def fit(root, task):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import (
        factor_log_prob,
        factor_template,
        transport_audit,
    )
    from scripts.feniks_forward_population import supervised_fit

    if task not in (0, 1):
        raise ValueError("factor task must be 0 or 1")
    _, cfg, digest = settings(root)
    out = root / FACTORS[task]
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    with np.load(root / "cache/train.npz") as f:
        features, targets = factor_data(f["x"], task)
    with np.load(root / "cache/validation.npz") as f:
        validation = factor_data(f["x"], task)
    training = dict(cfg["flow"], seed=cfg["seed"] + task)
    model = factor_template(
        training, targets.shape[1], features.shape[1], training["seed"]
    )
    model = supervised_fit(
        model,
        lambda net, c, t: -factor_log_prob(net, c, t),
        features,
        targets,
        validation,
        training,
        out,
    )
    audit = transport_audit(
        model,
        jnp.asarray(validation[0][:8]),
        jax.random.PRNGKey(730 + task),
        values=jnp.asarray(validation[1][:8]),
    )
    write(out / "transport.json", audit)
    if not audit["passed"]:
        raise RuntimeError(
            "Trained flow inverse/Jacobian contract failed; inspect transport.json"
        )
    history = pd.read_json(out / "training.jsonl", lines=True)
    finish(
        out,
        [out / "best.eqx", out / "training.jsonl", out / "transport.json"],
        digest,
        epoch=int(history.epoch.iloc[-1]),
        best_nll=float(history.best_nll.iloc[-1]),
        recent_best_nll_gain=float(
            history.best_nll.iloc[max(0, len(history) - 21)] - history.best_nll.iloc[-1]
        ),
        factor=FACTORS[task],
        truth_used_for_training=True,
        training_budget_exhausted=True,
    )


def projection_zero_table(original, projected, knots):
    from euclid_dsps.prior_learning.spline15d_schema import SFH_CONTRAST_NAMES

    rows = []
    for k, name in enumerate(SFH_CONTRAST_NAMES):
        left, right = (
            knots[f"spline_log_sfr_{k:02d}"],
            knots[f"spline_log_sfr_{k + 1:02d}"],
        )
        zero = projected[name].to_numpy() == 0
        rows.append(
            dict(
                parameter=name,
                rows=len(original),
                replay_zero_count=int(zero.sum()),
                equal_knots_count=int(np.sum(left == right)),
                equal_floor_knots_count=int(
                    np.sum(zero & (left <= -29.99) & (right <= -29.99))
                ),
                equal_nonfloor_knots_count=int(np.sum(zero & (left > -29.99))),
                max_replay_error=float(
                    np.max(abs(original[name].to_numpy() - projected[name].to_numpy()))
                ),
            )
        )
    return pd.DataFrame(rows)


def zeros(root):
    from euclid_dsps.prior_learning.spline15d import project_diffsky_frame_to_spline15d

    _, _, digest = settings(root)
    out = root / "sfh_zeros"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    original = pd.read_parquet(root / "cache/projection_train.parquet")
    projected, knots = project_diffsky_frame_to_spline15d(
        original,
        n_sfh_bins=read(root / "projection.json")["n_sfh_bins"],
        batch_size=256,
    )
    table = projection_zero_table(original, projected, knots)
    table.to_csv(out / "projection_zeros.csv", index=False)
    finish(
        out,
        [out / "projection_zeros.csv"],
        digest,
        replay_pass=bool(table.max_replay_error.max() <= 1e-5),
        conclusion="equal projected knots are not proof of physical atoms; float32 projection and SFH floors remain possible",
        dequantization_applied=False,
    )


def plots(out, truth, truth_x, draws, draw_x, names, history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for label, indices in (("physical", range(5)), ("sfh", range(5, 15))):
        fig, axes = plt.subplots(
            2, len(indices), figsize=(3 * len(indices), 6), squeeze=False
        )
        for col, k in enumerate(indices):
            for row, (a, b, space) in enumerate(
                ((truth, draws, "physical theta"), (truth_x, draw_x, "normalized x"))
            ):
                edges = np.histogram_bin_edges(np.r_[a[:, k], b[:, k]], bins=70)
                axes[row, col].hist(
                    a[:, k],
                    bins=edges,
                    density=True,
                    histtype="step",
                    label="held-out parent truth",
                    color="#222222",
                )
                axes[row, col].hist(
                    b[:, k],
                    bins=edges,
                    density=True,
                    histtype="step",
                    label="structured flow",
                    color="#b13d65",
                )
                axes[row, col].set_title(names[k], fontsize=8)
                axes[row, col].set_xlabel(space)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Capacity oracle only; unit parent weights; no selection")
        fig.tight_layout()
        fig.savefig(out / f"representation_{label}.png", dpi=130)
        plt.close(fig)
    for values, reference, space in ((draws, truth, "theta"), (draw_x, truth_x, "x")):
        fig, axes = plt.subplots(5, 5, figsize=(11, 11))
        for i in range(5):
            for j in range(5):
                ax = axes[i, j]
                if i == j:
                    edges = np.histogram_bin_edges(
                        np.r_[reference[:, i], values[:, i]], bins=60
                    )
                    ax.hist(
                        reference[:, i],
                        bins=edges,
                        density=True,
                        histtype="step",
                        color="#222222",
                    )
                    ax.hist(
                        values[:, i],
                        bins=edges,
                        density=True,
                        histtype="step",
                        color="#b13d65",
                    )
                elif i > j:
                    ax.scatter(
                        reference[::8, j],
                        reference[::8, i],
                        s=1,
                        alpha=0.12,
                        color="#222222",
                    )
                    ax.scatter(
                        values[::8, j], values[::8, i], s=1, alpha=0.12, color="#b13d65"
                    )
                else:
                    ax.axis("off")
                if i == 4:
                    ax.set_xlabel(names[j], fontsize=7)
                if j == 0:
                    ax.set_ylabel(names[i], fontsize=7)
        fig.suptitle(f"Physical joint in {space}: black truth, pink flow")
        fig.tight_layout()
        fig.savefig(out / f"physical_corner_{space}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, FACTORS, strict=True):
        h = history[name]
        for field in ("train_nll", "validation_nll", "best_nll"):
            ax.plot(h.epoch, h[field], label=field)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training.png", dpi=130)
    plt.close(fig)


def report(root):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import (
        StructuredPopulation,
        factor_template,
    )
    from scripts.report_feniks_forward_population import population_metrics

    _, cfg, digest = settings(root)
    out = root / "report"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    missing = [s for s in (*FACTORS, "sfh_zeros") if not complete(root / s, digest)]
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, ready_for_production=False))
        raise RuntimeError(f"Incomplete stages: {missing}. Resume only missing work.")
    transform = read(root / "coordinates.json")
    models = [
        eqx.tree_deserialise_leaves(
            root / s / "best.eqx",
            factor_template(
                cfg["flow"], 5 if i == 0 else 10, 1 if i == 0 else 5, cfg["seed"] + i
            ),
        )
        for i, s in enumerate(FACTORS)
    ]
    model = StructuredPopulation(*models, tuple(transform["names"]))
    sample = eqx.filter_jit(lambda m, k: m.sample(k, 512))
    draw_x = np.concatenate(
        [
            np.asarray(sample(model, jax.random.PRNGKey(cfg["seed"] + 1000 + i)))
            for i in range(int(np.ceil(cfg["draws"] / 512)))
        ]
    )[: cfg["draws"]]
    draws = np.asarray(to_theta(draw_x, transform))
    validate_theta(draws, transform)
    if not np.isfinite(draw_x).all():
        raise FloatingPointError("Nonfinite flow draws")
    with np.load(root / "cache/test.npz") as f:
        truth, truth_x = f["theta"], f["x"]
    log_prob = eqx.filter_jit(lambda m, x: m.log_prob(x))
    nll_x = np.concatenate(
        [
            np.asarray(-log_prob(model, jnp.asarray(chunk)))
            for chunk in np.array_split(
                truth_x, max(1, int(np.ceil(len(truth_x) / 512)))
            )
        ]
    )
    nll_theta = nll_x + np.asarray(log_abs_det_dtheta_dx(truth_x, transform))
    if not np.isfinite(nll_theta).all():
        raise FloatingPointError("Nonfinite held-out physical density")
    np.savez(out / "draws.npz", theta=draws, x=draw_x)
    marginals, joints = [], []
    for space, pred, target in (("theta", draws, truth), ("x", draw_x, truth_x)):
        marginal, joint = population_metrics(
            pred, target, transform["names"], seed=cfg["seed"]
        )
        marginals.append(marginal.assign(space=space))
        joints.append(joint.assign(space=space, comparison="flow_vs_heldout"))
        # Empirical reference split disagreement, NOT a formal universal error floor.
        with np.load(root / "cache/validation.npz") as f:
            _, reference = population_metrics(
                f["theta" if space == "theta" else "x"],
                target,
                transform["names"],
                seed=cfg["seed"],
            )
        joints.append(reference.assign(space=space, comparison="validation_vs_test"))
    pd.concat(marginals).to_csv(out / "marginals.csv", index=False)
    joint_table = pd.concat(joints)
    joint_table.to_csv(out / "joint.csv", index=False)
    # Show the dependence actually learned, not only separate 1D histograms.
    tc = pd.DataFrame(truth, columns=transform["names"]).corr(method="spearman")
    pc = pd.DataFrame(draws, columns=transform["names"]).corr(method="spearman")
    tc.to_csv(out / "truth_spearman.csv")
    pc.to_csv(out / "flow_spearman.csv")
    delta = abs((tc - pc).to_numpy()[:5, 5:])
    sfh = []
    for k in range(5, 15):
        scale = max(np.subtract(*np.percentile(truth[:, k], [75, 25])), 1e-8)
        sfh.append(
            dict(
                parameter=transform["names"][k],
                truth_exact_zero=float(np.mean(truth[:, k] == 0)),
                flow_exact_zero=float(np.mean(draws[:, k] == 0)),
                truth_within_001_iqr=float(np.mean(abs(truth[:, k]) < 0.01 * scale)),
                flow_within_001_iqr=float(np.mean(abs(draws[:, k]) < 0.01 * scale)),
            )
        )
    pd.DataFrame(sfh).to_csv(out / "sfh_zero_neighborhoods.csv", index=False)
    history = {
        s: pd.read_json(root / s / "training.jsonl", lines=True) for s in FACTORS
    }
    plots(out, truth, truth_x, draws, draw_x, transform["names"], history)
    cross_error = float(np.nanmax(delta)) if np.isfinite(delta).any() else None
    (out / "REPORT.md").write_text(
        "# Coherent representation oracle\n\n"
        "Read training.png, representation_physical.png, physical_corner_theta.png, "
        "physical_corner_x.png, representation_sfh.png and joint.csv. Compare "
        "flow-vs-heldout with validation-vs-test (not a universal acceptance threshold).\n\n"
        "Coordinates and both density factors used parent training truth. This is NOT "
        "photometric population recovery. All 15 coordinates are stochastic. "
        "SFH fitting cannot change the physical marginal. No clipping or jitter.\n\n"
        "A continuous density cannot reproduce genuine atoms exactly. Inspect "
        "sfh_zero_neighborhoods.csv and ../sfh_zeros/projection_zeros.csv. Equal "
        "nonfloor knots do not distinguish physical plateaus from float32 resolution. "
        "A small NLL is not sufficient evidence of a correct distribution.\n\n"
        "Finite training budget is not a convergence certificate. No production "
        "promotion is made, and no old banks or checkpoints have been reused.\n"
    )
    (out / "BLOCKED.json").unlink(missing_ok=True)
    finish(
        out,
        [p for p in out.iterdir() if p.is_file() and p.name != "FINAL.json"],
        digest,
        truth_used_for_training=True,
        ready_for_production=False,
        population_recovered=False,
        posterior_trained=False,
        dimensions=15,
        dequantization_applied=False,
        heldout_nll_x=float(nll_x.mean()),
        heldout_nll_theta=float(nll_theta.mean()),
        cross_spearman_max_abs_error=cross_error,
        max_physical_sw=float(
            joint_table.loc[
                (joint_table.space == "theta")
                & (joint_table.group == "physical")
                & (joint_table.comparison == "flow_vs_heldout"),
                "sliced_wasserstein",
            ].iloc[0]
        ),
        zero_replay_pass=read(root / "sfh_zeros/FINAL.json")["replay_pass"],
        next_action="review_representation_and_SFH_target_before_freezing_independent_reference",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "fit", "zeros", "report"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--old-spec", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--task", type=int, default=0)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(
            a.source.resolve(),
            a.old_spec.resolve(),
            a.root.resolve(),
            a.config.resolve(),
        )
    elif a.mode == "fit":
        fit(a.root.resolve(), a.task)
    else:
        {"zeros": zeros, "report": report}[a.mode](a.root.resolve())


if __name__ == "__main__":
    main()
