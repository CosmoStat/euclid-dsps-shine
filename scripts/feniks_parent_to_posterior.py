"""Fresh supervised 15D NPE under an immutable, explicitly exploratory parent.

j ~ u, theta ~ g_j, flux ~ simulator + saved noise; select on observed flux.
Conditioning on that flux makes selection redundant for p(theta | flux).
Thus selected simulation pairs have UNIT training weights, never 1/beta(theta).
No posterior sample is a training target or an input to population learning.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts import feniks_coherent_inference as ci
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_coherent_parent import complete, finish, photometer

settings = ci.settings
NAMES = ci.NAMES


def validate_config(cfg):
    b, p, e, s = (cfg[k] for k in ("bank", "posterior", "evaluation", "stopping"))
    for section in (
        cfg["resources"],
        {k: v for k, v in b.items() if k != "role_probabilities"},
    ):
        if any(type(v) is not int or v < 1 for v in section.values()):
            raise ValueError("Positive integer resources and bank sizes required")
    roles = np.asarray(b["role_probabilities"], float)
    if roles.shape != (3,) or np.any(roles <= 0) or not np.isclose(roles.sum(), 1):
        raise ValueError(
            "Three positive train/validation/evaluation probabilities required"
        )
    if not p["fixed_validation"] or p.get("initial_checkpoint"):
        raise ValueError("Train from scratch with fixed validation; no old checkpoint")
    for v in (
        cfg["seed"],
        p["experts"],
        p["epochs"],
        p["batch_size"],
        p["validation_limit"],
        e["objects"],
        e["draws_per_object"],
        s["block_epochs"],
        s["minimum_epochs"],
        s["patience_blocks"],
    ):
        if type(v) is not int or v < 1:
            raise ValueError("Positive integer training/evaluation sizes required")
    if p["experts"] < 2 or e["draws_per_object"] < 16:
        raise ValueError("At least two experts and 16 posterior draws required")
    if any(not 0 < e[k] < 1 for k in ("maximum_coverage_error", "maximum_pit_ks")):
        raise ValueError("Invalid diagnostic thresholds")


def initialize(source, root, config, exploratory=False):
    if not exploratory:
        raise ValueError(
            "Require --exploratory: this is not parent validation or production"
        )
    if root.exists() or source in root.parents or root in source.parents:
        raise ValueError("Use a new root outside the immutable source")
    m, original, contract = ci.settings(source)
    for stage in ("reference", "population", "qualification", "report"):
        if not complete(source / stage, contract):
            raise ValueError(f"Completed source {stage} required")
    cfg = yaml.safe_load(config.read_text())
    validate_config(cfg)
    if cfg["seed"] == original["seed"]:
        raise ValueError("Use a fresh simulation seed")
    parent = read(source / "population/parent.json")
    u, v, alpha = (np.asarray(parent[k], float) for k in ("u", "v", "alpha"))
    basis, _, _ = ci.require_reference(source, contract)
    k = basis["conditional"].shape[1]
    if any(a.shape != (k,) or not np.isfinite(a).all() for a in (u, v, alpha)):
        raise ValueError("Parent dimensions/values disagree with frozen basis")
    if (
        np.any(u < 0)
        or np.any(v < 0)
        or np.any(alpha <= 0)
        or np.any(alpha > 1)
        or not np.isclose(u.sum(), 1)
        or not np.isclose(v.sum(), 1)
    ):
        raise ValueError("Invalid normalized parent/selected weights or efficiencies")
    np.testing.assert_allclose(v, u * alpha / (u @ alpha), rtol=1e-8, atol=1e-10)
    if m.get("population_uses_q") or m.get("population_uses_target_truth"):
        raise ValueError("Source must be a blind parent, not q-fed or truth-fitted")
    decision = read(source / "report/DECISION.json")
    dataset = Path(m["source"])
    dataset_manifest = read(dataset / "MANIFEST.json")
    if m["selection"] != dataset_manifest["settings"]["selection"]:
        raise ValueError("Parent and dataset selection contracts differ")
    # Pin only assets used here, not the multi-GB reference simulation banks.
    paths = [
        source / "MANIFEST.json",
        source / "population/FINAL.json",
        source / "population/parent.json",
        source / "report/DECISION.json",
        dataset / "MANIFEST.json",
        dataset / "decoder.json",
        dataset / "noise.json",
        dataset / "dataset/selected_r29/test.parquet",
    ]
    source_files = {str(p): sha(p) for p in paths}
    source_files.update(dataset_manifest["assets"])
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "reference").mkdir()
    shutil.copyfile(config, root / "experiment.yaml")
    shutil.copyfile(source / "population/parent.json", root / "frozen_parent.json")
    shutil.copyfile(source / "report/DECISION.json", root / "PARENT_DECISION.json")
    reference_files = []
    for name in ("basis.npz", "coordinates.json", "feature_stats.json"):
        target = root / "reference" / name
        shutil.copyfile(source / "reference" / name, target)
        reference_files.append(target)
    frozen = [
        root / "experiment.yaml",
        root / "frozen_parent.json",
        root / "PARENT_DECISION.json",
        *reference_files,
    ]
    write(
        root / "MANIFEST.json",
        dict(
            source=str(dataset),
            parent_run=str(source),
            source_contract=contract,
            settings=cfg,
            selection=m["selection"],
            source_files=source_files,
            frozen_files={str(p.relative_to(root)): sha(p) for p in frozen},
            exploratory=True,
            inherited_parent_decision=decision,
            population_uses_q=False,
            population_uses_target_truth=False,
            posterior_target="fresh simulator theta",
            production_promotion=False,
        ),
    )
    digest = sha(root / "MANIFEST.json")
    finish(root / "reference", reference_files, digest, imported=True, dimensions=15)
    print(
        yaml.safe_dump(
            dict(
                mode="EXPLORATORY; all source failures retained; diagnostic jobs untouched",
                parent="frozen blind u and existing 15D native joint-kernel components",
                components=k,
                nuisance="all 10 SFH coordinates sampled with joint anchors and kernel noise",
                selection="saved noisy r<29; no inverse-selection weights in q",
                simulations=cfg["bank"]["shards"] * cfg["bank"]["rows_per_shard"],
                training="fresh selected pairs, unit-weight supervised forward-KL; no RWS",
                classifier="none trained or evaluated here; parent is frozen",
                posterior=cfg["posterior"],
                evaluation=cfg["evaluation"],
                resources=cfg["resources"],
                outputs="fresh banks, flow, transport audit, two-cohort 15D draws/calibration/PIT, selected aggregate closure, roadmap",
            )
        )
    )


def bank(root, task):
    from euclid_dsps.synthetic_diffsky.coherent_parent import observe

    m, cfg, digest = settings(root)
    b = cfg["bank"]
    if not 0 <= task < b["shards"]:
        raise ValueError("Invalid bank task")
    out = root / "banks" / f"shard_{task:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if complete(out, digest):
        return
    basis, spec, stats = ci.require_reference(root, digest)
    u = np.asarray(read(root / "frozen_parent.json")["u"])
    dataset = Path(m["source"])
    decoder, noise = read(dataset / "decoder.json"), read(dataset / "noise.json")
    predict = photometer(decoder, b["decoder_batch_size"])
    bands = [v["name"] for v in decoder["bands"]]
    receipts, counts = [], np.zeros(3, dtype=int)
    for start in range(0, b["rows_per_shard"], b["checkpoint_rows"]):
        block = out / f"block_{start:07d}"
        block.mkdir(exist_ok=True)
        if not complete(block, digest):
            n = min(b["checkpoint_rows"], b["rows_per_shard"] - start)
            # Disjoint deterministic RNG substreams for labels, kernels, noise, roles.
            seeds = np.random.SeedSequence(
                [cfg["seed"], task, start, 891]
            ).generate_state(4)
            labels = np.random.default_rng(seeds[0]).choice(len(u), n, p=u)
            x = ci.sample_basis(basis, labels, int(seeds[1]))
            theta = np.asarray(ci.to_theta(x, spec))
            ci.validate_theta(theta, spec)
            roles = np.random.default_rng(seeds[3]).choice(
                3, n, p=b["role_probabilities"]
            )
            frame = observe(
                pd.DataFrame(index=np.arange(n)),
                predict(theta),
                decoder["bands"],
                noise,
                seed=int(seeds[2]),
                selection=m["selection"],
            )
            selected = frame.selected_r29.to_numpy(dtype=bool)
            f = ci.features(frame, bands, stats)
            if not np.isfinite(f).all() or not np.isfinite(x).all():
                raise FloatingPointError("Nonfinite fresh-bank features/targets")
            np.savez(
                block / "bank.npz",
                x=x,
                theta=theta,
                features=f,
                component=labels,
                role=roles,
                selected=selected,
                row_id=task * b["rows_per_shard"] + start + np.arange(n),
            )
            finish(
                block,
                [block / "bank.npz"],
                digest,
                rows=n,
                selected_by_role=np.bincount(roles[selected], minlength=3).tolist(),
                parent_sha256=sha(root / "frozen_parent.json"),
                seeds=seeds.tolist(),
            )
        counts += read(block / "FINAL.json")["selected_by_role"]
        receipts.append(block / "FINAL.json")
        write(
            out / "PROGRESS.json",
            dict(
                done=min(start + b["checkpoint_rows"], b["rows_per_shard"]),
                total=b["rows_per_shard"],
                selected_by_role=counts.tolist(),
            ),
        )
    finish(
        out,
        receipts,
        digest,
        rows=b["rows_per_shard"],
        selected_by_role=counts.tolist(),
    )


def selected_rows(root, cfg, digest, role):
    """Load only one selected role; evaluation truth never enters training."""
    chunks = {name: [] for name in ("features", "x", "theta", "row_id")}
    for task in range(cfg["bank"]["shards"]):
        shard = root / "banks" / f"shard_{task:03d}"
        if not complete(shard, digest):
            raise ValueError(f"Missing bank {task}")
        # Follow the receipt inventory, not stray or partially written blocks.
        for path in read(shard / "FINAL.json")["artifacts"]:
            block = (shard / path).parent
            if not complete(block, digest):
                raise ValueError(f"Missing block {block}")
            with np.load(block / "bank.npz") as f:
                mask = f["selected"] & (f["role"] == role)
                for name in chunks:
                    chunks[name].append(f[name][mask])
    data = {name: np.concatenate(values) for name, values in chunks.items()}
    if not len(data["x"]) or len(np.unique(data["row_id"])) != len(data["x"]):
        raise ValueError("Empty or duplicated independent selected rows")
    return data


def template(cfg, input_dim):
    from euclid_dsps.amortized.structured_population import factor_template

    return factor_template(cfg["posterior"], 15, input_dim, cfg["seed"] + 10)


def train(root):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.structured_population import (
        factor_log_prob,
        transport_audit,
    )

    _, cfg, digest = settings(root)
    out = root / "posterior"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    train_rows, validation = (selected_rows(root, cfg, digest, r) for r in (0, 1))
    for name, data in (("training", train_rows), ("validation", validation)):
        if len(data["x"]) < cfg["bank"][f"minimum_{name}_rows"]:
            raise ValueError(f"Insufficient independent selected {name} rows")
    if np.intersect1d(train_rows["row_id"], validation["row_id"]).size:
        raise ValueError("Training/validation leakage")
    model = ci.bounded_fit(
        template(cfg, train_rows["features"].shape[1]),
        lambda net, f, theta_x: -factor_log_prob(net, f, theta_x),
        (train_rows["features"], train_rows["x"]),
        (validation["features"], validation["x"]),
        dict(cfg["posterior"], seed=cfg["seed"] + 10),
        cfg["stopping"],
        out,
    )
    transport = transport_audit(
        model,
        jnp.asarray(validation["features"][:8]),
        jax.random.PRNGKey(cfg["seed"] + 11),
        values=jnp.asarray(validation["x"][:8]),
    )
    write(out / "transport.json", transport)
    if not transport["passed"] or any(
        not np.isfinite(v) for row in transport["experts"] for v in row.values()
    ):
        raise FloatingPointError("Trained flow inverse/Jacobian audit failed")
    write(
        out / "training_measure.json",
        dict(
            dimensions=15,
            targets="fresh simulator theta, never q draws",
            weights="unit",
            training_rows=len(train_rows["x"]),
            validation_rows=len(validation["x"]),
            effective_training_rows=len(train_rows["x"]),
            evaluation_role_excluded=2,
            parent_sha256=sha(root / "frozen_parent.json"),
            population_updated=False,
            target_catalogue_truth_used=False,
            initial_checkpoint=None,
        ),
    )
    finish(
        out,
        [
            out / n
            for n in (
                "best.eqx",
                "STOP.json",
                "transport.json",
                "training_measure.json",
            )
        ],
        digest,
        dimensions=15,
        scientific_promotion=False,
    )


def evaluate(root):
    import equinox as eqx

    _, cfg, digest = settings(root)
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    if not complete(root / "posterior", digest):
        raise ValueError("Posterior incomplete")
    _, spec, stats = ci.require_reference(root, digest)
    data = selected_rows(root, cfg, digest, 2)
    n = cfg["evaluation"]["objects"]
    if len(data["x"]) < n or len(ci.observed(root, "test")[0]) < n:
        raise ValueError("Insufficient unique held-out rows for requested evaluation")
    model = eqx.tree_deserialise_leaves(
        root / "posterior/best.eqx", template(cfg, data["features"].shape[1])
    )
    ids = np.random.default_rng(cfg["seed"] + 12).choice(
        len(data["x"]), n, replace=False
    )
    # Separate checkpoints let evaluation resume without repeating the other cohort.
    for label in ("in_model", "coherent_target"):
        cohort = out / label
        cohort.mkdir(exist_ok=True)
        if complete(cohort, digest):
            continue
        if label == "in_model":
            ci.evaluate_posterior(
                root,
                model,
                cohort,
                dict(cfg, seed=cfg["seed"] + 20),
                spec,
                stats,
                bank_evaluation=(
                    data["features"][ids],
                    data["theta"][ids],
                    data["row_id"][ids],
                ),
            )
            for name in ("calibration.csv", "draws.npz"):
                (cohort / f"in_model_{name}").replace(cohort / name)
        else:
            ci.evaluate_posterior(
                root, model, cohort, dict(cfg, seed=cfg["seed"] + 30), spec, stats
            )
        finish(
            cohort,
            [cohort / "calibration.csv", cohort / "draws.npz"],
            digest,
            objects=n,
            draws_per_object=cfg["evaluation"]["draws_per_object"],
            checkpoint_sha256=sha(root / "posterior/best.eqx"),
            selected_population=True,
        )
    finish(
        out,
        [out / name / "FINAL.json" for name in ("in_model", "coherent_target")],
        digest,
    )


def report(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.feniks_avi_overnight import _corner, _marginals
    from scripts.report_feniks_forward_population import (
        diagnostic_plots,
        population_metrics,
    )

    m, cfg, digest = settings(root)
    out = root / "report"
    out.mkdir(exist_ok=True)
    if complete(out, digest):
        return
    missing = [s for s in ("posterior", "evaluation") if not complete(root / s, digest)]
    if missing:
        write(out / "BLOCKED.json", dict(missing=missing, scientific_promotion=False))
        return
    tables, joints, flags = [], [], {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    fig2, histaxes = plt.subplots(3, 5, figsize=(18, 10), layout="constrained")
    for label, color in (("in_model", "#167d91"), ("coherent_target", "#b95143")):
        cohort = root / "evaluation" / label
        if not complete(cohort, digest):
            raise ValueError(f"Missing cohort {label}")
        with np.load(cohort / "draws.npz") as f:
            draws, truth = f["draws"], f["truth"]
            if len(np.unique(f["evaluation_positions"])) != len(truth):
                raise ValueError("Duplicated evaluation galaxies")
        table = diagnostic_plots(out, draws, truth, NAMES, label)
        for index in range(min(2, len(truth))):
            _marginals(
                out / f"{label}_individual_{index:02d}_15d.png",
                {"posterior": draws[index]},
                truth[index],
                NAMES,
            )
            _corner(
                out / f"{label}_individual_{index:02d}_physical_corner.png",
                {"posterior": draws[index, :, :5]},
                truth[index, :5],
                NAMES[:5],
            )
        table["cohort"] = label
        table["coverage_flag"] = (
            (table.coverage_68 - 0.68).abs()
            <= cfg["evaluation"]["maximum_coverage_error"]
        ) & (
            (table.coverage_95 - 0.95).abs()
            <= cfg["evaluation"]["maximum_coverage_error"]
        )
        table["pit_flag"] = table.pit_ks <= cfg["evaluation"]["maximum_pit_ks"]
        tables.append(table)
        flags[label] = {}
        for group, take in (
            ("core_z_mass_metal", slice(0, 3)),
            ("dust", slice(3, 5)),
            ("sfh", slice(5, 15)),
        ):
            t = table.iloc[take]
            flags[label][group] = dict(
                coverage=bool(t.coverage_flag.all()), pit=bool(t.pit_flag.all())
            )
        for ax, column, target in zip(
            axes,
            ("coverage_68", "coverage_95", "pit_ks"),
            (0.68, 0.95, None),
            strict=True,
        ):
            ax.plot(np.arange(15), table[column], "o-", label=label, color=color)
            ax.set_title(column)
            if target is not None:
                ax.axhline(target, color="black", ls="--")
            ax.set_xticks(np.arange(15), NAMES, rotation=80, fontsize=7)
        # Dense joint draws, never posterior medians as a population estimator.
        flat = draws.reshape(-1, 15)
        ids = np.random.default_rng(cfg["seed"] + 40).choice(
            len(flat), min(32768, len(flat)), replace=False
        )
        marginal, joint = population_metrics(flat[ids], truth, NAMES, seed=cfg["seed"])
        marginal.to_csv(out / f"{label}_selected_aggregate_marginals.csv", index=False)
        joint["cohort"] = label
        joints.append(joint)
        for j, ax in enumerate(histaxes.flat):
            lo, hi = np.quantile(truth[:, j], [0.005, 0.995])
            if hi <= lo:
                lo, hi = lo - 0.5, hi + 0.5
            edges = np.linspace(lo, hi, 50)
            # Full probability denominators: plotting limits do not renormalize tails away.
            h = np.histogram(flat[:, j], edges)[0] / len(flat) / np.diff(edges)
            ht = np.histogram(truth[:, j], edges)[0] / len(truth) / np.diff(edges)
            ax.stairs(h, edges, color=color, label=f"{label}: q aggregate")
            ax.stairs(ht, edges, color=color, ls="--", label=f"{label}: selected truth")
            ax.set_title(NAMES[j], fontsize=9)
    axes[0].legend(fontsize=8)
    fig.savefig(out / "calibration_comparison.png", dpi=140)
    plt.close(fig)
    histaxes.flat[0].legend(fontsize=6)
    fig2.suptitle(
        "Selected aggregate closure, not parent recovery; view: truth 0.5-99.5%, metrics: full support"
    )
    fig2.savefig(out / "selected_aggregate_15d.png", dpi=140)
    plt.close(fig2)
    pd.concat(tables).to_csv(out / "calibration.csv", index=False)
    pd.concat(joints).to_csv(out / "selected_aggregate_joint.csv", index=False)
    history = pd.read_json(root / "posterior/training.jsonl", lines=True)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for column in ("train_nll", "validation_nll", "best_nll"):
        ax.plot(history.epoch, history[column], label=column)
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL in frozen latent coordinates")
    fig.savefig(out / "training.png", dpi=140)
    plt.close(fig)
    parent = m["inherited_parent_decision"]
    receipts = [
        read(root / "banks" / f"shard_{i:03d}/FINAL.json")
        for i in range(cfg["bank"]["shards"])
    ]
    counts = np.sum([r["selected_by_role"] for r in receipts], axis=0)
    generated = sum(r["rows"] for r in receipts)
    frozen = read(root / "frozen_parent.json")
    write(
        out / "bank_measure.json",
        dict(
            parent_draws=generated,
            selected_by_role=counts.tolist(),
            measured_alpha=float(counts.sum() / generated),
            frozen_alpha=float(np.dot(frozen["u"], frozen["alpha"])),
            training_measure=read(root / "posterior/training_measure.json"),
            evaluation_objects_per_cohort=cfg["evaluation"]["objects"],
        ),
    )
    decision = dict(
        exploratory=True,
        inherited_parent_decision=parent,
        calibration_flags=flags,
        stop=read(root / "posterior/STOP.json"),
        parent_updated=False,
        scientific_promotion=False,
        ready_for_production=False,
        target_test_used_for_development=True,
        next="inspect_in_model_vs_target_calibration_and_parallel_parent_recovery",
    )
    write(out / "DECISION.json", decision)
    text = (
        "# Fresh-parent posterior (exploratory)\n\n"
        "- Population weights/basis frozen; no q feedback. Dust and all SFH retained.\n"
        "- In-model: independent selected simulations from learned u. Tests the posterior under its training prior.\n"
        "- Coherent target: held-out catalogue under a different true parent. Tests transfer, not real-sky calibration.\n"
        "- Aggregate closure compares SELECTED distributions, never an aggregate to the unselected parent.\n"
        "- Unit-weight direct simulation targets; no RWS or inverse-beta posterior correction.\n"
        "- See calibration_comparison.png, training.png, both physical/sfh PIT figures, individual corners and selected_aggregate_15d.png.\n"
        "- Finite-cohort SW and calibration flags are diagnostic; no production approval.\n\n"
        f"Inherited parent failures: {[k for k, v in parent['gates'].items() if not v]}.\n"
        f"Strict capacity qualified: {parent.get('strict_qualification_passed', False)}.\n"
        f"Calibration flags: {flags}\n"
    )
    (out / "REPORT.md").write_text(text)
    (root / "ROADMAP_STATUS.md").write_text(text)
    finish(
        out,
        sorted(
            p
            for p in out.iterdir()
            if p.suffix in (".csv", ".json", ".md", ".png")
            and p.name not in ("FINAL.json", "BLOCKED.json", "FAILED.json")
        ),
        digest,
        scientific_promotion=False,
    )


def schedule(root):
    _, cfg, digest = settings(root)
    missing = [
        str(i)
        for i in range(cfg["bank"]["shards"])
        if not complete(root / "banks" / f"shard_{i:03d}", digest)
    ]
    print(f"MISSING_BANKS='{','.join(missing)}'")
    for stage in ("posterior", "evaluation", "report"):
        print(f"NEED_{stage.upper()}={int(not complete(root / stage, digest))}")
    for name, value in cfg["resources"].items():
        print(f"{name.upper()}={value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("init", "bank", "train", "evaluate", "report", "schedule")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "init":
        initialize(args.source.resolve(), root, args.config.resolve(), args.exploratory)
    else:
        stage = {"train": "posterior", "evaluate": "evaluation"}.get(
            args.mode, args.mode
        )
        out = (
            root / stage
            if args.mode != "bank"
            else root / "banks" / f"shard_{args.task:03d}"
        )
        try:
            if args.mode == "bank":
                bank(root, args.task)
            else:
                globals()[args.mode](root)
        except Exception as exc:
            if args.mode != "schedule" and root.is_dir():
                write(
                    out / "FAILED.json", dict(type=type(exc).__name__, message=str(exc))
                )
            raise


if __name__ == "__main__":
    main()
