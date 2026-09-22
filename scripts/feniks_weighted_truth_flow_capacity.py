"""Fit two independent 15D flows directly to the weighted true parent."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.feniks_avi_experiments import read, sha, write


def _status(path, expected):
    payload = read(path)
    if payload.get("status") != expected:
        raise ValueError(f"incomplete input {path}: {payload.get('status')}")
    return payload


def _normalized_weights(values):
    values = np.asarray(values, np.float64)
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or values.sum() <= 0
    ):
        raise ValueError("population weights must be finite, nonnegative and nonempty")
    return values / values.sum()


def _effective_size(weights):
    weights = _normalized_weights(weights)
    return float(1.0 / np.sum(weights**2))


def _split_indices(count, settings):
    rng = np.random.default_rng(settings["split_seed"])
    order = rng.permutation(count)
    train_end = int(settings["training_fraction"] * count)
    validation_end = train_end + int(settings["validation_fraction"] * count)
    train, validation, test = (
        order[:train_end],
        order[train_end:validation_end],
        order[validation_end:],
    )
    if min(map(len, (train, validation, test))) == 0:
        raise ValueError("weighted truth split contains an empty partition")
    return train, validation, test


def _weighted_resample(rng, indices, weights, count):
    indices = np.asarray(indices, np.int64)
    probabilities = _normalized_weights(np.asarray(weights)[indices])
    return rng.choice(indices, size=count, replace=True, p=probabilities)


def prepare(parent, root, config):
    from scripts.feniks_forward_population import contract as parent_contract

    if root.exists():
        raise FileExistsError(root)
    parent_manifest, _ = parent_contract(parent)
    _status(parent / "population/FINAL.json", "FORWARD_PARENT_COMPLETE")
    truth_path = Path(parent_manifest["blind_truth_parent"])
    settings = yaml.safe_load(config.read_text())
    if settings["replicas"] != 2:
        raise ValueError("the declared comparison requires exactly two replicas")
    truth = pd.read_parquet(
        truth_path, columns=["parent_row_index", "population_weight"]
    )
    weights = _normalized_weights(truth.population_weight)
    train, validation, test = _split_indices(len(truth), settings)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "report").mkdir()
    for replica in range(settings["replicas"]):
        (root / f"replica_{replica}").mkdir()
    np.savez(root / "split.npz", train=train, validation=validation, test=test)
    shutil.copy2(config, root / "capacity.yaml")
    inputs = {
        "parent_manifest": parent / "MANIFEST.json",
        "truth_parent": truth_path,
        "config": root / "capacity.yaml",
        "split": root / "split.npz",
    }
    write(
        root / "MANIFEST.json",
        dict(
            parent=str(parent.resolve()),
            source=parent_manifest["source"],
            settings=settings,
            inputs={
                name: {"path": str(path.resolve()), "sha256": sha(path)}
                for name, path in inputs.items()
            },
            target="population_weight-normalized true 15D parent",
            target_weight_sum=float(weights.sum()),
            target_weight_ess=_effective_size(weights),
            q_samples_used=False,
            classifier_used=False,
            selection_used=False,
            truth_used_for_training=True,
            truth_role="diagnostic flow-capacity target only",
            production_prior_modified=False,
        ),
    )
    print(
        json.dumps(
            dict(
                objects=len(truth),
                weight_ess=_effective_size(weights),
                replicas=settings["replicas"],
                epochs=settings["flow"]["epochs"],
            ),
            indent=2,
        )
    )


def contract(root):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"capacity input changed: {path}")
    return manifest, manifest["settings"]


def _runtime(manifest, destination):
    from euclid_dsps.amortized.forward_population_runtime import load_forward_runtime

    parent_manifest = read(Path(manifest["parent"]) / "MANIFEST.json")
    return load_forward_runtime(
        Path(manifest["source"]), destination, parent_manifest["settings"]
    )


def fit_replica(root, replica):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.avi_experiments import log_prob
    from euclid_dsps.amortized.forward_population import PHYSICAL
    from euclid_dsps.amortized.forward_population_runtime import posterior_template
    from euclid_dsps.amortized.latent import theta_to_x, x_to_theta
    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from scripts.feniks_forward_population import supervised_fit
    from scripts.report_feniks_forward_population import population_metrics

    manifest, settings = contract(root)
    if replica < 0 or replica >= settings["replicas"]:
        raise ValueError(f"invalid replica {replica}")
    out = root / f"replica_{replica}"
    write(out / "PROGRESS.json", dict(stage="loading_weighted_truth", replica=replica))
    model, runtime, _, source_config = _runtime(manifest, out / "runtime")
    parent_settings = read(Path(manifest["parent"]) / "MANIFEST.json")["settings"]
    names = tuple(runtime.latent_spec.names)
    truth = pd.read_parquet(Path(manifest["inputs"]["truth_parent"]["path"]))
    theta = truth[list(names)].to_numpy(np.float64)
    weights = _normalized_weights(truth.population_weight)
    x = np.asarray(theta_to_x(jnp.asarray(theta), runtime.latent_spec))
    if not np.isfinite(x).all():
        raise ValueError("true parent leaves the invertible latent support")
    with np.load(manifest["inputs"]["split"]["path"]) as split:
        train_ids, validation_ids, test_ids = (
            split["train"],
            split["validation"],
            split["test"],
        )
    seed = settings["seed"] + replica
    rng = np.random.default_rng(seed)
    training_ids = _weighted_resample(
        rng, train_ids, weights, settings["training_draws"]
    )
    validation_draw_ids = _weighted_resample(
        rng, validation_ids, weights, settings["validation_draws"]
    )
    heldout_draw_ids = _weighted_resample(
        rng, test_ids, weights, settings["heldout_draws"]
    )
    evaluation_rng = np.random.default_rng(settings["evaluation_seed"])
    evaluation_ids = _weighted_resample(
        evaluation_rng, np.arange(len(theta)), weights, settings["evaluation_draws"]
    )
    np.savez(
        out / "sample_identities.npz",
        training=training_ids,
        validation=validation_draw_ids,
        evaluation=evaluation_ids,
        heldout=heldout_draw_ids,
    )
    candidate = posterior_template(model, runtime, source_config, parent_settings)
    context_dim = candidate.input_dim

    def constant(count):
        return np.zeros((count, context_dim), np.float32)

    def loss(net, features, target):
        return -log_prob(model, net, features, target[None])[0]

    fit_settings = {**settings["flow"], "seed": seed}
    net = supervised_fit(
        candidate,
        loss,
        constant(len(training_ids)),
        x[training_ids],
        (constant(len(validation_draw_ids)), x[validation_draw_ids]),
        fit_settings,
        out,
    )
    evaluate = eqx.filter_jit(lambda features, target: loss(net, features, target))
    heldout_nll = np.concatenate(
        [
            np.asarray(
                evaluate(jnp.asarray(constant(len(chunk))), jnp.asarray(x[chunk]))
            )
            for chunk in np.array_split(
                heldout_draw_ids,
                max(1, int(np.ceil(len(heldout_draw_ids) / 512))),
            )
        ]
    )

    @eqx.filter_jit
    def sample(key):
        return sample_independent_mixture(
            model, net, key, jnp.zeros((1, context_dim)), 1024
        ).x[:, 0]

    flow_x = np.concatenate(
        [
            np.asarray(sample(jax.random.PRNGKey(seed + 10000 + index)))
            for index in range(int(np.ceil(settings["evaluation_draws"] / 1024)))
        ]
    )[: settings["evaluation_draws"]]
    target_x = x[evaluation_ids]
    target_theta = theta[evaluation_ids]
    flow_theta = np.asarray(x_to_theta(jnp.asarray(flow_x), runtime.latent_spec))
    np.savez(
        out / "density_draws.npz",
        target_x=target_x,
        target_theta=target_theta,
        flow_x=flow_x,
        flow_theta=flow_theta,
        names=np.asarray(names),
    )
    for space, predicted, target in (
        ("latent_x", flow_x, target_x),
        ("physical_theta", flow_theta, target_theta),
    ):
        marginal, joint = population_metrics(predicted, target, names, seed=seed)
        marginal.insert(0, "space", space)
        marginal.insert(0, "replica", replica)
        joint.insert(0, "space", space)
        joint.insert(0, "replica", replica)
        marginal.to_csv(out / f"{space}_marginal.csv", index=False)
        joint.to_csv(out / f"{space}_joint.csv", index=False)
    physical = [names.index(name) for name in PHYSICAL]
    finish = dict(
        status="WEIGHTED_TRUTH_FLOW_CAPACITY_COMPLETE",
        replica=replica,
        dimensions=len(names),
        target_weighting="population_weight",
        truth_role="diagnostic flow-capacity target only",
        unique_training_objects=len(np.unique(training_ids)),
        unique_validation_objects=len(np.unique(validation_draw_ids)),
        unique_heldout_objects=len(test_ids),
        heldout_nll=float(heldout_nll.mean()),
        heldout_nll_q99=float(np.quantile(heldout_nll, 0.99)),
        checkpoint_sha256=sha(out / "best.eqx"),
        target_redshift_mean=float(np.average(theta[:, physical[0]], weights=weights)),
        classifier_used=False,
    )
    write(out / "FINAL.json", finish)


def report(root):
    import matplotlib.pyplot as plt

    from euclid_dsps.amortized.forward_population import PHYSICAL
    from scripts.feniks_avi_overnight import _corner, _marginals
    from scripts.report_feniks_forward_population import population_metrics

    manifest, settings = contract(root)
    payloads = []
    finals = []
    for replica in range(settings["replicas"]):
        directory = root / f"replica_{replica}"
        finals.append(
            _status(directory / "FINAL.json", "WEIGHTED_TRUTH_FLOW_CAPACITY_COMPLETE")
        )
        with np.load(directory / "density_draws.npz") as saved:
            payloads.append({key: saved[key] for key in saved.files})
    names = tuple(payloads[0]["names"].tolist())
    for payload in payloads[1:]:
        if tuple(payload["names"].tolist()) != names:
            raise ValueError("replica latent names disagree")
        np.testing.assert_array_equal(payload["target_x"], payloads[0]["target_x"])
        np.testing.assert_array_equal(
            payload["target_theta"], payloads[0]["target_theta"]
        )
    labels = {
        f"Flow {index + 1}": payload["flow_theta"]
        for index, payload in enumerate(payloads)
    }
    latent_labels = {
        f"Flow {index + 1}": payload["flow_x"] for index, payload in enumerate(payloads)
    }
    _marginals(
        root / "report/weighted_truth_theta_15d.png",
        labels,
        payloads[0]["target_theta"],
        names,
    )
    _marginals(
        root / "report/weighted_truth_latent_x_15d.png",
        latent_labels,
        payloads[0]["target_x"],
        tuple(f"x({name})" for name in names),
    )
    physical = [names.index(name) for name in PHYSICAL]
    _corner(
        root / "report/weighted_truth_theta_physical_corner.png",
        {label: values[:, physical] for label, values in labels.items()},
        payloads[0]["target_theta"][:, physical],
        PHYSICAL,
    )
    _corner(
        root / "report/weighted_truth_latent_x_physical_corner.png",
        {label: values[:, physical] for label, values in latent_labels.items()},
        payloads[0]["target_x"][:, physical],
        tuple(f"x({name})" for name in PHYSICAL),
    )
    _corner(
        root / "report/weighted_truth_theta_target_corner.png",
        {},
        payloads[0]["target_theta"][:, physical],
        PHYSICAL,
    )
    _corner(
        root / "report/weighted_truth_latent_x_target_corner.png",
        {},
        payloads[0]["target_x"][:, physical],
        tuple(f"x({name})" for name in PHYSICAL),
    )
    metrics = []
    for replica in range(settings["replicas"]):
        for space in ("latent_x", "physical_theta"):
            metrics.append(
                pd.read_csv(root / f"replica_{replica}/{space}_marginal.csv")
            )
            metrics.append(pd.read_csv(root / f"replica_{replica}/{space}_joint.csv"))
    marginal = pd.concat(metrics[::2], ignore_index=True)
    joint = pd.concat(metrics[1::2], ignore_index=True)
    marginal.to_csv(root / "report/marginal_metrics.csv", index=False)
    joint.to_csv(root / "report/joint_metrics.csv", index=False)
    replica_marginal, replica_joint = population_metrics(
        payloads[0]["flow_theta"],
        payloads[1]["flow_theta"],
        names,
        seed=settings["seed"],
    )
    replica_marginal.to_csv(root / "report/replica_stability_marginal.csv", index=False)
    replica_joint.to_csv(root / "report/replica_stability_joint.csv", index=False)
    summary_panels = (
        (
            "A. Weighted parent truth in physical theta",
            root / "report/weighted_truth_theta_target_corner.png",
        ),
        (
            "B. Same truth after x = T(theta)",
            root / "report/weighted_truth_latent_x_target_corner.png",
        ),
        (
            "C. Two learned flows in normalized x",
            root / "report/weighted_truth_latent_x_physical_corner.png",
        ),
        (
            "D. Flow samples mapped back to physical theta",
            root / "report/weighted_truth_theta_physical_corner.png",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(24, 25), squeeze=False)
    for axis, (title, path) in zip(axes.flat, summary_panels, strict=True):
        axis.imshow(plt.imread(path))
        axis.set_title(title, fontsize=16)
        axis.axis("off")
    physical_joint = joint[
        (joint.space == "physical_theta") & (joint.group == "physical")
    ].sort_values("replica")
    latent_joint = joint[
        (joint.space == "latent_x") & (joint.group == "physical")
    ].sort_values("replica")
    stability = replica_joint.loc[
        replica_joint.group == "physical", "sliced_wasserstein"
    ].iloc[0]
    physical_text = " | ".join(
        f"Flow {int(row.replica) + 1}: physical joint SW={row.sliced_wasserstein:.4f}"
        for row in physical_joint.itertuples()
    )
    latent_text = " | ".join(
        f"Flow {int(row.replica) + 1}: latent joint SW={row.sliced_wasserstein:.4f}"
        for row in latent_joint.itertuples()
    )
    metric_text = (
        f"{physical_text}\n{latent_text} | flow-to-flow physical SW={stability:.4f}"
    )
    fig.suptitle(
        "Can the production 15D flow learn the exact weighted parent?",
        fontsize=22,
        y=0.995,
    )
    fig.text(0.5, 0.008, metric_text, ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.025, 1, 0.98))
    fig.savefig(root / "report/weighted_truth_flow_capacity_summary.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for replica in range(settings["replicas"]):
        rows = [
            json.loads(line)
            for line in (root / f"replica_{replica}/training.jsonl")
            .read_text()
            .splitlines()
        ]
        axes[0].plot(
            [row["epoch"] for row in rows],
            [row["train_nll"] for row in rows],
            label=f"Flow {replica + 1}",
        )
        axes[1].plot(
            [row["epoch"] for row in rows],
            [row["validation_nll"] for row in rows],
            label=f"Flow {replica + 1}",
        )
    axes[0].set(
        title="Weighted truth flow training", xlabel="Epoch", ylabel="Train NLL"
    )
    axes[1].set(title="Fixed validation", xlabel="Epoch", ylabel="Validation NLL")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(root / "report/training_curves.png", dpi=160)
    plt.close(fig)
    write(
        root / "report/FINAL.json",
        dict(
            status="WEIGHTED_TRUTH_FLOW_CAPACITY_REPORT_COMPLETE",
            replicas=finals,
            max_physical_sliced_wasserstein=float(
                physical_joint.sliced_wasserstein.max()
            ),
            target_weighting="population_weight",
            normalized_latent_plotted=True,
            physical_theta_plotted=True,
            summary_figure="weighted_truth_flow_capacity_summary.png",
            classifier_used=False,
            population_basis_used=False,
            production_prior_modified=False,
        ),
    )
    (root / "report/REPORT.md").write_text(
        "# Weighted true-parent flow capacity\n\n"
        "Two independently initialized 15D flows were trained directly on the "
        "population-weighted true parent in normalized latent coordinates. "
        "Inspect training_curves.png, weighted_truth_latent_x_15d.png, "
        "weighted_truth_theta_15d.png and "
        "weighted_truth_flow_capacity_summary.png.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/feniks_weighted_truth_flow_capacity.yaml"),
    )
    parser.add_argument("--replica", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        prepare(args.parent, args.root, args.config)
    elif args.stage == "fit":
        fit_replica(args.root, args.replica)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
