"""Continue weighted-truth flows and isolate resampling from capacity failure."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_weighted_truth_flow_capacity import (
    _normalized_weights,
    _status,
    _weighted_resample,
)


def _task(settings, task):
    replicas = int(settings["replicas"])
    arms = tuple(settings["arms"])
    if task < 0 or task >= replicas * len(arms):
        raise ValueError(f"invalid follow-up task {task}")
    return arms[task // replicas], task % replicas


def _exact_repeated_indices(indices, cycles):
    indices = np.asarray(indices, np.int64)
    if indices.ndim != 1 or not len(indices) or int(cycles) <= 0:
        raise ValueError("exact weighted training requires identities and cycles")
    return np.tile(indices, int(cycles))


def _weighted_loss_scale(weights, indices):
    selected = _normalized_weights(np.asarray(weights, np.float64)[indices])
    return selected * len(selected)


def prepare(source_capacity, root, config):
    from scripts.feniks_weighted_truth_flow_capacity import contract as source_contract

    if root.exists():
        raise FileExistsError(root)
    source_manifest, _ = source_contract(source_capacity)
    _status(
        source_capacity / "report/FINAL.json",
        "WEIGHTED_TRUTH_FLOW_CAPACITY_REPORT_COMPLETE",
    )
    settings = yaml.safe_load(config.read_text())
    if settings["replicas"] != 2 or tuple(settings["arms"]) != (
        "resampled",
        "exact_weighted",
    ):
        raise ValueError("follow-up requires two replicas and the declared two arms")
    milestones = list(settings["milestones"])
    if milestones != sorted(set(milestones)) or min(milestones) <= 0:
        raise ValueError("milestones must be unique positive increasing epochs")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "report").mkdir()
    for arm in settings["arms"]:
        for replica in range(settings["replicas"]):
            (root / arm / f"replica_{replica}").mkdir(parents=True)
    shutil.copy2(config, root / "followup.yaml")
    inputs = {
        "source_manifest": source_capacity / "MANIFEST.json",
        "source_report": source_capacity / "report/FINAL.json",
        "source_split": source_capacity / "split.npz",
        "truth_parent": Path(source_manifest["inputs"]["truth_parent"]["path"]),
        "config": root / "followup.yaml",
    }
    for replica in range(settings["replicas"]):
        checkpoint = source_capacity / f"replica_{replica}/best.eqx"
        receipt = _status(
            source_capacity / f"replica_{replica}/FINAL.json",
            "WEIGHTED_TRUTH_FLOW_CAPACITY_COMPLETE",
        )
        if sha(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError(f"source best checkpoint changed for replica {replica}")
        inputs[f"source_best_{replica}"] = checkpoint
        inputs[f"source_draws_{replica}"] = (
            source_capacity / f"replica_{replica}/density_draws.npz"
        )
    write(
        root / "MANIFEST.json",
        dict(
            source_capacity=str(source_capacity.resolve()),
            parent=source_manifest["parent"],
            source=source_manifest["source"],
            settings=settings,
            inputs={
                name: {"path": str(path.resolve()), "sha256": sha(path)}
                for name, path in inputs.items()
            },
            comparison=(
                "same best checkpoints: weighted resample versus exact weighted "
                "empirical continuation"
            ),
            new_forward_simulations=0,
            production_prior_modified=False,
        ),
    )
    print(
        json.dumps(
            dict(
                tasks=settings["replicas"] * len(settings["arms"]),
                arms=settings["arms"],
                added_epochs=max(milestones),
                learning_rate=settings["flow"]["learning_rate"],
                new_forward_simulations=0,
            ),
            indent=2,
        )
    )


def contract(root):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"follow-up input changed: {path}")
    return manifest, manifest["settings"]


def _constant(count, width):
    return np.zeros((int(count), int(width)), np.float32)


def _sample_flow(model, candidate, context_dim, seed, count):
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture

    chunk = 1024

    @eqx.filter_jit
    def sample(net, key):
        return sample_independent_mixture(
            model, net, key, jnp.zeros((1, context_dim)), chunk
        ).x[:, 0]

    return np.concatenate(
        [
            np.asarray(sample(candidate, jax.random.PRNGKey(seed + index)))
            for index in range(int(np.ceil(count / chunk)))
        ]
    )[:count]


def _metric_rows(predicted, target, names, *, arm, replica, epoch, seed):
    from scripts.report_feniks_forward_population import population_metrics

    marginal, joint = population_metrics(predicted, target, names, seed=seed)
    for frame in (marginal, joint):
        frame.insert(0, "epoch", int(epoch))
        frame.insert(0, "replica", int(replica))
        frame.insert(0, "arm", arm)
    return marginal, joint


def fit(root, task):
    import equinox as eqx
    import jax.numpy as jnp

    from euclid_dsps.amortized.avi_experiments import log_prob
    from euclid_dsps.amortized.forward_population_runtime import posterior_template
    from euclid_dsps.amortized.latent import theta_to_x, x_to_theta
    from scripts.feniks_forward_population import supervised_fit
    from scripts.feniks_weighted_truth_flow_capacity import _runtime

    manifest, settings = contract(root)
    arm, replica = _task(settings, task)
    out = root / arm / f"replica_{replica}"
    write(out / "PROGRESS.json", dict(stage="loading", epoch=0))
    model, runtime, _, source_config = _runtime(manifest, out / "runtime")
    parent_settings = read(Path(manifest["parent"]) / "MANIFEST.json")["settings"]
    names = tuple(runtime.latent_spec.names)
    truth = pd.read_parquet(manifest["inputs"]["truth_parent"]["path"])
    theta = truth[list(names)].to_numpy(np.float64)
    weights = _normalized_weights(truth.population_weight)
    x = np.asarray(theta_to_x(jnp.asarray(theta), runtime.latent_spec))
    with np.load(manifest["inputs"]["source_split"]["path"]) as split:
        train_ids, validation_ids, test_ids = (
            split["train"],
            split["validation"],
            split["test"],
        )
    seed = int(settings["seed"]) + replica
    rng = np.random.default_rng(seed)
    if arm == "resampled":
        training_ids = _weighted_resample(
            rng, train_ids, weights, settings["training_draws"]
        )
        validation_draw_ids = _weighted_resample(
            rng, validation_ids, weights, settings["validation_draws"]
        )
        training_targets = x[training_ids]
        validation_targets = x[validation_draw_ids]

        def loss(net, features, target):
            return -log_prob(model, net, features, target[None])[0]

    else:
        training_ids = _exact_repeated_indices(train_ids, settings["exact_cycles"])
        validation_draw_ids = validation_ids
        training_targets = np.column_stack(
            (
                x[training_ids],
                np.tile(
                    _weighted_loss_scale(weights, train_ids), settings["exact_cycles"]
                ),
            )
        )
        validation_targets = np.column_stack(
            (x[validation_ids], _weighted_loss_scale(weights, validation_ids))
        )

        def loss(net, features, target):
            return (
                -log_prob(model, net, features, target[None, :, : len(names)])[0]
                * target[:, len(names)]
            )

    candidate = posterior_template(model, runtime, source_config, parent_settings)
    context_dim = candidate.input_dim
    features = _constant(len(training_targets), context_dim)
    validation = (
        _constant(len(validation_targets), context_dim),
        validation_targets,
    )
    with np.load(manifest["inputs"]["source_draws_0"]["path"]) as source_draws:
        target_x = source_draws["target_x"][: settings["metric_draws"]]
        target_theta = source_draws["target_theta"][: settings["metric_draws"]]
    marginal_rows, joint_rows = [], []
    for milestone in settings["milestones"]:
        flow_settings = {
            **settings["flow"],
            "epochs": int(milestone),
            "seed": seed,
            "initial_checkpoint": manifest["inputs"][f"source_best_{replica}"]["path"],
        }
        candidate = supervised_fit(
            candidate,
            loss,
            features,
            training_targets,
            validation,
            flow_settings,
            out,
        )
        flow_x = _sample_flow(
            model,
            candidate,
            context_dim,
            seed + 100000 + int(milestone) * 100,
            settings["metric_draws"],
        )
        flow_theta = np.asarray(x_to_theta(jnp.asarray(flow_x), runtime.latent_spec))
        for space, predicted, target in (
            ("latent_x", flow_x, target_x),
            ("physical_theta", flow_theta, target_theta),
        ):
            marginal, joint = _metric_rows(
                predicted,
                target,
                names,
                arm=arm,
                replica=replica,
                epoch=milestone,
                seed=settings["metric_seed"],
            )
            marginal.insert(3, "space", space)
            joint.insert(3, "space", space)
            marginal_rows.append(marginal)
            joint_rows.append(joint)
        write(
            out / "METRIC_PROGRESS.json",
            dict(
                epoch=milestone,
                physical_sliced_wasserstein=float(
                    joint_rows[-1]
                    .loc[joint_rows[-1].group == "physical", "sliced_wasserstein"]
                    .iloc[0]
                ),
            ),
        )
    marginal = pd.concat(marginal_rows, ignore_index=True)
    joint = pd.concat(joint_rows, ignore_index=True)
    marginal.to_csv(out / "trajectory_marginal.csv", index=False)
    joint.to_csv(out / "trajectory_joint.csv", index=False)
    np.savez_compressed(
        out / "final_draws.npz",
        target_x=target_x.astype(np.float32),
        target_theta=target_theta.astype(np.float32),
        flow_x=flow_x.astype(np.float32),
        flow_theta=flow_theta.astype(np.float32),
        names=np.asarray(names),
    )

    def point_nll(net, f, target):
        return -log_prob(model, net, f, target[None])[0]

    evaluate = eqx.filter_jit(point_nll)
    heldout = []
    for ids in np.array_split(test_ids, max(1, int(np.ceil(len(test_ids) / 512)))):
        heldout.append(
            np.asarray(
                evaluate(
                    candidate,
                    jnp.asarray(_constant(len(ids), context_dim)),
                    jnp.asarray(x[ids]),
                )
            )
        )
    heldout = np.concatenate(heldout)
    test_weights = _normalized_weights(weights[test_ids])
    final_physical = joint[
        (joint.epoch == max(settings["milestones"]))
        & (joint.space == "physical_theta")
        & (joint.group == "physical")
    ].iloc[0]
    write(
        out / "FINAL.json",
        dict(
            status="WEIGHTED_TRUTH_FLOW_FOLLOWUP_COMPLETE",
            arm=arm,
            replica=replica,
            added_epochs=max(settings["milestones"]),
            training_rows=len(training_targets),
            unique_training_objects=len(np.unique(training_ids)),
            heldout_weighted_nll=float(np.sum(test_weights * heldout)),
            physical_sliced_wasserstein=float(final_physical.sliced_wasserstein),
            checkpoint_sha256=sha(out / "best.eqx"),
            new_forward_simulations=0,
        ),
    )


def report(root):
    import matplotlib.pyplot as plt

    from scripts.feniks_avi_overnight import _marginals

    manifest, settings = contract(root)
    (root / "report").mkdir(exist_ok=True)
    source = Path(manifest["source_capacity"])
    marginal_frames, joint_frames = [], []
    finals = []
    target = None
    final_series = {}
    for replica in range(settings["replicas"]):
        with np.load(source / f"replica_{replica}/density_draws.npz") as saved:
            names = tuple(saved["names"].tolist())
            count = settings["metric_draws"]
            source_target_x = saved["target_x"][:count]
            source_target_theta = saved["target_theta"][:count]
            source_flow_x = saved["flow_x"][:count]
            source_flow_theta = saved["flow_theta"][:count]
        for space, predicted, reference in (
            ("latent_x", source_flow_x, source_target_x),
            ("physical_theta", source_flow_theta, source_target_theta),
        ):
            marginal, joint = _metric_rows(
                predicted,
                reference,
                names,
                arm="initial",
                replica=replica,
                epoch=0,
                seed=settings["metric_seed"],
            )
            marginal.insert(3, "space", space)
            joint.insert(3, "space", space)
            marginal_frames.append(marginal)
            joint_frames.append(joint)
        for arm in settings["arms"]:
            directory = root / arm / f"replica_{replica}"
            finals.append(
                _status(
                    directory / "FINAL.json", "WEIGHTED_TRUTH_FLOW_FOLLOWUP_COMPLETE"
                )
            )
            marginal_frames.append(pd.read_csv(directory / "trajectory_marginal.csv"))
            joint_frames.append(pd.read_csv(directory / "trajectory_joint.csv"))
            with np.load(directory / "final_draws.npz") as saved:
                if target is None:
                    target = saved["target_theta"]
                else:
                    np.testing.assert_array_equal(target, saved["target_theta"])
                final_series[f"{arm} F{replica + 1}"] = saved["flow_theta"]
    marginal = pd.concat(marginal_frames, ignore_index=True)
    joint = pd.concat(joint_frames, ignore_index=True)
    marginal.to_csv(root / "report/marginal_trajectory.csv", index=False)
    joint.to_csv(root / "report/joint_trajectory.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), squeeze=False)
    for column, group in enumerate(("physical", "sfh")):
        axis = axes[0, column]
        for (arm, replica), frame in joint[
            (joint.space == "physical_theta") & (joint.group == group)
        ].groupby(["arm", "replica"]):
            axis.plot(
                frame.epoch,
                frame.sliced_wasserstein,
                marker="o",
                label=f"{arm} F{int(replica) + 1}",
            )
        axis.axhline(
            settings["decision"]["target_physical_sliced_wasserstein"],
            color="black",
            ls="--",
            lw=1,
        )
        axis.set(
            title=f"{group.upper()} joint density",
            xlabel="Additional epochs",
            ylabel="Sliced Wasserstein / truth IQR",
        )
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / "report/followup_trajectory.png", dpi=180)
    plt.close(fig)
    _marginals(
        root / "report/followup_final_theta_15d.png", final_series, target, names
    )
    physical = joint[(joint.space == "physical_theta") & (joint.group == "physical")]
    baseline = float(physical[physical.arm == "initial"].sliced_wasserstein.median())
    final_epoch = max(settings["milestones"])
    final_by_arm = (
        physical[physical.epoch == final_epoch]
        .groupby("arm")
        .sliced_wasserstein.median()
        .to_dict()
    )
    previous_epoch = settings["milestones"][-2]
    previous_by_arm = (
        physical[physical.epoch == previous_epoch]
        .groupby("arm")
        .sliced_wasserstein.median()
        .to_dict()
    )
    best_arm = min(final_by_arm, key=final_by_arm.get)
    target_gate = settings["decision"]["target_physical_sliced_wasserstein"]
    exact_advantage = final_by_arm["resampled"] - final_by_arm["exact_weighted"]
    continuing = previous_by_arm[best_arm] - final_by_arm[best_arm]
    if final_by_arm[best_arm] <= target_gate:
        conclusion = "target_recovered_with_additional_optimization"
    elif exact_advantage >= settings["decision"]["material_improvement"]:
        conclusion = "weighted_resampling_is_a_material_failure_mode"
    elif continuing >= settings["decision"]["continuing_improvement"]:
        conclusion = "optimization_not_yet_plateaued"
    else:
        conclusion = "current_flow_family_or_objective_remains_inadequate"
    write(
        root / "report/FINAL.json",
        dict(
            status="WEIGHTED_TRUTH_FLOW_FOLLOWUP_REPORT_COMPLETE",
            baseline_physical_sliced_wasserstein=baseline,
            final_physical_sliced_wasserstein=final_by_arm,
            exact_weighted_advantage=exact_advantage,
            last_milestone_improvement=continuing,
            best_arm=best_arm,
            conclusion=conclusion,
            tasks=finals,
            new_forward_simulations=0,
            production_prior_modified=False,
        ),
    )
    (root / "report/REPORT.md").write_text(
        "# Weighted-truth flow follow-up\n\n"
        "This controlled continuation starts both arms from the same immutable "
        "best checkpoints. It separates additional low-rate optimization from "
        "the weighted-resampling approximation. No forward simulation or "
        "production prior update is performed.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-capacity", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/feniks_weighted_truth_flow_followup.yaml"),
    )
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.source_capacity is None:
            parser.error("prepare requires --source-capacity")
        prepare(args.source_capacity, args.root, args.config)
    elif args.stage == "fit":
        fit(args.root, args.task)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
