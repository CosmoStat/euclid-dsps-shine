"""Continue the best photometric ratio classifiers to measured convergence."""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_ratio_followup import (
    ARMS,
    _classifier_metrics,
    _context,
    _copy_resume,
    _evaluate_population,
    apply_logit_offsets,
    contract,
    fit_marginal_logit_offsets,
)
from scripts.feniks_weighted_truth_flow_capacity import _status


def prepare(source: Path, root: Path, config: Path):
    _status(source / "report/FINAL.json", "RATIO_FOLLOWUP_REPORT_COMPLETE")
    source_manifest = read(source / "MANIFEST.json")
    ratio_root = Path(source_manifest["source"])
    settings = copy.deepcopy(source_manifest["settings"])
    convergence = yaml.safe_load(config.read_text())
    for key in (
        "epoch_block",
        "minimum_epochs",
        "maximum_epochs",
        "required_plateau_windows",
    ):
        if not isinstance(convergence[key], int) or convergence[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    start_epoch = settings["classifier"]["epochs"]
    if convergence["minimum_epochs"] <= start_epoch:
        raise ValueError("minimum_epochs must exceed the source epoch")
    if convergence["maximum_epochs"] < convergence["minimum_epochs"]:
        raise ValueError("maximum_epochs must be at least minimum_epochs")
    if settings["classifier"]["lr_decay_every"] % convergence["epoch_block"]:
        raise ValueError("epoch blocks must align with the learning-rate schedule")
    settings["metric_draws"] = convergence["metric_draws"]
    settings["metric_seed"] = convergence["metric_seed"]
    settings["convergence"] = convergence

    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "report").mkdir()
    shutil.copy2(config, root / "ratio_convergence.yaml")
    shutil.copy2(source / "splits.npz", root / "splits.npz")
    for arm in ARMS:
        destination = root / arm
        destination.mkdir()
        source_cell = source / f"{arm}_seed1"
        _status(source_cell / "FINAL.json", "RATIO_FOLLOWUP_CELL_COMPLETE")
        _copy_resume(source_cell, destination)

    inputs = {
        "source_manifest": ratio_root / "MANIFEST.json",
        "source_config": ratio_root / "ratio_ladder.yaml",
        "source_report": ratio_root / "report/FINAL.json",
        "source_physical_classifier": ratio_root / "physical_a/best.eqx",
        "followup_manifest": source / "MANIFEST.json",
        "followup_report": source / "report/FINAL.json",
        "source_noiseless_checkpoint": source
        / "noiseless_photometry_seed1/best.eqx",
        "source_noisy_checkpoint": source / "noisy_photometry_seed1/best.eqx",
        "config": root / "ratio_convergence.yaml",
        "splits": root / "splits.npz",
    }
    write(
        root / "MANIFEST.json",
        dict(
            source=str(ratio_root.resolve()),
            source_followup=str(source.resolve()),
            settings=settings,
            source_settings=source_manifest["source_settings"],
            inputs={
                key: dict(path=str(path.resolve()), sha256=sha(path))
                for key, path in inputs.items()
            },
            arms=list(ARMS),
            resumed_replica=1,
            shared_target_split=True,
            simulation_banks_reused=True,
            population_uses_q=False,
            production_prior_modified=False,
        ),
    )
    print(yaml.safe_dump(convergence, sort_keys=False))
    print("Two adaptive continuation cells; no forward simulation or posterior fit.")


def _history(path: Path):
    frame = pd.read_json(path, lines=True)
    return frame.sort_values("epoch").drop_duplicates("epoch", keep="last")


def _plateau_diagnostics(history: pd.DataFrame, settings: dict):
    window = settings["epoch_block"]
    required = settings["required_plateau_windows"]
    needed = (required + 1) * window
    if len(history) < needed:
        return [], False
    values = history.validation_nll.to_numpy()
    minima = [
        float(np.min(values[-(offset + 1) * window : -offset * window or None]))
        for offset in range(required, -1, -1)
    ]
    improvements = [
        float(before - after)
        for before, after in zip(minima[:-1], minima[1:], strict=True)
    ]
    plateau = all(
        improvement < settings["minimum_recent_nll_improvement"]
        for improvement in improvements
    )
    return improvements, bool(plateau)


def _write_trajectory(path: Path, rows: list[dict]):
    pd.DataFrame(rows).sort_values("epoch").drop_duplicates(
        "epoch", keep="last"
    ).to_csv(path, index=False)


def _load_trajectory(path: Path):
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def run(root: Path, task: int):
    import jax
    import jax.numpy as jnp

    from scripts.feniks_forward_population import (
        classifier_template,
        classify,
        supervised_fit,
    )
    from scripts.feniks_ratio_ladder import _runtime

    if task not in range(len(ARMS)):
        raise ValueError("ratio-convergence task must be 0 or 1")
    arm = ARMS[task]
    (
        manifest,
        settings,
        source_manifest,
        source_settings,
        reference,
        target,
        basis,
        splits,
        frequencies,
        alpha,
        eligible,
        u_true,
        v_true,
    ) = _context(root)
    convergence = settings["convergence"]
    out = root / arm
    source_final = read(
        Path(manifest["source_followup"]) / f"{arm}_seed1/FINAL.json"
    )
    classifier_seed = source_final["classifier_seed"]
    classifier_settings = {**settings["classifier"], "seed": classifier_seed}
    feature_key = "noiseless_features" if arm.startswith("noiseless") else "features"
    features = reference[feature_key]
    target_features = target[feature_key]
    labels = reference["component"]
    candidate = classifier_template(
        features.shape[1], basis.components, classifier_settings
    )

    def loss(net, values, targets):
        prediction = jax.nn.log_softmax(jax.vmap(net)(values), axis=-1)
        return -jnp.take_along_axis(prediction, targets[:, None], axis=1)[:, 0]

    runtime = _runtime(source_manifest, out)[1]
    weak_parent_mass = read(Path(source_manifest["parent"]) / "MANIFEST.json")[
        "settings"
    ]["weak_parent_mass_cap"]
    trajectory_path = out / "trajectory.csv"
    trajectory = _load_trajectory(trajectory_path)

    while True:
        resume = read(out / "RESUME.json")
        epoch = int(resume["epoch"])
        existing = {int(row["epoch"]) for row in trajectory}
        if epoch not in existing:
            current_settings = {**classifier_settings, "epochs": epoch}
            # With start == epochs, supervised_fit only restores the global best.
            classifier = supervised_fit(
                candidate,
                loss,
                features[splits["train"]],
                labels[splits["train"]],
                (features[splits["validation"]], labels[splits["validation"]]),
                current_settings,
                out,
            )
            logc_calibration = classify(
                classifier, features[splits["calibration"]]
            )
            offset, calibration = fit_marginal_logit_offsets(
                logc_calibration,
                frequencies,
                **settings["calibration"],
            )
            logc_audit = apply_logit_offsets(
                classify(classifier, features[splits["audit"]]), offset
            )
            logc_fit = apply_logit_offsets(
                classify(classifier, target_features[splits["target_fit"]]), offset
            )
            logc_heldout = apply_logit_offsets(
                classify(classifier, target_features[splits["target_heldout"]]),
                offset,
            )
            stage = out / f"epoch_{epoch:03d}"
            stage.mkdir(exist_ok=True)
            pd.DataFrame(
                dict(component=np.arange(len(offset)), logit_offset=offset)
            ).to_csv(stage / "calibration_offsets.csv", index=False)
            population = _evaluate_population(
                out=stage,
                name="calibrated",
                logc_fit=logc_fit,
                logc_heldout=logc_heldout,
                frequencies=frequencies,
                alpha=alpha,
                eligible=eligible,
                u_true=u_true,
                v_true=v_true,
                basis=basis,
                runtime=runtime,
                settings=settings,
                weak_parent_mass=weak_parent_mass,
            )
            classifier_metrics = _classifier_metrics(
                logc_audit, labels[splits["audit"]], frequencies
            )
            history = _history(out / "training.jsonl")
            improvements, nll_plateau = _plateau_diagnostics(
                history, convergence
            )
            previous_sw = (
                float(trajectory[-1]["parent_physical_sliced_wasserstein"])
                if trajectory
                else np.nan
            )
            parent_sw_change = (
                abs(
                    population["parent_physical_sliced_wasserstein"]
                    - previous_sw
                )
                if np.isfinite(previous_sw)
                else np.nan
            )
            ratio_pass = bool(
                classifier_metrics["ratio_moment_median_abs_error"]
                <= convergence["ratio_moment_median_abs_error"]
            )
            parent_stable = bool(
                np.isfinite(parent_sw_change)
                and parent_sw_change <= convergence["maximum_parent_sw_change"]
            )
            converged = bool(
                epoch >= convergence["minimum_epochs"]
                and nll_plateau
                and parent_stable
                and ratio_pass
            )
            row = dict(
                arm=arm,
                epoch=epoch,
                best_epoch=int(
                    history.loc[history.validation_nll.idxmin(), "epoch"]
                ),
                best_nll=float(history.validation_nll.min()),
                last_nll=float(history.validation_nll.iloc[-1]),
                recent_nll_improvements=";".join(
                    f"{value:.12g}" for value in improvements
                ),
                nll_plateau=nll_plateau,
                parent_sw_change=parent_sw_change,
                parent_stable=parent_stable,
                ratio_pass=ratio_pass,
                converged=converged,
                classifier_sha256=sha(out / "best.eqx"),
                calibration_gap=calibration[
                    "calibration_ratio_moment_max_abs_error"
                ],
                **classifier_metrics,
                **population,
            )
            trajectory.append(row)
            _write_trajectory(trajectory_path, trajectory)
            write(out / "PROGRESS.json", row)
            print(row, flush=True)
        else:
            row = next(row for row in trajectory if int(row["epoch"]) == epoch)
            converged = bool(row["converged"])

        if converged or epoch >= convergence["maximum_epochs"]:
            final = dict(
                status="RATIO_CONVERGENCE_CELL_COMPLETE",
                arm=arm,
                classifier_seed=classifier_seed,
                converged=bool(converged),
                maximum_epoch_reached=bool(
                    epoch >= convergence["maximum_epochs"] and not converged
                ),
                final_epoch=epoch,
                final=row,
                classifier_sha256=sha(out / "best.eqx"),
                shared_target_split=True,
                simulation_banks_reused=True,
                population_uses_q=False,
                production_prior_modified=False,
            )
            write(out / "FINAL.json", final)
            return

        next_epoch = min(
            epoch + convergence["epoch_block"], convergence["maximum_epochs"]
        )
        training_settings = {**classifier_settings, "epochs": next_epoch}
        candidate = supervised_fit(
            candidate,
            loss,
            features[splits["train"]],
            labels[splits["train"]],
            (features[splits["validation"]], labels[splits["validation"]]),
            training_settings,
            out,
        )


def report(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest, settings = contract(root)
    rows = []
    finals = {}
    for arm in ARMS:
        final = read(root / arm / "FINAL.json")
        if final.get("status") != "RATIO_CONVERGENCE_CELL_COMPLETE":
            raise ValueError(f"incomplete convergence arm {arm}")
        finals[arm] = final
        rows.append(pd.read_csv(root / arm / "trajectory.csv"))
    trajectory = pd.concat(rows, ignore_index=True)
    trajectory.to_csv(root / "report/trajectory.csv", index=False)

    source_report = read(Path(manifest["source_followup"]) / "report/FINAL.json")
    physical_oracle = float(source_report["physical_oracle_parent_sw"])
    source_contracts = settings["contracts"]
    final_rows = trajectory.sort_values("epoch").groupby("arm").tail(1)
    decisions = {
        "classifiers_converged": bool(
            all(finals[arm]["converged"] for arm in ARMS)
        ),
        "physical_parent_closure": bool(
            (
                final_rows.parent_physical_sliced_wasserstein
                <= physical_oracle
                + source_contracts["maximum_parent_sw_above_physical_oracle"]
            ).all()
        ),
        "nondegenerate_simplex_solution": bool(
            (
                final_rows.zero_parent_weight_fraction
                <= source_contracts["maximum_zero_weight_fraction"]
            ).all()
        ),
        "independent_ratio_normalization": bool(final_rows.ratio_pass.all()),
    }
    if not decisions["classifiers_converged"]:
        next_action = "classifier_not_converged_by_maximum_epoch"
    elif not (
        decisions["physical_parent_closure"]
        and decisions["nondegenerate_simplex_solution"]
    ):
        next_action = "regularize_or_reduce_population_weight_degrees_of_freedom"
    else:
        next_action = "repair_decoder_contract_then_run_one_end_to_end_parent_fit"

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    for arm, group in trajectory.groupby("arm"):
        group = group.sort_values("epoch")
        axes[0, 0].plot(group.epoch, group.best_nll, marker="o", label=arm)
        axes[0, 1].plot(
            group.epoch,
            group.parent_physical_sliced_wasserstein,
            marker="o",
            label=arm,
        )
        axes[1, 0].plot(
            group.epoch,
            group.ratio_moment_median_abs_error,
            marker="o",
            label=arm,
        )
        axes[1, 1].plot(
            group.epoch,
            group.zero_parent_weight_fraction,
            marker="o",
            label=arm,
        )
    axes[0, 0].set(title="Best validation NLL", xlabel="epoch", ylabel="NLL")
    axes[0, 1].axhline(physical_oracle, color="black", linestyle="--")
    axes[0, 1].set(
        title="Calibrated parent closure", xlabel="epoch", ylabel="physical SW"
    )
    axes[1, 0].axhline(
        settings["convergence"]["ratio_moment_median_abs_error"],
        color="black",
        linestyle="--",
    )
    axes[1, 0].set(
        title="Independent ratio normalization",
        xlabel="epoch",
        ylabel="median abs error",
    )
    axes[1, 1].axhline(
        source_contracts["maximum_zero_weight_fraction"],
        color="black",
        linestyle="--",
    )
    axes[1, 1].set(
        title="Simplex boundary", xlabel="epoch", ylabel="zero-weight fraction"
    )
    for axis in axes.flat:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "report/convergence_trajectory.png", dpi=170)
    plt.close(figure)

    (root / "report/REPORT.md").write_text(
        "# Adaptive photometric-ratio convergence\n\n"
        "Two independent seed-1 classifiers were resumed on the immutable "
        "banks and common splits. Every trajectory point includes a fresh "
        "reference-only calibration and parent solve.\n\n"
        f"Decision: `{next_action}`.\n"
    )
    write(
        root / "report/FINAL.json",
        dict(
            status="RATIO_CONVERGENCE_REPORT_COMPLETE",
            decisions=decisions,
            next_action=next_action,
            physical_oracle_parent_sw=physical_oracle,
            final_epochs={arm: finals[arm]["final_epoch"] for arm in ARMS},
            simulation_banks_reused=True,
            population_uses_q=False,
            production_prior_modified=False,
            production_ready=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source is None or args.config is None:
            parser.error("prepare requires --source and --config")
        prepare(args.source.resolve(), args.root.resolve(), args.config.resolve())
    elif args.mode == "run":
        run(args.root.resolve(), args.task)
    else:
        report(args.root.resolve())


if __name__ == "__main__":
    main()
