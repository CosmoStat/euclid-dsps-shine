"""Audit and regularize the frozen 128-component population inversion."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

from euclid_dsps.amortized.forward_population import (
    fit_selected_weights,
    parent_from_selected,
)
from euclid_dsps.amortized.population_regularization import fit_selected_weights_kl
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_ratio_followup import (
    ARMS,
    _context,
    apply_logit_offsets,
    fit_marginal_logit_offsets,
)
from scripts.feniks_weighted_truth_flow_capacity import _status


def contract(root: Path):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"population-inversion input changed: {path}")
    return manifest, manifest["settings"]


def prepare(source: Path, root: Path, config: Path):
    _status(source / "report/FINAL.json", "RATIO_CONVERGENCE_REPORT_COMPLETE")
    source_manifest = read(source / "MANIFEST.json")
    settings = yaml.safe_load(config.read_text())
    strengths = np.asarray(settings["strengths"], float)
    if (
        strengths.ndim != 1
        or len(strengths) < 2
        or strengths[0] != 0
        or np.any(np.diff(strengths) <= 0)
    ):
        raise ValueError("strengths must start at zero and increase strictly")
    for key in ("bootstraps", "metric_draws", "checkpoint_tail"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "report").mkdir()
    for arm in ARMS:
        (root / arm).mkdir()
    shutil.copy2(config, root / "population_inversion_audit.yaml")

    inputs = {
        "source_manifest": source / "MANIFEST.json",
        "source_report": source / "report/FINAL.json",
        "source_splits": source / "splits.npz",
        "config": root / "population_inversion_audit.yaml",
    }
    for arm in ARMS:
        inputs[f"{arm}_classifier"] = source / arm / "best.eqx"
        inputs[f"{arm}_trajectory"] = source / arm / "trajectory.csv"
        trajectory = pd.read_csv(source / arm / "trajectory.csv").sort_values("epoch")
        for epoch in trajectory.epoch.astype(int).tail(settings["checkpoint_tail"]):
            inputs[f"{arm}_epoch_{epoch}"] = (
                source / arm / f"epoch_{epoch:03d}/calibrated/weights.csv"
            )
    write(
        root / "MANIFEST.json",
        dict(
            source_convergence=str(source.resolve()),
            source_ratio=str(Path(source_manifest["source"]).resolve()),
            settings=settings,
            inputs={
                key: dict(path=str(path.resolve()), sha256=sha(path))
                for key, path in inputs.items()
            },
            arms=list(ARMS),
            classifier_frozen=True,
            simulation_banks_reused=True,
            population_uses_q=False,
            production_prior_modified=False,
            selection_rule=(
                "one-standard-error heldout likelihood, then minimum median "
                "bootstrap parent-density sliced Wasserstein"
            ),
        ),
    )
    print(yaml.safe_dump(settings, sort_keys=False))
    print("Frozen epoch-600 classifiers; no simulation or neural training.")


def _mixture_log_values(logc, frequencies, weights):
    weights = np.asarray(weights, float)
    with np.errstate(divide="ignore"):
        return logsumexp(
            np.asarray(logc, float)
            - np.log(np.asarray(frequencies, float))[None]
            + np.log(weights)[None],
            axis=1,
        )


def _effective_components(weights):
    weights = np.asarray(weights, float)
    positive = weights[weights > 0]
    return float(np.exp(-np.sum(positive * np.log(positive))))


def _json_scalar(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return str(value)


def _common_draws(basis, weights, uniforms, normals):
    cumulative = np.cumsum(np.asarray(weights, float))
    cumulative[-1] = 1.0
    labels = np.searchsorted(cumulative, uniforms, side="right")
    x = normals.copy()
    x[:, basis.indices] = (
        basis.centers[labels] + normals[:, basis.indices] * basis.scales[labels]
    )
    return x


def _physical_sw(predicted, truth, names, seed):
    from scripts.report_feniks_forward_population import population_metrics

    _, joint = population_metrics(predicted, truth, tuple(names), seed)
    return float(joint.loc[joint.group == "physical", "sliced_wasserstein"].iloc[0])


def _select_strength(path, heldout_values):
    """Apply a paired one-SE rule, then minimize bootstrap density instability."""
    path = path.copy()
    best_strength = float(path.loc[path.heldout_log_likelihood.idxmax(), "strength"])
    best_values = np.asarray(heldout_values[best_strength], float)
    for strength, values in heldout_values.items():
        difference = best_values - np.asarray(values, float)
        degradation = float(difference.mean())
        standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        mask = np.isclose(path.strength, strength)
        path.loc[mask, "heldout_degradation_from_best"] = degradation
        path.loc[mask, "heldout_difference_standard_error"] = standard_error
        path.loc[mask, "heldout_one_se"] = degradation <= max(standard_error, 1e-8)
    admissible = path[path.heldout_one_se.astype(bool)]
    selected = admissible.sort_values(
        ["bootstrap_parent_sw_median", "strength"], ascending=[True, True]
    ).iloc[0]
    return path, float(selected.strength)


def _fit(logc, frequencies, strength, *, alpha, eligible, weak_mass, weights=None, tol):
    kwargs = dict(
        alpha=alpha,
        eligible=eligible,
        weak_parent_mass=weak_mass,
        observation_weights=weights,
        tolerance=tol,
    )
    if strength == 0:
        return fit_selected_weights(logc, frequencies, **kwargs)
    return fit_selected_weights_kl(logc, frequencies, strength=strength, **kwargs)


def run(root: Path, task: int):
    import equinox as eqx
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import x_to_theta
    from scripts.feniks_forward_population import classifier_template, classify
    from scripts.feniks_ratio_ladder import _runtime

    if task not in range(len(ARMS)):
        raise ValueError("population-inversion task must be 0 or 1")
    manifest, audit = contract(root)
    source = Path(manifest["source_convergence"])
    (
        source_manifest,
        settings,
        ratio_manifest,
        _,
        reference,
        target,
        basis,
        splits,
        frequencies,
        alpha,
        eligible,
        u_true,
        _,
    ) = _context(source)
    arm = ARMS[task]
    out = root / arm
    write(out / "PROGRESS.json", dict(stage="classifier_logits", arm=arm))

    source_final = read(
        Path(source_manifest["source_followup"]) / f"{arm}_seed1/FINAL.json"
    )
    classifier_settings = {
        **settings["classifier"],
        "seed": source_final["classifier_seed"],
    }
    feature_key = "noiseless_features" if arm.startswith("noiseless") else "features"
    features = reference[feature_key]
    target_features = target[feature_key]
    candidate = classifier_template(
        features.shape[1], basis.components, classifier_settings
    )
    classifier = eqx.tree_deserialise_leaves(source / arm / "best.eqx", candidate)
    calibration_logits = classify(classifier, features[splits["calibration"]])
    offsets, calibration = fit_marginal_logit_offsets(
        calibration_logits, frequencies, **settings["calibration"]
    )
    logc_fit = apply_logit_offsets(
        classify(classifier, target_features[splits["target_fit"]]), offsets
    )
    logc_heldout = apply_logit_offsets(
        classify(classifier, target_features[splits["target_heldout"]]), offsets
    )
    np.savez_compressed(
        out / "classifier_logits.npz", fit=logc_fit, heldout=logc_heldout
    )

    runtime = _runtime(ratio_manifest, out)[1]
    weak_mass = read(Path(ratio_manifest["parent"]) / "MANIFEST.json")["settings"][
        "weak_parent_mass_cap"
    ]
    rng = np.random.default_rng(audit["metric_seed"])
    uniforms = rng.random(audit["metric_draws"])
    normals = rng.normal(size=(audit["metric_draws"], len(basis.names)))

    def theta(weights):
        x = _common_draws(basis, weights, uniforms, normals)
        return np.asarray(x_to_theta(jnp.asarray(x), runtime.latent_spec))

    truth_theta = theta(u_true)
    checkpoint_rows = []
    trajectory = pd.read_csv(source / arm / "trajectory.csv").sort_values("epoch")
    tail = trajectory.tail(audit["checkpoint_tail"])
    previous_theta = None
    for epoch in tail.epoch.astype(int):
        table = pd.read_csv(source / arm / f"epoch_{epoch:03d}/calibrated/weights.csv")
        current_theta = theta(table.fitted_parent.to_numpy())
        checkpoint_rows.append(
            dict(
                epoch=epoch,
                direct_previous_parent_sw=(
                    np.nan
                    if previous_theta is None
                    else _physical_sw(
                        current_theta, previous_theta, basis.names, audit["metric_seed"]
                    )
                ),
                truth_parent_sw=_physical_sw(
                    current_theta, truth_theta, basis.names, audit["metric_seed"]
                ),
            )
        )
        previous_theta = current_theta
    pd.DataFrame(checkpoint_rows).to_csv(out / "checkpoint_stability.csv", index=False)

    strengths = [float(value) for value in audit["strengths"]]
    path_rows, heldout_values, fitted, fitted_theta = [], {}, {}, {}
    write(out / "PROGRESS.json", dict(stage="regularization_path", arm=arm))
    for strength in strengths:
        selected, diagnostics = _fit(
            logc_fit,
            frequencies,
            strength,
            alpha=alpha,
            eligible=eligible,
            weak_mass=weak_mass,
            tol=audit["solver_tolerance"],
        )
        diagnostics = {
            key: value for key, value in diagnostics.items() if key != "strength"
        }
        parent = parent_from_selected(selected, alpha)
        fitted[strength] = (selected, parent)
        heldout_values[strength] = _mixture_log_values(
            logc_heldout, frequencies, selected
        )
        learned_theta = theta(parent)
        fitted_theta[strength] = learned_theta
        path_rows.append(
            dict(
                arm=arm,
                strength=strength,
                heldout_log_likelihood=float(heldout_values[strength].mean()),
                parent_physical_sliced_wasserstein=_physical_sw(
                    learned_theta, truth_theta, basis.names, audit["metric_seed"]
                ),
                effective_parent_components=_effective_components(parent),
                near_zero_parent_fraction=float(np.mean(parent <= 1e-6)),
                maximum_parent_weight=float(parent.max()),
                alpha_fitted=float(parent @ alpha),
                **diagnostics,
            )
        )
    path = pd.DataFrame(path_rows)

    bootstrap_rows = []
    write(out / "PROGRESS.json", dict(stage="bootstrap", arm=arm, complete=0))
    for repeat in range(audit["bootstraps"]):
        counts = np.bincount(
            rng.integers(len(logc_fit), size=len(logc_fit)), minlength=len(logc_fit)
        )
        for strength in strengths:
            selected, _ = _fit(
                logc_fit,
                frequencies,
                strength,
                alpha=alpha,
                eligible=eligible,
                weak_mass=weak_mass,
                weights=counts,
                tol=audit["solver_tolerance"],
            )
            parent = parent_from_selected(selected, alpha)
            bootstrap_rows.append(
                dict(
                    arm=arm,
                    replicate=repeat,
                    strength=strength,
                    direct_full_parent_sw=_physical_sw(
                        theta(parent),
                        fitted_theta[strength],
                        basis.names,
                        audit["metric_seed"],
                    ),
                    parent_weight_l1_to_full=float(
                        abs(parent - fitted[strength][1]).sum()
                    ),
                    effective_parent_components=_effective_components(parent),
                )
            )
        write(
            out / "PROGRESS.json",
            dict(stage="bootstrap", arm=arm, complete=repeat + 1),
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    stability = bootstrap.groupby("strength", as_index=False).agg(
        bootstrap_parent_sw_median=("direct_full_parent_sw", "median"),
        bootstrap_parent_sw_q90=("direct_full_parent_sw", lambda x: x.quantile(0.9)),
        bootstrap_weight_l1_median=("parent_weight_l1_to_full", "median"),
    )
    path = path.merge(stability, on="strength", validate="one_to_one")
    path, selected_strength = _select_strength(path, heldout_values)
    selected_row = path[np.isclose(path.strength, selected_strength)].iloc[0]

    path.to_csv(out / "regularization_path.csv", index=False)
    bootstrap.to_csv(out / "bootstrap.csv", index=False)
    selected_weights, selected_parent = fitted[selected_strength]
    pd.DataFrame(
        dict(
            component=np.arange(len(selected_parent)),
            selected_weight=selected_weights,
            parent_weight=selected_parent,
            alpha=alpha,
            reference_selected=frequencies,
            true_parent=u_true,
        )
    ).to_csv(out / "selected_weights.csv", index=False)
    write(
        out / "FINAL.json",
        dict(
            status="POPULATION_INVERSION_AUDIT_ARM_COMPLETE",
            arm=arm,
            selected_strength=selected_strength,
            selected={
                key: _json_scalar(value)
                for key, value in selected_row.to_dict().items()
                if key != "arm"
            },
            calibration=calibration,
            final_classifier_sha256=sha(source / arm / "best.eqx"),
            classifier_frozen=True,
            simulation_banks_reused=True,
            truth_used_for_selection=False,
            population_uses_q=False,
            production_prior_modified=False,
        ),
    )


def report(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest, settings = contract(root)
    source_report = read(Path(manifest["source_convergence"]) / "report/FINAL.json")
    oracle = float(source_report["physical_oracle_parent_sw"])
    paths, checkpoints, finals = [], [], {}
    for arm in ARMS:
        final = read(root / arm / "FINAL.json")
        if final.get("status") != "POPULATION_INVERSION_AUDIT_ARM_COMPLETE":
            raise ValueError(f"incomplete population inversion arm {arm}")
        finals[arm] = final
        paths.append(pd.read_csv(root / arm / "regularization_path.csv"))
        frame = pd.read_csv(root / arm / "checkpoint_stability.csv")
        frame.insert(0, "arm", arm)
        checkpoints.append(frame)
    path = pd.concat(paths, ignore_index=True)
    checkpoint = pd.concat(checkpoints, ignore_index=True)
    path.to_csv(root / "report/regularization_path.csv", index=False)
    checkpoint.to_csv(root / "report/checkpoint_stability.csv", index=False)
    chosen = path[
        path.apply(
            lambda row: np.isclose(row.strength, finals[row.arm]["selected_strength"]),
            axis=1,
        )
    ]
    contracts = settings["contracts"]
    decisions = {
        "heldout_one_se": bool(chosen.heldout_one_se.all()),
        "bootstrap_density_stable": bool(
            (
                chosen.bootstrap_parent_sw_median
                <= contracts["maximum_bootstrap_physical_sw"]
            ).all()
        ),
        "physical_parent_closure": bool(
            (
                chosen.parent_physical_sliced_wasserstein
                <= oracle + contracts["maximum_parent_sw_above_physical_oracle"]
            ).all()
        ),
    }
    if all(decisions.values()):
        next_action = "repair_decoder_contract_then_one_end_to_end_parent_fit"
    else:
        next_action = "reduce_or_rebuild_population_component_basis"

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for arm, group in path.groupby("arm"):
        group = group.sort_values("strength")
        x = np.arange(len(group))
        axes[0, 0].plot(x, group.heldout_degradation_from_best, marker="o", label=arm)
        axes[0, 1].plot(x, group.bootstrap_parent_sw_median, marker="o", label=arm)
        axes[1, 0].plot(
            x, group.parent_physical_sliced_wasserstein, marker="o", label=arm
        )
        axes[1, 1].plot(x, group.effective_parent_components, marker="o", label=arm)
        axes[1, 1].set_xticks(x, [f"{v:g}" for v in group.strength], rotation=45)
    axes[0, 0].set(title="Heldout degradation", ylabel="mean log-likelihood")
    axes[0, 1].set(title="Bootstrap density instability", ylabel="physical SW")
    axes[1, 0].axhline(oracle, color="black", linestyle="--")
    axes[1, 0].set(title="Truth closure (evaluation only)", ylabel="physical SW")
    axes[1, 1].set(title="Effective parent components", xlabel="KL strength")
    for axis in axes.flat:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "report/population_inversion_audit.png", dpi=170)
    plt.close(figure)

    write(
        root / "report/FINAL.json",
        dict(
            status="POPULATION_INVERSION_AUDIT_REPORT_COMPLETE",
            decisions=decisions,
            next_action=next_action,
            selected_strengths={arm: finals[arm]["selected_strength"] for arm in ARMS},
            physical_oracle_parent_sw=oracle,
            classifier_frozen=True,
            simulation_banks_reused=True,
            truth_used_for_selection=False,
            population_uses_q=False,
            production_prior_modified=False,
            production_ready=False,
        ),
    )
    (root / "report/REPORT.md").write_text(
        "# Frozen population-inversion audit\n\n"
        "Regularization is selected from held-out likelihood and bootstrap density "
        "stability. Known truth is used only for closure evaluation.\n\n"
        f"Decision: `{next_action}`.\n"
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
