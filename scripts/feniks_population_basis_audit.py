"""Audit identifiable hierarchical coarsenings of the 128-component parent."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.forward_population import parent_from_selected
from euclid_dsps.amortized.population_hierarchy import (
    aggregate_log_probabilities,
    aggregate_selection_efficiency,
    aggregate_vector,
    build_joint_hierarchy,
    component_confusion,
    expand_group_parent,
)
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_population_inversion_audit import (
    _common_draws,
    _effective_components,
    _fit,
    _json_scalar,
    _mixture_log_values,
    _physical_sw,
)
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
            raise ValueError(f"population-basis input changed: {path}")
    return manifest, manifest["settings"]


def prepare(source: Path, root: Path, config: Path):
    _status(source / "report/FINAL.json", "RATIO_CONVERGENCE_REPORT_COMPLETE")
    settings = yaml.safe_load(config.read_text())
    resolutions = [int(value) for value in settings["resolutions"]]
    strengths = [float(value) for value in settings["strengths"]]
    if sorted(set(resolutions)) != resolutions or resolutions[-1] != 128:
        raise ValueError("resolutions must increase uniquely and end at 128")
    if strengths[0] != 0 or np.any(np.diff(strengths) <= 0):
        raise ValueError("strengths must start at zero and increase")
    for key in ("bootstraps", "metric_draws", "response_rank"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ("logs", "hierarchy", "report", *ARMS):
        (root / name).mkdir()
    shutil.copy2(config, root / "population_basis_audit.yaml")
    inputs = {
        "source_manifest": source / "MANIFEST.json",
        "source_report": source / "report/FINAL.json",
        "source_splits": source / "splits.npz",
        "source_noiseless_classifier": source / "noiseless_photometry/best.eqx",
        "source_noisy_classifier": source / "noisy_photometry/best.eqx",
        "config": root / "population_basis_audit.yaml",
    }
    write(
        root / "MANIFEST.json",
        dict(
            source_convergence=str(source.resolve()),
            settings=settings,
            inputs={
                key: dict(path=str(path.resolve()), sha256=sha(path))
                for key, path in inputs.items()
            },
            arms=list(ARMS),
            hierarchy_source=(
                "joint normalized physical centers/scales and calibrated frozen "
                "classifier confusion on reference simulations only"
            ),
            shared_hierarchy=True,
            classifier_frozen=True,
            simulation_banks_reused=True,
            truth_used_for_hierarchy=False,
            truth_used_for_selection=False,
            population_uses_q=False,
            production_prior_modified=False,
        ),
    )
    print(yaml.safe_dump(settings, sort_keys=False))
    print("Shared hierarchy plus two inversion arms; no simulation or neural training.")


def _classifier_log_probabilities(source, arm, context, *, target=False):
    import equinox as eqx

    from scripts.feniks_forward_population import classifier_template, classify

    (
        source_manifest,
        settings,
        _,
        _,
        reference,
        target_bank,
        basis,
        splits,
        frequencies,
        _,
        _,
        _,
        _,
    ) = context
    source_final = read(
        Path(source_manifest["source_followup"]) / f"{arm}_seed1/FINAL.json"
    )
    classifier_settings = {
        **settings["classifier"],
        "seed": source_final["classifier_seed"],
    }
    feature_key = "noiseless_features" if arm.startswith("noiseless") else "features"
    candidate = classifier_template(
        reference[feature_key].shape[1], basis.components, classifier_settings
    )
    classifier = eqx.tree_deserialise_leaves(source / arm / "best.eqx", candidate)
    calibration_logits = classify(
        classifier, reference[feature_key][splits["calibration"]]
    )
    offsets, calibration = fit_marginal_logit_offsets(
        calibration_logits, frequencies, **settings["calibration"]
    )
    if target:
        return (
            apply_logit_offsets(
                classify(classifier, target_bank[feature_key][splits["target_fit"]]),
                offsets,
            ),
            apply_logit_offsets(
                classify(
                    classifier, target_bank[feature_key][splits["target_heldout"]]
                ),
                offsets,
            ),
            calibration,
        )
    audit = apply_logit_offsets(
        classify(classifier, reference[feature_key][splits["audit"]]), offsets
    )
    return audit, calibration


def hierarchy(root: Path):
    manifest, settings = contract(root)
    source = Path(manifest["source_convergence"])
    context = _context(source)
    reference, basis, splits = context[4], context[6], context[7]
    labels = reference["component"][splits["audit"]]
    confusions = []
    calibrations = {}
    for arm in ARMS:
        logc, calibration = _classifier_log_probabilities(source, arm, context)
        confusions.append(component_confusion(logc, labels, basis.components))
        calibrations[arm] = calibration
    groups, embedding = build_joint_hierarchy(
        basis.centers,
        basis.scales,
        confusions,
        settings["resolutions"],
        physical_weight=settings["hierarchy"]["physical_weight"],
        photometric_weight=settings["hierarchy"]["photometric_weight"],
        response_rank=settings["response_rank"],
        tail_component=0,
    )
    table = pd.DataFrame(dict(component=np.arange(basis.components)))
    for resolution, assignment in groups.items():
        table[f"group_{resolution}"] = assignment
    table.to_csv(root / "hierarchy/groups.csv", index=False)
    np.savez_compressed(root / "hierarchy/embedding.npz", embedding=embedding)
    write(
        root / "hierarchy/FINAL.json",
        dict(
            status="POPULATION_BASIS_HIERARCHY_COMPLETE",
            resolutions=list(groups),
            tail_component_preserved=True,
            shared_between_arms=True,
            truth_used=False,
            calibrations=calibrations,
            groups_sha256=sha(root / "hierarchy/groups.csv"),
        ),
    )


def _load_groups(root, settings, components):
    final = read(root / "hierarchy/FINAL.json")
    if final.get("status") != "POPULATION_BASIS_HIERARCHY_COMPLETE":
        raise ValueError("shared hierarchy incomplete")
    path = root / "hierarchy/groups.csv"
    if sha(path) != final["groups_sha256"]:
        raise ValueError("shared hierarchy changed")
    table = pd.read_csv(path)
    if not np.array_equal(table.component, np.arange(components)):
        raise ValueError("hierarchy component order changed")
    return {
        resolution: table[f"group_{resolution}"].to_numpy(int)
        for resolution in settings["resolutions"]
    }


def _select_candidate(path, heldout_values):
    path = path.copy()
    best_key = tuple(
        path.loc[path.heldout_log_likelihood.idxmax(), ["resolution", "strength"]]
    )
    best_values = np.asarray(heldout_values[best_key], float)
    for key, values in heldout_values.items():
        difference = best_values - np.asarray(values, float)
        degradation = float(difference.mean())
        standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        mask = (path.resolution == key[0]) & np.isclose(path.strength, key[1])
        path.loc[mask, "heldout_degradation_from_best"] = degradation
        path.loc[mask, "heldout_difference_standard_error"] = standard_error
        path.loc[mask, "heldout_one_se"] = degradation <= max(standard_error, 1e-8)
    admissible = path[path.heldout_one_se.astype(bool)]
    selected = admissible.sort_values(
        ["bootstrap_parent_sw_median", "resolution", "strength"],
        ascending=[True, True, True],
    ).iloc[0]
    return path, (int(selected.resolution), float(selected.strength))


def run(root: Path, task: int):
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import x_to_theta
    from scripts.feniks_ratio_ladder import _runtime

    if task not in range(len(ARMS)):
        raise ValueError("population-basis task must be 0 or 1")
    manifest, audit = contract(root)
    source = Path(manifest["source_convergence"])
    context = _context(source)
    (
        _,
        _,
        ratio_manifest,
        source_settings,
        reference,
        _,
        basis,
        _,
        frequencies,
        alpha,
        eligible,
        u_true,
        _,
    ) = context
    arm = ARMS[task]
    out = root / arm
    write(out / "PROGRESS.json", dict(stage="classifier_logits", arm=arm))
    logc_fit, logc_heldout, calibration = _classifier_log_probabilities(
        source, arm, context, target=True
    )
    write(out / "calibration.json", calibration)
    groups_by_resolution = _load_groups(root, audit, basis.components)

    labels = reference["component"]
    successes = np.bincount(
        labels[reference["selected"]], minlength=basis.components
    ).astype(float)
    # This is the finite-bank reference implied by the calibrated class prior.
    # It guarantees that grouping c and alpha commutes exactly with v -> u.
    reference_parent = parent_from_selected(frequencies, alpha)
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
    candidates = {}
    heldout_values = {}
    rows = []
    write(out / "PROGRESS.json", dict(stage="candidate_path", arm=arm))
    for resolution, groups in groups_by_resolution.items():
        group_fit = aggregate_log_probabilities(logc_fit, groups)
        group_heldout = aggregate_log_probabilities(logc_heldout, groups)
        group_frequency = aggregate_vector(frequencies, groups)
        group_frequency /= group_frequency.sum()
        group_reference, group_alpha = aggregate_selection_efficiency(
            reference_parent, alpha, groups
        )
        group_successes = aggregate_vector(successes, groups)
        if resolution == basis.components:
            group_eligible = eligible
        else:
            group_eligible = (
                group_successes
                >= source_settings["reference_min_selected_per_component"]
            ) & (group_alpha > 0)
        for strength in map(float, audit["strengths"]):
            selected, diagnostics = _fit(
                group_fit,
                group_frequency,
                strength,
                alpha=group_alpha,
                eligible=group_eligible,
                weak_mass=weak_mass,
                tol=audit["solver_tolerance"],
            )
            diagnostics = {
                key: value for key, value in diagnostics.items() if key != "strength"
            }
            parent = parent_from_selected(selected, group_alpha)
            expanded = expand_group_parent(parent, reference_parent, groups)
            expanded_theta = theta(expanded)
            key = (resolution, strength)
            candidates[key] = dict(
                selected=selected,
                parent=parent,
                expanded=expanded,
                group_alpha=group_alpha,
                group_frequency=group_frequency,
                groups=groups,
                fit=group_fit,
                eligible=group_eligible,
                theta=expanded_theta,
            )
            heldout_values[key] = _mixture_log_values(
                group_heldout, group_frequency, selected
            )
            rows.append(
                dict(
                    arm=arm,
                    resolution=resolution,
                    strength=strength,
                    heldout_log_likelihood=float(heldout_values[key].mean()),
                    parent_physical_sliced_wasserstein=_physical_sw(
                        expanded_theta, truth_theta, basis.names, audit["metric_seed"]
                    ),
                    effective_group_components=_effective_components(parent),
                    effective_original_components=_effective_components(expanded),
                    near_zero_group_fraction=float(np.mean(parent <= 1e-6)),
                    maximum_group_weight=float(parent.max()),
                    alpha_fitted=float(expanded @ alpha),
                    **diagnostics,
                )
            )
    path = pd.DataFrame(rows)

    bootstrap_rows = []
    total = audit["bootstraps"]
    for repeat in range(total):
        counts = np.bincount(
            rng.integers(len(logc_fit), size=len(logc_fit)), minlength=len(logc_fit)
        )
        for key, candidate in candidates.items():
            selected, _ = _fit(
                candidate["fit"],
                candidate["group_frequency"],
                key[1],
                alpha=candidate["group_alpha"],
                eligible=candidate["eligible"],
                weak_mass=weak_mass,
                weights=counts,
                tol=audit["solver_tolerance"],
            )
            parent = parent_from_selected(selected, candidate["group_alpha"])
            expanded = expand_group_parent(
                parent, reference_parent, candidate["groups"]
            )
            bootstrap_rows.append(
                dict(
                    arm=arm,
                    replicate=repeat,
                    resolution=key[0],
                    strength=key[1],
                    direct_full_parent_sw=_physical_sw(
                        theta(expanded),
                        candidate["theta"],
                        basis.names,
                        audit["metric_seed"],
                    ),
                    parent_weight_l1_to_full=float(
                        abs(expanded - candidate["expanded"]).sum()
                    ),
                )
            )
        write(
            out / "PROGRESS.json",
            dict(stage="bootstrap", arm=arm, complete=repeat + 1, total=total),
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    stability = bootstrap.groupby(["resolution", "strength"], as_index=False).agg(
        bootstrap_parent_sw_median=("direct_full_parent_sw", "median"),
        bootstrap_parent_sw_q90=("direct_full_parent_sw", lambda x: x.quantile(0.9)),
        bootstrap_weight_l1_median=("parent_weight_l1_to_full", "median"),
    )
    path = path.merge(stability, on=["resolution", "strength"], validate="one_to_one")
    path, selected_key = _select_candidate(path, heldout_values)
    selected_row = path[
        (path.resolution == selected_key[0])
        & np.isclose(path.strength, selected_key[1])
    ].iloc[0]
    selected = candidates[selected_key]

    path.to_csv(out / "candidate_path.csv", index=False)
    bootstrap.to_csv(out / "bootstrap.csv", index=False)
    pd.DataFrame(
        dict(
            component=np.arange(basis.components),
            group=selected["groups"],
            learned_parent=selected["expanded"],
            true_parent=u_true,
            reference_parent=reference_parent,
            alpha=alpha,
        )
    ).to_csv(out / "selected_weights.csv", index=False)
    write(
        out / "FINAL.json",
        dict(
            status="POPULATION_BASIS_AUDIT_ARM_COMPLETE",
            arm=arm,
            selected_resolution=selected_key[0],
            selected_strength=selected_key[1],
            selected={
                key: _json_scalar(value)
                for key, value in selected_row.to_dict().items()
                if key != "arm"
            },
            calibration=calibration,
            classifier_frozen=True,
            shared_hierarchy=True,
            simulation_banks_reused=True,
            truth_used_for_hierarchy=False,
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
    paths, finals = [], {}
    for arm in ARMS:
        final = read(root / arm / "FINAL.json")
        if final.get("status") != "POPULATION_BASIS_AUDIT_ARM_COMPLETE":
            raise ValueError(f"incomplete population-basis arm {arm}")
        finals[arm] = final
        paths.append(pd.read_csv(root / arm / "candidate_path.csv"))
    path = pd.concat(paths, ignore_index=True)
    path.to_csv(root / "report/candidate_path.csv", index=False)
    chosen = path[
        path.apply(
            lambda row: (
                row.resolution == finals[row.arm]["selected_resolution"]
                and np.isclose(row.strength, finals[row.arm]["selected_strength"])
            ),
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
        next_action = "replace_grouped_weights_with_smooth_low_rank_correction"

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    for row_index, arm in enumerate(ARMS):
        arm_path = path[path.arm == arm]
        for resolution, group in arm_path.groupby("resolution"):
            group = group.sort_values("strength")
            label = f"K={resolution}"
            axes[row_index, 0].plot(
                group.strength,
                group.heldout_degradation_from_best,
                marker="o",
                label=label,
            )
            axes[row_index, 1].plot(
                group.strength,
                group.bootstrap_parent_sw_median,
                marker="o",
                label=label,
            )
            axes[row_index, 2].plot(
                group.strength,
                group.parent_physical_sliced_wasserstein,
                marker="o",
                label=label,
            )
        selected = chosen[chosen.arm == arm].iloc[0]
        for column, metric in enumerate(
            (
                "heldout_degradation_from_best",
                "bootstrap_parent_sw_median",
                "parent_physical_sliced_wasserstein",
            )
        ):
            axes[row_index, column].scatter(
                [selected.strength],
                [selected[metric]],
                marker="*",
                s=180,
                color="black",
                zorder=5,
            )
            axes[row_index, column].set_xscale("symlog", linthresh=1e-4)
        axes[row_index, 0].set_ylabel(arm.replace("_photometry", ""))
        axes[row_index, 1].axhline(
            contracts["maximum_bootstrap_physical_sw"], color="black", linestyle="--"
        )
        axes[row_index, 2].axhline(oracle, color="black", linestyle="--")
    axes[0, 0].set_title("Heldout degradation")
    axes[0, 1].set_title("Bootstrap parent instability")
    axes[0, 2].set_title("Truth closure (evaluation only)")
    for axis in axes[-1]:
        axis.set_xlabel("KL strength")
    for axis in axes.flat:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(root / "report/population_basis_audit.png", dpi=170)
    plt.close(figure)

    write(
        root / "report/FINAL.json",
        dict(
            status="POPULATION_BASIS_AUDIT_REPORT_COMPLETE",
            decisions=decisions,
            next_action=next_action,
            selected={
                arm: dict(
                    resolution=finals[arm]["selected_resolution"],
                    strength=finals[arm]["selected_strength"],
                )
                for arm in ARMS
            },
            physical_oracle_parent_sw=oracle,
            classifier_frozen=True,
            shared_hierarchy=True,
            simulation_banks_reused=True,
            truth_used_for_hierarchy=False,
            truth_used_for_selection=False,
            population_uses_q=False,
            production_prior_modified=False,
            production_ready=False,
        ),
    )
    (root / "report/REPORT.md").write_text(
        "# Identifiable hierarchical population-basis audit\n\n"
        "The shared hierarchy uses only reference simulations, fixed physical "
        "geometry and frozen classifier responses. Resolution and regularization "
        "are selected without target truth.\n\n"
        f"Decision: `{next_action}`.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "hierarchy", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source is None or args.config is None:
            parser.error("prepare requires --source and --config")
        prepare(args.source.resolve(), args.root.resolve(), args.config.resolve())
    elif args.mode == "hierarchy":
        hierarchy(args.root.resolve())
    elif args.mode == "run":
        run(args.root.resolve(), args.task)
    else:
        report(args.root.resolve())


if __name__ == "__main__":
    main()
