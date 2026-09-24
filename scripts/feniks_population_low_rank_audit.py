"""Audit a smooth low-rank correction of frozen population components."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.forward_population import parent_from_selected
from euclid_dsps.amortized.population_low_rank import (
    build_spectral_modes,
    fit_low_rank_selected_weights,
)
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_population_basis_audit import (
    _classifier_log_probabilities,
)
from scripts.feniks_population_basis_audit import (
    contract as basis_contract,
)
from scripts.feniks_population_inversion_audit import (
    _common_draws,
    _effective_components,
    _json_scalar,
    _mixture_log_values,
    _physical_sw,
)
from scripts.feniks_ratio_followup import ARMS, _context
from scripts.feniks_weighted_truth_flow_capacity import _status


def contract(root: Path):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"population low-rank input changed: {path}")
    basis_contract(Path(manifest["source_basis_audit"]))
    return manifest, manifest["settings"]


def prepare(source: Path, root: Path, config: Path):
    _status(source / "report/FINAL.json", "POPULATION_BASIS_AUDIT_REPORT_COMPLETE")
    source_manifest, _ = basis_contract(source)
    settings = yaml.safe_load(config.read_text())
    ranks = [int(value) for value in settings["ranks"]]
    strengths = [float(value) for value in settings["strengths"]]
    if sorted(set(ranks)) != ranks or ranks[0] < 1 or ranks[-1] >= 128:
        raise ValueError("ranks must increase uniquely within [1, 127]")
    if strengths[0] != 0 or np.any(np.diff(strengths) <= 0):
        raise ValueError("strengths must start at zero and increase")
    for key in ("bootstraps", "metric_draws", "graph_neighbors"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ("logs", "spectral_basis", "report", *ARMS):
        (root / name).mkdir()
    shutil.copy2(config, root / "population_low_rank_audit.yaml")
    inputs = {
        "source_basis_manifest": source / "MANIFEST.json",
        "source_basis_report": source / "report/FINAL.json",
        "source_embedding": source / "hierarchy/embedding.npz",
        "source_groups": source / "hierarchy/groups.csv",
        "config": root / "population_low_rank_audit.yaml",
    }
    write(
        root / "MANIFEST.json",
        dict(
            source_basis_audit=str(source.resolve()),
            source_convergence=str(
                Path(source_manifest["source_convergence"]).resolve()
            ),
            settings=settings,
            inputs={
                key: dict(path=str(path.resolve()), sha256=sha(path))
                for key, path in inputs.items()
            },
            arms=list(ARMS),
            parameterization=(
                "v = c * (1 + Phi gamma), v>=floor; u=normalize(v/alpha)"
            ),
            objective="convex selected-mixture likelihood plus quadratic penalty",
            spectral_basis_source=(
                "joint physical and frozen-classifier response embedding from "
                "reference simulations only"
            ),
            classifier_frozen=True,
            simulation_banks_reused=True,
            truth_used_for_basis=False,
            truth_used_for_selection=False,
            population_uses_q=False,
            production_prior_modified=False,
        ),
    )
    print(yaml.safe_dump(settings, sort_keys=False))
    print("Smooth low-rank audit only; no DSPS, classifier or posterior training.")


def spectral_basis(root: Path):
    manifest, settings = contract(root)
    source_basis = Path(manifest["source_basis_audit"])
    convergence = Path(manifest["source_convergence"])
    context = _context(convergence)
    frequencies = context[8]
    embedding = np.load(source_basis / "hierarchy/embedding.npz")["embedding"]
    modes, diagnostics = build_spectral_modes(
        embedding,
        frequencies,
        neighbors=settings["graph_neighbors"],
        maximum_rank=max(settings["ranks"]),
        tail_component=0,
    )
    np.savez_compressed(
        root / "spectral_basis/modes.npz",
        modes=modes,
        embedding=embedding,
        eigenvalues=diagnostics["laplacian_eigenvalues"],
        reference_selected=frequencies,
    )
    table = pd.DataFrame({"component": np.arange(len(frequencies))})
    for index in range(modes.shape[1]):
        table[f"mode_{index + 1:03d}"] = modes[:, index]
    table.to_csv(root / "spectral_basis/modes.csv", index=False)
    write(
        root / "spectral_basis/FINAL.json",
        dict(
            status="POPULATION_LOW_RANK_BASIS_COMPLETE",
            maximum_rank=int(modes.shape[1]),
            graph_neighbors=diagnostics["neighbors"],
            graph_bandwidth=diagnostics["bandwidth"],
            graph_components=diagnostics["graph_components"],
            reference_centering_error=float(np.max(np.abs(frequencies @ modes))),
            modes_sha256=sha(root / "spectral_basis/modes.npz"),
            truth_used=False,
        ),
    )


def _load_modes(root: Path, maximum_rank: int):
    final = read(root / "spectral_basis/FINAL.json")
    path = root / "spectral_basis/modes.npz"
    if final.get("status") != "POPULATION_LOW_RANK_BASIS_COMPLETE":
        raise ValueError("spectral basis incomplete")
    if sha(path) != final["modes_sha256"]:
        raise ValueError("spectral basis changed")
    modes = np.load(path)["modes"]
    if modes.shape != (128, maximum_rank):
        raise ValueError("unexpected spectral mode shape")
    return modes


def _mark_heldout_admissible(path, heldout_values):
    """Mark candidates inside the paired heldout one-standard-error set."""
    path = path.copy()
    best_row = path.loc[path.heldout_log_likelihood.idxmax()]
    best_key = (int(best_row["rank"]), float(best_row.strength))
    best_values = np.asarray(heldout_values[best_key], dtype=np.float64)
    for key, values in heldout_values.items():
        difference = best_values - np.asarray(values, dtype=np.float64)
        degradation = float(difference.mean())
        standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        mask = (path["rank"] == key[0]) & np.isclose(path.strength, key[1])
        path.loc[mask, "heldout_degradation_from_best"] = degradation
        path.loc[mask, "heldout_difference_standard_error"] = standard_error
        path.loc[mask, "heldout_one_se"] = degradation <= max(standard_error, 1e-8)
    return path


def _select_candidate(path, heldout_values):
    """Use paired heldout one-SE admissibility, then bootstrap stability."""
    path = _mark_heldout_admissible(path, heldout_values)
    admissible = path[path.heldout_one_se.astype(bool)]
    if admissible.bootstrap_parent_sw_median.isna().any():
        raise ValueError("heldout-admissible candidate missing bootstrap metrics")
    selected = admissible.sort_values(
        ["bootstrap_parent_sw_median", "rank", "strength"],
        ascending=[True, True, True],
    ).iloc[0]
    return path, (int(selected["rank"]), float(selected.strength))


def run(root: Path, task: int):
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import x_to_theta
    from scripts.feniks_ratio_ladder import _runtime

    if task not in range(len(ARMS)):
        raise ValueError("population low-rank task must be 0 or 1")
    manifest, audit = contract(root)
    convergence = Path(manifest["source_convergence"])
    context = _context(convergence)
    (
        _,
        _,
        ratio_manifest,
        _,
        _,
        _,
        basis,
        _,
        frequencies,
        alpha,
        _,
        u_true,
        _,
    ) = context
    arm = ARMS[task]
    out = root / arm
    write(out / "PROGRESS.json", dict(stage="classifier_logits", arm=arm))
    logc_fit, logc_heldout, calibration = _classifier_log_probabilities(
        convergence, arm, context, target=True
    )
    write(out / "calibration.json", calibration)
    modes = _load_modes(root, max(audit["ranks"]))
    runtime = _runtime(ratio_manifest, out)[1]
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
    total_candidates = len(audit["ranks"]) * len(audit["strengths"])
    complete = 0
    for rank in map(int, audit["ranks"]):
        current_modes = modes[:, :rank]
        initial_coefficients = None
        for strength in map(float, audit["strengths"]):
            write(
                out / "PROGRESS.json",
                dict(
                    stage="candidate_path",
                    arm=arm,
                    complete=complete,
                    total=total_candidates,
                    rank=rank,
                    strength=strength,
                ),
            )
            selected, coefficients, diagnostics = fit_low_rank_selected_weights(
                logc_fit,
                frequencies,
                current_modes,
                strength=strength,
                weight_floor=audit["weight_floor"],
                tolerance=audit["solver_tolerance"],
                maximum_iterations=audit["solver_maximum_iterations"],
                initial_coefficients=initial_coefficients,
            )
            initial_coefficients = coefficients
            parent = parent_from_selected(selected, alpha)
            parent_theta = theta(parent)
            key = (rank, strength)
            candidates[key] = dict(
                selected=selected,
                parent=parent,
                coefficients=coefficients,
                modes=current_modes,
                theta=parent_theta,
            )
            heldout_values[key] = _mixture_log_values(
                logc_heldout, frequencies, selected
            )
            rows.append(
                dict(
                    arm=arm,
                    rank=rank,
                    strength=strength,
                    heldout_log_likelihood=float(heldout_values[key].mean()),
                    parent_physical_sliced_wasserstein=_physical_sw(
                        parent_theta, truth_theta, basis.names, audit["metric_seed"]
                    ),
                    effective_parent_components=_effective_components(parent),
                    near_zero_parent_fraction=float(np.mean(parent <= 1e-6)),
                    maximum_parent_weight=float(parent.max()),
                    selected_weight_sum=float(selected.sum()),
                    parent_weight_sum=float(parent.sum()),
                    alpha_fitted=float(parent @ alpha),
                    **diagnostics,
                )
            )
            complete += 1
    path = _mark_heldout_admissible(pd.DataFrame(rows), heldout_values)
    admissible_keys = {
        (int(row["rank"]), float(row.strength))
        for _, row in path[path.heldout_one_se.astype(bool)].iterrows()
    }
    if not admissible_keys:
        raise RuntimeError("no heldout-admissible low-rank candidates")
    path.to_csv(out / "candidate_path_prebootstrap.csv", index=False)

    bootstrap_rows = []
    for repeat in range(audit["bootstraps"]):
        counts = np.bincount(
            rng.integers(len(logc_fit), size=len(logc_fit)), minlength=len(logc_fit)
        )
        for key, candidate in candidates.items():
            if key not in admissible_keys:
                continue
            selected, coefficients, diagnostics = fit_low_rank_selected_weights(
                logc_fit,
                frequencies,
                candidate["modes"],
                strength=key[1],
                observation_weights=counts,
                weight_floor=audit["weight_floor"],
                tolerance=audit["solver_tolerance"],
                maximum_iterations=audit["solver_maximum_iterations"],
                initial_coefficients=candidate["coefficients"],
            )
            parent = parent_from_selected(selected, alpha)
            bootstrap_rows.append(
                dict(
                    arm=arm,
                    replicate=repeat,
                    rank=key[0],
                    strength=key[1],
                    direct_full_parent_sw=_physical_sw(
                        theta(parent),
                        candidate["theta"],
                        basis.names,
                        audit["metric_seed"],
                    ),
                    parent_weight_l1_to_full=float(
                        abs(parent - candidate["parent"]).sum()
                    ),
                    coefficient_l2=float(np.linalg.norm(coefficients)),
                    kkt_gap=diagnostics["kkt_gap"],
                )
            )
        write(
            out / "PROGRESS.json",
            dict(
                stage="bootstrap",
                arm=arm,
                complete=repeat + 1,
                total=audit["bootstraps"],
                candidates=len(admissible_keys),
            ),
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    stability = bootstrap.groupby(["rank", "strength"], as_index=False).agg(
        bootstrap_parent_sw_median=("direct_full_parent_sw", "median"),
        bootstrap_parent_sw_q90=("direct_full_parent_sw", lambda x: x.quantile(0.9)),
        bootstrap_weight_l1_median=("parent_weight_l1_to_full", "median"),
        bootstrap_maximum_kkt_gap=("kkt_gap", "max"),
    )
    path = path.merge(
        stability,
        on=["rank", "strength"],
        how="left",
        validate="one_to_one",
    )
    path, selected_key = _select_candidate(path, heldout_values)
    selected_row = path[
        (path["rank"] == selected_key[0]) & np.isclose(path.strength, selected_key[1])
    ].iloc[0]
    selected = candidates[selected_key]

    path.to_csv(out / "candidate_path.csv", index=False)
    bootstrap.to_csv(out / "bootstrap.csv", index=False)
    pd.DataFrame(
        dict(
            component=np.arange(basis.components),
            reference_selected=frequencies,
            learned_selected=selected["selected"],
            learned_parent=selected["parent"],
            true_parent=u_true,
            alpha=alpha,
        )
    ).to_csv(out / "selected_weights.csv", index=False)
    pd.DataFrame(
        dict(
            mode=np.arange(1, selected_key[0] + 1),
            coefficient=selected["coefficients"],
        )
    ).to_csv(out / "selected_coefficients.csv", index=False)
    write(
        out / "FINAL.json",
        dict(
            status="POPULATION_LOW_RANK_AUDIT_ARM_COMPLETE",
            arm=arm,
            selected_rank=selected_key[0],
            selected_strength=selected_key[1],
            selected={
                key: _json_scalar(value)
                for key, value in selected_row.to_dict().items()
                if key != "arm"
            },
            calibration=calibration,
            classifier_frozen=True,
            simulation_banks_reused=True,
            truth_used_for_basis=False,
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
    source_basis = Path(manifest["source_basis_audit"])
    source_report = read(source_basis / "report/FINAL.json")
    oracle = float(source_report["physical_oracle_parent_sw"])
    paths, finals = [], {}
    for arm in ARMS:
        final = read(root / arm / "FINAL.json")
        if final.get("status") != "POPULATION_LOW_RANK_AUDIT_ARM_COMPLETE":
            raise ValueError(f"incomplete low-rank arm {arm}")
        finals[arm] = final
        paths.append(pd.read_csv(root / arm / "candidate_path.csv"))
    path = pd.concat(paths, ignore_index=True)
    path.to_csv(root / "report/candidate_path.csv", index=False)
    chosen = path[
        path.apply(
            lambda row: (
                row["rank"] == finals[row.arm]["selected_rank"]
                and np.isclose(row.strength, finals[row.arm]["selected_strength"])
            ),
            axis=1,
        )
    ]
    contracts = settings["contracts"]
    decisions = {
        "heldout_one_se": bool(chosen.heldout_one_se.all()),
        "convex_solver_certified": bool(
            (chosen.kkt_gap <= settings["solver_tolerance"]).all()
            and (chosen.bootstrap_maximum_kkt_gap <= settings["solver_tolerance"]).all()
        ),
        "weights_normalized": bool(
            np.allclose(chosen.selected_weight_sum, 1.0, atol=1e-10)
            and np.allclose(chosen.parent_weight_sum, 1.0, atol=1e-10)
        ),
        "nondegenerate_parent": bool(
            (
                chosen.effective_parent_components
                >= contracts["minimum_effective_parent_components"]
            ).all()
            and (
                chosen.maximum_parent_weight
                <= contracts["maximum_single_parent_weight"]
            ).all()
        ),
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
        "rank_path_bracketed": bool((chosen["rank"] < max(settings["ranks"])).all()),
        "penalty_path_bracketed": bool(
            (chosen.strength < max(settings["strengths"])).all()
        ),
    }
    core = (
        "heldout_one_se",
        "convex_solver_certified",
        "weights_normalized",
        "nondegenerate_parent",
        "bootstrap_density_stable",
        "physical_parent_closure",
    )
    if (
        all(decisions[key] for key in core)
        and decisions["rank_path_bracketed"]
        and decisions["penalty_path_bracketed"]
    ):
        next_action = "repair_decoder_contract_then_one_end_to_end_parent_fit"
    elif not decisions["rank_path_bracketed"]:
        next_action = "extend_low_rank_path_before_scientific_decision"
    elif not decisions["penalty_path_bracketed"]:
        next_action = "extend_low_rank_penalty_path_before_scientific_decision"
    else:
        next_action = "smooth_low_rank_correction_inadequate_revisit_ratio_basis"

    figure, axes = plt.subplots(2, 4, figsize=(19, 9), sharex="col")
    metrics = (
        "heldout_degradation_from_best",
        "bootstrap_parent_sw_median",
        "parent_physical_sliced_wasserstein",
        "effective_parent_components",
    )
    titles = (
        "Heldout degradation",
        "Bootstrap parent instability",
        "Truth closure (evaluation only)",
        "Effective parent components",
    )
    for row_index, arm in enumerate(ARMS):
        arm_path = path[path.arm == arm]
        for strength, group in arm_path.groupby("strength"):
            group = group.sort_values("rank")
            for column, metric in enumerate(metrics):
                axes[row_index, column].plot(
                    group["rank"],
                    group[metric],
                    marker="o",
                    label=f"lambda={strength:g}",
                )
        selected = chosen[chosen.arm == arm].iloc[0]
        for column, metric in enumerate(metrics):
            axes[row_index, column].scatter(
                [selected["rank"]],
                [selected[metric]],
                marker="*",
                s=180,
                color="black",
                zorder=5,
            )
        axes[row_index, 0].set_ylabel(arm.replace("_photometry", ""))
        axes[row_index, 1].axhline(
            contracts["maximum_bootstrap_physical_sw"], color="black", linestyle="--"
        )
        axes[row_index, 2].axhline(oracle, color="black", linestyle="--")
        axes[row_index, 3].axhline(
            contracts["minimum_effective_parent_components"],
            color="black",
            linestyle="--",
        )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel("Spectral rank")
    for axis in axes.flat:
        axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(root / "report/population_low_rank_audit.png", dpi=170)
    plt.close(figure)

    spectral = np.load(root / "spectral_basis/modes.npz")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    eigenvalues = spectral["eigenvalues"]
    axes[0].plot(np.arange(1, len(eigenvalues)), eigenvalues[1:], marker=".")
    axes[0].set(
        xlabel="Nonconstant mode",
        ylabel="Graph eigenvalue",
        title="Spectral smoothness",
    )
    image = axes[1].imshow(spectral["modes"].T, aspect="auto", cmap="coolwarm")
    axes[1].set(
        xlabel="Population component", ylabel="Mode", title="Reference-centered modes"
    )
    figure.colorbar(image, ax=axes[1], shrink=0.8)
    figure.tight_layout()
    figure.savefig(root / "report/spectral_basis.png", dpi=170)
    plt.close(figure)

    write(
        root / "report/FINAL.json",
        dict(
            status="POPULATION_LOW_RANK_AUDIT_REPORT_COMPLETE",
            decisions=decisions,
            next_action=next_action,
            selected={
                arm: dict(
                    rank=finals[arm]["selected_rank"],
                    strength=finals[arm]["selected_strength"],
                )
                for arm in ARMS
            },
            physical_oracle_parent_sw=oracle,
            classifier_frozen=True,
            simulation_banks_reused=True,
            truth_used_for_basis=False,
            truth_used_for_selection=False,
            population_uses_q=False,
            production_prior_modified=False,
            production_ready=all(decisions.values()),
        ),
    )
    (root / "report/REPORT.md").write_text(
        "# Smooth low-rank population correction audit\n\n"
        "The selected mixture is fitted in a convex affine spectral subspace. "
        "The graph and all selection decisions are truth-free. Parent weights "
        "are reconstructed only through the explicit selection correction.\n\n"
        f"Decision: `{next_action}`.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "basis", "run", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.source is None or args.config is None:
            parser.error("prepare requires --source and --config")
        prepare(args.source.resolve(), args.root.resolve(), args.config.resolve())
    elif args.mode == "basis":
        spectral_basis(args.root.resolve())
    elif args.mode == "run":
        run(args.root.resolve(), args.task)
    else:
        report(args.root.resolve())


if __name__ == "__main__":
    main()
