"""Decide whether photometric component ratios converge and calibrate.

This workflow reuses immutable ratio-ladder banks. It never simulates new
objects, calls an individual posterior, or modifies a production population.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import logsumexp

from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_ratio_ladder import (
    _basis,
    _ece,
    _mixture_objective,
    _runtime,
    _stratified_split,
    _target_split,
    exact_selected_log_classifier,
)
from scripts.feniks_ratio_ladder import contract as ratio_contract
from scripts.feniks_weighted_truth_flow_capacity import _status

ARMS = ("noiseless_photometry", "noisy_photometry")
CELLS = tuple((arm, replica) for arm in ARMS for replica in range(2))


def _cell_name(arm: str, replica: int) -> str:
    return f"{arm}_seed{replica}"


def contract(root: Path):
    manifest = read(root / "MANIFEST.json")
    for item in manifest["inputs"].values():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError(f"ratio-followup input changed: {path}")
    return manifest, manifest["settings"]


def _bank_paths(source: Path, kind: str) -> list[Path]:
    _, settings = ratio_contract(source)
    if kind == "reference":
        return [
            source / "bank" / f"reference_{i:03d}"
            for i in range(settings["reference_shards"])
        ]
    if kind == "target":
        return [source / "bank/target"]
    raise ValueError(f"unknown bank kind {kind}")


def _load_bank_keys(source: Path, kind: str, keys: tuple[str, ...]):
    arrays = {key: [] for key in keys}
    for directory in _bank_paths(source, kind):
        receipt = read(directory / "FINAL.json")
        if receipt.get("status") != "RATIO_LADDER_BANK_COMPLETE":
            raise ValueError(f"incomplete source bank {directory}")
        bank = directory / "bank.npz"
        if sha(bank) != receipt["bank_sha256"]:
            raise ValueError(f"changed source bank {bank}")
        with np.load(bank) as payload:
            for key in keys:
                arrays[key].append(payload[key])
    return {key: np.concatenate(value) for key, value in arrays.items()}


def _split_calibration_audit(test: np.ndarray, labels: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    calibration, audit = [], []
    for component in np.unique(labels[test]):
        indices = np.asarray(test[labels[test] == component]).copy()
        if len(indices) < 2:
            raise ValueError(f"component {component} cannot be split for calibration")
        rng.shuffle(indices)
        midpoint = len(indices) // 2
        calibration.extend(indices[:midpoint])
        audit.extend(indices[midpoint:])
    calibration = np.asarray(calibration, np.int64)
    audit = np.asarray(audit, np.int64)
    rng.shuffle(calibration)
    rng.shuffle(audit)
    return calibration, audit


def fit_marginal_logit_offsets(
    logc: np.ndarray,
    frequencies: np.ndarray,
    *,
    l2: float = 1e-8,
    tolerance: float = 1e-10,
    maximum_iterations: int = 1000,
):
    """Fit convex class intercepts so reference posterior means match c_j."""
    logc = np.asarray(logc, np.float64)
    frequencies = np.asarray(frequencies, np.float64)
    if logc.ndim != 2 or logc.shape[1] != len(frequencies):
        raise ValueError("logits and selected frequencies have incompatible shapes")
    if np.any(frequencies <= 0) or not np.isclose(frequencies.sum(), 1):
        raise ValueError("strictly positive normalized frequencies required")

    def objective(offset):
        shifted = logc + offset[None]
        probability = np.exp(shifted - logsumexp(shifted, axis=1, keepdims=True))
        value = np.mean(logsumexp(shifted, axis=1)) - frequencies @ offset
        value += 0.5 * l2 * np.mean(offset**2)
        gradient = probability.mean(axis=0) - frequencies
        gradient += l2 * offset / len(offset)
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(logc.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": tolerance, "gtol": tolerance, "maxiter": maximum_iterations},
    )
    offset = np.asarray(result.x) - np.mean(result.x)
    calibrated = apply_logit_offsets(logc, offset)
    moment = np.exp(calibrated - np.log(frequencies)[None]).mean(axis=0)
    gap = float(np.max(abs(moment - 1)))
    if not result.success and gap > 1e-5:
        raise RuntimeError(f"logit calibration failed: {result.message}; gap={gap}")
    return offset, dict(
        success=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit),
        calibration_ratio_moment_max_abs_error=gap,
    )


def apply_logit_offsets(logc: np.ndarray, offset: np.ndarray):
    shifted = np.asarray(logc, np.float64) + np.asarray(offset)[None]
    return shifted - logsumexp(shifted, axis=1, keepdims=True)


def _copy_resume(source: Path, destination: Path):
    resume = read(source / "RESUME.json")
    required = (
        "best.eqx",
        "RESUME.json",
        resume["state_file"],
        "training.jsonl",
    )
    for name in required:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, destination / name)
    validation = source / "validation_positions.npy"
    if validation.is_file():
        shutil.copy2(validation, destination / validation.name)


def prepare(source: Path, root: Path, config: Path):
    _status(source / "report/FINAL.json", "RATIO_LADDER_REPORT_COMPLETE")
    source_manifest, source_settings = ratio_contract(source)
    settings = yaml.safe_load(config.read_text())
    if settings["classifier"]["epochs"] <= source_settings["classifier"]["epochs"]:
        raise ValueError("follow-up epoch horizon must exceed the source horizon")
    for key in ("width", "depth", "batch_size", "learning_rate"):
        if settings["classifier"][key] != source_settings["classifier"][key]:
            raise ValueError(f"resume classifier setting changed: {key}")
    if root.exists():
        raise FileExistsError(root)

    labels_selected = _load_bank_keys(source, "reference", ("component", "selected"))
    target_selected = _load_bank_keys(source, "target", ("selected",))["selected"]
    train, validation, test = _stratified_split(
        labels_selected["component"],
        labels_selected["selected"],
        source_settings["seed"],
    )
    calibration, audit = _split_calibration_audit(
        test, labels_selected["component"], settings["calibration_seed"]
    )
    target_fit, target_heldout = _target_split(target_selected, settings["seed"])

    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "report").mkdir()
    shutil.copy2(config, root / "ratio_followup.yaml")
    np.savez(
        root / "splits.npz",
        train=train,
        validation=validation,
        calibration=calibration,
        audit=audit,
        target_fit=target_fit,
        target_heldout=target_heldout,
    )
    for arm, replica in CELLS:
        directory = root / _cell_name(arm, replica)
        directory.mkdir()
        if replica == 0:
            _copy_resume(source / arm, directory)

    inputs = {
        "source_manifest": source / "MANIFEST.json",
        "source_config": source / "ratio_ladder.yaml",
        "source_report": source / "report/FINAL.json",
        "source_physical_classifier": source / "physical_a/best.eqx",
        "config": root / "ratio_followup.yaml",
        "splits": root / "splits.npz",
    }
    write(
        root / "MANIFEST.json",
        dict(
            source=str(source.resolve()),
            settings=settings,
            source_settings=source_settings,
            source_manifest_sha256=sha(source / "MANIFEST.json"),
            inputs={
                key: dict(path=str(path.resolve()), sha256=sha(path))
                for key, path in inputs.items()
            },
            cells=[_cell_name(*cell) for cell in CELLS],
            shared_target_split=True,
            calibration_source="disjoint selected-reference simulations",
            simulation_banks_reused=True,
            population_uses_q=False,
            production_prior_modified=False,
        ),
    )
    print(yaml.safe_dump(settings, sort_keys=False))
    print("Four classifier cells; source banks reused; no forward simulation.")


def _context(root: Path):
    manifest, settings = contract(root)
    source = Path(manifest["source"])
    source_manifest, source_settings = ratio_contract(source)
    reference = _load_bank_keys(
        source,
        "reference",
        ("x", "features", "noiseless_features", "selected", "component"),
    )
    target = _load_bank_keys(
        source,
        "target",
        ("x", "features", "noiseless_features", "selected", "component"),
    )
    basis = _basis(source)
    with np.load(root / "splits.npz") as payload:
        splits = {key: payload[key] for key in payload.files}
    labels = reference["component"]
    frequencies = np.bincount(
        labels[splits["train"]], minlength=basis.components
    ).astype(float)
    frequencies /= frequencies.sum()
    attempts = np.bincount(labels, minlength=basis.components).astype(float)
    successes = np.bincount(
        labels[reference["selected"]], minlength=basis.components
    ).astype(float)
    alpha = successes / attempts
    table = pd.read_csv(source_manifest["inputs"]["classifier_weights"]["path"])
    eligible = (
        table.eligible.to_numpy(bool)
        & (
            successes
            >= source_settings["reference_min_selected_per_component"]
        )
        & (alpha > 0)
    )
    u_true = pd.read_csv(source_manifest["inputs"]["clean_weights"]["path"])[
        "true_parent"
    ].to_numpy()
    from euclid_dsps.amortized.forward_population import selected_from_parent

    v_true = selected_from_parent(u_true, alpha)
    return (
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
    )


def _classifier_metrics(logc, labels, frequencies):
    ratio_moment = np.exp(logc - np.log(frequencies)[None]).mean(axis=0)
    return dict(
        classifier_nll=float(-logc[np.arange(len(labels)), labels].mean()),
        classifier_accuracy=float((logc.argmax(axis=1) == labels).mean()),
        classifier_ece=float(_ece(logc, labels)),
        ratio_moment_median_abs_error=float(np.median(abs(ratio_moment - 1))),
        ratio_moment_max_abs_error=float(np.max(abs(ratio_moment - 1))),
        ratio_moment_min=float(ratio_moment.min()),
        ratio_moment_max=float(ratio_moment.max()),
    )


def _training_summary(history_path: Path, contracts: dict):
    history = pd.read_json(history_path, lines=True)
    history = history.sort_values("epoch").drop_duplicates("epoch", keep="last")
    window = contracts["convergence_window"]
    if len(history) < 2 * window:
        raise ValueError("insufficient history for convergence decision")
    previous = history.validation_nll.iloc[-2 * window : -window].min()
    recent = history.validation_nll.iloc[-window:].min()
    improvement = float(previous - recent)
    return dict(
        epochs=int(history.epoch.max()),
        best_epoch=int(history.loc[history.validation_nll.idxmin(), "epoch"]),
        best_nll=float(history.validation_nll.min()),
        last_nll=float(history.validation_nll.iloc[-1]),
        recent_nll_improvement=improvement,
        converged=bool(improvement < contracts["minimum_recent_nll_improvement"]),
    )


def _evaluate_population(
    *,
    out: Path,
    name: str,
    logc_fit: np.ndarray,
    logc_heldout: np.ndarray,
    frequencies: np.ndarray,
    alpha: np.ndarray,
    eligible: np.ndarray,
    u_true: np.ndarray,
    v_true: np.ndarray,
    basis,
    runtime,
    settings: dict,
    weak_parent_mass: float,
):
    import jax.numpy as jnp

    from euclid_dsps.amortized.forward_population import (
        fit_selected_weights,
        parent_from_selected,
    )
    from euclid_dsps.amortized.latent import x_to_theta
    from scripts.feniks_clean_parent import likelihood_geometry
    from scripts.report_feniks_forward_population import population_metrics

    directory = out / name
    directory.mkdir(exist_ok=True)
    v, diagnostics = fit_selected_weights(
        logc_fit,
        frequencies,
        alpha=alpha,
        eligible=eligible,
        weak_parent_mass=weak_parent_mass,
    )
    u = parent_from_selected(v, alpha)
    pd.DataFrame(
        dict(
            component=np.arange(len(u)),
            true_parent=u_true,
            fitted_parent=u,
            true_selected=v_true,
            fitted_selected=v,
            alpha=alpha,
            eligible=eligible,
            frequency=frequencies,
        )
    ).to_csv(directory / "weights.csv", index=False)
    geometry = likelihood_geometry(logc_fit, frequencies, v)
    write(directory / "likelihood_geometry.json", geometry)

    draws = settings["metric_draws"]
    rng = np.random.default_rng(settings["metric_seed"])
    learned_x, _ = basis.sample(rng, draws, weights=u)
    truth_x, _ = basis.sample(rng, draws, weights=u_true)
    learned_theta = np.asarray(x_to_theta(jnp.asarray(learned_x), runtime.latent_spec))
    truth_theta = np.asarray(x_to_theta(jnp.asarray(truth_x), runtime.latent_spec))
    parent_marginal, parent_joint = population_metrics(
        learned_theta,
        truth_theta,
        tuple(runtime.latent_spec.names),
        settings["metric_seed"],
    )
    parent_marginal.to_csv(directory / "parent_marginal.csv", index=False)
    parent_joint.to_csv(directory / "parent_joint.csv", index=False)

    learned_selected_x, _ = basis.sample(rng, draws, weights=v)
    true_selected_x, _ = basis.sample(rng, draws, weights=v_true)
    learned_selected = np.asarray(
        x_to_theta(jnp.asarray(learned_selected_x), runtime.latent_spec)
    )
    true_selected = np.asarray(
        x_to_theta(jnp.asarray(true_selected_x), runtime.latent_spec)
    )
    selected_marginal, selected_joint = population_metrics(
        learned_selected,
        true_selected,
        tuple(runtime.latent_spec.names),
        settings["metric_seed"],
    )
    selected_marginal.to_csv(directory / "selected_marginal.csv", index=False)
    selected_joint.to_csv(directory / "selected_joint.csv", index=False)
    parent_sw = float(
        parent_joint.loc[parent_joint.group == "physical", "sliced_wasserstein"].iloc[0]
    )
    selected_sw = float(
        selected_joint.loc[
            selected_joint.group == "physical", "sliced_wasserstein"
        ].iloc[0]
    )
    eigenvalues = np.asarray(geometry["eigenvalues"], dtype=float)
    positive = eigenvalues[eigenvalues > 0]
    condition = (
        float(positive.max() / positive.min()) if len(positive) else float("inf")
    )
    result = dict(
        variant=name,
        parent_physical_sliced_wasserstein=parent_sw,
        selected_physical_sliced_wasserstein=selected_sw,
        parent_weight_l1=float(abs(u - u_true).sum()),
        selected_weight_l1=float(abs(v - v_true).sum()),
        alpha_true=float(u_true @ alpha),
        alpha_fitted=float(u @ alpha),
        heldout_fit_log_likelihood=_mixture_objective(
            logc_heldout, frequencies, v
        ),
        heldout_true_log_likelihood=_mixture_objective(
            logc_heldout, frequencies, v_true
        ),
        zero_parent_weight_fraction=float(np.mean(u <= 1e-12)),
        tangent_condition_number=condition,
        tangent_min_eigenvalue=float(eigenvalues.min()),
        **diagnostics,
    )
    result["heldout_fit_minus_true"] = (
        result["heldout_fit_log_likelihood"]
        - result["heldout_true_log_likelihood"]
    )
    write(directory / "FINAL.json", result)
    return result


def run_cell(root: Path, task: int):
    import jax
    import jax.numpy as jnp

    from scripts.feniks_forward_population import (
        classifier_template,
        classify,
        supervised_fit,
    )

    if task not in range(len(CELLS)):
        raise ValueError("ratio-followup task must be 0..3")
    arm, replica = CELLS[task]
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
    out = root / _cell_name(arm, replica)
    feature_key = "noiseless_features" if arm == "noiseless_photometry" else "features"
    features = reference[feature_key]
    target_features = target[feature_key]
    labels = reference["component"]
    source_task = 2 if arm == "noiseless_photometry" else 3
    classifier_seed = source_settings["seed"] + source_task
    if replica:
        classifier_seed += settings["independent_seed_offset"]
    classifier_settings = {**settings["classifier"], "seed": classifier_seed}
    candidate = classifier_template(
        features.shape[1], basis.components, classifier_settings
    )

    def loss(net, values, targets):
        prediction = jax.nn.log_softmax(jax.vmap(net)(values), axis=-1)
        return -jnp.take_along_axis(prediction, targets[:, None], axis=1)[:, 0]

    classifier = supervised_fit(
        candidate,
        loss,
        features[splits["train"]],
        labels[splits["train"]],
        (features[splits["validation"]], labels[splits["validation"]]),
        classifier_settings,
        out,
    )
    logc_calibration = classify(classifier, features[splits["calibration"]])
    logc_audit_raw = classify(classifier, features[splits["audit"]])
    logc_fit_raw = classify(classifier, target_features[splits["target_fit"]])
    logc_heldout_raw = classify(
        classifier, target_features[splits["target_heldout"]]
    )
    offset, calibration = fit_marginal_logit_offsets(
        logc_calibration,
        frequencies,
        **settings["calibration"],
    )
    pd.DataFrame(
        dict(component=np.arange(len(offset)), logit_offset=offset)
    ).to_csv(out / "calibration_offsets.csv", index=False)
    runtime = _runtime(source_manifest, out)[1]
    weak_parent_mass = read(Path(source_manifest["parent"]) / "MANIFEST.json")[
        "settings"
    ]["weak_parent_mass_cap"]
    results = {}
    for name, fit_logits, heldout_logits, audit_logits in (
        ("raw", logc_fit_raw, logc_heldout_raw, logc_audit_raw),
        (
            "calibrated",
            apply_logit_offsets(logc_fit_raw, offset),
            apply_logit_offsets(logc_heldout_raw, offset),
            apply_logit_offsets(logc_audit_raw, offset),
        ),
    ):
        population = _evaluate_population(
            out=out,
            name=name,
            logc_fit=fit_logits,
            logc_heldout=heldout_logits,
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
        results[name] = {
            **population,
            **_classifier_metrics(
                audit_logits, labels[splits["audit"]], frequencies
            ),
        }
    training = _training_summary(out / "training.jsonl", settings["contracts"])
    write(
        out / "FINAL.json",
        dict(
            status="RATIO_FOLLOWUP_CELL_COMPLETE",
            arm=arm,
            replica=replica,
            classifier_seed=classifier_seed,
            resumed=replica == 0,
            classifier_sha256=sha(out / "best.eqx"),
            training=training,
            calibration=calibration,
            raw=results["raw"],
            calibrated=results["calibrated"],
            shared_target_split_sha256=manifest["inputs"]["splits"]["sha256"],
            population_uses_q=False,
            production_prior_modified=False,
        ),
    )


def _baseline_rows(root: Path, context):
    import equinox as eqx

    from scripts.feniks_forward_population import classifier_template, classify

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
    ) = context
    runtime = _runtime(source_manifest, root / "report/oracle_runtime")[1]
    weak_parent_mass = read(Path(source_manifest["parent"]) / "MANIFEST.json")[
        "settings"
    ]["weak_parent_mass_cap"]
    rows = []

    exact_fit = exact_selected_log_classifier(
        target["x"][splits["target_fit"]], basis, alpha, frequencies
    )
    exact_heldout = exact_selected_log_classifier(
        target["x"][splits["target_heldout"]], basis, alpha, frequencies
    )
    exact_audit = exact_selected_log_classifier(
        reference["x"][splits["audit"]], basis, alpha, frequencies
    )
    exact = _evaluate_population(
        out=root / "report",
        name="exact_theta",
        logc_fit=exact_fit,
        logc_heldout=exact_heldout,
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
    rows.append(
        {
            "arm": "exact_theta",
            "replica": -1,
            "calibration": "analytic",
            **exact,
            **_classifier_metrics(
                exact_audit, reference["component"][splits["audit"]], frequencies
            ),
        }
    )

    original_settings = {
        **source_settings["classifier"],
        "seed": source_settings["seed"] + 1,
    }
    candidate = classifier_template(5, basis.components, original_settings)
    classifier = eqx.tree_deserialise_leaves(
        Path(manifest["source"]) / "physical_a/best.eqx", candidate
    )
    physical = reference["x"][:, basis.indices]
    target_physical = target["x"][:, basis.indices]
    calibration_logits = classify(classifier, physical[splits["calibration"]])
    audit_raw = classify(classifier, physical[splits["audit"]])
    fit_raw = classify(classifier, target_physical[splits["target_fit"]])
    heldout_raw = classify(classifier, target_physical[splits["target_heldout"]])
    offset, _ = fit_marginal_logit_offsets(
        calibration_logits, frequencies, **settings["calibration"]
    )
    for label, fit_logits, heldout_logits, audit_logits in (
        ("raw", fit_raw, heldout_raw, audit_raw),
        (
            "calibrated",
            apply_logit_offsets(fit_raw, offset),
            apply_logit_offsets(heldout_raw, offset),
            apply_logit_offsets(audit_raw, offset),
        ),
    ):
        result = _evaluate_population(
            out=root / "report",
            name=f"physical_a_{label}",
            logc_fit=fit_logits,
            logc_heldout=heldout_logits,
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
        rows.append(
            {
                "arm": "physical_a",
                "replica": -1,
                "calibration": label,
                **result,
                **_classifier_metrics(
                    audit_logits,
                    reference["component"][splits["audit"]],
                    frequencies,
                ),
            }
        )
    return rows


def report(root: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = _context(root)
    _, settings = contract(root)
    rows = _baseline_rows(root, context)
    training_rows = []
    for arm, replica in CELLS:
        final = read(root / _cell_name(arm, replica) / "FINAL.json")
        if final.get("status") != "RATIO_FOLLOWUP_CELL_COMPLETE":
            raise ValueError(f"incomplete follow-up cell {arm} seed {replica}")
        training_rows.append({"arm": arm, "replica": replica, **final["training"]})
        for calibration in ("raw", "calibrated"):
            rows.append(
                {
                    "arm": arm,
                    "replica": replica,
                    "calibration": calibration,
                    **final[calibration],
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "report/summary.csv", index=False)
    training = pd.DataFrame(training_rows)
    training.to_csv(root / "report/convergence.csv", index=False)

    contracts = settings["contracts"]
    candidates = summary[
        summary.arm.isin(ARMS) & (summary.calibration == "calibrated")
    ]
    physical_oracle = float(
        summary.loc[summary.arm == "physical_a", "parent_physical_sliced_wasserstein"].min()
    )
    seed_spread = candidates.groupby("arm").parent_physical_sliced_wasserstein.agg(
        lambda value: float(value.max() - value.min())
    )
    decisions = {
        "classifier_converged": bool(training.converged.all()),
        "independent_ratio_normalization": bool(
            (
                candidates.ratio_moment_median_abs_error
                <= contracts["ratio_moment_median_abs_error"]
            ).all()
        ),
        "heldout_population_objective": bool(
            (
                candidates.heldout_fit_minus_true.abs()
                <= contracts["heldout_fit_minus_true_abs"]
            ).all()
        ),
        "physical_parent_closure": bool(
            (
                candidates.parent_physical_sliced_wasserstein
                <= physical_oracle
                + contracts["maximum_parent_sw_above_physical_oracle"]
            ).all()
        ),
        "seed_stability": bool(
            (seed_spread <= contracts["maximum_seed_parent_sw_spread"]).all()
        ),
        "nondegenerate_simplex_solution": bool(
            (
                candidates.zero_parent_weight_fraction
                <= contracts["maximum_zero_weight_fraction"]
            ).all()
        ),
    }
    if not decisions["classifier_converged"]:
        next_action = "continue_or_redesign_classifier_before_population_changes"
    elif not decisions["independent_ratio_normalization"]:
        next_action = "replace_or_strengthen_density_ratio_estimator"
    elif not decisions["seed_stability"]:
        next_action = "ratio_estimator_seed_instability"
    elif not (
        decisions["heldout_population_objective"]
        and decisions["physical_parent_closure"]
        and decisions["nondegenerate_simplex_solution"]
    ):
        next_action = "regularize_or_reduce_population_weight_degrees_of_freedom"
    else:
        next_action = "repair_decoder_contract_then_run_one_end_to_end_parent_fit"

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    display = summary.copy()
    display["label"] = display.apply(
        lambda row: f"{row.arm}\ns{row.replica} {row.calibration}", axis=1
    )
    axes[0, 0].bar(display.label, display.parent_physical_sliced_wasserstein)
    axes[0, 0].axhline(physical_oracle, color="black", linestyle="--")
    axes[0, 0].set(title="Parent physical closure", ylabel="sliced Wasserstein")
    axes[0, 1].bar(display.label, display.ratio_moment_median_abs_error)
    axes[0, 1].axhline(
        contracts["ratio_moment_median_abs_error"], color="black", linestyle="--"
    )
    axes[0, 1].set(title="Independent ratio normalization", ylabel="median abs error")
    axes[1, 0].bar(display.label, display.heldout_fit_minus_true)
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(title="Held-out fit minus known truth", ylabel="mean log likelihood")
    axes[1, 1].bar(display.label, display.zero_parent_weight_fraction)
    axes[1, 1].axhline(
        contracts["maximum_zero_weight_fraction"], color="black", linestyle="--"
    )
    axes[1, 1].set(title="Simplex boundary", ylabel="zero parent-weight fraction")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelrotation=75, labelsize=7)
    figure.tight_layout()
    figure.savefig(root / "report/ratio_followup_summary.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arm, replica in CELLS:
        history = pd.read_json(
            root / _cell_name(arm, replica) / "training.jsonl", lines=True
        )
        history = history.sort_values("epoch").drop_duplicates("epoch", keep="last")
        label = f"{arm} seed {replica}"
        axes[0].plot(history.epoch, history.validation_nll, label=label)
        axes[1].plot(history.epoch, history.train_nll, label=label)
    axes[0].set(title="Validation convergence", xlabel="epoch", ylabel="NLL")
    axes[1].set(title="Training convergence", xlabel="epoch", ylabel="NLL")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "report/classifier_convergence.png", dpi=170)
    plt.close(figure)

    (root / "report/REPORT.md").write_text(
        "# FENIKS photometric-ratio decision\n\n"
        "All classifier cells reuse the original simulation banks and share "
        "identical target rows. Calibration uses only a disjoint reference "
        "simulation split. See `summary.csv` and `convergence.csv`.\n\n"
        f"Decision: `{next_action}`. This diagnostic does not promote or modify "
        "a production parent.\n"
    )
    write(
        root / "report/FINAL.json",
        dict(
            status="RATIO_FOLLOWUP_REPORT_COMPLETE",
            decisions=decisions,
            next_action=next_action,
            physical_oracle_parent_sw=physical_oracle,
            seed_parent_sw_spread={key: float(value) for key, value in seed_spread.items()},
            shared_target_split=True,
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
        run_cell(args.root.resolve(), args.task)
    else:
        report(args.root.resolve())


if __name__ == "__main__":
    main()
