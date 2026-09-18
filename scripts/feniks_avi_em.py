"""Selection-corrected generalized EM seeded from the latest AVI model."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from euclid_dsps.amortized.avi_experiments import Arm
from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from scripts.feniks_avi_experiments import read, sha, write

PHYSICAL = tuple(FENIKS_SPLINE15D_PARAMETERS[:5])
SFH = tuple(FENIKS_SPLINE15D_PARAMETERS[5:])


def _files_hash(paths) -> dict[str, str]:
    return {str(Path(path).resolve()): sha(path) for path in paths}


def _check_hashes(manifest: dict[str, Any]) -> None:
    for path, digest in manifest.get("hashes", {}).items():
        if sha(path) != digest:
            raise ValueError(f"changed EM input: {path}")


def _require_final(
    path: Path,
    status: str,
    *,
    manifest_path: Path | None = None,
    artifact: str | None = None,
) -> dict[str, Any]:
    value = read(path)
    if value.get("status") != status:
        raise ValueError(f"expected {status}: {path}")
    if manifest_path is not None and value.get("manifest_sha256") != sha(manifest_path):
        raise ValueError(f"stale completion receipt: {path}")
    if artifact is not None:
        target = path.parent / artifact
        expected = value.get(f"{Path(artifact).stem}_sha256")
        if not target.is_file() or (expected is not None and sha(target) != expected):
            raise ValueError(f"invalid completed artifact: {target}")
    return value


def _copy_runtime_inputs(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "logs").mkdir()
    for name in (
        "source_config.yaml",
        "train.npy",
        "validation.npy",
        "teachers.npz",
        "teachers.json",
    ):
        shutil.copy2(source / name, destination / name)


def _runtime_hashes(root: Path) -> dict[str, str]:
    return _files_hash(
        root / name
        for name in (
            "source_config.yaml",
            "train.npy",
            "validation.npy",
            "teachers.npz",
            "teachers.json",
        )
    )


def prepare(
    source: Path,
    training: Path,
    prior_followup: Path,
    selection: Path,
    root: Path,
    inference_root: Path,
    *,
    cycles: int,
    q_epochs: int,
) -> None:
    """Freeze the truth-free EM inputs and the isolated closure reference."""
    from scripts.feniks_avi_experiments import check_inputs

    if root.exists() or inference_root.exists():
        raise FileExistsError("new EM and inference roots are required")
    if cycles < 1 or q_epochs < 3 or q_epochs % 3:
        raise ValueError(
            "cycles must be positive and q_epochs a positive multiple of 3"
        )
    source_manifest = check_inputs(source)
    training_manifest = read(training / "MANIFEST.json")
    _check_hashes(training_manifest)
    prior_manifest = read(prior_followup / "MANIFEST.json")
    _check_hashes(prior_manifest)
    encoder_dir = training / "arms/Q_latest_refresh"
    prior_dir = prior_followup / "arms/P_latest_prior"
    _require_final(
        encoder_dir / "FINAL.json",
        "TRAINING_COMPLETE",
        manifest_path=training / "MANIFEST.json",
        artifact="encoder.eqx",
    )
    _require_final(
        prior_dir / "FINAL.json",
        "PRIOR_TRAINING_COMPLETE",
        manifest_path=prior_followup / "MANIFEST.json",
        artifact="prior.eqx",
    )
    selection_final = _require_final(
        selection / "selection/FINAL.json", "EXACT_OBSERVED_SELECTION_COMPLETE"
    )
    true_parent = selection / "selection/true_parent.parquet"
    true_selected = selection / "selection/true_selected.parquet"
    for path in (true_parent, true_selected):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    if int(selection_final["parent_objects"]) == int(
        selection_final["selected_objects"]
    ):
        raise ValueError("selection closure did not create distinct populations")

    _copy_runtime_inputs(source, root)
    input_paths = [
        source / "MANIFEST.json",
        training / "MANIFEST.json",
        encoder_dir / "FINAL.json",
        encoder_dir / "encoder.eqx",
        prior_followup / "MANIFEST.json",
        prior_dir / "FINAL.json",
        prior_dir / "prior.eqx",
        selection / "selection/FINAL.json",
        true_parent,
        true_selected,
        *(
            root / name
            for name in (
                "source_config.yaml",
                "train.npy",
                "validation.npy",
                "teachers.npz",
                "teachers.json",
            )
        ),
    ]
    manifest = {
        **{
            key: source_manifest[key]
            for key in (
                "source",
                "seed",
                "train_rows",
                "validation_rows",
                "validation_catalog",
                "global_batch",
                "local_microbatch",
                "accumulation",
                "gpus",
                "particles",
                "decoder_draw_block",
                "validation_particles",
                "learning_rate",
                "warmup_fraction",
                "cycle",
            )
        },
        "version": 1,
        "suite": "selection_corrected_avi_generalized_em_v1",
        "cycles": int(cycles),
        "q_epochs_per_cycle": int(q_epochs),
        "prior_sweeps_per_cycle": 1,
        "prior_macro_objects": 1024,
        "prior_particles": 256,
        "prior_learning_rate": 1.0e-5,
        "prior_trust_strength": 0.2,
        "prior_maximum_kl_per_dimension": 0.02,
        "initial_encoder": str((encoder_dir / "encoder.eqx").resolve()),
        "initial_prior": str((prior_dir / "prior.eqx").resolve()),
        "source_training": str(source.resolve()),
        "source_q_refresh": str(training.resolve()),
        "source_prior_followup": str(prior_followup.resolve()),
        "selection_root": str(selection.resolve()),
        "selection_final": selection_final,
        "true_parent": str(true_parent.resolve()),
        "true_selected": str(true_selected.resolve()),
        "inference_root": str(inference_root.resolve()),
        "em_contract": {
            "e_step": "ordinary full_15d logtarget-logproposal weights",
            "m_step": "-E_posterior[log p_parent] + log alpha_eta",
            "selected_fixed_point": "aggregate posterior == beta * p_parent / alpha",
            "parent_projection": "aggregate posterior / beta with joint weights",
            "selection_in_object_weights": False,
            "selection_in_prior_loss": True,
        },
        "truth_used_for_training_or_checkpoint_selection": False,
        "truth_role": "post-training fixed-cohort closure only",
        "scientific_promotion": False,
        "hashes": _files_hash(input_paths),
    }
    write(root / "MANIFEST.json", manifest)
    print(
        f"Prepared selection-corrected EM root={root} cycles={cycles} "
        f"q_epochs_per_cycle={q_epochs}",
        flush=True,
    )


def prepare_from_components(
    runtime_root: Path,
    initial_encoder: Path,
    initial_prior: Path,
    selection: Path,
    root: Path,
    inference_root: Path,
    *,
    cycles: int,
    q_epochs: int,
    prior_sweeps: int,
    e_step_mode: str,
    selection_objective_enabled: bool,
    track: str,
    cycle_offset: int = 0,
) -> None:
    """Prepare an EM trajectory from explicit immutable components.

    This entry point is used by the SBEB benchmark so warm and scratch
    encoders can share the same observed cohort, prior and selection contract.
    """
    if root.exists() or inference_root.exists():
        raise FileExistsError("new EM and inference roots are required")
    if cycles < 1 or q_epochs < 3 or q_epochs % 3:
        raise ValueError(
            "cycles must be positive and q_epochs a positive multiple of 3"
        )
    if prior_sweeps < 1:
        raise ValueError("prior_sweeps must be positive")
    if cycle_offset < 0:
        raise ValueError("cycle_offset must be non-negative")
    if e_step_mode not in {"raw_q", "ordinary_iw"}:
        raise ValueError(f"unknown E-step mode: {e_step_mode}")
    runtime_manifest = read(runtime_root / "MANIFEST.json")
    _check_hashes(runtime_manifest)
    required = (
        "source",
        "seed",
        "train_rows",
        "validation_rows",
        "validation_catalog",
        "global_batch",
        "local_microbatch",
        "accumulation",
        "gpus",
        "particles",
        "decoder_draw_block",
        "validation_particles",
        "learning_rate",
        "warmup_fraction",
        "cycle",
    )
    missing = [key for key in required if key not in runtime_manifest]
    if missing:
        raise ValueError(f"runtime manifest is missing fields: {missing}")
    for path in (initial_encoder, initial_prior):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    selection_final = _require_final(
        selection / "selection/FINAL.json", "EXACT_OBSERVED_SELECTION_COMPLETE"
    )
    true_parent = selection / "selection/true_parent.parquet"
    true_selected = selection / "selection/true_selected.parquet"
    for path in (true_parent, true_selected):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    if int(selection_final["parent_objects"]) == int(
        selection_final["selected_objects"]
    ):
        raise ValueError("selection closure did not create distinct populations")

    _copy_runtime_inputs(runtime_root, root)
    inputs = [
        runtime_root / "MANIFEST.json",
        initial_encoder,
        initial_prior,
        selection / "selection/FINAL.json",
        true_parent,
        true_selected,
        *(
            root / name
            for name in (
                "source_config.yaml",
                "train.npy",
                "validation.npy",
                "teachers.npz",
                "teachers.json",
            )
        ),
    ]
    manifest = {
        **{key: runtime_manifest[key] for key in required},
        "version": 2,
        "suite": "feniks_sbeb_multicycle_em_v2",
        "track": track,
        "cycles": int(cycles),
        "cycle_offset": int(cycle_offset),
        "q_epochs_per_cycle": int(q_epochs),
        "prior_sweeps_per_cycle": int(prior_sweeps),
        "prior_macro_objects": int(runtime_manifest.get("prior_macro_objects", 1024)),
        "prior_particles": int(runtime_manifest.get("prior_particles", 512)),
        "prior_learning_rate": float(
            runtime_manifest.get("prior_learning_rate", 1.0e-5)
        ),
        "prior_trust_strength": float(
            runtime_manifest.get("prior_trust_strength", 0.2)
        ),
        "prior_maximum_kl_per_dimension": float(
            runtime_manifest.get("prior_maximum_kl_per_dimension", 0.02)
        ),
        "initial_encoder": str(initial_encoder.resolve()),
        "initial_prior": str(initial_prior.resolve()),
        "source_training": str(runtime_root.resolve()),
        "source_q_refresh": str(runtime_root.resolve()),
        "source_prior_followup": str(runtime_root.resolve()),
        "selection_root": str(selection.resolve()),
        "selection_final": selection_final,
        "true_parent": str(true_parent.resolve()),
        "true_selected": str(true_selected.resolve()),
        "inference_root": str(inference_root.resolve()),
        "e_step_mode": e_step_mode,
        "selection_objective_enabled": bool(selection_objective_enabled),
        "em_contract": {
            "e_step": (
                "unweighted dense joint q draws"
                if e_step_mode == "raw_q"
                else "ordinary full_15d logtarget-logproposal weights"
            ),
            "m_step": (
                "-E_posterior[log p_parent] + log alpha_eta"
                if selection_objective_enabled
                else "-E_posterior[log p_selected] (selection ablation)"
            ),
            "selected_fixed_point": "aggregate posterior == beta * p_parent / alpha",
            "parent_projection": "aggregate posterior / beta with joint weights",
            "selection_in_object_weights": False,
            "selection_in_prior_loss": bool(selection_objective_enabled),
        },
        "truth_used_for_training_or_checkpoint_selection": False,
        "truth_role": "post-training blind-cohort closure only",
        "scientific_promotion": False,
        "hashes": _files_hash(inputs),
    }
    write(root / "MANIFEST.json", manifest)
    print(
        f"Prepared SBEB EM track={track} cycles={cycles} "
        f"q_epochs={q_epochs} prior_sweeps={prior_sweeps}",
        flush=True,
    )


def _manifest(root: Path) -> dict[str, Any]:
    value = read(root / "MANIFEST.json")
    _check_hashes(value)
    return value


def _cycle_directory(root: Path, cycle: int) -> Path:
    return root / "cycles" / f"cycle_{cycle:02d}"


def _components(root: Path, cycle: int) -> tuple[Path, Path]:
    manifest = _manifest(root)
    if cycle == 0:
        return Path(manifest["initial_encoder"]), Path(manifest["initial_prior"])
    previous = _cycle_directory(root, cycle) / "FINAL.json"
    receipt = _require_final(previous, "EM_CYCLE_COMPLETE")
    encoder = Path(receipt["encoder"])
    prior = Path(receipt["prior"])
    if (
        sha(encoder) != receipt["encoder_sha256"]
        or sha(prior) != receipt["prior_sha256"]
    ):
        raise ValueError(f"cycle {cycle} component provenance changed")
    return encoder, prior


def _base_stage_manifest(manifest: dict[str, Any], stage_root: Path) -> dict[str, Any]:
    return {
        **{
            key: manifest[key]
            for key in (
                "source",
                "train_rows",
                "validation_rows",
                "validation_catalog",
                "global_batch",
                "local_microbatch",
                "accumulation",
                "gpus",
                "particles",
                "decoder_draw_block",
                "validation_particles",
                "learning_rate",
                "warmup_fraction",
                "cycle",
            )
        },
        "hashes": _runtime_hashes(stage_root),
        "truth_used": False,
        "scientific_promotion": False,
    }


def prepare_mstep(root: Path, cycle: int) -> Path:
    manifest = _manifest(root)
    if cycle < 1 or cycle > int(manifest["cycles"]):
        raise ValueError("cycle lies outside the prepared EM contract")
    encoder, prior = _components(root, cycle - 1)
    stage_root = _cycle_directory(root, cycle) / "mstep"
    if stage_root.exists():
        existing = read(stage_root / "MANIFEST.json")
        _check_hashes(existing)
        return stage_root
    stage_root.parent.mkdir(parents=True, exist_ok=True)
    _copy_runtime_inputs(root, stage_root)
    arm = Arm(
        f"P_em_{cycle:02d}",
        experts=4,
        kind="prior",
        prior_initialization="learned_source",
    )
    stage = _base_stage_manifest(manifest, stage_root)
    global_cycle = int(manifest.get("cycle_offset", 0)) + cycle
    stage.update(
        {
            "version": 1,
            "suite": "selection_corrected_avi_em_mstep_v1",
            "em_cycle": cycle,
            "global_em_cycle": global_cycle,
            "arms": [asdict(arm)],
            "seed": int(manifest["seed"]) + 10_000 * global_cycle,
            "epochs": 1,
            "upstream_training": str(root.resolve()),
            "upstream_b_encoder": str(encoder.resolve()),
            "initial_prior_checkpoint_by_arm": {arm.name: str(prior.resolve())},
            "prior_sweeps": int(manifest["prior_sweeps_per_cycle"]),
            "prior_macro_objects": int(manifest["prior_macro_objects"]),
            "prior_particles": int(manifest["prior_particles"]),
            "prior_learning_rate": float(manifest["prior_learning_rate"]),
            "prior_trust_strength": float(manifest["prior_trust_strength"]),
            "prior_maximum_kl_per_dimension": float(
                manifest["prior_maximum_kl_per_dimension"]
            ),
            "selection_correction": (
                "required +log_alpha_eta in parent-prior loss"
                if manifest.get("selection_objective_enabled", True)
                else "disabled selected-density ablation"
            ),
            "selection_in_object_weights": False,
            "posterior_weight_contract": (
                "unweighted dense joint q draws"
                if manifest.get("e_step_mode", "ordinary_iw") == "raw_q"
                else "ordinary full_15d logtarget-logproposal"
            ),
            "e_step_mode": manifest.get("e_step_mode", "ordinary_iw"),
            "selection_objective_enabled": bool(
                manifest.get("selection_objective_enabled", True)
            ),
        }
    )
    stage["hashes"].update(_files_hash([root / "MANIFEST.json", encoder, prior]))
    write(stage_root / "MANIFEST.json", stage)
    print(f"Prepared EM cycle {cycle} M-step: {stage_root}", flush=True)
    return stage_root


def prepare_qstep(root: Path, cycle: int) -> Path:
    manifest = _manifest(root)
    encoder, _prior = _components(root, cycle - 1)
    mstep = _cycle_directory(root, cycle) / "mstep"
    mstep_manifest = mstep / "MANIFEST.json"
    arm_name = f"P_em_{cycle:02d}"
    _require_final(
        mstep / "arms" / arm_name / "FINAL.json",
        "PRIOR_TRAINING_COMPLETE",
        manifest_path=mstep_manifest,
        artifact="prior.eqx",
    )
    prior = mstep / "arms" / arm_name / "prior.eqx"
    stage_root = _cycle_directory(root, cycle) / "qstep"
    if stage_root.exists():
        existing = read(stage_root / "MANIFEST.json")
        _check_hashes(existing)
        return stage_root
    _copy_runtime_inputs(root, stage_root)
    arm = Arm(f"Q_em_{cycle:02d}", experts=4)
    stage = _base_stage_manifest(manifest, stage_root)
    global_cycle = int(manifest.get("cycle_offset", 0)) + cycle
    stage.update(
        {
            "version": 1,
            "suite": "selection_corrected_avi_em_qstep_v1",
            "em_cycle": cycle,
            "global_em_cycle": global_cycle,
            "arms": [asdict(arm)],
            "seed": int(manifest["seed"]) + 10_000 * global_cycle + 1,
            "epochs": int(manifest["q_epochs_per_cycle"]),
            "bootstrap_epochs": 0,
            "initial_encoder_checkpoint_by_arm": {arm.name: str(encoder.resolve())},
            "initial_prior_checkpoint_by_arm": {arm.name: str(prior.resolve())},
            "prior_frozen": True,
            "decoder_frozen": True,
            "selection_correction": (
                "selection-aware sleep; log_alpha is constant in the q-only loss"
            ),
            "selection_in_object_weights": False,
            "population_training_started": False,
        }
    )
    stage["hashes"].update(
        _files_hash(
            [
                root / "MANIFEST.json",
                encoder,
                prior,
                mstep / "arms" / arm_name / "FINAL.json",
            ]
        )
    )
    write(stage_root / "MANIFEST.json", stage)
    print(f"Prepared EM cycle {cycle} q-step: {stage_root}", flush=True)
    return stage_root


def finalize_cycle(root: Path, cycle: int) -> None:
    cycle_root = _cycle_directory(root, cycle)
    mstep = cycle_root / "mstep"
    qstep = cycle_root / "qstep"
    p_name = f"P_em_{cycle:02d}"
    q_name = f"Q_em_{cycle:02d}"
    p_final = _require_final(
        mstep / "arms" / p_name / "FINAL.json",
        "PRIOR_TRAINING_COMPLETE",
        manifest_path=mstep / "MANIFEST.json",
        artifact="prior.eqx",
    )
    q_final = _require_final(
        qstep / "arms" / q_name / "FINAL.json",
        "TRAINING_COMPLETE",
        manifest_path=qstep / "MANIFEST.json",
        artifact="encoder.eqx",
    )
    prior = mstep / "arms" / p_name / "prior.eqx"
    encoder = qstep / "arms" / q_name / "encoder.eqx"
    history = pd.read_csv(mstep / "arms" / p_name / "prior_training.csv")
    last = history.iloc[-1]
    write(
        cycle_root / "FINAL.json",
        {
            "status": "EM_CYCLE_COMPLETE",
            "cycle": cycle,
            "encoder": str(encoder.resolve()),
            "encoder_sha256": sha(encoder),
            "prior": str(prior.resolve()),
            "prior_sha256": sha(prior),
            "mstep_final_sha256": sha(mstep / "arms" / p_name / "FINAL.json"),
            "qstep_final_sha256": sha(qstep / "arms" / q_name / "FINAL.json"),
            "selection_log_alpha": float(last["log_alpha"]),
            "posterior_ess_median": float(last["posterior_ess_median"]),
            "posterior_max_weight_q90": float(last["posterior_max_weight_q90"]),
            "mstep": p_final,
            "qstep": q_final,
            "truth_used": False,
        },
    )
    print(f"EM cycle {cycle} complete", flush=True)


def _truth_table(path: Path, rows: np.ndarray) -> pd.DataFrame:
    names = list(FENIKS_SPLINE15D_PARAMETERS)
    frame = pd.read_parquet(path, columns=names).iloc[rows].copy()
    if not np.isfinite(frame[names].to_numpy(np.float64)).all():
        raise ValueError("finite validation truth is required after EM training")
    frame.insert(0, "row_index", rows)
    frame.insert(0, "object_id", rows)
    return frame


def prepare_inference(root: Path, inference_root: Path, *, particles: int) -> None:
    manifest = _manifest(root)
    if inference_root.exists():
        raise FileExistsError(inference_root)
    if particles < 256 or particles % 8:
        raise ValueError("particles must be >=256 and divisible by eight")
    variants = []
    cycle_offset = int(manifest.get("cycle_offset", 0))
    dependency_paths: list[Path] = [root / "MANIFEST.json"]
    for cycle in range(0, int(manifest["cycles"]) + 1):
        encoder, prior = _components(root, cycle)
        if cycle:
            dependency_paths.append(_cycle_directory(root, cycle) / "FINAL.json")
        dependency_paths.extend((encoder, prior))
        variants.append(
            {
                "name": f"cycle_{cycle:02d}",
                "global_cycle": cycle_offset + cycle,
                "encoder": str(encoder.resolve()),
                "prior": str(prior.resolve()),
                "prior_label": f"em_cycle_{cycle:02d}",
                "emit_prior_population": True,
            }
        )
    inference_root.mkdir(parents=True)
    (inference_root / "logs").mkdir()
    validation_rows = np.load(root / "validation.npy", allow_pickle=False)
    validation_truth = _truth_table(
        Path(manifest["validation_catalog"]), validation_rows
    )
    validation_truth.to_parquet(inference_root / "inference_truth.parquet", index=False)
    shutil.copy2(manifest["true_parent"], inference_root / "true_parent.parquet")
    shutil.copy2(manifest["true_selected"], inference_root / "true_selected.parquet")
    truth_paths = [
        inference_root / "inference_truth.parquet",
        inference_root / "true_parent.parquet",
        inference_root / "true_selected.parquet",
    ]
    write(
        inference_root / "MANIFEST.json",
        {
            "version": 1,
            "suite": "selection_corrected_avi_em_fixed_point_inference_v1",
            "training": str(root.resolve()),
            "source": manifest["source"],
            "config": str((root / "source_config.yaml").resolve()),
            "train_indices": str((root / "train.npy").resolve()),
            "validation_indices": str((root / "validation.npy").resolve()),
            "validation_catalog": manifest["validation_catalog"],
            "variants": variants,
            "particles": int(particles),
            "saved_draws": 512,
            "replicas": 2,
            "objects": len(validation_rows),
            "seed": int(manifest["seed"]) + 80_000_000,
            "prior_samples": 65536,
            "prior_selected_resamples": 65536,
            "posterior_weight_contract": "ordinary full_15d logtarget-logproposal",
            "truth_used_for_sampling_or_weighting": False,
            "truth_role": "report only after all EM cycles",
            "scientific_promotion": False,
            "hashes": {
                **_files_hash(dependency_paths),
                **_files_hash(
                    [
                        root / "source_config.yaml",
                        root / "train.npy",
                        root / "validation.npy",
                        *truth_paths,
                    ]
                ),
            },
        },
    )
    print(
        f"Prepared EM inference root with {len(variants)} cycles: {inference_root}",
        flush=True,
    )


def infer(inference_root: Path, task: int) -> None:
    from scripts.feniks_avi_overnight import infer as run_inference

    run_inference(inference_root, task)


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = values.sum()
    if not np.isfinite(values).all() or np.any(values < 0) or total <= 0:
        raise ValueError("invalid population weights")
    return values / total


def _resample(
    values: np.ndarray, weights: np.ndarray, *, count: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(values)[rng.choice(len(values), count, replace=True, p=weights)]


def _distance_rows(
    cycle: int,
    distributions: dict[str, np.ndarray],
    comparisons: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    rows = []
    for left, right in comparisons:
        for index, name in enumerate(FENIKS_SPLINE15D_PARAMETERS):
            reference = distributions[right][:, index]
            scale = max(
                float(np.quantile(reference, 0.75) - np.quantile(reference, 0.25)),
                1.0e-6,
            )
            rows.append(
                {
                    "cycle": cycle,
                    "comparison": f"{left}_vs_{right}",
                    "parameter": name,
                    "group": "physical" if name in PHYSICAL else "sfh",
                    "wasserstein_over_reference_iqr": float(
                        wasserstein_distance(distributions[left][:, index], reference)
                        / scale
                    ),
                }
            )
    return rows


def _plot_fixed_point(summary: pd.DataFrame, alpha: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = (
        (
            "aggregate_selected_vs_prior_selected",
            "physical",
            "Selected fixed point, 5D",
        ),
        ("aggregate_selected_vs_prior_selected", "sfh", "Selected fixed point, SFH"),
        ("aggregate_parent_vs_prior_parent", "physical", "Parent fixed point, 5D"),
        ("prior_selected_vs_truth_selected", "physical", "Selected truth closure, 5D"),
    )
    for axis, (comparison, group, title) in zip(axes.flat, panels, strict=True):
        part = summary[summary.comparison.eq(comparison) & summary.group.eq(group)]
        axis.plot(part.cycle, part.median_wasserstein_over_iqr, marker="o")
        axis.set_title(title)
        axis.set_ylabel("Median Wasserstein / IQR")
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("EM cycle")
    axes[1, 1].set_xlabel("EM cycle")
    fig.suptitle(
        f"Selection-corrected generalized EM | true weighted alpha={alpha:.4f}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def _plot_population_overlay(
    distributions: dict[str, np.ndarray], path: Path, *, parent: bool
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = PHYSICAL
    labels = (
        (
            ("truth_parent", "True parent", "#202124"),
            ("aggregate_parent", "Posterior aggregate / beta", "#D55E00"),
            ("prior_parent", "Learned parent prior", "#0072B2"),
        )
        if parent
        else (
            ("truth_selected", "True selected", "#202124"),
            ("aggregate_selected", "Posterior aggregate", "#D55E00"),
            ("prior_selected", "Induced selected prior", "#0072B2"),
        )
    )
    fig, axes = plt.subplots(1, len(names), figsize=(17, 3.4))
    for axis, name in zip(axes, names, strict=True):
        index = FENIKS_SPLINE15D_PARAMETERS.index(name)
        combined = np.concatenate(
            [distributions[key][:, index] for key, _, _ in labels]
        )
        low, high = np.quantile(combined, [0.005, 0.995])
        bins = np.linspace(low, high, 55)
        for key, label, color in labels:
            axis.hist(
                distributions[key][:, index],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=label,
                color=color,
            )
        axis.set_title(name.replace("log10_", "").replace("_", " "))
        axis.set_yticks([])
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Parent population" if parent else "Selected population", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def report(inference_root: Path) -> None:
    from scripts.build_feniks_avi_wrapup import plot_corner
    from scripts.feniks_avi_overnight import population_weights
    from scripts.feniks_sbeb_benchmark import (
        _population_corner,
        _posterior_calibration_artifacts,
        _representative_galaxies,
    )

    manifest = read(inference_root / "MANIFEST.json")
    _check_hashes(manifest)
    destination = inference_root / "report"
    destination.mkdir(exist_ok=True)
    inference_truth_path = inference_root / "inference_truth.parquet"
    has_individual_truth = inference_truth_path.is_file()
    if has_individual_truth:
        truth_individual = pd.read_parquet(inference_truth_path).sort_values(
            "row_index"
        )
        individual_types = _representative_galaxies(truth_individual)
        truth_indexed = truth_individual.set_index("row_index")
    else:
        truth_individual = pd.DataFrame()
        individual_types = pd.DataFrame(columns=("type", "row_index"))
        truth_indexed = pd.DataFrame()
    truth_parent = pd.read_parquet(inference_root / "true_parent.parquet")
    truth_selected = pd.read_parquet(inference_root / "true_selected.parquet")
    names = list(FENIKS_SPLINE15D_PARAMETERS)
    count = 65536
    truth_parent_draws = _resample(
        truth_parent[names].to_numpy(np.float64),
        _normalized(truth_parent["population_weight"].to_numpy(np.float64)),
        count=count,
        seed=26091381,
    )
    truth_selected_draws = _resample(
        truth_selected[names].to_numpy(np.float64),
        _normalized(truth_selected["population_weight"].to_numpy(np.float64)),
        count=count,
        seed=26091382,
    )
    distance_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    final_distributions = None
    individual_corner_count = 0
    comparisons = (
        ("aggregate_selected", "prior_selected"),
        ("aggregate_parent", "prior_parent"),
        ("aggregate_selected", "truth_selected"),
        ("prior_selected", "truth_selected"),
        ("aggregate_parent", "truth_parent"),
        ("prior_parent", "truth_parent"),
    )
    for local_cycle, variant in enumerate(manifest["variants"]):
        cycle = int(variant.get("global_cycle", local_cycle))
        arm = inference_root / "arms" / variant["name"]
        _require_final(
            arm / "FINAL.json",
            "INFERENCE_COMPLETE",
            manifest_path=inference_root / "MANIFEST.json",
        )
        parts = sorted((arm / "bank_0").glob("part_*.parquet"))
        if not parts:
            raise FileNotFoundError(f"missing exact bank parts: {arm}")
        bank = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
        selected_weight, parent_weight, support = population_weights(bank)
        values = bank[names].to_numpy(np.float64)
        prior = np.load(arm / "prior_population.npz", allow_pickle=False)
        distributions = {
            "aggregate_selected": _resample(
                values, selected_weight, count=count, seed=26092000 + cycle
            ),
            "aggregate_parent": _resample(
                values, parent_weight, count=count, seed=26093000 + cycle
            ),
            "prior_selected": np.asarray(prior["selected_theta"], dtype=np.float64),
            "prior_parent": np.asarray(prior["theta"], dtype=np.float64),
            "truth_selected": truth_selected_draws,
            "truth_parent": truth_parent_draws,
        }
        cycle_dir = destination / f"cycle_{cycle:02d}"
        _population_corner(
            cycle_dir / "selected_population_corner_5d.png",
            distributions["aggregate_selected"][:8192],
            distributions["prior_selected"][:8192],
            distributions["truth_selected"][:8192],
            title=f"EM cycle {cycle:02d} | selected population",
        )
        _population_corner(
            cycle_dir / "parent_population_corner_5d.png",
            distributions["aggregate_parent"][:8192],
            distributions["prior_parent"][:8192],
            distributions["truth_parent"][:8192],
            title=f"EM cycle {cycle:02d} | recovered parent population",
        )
        raw_path = arm / "raw_0.parquet"
        corrected_path = arm / "is_0.parquet"
        calibration_summary: dict[str, float] = {}
        if has_individual_truth and raw_path.is_file() and corrected_path.is_file():
            raw = pd.read_parquet(raw_path)
            corrected = pd.read_parquet(corrected_path)
            calibration_summary = _posterior_calibration_artifacts(
                corrected,
                truth_individual,
                cycle_dir / "calibration",
                title=f"EM cycle {cycle:02d}",
            )
            for order, galaxy in individual_types.iterrows():
                row_index = int(galaxy.row_index)
                before = raw.loc[raw.row_index.eq(row_index), list(PHYSICAL)].to_numpy(
                    np.float64
                )
                current = corrected.loc[
                    corrected.row_index.eq(row_index), list(PHYSICAL)
                ].to_numpy(np.float64)
                truth_value = truth_indexed.loc[row_index, list(PHYSICAL)].to_numpy(
                    np.float64
                )
                label = str(galaxy["type"]).replace(" ", "_").replace("-", "_")
                plot_corner(
                    cycle_dir
                    / "individual"
                    / f"{order + 1:02d}_{label}_row_{row_index}.png",
                    before,
                    current,
                    truth_value,
                    f"EM cycle {cycle:02d} | {galaxy['type']} | row {row_index}",
                )
                individual_corner_count += 1
        distance_rows.extend(_distance_rows(cycle, distributions, comparisons))
        inference_metrics = pd.read_csv(arm / "metrics.csv")
        support_rows.append(
            {
                "cycle": cycle,
                "median_object_ess_fraction": float(
                    inference_metrics.ess_fraction.median()
                ),
                "q10_object_ess_fraction": float(
                    inference_metrics.ess_fraction.quantile(0.1)
                ),
                "median_object_max_weight": float(
                    inference_metrics.max_weight.median()
                ),
                "selected_aggregate_ess": support["selected_ess"],
                "parent_projection_ess": support["parent_projection_ess"],
                "parent_projection_max_weight": support["parent_projection_max_weight"],
                "prior_alpha": float(np.mean(prior["beta"])),
                **calibration_summary,
            }
        )
        final_distributions = distributions
        del bank, values, prior
    distances = pd.DataFrame(distance_rows)
    distances.to_csv(destination / "fixed_point_distances.csv", index=False)
    summary = (
        distances.groupby(["cycle", "comparison", "group"], as_index=False)
        .wasserstein_over_reference_iqr.median()
        .rename(
            columns={"wasserstein_over_reference_iqr": "median_wasserstein_over_iqr"}
        )
    )
    summary.to_csv(destination / "fixed_point_summary.csv", index=False)
    support = pd.DataFrame(support_rows)
    support.to_csv(destination / "support_by_cycle.csv", index=False)
    true_alpha = float(
        read(
            Path(_manifest(Path(manifest["training"]))["selection_root"])
            / "selection/FINAL.json"
        )["weighted_alpha"]
    )
    _plot_fixed_point(summary, true_alpha, destination / "em_fixed_point.png")
    if final_distributions is None:
        raise RuntimeError("no completed EM inference variants")
    _plot_population_overlay(
        final_distributions, destination / "final_selected_population.png", parent=False
    )
    _plot_population_overlay(
        final_distributions, destination / "final_parent_population.png", parent=True
    )
    final_cycle = int(summary.cycle.max())
    final = summary[summary.cycle.eq(final_cycle)].set_index(["comparison", "group"])
    selected_fixed = float(
        final.loc[
            ("aggregate_selected_vs_prior_selected", "physical"),
            "median_wasserstein_over_iqr",
        ]
    )
    parent_fixed = float(
        final.loc[
            ("aggregate_parent_vs_prior_parent", "physical"),
            "median_wasserstein_over_iqr",
        ]
    )
    report_text = f"""# Selection-corrected AVI generalized EM

## Contract

- E-step: ordinary full-joint 15D importance weights.
- M-step: parent-prior cross entropy plus `log(alpha_eta)`.
- The object-level posterior weights never contain `beta`.
- Truth was isolated until this final report.

## Final fixed point

- Selected aggregate vs induced selected prior, physical 5D median W1/IQR: {selected_fixed:.4f}
- Inverse-beta aggregate vs parent prior, physical 5D median W1/IQR: {parent_fixed:.4f}
- True weighted selection fraction: {true_alpha:.6f}
- Learned final selection fraction: {support.iloc[-1].prior_alpha:.6f}
- Final median object ESS fraction: {support.iloc[-1].median_object_ess_fraction:.6f}

These are convergence diagnostics, not automatic scientific promotion. Inspect
`fixed_point_distances.csv`, `support_by_cycle.csv`, and the two population plots.
"""
    (destination / "REPORT.md").write_text(report_text, encoding="utf-8")
    artifacts = {
        str(path.relative_to(inference_root)): sha(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "FINAL.json"
    }
    write(
        destination / "FINAL.json",
        {
            "status": "AVI_EM_REPORT_COMPLETE",
            "manifest_sha256": sha(inference_root / "MANIFEST.json"),
            "cycles": final_cycle,
            "selected_fixed_point_physical_5d": selected_fixed,
            "parent_fixed_point_physical_5d": parent_fixed,
            "population_corners": 2 * len(manifest["variants"]),
            "individual_corners": individual_corner_count,
            "truth_used_for_training_or_checkpoint_selection": False,
            "artifacts": artifacts,
            "scientific_promotion": False,
        },
    )
    print(report_text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "prepare-components",
            "prepare-mstep",
            "prepare-qstep",
            "finalize-cycle",
            "prepare-inference",
            "infer",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--prior-followup", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--initial-encoder", type=Path)
    parser.add_argument("--initial-prior", type=Path)
    parser.add_argument("--inference-root", type=Path)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--q-epochs", type=int, default=3)
    parser.add_argument("--prior-sweeps", type=int, default=5)
    parser.add_argument(
        "--e-step-mode", choices=("ordinary_iw", "raw_q"), default="ordinary_iw"
    )
    parser.add_argument(
        "--selection-objective",
        choices=("corrected", "naive"),
        default="corrected",
    )
    parser.add_argument("--track", default="em")
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--cycle-offset", type=int, default=0)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.mode == "prepare":
        required = (
            args.source,
            args.training,
            args.prior_followup,
            args.selection,
            args.inference_root,
        )
        if any(value is None for value in required):
            parser.error(
                "prepare requires --source, --training, --prior-followup, "
                "--selection and --inference-root"
            )
        prepare(
            args.source.resolve(),
            args.training.resolve(),
            args.prior_followup.resolve(),
            args.selection.resolve(),
            args.root,
            args.inference_root.resolve(),
            cycles=args.cycles,
            q_epochs=args.q_epochs,
        )
    elif args.mode == "prepare-components":
        required = (
            args.runtime_root,
            args.initial_encoder,
            args.initial_prior,
            args.selection,
            args.inference_root,
        )
        if any(value is None for value in required):
            parser.error(
                "prepare-components requires --runtime-root, --initial-encoder, "
                "--initial-prior, --selection and --inference-root"
            )
        prepare_from_components(
            args.runtime_root.resolve(),
            args.initial_encoder.resolve(),
            args.initial_prior.resolve(),
            args.selection.resolve(),
            args.root,
            args.inference_root.resolve(),
            cycles=args.cycles,
            q_epochs=args.q_epochs,
            prior_sweeps=args.prior_sweeps,
            e_step_mode=args.e_step_mode,
            selection_objective_enabled=args.selection_objective == "corrected",
            track=args.track,
            cycle_offset=args.cycle_offset,
        )
    elif args.mode in {"prepare-mstep", "prepare-qstep", "finalize-cycle"}:
        if args.cycle is None:
            parser.error(f"{args.mode} requires --cycle")
        {
            "prepare-mstep": prepare_mstep,
            "prepare-qstep": prepare_qstep,
            "finalize-cycle": finalize_cycle,
        }[args.mode](args.root, args.cycle)
    elif args.mode == "prepare-inference":
        if args.inference_root is None:
            parser.error("prepare-inference requires --inference-root")
        prepare_inference(
            args.root, args.inference_root.resolve(), particles=args.particles
        )
    elif args.mode == "infer":
        if args.task is None:
            parser.error("infer requires --task")
        infer(args.root, args.task)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
