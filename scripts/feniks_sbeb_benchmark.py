"""Long-form SBEB benchmark across selection cuts and initializations.

The suite keeps truth out of every optimization step.  Canonical truth is
joined only so the dependent reports can evaluate the frozen blind split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import shutil
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kstest, wasserstein_distance

from euclid_dsps.amortized.avi_experiments import Arm
from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from euclid_dsps.observation_arrays import photometry_arrays_from_dataframe
from scripts.feniks_avi_experiments import read, sha, source_config_text, write
from scripts.feniks_avi_next_validation import observed_selection_mask

CUTS = (25, 27, 29)
INITIALIZATIONS = ("warm", "scratch")
E_STEPS = ("raw_q", "ordinary_iw")
SELECTION_OBJECTIVES = ("naive", "corrected")
PHYSICAL = tuple(FENIKS_SPLINE15D_PARAMETERS[:5])
MIN_TRAIN = 4096
MIN_VALIDATION = 512
MIN_BLIND_SELECTED = 1000


def _files_hash(paths) -> dict[str, str]:
    return {str(Path(path).resolve()): sha(path) for path in paths}


def _tree_artifacts(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """Hash every regular report artifact using paths relative to its root."""
    return {
        str(path.relative_to(root)): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }


def _check_hashes(manifest: dict[str, Any]) -> None:
    for path, digest in manifest.get("hashes", {}).items():
        if sha(path) != digest:
            raise ValueError(f"changed SBEB input: {path}")


def _copy_runtime(source: Path, destination: Path) -> None:
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


def _stable_bucket(values: pd.Series) -> np.ndarray:
    def bucket(value: Any) -> int:
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % 10_000

    return np.fromiter((bucket(value) for value in values), np.int64, len(values))


def _weighted_alpha(weights: np.ndarray, selected: np.ndarray) -> float:
    weights = np.asarray(weights, np.float64)
    if (
        not np.isfinite(weights).all()
        or np.any(weights < 0)
        or float(weights.sum()) <= 0
    ):
        raise ValueError("invalid population weights")
    return float(weights[selected].sum() / weights.sum())


def _resolve_warm_components(training: Path, prior: Path) -> tuple[Path, Path]:
    encoder = training / "arms/Q_latest_refresh/encoder.eqx"
    prior_path = prior / "arms/P_latest_prior/prior.eqx"
    for path in (encoder, prior_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    for final_path, status in (
        (encoder.parent / "FINAL.json", "TRAINING_COMPLETE"),
        (prior_path.parent / "FINAL.json", "PRIOR_TRAINING_COMPLETE"),
    ):
        if read(final_path).get("status") != status:
            raise ValueError(f"expected {status}: {final_path}")
    return encoder, prior_path


def _joined_catalog(selection: Path, destination: Path) -> tuple[pd.DataFrame, str]:
    final = read(selection / "selection/FINAL.json")
    parent_catalog = Path(final["parent_catalog"])
    parent_truth = pd.read_parquet(selection / "selection/true_parent.parquet")
    observed = pd.read_parquet(parent_catalog)
    identity = str(final["identity_column"])
    if identity not in observed or identity not in parent_truth:
        raise ValueError(f"selection identity is missing: {identity}")
    truth_columns = [
        identity,
        "population_weight",
        *FENIKS_SPLINE15D_PARAMETERS,
    ]
    if (
        parent_truth[identity].duplicated().any()
        or observed[identity].duplicated().any()
    ):
        raise ValueError("SBEB parent identities must be unique")
    drop = [name for name in FENIKS_SPLINE15D_PARAMETERS if name in observed]
    drop += ["population_weight"] if "population_weight" in observed else []
    observed = observed.drop(columns=drop)
    joined = observed.merge(
        parent_truth[truth_columns], on=identity, how="left", validate="one_to_one"
    )
    if len(joined) != len(parent_truth):
        raise ValueError("observed/truth parent join changed the population")
    values = joined[list(FENIKS_SPLINE15D_PARAMETERS)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite canonical 15D closure truth is required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(destination, index=False)
    return joined, identity


def _selection_config(config: dict[str, Any], cut: int) -> dict[str, Any]:
    config = copy.deepcopy(config)
    objective = config["amortized"]["objective"]
    correction = objective["selection_correction"]
    sleep = objective["sleep"]["selection"]
    for settings in (correction, sleep):
        if settings.get("band") != "lsst_r":
            raise ValueError("SBEB benchmark requires observed lsst_r selection")
        settings["enabled"] = True
        settings["max_mag_ab"] = float(cut)
    config["truth"] = {"parameter_columns": {}}
    config["extra_columns"] = []
    config["amortized"].setdefault("data", {}).update(
        use_redshift_for_split=False,
        stratify_column=None,
        redshift_bins=[],
    )
    return config


def _dummy_teachers(
    frame: pd.DataFrame, rows: np.ndarray, config: dict[str, Any], destination: Path
) -> None:
    chosen = np.asarray(rows[: min(16, len(rows))], np.int64)
    if len(chosen) == 0:
        raise ValueError("cannot build the zero-weight teacher payload")
    observations = photometry_arrays_from_dataframe(
        frame.iloc[chosen], config["bands"]
    )
    np.savez(
        destination / "teachers.npz",
        x=np.zeros((len(chosen), 64, 15), dtype=np.float64),
        flux=observations.flux,
        flux_err=observations.flux_err,
        mask=observations.mask,
        rows=chosen,
    )
    write(
        destination / "teachers.json",
        {
            "enabled": False,
            "role": "shape-valid zero-weight payload; no teacher loss in SBEB arms",
            "rows": chosen.tolist(),
        },
    )


def _runtime_manifest(
    source_manifest: dict[str, Any],
    runtime: Path,
    *,
    cut: int,
    train_rows: int,
    validation_rows: int,
    catalog: Path,
    seed: int,
) -> dict[str, Any]:
    copied = {
        key: source_manifest[key]
        for key in (
            "source",
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
    }
    return {
        **copied,
        "version": 1,
        "suite": "feniks_sbeb_cut_runtime_v1",
        "cut": int(cut),
        "seed": int(seed),
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "validation_catalog": str(catalog.resolve()),
        "prior_macro_objects": 1024,
        "prior_particles": 512,
        "prior_learning_rate": 1.0e-5,
        "prior_trust_strength": 0.2,
        "prior_maximum_kl_per_dimension": 0.02,
        "truth_used": False,
        "scientific_promotion": False,
        "hashes": _files_hash(
            [
                runtime / "source_config.yaml",
                runtime / "train.npy",
                runtime / "validation.npy",
                runtime / "teachers.npz",
                runtime / "teachers.json",
                catalog,
            ]
        ),
    }


def _factor_cells() -> list[dict[str, Any]]:
    cells = []
    for cut in CUTS:
        for initialization in INITIALIZATIONS:
            for e_step in E_STEPS:
                for selection in SELECTION_OBJECTIVES:
                    short = "raw" if e_step == "raw_q" else "iw"
                    cells.append(
                        {
                            "name": f"{initialization}_{short}_{selection}_r{cut}",
                            "cut": cut,
                            "initialization": initialization,
                            "e_step_mode": e_step,
                            "selection_objective": selection,
                        }
                    )
    return cells


def _trajectory_tracks() -> list[dict[str, Any]]:
    tracks = []
    for initialization in INITIALIZATIONS:
        tracks.append(
            {
                "name": f"{initialization}_raw_r29",
                "cut": 29,
                "initialization": initialization,
                "e_step_mode": "raw_q",
            }
        )
        for cut in CUTS:
            tracks.append(
                {
                    "name": f"{initialization}_iw_r{cut}",
                    "cut": cut,
                    "initialization": initialization,
                    "e_step_mode": "ordinary_iw",
                }
            )
    return tracks


def prepare(
    source: Path,
    warm_training: Path,
    warm_prior_root: Path,
    selection: Path,
    nuts_root: Path,
    root: Path,
    *,
    scratch_epochs: int,
    cycles: int,
    q_epochs: int,
    prior_sweeps: int,
) -> None:
    """Freeze all cohorts and immutable contracts before submitting GPUs."""
    from scripts.feniks_avi_experiments import check_inputs

    if root.exists():
        raise FileExistsError(root)
    if scratch_epochs < 60 or q_epochs < 3 or q_epochs % 3 or cycles < 1:
        raise ValueError("scratch>=60, cycles>=1 and q_epochs divisible by 3 required")
    source_manifest = check_inputs(source)
    warm_encoder, warm_prior = _resolve_warm_components(warm_training, warm_prior_root)
    nuts_manifest = read(nuts_root / "MANIFEST.json")
    cases = list(nuts_manifest.get("cases", ()))
    if len(cases) != 8 or len(set(cases)) != 8:
        raise ValueError("the exact eight-case historical NUTS cohort is required")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    combined_path = root / "cohort/observed_with_closure_truth.parquet"
    frame, identity = _joined_catalog(selection, combined_path)
    buckets = _stable_bucket(frame[identity])
    source_config = yaml.safe_load((source / "source_config.yaml").read_text())
    counts = []
    runtime_paths: dict[str, str] = {}
    selection_paths: dict[str, str] = {}

    for cut in CUTS:
        selected, selection_summary = observed_selection_mask(
            frame, band="lsst_r", max_mag_ab=float(cut)
        )
        train = np.flatnonzero(selected & (buckets < 7000)).astype(np.int64)
        validation_all = np.flatnonzero(
            selected & (buckets >= 7000) & (buckets < 8500)
        ).astype(np.int64)
        validation = validation_all[:MIN_VALIDATION]
        blind = np.flatnonzero(buckets >= 8500).astype(np.int64)
        blind_selected = blind[selected[blind]]
        if len(train) < MIN_TRAIN:
            raise ValueError(f"r<{cut} has only {len(train)} training galaxies")
        if len(validation_all) < MIN_VALIDATION:
            raise ValueError(
                f"r<{cut} has only {len(validation_all)} validation galaxies"
            )
        if len(blind_selected) < MIN_BLIND_SELECTED:
            raise ValueError(
                f"r<{cut} has only {len(blind_selected)} blind selected galaxies"
            )

        selection_root = root / "cohorts" / f"r{cut}"
        selection_out = selection_root / "selection"
        selection_out.mkdir(parents=True)
        truth_columns = [
            identity,
            "population_weight",
            *FENIKS_SPLINE15D_PARAMETERS,
        ]
        parent_truth = frame.iloc[blind][truth_columns].copy()
        parent_truth.insert(0, "parent_row_index", blind)
        selected_truth = frame.iloc[blind_selected][truth_columns].copy()
        selected_truth.insert(0, "parent_row_index", blind_selected)
        parent_truth.to_parquet(selection_out / "true_parent.parquet", index=False)
        selected_truth.to_parquet(selection_out / "true_selected.parquet", index=False)
        pd.DataFrame(
            {
                "parent_row_index": np.arange(len(frame), dtype=np.int64),
                identity: frame[identity].to_numpy(),
                "split_bucket": buckets,
                "selected": selected,
            }
        ).to_parquet(selection_out / "selection_identities.parquet", index=False)
        blind_weights = frame.iloc[blind].population_weight.to_numpy(np.float64)
        blind_mask = selected[blind]
        closure = {
            **selection_summary,
            "status": "EXACT_OBSERVED_SELECTION_COMPLETE",
            "parent_objects": int(len(blind)),
            "selected_objects": int(len(blind_selected)),
            "weighted_alpha": _weighted_alpha(blind_weights, blind_mask),
            "full_parent_objects": int(len(frame)),
            "full_selected_objects": int(selected.sum()),
            "training_selected_objects": int(len(train)),
            "validation_selected_objects": int(len(validation_all)),
            "blind_parent_objects": int(len(blind)),
            "blind_selected_objects": int(len(blind_selected)),
            "identity_column": identity,
            "split_contract": "sha256(object_id) modulo 10000: 70/15/15",
            "truth_used_to_select": False,
            "truth_role": "blind dependent closure only",
        }
        write(selection_out / "FINAL.json", closure)

        runtime = root / "runtime" / f"r{cut}"
        runtime.mkdir(parents=True)
        (runtime / "logs").mkdir()
        np.save(runtime / "train.npy", train, allow_pickle=False)
        np.save(runtime / "validation.npy", validation, allow_pickle=False)
        config = _selection_config(source_config, cut)
        config["catalog_path"] = str(combined_path.resolve())
        (runtime / "source_config.yaml").write_text(
            source_config_text(config, source_manifest["source"]["checkpoint"]),
            encoding="utf-8",
        )
        _dummy_teachers(frame, train, config, runtime)
        runtime_manifest = _runtime_manifest(
            source_manifest,
            runtime,
            cut=cut,
            train_rows=len(train),
            validation_rows=len(validation),
            catalog=combined_path,
            seed=26091600 + cut,
        )
        write(runtime / "MANIFEST.json", runtime_manifest)
        runtime_paths[str(cut)] = str(runtime.resolve())
        selection_paths[str(cut)] = str(selection_root.resolve())
        counts.append(
            {
                "cut": cut,
                "full_parent": len(frame),
                "full_selected": int(selected.sum()),
                "train_selected": len(train),
                "validation_selected": len(validation_all),
                "blind_parent": len(blind),
                "blind_selected": len(blind_selected),
                "blind_weighted_alpha": closure["weighted_alpha"],
            }
        )

        bootstrap = root / "bootstrap" / f"r{cut}"
        _copy_runtime(runtime, bootstrap)
        arm = Arm(f"scratch_r{cut}", scratch=True, experts=4)
        bootstrap_manifest = {
            **{
                key: runtime_manifest[key]
                for key in runtime_manifest
                if key not in {"hashes", "suite", "version"}
            },
            "version": 1,
            "suite": "feniks_sbeb_scratch_bootstrap_v1",
            "arms": [asdict(arm)],
            "epochs": int(scratch_epochs),
            "bootstrap_epochs": int(scratch_epochs),
            "cycle": ["sleep", "sleep", "wake"],
            "initial_prior_checkpoint_by_arm": {arm.name: None},
            "prior_frozen": True,
            "decoder_frozen": True,
            "initialization_contract": (
                "random four-expert q; embedded source prior; pure selected sleep"
            ),
            "truth_used": False,
            "scientific_promotion": False,
            "hashes": {
                **_files_hash(
                    [
                        bootstrap / "source_config.yaml",
                        bootstrap / "train.npy",
                        bootstrap / "validation.npy",
                        bootstrap / "teachers.npz",
                        bootstrap / "teachers.json",
                    ]
                )
            },
        }
        write(bootstrap / "MANIFEST.json", bootstrap_manifest)

    pd.DataFrame(counts).to_csv(root / "cohort_counts.csv", index=False)
    manifest = {
        "version": 1,
        "suite": "feniks_long_sbeb_selection_benchmark_v1",
        "source": str(source.resolve()),
        "warm_training": str(warm_training.resolve()),
        "warm_prior_root": str(warm_prior_root.resolve()),
        "warm_encoder": str(warm_encoder.resolve()),
        "warm_prior": str(warm_prior.resolve()),
        "selection_source": str(selection.resolve()),
        "nuts_root": str(nuts_root.resolve()),
        "nuts_cases": cases,
        "combined_catalog": str(combined_path.resolve()),
        "identity_column": identity,
        "cuts": list(CUTS),
        "runtime_roots": runtime_paths,
        "selection_roots": selection_paths,
        "factor_cells": _factor_cells(),
        "trajectory_tracks": _trajectory_tracks(),
        "scratch_epochs": int(scratch_epochs),
        "cycles": int(cycles),
        "q_epochs_per_cycle": int(q_epochs),
        "prior_sweeps_per_cycle": int(prior_sweeps),
        "factor_prior_sweeps": 5,
        "factor_particles": 512,
        "inference_particles": 4096,
        "inference_replicas": 2,
        "split_contract": {
            "train": "bucket < 7000",
            "validation": "7000 <= bucket < 8500",
            "blind": "bucket >= 8500",
            "minimum_train_selected": MIN_TRAIN,
            "minimum_validation_selected": MIN_VALIDATION,
            "minimum_blind_selected": MIN_BLIND_SELECTED,
        },
        "truth_used_for_training_or_checkpoint_selection": False,
        "scientific_promotion": False,
        "hashes": _files_hash(
            [
                source / "MANIFEST.json",
                warm_encoder,
                warm_prior,
                selection / "selection/FINAL.json",
                selection / "selection/true_parent.parquet",
                nuts_root / "MANIFEST.json",
                nuts_root / "OBSERVED_COHORT.csv",
                combined_path,
                root / "cohort_counts.csv",
            ]
        ),
    }
    write(root / "MANIFEST.json", manifest)
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print(f"Prepared long SBEB benchmark: {root}", flush=True)


def _manifest(root: Path) -> dict[str, Any]:
    manifest = read(root / "MANIFEST.json")
    _check_hashes(manifest)
    return manifest


def _encoder_for(manifest: dict[str, Any], root: Path, cell: dict[str, Any]) -> Path:
    if cell["initialization"] == "warm":
        return Path(manifest["warm_encoder"])
    cut = int(cell["cut"])
    return root / f"bootstrap/r{cut}/arms/scratch_r{cut}/encoder.eqx"


def prepare_factor_cell(root: Path, task: int) -> Path:
    manifest = _manifest(root)
    cell = manifest["factor_cells"][task]
    destination = root / "factor/cells" / cell["name"]
    if destination.exists():
        _check_hashes(read(destination / "MANIFEST.json"))
        return destination
    runtime = Path(manifest["runtime_roots"][str(cell["cut"])])
    runtime_manifest = read(runtime / "MANIFEST.json")
    encoder = _encoder_for(manifest, root, cell)
    prior = Path(manifest["warm_prior"])
    for path in (encoder, prior):
        if not path.is_file():
            raise FileNotFoundError(path)
    _copy_runtime(runtime, destination)
    arm = Arm(
        f"P_{cell['name']}",
        experts=4,
        kind="prior",
        prior_initialization=(
            "learned_source"
            if cell["initialization"] == "warm"
            else "identity_standard_normal"
        ),
    )
    stage = {
        **{
            key: runtime_manifest[key]
            for key in runtime_manifest
            if key not in {"hashes", "suite", "version"}
        },
        "version": 1,
        "suite": "feniks_sbeb_frozen_q_factor_cell_v1",
        "cell": cell,
        "arms": [asdict(arm)],
        "epochs": 1,
        "upstream_training": str(runtime.resolve()),
        "upstream_b_encoder": str(encoder.resolve()),
        "initial_prior_checkpoint_by_arm": {
            arm.name: (
                str(prior.resolve()) if cell["initialization"] == "warm" else None
            )
        },
        "prior_sweeps": int(manifest["factor_prior_sweeps"]),
        "prior_macro_objects": 1024,
        "prior_particles": int(manifest["factor_particles"]),
        "prior_learning_rate": 1.0e-5,
        "prior_trust_strength": 0.2,
        "prior_maximum_kl_per_dimension": 0.02,
        "e_step_mode": cell["e_step_mode"],
        "selection_objective_enabled": cell["selection_objective"] == "corrected",
        "selection_in_object_weights": False,
        "truth_used": False,
        "scientific_promotion": False,
        "hashes": _files_hash(
            [
                destination / "source_config.yaml",
                destination / "train.npy",
                destination / "validation.npy",
                destination / "teachers.npz",
                destination / "teachers.json",
                encoder,
                *([prior] if cell["initialization"] == "warm" else []),
            ]
        ),
    }
    write(destination / "MANIFEST.json", stage)
    return destination


def run_bootstrap(root: Path, task: int, *, max_hours: float) -> None:
    from scripts.feniks_avi_experiments import run

    manifest = _manifest(root)
    cut = int(manifest["cuts"][task])
    stage = root / f"bootstrap/r{cut}"
    run(SimpleNamespace(root=stage, task=0, mode="preflight", max_hours=max_hours))
    run(SimpleNamespace(root=stage, task=0, mode="train", max_hours=max_hours))


def run_factor(root: Path, task: int) -> None:
    from scripts.feniks_avi_next_experiments import run_prior

    stage = prepare_factor_cell(root, task)
    run_prior(stage, 0, preflight=True)
    run_prior(stage, 0, preflight=False)


def prepare_factor_inference(root: Path) -> None:
    manifest = _manifest(root)
    for cut in CUTS:
        destination = root / f"factor/inference/r{cut}"
        if destination.exists():
            continue
        runtime = Path(manifest["runtime_roots"][str(cut)])
        runtime_manifest = read(runtime / "MANIFEST.json")
        cells = [cell for cell in manifest["factor_cells"] if cell["cut"] == cut]
        variants = []
        dependencies: list[Path] = [root / "MANIFEST.json"]
        for cell in cells:
            stage = root / "factor/cells" / cell["name"]
            arm = f"P_{cell['name']}"
            final = stage / "arms" / arm / "FINAL.json"
            if read(final).get("status") != "PRIOR_TRAINING_COMPLETE":
                raise ValueError(f"incomplete factor cell: {cell['name']}")
            encoder = _encoder_for(manifest, root, cell)
            prior = stage / "arms" / arm / "prior.eqx"
            variants.append(
                {
                    "name": cell["name"],
                    "encoder": str(encoder.resolve()),
                    "prior": str(prior.resolve()),
                    "prior_label": cell["name"],
                    "emit_prior_population": True,
                }
            )
            dependencies.extend((stage / "MANIFEST.json", final, encoder, prior))
        destination.mkdir(parents=True)
        (destination / "logs").mkdir()
        rows = np.load(runtime / "validation.npy", allow_pickle=False)
        truth = pd.read_parquet(runtime_manifest["validation_catalog"]).iloc[rows]
        truth = truth[[*FENIKS_SPLINE15D_PARAMETERS]].copy()
        truth.insert(0, "row_index", rows)
        truth.insert(0, "object_id", rows)
        truth.to_parquet(destination / "inference_truth.parquet", index=False)
        selection = Path(manifest["selection_roots"][str(cut)]) / "selection"
        shutil.copy2(
            selection / "true_parent.parquet", destination / "true_parent.parquet"
        )
        shutil.copy2(
            selection / "true_selected.parquet", destination / "true_selected.parquet"
        )
        dependencies.extend(
            (
                runtime / "source_config.yaml",
                runtime / "train.npy",
                runtime / "validation.npy",
                destination / "inference_truth.parquet",
                destination / "true_parent.parquet",
                destination / "true_selected.parquet",
            )
        )
        write(
            destination / "MANIFEST.json",
            {
                "version": 1,
                "suite": "feniks_sbeb_factor_inference_v1",
                "cut": cut,
                "training": str(root.resolve()),
                "source": runtime_manifest["source"],
                "config": str((runtime / "source_config.yaml").resolve()),
                "train_indices": str((runtime / "train.npy").resolve()),
                "validation_indices": str((runtime / "validation.npy").resolve()),
                "validation_catalog": runtime_manifest["validation_catalog"],
                "variants": variants,
                "particles": int(manifest["inference_particles"]),
                "saved_draws": 512,
                "replicas": int(manifest["inference_replicas"]),
                "objects": len(rows),
                "seed": 26091640 + cut,
                "prior_samples": 32768,
                "prior_selected_resamples": 32768,
                "selection_in_object_weights": False,
                "truth_used_for_sampling_or_weighting": False,
                "scientific_promotion": False,
                "hashes": _files_hash(dependencies),
            },
        )
    write(
        root / "factor/inference/FINAL.json",
        {"status": "FACTOR_INFERENCE_PREPARED", "cuts": list(CUTS)},
    )


def run_factor_inference(root: Path, task: int) -> None:
    from scripts.feniks_avi_overnight import infer

    cut_index, local_task = divmod(task, 8)
    infer(root / f"factor/inference/r{CUTS[cut_index]}", local_task)


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("invalid weights")
    return values / values.sum()


def _resample(values: np.ndarray, weights: np.ndarray, seed: int, count=8192):
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), count, replace=True, p=_normalized(weights))]


def _population_corner(
    path: Path,
    aggregate: np.ndarray,
    learned: np.ndarray,
    truth: np.ndarray,
    *,
    title: str,
) -> None:
    import corner
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [name.replace("log10_", "").replace("_", " ") for name in PHYSICAL]
    arrays = [aggregate[:, :5], learned[:, :5], truth[:, :5]]
    ranges = []
    for index in range(5):
        joined = np.concatenate([value[:, index] for value in arrays])
        low, high = np.quantile(joined, [0.005, 0.995])
        ranges.append((float(low), float(high)))
    figure = corner.corner(
        truth[:, :5],
        labels=labels,
        color="#202124",
        range=ranges,
        bins=30,
        smooth=1.0,
        plot_datapoints=False,
        levels=(0.5, 0.8, 0.95),
        hist_kwargs={"linewidth": 1.8},
        contour_kwargs={"linewidths": 1.6},
        max_n_ticks=4,
    )
    for values, color, linestyle in (
        (aggregate, "#D55E00", "-"),
        (learned, "#0072B2", "--"),
    ):
        corner.corner(
            values[:, :5],
            fig=figure,
            color=color,
            range=ranges,
            bins=30,
            smooth=1.0,
            plot_datapoints=False,
            levels=(0.5, 0.8, 0.95),
            hist_kwargs={"linewidth": 1.6, "linestyle": linestyle},
            contour_kwargs={"linewidths": 1.4, "linestyles": linestyle},
            max_n_ticks=4,
        )
    figure.legend(
        handles=[
            plt.Line2D([], [], color="#202124", label="True population"),
            plt.Line2D([], [], color="#D55E00", label="Posterior aggregate"),
            plt.Line2D([], [], color="#0072B2", linestyle="--", label="Learned prior"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=False,
    )
    figure.suptitle(title, fontsize=14, y=1.01)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _representative_galaxies(truth: pd.DataFrame) -> pd.DataFrame:
    """Choose the eight scientific strata, with a small-fixture fallback."""
    from scripts.build_feniks_avi_wrapup import _select_galaxy_types

    if len(truth) >= 8:
        return _select_galaxy_types(truth)
    return pd.DataFrame(
        {
            "type": [f"fixture object {index}" for index in range(len(truth))],
            "row_index": truth.row_index.to_numpy(np.int64),
        }
    )


def _posterior_calibration_artifacts(
    draws: pd.DataFrame,
    truth: pd.DataFrame,
    destination: Path,
    *,
    title: str,
) -> dict[str, float]:
    """Write physical PIT, coverage and truth-versus-posterior plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth_indexed = truth.set_index("row_index")
    grouped = draws.groupby("row_index", sort=True)
    rows = np.asarray(sorted(grouped.groups), np.int64)
    if not set(rows).issubset(set(truth_indexed.index)):
        raise ValueError("posterior and truth identities differ")
    counts = grouped.size().to_numpy()
    if len(counts) == 0 or np.unique(counts).size != 1:
        raise ValueError("dense equal-sized posterior draws are required")
    ordered = draws.sort_values(["row_index", "sample_id"])
    cube = (
        ordered[list(PHYSICAL)]
        .to_numpy(np.float64)
        .reshape(len(rows), int(counts[0]), len(PHYSICAL))
    )
    target = truth_indexed.loc[rows, list(PHYSICAL)].to_numpy(np.float64)
    pit = np.mean(cube <= target[:, None, :], axis=1)
    records = []
    for index, name in enumerate(PHYSICAL):
        ks = kstest(pit[:, index], "uniform")
        record = {
            "parameter": name,
            "pit_mean": float(pit[:, index].mean()),
            "pit_ks": float(ks.statistic),
            "pit_ks_pvalue": float(ks.pvalue),
        }
        for level in (0.68, 0.95):
            tail = (1.0 - level) / 2.0
            low = np.quantile(cube[:, :, index], tail, axis=1)
            high = np.quantile(cube[:, :, index], 1.0 - tail, axis=1)
            record[f"coverage_{int(level * 100)}"] = float(
                np.mean((target[:, index] >= low) & (target[:, index] <= high))
            )
        records.append(record)
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(
        destination / "posterior_calibration_5d.csv", index=False
    )

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for index, (axis, name) in enumerate(zip(axes, PHYSICAL, strict=True)):
        axis.hist(pit[:, index], bins=bins, density=True, color="#3B6EA8", alpha=0.82)
        axis.axhline(1.0, color="#202124", linestyle="--", linewidth=1)
        axis.set_title(name.replace("log10_", "").replace("_", " "))
        axis.set_xlabel("PIT")
    axes[0].set_ylabel("Density")
    fig.suptitle(f"{title} | physical posterior PIT", fontsize=13)
    fig.tight_layout()
    fig.savefig(destination / "posterior_pit_5d.png", dpi=180)
    plt.close(fig)

    median = np.median(cube, axis=1)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    for index, (axis, name) in enumerate(zip(axes, PHYSICAL, strict=True)):
        axis.scatter(
            target[:, index], median[:, index], s=8, alpha=0.35, color="#D55E00"
        )
        low = float(min(target[:, index].min(), median[:, index].min()))
        high = float(max(target[:, index].max(), median[:, index].max()))
        axis.plot([low, high], [low, high], color="#202124", linestyle="--")
        axis.set_title(name.replace("log10_", "").replace("_", " "))
        axis.set_xlabel("Truth")
        axis.set_ylabel("Posterior median")
    fig.suptitle(f"{title} | truth versus posterior", fontsize=13)
    fig.tight_layout()
    fig.savefig(destination / "truth_vs_posterior_5d.png", dpi=180)
    plt.close(fig)
    result = pd.DataFrame(records)
    return {
        "mean_physical_pit_ks": float(result.pit_ks.mean()),
        "mean_physical_coverage_68": float(result.coverage_68.mean()),
        "mean_physical_coverage_95": float(result.coverage_95.mean()),
    }


def _prior_truth_inclusion(learned: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pit = np.column_stack(
        [
            np.searchsorted(np.sort(learned[:, index]), truth[:, index], side="right")
            / len(learned)
            for index in range(5)
        ]
    )
    ks = [kstest(pit[:, index], "uniform").statistic for index in range(5)]
    return {
        "mean_prior_truth_pit_ks": float(np.mean(ks)),
        "central_95_truth_in_prior": float(np.mean((pit >= 0.025) & (pit <= 0.975))),
    }


def factor_report(root: Path) -> None:
    from scripts.build_feniks_avi_wrapup import plot_corner
    from scripts.feniks_avi_overnight import population_weights

    manifest = _manifest(root)
    destination = root / "factor/report"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for cut in CUTS:
        inference = root / f"factor/inference/r{cut}"
        inference_manifest = read(inference / "MANIFEST.json")
        individual_truth = pd.read_parquet(inference / "inference_truth.parquet")
        individual_types = _representative_galaxies(individual_truth)
        truth_indexed = individual_truth.set_index("row_index")
        truth_parent = pd.read_parquet(inference / "true_parent.parquet")
        truth_selected = pd.read_parquet(inference / "true_selected.parquet")
        names = list(FENIKS_SPLINE15D_PARAMETERS)
        tp = _resample(
            truth_parent[names].to_numpy(np.float64),
            truth_parent.population_weight.to_numpy(np.float64),
            26091700 + cut,
        )
        ts = _resample(
            truth_selected[names].to_numpy(np.float64),
            truth_selected.population_weight.to_numpy(np.float64),
            26091800 + cut,
        )
        for index, variant in enumerate(inference_manifest["variants"]):
            arm = inference / "arms" / variant["name"]
            if read(arm / "FINAL.json").get("status") != "INFERENCE_COMPLETE":
                raise ValueError(f"incomplete factor inference: {variant['name']}")
            metrics = pd.read_csv(arm / "metrics.csv")
            parts = sorted((arm / "bank_0").glob("part_*.parquet"))
            bank = pd.concat(
                [pd.read_parquet(path) for path in parts], ignore_index=True
            )
            selected_weight, parent_weight, support = population_weights(bank)
            values = bank[names].to_numpy(np.float64)
            selected_aggregate = _resample(
                values, selected_weight, 26091900 + 100 * cut + index
            )
            parent_aggregate = _resample(
                values, parent_weight, 26092000 + 100 * cut + index
            )
            prior = np.load(arm / "prior_population.npz", allow_pickle=False)
            learned_parent = np.asarray(prior["theta"], np.float64)
            learned_selected = np.asarray(prior["selected_theta"], np.float64)
            cell = next(
                value
                for value in manifest["factor_cells"]
                if value["name"] == variant["name"]
            )
            record = {
                **cell,
                "median_ess_fraction": float(metrics.ess_fraction.median()),
                "q10_ess_fraction": float(metrics.ess_fraction.quantile(0.1)),
                "median_max_weight": float(metrics.max_weight.median()),
                "aggregate_selected_ess": float(support["selected_ess"]),
                "aggregate_parent_ess": float(support["parent_projection_ess"]),
                "learned_alpha": float(np.mean(prior["beta"])),
            }
            record.update(
                {
                    f"selected_{key}": value
                    for key, value in _prior_truth_inclusion(
                        learned_selected, ts
                    ).items()
                }
            )
            record.update(
                {
                    f"parent_{key}": value
                    for key, value in _prior_truth_inclusion(learned_parent, tp).items()
                }
            )
            for population, estimate, truth in (
                ("selected", selected_aggregate, ts),
                ("parent", parent_aggregate, tp),
            ):
                distances = []
                for parameter in range(5):
                    scale = max(
                        float(
                            np.quantile(truth[:, parameter], 0.75)
                            - np.quantile(truth[:, parameter], 0.25)
                        ),
                        1.0e-6,
                    )
                    distances.append(
                        wasserstein_distance(
                            estimate[:, parameter], truth[:, parameter]
                        )
                        / scale
                    )
                record[f"{population}_physical_w1_over_iqr"] = float(
                    np.median(distances)
                )
            plot_root = destination / "population" / variant["name"]
            _population_corner(
                plot_root / "selected_corner_5d.png",
                selected_aggregate,
                learned_selected,
                ts,
                title=f"{variant['name']} | selected r<{cut}",
            )
            _population_corner(
                plot_root / "parent_corner_5d.png",
                parent_aggregate,
                learned_parent,
                tp,
                title=f"{variant['name']} | parent recovered from r<{cut}",
            )
            raw = pd.read_parquet(arm / "raw_0.parquet")
            corrected = pd.read_parquet(arm / "is_0.parquet")
            record.update(
                _posterior_calibration_artifacts(
                    corrected,
                    individual_truth,
                    destination / "calibration" / variant["name"],
                    title=variant["name"],
                )
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
                    destination
                    / "individual"
                    / variant["name"]
                    / f"{order + 1:02d}_{label}_row_{row_index}.png",
                    before,
                    current,
                    truth_value,
                    f"{variant['name']} | {galaxy['type']} | row {row_index}",
                )
            rows.append(record)
    result = pd.DataFrame(rows)
    result.to_csv(destination / "factor_scorecard.csv", index=False)
    write(
        destination / "FINAL.json",
        {
            "status": "SBEB_FACTOR_REPORT_COMPLETE",
            "cells": len(result),
            "population_corners": len(result) * 2,
            "individual_corners": len(result) * len(individual_types),
            "truth_used_for_training_or_checkpoint_selection": False,
            "scorecard_sha256": sha(destination / "factor_scorecard.csv"),
            "artifacts": _tree_artifacts(destination, exclude=("FINAL.json",)),
        },
    )
    print(result.to_string(index=False), flush=True)


def prepare_trajectories(root: Path) -> None:
    from scripts.feniks_avi_em import prepare_from_components

    manifest = _manifest(root)
    for track in manifest["trajectory_tracks"]:
        runtime = Path(manifest["runtime_roots"][str(track["cut"])])
        encoder = _encoder_for(manifest, root, track)
        short = "raw" if track["e_step_mode"] == "raw_q" else "iw"
        factor_name = (
            f"{track['initialization']}_{short}_corrected_r{track['cut']}"
        )
        factor_stage = root / "factor/cells" / factor_name
        factor_arm = factor_stage / "arms" / f"P_{factor_name}"
        factor_final = factor_arm / "FINAL.json"
        if read(factor_final).get("status") != "PRIOR_TRAINING_COMPLETE":
            raise ValueError(f"incomplete corrected factor prior: {factor_name}")
        prior = factor_arm / "prior.eqx"
        training = root / "trajectories" / track["name"] / "em"
        inference = root / "trajectories" / track["name"] / "inference"
        if training.exists():
            continue
        prepare_from_components(
            runtime,
            encoder,
            prior,
            Path(manifest["selection_roots"][str(track["cut"])]),
            training,
            inference,
            cycles=int(manifest["cycles"]),
            q_epochs=int(manifest["q_epochs_per_cycle"]),
            prior_sweeps=int(manifest["prior_sweeps_per_cycle"]),
            e_step_mode=track["e_step_mode"],
            selection_objective_enabled=True,
            track=track["name"],
        )
    write(
        root / "trajectories/PREPARED.json",
        {
            "status": "SBEB_TRAJECTORIES_PREPARED",
            "tracks": [track["name"] for track in manifest["trajectory_tracks"]],
        },
    )


def run_em_cycle(root: Path, task: int, cycle: int, *, max_hours: float) -> None:
    from scripts.feniks_avi_em import (
        finalize_cycle,
        prepare_mstep,
        prepare_qstep,
    )
    from scripts.feniks_avi_experiments import run as run_encoder
    from scripts.feniks_avi_next_experiments import run_prior

    manifest = _manifest(root)
    track = manifest["trajectory_tracks"][task]
    training = root / "trajectories" / track["name"] / "em"
    mstep = prepare_mstep(training, cycle)
    run_prior(mstep, 0, preflight=True)
    run_prior(mstep, 0, preflight=False)
    qstep = prepare_qstep(training, cycle)
    run_encoder(
        SimpleNamespace(root=qstep, task=0, mode="preflight", max_hours=max_hours)
    )
    run_encoder(SimpleNamespace(root=qstep, task=0, mode="train", max_hours=max_hours))
    finalize_cycle(training, cycle)


def prepare_endpoint_factorials(root: Path) -> None:
    """Freeze Q0/Q4 x P0/P4 contracts for every completed trajectory."""
    from scripts.feniks_avi_em_factorial import prepare as prepare_factorial

    manifest = _manifest(root)
    for track in manifest["trajectory_tracks"]:
        base = root / "trajectories" / track["name"]
        destination = base / "endpoint_factorial"
        if destination.exists():
            continue
        prepare_factorial(
            base / "em",
            destination,
            particles=int(manifest["inference_particles"]),
        )
    write(
        root / "trajectories/ENDPOINT_FACTORIALS_PREPARED.json",
        {
            "status": "SBEB_ENDPOINT_FACTORIALS_PREPARED",
            "tracks": [track["name"] for track in manifest["trajectory_tracks"]],
        },
    )


def run_endpoint_factorial_inference(root: Path, task: int) -> None:
    from scripts.feniks_avi_em_factorial import infer

    manifest = _manifest(root)
    track_index, variant = divmod(task, 4)
    track = manifest["trajectory_tracks"][track_index]
    infer(root / "trajectories" / track["name"] / "endpoint_factorial", variant)


def run_endpoint_factorial_report(root: Path, task: int) -> None:
    from scripts.feniks_avi_em_factorial import report

    manifest = _manifest(root)
    track = manifest["trajectory_tracks"][task]
    report(root / "trajectories" / track["name"] / "endpoint_factorial")


def prepare_em_inference(root: Path) -> None:
    from scripts.feniks_avi_em import prepare_inference

    manifest = _manifest(root)
    for track in manifest["trajectory_tracks"]:
        base = root / "trajectories" / track["name"]
        if (base / "inference").exists():
            continue
        prepare_inference(
            base / "em",
            base / "inference",
            particles=int(manifest["inference_particles"]),
        )
    write(
        root / "trajectories/INFERENCE_PREPARED.json",
        {"status": "SBEB_TRAJECTORY_INFERENCE_PREPARED"},
    )


def run_em_inference(root: Path, task: int) -> None:
    from scripts.feniks_avi_em import infer

    manifest = _manifest(root)
    variants = int(manifest["cycles"]) + 1
    track_index, variant = divmod(task, variants)
    track = manifest["trajectory_tracks"][track_index]
    infer(root / "trajectories" / track["name"] / "inference", variant)


def run_em_report(root: Path, task: int) -> None:
    from scripts.feniks_avi_em import report

    manifest = _manifest(root)
    track = manifest["trajectory_tracks"][task]
    report(root / "trajectories" / track["name"] / "inference")


def prepare_nuts_inference(root: Path, task: int) -> Path:
    from scripts.feniks_avi_em import _components

    manifest = _manifest(root)
    track = manifest["trajectory_tracks"][task]
    destination = root / "nuts" / track["name"]
    if destination.exists():
        return destination
    runtime = Path(manifest["runtime_roots"][str(track["cut"])])
    runtime_manifest = read(runtime / "MANIFEST.json")
    training = root / "trajectories" / track["name"] / "em"
    encoder, prior = _components(training, int(manifest["cycles"]))
    nuts_root = Path(manifest["nuts_root"])
    dependencies = [
        root / "MANIFEST.json",
        runtime / "source_config.yaml",
        runtime / "train.npy",
        runtime / "validation.npy",
        encoder,
        prior,
        nuts_root / "MANIFEST.json",
        nuts_root / "OBSERVED_COHORT.csv",
    ]
    for case in manifest["nuts_cases"]:
        dependencies.append(nuts_root / case / "observation.npz")
    destination.mkdir(parents=True)
    (destination / "logs").mkdir()
    write(
        destination / "MANIFEST.json",
        {
            "version": 1,
            "suite": "feniks_sbeb_final_nuts_inference_v1",
            "track": track,
            "nuts_root": str(nuts_root.resolve()),
            "config": str((runtime / "source_config.yaml").resolve()),
            "train_indices": str((runtime / "train.npy").resolve()),
            "validation_indices": str((runtime / "validation.npy").resolve()),
            "validation_catalog": runtime_manifest["validation_catalog"],
            "feature_stats": runtime_manifest["source"]["feature_stats"],
            "source_checkpoint": runtime_manifest["source"]["checkpoint"],
            "encoder": str(encoder.resolve()),
            "latest_prior": str(prior.resolve()),
            "cases": manifest["nuts_cases"],
            "particles": int(manifest["inference_particles"]),
            "replicas": int(manifest["inference_replicas"]),
            "seed": 26092100 + task,
            "encoder_template_seed": int(runtime_manifest["seed"]),
            "truth_used_for_sampling_or_weighting": False,
            "scientific_promotion": False,
            "hashes": _files_hash(dependencies),
        },
    )
    return destination


def run_nuts_inference(root: Path, task: int) -> None:
    from scripts.feniks_avi_next_validation import infer

    destination = prepare_nuts_inference(root, task)
    infer(destination, preflight=False)


def final_report(root: Path) -> None:
    from scripts.build_feniks_avi_wrapup import (
        _match_nuts_truth,
        plot_nuts_corner,
    )
    from scripts.feniks_avi_next_validation import _load_nuts_x

    manifest = _manifest(root)
    destination = root / "report"
    destination.mkdir(parents=True, exist_ok=True)
    factor = pd.read_csv(root / "factor/report/factor_scorecard.csv")
    factor.to_csv(destination / "factor_scorecard.csv", index=False)
    trajectory_rows = []
    endpoint_cells = []
    endpoint_effects = []
    for track_index, track in enumerate(manifest["trajectory_tracks"]):
        inference = root / "trajectories" / track["name"] / "inference"
        report_final = read(inference / "report/FINAL.json")
        if report_final.get("status") != "AVI_EM_REPORT_COMPLETE":
            raise ValueError(f"incomplete trajectory report: {track['name']}")
        endpoint = root / "trajectories" / track["name"] / "endpoint_factorial"
        endpoint_final = read(endpoint / "report/FACTORIAL_FINAL.json")
        if endpoint_final.get("status") != "AVI_EM_FACTORIAL_REPORT_COMPLETE":
            raise ValueError(f"incomplete endpoint factorial: {track['name']}")
        cells = pd.read_csv(endpoint / "report/factorial_cells.csv")
        cells.insert(0, "track", track["name"])
        endpoint_cells.append(cells)
        effects = pd.read_csv(endpoint / "report/factorial_effects.csv")
        effects.insert(0, "track", track["name"])
        endpoint_effects.append(effects)
        support = pd.read_csv(inference / "report/support_by_cycle.csv")
        fixed = pd.read_csv(inference / "report/fixed_point_summary.csv")
        for row in support.itertuples(index=False):
            trajectory_rows.append(
                {
                    **track,
                    "cycle": int(row.cycle),
                    "median_ess_fraction": float(row.median_object_ess_fraction),
                    "q10_ess_fraction": float(row.q10_object_ess_fraction),
                    "learned_alpha": float(row.prior_alpha),
                    "selected_fixed_point_5d": float(
                        fixed[
                            fixed.cycle.eq(row.cycle)
                            & fixed.comparison.eq(
                                "aggregate_selected_vs_prior_selected"
                            )
                            & fixed.group.eq("physical")
                        ].median_wasserstein_over_iqr.iloc[0]
                    ),
                    "parent_truth_5d": float(
                        fixed[
                            fixed.cycle.eq(row.cycle)
                            & fixed.comparison.eq("aggregate_parent_vs_truth_parent")
                            & fixed.group.eq("physical")
                        ].median_wasserstein_over_iqr.iloc[0]
                    ),
                }
            )
        nuts = root / "nuts" / track["name"]
        bank_path = nuts / "inference/q_latest_nuts_bank.npz"
        if not bank_path.is_file():
            raise FileNotFoundError(bank_path)
        truth_dir = destination / "nuts_truth" / track["name"]
        (truth_dir / "tables").mkdir(parents=True, exist_ok=True)
        runtime_config = (
            Path(manifest["runtime_roots"][str(track["cut"])]) / "source_config.yaml"
        )
        truth_by_case = _match_nuts_truth(
            Path(manifest["nuts_root"]), runtime_config, truth_dir
        )
        import jax.numpy as jnp
        import yaml

        from euclid_dsps.amortized.latent import latent_spec_from_config, x_to_theta

        spec = latent_spec_from_config(yaml.safe_load(runtime_config.read_text()))
        with np.load(bank_path, allow_pickle=False) as archive:
            theta = np.asarray(archive["theta"], np.float64)
            weights = np.asarray(archive["source_weight"], np.float64)
        for case_index, case in enumerate(manifest["nuts_cases"]):
            encoder = _resample(
                theta[0, case_index],
                weights[0, case_index],
                26092200 + 100 * track_index + case_index,
                count=4096,
            )
            nuts_theta = _load_nuts_x(Path(manifest["nuts_root"]), case)
            # The historical NUTS archive is already in latent x; the plotting
            # helper used by the validation suite performs the x->theta mapping.
            nuts_theta = np.asarray(
                x_to_theta(jnp.asarray(nuts_theta), spec), np.float64
            )
            rng = np.random.default_rng(26092300 + case_index)
            nuts_theta = nuts_theta[
                rng.choice(len(nuts_theta), 4096, replace=len(nuts_theta) < 4096)
            ]
            truth_values = np.asarray(truth_by_case[case], np.float64)
            plot_nuts_corner(
                destination / "nuts" / track["name"] / f"{case}.png",
                encoder,
                nuts_theta,
                truth_values,
                f"{track['name']} | {case} | source-target comparison",
            )
    trajectories = pd.DataFrame(trajectory_rows)
    trajectories.to_csv(destination / "trajectory_scorecard.csv", index=False)
    pd.concat(endpoint_cells, ignore_index=True).to_csv(
        destination / "endpoint_factorial_cells.csv", index=False
    )
    pd.concat(endpoint_effects, ignore_index=True).to_csv(
        destination / "endpoint_factorial_effects.csv", index=False
    )
    best = factor.sort_values(
        ["parent_physical_w1_over_iqr", "median_ess_fraction"],
        ascending=[True, False],
    ).iloc[0]
    report = f"""# FENIKS long SBEB benchmark

## Contract

- Cuts: r < 25, 27 and 29 on saved noisy observed flux.
- Initialization: warm q/learned prior versus a 180-epoch random q/identity prior start.
- Frozen-q factor: raw-q / ordinary-IW by naive / selection-corrected M-step.
- Scientific trajectories: four complete EM cycles, five prior sweeps and 24 q epochs per cycle.
- Truth is used only in the dependent blind report.

## Outputs

- Factor scorecard: `factor_scorecard.csv`.
- Per-cycle trajectory scorecard: `trajectory_scorecard.csv`.
- Endpoint Q0/Q4 x P0/P4 cells and effects: `endpoint_factorial_cells.csv`
  and `endpoint_factorial_effects.csv`.
- Population corners: `../factor/report/population/` and each trajectory report.
- Target-compatible Encoder/NUTS/Truth corners: `nuts/`.

The best frozen-q parent closure cell by median physical W1/IQR is
`{best["name"]}` ({best["parent_physical_w1_over_iqr"]:.4f}). This is a
diagnostic ranking, not an automatic scientific promotion.
"""
    (destination / "REPORT.md").write_text(report, encoding="utf-8")
    write(
        destination / "FINAL.json",
        {
            "status": "FENIKS_SBEB_BENCHMARK_COMPLETE",
            "factor_cells": len(factor),
            "trajectory_rows": len(trajectories),
            "endpoint_factorial_cells": sum(len(frame) for frame in endpoint_cells),
            "nuts_corners": len(manifest["trajectory_tracks"])
            * len(manifest["nuts_cases"]),
            "truth_used_for_training_or_checkpoint_selection": False,
            "factor_scorecard_sha256": sha(destination / "factor_scorecard.csv"),
            "trajectory_scorecard_sha256": sha(
                destination / "trajectory_scorecard.csv"
            ),
            "artifacts": _tree_artifacts(destination, exclude=("FINAL.json",)),
        },
    )
    print(report, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "bootstrap",
            "factor",
            "prepare-factor-inference",
            "factor-infer",
            "factor-report",
            "prepare-trajectories",
            "em-cycle",
            "prepare-endpoint-factorials",
            "endpoint-factorial-infer",
            "endpoint-factorial-report",
            "prepare-em-inference",
            "em-infer",
            "em-report",
            "nuts-infer",
            "report",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--warm-training", type=Path)
    parser.add_argument("--warm-prior", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--nuts-root", type=Path)
    parser.add_argument("--task", type=int)
    parser.add_argument("--cycle", type=int)
    parser.add_argument("--scratch-epochs", type=int, default=180)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--q-epochs", type=int, default=24)
    parser.add_argument("--prior-sweeps", type=int, default=5)
    parser.add_argument("--max-hours", type=float, default=18.0)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.mode == "prepare":
        required = (
            args.source,
            args.warm_training,
            args.warm_prior,
            args.selection,
            args.nuts_root,
        )
        if any(path is None for path in required):
            parser.error(
                "prepare requires --source, --warm-training, --warm-prior, "
                "--selection and --nuts-root"
            )
        prepare(
            args.source.resolve(),
            args.warm_training.resolve(),
            args.warm_prior.resolve(),
            args.selection.resolve(),
            args.nuts_root.resolve(),
            args.root,
            scratch_epochs=args.scratch_epochs,
            cycles=args.cycles,
            q_epochs=args.q_epochs,
            prior_sweeps=args.prior_sweeps,
        )
        return
    if (
        args.mode
        in {
            "bootstrap",
            "factor",
            "factor-infer",
            "em-cycle",
            "endpoint-factorial-infer",
            "endpoint-factorial-report",
            "em-infer",
            "em-report",
            "nuts-infer",
        }
        and args.task is None
    ):
        parser.error(f"{args.mode} requires --task")
    if args.mode == "bootstrap":
        run_bootstrap(args.root, args.task, max_hours=args.max_hours)
    elif args.mode == "factor":
        run_factor(args.root, args.task)
    elif args.mode == "prepare-factor-inference":
        prepare_factor_inference(args.root)
    elif args.mode == "factor-infer":
        run_factor_inference(args.root, args.task)
    elif args.mode == "factor-report":
        factor_report(args.root)
    elif args.mode == "prepare-trajectories":
        prepare_trajectories(args.root)
    elif args.mode == "em-cycle":
        if args.cycle is None:
            parser.error("em-cycle requires --cycle")
        run_em_cycle(args.root, args.task, args.cycle, max_hours=args.max_hours)
    elif args.mode == "prepare-endpoint-factorials":
        prepare_endpoint_factorials(args.root)
    elif args.mode == "endpoint-factorial-infer":
        run_endpoint_factorial_inference(args.root, args.task)
    elif args.mode == "endpoint-factorial-report":
        run_endpoint_factorial_report(args.root, args.task)
    elif args.mode == "prepare-em-inference":
        prepare_em_inference(args.root)
    elif args.mode == "em-infer":
        run_em_inference(args.root, args.task)
    elif args.mode == "em-report":
        run_em_report(args.root, args.task)
    elif args.mode == "nuts-infer":
        run_nuts_inference(args.root, args.task)
    else:
        final_report(args.root)


if __name__ == "__main__":
    main()
