#!/usr/bin/env python3
"""Bounded, truth-free continuation and matched validation of a frozen parent."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Also support direct execution without relying on an inherited PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml

from euclid_dsps.amortized.population_vem import require_git_commit, sha256_file
from euclid_dsps.config import load_config
from scripts.prepare_feniks_sc_drws_topology_npe_pilot import (
    _representative_observed_rows,
)

ARMS = ("A", "B", "C", "D")


def read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def verified_artifacts(receipt: dict) -> dict:
    for key in ("checkpoint", "config", "feature_stats"):
        if sha256_file(Path(receipt[key])) != receipt[key + "_sha256"]:
            raise ValueError(f"artifact changed: {receipt[key]}")
    return receipt


def split_confirmation(validation, development, train, count):
    validation = np.asarray(validation, dtype=np.int64)
    if (
        len(np.unique(validation)) != len(validation)
        or np.intersect1d(validation, train).size
    ):
        raise ValueError("training/validation identities must be unique and disjoint")
    available = np.setdiff1d(validation, development)
    if count <= 0 or len(available) < count:
        raise ValueError("not enough validation rows for reserved confirmation")
    return available


def prepare(source_root: Path, root: Path, recipe_path: Path, repo: Path) -> dict:
    source_root, root = source_root.resolve(), root.resolve()
    if root.exists():
        raise FileExistsError(f"preserve existing experiment; use a new root: {root}")
    recipe = yaml.safe_load(recipe_path.read_text())
    if recipe["observed_weights"] != {"B": 0.0, "C": 0.0001, "D": 0.001}:
        raise ValueError("this bounded experiment has three prespecified arms")
    old = read(source_root / "RUN_MANIFEST.json")
    source = verified_artifacts(read(source_root / "arms/B/ARM_COMPLETE.json"))
    historical = verified_artifacts(old["source"])
    if (
        source.get("status") != "COMPLETE"
        or not source.get("prior_bitwise_unchanged")
        or source.get("truth_used_for_training_or_checkpoint_selection") is not False
        or source["topology"]["minimum_transform_count"] < 2
    ):
        raise ValueError("source must be a certified corrected frozen-parent B")
    config = load_config(source["config"])
    if config.get("truth", {}).get("parameter_columns"):
        raise ValueError("catalogue truth is forbidden")
    objective = config["amortized"]["objective"]
    if (
        objective.get("mode") != "reweighted_wake_sleep"
        or not objective.get("sleep", {}).get("enabled")
        or objective.get("wake", {}).get("train_encoder") is not False
        or objective.get("wake", {}).get("train_prior") is not False
        or config["amortized"]["prior"].get("train_jointly") is not False
        or objective.get("npe_weight", 0) != 0
        or objective.get("prior_truth_weight", 0) != 0
    ):
        raise ValueError(
            "source must be pure simulation sleep with frozen prior and no teacher"
        )
    dataset = Path(old["dataset"]["path"])
    if sha256_file(dataset) != old["dataset"]["sha256"]:
        raise ValueError("source dataset changed")
    config["catalog_path"] = str(dataset)
    cohorts = {
        k: np.load(old["cohorts"][k]["path"], allow_pickle=False)
        for k in ("train", "validation", "validation_pilot", "support_pilot")
    }
    if (
        np.setdiff1d(cohorts["validation_pilot"], cohorts["validation"]).size
        or np.setdiff1d(cohorts["support_pilot"], cohorts["validation_pilot"]).size
    ):
        raise ValueError("pilot/support must remain subsets of observed validation")
    if len(cohorts["validation_pilot"]) != recipe["validation_objects"]:
        raise ValueError("development cohort size differs from recipe")
    if len(cohorts["support_pilot"]) != recipe["support_objects"]:
        raise ValueError("support cohort size differs from recipe")
    available = split_confirmation(
        cohorts["validation"],
        cohorts["validation_pilot"],
        cohorts["train"],
        recipe["confirmation_objects"],
    )
    confirmation, audit = _representative_observed_rows(
        config, available, count=recipe["confirmation_objects"]
    )
    cohorts["confirmation"] = confirmation
    if len(cohorts["train"]) < 2:
        raise ValueError("two training rows are required for the SED smoke")
    cohorts["smoke"] = cohorts["train"][:2]
    cohorts["confirmation_support"] = confirmation[
        np.rint(
            np.linspace(0, len(confirmation) - 1, recipe["support_objects"])
        ).astype(int)
    ]
    cohorts["validation"] = np.setdiff1d(cohorts["validation"], confirmation)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        (staging / "manifests").mkdir()
        cohort_manifest = {}
        for name, values in cohorts.items():
            path = staging / "manifests" / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            cohort_manifest[name] = {
                "path": str(root / path.relative_to(staging)),
                "sha256": sha256_file(path),
                "objects": len(values),
            }
        cache = str(root / "cache/frozen_parent_sleep.npz")
        configs = {}
        for arm in ("A", "S", "B", "C", "D"):
            cfg = copy.deepcopy(
                load_config(historical["config"]) if arm == "A" else config
            )
            cfg["catalog_path"] = str(dataset)
            cfg["truth"] = {"parameter_columns": {}}
            cfg["extra_columns"] = []
            amortized = cfg["amortized"]
            amortized.setdefault("data", {}).update(
                use_redshift_for_split=False,
                stratify_column=None,
                redshift_bins=[],
                selection_mode="sequential",
            )
            amortized["truth_free_validation"] = {
                **recipe["truth_free_validation"],
                "projection_scale_bank": cache,
            }
            amortized.setdefault("inference", {}).update(
                resume_shards=True,
                write_residual_samples=True,
                write_truth_diagnostics=False,
                write_truth_snapshot=False,
            )
            if arm != "A":
                objective = amortized["objective"]
                objective["sleep"]["conditioning_mask"] = recipe["conditioning_mask"]
                objective["sleep"]["noiseless_flux_cache"] = {
                    "enabled": True,
                    "path": cache,
                    "candidates": 131072,
                    "decoder_batch_size": 256,
                }
                weight = recipe["observed_weights"].get(arm, 0.0)
                objective["observed_elbo"] = {
                    "enabled": weight > 0,
                    "weight": weight,
                    "sleep_weight": 1.0,
                    "n_samples": recipe["observed_draws"],
                    "require_all_finite": True,
                    "common_sleep_random_numbers": True,
                }
                amortized["training"].update(
                    data_parallel="single",
                    learning_rate=recipe["learning_rate"],
                    validation_every=1,
                    validation_sleep_seed=recipe["seed"],
                    best_checkpoint_min_epoch=1,
                    best_checkpoint_metric=(
                        "validation_loss" if weight else "validation_sleep_nll"
                    ),
                )
            path = staging / "manifests" / f"{arm}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            configs[arm] = {
                "path": str(root / path.relative_to(staging)),
                "sha256": sha256_file(path),
            }
        manifest = {
            "method": recipe["method"],
            "status": "PREPARED",
            "code_commit": commit,
            "source": source,
            "historical": historical,
            "source_root": str(source_root),
            "source_manifest_sha256": sha256_file(source_root / "RUN_MANIFEST.json"),
            "dataset": old["dataset"],
            "cohorts": cohort_manifest,
            "configs": configs,
            "recipe": recipe,
            "confirmation_cohort_audit": audit,
            "confirmation_limit": "reserved for this continuation; not a new unseen catalogue",
            "truth_used": False,
            "scientific_promotion": False,
            "population_training_started": False,
        }
        write(staging / "RUN_MANIFEST.json", manifest)
        staging.rename(root)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest


def execute(command: list) -> None:
    command = [str(x) for x in command]
    print("[balanced-npe] " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def cli(config: str | Path, *args) -> None:
    execute([sys.executable, "-m", "euclid_dsps.cli", "--config", config, *args])


def certify_training(root: Path, arm: str) -> dict:
    from scripts.finalize_feniks_sc_drws_topology_npe_arm import finalize

    value = finalize(root=root, arm=arm)
    if arm == "S":
        cache = root / "cache/frozen_parent_sleep.npz"
        write(
            root / "TRAINING_CACHE_FROZEN.json",
            {
                "path": str(cache),
                "sha256": sha256_file(cache),
                "sidecar_sha256": sha256_file(cache.with_suffix(".npz.json")),
                "truth_used": False,
            },
        )
    return value


def verify_cache(root: Path) -> None:
    receipt = read(root / "TRAINING_CACHE_FROZEN.json")
    path = Path(receipt["path"])
    if (
        sha256_file(path) != receipt["sha256"]
        or sha256_file(path.with_suffix(".npz.json")) != receipt["sidecar_sha256"]
    ):
        raise ValueError("frozen training simulation bank changed")


def train(root: Path, manifest: dict, arm: str) -> dict:
    if arm not in {"S", "B", "C", "D"}:
        raise ValueError("training arm must be S/B/C/D")
    base = root / "arms" / arm
    receipt = base / "ARM_COMPLETE.json"
    if receipt.exists():
        if arm == "S" and not (root / "TRAINING_CACHE_FROZEN.json").exists():
            return certify_training(root, arm)
        return verified_artifacts(read(receipt))
    if arm != "S":
        verify_cache(root)
    out = base / "train"
    if (out / "training_summary.json").exists():
        return certify_training(root, arm)
    if out.exists():
        raise FileExistsError(f"partial training preserved; do not overwrite: {out}")
    initial = (
        manifest["source"]
        if arm == "S"
        else verified_artifacts(read(root / "arms/S/ARM_COMPLETE.json"))
    )
    cfg = manifest["configs"][arm]["path"]
    base.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg, base / "runtime_config.yaml")
    if arm == "S" and not (root / "SED_SMOKE_COMPLETE.json").exists():
        smoke = root / "smoke/inference"
        cli(
            cfg,
            "amortized-infer-diffsky",
            "--runtime",
            "gpu",
            "--dataset",
            manifest["dataset"]["path"],
            "--out",
            smoke,
            "--checkpoint",
            initial["checkpoint"],
            "--feature-stats",
            initial["feature_stats"],
            "--row-indices-file",
            manifest["cohorts"]["smoke"]["path"],
            "--selection-mode",
            "sequential",
            "--limit",
            2,
            "--batch-size",
            2,
            "--jax-batch-size",
            2,
            "--posterior-samples",
            4,
            "--prior-samples",
            1,
            "--decoder-sample-chunk-size",
            1,
            "--prior-predictive-batch-size",
            1,
            "--shard-outputs",
            "--seed",
            manifest["recipe"]["seed"],
        )
        cli(
            cfg, "amortized-finalize-inference", "--out", smoke, "--limit", 2, "--quiet"
        )
        write(
            root / "SED_SMOKE_COMPLETE.json",
            {
                "status": "COMPLETE",
                "objects": 2,
                "draws": 4,
                "truth_used": False,
                "scientific_promotion": False,
            },
        )
    epochs = manifest["recipe"]["anchor_epochs" if arm == "S" else "candidate_epochs"]
    cli(
        cfg,
        "amortized-train-diffsky",
        "--runtime",
        "gpu",
        "--dataset",
        manifest["dataset"]["path"],
        "--out",
        out,
        "--train-indices-file",
        manifest["cohorts"]["train"]["path"],
        "--validation-indices-file",
        manifest["cohorts"]["validation"]["path"],
        "--initial-checkpoint",
        initial["checkpoint"],
        "--fixed-feature-stats",
        manifest["source"]["feature_stats"],
        "--epochs",
        epochs,
        "--batch-size",
        512,
        "--jax-batch-size",
        256,
        "--n-samples",
        1,
        "--kl-annealing-epochs",
        0,
        "--kl-weight-max",
        0,
        "--validation-every",
        1,
        "--best-checkpoint-min-epoch",
        1,
        "--data-parallel",
        "single",
        "--freeze-prior",
        "--seed",
        manifest["recipe"]["seed"],
        "--no-progress",
    )
    return certify_training(root, arm)


def validate(root: Path, manifest: dict, arm: str, *, confirmation: bool = False) -> dict:
    verify_cache(root)
    artifact = verified_artifacts(
        manifest["historical"]
        if arm == "A"
        else read(root / "arms" / arm / "ARM_COMPLETE.json")
    )
    config = manifest["configs"][arm]["path"]
    out = root / ("confirmation" if confirmation else "validation") / arm
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "VALIDATION_COMPLETE.json"
    if receipt.exists():
        completed = read(receipt)
        if completed["checkpoint_sha256"] != artifact["checkpoint_sha256"]:
            raise ValueError("completed validation belongs to another checkpoint")
        return completed
    seed = manifest["recipe"]["seed"] + (10000 if confirmation else 1000)
    for label, cohort, draws in (
        ("tracking_k256", "confirmation" if confirmation else "validation_pilot", 256),
        (
            "support_k1024",
            "confirmation_support" if confirmation else "support_pilot",
            1024,
        ),
    ):
        rows = manifest["cohorts"][cohort]
        target = out / label / "inference"
        summary = out / label / "summary/TRUTH_FREE_POSTERIOR_VALIDATION.json"
        if summary.exists():
            continue
        print(f"[balanced-npe] arm={arm} stage={label}", flush=True)
        cli(
            config,
            "amortized-infer-diffsky",
            "--runtime",
            "gpu",
            "--dataset",
            manifest["dataset"]["path"],
            "--out",
            target,
            "--checkpoint",
            artifact["checkpoint"],
            "--feature-stats",
            artifact["feature_stats"],
            "--row-indices-file",
            rows["path"],
            "--selection-mode",
            "sequential",
            "--limit",
            rows["objects"],
            "--batch-size",
            manifest["recipe"]["inference_batch_size"],
            "--jax-batch-size",
            manifest["recipe"]["inference_batch_size"],
            "--posterior-samples",
            draws,
            "--prior-samples",
            1,
            "--decoder-sample-chunk-size",
            1,
            "--prior-predictive-batch-size",
            1,
            "--shard-outputs",
            "--seed",
            seed + draws,
        )
        cli(
            config,
            "amortized-finalize-inference",
            "--out",
            target,
            "--limit",
            rows["objects"],
            "--quiet",
        )
        execute(
            [
                sys.executable,
                "scripts/summarize_feniks_sc_drws_truth_free_posterior.py",
                "--inference",
                target,
                "--out",
                summary.parent,
            ]
        )
    internal = out / "internal/INTERNAL_TRUTH_FREE_VALIDATION.json"
    if not internal.exists():
        rows = manifest["cohorts"][
            "confirmation" if confirmation else "validation_pilot"
        ]
        execute(
            [
                sys.executable,
                "scripts/evaluate_feniks_sc_drws_topology_npe_internal.py",
                "--config",
                config,
                "--checkpoint",
                artifact["checkpoint"],
                "--feature-stats",
                artifact["feature_stats"],
                "--dataset",
                manifest["dataset"]["path"],
                "--row-indices",
                rows["path"],
                "--out",
                internal.parent,
                "--objects",
                rows["objects"],
                "--posterior-draws",
                64,
                "--decoder-sample-chunk-size",
                1,
                "--seed",
                seed,
            ]
        )
    value = {
        "status": "COMPLETE",
        "arm": arm,
        "truth_used": False,
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "tracking": read(
            out / "tracking_k256/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json"
        ),
        "support": read(
            out / "support_k1024/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json"
        ),
        "internal": read(internal),
        "scientific_promotion": False,
    }
    write(receipt, value)
    return value


def checks(value: dict) -> dict[str, bool]:
    internal = value["internal"]
    return {
        "truth_free": value.get("truth_used") is False
        and internal.get("truth_used") is False,
        "tracking_support": value["tracking"]["technical_gate"]["status"] == "PASS",
        "support_k1024": value["support"]["technical_gate"]["status"] == "PASS",
        "held_out": internal["held_out_band"]["status"] == "PASS",
        "simulated_marginals": internal["model_generated_calibration"]["status"]
        == "PASS",
        "simulated_joint_projections": internal["joint_projection_calibration"][
            "status"
        ]
        == "PASS",
        "masked_simulated_marginals": internal["masked_model_generated_calibration"][
            "status"
        ]
        == "PASS",
        "masked_simulated_joint_projections": internal[
            "masked_joint_projection_calibration"
        ]["status"]
        == "PASS",
    }


def choose(values: dict, maximum_nll_increase: float) -> dict:
    fingerprints = {row["internal"]["simulation_sha256"] for row in values.values()}
    if len(fingerprints) != 1:
        raise ValueError("arms did not evaluate the same simulated input bank")
    reference_nll = values["B"]["internal"]["simulation_nll"]
    all_checks = {}
    for arm, value in values.items():
        gate = checks(value)
        nll = value["internal"]["simulation_nll"]
        gate["simulation_nll_guard"] = bool(
            np.isfinite(nll) and nll <= reference_nll + maximum_nll_increase
        )
        all_checks[arm] = gate
    eligible = [arm for arm in ("B", "C", "D") if all(all_checks[arm].values())]
    winner = (
        min(eligible, key=lambda arm: values[arm]["internal"]["simulation_nll"])
        if eligible
        else None
    )
    return {
        "winner": winner,
        "checks": all_checks,
        "status": "FROZEN" if winner else "NO_ELIGIBLE_ARM",
        "selection": "all prespecified gates, then common-simulation NLL",
        "truth_used": False,
        "scientific_promotion": False,
    }


def select(root: Path, manifest: dict) -> dict:
    results = {
        arm: read(root / "validation" / arm / "VALIDATION_COMPLETE.json")
        for arm in ARMS
    }
    value = choose(
        results,
        manifest["recipe"]["maximum_simulation_nll_increase"],
    )
    write_comparison(root, results)
    if value["winner"]:
        value["artifacts"] = verified_artifacts(
            read(root / "arms" / value["winner"] / "ARM_COMPLETE.json")
        )
    write(root / "CANDIDATE_FROZEN.json", value)
    return value


def write_comparison(root: Path, results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for arm, result in results.items():
        support = result["support"]["support"]
        internal = result["internal"]
        rows.append(
            {
                "arm": arm,
                "ess_fraction_median": support["raw_ess"]["fraction_median"],
                "ess_fraction_q10": support["raw_ess"]["fraction_q10"],
                "bad_k_fraction": support["pareto_k"]["gt_0p7_or_nonfinite_fraction"],
                "nonfinite_k_fraction": support["pareto_k"]["nonfinite_fraction"],
                "maximum_weight_p90": support["maximum_raw_weight"]["p90"],
                "simulation_nll": internal["simulation_nll"],
                "joint_projection_ks_max": internal["joint_projection_calibration"][
                    "maximum_coordinate_pit_ks"
                ],
                "masked_joint_projection_ks_max": internal[
                    "masked_joint_projection_calibration"
                ]["maximum_coordinate_pit_ks"],
                "heldout_pass": internal["held_out_band"]["status"] == "PASS",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "matched_validation.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for axis, column, threshold, label in zip(
        axes.flat,
        (
            "ess_fraction_median",
            "bad_k_fraction",
            "maximum_weight_p90",
            "joint_projection_ks_max",
        ),
        (0.05, 0.20, 0.80, 0.12),
        (
            "K1024 median raw ESS/K (higher is better)",
            "Pareto k > 0.7 or nonfinite fraction (lower is better)",
            "P90 maximum raw weight (lower is better)",
            "Maximum simulated projection KS (lower is better)",
        ),
        strict=True,
    ):
        axis.bar(
            frame["arm"],
            frame[column],
            color=["#777777", "#377eb8", "#4daf4a", "#e41a1c"],
        )
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_title(label, fontsize=9)
        axis.set_ylim(bottom=0)
    fig.suptitle("Frozen-parent development comparison; no scientific promotion")
    fig.savefig(root / "matched_validation.png", dpi=160)
    plt.close(fig)


def confirm(root: Path, manifest: dict) -> dict:
    winner = read(root / "CANDIDATE_FROZEN.json")["winner"]
    if winner is None:
        value = {"status": "SKIPPED", "reason": "no eligible development candidate"}
    else:
        value = validate(root, manifest, winner, confirmation=True)
    write(root / "CONFIRMATION_COMPLETE.json", value)
    return value


def close(root: Path, manifest: dict) -> dict:
    candidate = read(root / "CANDIDATE_FROZEN.json")
    confirmation = read(root / "CONFIRMATION_COMPLETE.json")
    gate = checks(confirmation) if candidate["winner"] else {}
    passed = bool(gate) and all(gate.values())
    value = {
        "status": "DIAGNOSTIC_COMPLETE",
        "candidate": candidate["winner"],
        "confirmation_checks": gate,
        "posterior_technical_ready": passed,
        "population_training_started": False,
        "scientific_promotion": False,
        "truth_used": False,
        "population": "NOT_AUTHORIZED_BY_THIS_EXPERIMENT",
    }
    write(root / "BALANCED_NPE_COMPLETE.json", value)
    return value


def monitor(root: Path) -> None:
    print(f"root={root}")
    for arm in ("S", "B", "C", "D"):
        path = root / "arms" / arm / "ARM_COMPLETE.json"
        if path.exists():
            x = read(path)
            print(
                f"train {arm}: COMPLETE {x['best_checkpoint_metric']}={x['best_checkpoint_value']:.5f}"
            )
        else:
            print(f"train {arm}: waiting/in progress")
    for arm in ARMS:
        path = root / "validation" / arm / "VALIDATION_COMPLETE.json"
        if path.exists():
            print(f"validation {arm}: {checks(read(path))}")
        else:
            for stage, total in (("tracking_k256", 32), ("support_k1024", 16)):
                directory = path.parent / stage / "inference/shard_metadata"
                print(
                    f"{arm} {stage}: {len(list(directory.glob('batch_*.json')))}/{total} batches"
                )
    for name in ("CANDIDATE_FROZEN.json", "BALANCED_NPE_COMPLETE.json"):
        if (root / name).exists():
            print(name, json.dumps(read(root / name), sort_keys=True))


def submit(root: Path, manifest: dict) -> dict:
    """Persist each dependency immediately; interrupted submission is resumable."""
    snapshot = Path(os.environ["REPO_DIR"])
    require_git_commit(snapshot, manifest["code_commit"])
    ledger = root / "SUBMISSION.json"
    state = read(ledger) if ledger.exists() else {"jobs": {}, "status": "SUBMITTING"}
    if state.get("inflight"):
        raise RuntimeError(
            "ambiguous interrupted sbatch; inspect squeue before resubmission"
        )
    log = Path(os.environ["BALANCED_LOG_ROOT"])
    gpu = snapshot / "scripts/feniks_sc_drws_balanced_npe.slurm"
    cpu = snapshot / "scripts/feniks_sc_drws_balanced_npe_finalize.slurm"
    stages = (
        ("anchor", "train", None, None, gpu, "04:00:00", "S"),
        ("candidates", "train", "0-2%3", "anchor", gpu, "04:00:00", ""),
        ("validation", "validate", "0-3%4", "candidates", gpu, "10:00:00", ""),
        ("selection", "select", None, "validation", cpu, "00:30:00", ""),
        ("confirmation", "confirm", None, "selection", gpu, "10:00:00", ""),
        ("closure", "close", None, "confirmation", cpu, "00:30:00", ""),
    )
    for name, action, array, parent, script, limit, arm in stages:
        if name in state["jobs"]:
            continue
        job_pattern = "%A_%a" if array else "%j"
        command = [
            "sbatch",
            "--parsable",
            f"--time={limit}",
            f"--output={log}/{name}-{job_pattern}.out",
            f"--error={log}/{name}-{job_pattern}.err",
            f"--export=ALL,ACTION={action},ARM={arm}",
        ]
        if array:
            command.append(f"--array={array}")
        if parent:
            command.append(f"--dependency=afterok:{state['jobs'][parent]}")
        command.append(str(script))
        state["inflight"] = {"stage": name, "command": command}
        write(ledger, state)
        try:
            result = subprocess.check_output(command, text=True).strip().split(";")[0]
        except subprocess.CalledProcessError:
            state.pop("inflight")
            write(ledger, state)
            raise
        if not result.isdigit():
            raise ValueError(f"unexpected sbatch response: {result}")
        state["jobs"][name] = result
        state.pop("inflight")
        write(ledger, state)
        environment = {
            "BALANCED_ROOT": str(root),
            "BALANCED_LOG_ROOT": str(log),
            "ALL_JOBS": ",".join(state["jobs"].values()),
            "JOB_REPO_DIR": str(snapshot),
            **{f"{key.upper()}_JOB": value for key, value in state["jobs"].items()},
        }
        Path(os.environ["BALANCED_ENV"]).write_text(
            "".join(
                f"export {key}={shlex.quote(value)}\n"
                for key, value in environment.items()
            )
        )
    state.update(
        status="SUBMITTED",
        population_training_started=False,
        maximum_concurrent_gpus=4,
        gpu_hours_allocation_ceiling=66,
    )
    write(ledger, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "submit",
            "train",
            "validate",
            "select",
            "confirm",
            "close",
            "monitor",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--arm", choices=("A", "S", "B", "C", "D"))
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path("configs/experiments/feniks_sc_drws_r29_balanced_npe.yaml"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    repo = Path(__file__).resolve().parents[1]
    if args.action == "prepare":
        if args.source_root is None:
            parser.error("prepare requires --source-root")
        value = prepare(args.source_root, root, args.recipe, repo)
    elif args.action == "monitor":
        monitor(root)
        return
    else:
        manifest = read(root / "RUN_MANIFEST.json")
        require_git_commit(repo, manifest["code_commit"])
        for item in (*manifest["configs"].values(), *manifest["cohorts"].values()):
            if sha256_file(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"manifest input changed: {item['path']}")
        if args.action in {"train", "validate"} and not args.arm:
            parser.error("train/validate requires --arm")
        with (root / f".{args.action}-{args.arm or 'global'}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action in {"train", "validate"}:
                value = globals()[args.action](root, manifest, args.arm)
            else:
                value = globals()[args.action](root, manifest)
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
