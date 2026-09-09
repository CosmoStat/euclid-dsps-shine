"""Sequential, fail-closed corrected-numerics pilot, never population training."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.population_vem import sha256_file


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    def finite(item):
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temp.replace(path)


def verify_reference(path, source_root):
    path = Path(path).resolve()
    final, manifest = read(path / "FINAL.json"), read(path / "RUN_MANIFEST.json")
    if (
        final.get("status") != "REDSHIFT_PRECISION_COMPLETE"
        or manifest.get("mode") != "redshift_precision_audit"
        or manifest["source_manifest_sha256"]
        != sha256_file(Path(source_root) / "RUN_MANIFEST.json")
    ):
        raise ValueError("completed same-source redshift precision diagnostic required")
    if "REDSHIFT_PRECISION.json" not in final["artifacts"]:
        raise ValueError("missing redshift precision report")
    for name, item in final["artifacts"].items():
        if sha256_file(path / name) != item["sha256"]:
            raise ValueError(f"reference artifact changed: {name}")
    report = read(path / "REDSHIFT_PRECISION.json")
    candidates = [b for b in report["branches"] if b["name"] == "zpath64_full"]
    if (
        len(candidates) != 1
        or not candidates[0]["checks"]
        or any(c["status"] != "PASS" for c in candidates[0]["checks"])
    ):
        raise ValueError("zpath64_full reference has unresolved checks")
    if report.get("prior_bitwise_unchanged") is not True:
        raise ValueError("reference prior identity not certified")
    return dict(
        path=str(path),
        final_sha256=sha256_file(path / "FINAL.json"),
        manifest_sha256=sha256_file(path / "RUN_MANIFEST.json"),
    )


def candidate_configuration(config):
    config = copy.deepcopy(config)
    config["model"].update(
        photometry_integrator="merged_gauss4_v1",
        mdf_weight_precision="float64_v1",
        spline_precision="float64_v1",
    )
    a = config["amortized"]
    a.setdefault("latent", {})["arithmetic_precision"] = "float64_v1"
    a["likelihood"]["arithmetic_precision"] = "float64_v1"
    a.setdefault("population_vi", {})["enabled"] = False
    a["prior"]["train_jointly"] = False
    a.setdefault("inference", {}).update(
        write_truth_snapshot=False,
        write_truth_diagnostics=False,
        write_residual_samples=True,
        resume_shards=True,
    )
    return config


def require_qualification(report):
    cases = [c for c in report.get("cases", []) if c["variant"] == "spline64"]
    if (
        report.get("candidate_numerical_checks") != "PASS"
        or len(cases) != 6
        or any(c["numerical_checks"] != "PASS" for c in cases)
    ):
        raise ValueError("full 15D candidate qualification did not pass; no training")


def select_cohorts(root, source_root, recipe):
    source = read(Path(source_root) / "RUN_MANIFEST.json")
    arrays = {}
    for name in ("train", "validation", "validation_pilot"):
        item = source["cohorts"][name]
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"source cohort changed: {name}")
        arrays[name] = np.load(item["path"], allow_pickle=False)
        if len(np.unique(arrays[name])) != len(arrays[name]):
            raise ValueError("duplicate cohort indices")
    if np.intersect1d(
        arrays["train"], np.union1d(arrays["validation"], arrays["validation_pilot"])
    ).size:
        raise ValueError("training and validation overlap")
    rng = np.random.default_rng(recipe["seed"])
    selected = {}
    selected["tracking"] = np.sort(
        rng.choice(
            arrays["validation_pilot"], recipe["tracking_objects"], replace=False
        )
    )
    selected["validation"] = np.sort(
        rng.choice(
            np.setdiff1d(arrays["validation"], arrays["validation_pilot"]),
            recipe["validation_objects"],
            replace=False,
        )
    )
    selected["train"] = np.sort(
        rng.choice(arrays["train"], recipe["train_objects"], replace=False)
    )
    selected["smoke_train"] = selected["train"][: recipe["smoke_train_objects"]]
    selected["smoke_validation"] = selected["validation"][
        : recipe["smoke_validation_objects"]
    ]
    result = {}
    for name, values in selected.items():
        path = root / "cohorts" / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values, allow_pickle=False)
        result[name] = dict(
            path=str(path), sha256=sha256_file(path), objects=len(values)
        )
    write(
        root / "COHORTS.json",
        dict(
            cohorts=result,
            truth_used=False,
            selection="uniform seeded observed-row indices; no catalogue parameters",
        ),
    )
    return result


class Commands:
    def __init__(self, root, seconds):
        self.root = root
        self.deadline = time.monotonic() + seconds

    def run(self, stage, command):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("night wall-clock budget exhausted")
        command = [str(c) for c in command]
        write(
            self.root / "NIGHT_PROGRESS.json",
            dict(stage=stage, command=command, remaining_seconds=remaining),
        )
        print("[precision-night] " + " ".join(command), flush=True)
        # Each child is synchronous; timeout kills and waits for this direct process.
        subprocess.run(command, check=True, timeout=remaining)

    def cli(self, stage, config, *args):
        self.run(
            stage, [sys.executable, "-m", "euclid_dsps.cli", "--config", config, *args]
        )


def certify_training(out, config, frozen_hash, topology_hash, stats_hash, *, observed):
    from euclid_dsps.amortized.posterior import conditional_flow_topology
    from euclid_dsps.amortized.train import _array_tree_sha256, load_checkpoint

    summary = read(out / "training_summary.json")
    checkpoint = out / "checkpoints/best.eqx"
    trained = load_checkpoint(checkpoint, config)
    if (
        _array_tree_sha256((trained.prior, trained.sed_scale, trained.band_calibration))
        != frozen_hash
    ):
        raise ValueError("prior/calibration changed during frozen phase")
    topology = conditional_flow_topology(trained.encoder)
    if topology["fingerprint_sha256"] != topology_hash:
        raise ValueError("posterior topology changed")
    if sha256_file(out / "feature_stats.json") != stats_hash:
        raise ValueError("fixed feature statistics changed")
    rows = pd.read_csv(out / "training_log.csv")
    train = rows.loc[rows["split"] == "train"]
    if (
        train.empty
        or not np.isfinite(train["loss"]).all()
        or not (train["update_applied"] > 0.5).all()
    ):
        raise ValueError("nonfinite loss or skipped optimizer update")
    if (
        not np.isfinite(summary["best_loss"])
        or summary.get("prior_train_jointly") is not False
    ):
        raise ValueError("invalid training summary")
    audit = summary.get("objective_component_gradient_audit") or {}
    prefix = "observed_after_freeze" if observed else "sleep_after_freeze"
    if (
        not np.isfinite(audit.get(prefix + "_encoder_grad_norm", np.nan))
        or audit[prefix + "_encoder_grad_norm"] <= 0
    ):
        raise ValueError("missing finite nonzero encoder gradient audit")
    if audit.get(prefix + "_prior_grad_norm") != 0:
        raise ValueError("prior gradient was not masked")
    receipt = dict(
        status="COMPLETE",
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256_file(checkpoint),
        checkpoint_sidecar_sha256=sha256_file(checkpoint.with_suffix(".eqx.json")),
        prior_bitwise_unchanged=True,
        feature_stats_sha256=stats_hash,
        best_checkpoint_metric=summary["best_checkpoint_metric"],
        best_loss=summary["best_loss"],
        decoder_budget=summary["decoder_budget"],
        updates=len(train),
        truth_used=False,
        scientific_promotion=False,
    )
    write(out.parent / "ARM_COMPLETE.json", receipt)
    return checkpoint


def continue_night(
    root, manifest, config, spec, model, stats, qualification, qualification_seconds
):
    from euclid_dsps.amortized.posterior import conditional_flow_topology
    from euclid_dsps.amortized.train import (
        _array_tree_sha256,
        load_checkpoint,
        save_checkpoint,
    )

    root = Path(root)
    try:
        require_qualification(qualification)
    except ValueError as exc:
        result = dict(
            status="BLOCKED_NUMERICAL_QUALIFICATION",
            error=str(exc),
            training_started=False,
            scientific_promotion=False,
            population_training_started=False,
        )
        write(root / "NIGHT_FINAL.json", result)
        return result
    recipe = manifest["night_recipe"]
    commands = Commands(root, recipe["maximum_seconds"] - qualification_seconds)
    cohorts = select_cohorts(root, manifest["source_root"], recipe)
    frozen_hash = _array_tree_sha256(
        (model.prior, model.sed_scale, model.band_calibration)
    )
    topology_hash = conditional_flow_topology(model.encoder)["fingerprint_sha256"]
    stats_path = Path(manifest["source"]["feature_stats"])
    stats_hash = sha256_file(stats_path)
    initial = root / "initial/model.eqx"
    save_checkpoint(
        initial,
        model,
        config=config,
        latent_spec=spec,
        feature_stats=stats,
        epoch=0,
        metric=0.0,
        metric_name="numerical_migration_not_selection",
    )
    restored = load_checkpoint(initial, config)
    if _array_tree_sha256(restored) != _array_tree_sha256(model):
        raise ValueError("explicit checkpoint migration changed model arrays")
    write(
        root / "MIGRATION.json",
        dict(
            status="PASS",
            source=manifest["source"],
            checkpoint_sha256=sha256_file(initial),
            frozen_array_sha256=frozen_hash,
            scope="identical model arrays; versioned x/theta, SED and likelihood arithmetic",
            historical_checkpoints_modified=False,
            bank_reuse=False,
            scientific_promotion=False,
        ),
    )
    dataset = manifest["dataset"]["path"]
    bank = root / "cache/precision64_sleep.npz"
    configs = {}
    for arm in ("smoke", "A", "B", "C"):
        cfg = copy.deepcopy(config)
        a = cfg["amortized"]
        observed = arm in {"smoke", "C"}
        a["objective"]["sleep"]["noiseless_flux_cache"] = dict(
            enabled=True,
            path=str(root / "smoke/cache.npz" if arm == "smoke" else bank),
            candidates=recipe[
                "smoke_candidates" if arm == "smoke" else "sleep_candidates"
            ],
            decoder_batch_size=recipe["decoder_batch_size"],
        )
        a["objective"]["observed_elbo"] = dict(
            enabled=observed,
            weight=recipe["observed_weight"] if observed else 0.0,
            sleep_weight=1.0,
            n_samples=recipe["observed_draws"],
            require_all_finite=True,
            common_sleep_random_numbers=True,
        )
        a["training"].update(
            data_parallel="single",
            learning_rate=recipe["learning_rate"],
            component_gradient_audit_first_batch=True,
            component_gradient_audit_objects=2,
            validation_every=1,
            validation_sleep_seed=recipe["seed"],
            best_checkpoint_min_epoch=1,
            best_checkpoint_metric="validation_loss"
            if observed
            else "validation_sleep_nll",
        )
        a.setdefault("truth_free_validation", {})["projection_scale_bank"] = str(bank)
        path = root / "configs" / f"{arm}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[arm] = (cfg, path)
    jax.clear_caches()
    checkpoints = {"A": initial}
    for arm, epochs, origin in (
        ("smoke", 1, "A"),
        ("B", recipe["sleep_epochs"], "A"),
        ("C", recipe["elbo_epochs"], "B"),
    ):
        out = root / "arms" / arm / "train"
        prefix = "smoke_" if arm == "smoke" else ""
        commands.cli(
            "train_" + arm,
            configs[arm][1],
            "amortized-train-diffsky",
            "--runtime",
            "gpu",
            "--dataset",
            dataset,
            "--out",
            out,
            "--train-indices-file",
            cohorts[prefix + "train"]["path"],
            "--validation-indices-file",
            cohorts[prefix + "validation"]["path"],
            "--initial-checkpoint",
            checkpoints[origin],
            "--fixed-feature-stats",
            stats_path,
            "--epochs",
            epochs,
            "--batch-size",
            recipe["jax_batch_size"],
            "--jax-batch-size",
            recipe["jax_batch_size"],
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
            recipe["seed"],
            "--no-progress",
        )
        checkpoints[arm] = certify_training(
            out,
            configs[arm][0],
            frozen_hash,
            topology_hash,
            stats_hash,
            observed=arm != "B",
        )
    summaries = {}
    for arm in ("A", "B", "C"):
        base = root / "validation" / arm
        inference = base / "tracking_k256/inference"
        cfg = configs[arm][1]
        commands.cli(
            "infer_" + arm,
            cfg,
            "amortized-infer-diffsky",
            "--runtime",
            "gpu",
            "--dataset",
            dataset,
            "--out",
            inference,
            "--checkpoint",
            checkpoints[arm],
            "--feature-stats",
            stats_path,
            "--row-indices-file",
            cohorts["tracking"]["path"],
            "--selection-mode",
            "sequential",
            "--limit",
            recipe["tracking_objects"],
            "--batch-size",
            2,
            "--jax-batch-size",
            2,
            "--posterior-samples",
            recipe["tracking_draws"],
            "--prior-samples",
            1,
            "--decoder-sample-chunk-size",
            1,
            "--prior-predictive-batch-size",
            1,
            "--shard-outputs",
            "--seed",
            recipe["seed"] + 256,
        )
        commands.cli(
            "finalize_" + arm,
            cfg,
            "amortized-finalize-inference",
            "--out",
            inference,
            "--limit",
            recipe["tracking_objects"],
            "--quiet",
        )
        commands.run(
            "summarize_" + arm,
            [
                sys.executable,
                "scripts/summarize_feniks_sc_drws_truth_free_posterior.py",
                "--inference",
                inference,
                "--out",
                base / "tracking_k256/summary",
            ],
        )
        commands.run(
            "internal_" + arm,
            [
                sys.executable,
                "scripts/evaluate_feniks_sc_drws_topology_npe_internal.py",
                "--config",
                cfg,
                "--checkpoint",
                checkpoints[arm],
                "--feature-stats",
                stats_path,
                "--dataset",
                dataset,
                "--row-indices",
                cohorts["tracking"]["path"],
                "--out",
                base / "internal",
                "--objects",
                recipe["tracking_objects"],
                "--posterior-draws",
                64,
                "--decoder-sample-chunk-size",
                1,
                "--seed",
                recipe["seed"],
            ],
        )
        summaries[arm] = read(
            base / "tracking_k256/summary/TRUTH_FREE_POSTERIOR_VALIDATION.json"
        )
    result = dict(
        status="PRECISION_NIGHT_DIAGNOSTIC_COMPLETE",
        summaries=summaries,
        scientific_promotion=False,
        population_training_started=False,
        truth_used=False,
        interpretation="A unchanged weights under new numerics; B sleep; C continuation sleep+ELBO. No independent final test.",
    )
    write(root / "NIGHT_FINAL.json", result)
    write(
        root / "NIGHT_ARTIFACTS.json",
        {
            str(p.relative_to(root)): sha256_file(p)
            for p in sorted(root.rglob("*"))
            if p.is_file()
            and p.suffix in {".json", ".yaml", ".npy", ".eqx"}
            and p.name != "NIGHT_ARTIFACTS.json"
        },
    )
    return result
