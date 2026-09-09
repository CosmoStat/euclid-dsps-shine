#!/usr/bin/env python3
"""Bounded same-family local-VI experiment; no population training or promotion."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from euclid_dsps.amortized.data import load_photometry_arrays_from_config
from euclid_dsps.amortized.features import make_encoder_features, read_feature_stats
from euclid_dsps.amortized.infer import _model_flux_from_x_sample_chunks
from euclid_dsps.amortized.latent import latent_spec_from_config, theta_to_x, x_to_theta
from euclid_dsps.amortized.likelihood import (
    photometric_normalized_residual,
    photometric_sigma_eff,
)
from euclid_dsps.amortized.local_vi_diagnostic import (
    Budget,
    BudgetExceeded,
    analytic_controls,
    assert_close,
    gradient_audit,
    initialize,
    log_prob,
    make_step,
    perturb,
    sample,
)
from euclid_dsps.amortized.npe_validation import (
    assert_truth_free_columns,
    summarize_truth_free_joint_bank,
)
from euclid_dsps.amortized.population_vem import require_git_commit, sha256_file
from euclid_dsps.amortized.posterior import (
    conditional_flow_topology,
    posterior_log_prob,
    sample_posterior,
)
from euclid_dsps.amortized.posterior_target import (
    PosteriorObservation,
    _apply_model_calibration,
    posterior_log_target,
    posterior_log_target_from_model_flux,
)
from euclid_dsps.amortized.train import (
    _array_tree_sha256,
    _sample_sleep_noise,
    _sleep_encoder_features,
    _sleep_observed_selection_mask,
    _sleep_runtime_config,
    load_checkpoint,
)
from euclid_dsps.config import load_config
from euclid_dsps.filters import load_filters
from euclid_dsps.io import required_catalog_columns
from euclid_dsps.model import dynamic_model_args, load_context
from scripts.prepare_feniks_sc_drws_topology_npe_pilot import (
    _representative_observed_rows,
)
from scripts.run_feniks_sc_drws_balanced_npe import read, verified_artifacts, write


def observed_only_config(config):
    """Remove legacy reporting/fit column references before any catalogue read."""
    config = copy.deepcopy(config)
    config["truth"] = {"parameter_columns": {}, "redshift_column": None}
    config.setdefault("redshift", {}).update(column=None, truth_column=None)
    config.setdefault("model", {})["parameter_columns"] = {}
    config["extra_columns"] = []
    allowed = {
        str(band[key])
        for band in config["bands"]
        for key in ("column", "error_column")
        if band.get(key)
    }
    actual = set(required_catalog_columns(config))
    if not actual <= allowed:
        raise ValueError(f"non-photometric catalogue columns: {actual - allowed}")
    check_config(config)
    return config


def check_config(config):
    if config.get("truth", {}).get("parameter_columns"):
        raise ValueError("catalogue truth forbidden")
    assert_truth_free_columns(required_catalog_columns(config))
    identifiers = [
        config.get("dataset", {}).get(key) for key in ("id_column", "object_id_column")
    ] + [config.get("object_id_column")]
    assert_truth_free_columns([value for value in identifiers if value])
    likelihood = config["amortized"]["likelihood"]
    if likelihood["type"] != "gaussian" or any(
        likelihood.get(key, 0) != 0 for key in ("error_floor_frac", "error_jitter")
    ):
        raise ValueError(
            "this diagnostic is certified only for Gaussian zero-floor likelihood"
        )
    sleep = config["amortized"]["objective"]["sleep"]
    if (
        sleep["error_model"] != "observed_catalog"
        or sleep["noise_family"] != "match_likelihood"
    ):
        raise ValueError("simulation context/noise contract changed")


def verify_photometry_reference(path, source_root):
    path = Path(path).resolve()
    final = read(path / "FINAL.json")
    if final.get("status") != "PHOTOMETRY_REFERENCE_COMPLETE":
        raise ValueError("fixed-spectrum qualification incomplete")
    for name, item in final["artifacts"].items():
        if sha256_file(path / name) != item["sha256"]:
            raise ValueError(f"reference artifact changed: {name}")
    report = read(path / "PHOTOMETRY_REFERENCE.json")
    if report.get("numerical_reference_checks") != "PASS" or len(report["points"]) != 3:
        raise ValueError("fixed-spectrum numerical checks must pass at three points")
    manifest = read(path / "RUN_MANIFEST.json")
    if Path(manifest["source_root"]).resolve() != Path(
        source_root
    ).resolve() or manifest["source_manifest_sha256"] != sha256_file(
        Path(source_root) / "RUN_MANIFEST.json"
    ):
        raise ValueError("reference belongs to another source experiment")
    return dict(
        path=str(path),
        final_sha256=sha256_file(path / "FINAL.json"),
        manifest_sha256=sha256_file(path / "RUN_MANIFEST.json"),
    )


def verify_decoder_reference(
    path, source_root, *, expected_mode="full_decoder_qualification"
):
    path = Path(path).resolve()
    final = read(path / "FINAL.json")
    required = {"FULL_DECODER_QUALIFICATION.json", "QUALIFICATION_POINTS.npz"}
    if final.get(
        "status"
    ) != "FULL_DECODER_QUALIFICATION_COMPLETE" or not required <= set(
        final["artifacts"]
    ):
        raise ValueError("complete full decoder reference and saved points required")
    for name, item in final["artifacts"].items():
        if sha256_file(path / name) != item["sha256"]:
            raise ValueError(f"decoder reference artifact changed: {name}")
    old = read(path / "RUN_MANIFEST.json")
    if (
        old["mode"] != expected_mode
        or Path(old["source_root"]).resolve() != Path(source_root).resolve()
        or old["source_manifest_sha256"]
        != sha256_file(Path(source_root) / "RUN_MANIFEST.json")
    ):
        raise ValueError("decoder reference belongs to another source or mode")
    for name, field in (
        ("config.yaml", "config_sha256"),
        ("observed_rows.npy", "rows_sha256"),
    ):
        if sha256_file(path / name) != old[field]:
            raise ValueError(f"decoder reference input changed: {name}")
    return dict(
        path=str(path),
        final_sha256=sha256_file(path / "FINAL.json"),
        manifest_sha256=sha256_file(path / "RUN_MANIFEST.json"),
    )


def verify_resolution_reference(path, source_root):
    receipt = verify_decoder_reference(
        path, source_root, expected_mode="mdf_precision_qualification"
    )
    report = read(Path(path) / "FULL_DECODER_QUALIFICATION.json")
    if (
        report.get("mdf_reference_checks") != "PASS"
        or report.get("prior_bitwise_unchanged") is not True
    ):
        raise ValueError("MDF reference and frozen prior must pass")
    candidate = report["variant_labels"][-1]
    pending = []
    for case in report["cases"]:
        if case["variant"] == candidate and not all(
            c["passed"] and c["reverse_passed"]
            for c in case["canonical_centered_gradient"]
        ):
            raise ValueError("resolve gradient identities before residual audit")
        if case["variant"] == candidate:
            pending.extend(
                c
                for c in case["checks"]
                if c["status"] != "PASS" and c["component"] != "canonical_loglike"
            )
    if not 1 <= len(pending) <= 8:
        raise ValueError("bounded residual audit requires 1..8 unresolved checks")
    if any(c["component"] == "logprior" for c in pending):
        raise ValueError("residual audit does not resolve logprior failures")
    return receipt


def verify_redshift_precision_reference(path, source_root):
    path = Path(path).resolve()
    final, manifest = read(path / "FINAL.json"), read(path / "RUN_MANIFEST.json")
    if (
        final.get("status") != "TARGET_RESOLUTION_COMPLETE"
        or manifest.get("mode") != "target_resolution_audit"
        or manifest["source_manifest_sha256"]
        != sha256_file(Path(source_root) / "RUN_MANIFEST.json")
    ):
        raise ValueError("complete same-source target resolution reference required")
    for name in (
        "TARGET_RESOLUTION.json",
        "TARGET_RESOLUTION_SNAPSHOT.json",
        "QUALIFICATION_POINTS.npz",
    ):
        if name not in final["artifacts"]:
            raise ValueError(f"missing resolution artifact: {name}")
    for name, item in final["artifacts"].items():
        if sha256_file(path / name) != item["sha256"]:
            raise ValueError(f"resolution reference artifact changed: {name}")
    report = read(path / "TARGET_RESOLUTION.json")
    remaining = [c for c in report["checks"] if c["status"] != "PASS"]
    if len(remaining) != 1 or any(
        remaining[0].get(k) != v
        for k, v in dict(
            point_index=4, coordinate="z_obs", component="lsst_z", status="INCONCLUSIVE"
        ).items()
    ):
        raise ValueError(
            "this bounded diagnostic requires only point4 z_obs lsst_z unresolved"
        )
    mdf = manifest["mdf_precision_reference"]
    if verify_resolution_reference(mdf["path"], source_root) != mdf:
        raise ValueError("MDF source reference changed")
    with (
        np.load(path / "QUALIFICATION_POINTS.npz", allow_pickle=False) as current,
        np.load(
            Path(mdf["path"]) / "QUALIFICATION_POINTS.npz", allow_pickle=False
        ) as previous,
    ):
        for key in ("x", "row_indices", "origins", "coordinate_names"):
            if not np.array_equal(current[key], previous[key]):
                raise ValueError(f"resolution reference points changed: {key}")
    return dict(
        path=str(path),
        final_sha256=sha256_file(path / "FINAL.json"),
        manifest_sha256=sha256_file(path / "RUN_MANIFEST.json"),
    )


def prepare(
    root,
    source_root,
    objects=16,
    steps=64,
    draws=128,
    gradient_isolation=False,
    redshift_decomposition=False,
    photometry_reference=False,
    full_decoder_reference=None,
    mdf_precision_reference=None,
    target_resolution_reference=None,
    redshift_precision_reference=None,
    precision_night_reference=None,
    qualified_night_root=None,
    local_arm="C",
    controlled_optimization=False,
    support_probe_root=None,
    long_optimization=False,
):
    root, source_root = root.resolve(), source_root.resolve()
    if (
        sum(
            (
                gradient_isolation,
                redshift_decomposition,
                photometry_reference,
                full_decoder_reference is not None,
                mdf_precision_reference is not None,
                target_resolution_reference is not None,
                redshift_precision_reference is not None,
                precision_night_reference is not None,
                qualified_night_root is not None,
            )
        )
        > 1
    ):
        raise ValueError("choose only one diagnostic mode")
    if qualified_night_root is not None:
        from scripts.feniks_qualified_local_vi import prepare as prepare_qualified

        return prepare_qualified(
            root,
            source_root,
            qualified_night_root,
            objects,
            steps,
            draws,
            local_arm,
            controlled=controlled_optimization,
            support_probe_root=support_probe_root,
            long_optimization=long_optimization,
        )
    if controlled_optimization or support_probe_root is not None or long_optimization:
        raise ValueError("controlled optimization requires a qualified night")
    if root.exists():
        raise FileExistsError(f"preserve existing diagnostic: {root}")
    qualification_reference = (
        None
        if full_decoder_reference is None
        else verify_photometry_reference(full_decoder_reference, source_root)
    )
    mdf_reference = None
    redshift_reference = None
    night_reference = None
    if precision_night_reference is not None:
        from scripts.feniks_precision_night import verify_reference

        night_reference = verify_reference(precision_night_reference, source_root)
        mdf_reference = read(Path(night_reference["path"]) / "RUN_MANIFEST.json")[
            "mdf_precision_reference"
        ]
    if redshift_precision_reference is not None:
        redshift_reference = verify_redshift_precision_reference(
            redshift_precision_reference, source_root
        )
        mdf_reference = read(Path(redshift_reference["path"]) / "RUN_MANIFEST.json")[
            "mdf_precision_reference"
        ]
    if mdf_precision_reference is not None or target_resolution_reference is not None:
        mdf_reference = (
            verify_resolution_reference(target_resolution_reference, source_root)
            if target_resolution_reference is not None
            else verify_decoder_reference(mdf_precision_reference, source_root)
        )
    if mdf_reference is not None:
        objects = read(Path(mdf_reference["path"]) / "RUN_MANIFEST.json")[
            "objects_per_group"
        ]
    if not 2 <= objects <= 16 or not 1 <= steps <= 64 or not 32 <= draws <= 128:
        raise ValueError(
            "bounded budget: 2..16 objects/group, 1..64 steps, 32..128 evaluation draws"
        )
    old = read(source_root / "RUN_MANIFEST.json")
    artifact = verified_artifacts(read(source_root / "arms/B/ARM_COMPLETE.json"))
    if (
        artifact.get("status") != "COMPLETE"
        or artifact.get("prior_bitwise_unchanged") is not True
    ):
        raise ValueError("source B must be certified complete with frozen prior")
    if artifact.get("truth_used_for_training_or_checkpoint_selection") is not False:
        raise ValueError("source training truth boundary absent")
    config = observed_only_config(load_config(artifact["config"]))
    config["catalog_path"] = old["dataset"]["path"]
    if sha256_file(Path(config["catalog_path"])) != old["dataset"]["sha256"]:
        raise ValueError("source dataset changed")
    cohort = old["cohorts"]["validation_pilot"]
    if sha256_file(Path(cohort["path"])) != cohort["sha256"]:
        raise ValueError("source cohort changed")
    rows, audit = _representative_observed_rows(
        config, np.load(cohort["path"], allow_pickle=False), count=objects
    )
    if len(np.unique(rows)) != objects:
        raise ValueError("duplicate cohort rows")
    photometry = load_photometry_arrays_from_config(
        config, batch_size=objects, row_indices=rows
    )
    if not np.all(photometry.mask) or not np.all(
        np.isfinite(photometry.flux_err) & (photometry.flux_err > 0)
    ):
        raise ValueError(
            "unsupported incomplete photometry; no silent cohort filtering"
        )
    train = old["cohorts"]["train"]
    if (
        sha256_file(Path(train["path"])) != train["sha256"]
        or np.intersect1d(rows, np.load(train["path"], allow_pickle=False)).size
    ):
        raise ValueError("training/cohort overlap or changed identities")
    cache_receipt = read(source_root / "TRAINING_CACHE_FROZEN.json")
    cache = Path(cache_receipt["path"])
    for path, expected in (
        (cache, cache_receipt["sha256"]),
        (cache.with_suffix(".npz.json"), cache_receipt["sidecar_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError("source simulation cache changed")
    root.mkdir(parents=True)
    np.save(root / "observed_rows.npy", rows, allow_pickle=False)
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = dict(
        method="fixed_parent_same_family_local_vi_diagnostic_v1",
        mode="precision_night"
        if precision_night_reference is not None
        else "redshift_precision_audit"
        if redshift_precision_reference is not None
        else "target_resolution_audit"
        if target_resolution_reference is not None
        else "mdf_precision_qualification"
        if mdf_precision_reference is not None
        else "full_decoder_qualification"
        if full_decoder_reference is not None
        else "photometry_reference"
        if photometry_reference
        else "redshift_decomposition"
        if redshift_decomposition
        else "gradient_isolation"
        if gradient_isolation
        else "local_vi",
        status="PREPARED",
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_root=str(source_root),
        source=artifact,
        dataset=old["dataset"],
        source_manifest_sha256=sha256_file(source_root / "RUN_MANIFEST.json"),
        cache=cache_receipt,
        config_sha256=sha256_file(root / "config.yaml"),
        rows_sha256=sha256_file(root / "observed_rows.npy"),
        cohort=audit,
        catalogue_columns=required_catalog_columns(config),
        removed_legacy_read_references=[
            "truth",
            "redshift.column",
            "redshift.truth_column",
            "model.parameter_columns",
            "extra_columns",
        ],
        objects_per_group=objects,
        steps=steps,
        evaluation_draws=draws,
        starts=2,
        gradient_draws=4,
        learning_rate=0.001,
        seed=260908,
        seconds=2400
        if (redshift_decomposition or photometry_reference)
        else 1080
        if gradient_isolation
        else 9900,
        maximum_decoder_evaluations=1000
        if (redshift_decomposition or photometry_reference)
        else 500
        if gradient_isolation
        else 45000,
        maximum_gpus=1,
        maximum_nodes=1,
        allocation_gpu_hours=0.75
        if (redshift_decomposition or photometry_reference)
        else 1 / 3
        if gradient_isolation
        else 3,
        decomposition_cache_indices=[0, 1, 2]
        if (redshift_decomposition or photometry_reference)
        else [],
        truth_used=False,
        scientific_promotion=False,
        population_training_started=False,
        interpretation="diagnostic only; observed validation has been used in earlier experiments",
    )
    if full_decoder_reference is not None or mdf_reference is not None:
        manifest.update(
            qualification_reference=qualification_reference,
            seconds=4800,
            maximum_decoder_evaluations=6000,
            allocation_gpu_hours=1.5,
            decomposition_cache_indices=[0, 1, 2],
            steps=0,
            starts=0,
            candidate_integrator="merged_gauss4_v1",
        )
    if mdf_reference is not None:
        manifest.update(
            mdf_precision_reference=mdf_reference,
            candidate_mdf_weight_precision="float64_v1",
        )
        if (
            sha256_file(root / "observed_rows.npy")
            != read(Path(mdf_reference["path"]) / "RUN_MANIFEST.json")["rows_sha256"]
        ):
            raise ValueError("MDF comparison must replay identical observed rows")
    if target_resolution_reference is not None:
        manifest.update(
            seconds=2400, maximum_decoder_evaluations=1000, allocation_gpu_hours=0.75
        )
    if redshift_reference is not None:
        manifest.update(
            redshift_precision_reference=redshift_reference,
            seconds=2400,
            maximum_decoder_evaluations=1000,
            allocation_gpu_hours=0.75,
        )
    if night_reference is not None:
        recipe_path = Path("configs/experiments/feniks_sc_drws_precision_night.yaml")
        recipe = yaml.safe_load(recipe_path.read_text())
        manifest.update(
            precision_night_reference=night_reference,
            night_recipe=recipe,
            night_recipe_sha256=sha256_file(recipe_path),
            seconds=4800,
            maximum_decoder_evaluations=6000,
            allocation_gpu_hours=10,
            interpretation="gated fixed-parent numerical pilot; no population promotion",
        )
    write(root / "RUN_MANIFEST.json", manifest)
    return manifest


def finite_json(value):
    """Nonfinite diagnostics are explicit nulls, never converted into success."""
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def contract_audit(
    root, model, config, stats, spec, context, model_args, observation, target, budget
):
    features = make_encoder_features(*observation[:2], stats, observation.mask)
    sleep = _sleep_runtime_config(config, stats)
    result = {"analytic_controls": analytic_controls()}
    result["features_delta"] = assert_close(
        "sleep/inference features",
        _sleep_encoder_features(*observation, sleep),
        features,
    )
    altered = observation._replace(mask=observation.mask.at[:, 0].set(False))
    result["masked_features_delta"] = assert_close(
        "masked sleep/inference features",
        _sleep_encoder_features(*altered, sleep),
        make_encoder_features(altered.flux, altered.flux_err, stats, altered.mask),
    )
    parameters, encoded_context = initialize(model, features)
    key = jax.random.PRNGKey(61)
    x, logq = sample(model.encoder, parameters, encoded_context, key, 4)
    original = sample_posterior(model, key, features, 4)
    result["amortized_sample_delta"] = assert_close(
        "amortized/local sample", x, original.x
    )
    result["amortized_logq_delta"] = assert_close(
        "amortized/local density", logq, original.logq
    )
    result["inverse_logq_delta"] = assert_close(
        "inverse logq", logq, log_prob(model.encoder, parameters, encoded_context, x)
    )
    result["canonical_logq_delta"] = assert_close(
        "posterior_log_prob", logq, posterior_log_prob(model, features, x)
    )
    eqx.tree_serialise_leaves(root / "AUDIT_LOCAL.eqx", parameters)
    restored = eqx.tree_deserialise_leaves(root / "AUDIT_LOCAL.eqx", parameters)
    result["serialization_delta"] = assert_close(
        "local restore", logq, log_prob(model.encoder, restored, encoded_context, x)
    )
    # Audit at parent-generated inputs as well as q inputs, with exact unit scaling.
    budget.charge(8)
    canonical = target(x, observation)
    raw = _model_flux_from_x_sample_chunks(
        x, spec, context, model_args, spec.names, sample_chunk_size=1
    )
    calibration = {"calibration": config.get("calibration", {})}
    calibrated = _apply_model_calibration(model, raw, calibration)
    exported = posterior_log_target_from_model_flux(
        model, x, observation, spec, calibrated, config["amortized"]["likelihood"]
    )
    result["target_delta"] = assert_close(
        "ELBO/export target", canonical.logtarget, exported.logtarget
    )
    result["x_roundtrip_delta"] = assert_close(
        "theta/x", theta_to_x(x_to_theta(x, spec), spec), x, atol=0.01, rtol=0.001
    )
    effective = photometric_sigma_eff(
        *observation[:1],
        calibrated,
        observation.flux_err,
        observation.mask,
        error_floor_frac=0,
        error_jitter=0,
    )
    ratio = effective / observation.flux_err
    result["noise_scale_delta"] = assert_close(
        "noise/likelihood scale",
        np.asarray(ratio)[np.broadcast_to(np.asarray(observation.mask), ratio.shape)],
        np.ones(
            int(np.sum(np.broadcast_to(np.asarray(observation.mask), ratio.shape)))
        ),
    )
    cache = read(root / "RUN_MANIFEST.json")["cache"]
    with np.load(cache["path"], allow_pickle=False) as bank:
        cache_x = jnp.asarray(bank["x"][:2])[:, None, :]
        expected = bank["model_flux"][:2, None, :]
    budget.charge(2)
    cached_live = target(cache_x, observation).model_flux_raw
    unit = np.maximum(np.abs(expected), np.asarray(stats.flux_scale)[None, None, :])
    result["cache_flux_scaled_delta"] = assert_close(
        "cache/live flux", cached_live / unit, expected / unit, atol=0.002, rtol=0.002
    )
    # Promote the perturbation input itself: an x64 target alone cannot recover
    # perturbations already rounded in a float32 finite-difference stencil.
    point = cache_x[0:1]
    if getattr(spec, "arithmetic_precision", "float32_legacy") == "float64_v1":
        point = point.astype(jnp.float64)
    direction = jax.random.normal(
        jax.random.PRNGKey(62), point.shape, dtype=point.dtype
    )
    direction /= jnp.linalg.norm(direction)

    def objective(z):
        values = target(z, observation)
        return {
            name: jnp.sum(getattr(values, name))
            for name in ("loglike", "logprior", "logtarget")
        }

    gradient = gradient_audit(objective, point, direction, budget)
    write(root / "GRADIENT_AUDIT.json", finite_json(gradient))
    result["gradient_audit"] = {
        "status": gradient["status"],
        "path": str(root / "GRADIENT_AUDIT.json"),
        "sha256": sha256_file(root / "GRADIENT_AUDIT.json"),
    }
    if gradient["status"] != "PASS":
        result.update(
            status=gradient["status"], truth_used=False, scientific_promotion=False
        )
        write(root / "CONTRACT_AUDIT.json", finite_json(result))
        raise ValueError(
            f"gradient audit {gradient['status']}; inspect {root / 'GRADIENT_AUDIT.json'}; "
            "no local optimization started"
        )
    result.update(status="PASS", truth_used=False, scientific_promotion=False)
    write(root / "CONTRACT_AUDIT.json", result)
    return result


def generate_cases(model, arrays, config, target, budget, seed, count):
    """Generate at each fixed observed context, then select on noisy r flux."""
    stats_sleep = config["sleep_runtime"]
    selection_band = int(stats_sleep["selection_band_index"])
    cases = []
    for index in range(count):
        observation = PosteriorObservation(
            *[
                jnp.asarray(value[index : index + 1])
                for value in (arrays.flux, arrays.flux_err, arrays.mask)
            ]
        )
        cases.append(("observed", index, observation, None))
        accepted = None
        # Explicit bounded rejection; no changing the generative distribution.
        for attempt in range(8):
            key = jax.random.fold_in(jax.random.PRNGKey(seed), index * 8 + attempt)
            prior_key, noise_key = jax.random.split(key)
            x = model.prior.sample(prior_key, 4)[:, None, :]
            budget.charge(4)
            values = target(x, observation)
            errors = jnp.broadcast_to(observation.flux_err, values.model_flux.shape)
            noise, _ = _sample_sleep_noise(
                noise_key,
                errors,
                sleep=stats_sleep,
                likelihood_config=config["amortized"]["likelihood"],
            )
            noisy = values.model_flux + noise
            valid = _sleep_observed_selection_mask(
                noisy[:, 0],
                values.physical_valid[:, 0],
                band_index=selection_band,
                flux_min=stats_sleep["selection_flux_min_fnu_cgs"],
                observed_mask=jnp.broadcast_to(
                    observation.mask, (4, observation.mask.shape[-1])
                ),
            )
            choices = np.flatnonzero(np.asarray(valid))
            if choices.size:
                chosen = int(choices[0])
                accepted = (
                    "simulated",
                    index,
                    observation._replace(flux=noisy[chosen]),
                    x[chosen],
                )
                break
        if accepted is None:
            raise ValueError(f"simulation selection exhausted at fixed context {index}")
        cases.append(accepted)
    return cases


def evaluate_distribution(
    folder,
    encoder,
    parameters,
    encoded_context,
    observation,
    target,
    spec,
    budget,
    seed,
    draws,
    generated=None,
):
    chunks = []
    for replicate in range(2):
        x, logq = sample(
            encoder,
            parameters,
            encoded_context,
            jax.random.PRNGKey(seed + replicate),
            draws,
        )
        parts = []
        for start in range(0, draws, 4):
            budget.charge(min(4, draws - start))
            parts.append(jax.device_get(target(x[start : start + 4], observation)))
        parts = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *parts)
        assert_close(
            "evaluation inverse logq",
            logq,
            log_prob(encoder, parameters, encoded_context, x),
            atol=0.002,
            rtol=0.001,
        )
        chunks.append(
            dict(
                x=np.asarray(x),
                logq=np.asarray(logq),
                logprior=parts.logprior,
                loglike=parts.loglike,
                model_flux=parts.model_flux,
            )
        )
    values = {
        name: np.concatenate([p[name] for p in chunks], axis=0) for name in chunks[0]
    }
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "direct_draws.npz", **values)
    physical = np.asarray(x_to_theta(jnp.asarray(values["x"]), spec))[:, 0, :]
    frame = pd.DataFrame(physical, columns=spec.names)
    frame["object_id"] = 0
    frame["sample_id"] = np.arange(len(frame))
    for name in ("logq", "logprior", "loglike"):
        frame[name] = values[name][:, 0]
    summary, _ = summarize_truth_free_joint_bank(
        frame, parameter_names=spec.names, identity_column="object_id"
    )
    logz, replicates = [], []
    for part in chunks:
        weights = part["loglike"] + part["logprior"] - part["logq"]
        logz.append(float(jax.scipy.special.logsumexp(weights[:, 0]) - np.log(draws)))
        logw = weights[:, 0].astype(np.float64)
        normalized = np.exp(logw - np.max(logw))
        normalized /= np.sum(normalized)
        replicates.append(
            dict(
                draws=draws,
                raw_ess=float(1 / np.sum(normalized**2)),
                maximum_raw_weight=float(np.max(normalized)),
                latent_mean=part["x"][:, 0].mean(axis=0).tolist(),
                latent_std=part["x"][:, 0].std(axis=0).tolist(),
            )
        )
    residual = np.asarray(
        photometric_normalized_residual(
            observation.flux,
            values["model_flux"],
            observation.flux_err,
            observation.mask,
            error_floor_frac=0,
            error_jitter=0,
        )
    )[:, 0, :]
    valid_residuals = residual[:, np.asarray(observation.mask)[0]]
    summary.update(
        replicate_log_evidence=logz,
        independent_replicates=replicates,
        pooled_draws=2 * draws,
        density_means={
            name: float(np.mean(values[name]))
            for name in ("logq", "logprior", "loglike")
        },
        evaluation_contract="Two fresh direct-draw replicates; pooled support is not either replicate's support. No resampling or best-start selection.",
        replicate_abs_log_evidence_delta=abs(logz[0] - logz[1]),
        negative_elbo=float(
            np.mean(values["logq"] - values["logprior"] - values["loglike"])
        ),
        residual_median_abs=float(np.median(np.abs(valid_residuals))),
        residual_rms=float(np.sqrt(np.mean(valid_residuals**2))),
        residual_fraction_abs_gt5=float(np.mean(np.abs(valid_residuals) > 5)),
        residual_per_band_median_abs=np.median(np.abs(residual), axis=0).tolist(),
        latent_std=np.std(values["x"][:, 0], axis=0).tolist(),
        latent_covariance=np.cov(values["x"][:, 0], rowvar=False).tolist(),
        direct_draws_sha256=sha256_file(folder / "direct_draws.npz"),
        truth_used=False,
        scientific_promotion=False,
    )
    if generated is not None:
        budget.charge(1)
        generated_values = target(generated[None], observation)
        generated_residual = np.asarray(
            photometric_normalized_residual(
                observation.flux,
                generated_values.model_flux,
                observation.flux_err,
                observation.mask,
                error_floor_frac=0,
                error_jitter=0,
            )
        )[0, 0]
        summary["generated_reference_residual_by_band"] = generated_residual.tolist()
        summary["generated_reference_flux"] = np.asarray(generated_values.model_flux)[
            0, 0
        ].tolist()
        summary["generated_parameter_ranks"] = np.mean(
            values["x"][:, 0] < np.asarray(generated)[0], axis=0
        ).tolist()
        summary["generated_loglike_rank"] = float(
            np.mean(values["loglike"] < np.asarray(generated_values.loglike))
        )
        summary["generated_ranks_interpretation"] = (
            "small-cohort diagnostic, not a calibration certificate; generated labels never used in optimization"
        )
    write(folder / "SUMMARY.json", finite_json(summary))
    return summary


def run(root):
    manifest = read(root / "RUN_MANIFEST.json")
    require_git_commit(Path(__file__).resolve().parents[1], manifest["code_commit"])
    if jax.default_backend() != "gpu" or len(jax.local_devices()) != 1:
        raise ValueError("expected exactly one visible GPU")
    if (root / "FINAL.json").exists():
        raise FileExistsError("final receipt exists; preserve this diagnostic")
    if (root / "PROGRESS.json").exists():
        raise FileExistsError("partial attempt preserved; choose a new diagnostic root")
    budget = Budget(manifest["seconds"], manifest["maximum_decoder_evaluations"])
    write(root / "PROGRESS.json", {"stage": "loading", "budget": budget.snapshot()})
    try:
        if "support_probe" in manifest:
            from scripts.feniks_support_probe import verify_source

            verify_source(manifest["support_probe"])
        if "qualified_night" in manifest:
            from scripts.feniks_qualified_local_vi import verify_night

            ref = manifest["qualified_night"]
            verified = verify_night(Path(ref["path"]), ref["arm"])
            if verified["inventory_sha256"] != ref["inventory_sha256"]:
                raise ValueError("qualified night inventory changed after preparation")
        if manifest.get("mode") == "precision_night":
            recipe_path = (
                Path(__file__).resolve().parents[1]
                / "configs/experiments/feniks_sc_drws_precision_night.yaml"
            )
            if (
                sha256_file(recipe_path) != manifest["night_recipe_sha256"]
                or yaml.safe_load(recipe_path.read_text()) != manifest["night_recipe"]
            ):
                raise ValueError("night recipe differs from immutable source")
        for name, expected in (
            ("config.yaml", manifest["config_sha256"]),
            ("observed_rows.npy", manifest["rows_sha256"]),
        ):
            if sha256_file(root / name) != expected:
                raise ValueError("diagnostic input changed")
        verified_artifacts(manifest["source"])
        if (
            sha256_file(Path(manifest["dataset"]["path"]))
            != manifest["dataset"]["sha256"]
        ):
            raise ValueError("dataset changed")
        config = load_config(root / "config.yaml")
        config["catalog_path"] = manifest["dataset"]["path"]
        check_config(config)
        stats = read_feature_stats(Path(manifest["source"]["feature_stats"]))
        rows = np.load(root / "observed_rows.npy", allow_pickle=False)
        arrays = load_photometry_arrays_from_config(
            config, batch_size=len(rows), row_indices=rows
        )
        if tuple(arrays.band_names) != tuple(stats.band_names) or not np.array_equal(
            arrays.row_index, rows
        ):
            raise ValueError("band or object identity order mismatch")
        if not np.all(arrays.mask) or not np.all(
            np.isfinite(arrays.flux_err) & (arrays.flux_err > 0)
        ):
            raise ValueError(
                "initial local diagnostic requires complete finite photometry; do not silently filter cases"
            )
        model = load_checkpoint(Path(manifest["source"]["checkpoint"]), config)
        spec = latent_spec_from_config(config)
        topology = conditional_flow_topology(model.encoder, coordinate_names=spec.names)
        if topology["minimum_transform_count"] < 2:
            raise ValueError("source topology not fully covered")
        frozen_hash = _array_tree_sha256(
            (model.prior, model.sed_scale, model.band_calibration)
        )
        if (
            "qualified_night" in manifest
            and frozen_hash != manifest["qualified_night"]["frozen_array_sha256"]
        ):
            raise ValueError("qualified prior/calibration arrays changed")
        cache = manifest["cache"]
        for path, expected in (
            (Path(cache["path"]), cache["sha256"]),
            (Path(cache["path"]).with_suffix(".npz.json"), cache["sidecar_sha256"]),
        ):
            if sha256_file(path) != expected:
                raise ValueError("source cache changed")
        if (
            _array_tree_sha256(model.prior)
            != read(Path(cache["path"]).with_suffix(".npz.json"))[
                "prior_fingerprint_sha256"
            ]
        ):
            raise ValueError("cache generator prior differs")
        filters = load_filters(config["bands"])
        context = load_context(
            config["ssp_path"],
            filters,
            n_sfh_bins=int(config.get("model", {}).get("n_sfh_bins", 96)),
            cosmos_config=config.get("cosmos_sed"),
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=config.get("model"),
        )
        model_args = dynamic_model_args(context)
        likelihood = config["amortized"]["likelihood"]
        calibration = {"calibration": config.get("calibration", {})}

        @eqx.filter_jit
        def target(x, observation):
            return posterior_log_target(
                model,
                x,
                observation,
                spec,
                context,
                model_args,
                spec.names,
                likelihood,
                calibration,
            )

        first = PosteriorObservation(
            *[jnp.asarray(x[:1]) for x in (arrays.flux, arrays.flux_err, arrays.mask)]
        )
        if manifest.get("mode") in {
            "redshift_decomposition",
            "photometry_reference",
            "full_decoder_qualification",
            "mdf_precision_qualification",
            "target_resolution_audit",
            "redshift_precision_audit",
            "precision_night",
        }:
            from euclid_dsps.amortized.redshift_decomposition import decompose_redshift
            from euclid_dsps.calibration import (
                global_sed_scale_config,
                per_band_flux_calibration_config,
            )

            if (
                global_sed_scale_config(calibration).enabled
                or per_band_flux_calibration_config(calibration).enabled
            ):
                raise ValueError(
                    "branch diagnostic currently requires disabled flux calibration; never silently remove it"
                )
            with np.load(cache["path"], allow_pickle=False) as bank:
                points = jnp.asarray(bank["x"][manifest["decomposition_cache_indices"]])
                expected_flux = bank["model_flux"][
                    manifest["decomposition_cache_indices"]
                ]
            cache_checks = []
            for index, point in enumerate(points):
                budget.charge(1)
                live = target(point[None, None, :], first).model_flux_raw.reshape(-1)
                unit = np.maximum(
                    np.abs(expected_flux[index]), np.asarray(stats.flux_scale)
                )
                delta = assert_close(
                    "cache/live flux",
                    live / unit,
                    expected_flux[index] / unit,
                    atol=0.002,
                    rtol=0.002,
                )
                cache_checks.append(dict(point_index=index, scaled_max_delta=delta))
            write(
                root / "CACHE_FLUX_AUDIT.json", dict(status="PASS", points=cache_checks)
            )

            def branch_progress(index, branch, measurements, completed):
                pd.DataFrame(measurements).to_csv(
                    root / "redshift_decomposition.csv", index=False
                )
                write(
                    root / "REDSHIFT_DECOMPOSITION_PARTIAL.json",
                    finite_json(dict(points=completed)),
                )
                write(
                    root / "PROGRESS.json",
                    dict(
                        stage="redshift_decomposition",
                        point_index=index,
                        branch=branch,
                        budget=budget.snapshot(),
                    ),
                )

            if manifest["mode"] in {
                "full_decoder_qualification",
                "mdf_precision_qualification",
                "target_resolution_audit",
                "redshift_precision_audit",
                "precision_night",
            }:
                from euclid_dsps.amortized.decoder_qualification import qualify
                from euclid_dsps.model import photometry_numerics

                redshift_precision_mode = manifest["mode"] == "redshift_precision_audit"
                night_mode = manifest["mode"] == "precision_night"
                resolution_mode = (
                    manifest["mode"] == "target_resolution_audit"
                    or redshift_precision_mode
                    or night_mode
                )
                mdf_mode = (
                    manifest["mode"] == "mdf_precision_qualification" or resolution_mode
                )
                reference = (
                    manifest["mdf_precision_reference"]
                    if mdf_mode
                    else manifest["qualification_reference"]
                )
                verify_reference = (
                    verify_resolution_reference
                    if resolution_mode
                    else verify_decoder_reference
                    if mdf_mode
                    else verify_photometry_reference
                )
                if (
                    verify_reference(reference["path"], manifest["source_root"])
                    != reference
                ):
                    raise ValueError("qualification reference receipt changed")
                if (
                    photometry_numerics(config.get("model"))["integrator"]
                    != "legacy_trapezoid_v1"
                ):
                    raise ValueError(
                        "qualification requires a historical source target"
                    )
                corrected = copy.copy(context)
                corrected.model_config = dict(
                    context.model_config,
                    photometry_integrator=manifest["candidate_integrator"],
                )
                baseline_target = target
                labels = ("legacy", "merged")
                if mdf_mode:
                    if (
                        context.model_config.get("stellar_metallicity_model")
                        != "lognormal_mdf_fixed_scatter"
                    ):
                        raise ValueError(
                            "MDF precision diagnostic requires fixed-scatter MDF"
                        )
                    baseline_context = copy.copy(corrected)
                    baseline_args = dynamic_model_args(baseline_context)

                    @eqx.filter_jit
                    def baseline_target(x, observation):
                        return posterior_log_target(
                            model,
                            x,
                            observation,
                            spec,
                            baseline_context,
                            baseline_args,
                            spec.names,
                            likelihood,
                            calibration,
                        )

                    corrected.model_config = dict(
                        corrected.model_config,
                        mdf_weight_precision=manifest["candidate_mdf_weight_precision"],
                    )
                    labels = ("merged_mdf32", "merged_mdf64")
                candidate_spec = spec
                candidate_likelihood = likelihood
                if night_mode:
                    from scripts.feniks_precision_night import candidate_configuration

                    corrected.model_config["spline_precision"] = "float64_v1"
                corrected_args = dynamic_model_args(corrected)
                candidate_config = copy.deepcopy(config)
                candidate_config["model"]["photometry_integrator"] = manifest[
                    "candidate_integrator"
                ]
                if mdf_mode:
                    candidate_config["model"]["mdf_weight_precision"] = manifest[
                        "candidate_mdf_weight_precision"
                    ]
                if night_mode:
                    candidate_config = candidate_configuration(candidate_config)
                    candidate_spec = latent_spec_from_config(candidate_config)
                    candidate_likelihood = candidate_config["amortized"]["likelihood"]
                (root / "candidate_config.yaml").write_text(
                    yaml.safe_dump(candidate_config, sort_keys=False)
                )

                @eqx.filter_jit
                def corrected_target(x, observation):
                    return posterior_log_target(
                        model,
                        x,
                        observation,
                        candidate_spec,
                        corrected,
                        corrected_args,
                        spec.names,
                        candidate_likelihood,
                        calibration,
                    )

                n_contexts = min(3, len(rows))
                contexts = [
                    PosteriorObservation(
                        *[
                            jnp.asarray(a[i : i + 1])
                            for a in (arrays.flux, arrays.flux_err, arrays.mask)
                        ]
                    )
                    for i in range(n_contexts)
                ]
                features = make_encoder_features(
                    jnp.asarray(arrays.flux[:n_contexts]),
                    jnp.asarray(arrays.flux_err[:n_contexts]),
                    stats,
                    jnp.asarray(arrays.mask[:n_contexts]),
                )
                q_points = sample_posterior(
                    model, jax.random.PRNGKey(260908), features, 1
                ).x.reshape(-1, len(spec.names))
                points = jnp.concatenate((points, q_points), axis=0)
                observations = [contexts[i % n_contexts] for i in range(3)] + contexts
                np.savez_compressed(
                    root / "QUALIFICATION_POINTS.npz",
                    x=np.asarray(points),
                    row_indices=np.array(
                        [rows[i % n_contexts] for i in range(3)]
                        + list(rows[:n_contexts])
                    ),
                    origins=np.array(
                        ["frozen_parent_cache"] * 3 + ["direct_q"] * n_contexts
                    ),
                    coordinate_names=np.asarray(spec.names),
                )
                if mdf_mode:
                    with (
                        np.load(
                            Path(reference["path"]) / "QUALIFICATION_POINTS.npz",
                            allow_pickle=False,
                        ) as previous,
                        np.load(
                            root / "QUALIFICATION_POINTS.npz", allow_pickle=False
                        ) as replay,
                    ):
                        for key in ("x", "row_indices", "origins", "coordinate_names"):
                            if not np.array_equal(previous[key], replay[key]):
                                raise ValueError(f"qualification replay differs: {key}")
                if mdf_mode and not resolution_mode:
                    from euclid_dsps.amortized.mdf_precision import probe
                    from euclid_dsps.model import (
                        _context_ssp_lgmet,
                        log10_stellar_metallicity_to_absolute_jax,
                    )

                    theta = x_to_theta(points, spec)
                    centers = jax.vmap(
                        lambda m: log10_stellar_metallicity_to_absolute_jax(
                            m, context.z_sun
                        )
                    )(theta[:, spec.names.index("log10_stellar_metallicity")])
                    scatter = float(
                        context.model_config.get("stellar_metallicity_scatter_dex", 0.2)
                    )
                    weight_report, weight_rows = probe(
                        _context_ssp_lgmet(context),
                        np.asarray(centers),
                        scatter,
                        budget,
                    )
                    write(root / "MDF_WEIGHT_PROBES.json", finite_json(weight_report))
                    pd.DataFrame(weight_rows).to_csv(
                        root / "mdf_weight_probes.csv", index=False
                    )

                def qualification_progress(label, i, stage, measurements, completed):
                    pd.DataFrame(measurements).to_csv(
                        root / "decoder_qualification.csv", index=False
                    )
                    write(
                        root / "FULL_DECODER_PARTIAL.json",
                        finite_json(dict(cases=completed)),
                    )
                    write(
                        root / "PROGRESS.json",
                        dict(
                            stage="full_decoder_qualification",
                            variant=label,
                            point_index=i,
                            branch=stage,
                            budget=budget.snapshot(),
                        ),
                    )

                if night_mode:
                    from scripts.feniks_precision_night import verify_reference

                    ref = manifest["precision_night_reference"]
                    if verify_reference(ref["path"], manifest["source_root"]) != ref:
                        raise ValueError("night source changed")
                    result, _ = qualify(
                        target,
                        corrected_target,
                        points,
                        observations,
                        budget,
                        names=spec.names,
                        bands=stats.band_names,
                        progress=qualification_progress,
                        labels=("legacy", "spline64"),
                        candidate_only=True,
                        candidate_float64=True,
                    )
                elif redshift_precision_mode:
                    from euclid_dsps.amortized.redshift_precision import (
                        analyze,
                        collect,
                    )

                    resolution_reference = manifest["redshift_precision_reference"]
                    if (
                        verify_redshift_precision_reference(
                            resolution_reference["path"], manifest["source_root"]
                        )
                        != resolution_reference
                    ):
                        raise ValueError("redshift precision source changed")

                    def redshift_progress(snapshot):
                        write(
                            root / "REDSHIFT_PRECISION_SNAPSHOT.json",
                            finite_json(snapshot),
                        )
                        partial, table = analyze(snapshot)
                        write(
                            root / "REDSHIFT_PRECISION_PARTIAL.json",
                            finite_json(partial),
                        )
                        pd.DataFrame(table).to_csv(
                            root / "redshift_precision.csv", index=False
                        )
                        write(
                            root / "PROGRESS.json",
                            dict(
                                stage="redshift_precision_audit",
                                completed_branches=len(snapshot["branches"]),
                                budget=budget.snapshot(),
                            ),
                        )

                    snapshot = collect(
                        corrected,
                        spec,
                        points[4],
                        observations[4],
                        corrected_target,
                        read(
                            Path(resolution_reference["path"])
                            / "TARGET_RESOLUTION.json"
                        ),
                        budget,
                        bands=tuple(stats.band_names),
                        progress=redshift_progress,
                    )
                    result, _ = analyze(snapshot)
                elif resolution_mode:
                    from euclid_dsps.amortized.target_resolution import analyze, collect

                    def resolution_progress(snapshot):
                        write(root / "TARGET_RESOLUTION_SNAPSHOT.json", snapshot)
                        partial, table = analyze(snapshot)
                        write(
                            root / "TARGET_RESOLUTION_PARTIAL.json",
                            finite_json(partial),
                        )
                        pd.DataFrame(table).to_csv(
                            root / "target_resolution.csv", index=False
                        )
                        write(
                            root / "PROGRESS.json",
                            dict(
                                stage="target_resolution_audit",
                                completed_checks=len(snapshot["cases"]),
                                budget=budget.snapshot(),
                            ),
                        )

                    source_report = read(
                        Path(reference["path"]) / "FULL_DECODER_QUALIFICATION.json"
                    )
                    snapshot = collect(
                        corrected_target,
                        points,
                        observations,
                        source_report,
                        budget,
                        names=tuple(spec.names),
                        bands=tuple(stats.band_names),
                        progress=resolution_progress,
                    )
                    result, _ = analyze(snapshot)
                else:
                    result, _ = qualify(
                        baseline_target,
                        corrected_target,
                        points,
                        observations,
                        budget,
                        names=spec.names,
                        bands=stats.band_names,
                        progress=qualification_progress,
                        labels=labels,
                    )
                result["candidate_numerics"] = photometry_numerics(
                    corrected.model_config
                )
                result["old_sleep_cache_used_only_for_parameters_and_legacy_check"] = (
                    True
                )
                report_name = "FULL_DECODER_QUALIFICATION.json"
                artifacts = (
                    "CACHE_FLUX_AUDIT.json",
                    report_name,
                    "decoder_qualification.csv",
                    "QUALIFICATION_POINTS.npz",
                    "candidate_config.yaml",
                )
                if mdf_mode and not resolution_mode:
                    result["precision_scope"] = (
                        "MDF weights and induced contractions only; inputs, assets and downstream casts remain mixed precision"
                    )
                    result["identical_source_points_verified"] = True
                    result["mdf_reference_checks"] = weight_report[
                        "candidate_reference_checks"
                    ]
                    if weight_report["candidate_reference_checks"] != "PASS":
                        result["candidate_numerical_checks"] = "NOT_PASSED"
                        result["next_stage"] = "INVESTIGATE_MDF_WEIGHTS"
                    artifacts += ("MDF_WEIGHT_PROBES.json", "mdf_weight_probes.csv")
                if resolution_mode and not night_mode:
                    report_name = "TARGET_RESOLUTION.json"
                    result["identical_source_points_verified"] = True
                    result["source_reference"] = reference
                    artifacts = (
                        "CACHE_FLUX_AUDIT.json",
                        report_name,
                        "target_resolution.csv",
                        "TARGET_RESOLUTION_SNAPSHOT.json",
                        "QUALIFICATION_POINTS.npz",
                        "candidate_config.yaml",
                    )
                if redshift_precision_mode:
                    report_name = "REDSHIFT_PRECISION.json"
                    result["source_resolution_reference"] = resolution_reference
                    artifacts = (
                        "CACHE_FLUX_AUDIT.json",
                        report_name,
                        "redshift_precision.csv",
                        "REDSHIFT_PRECISION_SNAPSHOT.json",
                        "QUALIFICATION_POINTS.npz",
                        "candidate_config.yaml",
                    )
            elif manifest["mode"] == "photometry_reference":
                from euclid_dsps.amortized.photometry_reference import (
                    analyze_snapshot,
                    export_spectra,
                    write_progress,
                )

                snapshot_arrays = export_spectra(
                    root,
                    context,
                    spec,
                    points,
                    first,
                    budget,
                    band_names=stats.band_names,
                )
                write(
                    root / "SNAPSHOT.json",
                    dict(
                        status="EXPORTED",
                        code_commit=manifest["code_commit"],
                        truth_used=False,
                        archive="FIXED_SPECTRA.npz",
                        sha256=sha256_file(root / "FIXED_SPECTRA.npz"),
                    ),
                )
                result, _ = analyze_snapshot(
                    snapshot_arrays,
                    budget,
                    progress=lambda i, s, r, p: write_progress(
                        root, i, s, r, p, budget
                    ),
                )
                report_name = "PHOTOMETRY_REFERENCE.json"
                artifacts = (
                    "CACHE_FLUX_AUDIT.json",
                    report_name,
                    "photometry_reference.csv",
                    "SNAPSHOT.json",
                    "FIXED_SPECTRA.npz",
                )
            else:
                result, _ = decompose_redshift(
                    context,
                    spec,
                    points,
                    first,
                    budget,
                    band_names=stats.band_names,
                    progress=branch_progress,
                    canonical_flux=lambda point: (
                        target(point[None, None, :], first).model_flux_raw
                    ),
                )
                report_name = "REDSHIFT_DECOMPOSITION.json"
                artifacts = (
                    "CACHE_FLUX_AUDIT.json",
                    report_name,
                    "redshift_decomposition.csv",
                )
            if (
                _array_tree_sha256(
                    (model.prior, model.sed_scale, model.band_calibration)
                )
                != frozen_hash
            ):
                raise ValueError("frozen model changed")
            result.update(
                prior_bitwise_unchanged=True,
                row_index=int(
                    rows[1 if manifest["mode"] == "redshift_precision_audit" else 0]
                ),
                object_id=str(
                    arrays.object_id[
                        1 if manifest["mode"] == "redshift_precision_audit" else 0
                    ]
                ),
                budget=budget.snapshot(),
                runtime=dict(
                    python=sys.version,
                    jax=jax.__version__,
                    backend=jax.default_backend(),
                    devices=[str(d) for d in jax.local_devices()],
                    jax_enable_x64=bool(jax.config.x64_enabled),
                ),
            )
            write(root / report_name, finite_json(result))
            final = dict(
                status=result["status"],
                scientific_promotion=False,
                local_optimization_started=False,
                population_training_started=False,
                truth_used=False,
                budget=budget.snapshot(),
                artifacts={
                    name: dict(path=str(root / name), sha256=sha256_file(root / name))
                    for name in artifacts
                },
            )
            write(root / "FINAL.json", final)
            if manifest["mode"] == "precision_night":
                from scripts.feniks_precision_night import continue_night

                return continue_night(
                    root,
                    manifest,
                    candidate_config,
                    candidate_spec,
                    model,
                    stats,
                    result,
                    budget.snapshot()["elapsed_seconds"],
                )
            return final
        if manifest.get("mode") == "gradient_isolation":
            from euclid_dsps.amortized.gradient_isolation import isolate_gradient

            with np.load(cache["path"], allow_pickle=False) as bank:
                point = jnp.asarray(bank["x"][:1])[:, None, :]
            direction = jax.random.normal(
                jax.random.PRNGKey(62), point.shape, dtype=point.dtype
            )
            direction /= jnp.linalg.norm(direction)

            def progress(label, measurements):
                pd.DataFrame(measurements).to_csv(
                    root / "gradient_isolation.csv", index=False
                )
                write(
                    root / "PROGRESS.json",
                    dict(
                        stage="gradient_isolation",
                        direction=label,
                        budget=budget.snapshot(),
                    ),
                )

            write(
                root / "PROGRESS.json",
                dict(stage="gradient_isolation_jacobian", budget=budget.snapshot()),
            )
            isolated, measurements = isolate_gradient(
                target,
                first,
                point,
                direction,
                budget,
                band_names=stats.band_names,
                coordinate_names=spec.names,
                progress=progress,
            )
            if (
                _array_tree_sha256(
                    (model.prior, model.sed_scale, model.band_calibration)
                )
                != frozen_hash
            ):
                raise ValueError("frozen model changed")
            isolated.update(
                budget=budget.snapshot(),
                prior_bitwise_unchanged=True,
                row_index=int(rows[0]),
                object_id=str(arrays.object_id[0]),
            )
            write(root / "GRADIENT_ISOLATION.json", finite_json(isolated))
            final = dict(
                status="GRADIENT_ISOLATION_COMPLETE",
                budget=budget.snapshot(),
                scientific_promotion=False,
                local_optimization_started=False,
                population_training_started=False,
                truth_used=False,
                artifacts={
                    name: dict(path=str(root / name), sha256=sha256_file(root / name))
                    for name in ("GRADIENT_ISOLATION.json", "gradient_isolation.csv")
                },
            )
            write(root / "FINAL.json", final)
            return final
        write(
            root / "PROGRESS.json",
            dict(stage="contract_audit", budget=budget.snapshot()),
        )
        contract_audit(
            root, model, config, stats, spec, context, model_args, first, target, budget
        )
        config["sleep_runtime"] = _sleep_runtime_config(config, stats)
        write(
            root / "PROGRESS.json", dict(stage="simulation", budget=budget.snapshot())
        )
        cases = generate_cases(
            model, arrays, config, target, budget, manifest["seed"], len(rows)
        )
        np.savez_compressed(
            root / "SIMULATED_INPUTS.npz",
            generated_x=np.concatenate(
                [np.asarray(c[3]) for c in cases if c[0] == "simulated"]
            ),
            flux=np.concatenate(
                [np.asarray(c[2].flux) for c in cases if c[0] == "simulated"]
            ),
            flux_err=arrays.flux_err,
            mask=arrays.mask,
        )
        regimes = manifest.get(
            "optimization_regimes",
            [
                dict(
                    name="original",
                    learning_rate=manifest["learning_rate"],
                    gradient_draws=manifest["gradient_draws"],
                )
            ],
        )
        if "support_probe" in manifest:
            from scripts.feniks_support_probe import check_simulations

            check_simulations(
                Path(manifest["support_probe"]["path"]) / "SIMULATED_INPUTS.npz",
                root / "SIMULATED_INPUTS.npz",
            )
        optimizers = (
            []
            if "support_probe" in manifest
            else [
                make_step(
                    model.encoder,
                    target,
                    draws=r["gradient_draws"],
                    learning_rate=r["learning_rate"],
                )
                for r in regimes
            ]
        )
        completed, started_cases = [], time.monotonic()
        for case_number, (group, index, observation, generated) in enumerate(cases):
            case_start = time.monotonic()
            folder = root / "cases" / f"{group}_{index:03d}"
            features = make_encoder_features(
                observation.flux, observation.flux_err, stats, observation.mask
            )
            initial, encoded_context = initialize(model, features)
            seed = manifest["seed"] + 100000 + case_number * 1000
            if manifest["method"] == "qualified_long_local_vi_v1":
                seed = manifest["seed"] + 3000000 + case_number * 100000
            if "support_probe" in manifest:
                seed += 2000000
            write(
                root / "PROGRESS.json",
                dict(
                    stage="amortized_evaluation",
                    group=group,
                    object=index,
                    cases_complete=len(completed),
                    budget=budget.snapshot(),
                ),
            )
            baseline = evaluate_distribution(
                folder / "amortized",
                model.encoder,
                initial,
                encoded_context,
                observation,
                target,
                spec,
                budget,
                seed,
                manifest["evaluation_draws"],
                generated,
            )
            fitted = []
            for start in range(2 * len(regimes)):
                regime = regimes[start // 2]
                if "support_probe" not in manifest:
                    optimizer, step = optimizers[start // 2]
                initialization = start % 2
                parameters = (
                    initial
                    if initialization == 0
                    else perturb(initial, jax.random.PRNGKey(seed + 20))
                )
                if "support_probe" in manifest:
                    from euclid_dsps.amortized.local_vi_diagnostic import (
                        MixtureParameters,
                        broaden,
                    )

                    source_path = (
                        Path(manifest["support_probe"]["path"])
                        / "cases"
                        / f"{group}_{index:03d}"
                        / f"start_{4 + initialization}"
                        / "parameters.eqx"
                    )
                    local = eqx.tree_deserialise_leaves(source_path, initial)
                    parameters = broaden(model.encoder, local, regime["factor"])
                    if regime["mixture"]:
                        parameters = MixtureParameters(parameters, initial)
                local_folder = folder / f"start_{start}"
                local_folder.mkdir(parents=True, exist_ok=True)
                write(
                    local_folder / "REGIME.json",
                    dict(
                        **regime,
                        initialization=initialization,
                        scientific_promotion=False,
                    ),
                )
                state = (
                    None
                    if "support_probe" in manifest
                    else optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
                )
                history = []
                from scripts.feniks_qualified_local_vi import optimization_seed

                for iteration in range(
                    0 if "support_probe" in manifest else manifest["steps"]
                ):
                    budget.charge(regime["gradient_draws"], gradient=True)
                    parameters, state, metrics = step(
                        parameters,
                        state,
                        encoded_context,
                        observation,
                        jax.random.PRNGKey(
                            optimization_seed(manifest, seed, initialization, iteration)
                        ),
                    )
                    metrics = {
                        k: np.asarray(v).item()
                        for k, v in jax.device_get(metrics).items()
                    }
                    history.append({"step": iteration + 1, **metrics})
                    pd.DataFrame(history).to_csv(
                        local_folder / "optimization.csv", index=False
                    )
                    if not metrics["finite"]:
                        write(
                            folder / f"start_{start}" / "NONFINITE.json",
                            finite_json(history[-1]),
                        )
                        raise ValueError(
                            "nonfinite local update; diagnostic stopped without dropping draws"
                        )
                    if iteration % 8 == 0 or iteration + 1 == manifest["steps"]:
                        write(
                            root / "PROGRESS.json",
                            dict(
                                stage="local_vi",
                                group=group,
                                object=index,
                                start=start,
                                step=iteration + 1,
                                cases_complete=len(completed),
                                budget=budget.snapshot(),
                            ),
                        )
                        print(
                            f"[local-vi] {group} {index} start={start} step={iteration + 1} loss={metrics['negative_elbo']:.6g}",
                            flush=True,
                        )
                    if iteration + 1 in manifest.get("trajectory_steps", []):
                        checkpoint_folder = local_folder / f"step_{iteration + 1:04d}"
                        checkpoint_folder.mkdir(parents=True, exist_ok=True)
                        eqx.tree_serialise_leaves(
                            checkpoint_folder / "parameters.eqx", parameters
                        )
                        intermediate = eqx.tree_deserialise_leaves(
                            checkpoint_folder / "parameters.eqx", initial
                        )
                        evaluate_distribution(
                            checkpoint_folder,
                            model.encoder,
                            intermediate,
                            encoded_context,
                            observation,
                            target,
                            spec,
                            budget,
                            seed + 500 + initialization * 10,
                            manifest["evaluation_draws"],
                            generated,
                        )
                local_folder = folder / f"start_{start}"
                local_folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history).to_csv(
                    local_folder / "optimization.csv", index=False
                )
                eqx.tree_serialise_leaves(local_folder / "parameters.eqx", parameters)
                restored = eqx.tree_deserialise_leaves(
                    local_folder / "parameters.eqx",
                    parameters if "support_probe" in manifest else initial,
                )
                # Evaluate only fresh direct draws of the restored final iterate.
                write(
                    root / "PROGRESS.json",
                    dict(
                        stage="local_evaluation",
                        group=group,
                        object=index,
                        start=start,
                        cases_complete=len(completed),
                        budget=budget.snapshot(),
                    ),
                )
                evaluated = evaluate_distribution(
                    local_folder,
                    model.encoder,
                    restored,
                    encoded_context,
                    observation,
                    target,
                    spec,
                    budget,
                    seed + 500 + initialization * 10,
                    manifest["evaluation_draws"],
                    generated,
                )
                evaluated["checkpoint_sha256"] = sha256_file(
                    local_folder / "parameters.eqx"
                )
                evaluated["regime"] = regime["name"]
                evaluated["initialization"] = initialization
                fitted.append(evaluated)
            if (
                _array_tree_sha256(
                    (model.prior, model.sed_scale, model.band_calibration)
                )
                != frozen_hash
            ):
                raise ValueError("frozen model changed")
            record = dict(
                group=group,
                index=index,
                row_index=int(rows[index]),
                object_id=str(arrays.object_id[index]),
                baseline=baseline,
                local_starts=fitted,
                prior_bitwise_unchanged=True,
                elapsed_seconds=time.monotonic() - case_start,
                truth_used=False,
                scientific_promotion=False,
            )
            write(folder / "COMPLETE.json", finite_json(record))
            completed.append(record)
            report(root, completed)
            # First observed + simulated cases are the actual cost microbenchmark.
            if len(completed) == 2:
                estimate = (time.monotonic() - started_cases) / 2 * (len(cases) - 2)
                remaining = budget.seconds - (time.monotonic() - budget.started)
                write(
                    root / "COST_PREFLIGHT.json",
                    dict(
                        estimated_remaining_seconds=estimate,
                        remaining_budget_seconds=remaining,
                        safety_factor=1.25,
                        cases_measured=2,
                    ),
                )
                if 1.25 * estimate > remaining:
                    raise BudgetExceeded(
                        "two-case timing predicts exceeding the fixed allocation; no automatic extension"
                    )
        final = dict(
            status="DIAGNOSTIC_COMPLETE",
            cases_complete=len(completed),
            budget=budget.snapshot(),
            topology=topology,
            prior_bitwise_unchanged=True,
            frozen_model_sha256=frozen_hash,
            truth_used=False,
            scientific_promotion=False,
            population_training_started=False,
            interpretation=manifest.get(
                "interpretation",
                "same-family local optimization diagnostic, not a posterior or calibration certificate",
            ),
            artifacts={
                name: {"path": str(root / name), "sha256": sha256_file(root / name)}
                for name in (
                    "paired_comparison.csv",
                    "paired_comparison.png",
                    "CONTRACT_AUDIT.json",
                    "SIMULATED_INPUTS.npz",
                )
            },
        )
    except BudgetExceeded as error:
        final = dict(
            status="BUDGET_STOP",
            reason=str(error),
            budget=budget.snapshot(),
            scientific_promotion=False,
            population_training_started=False,
            truth_used=False,
        )
        if manifest.get("mode") == "precision_night":
            write(root / "NIGHT_FINAL.json", dict(final, training_started=False))
    except Exception as error:
        write(
            root / "FAILED.json",
            dict(
                status="FAILED",
                error=str(error),
                budget=budget.snapshot(),
                scientific_promotion=False,
                truth_used=False,
            ),
        )
        raise
    write(root / "FINAL.json", final)
    return final


def report(root, records):
    rows = []
    for record in records:
        for name, value in [
            ("amortized", record["baseline"]),
            *[(f"local_{i}", v) for i, v in enumerate(record["local_starts"])],
        ]:
            rows.append(
                dict(
                    group=record["group"],
                    index=record["index"],
                    variant=name,
                    ess=value["raw_ess"]["median"],
                    ess_fraction=value["raw_ess"]["fraction_median"],
                    max_weight=value["maximum_raw_weight"]["median"],
                    bad_k=value["pareto_k"]["gt_0p7_or_nonfinite_fraction"],
                    nonfinite_k=value["pareto_k"]["nonfinite_fraction"],
                    negative_elbo=value["negative_elbo"],
                    evidence_delta=value["replicate_abs_log_evidence_delta"],
                    residual_median=value["residual_median_abs"],
                    residual_rms=value["residual_rms"],
                    residual_fraction_gt5=value["residual_fraction_abs_gt5"],
                )
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "paired_comparison.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("ess_fraction", "residual_median", "evidence_delta")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for row, group in enumerate(("observed", "simulated")):
        selected = frame[frame["group"] == group]
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            for _, case in selected.groupby("index"):
                case = case.set_index("variant").reindex(
                    ["amortized", "local_0", "local_1"]
                )
                ax.plot(range(3), case[metric], "o-", alpha=0.5, linewidth=0.8)
            ax.set_xticks(range(3), ["amortized", "local 0", "local 1"])
            ax.set_title(f"{group}: {metric}")
            ax.grid(alpha=0.2)
    fig.suptitle("Paired direct-draw diagnostics; no scientific promotion")
    fig.tight_layout()
    fig.savefig(root / "paired_comparison.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--objects", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument("--gradient-isolation", action="store_true")
    parser.add_argument("--redshift-decomposition", action="store_true")
    parser.add_argument("--photometry-reference", action="store_true")
    parser.add_argument("--full-decoder-reference", type=Path)
    parser.add_argument("--mdf-precision-reference", type=Path)
    parser.add_argument("--target-resolution-reference", type=Path)
    parser.add_argument("--redshift-precision-reference", type=Path)
    parser.add_argument("--precision-night-reference", type=Path)
    parser.add_argument("--qualified-night-root", type=Path)
    parser.add_argument("--local-arm", choices=("B", "C"), default="C")
    parser.add_argument("--controlled-optimization", action="store_true")
    parser.add_argument("--long-optimization", action="store_true")
    parser.add_argument("--support-probe-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source_root is None:
            parser.error("prepare requires --source-root")
        value = prepare(
            args.root,
            args.source_root,
            args.objects,
            args.steps,
            args.draws,
            args.gradient_isolation,
            args.redshift_decomposition,
            args.photometry_reference,
            args.full_decoder_reference,
            args.mdf_precision_reference,
            args.target_resolution_reference,
            args.redshift_precision_reference,
            args.precision_night_reference,
            args.qualified_night_root,
            args.local_arm,
            args.controlled_optimization,
            args.support_probe_root,
            args.long_optimization,
        )
    else:
        with (args.root / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value = run(args.root)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
