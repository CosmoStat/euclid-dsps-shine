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


def prepare(root, source_root, objects=16, steps=64, draws=128):
    root, source_root = root.resolve(), source_root.resolve()
    if root.exists():
        raise FileExistsError(f"preserve existing diagnostic: {root}")
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
        seconds=9900,
        maximum_decoder_evaluations=45000,
        maximum_gpus=1,
        maximum_nodes=1,
        allocation_gpu_hours=3,
        truth_used=False,
        scientific_promotion=False,
        population_training_started=False,
        interpretation="diagnostic only; observed validation has been used in earlier experiments",
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
    point = cache_x[0:1]
    direction = jax.random.normal(
        jax.random.PRNGKey(62), point.shape, dtype=point.dtype
    )
    direction /= jnp.linalg.norm(direction)

    def objective(z):
        return jnp.sum(target(z, observation).logtarget)

    budget.charge(1, gradient=True)
    derivative = jnp.sum(jax.grad(objective)(point) * direction)
    estimates = []
    for epsilon in (0.01, 0.005):
        budget.charge(2)
        estimates.append(
            (
                objective(point + epsilon * direction)
                - objective(point - epsilon * direction)
            )
            / (2 * epsilon)
        )
    assert_close(
        "finite difference convergence", estimates[0], estimates[1], atol=0.1, rtol=0.05
    )
    result["gradient_delta"] = assert_close(
        "target input gradient", derivative, estimates[1], atol=0.1, rtol=0.05
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
    logz = []
    for part in chunks:
        weights = part["loglike"] + part["logprior"] - part["logq"]
        logz.append(float(jax.scipy.special.logsumexp(weights[:, 0]) - np.log(draws)))
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
        optimizer, step = make_step(
            model.encoder, target, draws=4, learning_rate=manifest["learning_rate"]
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
            for start in range(2):
                parameters = (
                    initial
                    if start == 0
                    else perturb(initial, jax.random.PRNGKey(seed + 20))
                )
                state = optimizer.init(eqx.filter(parameters, eqx.is_inexact_array))
                history = []
                for iteration in range(manifest["steps"]):
                    budget.charge(4, gradient=True)
                    parameters, state, metrics = step(
                        parameters,
                        state,
                        encoded_context,
                        observation,
                        jax.random.PRNGKey(seed + 100 + iteration + start * 100),
                    )
                    metrics = {
                        k: np.asarray(v).item()
                        for k, v in jax.device_get(metrics).items()
                    }
                    history.append({"step": iteration + 1, **metrics})
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
                            f"[local-vi] {group} {index} start={start} step={iteration+1} loss={metrics['negative_elbo']:.6g}",
                            flush=True,
                        )
                local_folder = folder / f"start_{start}"
                local_folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history).to_csv(
                    local_folder / "optimization.csv", index=False
                )
                eqx.tree_serialise_leaves(local_folder / "parameters.eqx", parameters)
                restored = eqx.tree_deserialise_leaves(
                    local_folder / "parameters.eqx", initial
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
                    seed + 500 + start * 10,
                    manifest["evaluation_draws"],
                    generated,
                )
                evaluated["checkpoint_sha256"] = sha256_file(
                    local_folder / "parameters.eqx"
                )
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
            interpretation="same-family local optimization diagnostic, not a posterior or calibration certificate",
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
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source_root is None:
            parser.error("prepare requires --source-root")
        value = prepare(
            args.root, args.source_root, args.objects, args.steps, args.draws
        )
    else:
        with (args.root / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value = run(args.root)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
