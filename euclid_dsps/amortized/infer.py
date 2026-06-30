"""Inference entry points for trained amortized FS2 models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.calibration import (
    alpha_metadata,
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    per_band_flux_calibration_config,
)
from euclid_dsps.diffsky_redshift_ablation import write_redshift_metrics_for_run
from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import DERIVED_QUANTITY_NAMES, dynamic_model_args, load_context
from euclid_dsps.parameter_vectors import derived_from_theta_matrix_jax

from .catalog import (
    learned_prior_samples_frame,
    posterior_predictive_flux_frame,
    posterior_samples_frame,
    posterior_summary_frame,
)
from .catalog_identity import (
    select_catalog_row_indices,
    write_catalog_fingerprint,
    write_truth_snapshot,
)
from .collapse_gates import write_inference_collapse_gate
from .config import amortized_config
from .data import (
    iter_photometry_batches_from_arrays,
    iter_photometry_batches_from_config,
    load_photometry_arrays_from_config,
)
from .decoder import model_flux_from_x
from .diagnostics import (
    feature_diagnostics_frame,
    posterior_predictive_residual_frame,
    posterior_predictive_residual_summary_frame,
    summarize_inference_outputs,
)
from .elbo import is_deterministic_reconstruction, objective_mode
from .features import read_feature_stats
from .latent import latent_spec_from_config, x_to_theta
from .likelihood import photometric_loglike
from .train import (
    _effective_jax_batch_size,
    _per_band_flux_calibration_summary,
    load_checkpoint,
    write_per_band_flux_calibration_artifacts,
)
from .truth_diagnostics import write_extended_truth_diagnostics


def infer_amortized_fs2(
    config: dict[str, Any],
    out_dir: Path,
    *,
    checkpoint: Path,
    limit: int | None,
    batch_size: int,
    posterior_samples: int,
    prior_samples: int = 8192,
    seed: int,
    feature_stats_path: Path | None = None,
    decoder_sample_chunk_size: int = 1,
    prior_predictive_batch_size: int | None = None,
    shard_outputs: bool | None = None,
    resume_shards: bool | None = None,
    write_posterior_predictive: bool | None = None,
    write_residual_samples: bool | None = None,
    combine_sample_shards: bool | None = None,
    combine_summary_shards: bool | None = None,
    selection_mode: str | None = None,
    stratified_strategy: str | None = None,
    selection_seed: int | None = None,
    row_indices_file: str | Path | None = None,
    verbose: bool = True,
    dataset_label: str = "FS2",
) -> None:
    """Run amortized posterior inference for configured catalog rows."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if int(posterior_samples) <= 0:
        raise ValueError("posterior_samples must be positive")
    if int(prior_samples) <= 0:
        raise ValueError("prior_samples must be positive")
    if int(decoder_sample_chunk_size) <= 0:
        raise ValueError("decoder_sample_chunk_size must be positive")
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    objective_mode_name = objective_mode(cfg.get("objective", {}))
    deterministic_reconstruction = is_deterministic_reconstruction(
        cfg.get("objective", {})
    )
    inference_cfg = cfg.get("inference", {})
    selection_mode = str(
        selection_mode
        if selection_mode is not None
        else inference_cfg.get("selection_mode", "sequential")
    )
    stratified_strategy = str(
        stratified_strategy
        if stratified_strategy is not None
        else inference_cfg.get("stratified_strategy", "balanced")
    )
    selection_seed = int(
        selection_seed
        if selection_seed is not None
        else inference_cfg.get("selection_seed", seed)
    )
    redshift_bins_for_selection = inference_cfg.get(
        "redshift_bins",
        ((config.get("amortized", {}) or {}).get("data", {}) or {}).get(
            "redshift_bins",
            None,
        ),
    )
    catalog_identity = write_catalog_fingerprint(
        out,
        config,
        redshift_bins=redshift_bins_for_selection,
    )
    row_indices, selection_summary = select_catalog_row_indices(
        config,
        limit=limit,
        selection_mode=selection_mode,
        stratified_strategy=stratified_strategy,
        seed=selection_seed,
        redshift_bins=redshift_bins_for_selection,
        row_indices_file=row_indices_file,
    )
    if row_indices is not None:
        np.save(out / "inference_indices.npy", row_indices)
        selection_summary["row_indices_path"] = "inference_indices.npy"
    write_json(out / "inference_selection.json", selection_summary)
    truth_snapshot = write_truth_snapshot(
        out,
        config,
        row_indices=row_indices,
        limit=limit,
        batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
    )
    jax_batch_size = _effective_jax_batch_size(
        inference_cfg,
        int(batch_size),
    )
    if prior_predictive_batch_size is None:
        prior_predictive_batch_size = int(
            inference_cfg.get("prior_predictive_batch_size", 256)
        )
    prior_predictive_batch_size = int(prior_predictive_batch_size)
    if prior_predictive_batch_size <= 0:
        raise ValueError("prior_predictive_batch_size must be positive")
    shard_outputs = (
        bool(inference_cfg.get("shard_outputs", False))
        if shard_outputs is None
        else bool(shard_outputs)
    )
    resume_shards = (
        bool(inference_cfg.get("resume_shards", True))
        if resume_shards is None
        else bool(resume_shards)
    )
    write_posterior_predictive = (
        bool(inference_cfg.get("write_posterior_predictive", True))
        if write_posterior_predictive is None
        else bool(write_posterior_predictive)
    )
    write_residual_samples = (
        bool(inference_cfg.get("write_residual_samples", True))
        if write_residual_samples is None
        else bool(write_residual_samples)
    )
    combine_sample_shards = (
        bool(inference_cfg.get("combine_sample_shards", False))
        if combine_sample_shards is None
        else bool(combine_sample_shards)
    )
    combine_summary_shards = (
        bool(inference_cfg.get("combine_summary_shards", True))
        if combine_summary_shards is None
        else bool(combine_summary_shards)
    )
    checkpoint = Path(checkpoint)
    if feature_stats_path is None:
        feature_stats_path = checkpoint.parent.parent / "feature_stats.json"
    if verbose:
        print(f"[amortized] {dataset_label} amortized inference")
        print(f"[amortized] checkpoint: {checkpoint}")
        print(f"[amortized] output directory: {out}")
        print(
            "[amortized] run config: "
            f"limit={limit} batch_size={int(batch_size)} "
            f"posterior_samples={int(posterior_samples)} "
            f"prior_samples={int(prior_samples)} "
            f"decoder_sample_chunk_size={int(decoder_sample_chunk_size)} "
            f"prior_predictive_batch_size={int(prior_predictive_batch_size)} "
            f"shard_outputs={shard_outputs} resume_shards={resume_shards} "
            f"write_posterior_predictive={write_posterior_predictive} "
            f"write_residual_samples={write_residual_samples} "
            f"selection_mode={selection_mode} "
            f"stratified_strategy={stratified_strategy} "
            f"objective={objective_mode_name}"
        )
        print(
            "[amortized] inference selection: "
            f"rows={selection_summary.get('selected_rows')} "
            f"truth_rows={len(truth_snapshot)} "
            f"row_indices={selection_summary.get('row_indices_path')}"
        )
        if jax_batch_size != int(batch_size):
            print(
                "[amortized] capping JAX/DSPS inference batch size: "
                f"requested_batch_size={int(batch_size)} "
                f"jax_batch_size={jax_batch_size}"
            )
        print(
            f"[amortized] JAX backend: {jax.default_backend()} devices={jax.devices()}"
        )
        print(f"[amortized] feature stats: {feature_stats_path}")
    feature_stats = read_feature_stats(feature_stats_path)
    model = load_checkpoint(checkpoint, config)
    calibration_runtime_config = {"calibration": config.get("calibration", {}) or {}}
    scale_cfg = global_sed_scale_config(calibration_runtime_config)
    log_alpha_sed = (
        float(np.asarray(jax.device_get(model.sed_scale.log_alpha_sed)))
        if scale_cfg.enabled
        else 0.0
    )
    alpha_sed = float(np.exp(log_alpha_sed))
    band_calibration_cfg = per_band_flux_calibration_config(calibration_runtime_config)
    band_names = tuple(str(band["name"]) for band in config["bands"])
    log_alpha_band = (
        jax.device_get(model.band_calibration.log_alpha_band)
        if band_calibration_cfg.enabled and model.band_calibration is not None
        else np.zeros((len(band_names),), dtype=np.float32)
    )
    if verbose:
        print("[amortized] loading configured filters...")
    filters = load_filters(config["bands"])
    if verbose:
        print(f"[amortized] loading DSPS context from {config['ssp_path']}...")
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    if verbose:
        print(f"[amortized] DSPS context ready: {len(filters)} filters")
        print(
            "[amortized] global SED scale: "
            f"enabled={scale_cfg.enabled} mode={scale_cfg.mode} "
            f"alpha_sed={alpha_sed:.6g}"
        )
        print(
            "[amortized] per-band flux calibration: "
            f"enabled={band_calibration_cfg.enabled} "
            f"mode={band_calibration_cfg.mode} "
            f"trainable={band_calibration_cfg.trainable}"
        )
    latent_spec = latent_spec_from_config(config)
    likelihood_cfg = cfg["likelihood"]
    key = jax.random.PRNGKey(int(seed))

    sample_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    predictive_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    residual_summary_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    row_counts = {
        "samples_rows": 0,
        "summary_rows": 0,
        "predictive_rows": 0,
        "residual_rows": 0,
        "residual_summary_rows": 0,
        "feature_diagnostics_rows": 0,
        "parameter_bound_diagnostics_rows": 0,
    }
    shards_written: list[dict[str, Any]] = []
    shards_skipped: list[dict[str, Any]] = []
    if shard_outputs:
        _ensure_inference_shard_dirs(
            out,
            write_posterior_predictive=write_posterior_predictive,
            write_residual_samples=write_residual_samples,
        )
    dense_selected_batches = bool(inference_cfg.get("dense_selected_batches", True))
    run_signature = {
        "catalog_path": str(config.get("catalog_path")),
        "checkpoint": str(checkpoint),
        "feature_stats_path": str(feature_stats_path),
        "batch_size": int(batch_size),
        "jax_batch_size": int(jax_batch_size),
        "limit": int(limit) if limit is not None else None,
        "selection_mode": selection_mode,
        "stratified_strategy": stratified_strategy,
        "selection_seed": int(selection_seed),
        "row_indices_file": str(row_indices_file) if row_indices_file else None,
        "posterior_samples": int(posterior_samples),
        "effective_posterior_samples": (
            1 if deterministic_reconstruction else int(posterior_samples)
        ),
        "objective_mode": objective_mode_name,
        "decoder_sample_chunk_size": int(decoder_sample_chunk_size),
        "write_posterior_predictive": bool(write_posterior_predictive),
        "write_residual_samples": bool(write_residual_samples),
        "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
        "per_band_calibration_enabled": bool(band_calibration_cfg.enabled),
        "dense_selected_batches": bool(dense_selected_batches),
    }
    if row_indices is not None and dense_selected_batches:
        if verbose:
            print(
                "[amortized] loading selected inference rows for dense batching: "
                f"rows={len(row_indices)} jax_batch_size={int(jax_batch_size)}"
            )
        selected_arrays = load_photometry_arrays_from_config(
            config,
            batch_size=int(inference_cfg.get("catalog_batch_size", 10_000)),
            row_indices=row_indices,
        )
        batch_iterable = iter_photometry_batches_from_arrays(
            selected_arrays,
            batch_size=int(jax_batch_size),
            feature_stats=feature_stats,
        )
    else:
        batch_iterable = iter_photometry_batches_from_config(
            config,
            batch_size=int(jax_batch_size),
            limit=limit,
            feature_stats=feature_stats,
            row_indices=row_indices,
        )

    n_objects_total = 0
    for batch_index, batch in enumerate(batch_iterable, start=1):
        if verbose:
            print(
                "[amortized] inference batch "
                f"{batch_index}: n_objects={int(batch.flux.shape[0])}"
            )
        key, sample_key = jax.random.split(key)
        shard_paths = _inference_shard_paths(out, batch_index) if shard_outputs else {}
        shard_signature = _inference_shard_signature(
            run_signature,
            batch_index=batch_index,
            object_id=batch.object_id,
            row_index=batch.row_index,
        )
        if (
            shard_outputs
            and resume_shards
            and _inference_shard_complete(
                shard_paths,
                write_posterior_predictive=write_posterior_predictive,
                write_residual_samples=write_residual_samples,
                signature=shard_signature,
            )
        ):
            counts = _inference_shard_row_counts(
                shard_paths,
                write_posterior_predictive=write_posterior_predictive,
                write_residual_samples=write_residual_samples,
            )
            for key_name, value in counts.items():
                row_counts[key_name] += int(value)
            n_objects_total += int(counts.get("summary_rows", 0))
            if combine_summary_shards:
                summary_frames.append(pd.read_parquet(shard_paths["summary"]))
                feature_frames.append(pd.read_parquet(shard_paths["features"]))
                residual_summary_frames.append(
                    pd.read_parquet(shard_paths["residual_summary"])
                )
            if combine_sample_shards:
                sample_frames.append(pd.read_parquet(shard_paths["samples"]))
            shards_skipped.append(
                _inference_shard_manifest_record(batch_index, shard_paths, counts)
            )
            if verbose:
                print(f"[amortized] shard batch {batch_index} exists; skipping")
            continue
        if deterministic_reconstruction:
            mean, _log_std = model.encoder(batch.features)
            x_samples = mean[None, ...]
            logq = jnp.zeros(mean.shape[:-1], dtype=mean.dtype)[None, ...]
        else:
            x_samples, logq = model.encoder.sample_and_log_prob(
                sample_key,
                batch.features,
                int(posterior_samples),
            )
        theta = x_to_theta(x_samples, latent_spec)
        model_flux_raw = _model_flux_from_x_sample_chunks(
            x_samples,
            latent_spec,
            context,
            model_args,
            latent_spec.names,
            sample_chunk_size=int(decoder_sample_chunk_size),
        )
        model_flux = (
            apply_global_sed_scale_to_flux(
                model_flux_raw,
                jnp.asarray(log_alpha_sed, dtype=model_flux_raw.dtype),
            )
            if scale_cfg.enabled
            else model_flux_raw
        )
        model_flux = (
            apply_per_band_flux_calibration_to_flux(
                model_flux,
                jnp.asarray(log_alpha_band, dtype=model_flux.dtype),
            )
            if band_calibration_cfg.enabled
            else model_flux
        )
        logprior = (
            jnp.zeros_like(logq)
            if deterministic_reconstruction
            else model.prior.log_prob(x_samples)
        )
        loglike = photometric_loglike(
            obs_flux=batch.flux,
            model_flux=model_flux,
            obs_err=batch.flux_err,
            mask=batch.mask,
            likelihood_type=str(likelihood_cfg.get("type", "student_t")),
            student_t_dof=float(likelihood_cfg.get("student_t_dof", 2.0)),
            error_floor_frac=float(likelihood_cfg.get("error_floor_frac", 0.02)),
            error_jitter=float(likelihood_cfg.get("error_jitter", 0.0)),
        )
        chi2 = _posterior_predictive_chi2(batch, model_flux, likelihood_cfg)
        object_id = np.asarray(batch.object_id)
        theta_np = jax.device_get(theta)
        logq_np = jax.device_get(logq)
        logprior_np = jax.device_get(logprior)
        loglike_np = jax.device_get(loglike)
        model_flux_raw_np = jax.device_get(model_flux_raw)
        model_flux_np = jax.device_get(model_flux)
        mask_np = jax.device_get(batch.mask)
        sample_frame = posterior_samples_frame(
            object_id,
            theta_np,
            latent_spec.names,
            logq_np,
            logprior_np,
            loglike_np,
            row_index=batch.row_index,
            log_alpha_sed=log_alpha_sed,
            alpha_sed=alpha_sed,
        )
        summary_frame = posterior_summary_frame(
            object_id,
            theta_np,
            latent_spec.names,
            loglike_np,
            jax.device_get(chi2),
            mask_np,
            row_index=batch.row_index,
            log_alpha_sed=log_alpha_sed,
            alpha_sed=alpha_sed,
        )
        _add_posterior_summary_derived_columns(
            summary_frame,
            context,
            model_args,
            latent_spec.names,
            alpha_sed=alpha_sed,
        )
        predictive_frame = (
            posterior_predictive_flux_frame(
                object_id,
                model_flux_np,
                band_names,
                row_index=batch.row_index,
                model_flux_raw=model_flux_raw_np,
                log_alpha_sed=log_alpha_sed,
                alpha_sed=alpha_sed,
            )
            if write_posterior_predictive
            else pd.DataFrame()
        )
        residual_frame = (
            posterior_predictive_residual_frame(
                object_id,
                jax.device_get(batch.flux),
                jax.device_get(batch.flux_err),
                mask_np,
                model_flux_np,
                band_names,
                row_index=batch.row_index,
                likelihood_config=likelihood_cfg,
            )
            if write_residual_samples
            else pd.DataFrame()
        )
        residual_summary_frame = posterior_predictive_residual_summary_frame(
            object_id,
            jax.device_get(batch.flux),
            jax.device_get(batch.flux_err),
            mask_np,
            model_flux_np,
            band_names,
            row_index=batch.row_index,
            likelihood_config=likelihood_cfg,
        )
        feature_frame = feature_diagnostics_frame(
            object_id,
            jax.device_get(batch.features),
            row_index=batch.row_index,
            n_flux_bands=len(band_names),
        )
        if shard_outputs:
            sample_frame.to_parquet(shard_paths["samples"], index=False)
            summary_frame.to_parquet(shard_paths["summary"], index=False)
            feature_frame.to_parquet(shard_paths["features"], index=False)
            residual_summary_frame.to_parquet(
                shard_paths["residual_summary"],
                index=False,
            )
            if write_posterior_predictive:
                predictive_frame.to_parquet(shard_paths["predictive"], index=False)
            if write_residual_samples:
                residual_frame.to_parquet(shard_paths["residuals"], index=False)
            counts = {
                "samples_rows": int(len(sample_frame)),
                "summary_rows": int(len(summary_frame)),
                "predictive_rows": int(len(predictive_frame)),
                "residual_rows": int(len(residual_frame)),
                "residual_summary_rows": int(len(residual_summary_frame)),
                "feature_diagnostics_rows": int(len(feature_frame)),
            }
            _write_inference_shard_metadata(
                shard_paths,
                signature=shard_signature,
                counts=counts,
                batch_index=batch_index,
            )
            for key_name, value in counts.items():
                row_counts[key_name] += int(value)
            shards_written.append(
                _inference_shard_manifest_record(batch_index, shard_paths, counts)
            )
            if combine_summary_shards:
                summary_frames.append(summary_frame)
                feature_frames.append(feature_frame)
                residual_summary_frames.append(residual_summary_frame)
            if combine_sample_shards:
                sample_frames.append(sample_frame)
        else:
            sample_frames.append(sample_frame)
            summary_frames.append(summary_frame)
            predictive_frames.append(predictive_frame)
            residual_frames.append(residual_frame)
            residual_summary_frames.append(residual_summary_frame)
        n_objects_total += int(batch.flux.shape[0])

    if verbose:
        print("[amortized] combining posterior inference frames...")
    samples = (
        pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    )
    summary = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame()
    )
    predictive = (
        pd.concat(predictive_frames, ignore_index=True)
        if predictive_frames
        else pd.DataFrame()
    )
    residuals = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames
        else pd.DataFrame()
    )
    feature_diagnostics = (
        pd.concat(feature_frames, ignore_index=True)
        if feature_frames
        else pd.DataFrame()
    )
    residual_summary = (
        pd.concat(residual_summary_frames, ignore_index=True)
        if residual_summary_frames
        else pd.DataFrame()
    )
    parameter_bound_diagnostics = _write_parameter_bound_diagnostics(
        summary,
        latent_spec,
        out,
    )
    row_counts["parameter_bound_diagnostics_rows"] = int(
        len(parameter_bound_diagnostics)
    )
    if not shard_outputs:
        row_counts = {
            "samples_rows": int(len(samples)),
            "summary_rows": int(len(summary)),
            "predictive_rows": int(len(predictive)),
            "residual_rows": int(len(residuals)),
            "residual_summary_rows": int(len(residual_summary)),
            "feature_diagnostics_rows": int(len(feature_diagnostics)),
            "parameter_bound_diagnostics_rows": int(len(parameter_bound_diagnostics)),
        }
    key, prior_key = jax.random.split(key)
    if verbose:
        print(
            "[amortized] prior predictive start: "
            f"prior_samples={int(prior_samples)} "
            f"prior_predictive_batch_size={int(prior_predictive_batch_size)}"
        )
    prior_x = model.prior.sample(prior_key, int(prior_samples))
    prior_theta = x_to_theta(prior_x, latent_spec)
    prior_logprob = model.prior.log_prob(prior_x)
    prior_flux_raw_2d = _model_flux_from_x_2d_chunks(
        prior_x,
        latent_spec,
        context,
        model_args,
        latent_spec.names,
        batch_size=int(prior_predictive_batch_size),
    )
    prior_flux_raw = prior_flux_raw_2d[:, None, :]
    prior_flux = (
        apply_global_sed_scale_to_flux(
            prior_flux_raw,
            jnp.asarray(log_alpha_sed, dtype=prior_flux_raw.dtype),
        )
        if scale_cfg.enabled
        else prior_flux_raw
    )
    prior_flux = (
        apply_per_band_flux_calibration_to_flux(
            prior_flux,
            jnp.asarray(log_alpha_band, dtype=prior_flux.dtype),
        )
        if band_calibration_cfg.enabled
        else prior_flux
    )
    learned_prior = learned_prior_samples_frame(
        jax.device_get(prior_x),
        jax.device_get(prior_theta),
        latent_spec.names,
        jax.device_get(prior_logprob),
        derived=_derived_columns_from_theta(
            context,
            model_args,
            jax.device_get(prior_theta),
            latent_spec.names,
            alpha_sed=alpha_sed,
            summary=False,
        ),
        log_alpha_sed=log_alpha_sed,
        alpha_sed=alpha_sed,
    )
    prior_predictive = posterior_predictive_flux_frame(
        np.asarray(["prior"], dtype=object),
        jax.device_get(prior_flux),
        band_names,
        model_flux_raw=jax.device_get(prior_flux_raw),
        log_alpha_sed=log_alpha_sed,
        alpha_sed=alpha_sed,
    ).rename(columns={"sample_id": "prior_sample_id"})
    if verbose:
        print("[amortized] writing inference parquet outputs...")
    if shard_outputs:
        if combine_sample_shards and not samples.empty:
            samples.to_parquet(out / "posterior_samples.parquet", index=False)
        if combine_summary_shards:
            summary.to_parquet(out / "posterior_summary.parquet", index=False)
            feature_diagnostics.to_parquet(
                out / "feature_diagnostics.parquet", index=False
            )
            residual_summary.to_parquet(
                out / "posterior_predictive_residual_summary.parquet",
                index=False,
            )
        _write_shard_manifest(
            out,
            {
                "shard_outputs": True,
                "resume_shards": bool(resume_shards),
                "write_posterior_predictive": bool(write_posterior_predictive),
                "write_residual_samples": bool(write_residual_samples),
                "combine_sample_shards": bool(combine_sample_shards),
                "combine_summary_shards": bool(combine_summary_shards),
                "run_signature": run_signature,
                "n_shards_written": int(len(shards_written)),
                "n_shards_skipped": int(len(shards_skipped)),
                "shards_written": shards_written,
                "shards_skipped": shards_skipped,
                **row_counts,
            },
        )
    else:
        samples.to_parquet(out / "posterior_samples.parquet", index=False)
        summary.to_parquet(out / "posterior_summary.parquet", index=False)
        if write_posterior_predictive:
            predictive.to_parquet(
                out / "posterior_predictive_flux.parquet", index=False
            )
        if write_residual_samples:
            residuals.to_parquet(
                out / "posterior_predictive_residuals.parquet",
                index=False,
            )
        residual_summary.to_parquet(
            out / "posterior_predictive_residual_summary.parquet",
            index=False,
        )
        feature_diagnostics.to_parquet(out / "feature_diagnostics.parquet", index=False)
    prior_predictive.to_parquet(out / "prior_predictive_flux.parquet", index=False)
    learned_prior.to_parquet(out / "learned_prior_samples.parquet", index=False)
    learned_prior.to_parquet(
        out / "learned_or_loaded_prior_samples.parquet",
        index=False,
    )
    write_json(out / "normalized_config.json", config)
    global_sed_scale_payload = alpha_metadata(
        log_alpha_sed,
        scale_cfg.prior_sigma_log_alpha,
    ) | {
        "enabled": bool(scale_cfg.enabled),
        "trainable": bool(scale_cfg.trainable),
        "mode": scale_cfg.mode,
    }
    per_band_payload = _per_band_flux_calibration_summary(
        model,
        config,
        band_names=band_names,
        config_block=band_calibration_cfg,
        trainable=band_calibration_cfg.trainable,
    )
    write_per_band_flux_calibration_artifacts(
        out,
        model,
        config,
        band_names=band_names,
        config_block=band_calibration_cfg,
        trainable=band_calibration_cfg.trainable,
    )
    write_json(
        out / "inference_summary.json",
        {
            "objective_mode": objective_mode_name,
            "deterministic_reconstruction": bool(deterministic_reconstruction),
            "requested_posterior_samples": int(posterior_samples),
            "effective_posterior_samples": (
                1 if deterministic_reconstruction else int(posterior_samples)
            ),
            "parameter_bound_diagnostics": (
                "parameter_bound_diagnostics.csv"
                if not parameter_bound_diagnostics.empty
                else None
            ),
            "global_sed_scale": global_sed_scale_payload,
            "per_band_flux_calibration": per_band_payload,
        },
    )
    try:
        metric_outputs = {
            key: str(value)
            for key, value in write_redshift_metrics_for_run(
                dataset_path=config["catalog_path"],
                run_dir=out,
                out_dir=out,
                label=dataset_label,
            ).items()
        }
    except Exception as exc:
        metric_outputs = {"warning": str(exc)}
    try:
        metric_outputs.update(
            {
                f"extended_{key}": str(value)
                for key, value in write_extended_truth_diagnostics(out).items()
            }
        )
    except Exception as exc:
        metric_outputs["extended_truth_warning"] = str(exc)
    try:
        gate = write_inference_collapse_gate(out)
        metric_outputs["collapse_gate"] = str(out / "collapse_gate.json")
        metric_outputs["collapse_gate_status"] = str(gate.get("status"))
    except Exception as exc:
        metric_outputs["collapse_gate_warning"] = str(exc)
    write_json(
        out / "inference_summary.json",
        {
            "checkpoint": str(checkpoint),
            "feature_stats_path": str(feature_stats_path),
            "limit": limit,
            "selection": selection_summary,
            "catalog_fingerprint": catalog_identity,
            "truth_snapshot_rows": int(len(truth_snapshot)),
            "batch_size": int(batch_size),
            "jax_batch_size": int(jax_batch_size),
            "posterior_samples": int(posterior_samples),
            "prior_samples": int(prior_samples),
            "decoder_sample_chunk_size": int(decoder_sample_chunk_size),
            "prior_predictive_batch_size": int(prior_predictive_batch_size),
            "shard_outputs": bool(shard_outputs),
            "resume_shards": bool(resume_shards),
            "write_posterior_predictive": bool(write_posterior_predictive),
            "write_residual_samples": bool(write_residual_samples),
            "combine_sample_shards": bool(combine_sample_shards),
            "combine_summary_shards": bool(combine_summary_shards),
            "dense_selected_batches": bool(dense_selected_batches),
            "n_objects": int(n_objects_total),
            "samples_rows": int(row_counts["samples_rows"]),
            "summary_rows": int(row_counts["summary_rows"]),
            "predictive_rows": int(row_counts["predictive_rows"]),
            "prior_predictive_rows": int(len(prior_predictive)),
            "residual_rows": int(row_counts["residual_rows"]),
            "residual_summary_rows": int(row_counts["residual_summary_rows"]),
            "feature_diagnostics_rows": int(row_counts["feature_diagnostics_rows"]),
            "learned_prior_rows": int(len(learned_prior)),
            "n_shards_written": int(len(shards_written)),
            "n_shards_skipped": int(len(shards_skipped)),
            "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
            "prior_train_jointly": bool(cfg["prior"].get("train_jointly", True)),
            "global_sed_scale": global_sed_scale_payload,
            "per_band_flux_calibration": per_band_payload,
            "metric_outputs": metric_outputs,
        },
    )
    if verbose:
        print("[amortized] writing posterior predictive diagnostics...")
    summarize_inference_outputs(
        out / "posterior_summary.parquet",
        out,
        config=config,
        limit=limit,
        row_indices=row_indices,
    )
    if verbose:
        print("[amortized] inference complete")
        print(f"[amortized] summary: {out / 'inference_summary.json'}")


def finalize_amortized_inference(
    config: dict[str, Any],
    out_dir: Path,
    *,
    limit: int | None = None,
    combine_sample_shards: bool = False,
    dataset_label: str = "FS2",
    verbose: bool = True,
) -> dict[str, Any]:
    """Combine sharded inference outputs and write diagnostics/plots.

    This is intentionally independent from the GPU inference loop so a Slurm
    run that hit walltime can still be summarized safely on login/CPU nodes.
    """
    out = Path(out_dir)
    if not out.exists():
        raise FileNotFoundError(f"Missing inference output directory: {out}")
    shard_records = _discover_shard_records(out)
    complete_records = [record for record in shard_records if record["complete"]]
    if verbose:
        print(
            "[amortized] finalize inference: "
            f"run={out} complete_shards={len(complete_records)} "
            f"all_shards={len(shard_records)}"
        )
    frames = _combine_inference_shard_tables(
        out,
        complete_records,
        combine_sample_shards=bool(combine_sample_shards),
        verbose=verbose,
    )
    selection = _read_json_if_exists(out / "inference_selection.json")
    expected = selection.get("selected_rows")
    processed = int(len(frames.get("summary", pd.DataFrame())))
    row_indices = _load_inference_indices(out)
    manifest_payload = {
        "shard_outputs": True,
        "finalized": True,
        "n_shards": int(len(complete_records)),
        "summary_rows": processed,
        "samples_rows": int(_frame_len(frames.get("samples"))),
        "residual_summary_rows": int(_frame_len(frames.get("residual_summary"))),
        "feature_diagnostics_rows": int(_frame_len(frames.get("features"))),
        "expected_selected_rows": int(expected) if expected is not None else None,
        "incomplete": bool(expected is not None and processed < int(expected)),
        "shards_written": [
            _inference_shard_manifest_record(
                int(record["batch"]),
                _inference_shard_paths(out, int(record["batch"])),
                dict(record.get("counts", {})),
            )
            for record in complete_records
        ],
        "shards_skipped": [],
    }
    _write_shard_manifest(out, manifest_payload)
    metric_outputs: dict[str, str] = {}
    try:
        metric_outputs = {
            key: str(value)
            for key, value in write_redshift_metrics_for_run(
                dataset_path=config["catalog_path"],
                run_dir=out,
                out_dir=out,
                label=dataset_label,
            ).items()
        }
    except Exception as exc:
        metric_outputs = {"warning": str(exc)}
    try:
        metric_outputs.update(
            {
                f"extended_{key}": str(value)
                for key, value in write_extended_truth_diagnostics(out).items()
            }
        )
    except Exception as exc:
        metric_outputs["extended_truth_warning"] = str(exc)
    try:
        gate = write_inference_collapse_gate(out)
        metric_outputs["collapse_gate"] = str(out / "collapse_gate.json")
        metric_outputs["collapse_gate_status"] = str(gate.get("status"))
    except Exception as exc:
        metric_outputs["collapse_gate_warning"] = str(exc)
    summarize_inference_outputs(
        out / "posterior_summary.parquet",
        out,
        config=config,
        limit=limit,
        row_indices=row_indices,
    )
    payload = {
        "run_dir": str(out),
        "n_processed": processed,
        "expected_selected_rows": int(expected) if expected is not None else None,
        "complete": bool(expected is None or processed >= int(expected)),
        "n_complete_shards": int(len(complete_records)),
        "n_discovered_shards": int(len(shard_records)),
        "combine_sample_shards": bool(combine_sample_shards),
        "metric_outputs": metric_outputs,
        "posterior_summary": str(out / "posterior_summary.parquet"),
        "posterior_diagnostics_summary": str(
            out / "posterior_diagnostics_summary.json"
        ),
    }
    summary_name = (
        "inference_summary.json"
        if payload["complete"]
        else "inference_incomplete_summary.json"
    )
    write_json(out / summary_name, payload)
    if not payload["complete"]:
        write_json(out / "inference_incomplete_summary.json", payload)
    if verbose:
        print(f"[amortized] finalize summary: {out / summary_name}")
    return payload


def _discover_shard_records(out: Path) -> list[dict[str, Any]]:
    metadata_dir = _inference_shard_dirs(out)["metadata"]
    records: list[dict[str, Any]] = []
    if metadata_dir.exists():
        for path in sorted(metadata_dir.glob("batch_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            batch = int(payload.get("batch", _batch_index_from_path(path)))
            paths = _inference_shard_paths(out, batch)
            counts = dict(payload.get("counts", {}) or {})
            records.append(
                {
                    "batch": batch,
                    "metadata": path,
                    "counts": counts,
                    "complete": _basic_shard_tables_exist(paths),
                }
            )
    if records:
        return sorted(records, key=lambda record: int(record["batch"]))
    summary_dir = _inference_shard_dirs(out)["summary"]
    if not summary_dir.exists():
        return []
    for path in sorted(summary_dir.glob("batch_*.parquet")):
        batch = _batch_index_from_path(path)
        paths = _inference_shard_paths(out, batch)
        records.append(
            {
                "batch": batch,
                "metadata": paths["metadata"],
                "counts": _inference_shard_row_counts(
                    paths,
                    write_posterior_predictive=paths["predictive"].exists(),
                    write_residual_samples=paths["residuals"].exists(),
                )
                if _basic_shard_tables_exist(paths)
                else {},
                "complete": _basic_shard_tables_exist(paths),
            }
        )
    return records


def _combine_inference_shard_tables(
    out: Path,
    records: list[dict[str, Any]],
    *,
    combine_sample_shards: bool,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    table_specs = {
        "summary": ("summary", out / "posterior_summary.parquet", True),
        "features": ("features", out / "feature_diagnostics.parquet", True),
        "residual_summary": (
            "residual_summary",
            out / "posterior_predictive_residual_summary.parquet",
            True,
        ),
        "samples": (
            "samples",
            out / "posterior_samples.parquet",
            combine_sample_shards,
        ),
    }
    frames: dict[str, pd.DataFrame] = {}
    for name, (path_key, output_path, should_write) in table_specs.items():
        if verbose:
            print(
                "[amortized] combine shards: "
                f"table={name} write={bool(should_write)}"
            )
        paths = [
            _inference_shard_paths(out, int(record["batch"]))[path_key]
            for record in records
        ]
        existing = [path for path in paths if path.exists() and path.stat().st_size > 0]
        pieces = []
        total = len(existing)
        if verbose:
            print(
                "[amortized] combine shards: "
                f"table={name} files={total}"
            )
        for index, path in enumerate(existing, start=1):
            pieces.append(pd.read_parquet(path))
            if verbose and (index == 1 or index == total or index % 25 == 0):
                print(
                    "[amortized] combine shards: "
                    f"table={name} read={index}/{total}"
                )
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        frames[name] = frame
        if should_write and not frame.empty:
            if verbose:
                print(
                    "[amortized] combine shards: "
                    f"table={name} rows={len(frame)} -> {output_path}"
                )
            frame.to_parquet(output_path, index=False)
        elif verbose:
            print(
                "[amortized] combine shards: "
                f"table={name} rows={len(frame)} skipped_write={not should_write}"
            )
    return frames


def _basic_shard_tables_exist(paths: dict[str, Path]) -> bool:
    required = ["samples", "summary", "residual_summary", "features"]
    return all(
        paths[key].exists() and paths[key].stat().st_size > 0 for key in required
    )


def _batch_index_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_inference_indices(out: Path) -> np.ndarray | None:
    path = out / "inference_indices.npy"
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _frame_len(frame: pd.DataFrame | None) -> int:
    return 0 if frame is None else int(len(frame))


def _ensure_inference_shard_dirs(
    out: Path,
    *,
    write_posterior_predictive: bool,
    write_residual_samples: bool,
) -> None:
    for key, directory in _inference_shard_dirs(out).items():
        if key == "predictive" and not write_posterior_predictive:
            continue
        if key == "residuals" and not write_residual_samples:
            continue
        directory.mkdir(parents=True, exist_ok=True)


def _inference_shard_dirs(out: Path) -> dict[str, Path]:
    return {
        "samples": out / "posterior_samples",
        "summary": out / "posterior_summary",
        "predictive": out / "posterior_predictive_flux",
        "residuals": out / "posterior_predictive_residuals",
        "residual_summary": out / "posterior_predictive_residual_summary",
        "features": out / "feature_diagnostics",
        "metadata": out / "shard_metadata",
    }


def _inference_shard_paths(out: Path, batch_index: int) -> dict[str, Path]:
    filename = f"batch_{int(batch_index):06d}.parquet"
    paths = {
        key: directory / filename
        for key, directory in _inference_shard_dirs(out).items()
    }
    paths["metadata"] = _inference_shard_dirs(out)["metadata"] / (
        f"batch_{int(batch_index):06d}.json"
    )
    return paths


def _inference_shard_signature(
    run_signature: dict[str, Any],
    *,
    batch_index: int,
    object_id,
    row_index=None,
) -> dict[str, Any]:
    object_id = np.asarray(object_id)
    row_index = np.asarray(row_index, dtype=np.int64) if row_index is not None else None
    return {
        **run_signature,
        "batch_index": int(batch_index),
        "batch_n_objects": int(object_id.shape[0]),
        "batch_first_object_id": _jsonable_object_id(object_id[0])
        if object_id.size
        else None,
        "batch_last_object_id": _jsonable_object_id(object_id[-1])
        if object_id.size
        else None,
        "batch_object_id_digest": _object_id_digest(object_id),
        "batch_first_row_index": int(row_index[0])
        if row_index is not None and row_index.size
        else None,
        "batch_last_row_index": int(row_index[-1])
        if row_index is not None and row_index.size
        else None,
        "batch_row_index_digest": _row_index_digest(row_index)
        if row_index is not None
        else None,
    }


def _jsonable_object_id(value) -> int | float | str | None:
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _object_id_digest(object_id) -> str:
    values = np.asarray(object_id)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(str(values.shape).encode("utf-8"))
    if values.dtype.kind in "biufc":
        digest.update(np.ascontiguousarray(values).view(np.uint8))
    else:
        for value in values.reshape(-1):
            digest.update(repr(_jsonable_object_id(value)).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _row_index_digest(row_index) -> str:
    values = np.asarray(row_index, dtype=np.int64)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def _inference_shard_complete(
    paths: dict[str, Path],
    *,
    write_posterior_predictive: bool,
    write_residual_samples: bool,
    signature: dict[str, Any],
) -> bool:
    required = ["samples", "summary", "residual_summary", "features"]
    if write_posterior_predictive:
        required.append("predictive")
    if write_residual_samples:
        required.append("residuals")
    if not all(
        paths[key].exists() and paths[key].stat().st_size > 0 for key in required
    ):
        return False
    return _inference_shard_signature_matches(paths["metadata"], signature)


def _inference_shard_row_counts(
    paths: dict[str, Path],
    *,
    write_posterior_predictive: bool,
    write_residual_samples: bool,
) -> dict[str, int]:
    return {
        "samples_rows": _parquet_row_count(paths["samples"]),
        "summary_rows": _parquet_row_count(paths["summary"]),
        "predictive_rows": (
            _parquet_row_count(paths["predictive"]) if write_posterior_predictive else 0
        ),
        "residual_rows": (
            _parquet_row_count(paths["residuals"]) if write_residual_samples else 0
        ),
        "residual_summary_rows": _parquet_row_count(paths["residual_summary"]),
        "feature_diagnostics_rows": _parquet_row_count(paths["features"]),
    }


def _inference_shard_manifest_record(
    batch_index: int,
    paths: dict[str, Path],
    counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "batch": int(batch_index),
        "samples_path": str(paths["samples"]),
        "summary_path": str(paths["summary"]),
        "residual_summary_path": str(paths["residual_summary"]),
        "feature_diagnostics_path": str(paths["features"]),
        "predictive_path": str(paths["predictive"]),
        "residuals_path": str(paths["residuals"]),
        "metadata_path": str(paths["metadata"]),
        **{key: int(value) for key, value in counts.items()},
    }


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return int(len(pd.read_parquet(path)))


def _write_inference_shard_metadata(
    paths: dict[str, Path],
    *,
    signature: dict[str, Any],
    counts: dict[str, int],
    batch_index: int,
) -> None:
    write_json(
        paths["metadata"],
        {
            "batch": int(batch_index),
            "signature": signature,
            "counts": {key: int(value) for key, value in counts.items()},
        },
    )


def _inference_shard_signature_matches(
    path: Path,
    signature: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    saved = payload.get("signature", {})
    if not isinstance(saved, dict):
        return False
    return all(saved.get(key) == value for key, value in signature.items())


def _write_shard_manifest(out: Path, payload: dict[str, Any]) -> None:
    write_json(out / "posterior_shards_manifest.json", payload)


def _add_posterior_summary_derived_columns(
    summary: pd.DataFrame,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    *,
    alpha_sed: float,
) -> None:
    if summary.empty:
        return
    columns = [f"{name}_median" for name in parameter_names]
    if any(column not in summary for column in columns):
        return
    theta = summary[columns].to_numpy(dtype=float)
    derived = _derived_columns_from_theta(
        context,
        model_args,
        theta,
        parameter_names,
        alpha_sed=alpha_sed,
        summary=True,
    )
    for name, values in derived.items():
        summary[name] = values


def _derived_columns_from_theta(
    context,
    model_args,
    theta: np.ndarray,
    parameter_names: tuple[str, ...],
    *,
    alpha_sed: float,
    summary: bool,
) -> dict[str, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return {}
    derived_values = np.asarray(
        jax.device_get(
            derived_from_theta_matrix_jax(
                context,
                model_args,
                jnp.asarray(theta, dtype=jnp.float32),
                parameter_names,
            )
        ),
        dtype=float,
    )
    derived = {
        name: derived_values[:, index]
        for index, name in enumerate(DERIVED_QUANTITY_NAMES)
    }
    log_sfr = np.asarray(derived["log10_sfr_at_obs"], dtype=float)
    mass_index = (
        parameter_names.index("log10_stellar_mass")
        if "log10_stellar_mass" in parameter_names
        else None
    )
    log_mass = (
        theta[:, mass_index]
        if mass_index is not None
        else np.full(theta.shape[0], np.nan)
    )
    log_alpha = float(np.log10(max(float(alpha_sed), 1.0e-300)))
    out: dict[str, np.ndarray] = {
        "sfr_at_obs_msun_per_yr": np.asarray(
            derived["sfr_at_obs_msun_per_yr"],
            dtype=float,
        ),
        "log10_sfr_at_obs": log_sfr,
        "log10_sfr_at_obs_alpha_corrected": log_sfr + log_alpha,
        "log10_ssfr_at_obs": log_sfr - log_mass,
        "log10_ssfr_at_obs_alpha_corrected": (log_sfr + log_alpha)
        - (log_mass + log_alpha),
    }
    for index in range(1, 8):
        name = f"sfr_bin_{index}"
        if name in derived:
            out[name] = np.asarray(derived[name], dtype=float)
    if summary:
        renamed = {}
        for name, values in out.items():
            if name.endswith("_alpha_corrected"):
                renamed[name] = values
            else:
                renamed[f"{name}_median"] = values
        return renamed
    return out


def _model_flux_from_x_sample_chunks(
    x_samples: jnp.ndarray,
    latent_spec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    *,
    sample_chunk_size: int,
) -> jnp.ndarray:
    """Decode posterior samples in small chunks to cap DSPS peak memory."""
    x_samples = jnp.asarray(x_samples)
    if int(sample_chunk_size) <= 0:
        raise ValueError("sample_chunk_size must be positive")
    if x_samples.ndim != 3:
        return model_flux_from_x(
            x_samples,
            latent_spec,
            context,
            model_args,
            parameter_names,
        )
    chunks = []
    for start in range(0, int(x_samples.shape[0]), int(sample_chunk_size)):
        x_chunk = x_samples[start : start + int(sample_chunk_size)]
        flux_chunk = model_flux_from_x(
            x_chunk,
            latent_spec,
            context,
            model_args,
            parameter_names,
        )
        chunks.append(jax.block_until_ready(flux_chunk))
    return jnp.concatenate(chunks, axis=0)


def _model_flux_from_x_2d_chunks(
    x: jnp.ndarray,
    latent_spec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    *,
    batch_size: int,
) -> jnp.ndarray:
    """Decode a 2D latent matrix in object/prior-sample batches."""
    x = jnp.asarray(x)
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    if x.ndim != 2:
        raise ValueError(f"x must be [N,D], got {x.shape}")
    chunks = []
    for start in range(0, int(x.shape[0]), int(batch_size)):
        x_chunk = x[start : start + int(batch_size)]
        flux_chunk = model_flux_from_x(
            x_chunk,
            latent_spec,
            context,
            model_args,
            parameter_names,
        )
        chunks.append(jax.block_until_ready(flux_chunk))
    return jnp.concatenate(chunks, axis=0)


def _write_parameter_bound_diagnostics(
    summary: pd.DataFrame,
    latent_spec,
    out: Path,
) -> pd.DataFrame:
    rows = []
    if summary.empty:
        return pd.DataFrame()
    lower = np.asarray(latent_spec.lower, dtype=float)
    upper = np.asarray(latent_spec.upper, dtype=float)
    span = np.maximum(upper - lower, 1.0e-12)
    for index, name in enumerate(latent_spec.names):
        column = f"{name}_median"
        if column not in summary:
            continue
        values = pd.to_numeric(summary[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        unit = (finite - lower[index]) / span[index]
        nearest = np.minimum(unit, 1.0 - unit)
        rows.append(
            {
                "parameter": str(name),
                "n_objects": int(finite.size),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "median_value": float(np.nanmedian(finite)),
                "median_unit_position": float(np.nanmedian(unit)),
                "frac_within_1pct_lower": float(np.mean(unit < 0.01)),
                "frac_within_1pct_upper": float(np.mean(unit > 0.99)),
                "frac_within_1pct_boundary": float(np.mean(nearest < 0.01)),
                "frac_within_5pct_boundary": float(np.mean(nearest < 0.05)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame.to_csv(out / "parameter_bound_diagnostics.csv", index=False)
    frame.to_parquet(out / "parameter_bound_diagnostics.parquet", index=False)
    return frame


def _posterior_predictive_chi2(batch, model_flux, likelihood_config: dict[str, Any]):
    obs = np.asarray(jax.device_get(batch.flux), dtype=float)[None, :, :]
    err = np.asarray(jax.device_get(batch.flux_err), dtype=float)[None, :, :]
    mask = np.asarray(jax.device_get(batch.mask), dtype=bool)[None, :, :]
    model = np.asarray(jax.device_get(model_flux), dtype=float)
    unit = np.maximum(np.maximum(np.abs(obs), err), 1.0e-30)
    obs_scaled = obs / unit
    model_scaled = model / unit
    err_scaled = err / unit
    sigma2 = err_scaled**2
    floor = float(likelihood_config.get("error_floor_frac", 0.0))
    if floor:
        sigma2 = sigma2 + (floor * np.abs(model_scaled)) ** 2
    jitter = float(likelihood_config.get("error_jitter", 0.0))
    if jitter:
        sigma2 = sigma2 + (jitter / unit) ** 2
    sigma = np.sqrt(np.maximum(sigma2, 1.0e-12))
    chi = (model_scaled - obs_scaled) / sigma
    return np.sum(np.where(mask, chi**2, 0.0), axis=-1)
