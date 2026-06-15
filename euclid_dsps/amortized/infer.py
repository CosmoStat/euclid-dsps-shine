"""Inference entry points for trained amortized FS2 models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.calibration import (
    alpha_metadata,
    apply_global_sed_scale_to_flux,
    global_sed_scale_config,
)
from euclid_dsps.diffsky_redshift_ablation import write_redshift_metrics_for_run
from euclid_dsps.filters import load_filters
from euclid_dsps.io import ensure_dir, write_json
from euclid_dsps.model import dynamic_model_args, load_context

from .catalog import (
    learned_prior_samples_frame,
    posterior_predictive_flux_frame,
    posterior_samples_frame,
    posterior_summary_frame,
)
from .config import amortized_config
from .data import iter_photometry_batches_from_config
from .decoder import model_flux_from_x
from .diagnostics import (
    feature_diagnostics_frame,
    posterior_predictive_residual_frame,
    posterior_predictive_residual_summary_frame,
    summarize_inference_outputs,
)
from .features import read_feature_stats
from .latent import latent_spec_from_config, x_to_theta
from .likelihood import photometric_loglike
from .train import _effective_jax_batch_size, load_checkpoint


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
    inference_cfg = cfg.get("inference", {})
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
            f"write_residual_samples={write_residual_samples}"
        )
        if jax_batch_size != int(batch_size):
            print(
                "[amortized] capping JAX/DSPS inference batch size: "
                f"requested_batch_size={int(batch_size)} "
                f"jax_batch_size={jax_batch_size}"
            )
        print(f"[amortized] JAX backend: {jax.default_backend()} devices={jax.devices()}")
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
    }
    shards_written: list[dict[str, Any]] = []
    shards_skipped: list[dict[str, Any]] = []
    if shard_outputs:
        _ensure_inference_shard_dirs(
            out,
            write_posterior_predictive=write_posterior_predictive,
            write_residual_samples=write_residual_samples,
        )
    run_signature = {
        "catalog_path": str(config.get("catalog_path")),
        "checkpoint": str(checkpoint),
        "feature_stats_path": str(feature_stats_path),
        "batch_size": int(batch_size),
        "jax_batch_size": int(jax_batch_size),
        "limit": int(limit) if limit is not None else None,
        "posterior_samples": int(posterior_samples),
        "decoder_sample_chunk_size": int(decoder_sample_chunk_size),
        "write_posterior_predictive": bool(write_posterior_predictive),
        "write_residual_samples": bool(write_residual_samples),
        "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
    }
    n_objects_total = 0
    band_names = tuple(str(band["name"]) for band in config["bands"])
    for batch_index, batch in enumerate(
        iter_photometry_batches_from_config(
            config,
            batch_size=int(jax_batch_size),
            limit=limit,
            feature_stats=feature_stats,
        ),
        start=1,
    ):
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
        logprior = model.prior.log_prob(x_samples)
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
        chi2 = _posterior_predictive_chi2(batch, model_flux)
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
            log_alpha_sed=log_alpha_sed,
            alpha_sed=alpha_sed,
        )
        predictive_frame = (
            posterior_predictive_flux_frame(
                object_id,
                model_flux_np,
                band_names,
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
        )
        feature_frame = feature_diagnostics_frame(
            object_id,
            jax.device_get(batch.features),
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
        pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    )
    residual_summary = (
        pd.concat(residual_summary_frames, ignore_index=True)
        if residual_summary_frames
        else pd.DataFrame()
    )
    if not shard_outputs:
        row_counts = {
            "samples_rows": int(len(samples)),
            "summary_rows": int(len(summary)),
            "predictive_rows": int(len(predictive)),
            "residual_rows": int(len(residuals)),
            "residual_summary_rows": int(len(residual_summary)),
            "feature_diagnostics_rows": int(len(feature_diagnostics)),
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
    learned_prior = learned_prior_samples_frame(
        jax.device_get(prior_x),
        jax.device_get(prior_theta),
        latent_spec.names,
        jax.device_get(prior_logprob),
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
            feature_diagnostics.to_parquet(out / "feature_diagnostics.parquet", index=False)
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
            predictive.to_parquet(out / "posterior_predictive_flux.parquet", index=False)
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
    write_json(
        out / "inference_summary.json",
        {"global_sed_scale": global_sed_scale_payload},
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
    write_json(
        out / "inference_summary.json",
        {
            "checkpoint": str(checkpoint),
            "feature_stats_path": str(feature_stats_path),
            "limit": limit,
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
    )
    if verbose:
        print("[amortized] inference complete")
        print(f"[amortized] summary: {out / 'inference_summary.json'}")


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
) -> dict[str, Any]:
    object_id = np.asarray(object_id)
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
    if not all(paths[key].exists() and paths[key].stat().st_size > 0 for key in required):
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
            _parquet_row_count(paths["predictive"])
            if write_posterior_predictive
            else 0
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


def _posterior_predictive_chi2(batch, model_flux):
    obs = batch.flux[None, :, :]
    err = batch.flux_err[None, :, :]
    mask = batch.mask[None, :, :]
    chi = np.asarray((model_flux - obs) / err)
    valid = np.asarray(mask)
    return np.sum(np.where(valid, chi**2, 0.0), axis=-1)
