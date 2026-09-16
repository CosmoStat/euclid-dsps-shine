"""Selection closure, matched NUTS inference, and SFH identifiability audit."""

from __future__ import annotations

import argparse
import copy
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from euclid_dsps.photometry import abmag_to_fnu_cgs
from scripts.feniks_avi_experiments import read, sha, write

PHYSICAL = tuple(FENIKS_SPLINE15D_PARAMETERS[:5])
SFH = tuple(FENIKS_SPLINE15D_PARAMETERS[5:])


def observed_selection_mask(
    frame: pd.DataFrame, *, band: str = "lsst_r", max_mag_ab: float = 29.0
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the configured hard cut to the saved noisy observed flux."""
    flux_column = f"flux_{band}"
    error_column = f"fluxerr_{band}"
    mask_column = f"mask_{band}"
    missing = [c for c in (flux_column, error_column, mask_column) if c not in frame]
    if missing:
        raise ValueError(f"selection columns missing: {missing}")
    flux = pd.to_numeric(frame[flux_column], errors="coerce").to_numpy(np.float64)
    error = pd.to_numeric(frame[error_column], errors="coerce").to_numpy(np.float64)
    valid = frame[mask_column].to_numpy(bool, copy=True)
    valid &= np.isfinite(flux) & np.isfinite(error) & (error > 0)
    threshold = float(abmag_to_fnu_cgs(max_mag_ab))
    selected = valid & (flux > threshold)
    if not selected.any() or selected.all():
        raise ValueError(
            "observed selection must produce distinct nonempty parent and selected cohorts"
        )
    n = int(len(frame))
    k = int(selected.sum())
    alpha = k / n
    se = math.sqrt(alpha * (1.0 - alpha) / n)
    return selected, {
        "band": band,
        "max_mag_ab": float(max_mag_ab),
        "flux_column": flux_column,
        "error_column": error_column,
        "mask_column": mask_column,
        "flux_limit_fnu_cgs": threshold,
        "parent_objects": n,
        "selected_objects": k,
        "empirical_alpha": alpha,
        "empirical_alpha_standard_error": se,
        "selection_rule": f"{mask_column} and {flux_column} > flux(AB={max_mag_ab:g})",
        "noise_contract": "selection applied to the catalog's saved noisy observed flux",
    }


def build_selection_closure(
    parent_catalog: Path,
    out: Path,
    *,
    band: str = "lsst_r",
    max_mag_ab: float = 29.0,
    truth_catalogs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Persist a row-identical parent/selected truth closure and its alpha."""
    identity = "object_id"
    columns = [identity, f"flux_{band}", f"fluxerr_{band}", f"mask_{band}"]
    schema = pd.read_parquet(parent_catalog, columns=None).columns
    if identity not in schema:
        identity = "row_index"
        columns[0] = FENIKS_SPLINE15D_PARAMETERS[0]
    weight_column = "galaxy_weight" if "galaxy_weight" in schema else None
    truth_in_parent = all(name in schema for name in FENIKS_SPLINE15D_PARAMETERS)
    needed = list(
        dict.fromkeys(
            [
                *columns,
                *([weight_column] if weight_column else []),
                *(FENIKS_SPLINE15D_PARAMETERS if truth_in_parent else ()),
            ]
        )
    )
    frame = pd.read_parquet(parent_catalog, columns=needed)
    if identity == "row_index":
        frame.insert(0, identity, np.arange(len(frame), dtype=np.int64))
    if frame[identity].duplicated().any():
        raise ValueError(f"parent identities are not unique: {identity}")
    if not truth_in_parent:
        if not truth_catalogs or identity != "object_id":
            raise ValueError(
                "separate exact truth catalogs keyed by object_id are required"
            )
        truth = pd.concat(
            [
                pd.read_parquet(path, columns=[identity, *FENIKS_SPLINE15D_PARAMETERS])
                for path in truth_catalogs
            ],
            ignore_index=True,
        )
        if truth[identity].duplicated().any():
            raise ValueError("exact truth identities are not unique")
        frame = frame.merge(truth, on=identity, how="left", validate="one_to_one")
    values = frame[list(FENIKS_SPLINE15D_PARAMETERS)].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("finite canonical 15D truth is required for closure")
    selected, summary = observed_selection_mask(frame, band=band, max_mag_ab=max_mag_ab)
    out.mkdir(parents=True, exist_ok=True)
    frame["population_weight"] = (
        pd.to_numeric(frame[weight_column], errors="coerce").to_numpy(np.float64)
        if weight_column
        else np.ones(len(frame), dtype=np.float64)
    )
    if (
        not np.isfinite(frame["population_weight"]).all()
        or (frame["population_weight"] < 0).any()
        or frame["population_weight"].sum() <= 0
    ):
        raise ValueError(
            "finite nonnegative population weights with positive mass required"
        )
    parent = frame[[identity, "population_weight", *FENIKS_SPLINE15D_PARAMETERS]].copy()
    parent.insert(0, "parent_row_index", np.arange(len(parent), dtype=np.int64))
    chosen = parent.loc[selected].copy()
    parent.to_parquet(out / "true_parent.parquet", index=False)
    chosen.to_parquet(out / "true_selected.parquet", index=False)
    pd.DataFrame(
        {
            "parent_row_index": np.arange(len(frame), dtype=np.int64),
            identity: frame[identity].to_numpy(),
            "selected": selected,
            "observed_flux": frame[f"flux_{band}"].to_numpy(np.float64),
            "observed_flux_error": frame[f"fluxerr_{band}"].to_numpy(np.float64),
            "observed_mask": frame[f"mask_{band}"].to_numpy(bool),
        }
    ).to_parquet(out / "selection_identities.parquet", index=False)
    summary.update(
        {
            "status": "EXACT_OBSERVED_SELECTION_COMPLETE",
            "parent_catalog": str(parent_catalog.resolve()),
            "parent_catalog_sha256": sha(parent_catalog),
            "identity_column": identity,
            "population_weight_column": weight_column,
            "weighted_alpha": float(
                frame.loc[selected, "population_weight"].sum()
                / frame["population_weight"].sum()
            ),
            "truth_catalogs": [str(path.resolve()) for path in truth_catalogs],
            "truth_used_to_select": False,
            "truth_role": "closure after selection only",
        }
    )
    write(out / "FINAL.json", summary)
    return summary


def _critical_nuts_inputs(nuts_root: Path, cases: list[str]) -> list[Path]:
    paths = [nuts_root / "MANIFEST.json", nuts_root / "OBSERVED_COHORT.csv"]
    for case in cases:
        target = nuts_root / "nuts" / case / "B_dense_depth6"
        paths.extend([nuts_root / case / "observation.npz", target / "FINAL.json"])
        for chain in range(8):
            chunks = [
                path
                for path in sorted(
                    (target / f"chain_{chain}/chunks").glob("part_*.parquet")
                )
                if not path.stem.endswith("_info")
            ]
            if not chunks:
                raise FileNotFoundError(f"missing NUTS chunks: {case} chain {chain}")
            paths.extend(chunks)
    return paths


def prepare(overnight: Path, nuts_root: Path, root: Path) -> None:
    """Freeze the validation inputs and build the independent selection cohort."""
    import yaml

    if root.exists():
        raise FileExistsError(root)
    overnight_manifest = read(overnight / "MANIFEST.json")
    overnight_final = read(overnight / "report/FINAL.json")
    if overnight_final.get("status") != "OVERNIGHT_REPORT_COMPLETE":
        raise ValueError("completed overnight report required")
    variants = {item["name"]: item for item in overnight_manifest["variants"]}
    if "Q_latest_refresh" not in variants:
        raise ValueError("Q_latest_refresh is absent from overnight manifest")
    variant = variants["Q_latest_refresh"]
    encoder_path = Path(variant["encoder"])
    prior_path = Path(variant["prior"])
    encoder_final = encoder_path.parent / "FINAL.json"
    prior_final = prior_path.parent / "FINAL.json"
    for path in (encoder_path, prior_path, encoder_final, prior_final):
        if not path.is_file():
            raise FileNotFoundError(path)
    if read(encoder_final).get("status") != "TRAINING_COMPLETE":
        raise ValueError("Q_latest_refresh training is incomplete")
    if read(prior_final).get("status") != "PRIOR_TRAINING_COMPLETE":
        raise ValueError("P_latest_prior training is incomplete")
    nuts_manifest = read(nuts_root / "MANIFEST.json")
    cases = list(nuts_manifest["cases"])
    if len(cases) != 8 or len(set(cases)) != 8:
        raise ValueError("the frozen eight-case NUTS cohort is required")
    critical = _critical_nuts_inputs(nuts_root, cases)
    for path in critical:
        if not path.is_file():
            raise FileNotFoundError(path)
    for case in cases:
        final = read(nuts_root / "nuts" / case / "B_dense_depth6/FINAL.json")
        if final.get("status") != "SAMPLING_COMPLETE":
            raise ValueError(f"historical NUTS sampling incomplete: {case}")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    repository = Path(__file__).resolve().parents[1]
    selection_parent = (
        repository
        / "Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/all_50k.parquet"
    )
    truth_root = repository / "Data/diffsky/synthetic/feniks_260617_spline15d"
    selection_truth = tuple(
        truth_root / f"{split}_exact.parquet"
        for split in ("train", "validation", "test")
    )
    source_config = yaml.safe_load(Path(overnight_manifest["config"]).read_text())
    selection_config = source_config["amortized"]["objective"]["selection_correction"]
    if (
        not selection_config.get("enabled")
        or selection_config.get("kind") != "observed_magnitude_limit"
    ):
        raise ValueError("enabled observed-magnitude selection correction required")
    selection = build_selection_closure(
        selection_parent,
        root / "selection",
        band=str(selection_config["band"]),
        max_mag_ab=float(selection_config["max_mag_ab"]),
        truth_catalogs=selection_truth,
    )
    source_training = Path(overnight_manifest["source_training"])
    source_prior_checkpoint = Path(overnight_manifest["source"]["checkpoint"])
    inputs = [
        overnight / "MANIFEST.json",
        overnight / "report/FINAL.json",
        Path(overnight_manifest["config"]),
        Path(overnight_manifest["train_indices"]),
        Path(overnight_manifest["validation_indices"]),
        Path(overnight_manifest["validation_catalog"]),
        Path(overnight_manifest["parent_catalog"]),
        source_prior_checkpoint,
        Path(overnight_manifest["source"]["feature_stats"]),
        Path(variant["encoder"]),
        Path(variant["prior"]),
        encoder_final,
        prior_final,
        selection_parent,
        *selection_truth,
        *critical,
    ]
    manifest = {
        "version": 1,
        "suite": "avi_selection_nuts_sfh_validation_v1",
        "overnight": str(overnight.resolve()),
        "nuts_root": str(nuts_root.resolve()),
        "source_training": str(source_training.resolve()),
        "config": overnight_manifest["config"],
        "train_indices": overnight_manifest["train_indices"],
        "validation_indices": overnight_manifest["validation_indices"],
        "validation_catalog": overnight_manifest["validation_catalog"],
        "feature_stats": overnight_manifest["source"]["feature_stats"],
        "source_checkpoint": str(source_prior_checkpoint.resolve()),
        "encoder": variant["encoder"],
        "latest_prior": variant["prior"],
        "cases": cases,
        "particles": 4096,
        "replicas": 2,
        "seed": 26091331,
        "encoder_template_seed": int(overnight_manifest["seed"]),
        "posterior_targets": {
            "source": "same source prior target as historical NUTS",
            "latest": "current P_latest_prior target",
        },
        "selection": selection,
        "selection_parent_catalog": str(selection_parent.resolve()),
        "selection_truth_catalogs": [str(path.resolve()) for path in selection_truth],
        "truth_used_for_sampling_or_weighting": False,
        "truth_role": "selection closure and post-inference display only",
        "scientific_promotion": False,
        "hashes": {str(path.resolve()): sha(path) for path in inputs},
    }
    write(root / "MANIFEST.json", manifest)
    print(f"Prepared next validation root: {root}", flush=True)


def _checked_manifest(root: Path) -> dict[str, Any]:
    manifest = read(root / "MANIFEST.json")
    for path, digest in manifest["hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"changed validation input: {path}")
    return manifest


def _load_observations(manifest: dict[str, Any]) -> tuple[np.ndarray, ...]:
    root = Path(manifest["nuts_root"])
    arrays: list[list[np.ndarray]] = [[], [], []]
    for case in manifest["cases"]:
        with np.load(root / case / "observation.npz", allow_pickle=False) as data:
            for target, name in zip(arrays, ("flux", "flux_err", "mask"), strict=True):
                target.append(np.asarray(data[name]).reshape(-1))
    return tuple(np.stack(values) for values in arrays)


def infer(root: Path, *, preflight: bool = False, platform: str = "gpu") -> None:
    """Infer Q_latest on the NUTS observations under source and latest targets."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        prepare_adaptive_training_runtime,
    )
    from euclid_dsps.amortized.avi_experiments import Arm, log_prob, normalized_weights
    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.latent import x_to_theta
    from euclid_dsps.amortized.posterior_target import posterior_log_target
    from euclid_dsps.amortized.proposal_expressivity import sample_independent_mixture
    from euclid_dsps.amortized.train import LossBatch, load_checkpoint
    from euclid_dsps.config import load_config
    from scripts.feniks_avi_experiments import (
        initialize_candidate,
        initialize_transport,
        load_optional_component,
    )

    manifest = _checked_manifest(root)
    destination = root / ("preflight" if preflight else "inference")
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / "FINAL.json"
    if final.exists():
        return
    devices = tuple(jax.local_devices())
    if len(devices) != 4 or any(d.platform != platform for d in devices):
        raise ValueError("exactly four requested devices are required")
    if not jax.config.x64_enabled:
        raise ValueError("float64 JAX is required")
    started = time.monotonic()
    write(destination / "PROGRESS.json", {"stage": "loading", "percent": 0.0})
    config = load_config(manifest["config"])
    source_model = load_checkpoint(manifest["source_checkpoint"], config)
    config = copy.deepcopy(config)
    config["amortized"]["encoder"]["transport_float64"] = True
    source_model = eqx.tree_at(
        lambda model: model.encoder,
        source_model,
        initialize_transport(source_model.encoder),
    )
    latest_model = eqx.tree_at(
        lambda model: model.prior,
        source_model,
        load_optional_component(manifest["latest_prior"], source_model.prior),
    )
    runtime_dir = destination / "runtime"
    runtime = prepare_adaptive_training_runtime(
        config,
        runtime_dir,
        train_indices_file=manifest["train_indices"],
        validation_indices_file=manifest["validation_indices"],
        validation_catalog_path=manifest["validation_catalog"],
        fixed_feature_stats_path=manifest["feature_stats"],
        train_population_prior=False,
    )
    if runtime.likelihood_config.get("type", "gaussian") != "gaussian":
        raise ValueError("the historical NUTS comparison requires the Gaussian target")
    candidate = initialize_candidate(
        source_model,
        config,
        runtime.latent_spec,
        Arm("Q_latest", experts=4),
        int(manifest["encoder_template_seed"]),
    )
    candidate = eqx.tree_deserialise_leaves(manifest["encoder"], candidate)
    flux, flux_err, mask = _load_observations(manifest)
    features = make_encoder_features(flux, flux_err, runtime.feature_stats, mask)
    batch = jax.device_get(
        LossBatch(
            jnp.asarray(flux),
            jnp.asarray(flux_err),
            jnp.asarray(mask),
            features,
            jnp.zeros((len(flux), 0), jnp.float32),
        )
    )
    particles = 256 if preflight else int(manifest["particles"])
    replicas = 1 if preflight else int(manifest["replicas"])

    @partial(eqx.filter_pmap, in_axes=(None, 0, 0), devices=devices)
    def bank(active_candidate, local_batch, key):
        x = sample_independent_mixture(
            source_model, active_candidate, key, local_batch.features, particles
        ).x
        logq = log_prob(source_model, active_candidate, local_batch.features, x)

        def target(model, block):
            result = posterior_log_target(
                model,
                block,
                local_batch,
                runtime.latent_spec,
                runtime.context,
                runtime.model_args,
                runtime.parameter_names,
                runtime.likelihood_config,
                runtime.calibration_config,
            )
            return result.logtarget, result.model_flux

        source_target, model_flux = jax.lax.map(
            lambda xx: target(source_model, xx), x.reshape((-1, 8) + x.shape[1:])
        )
        latest_target, _ = jax.lax.map(
            lambda xx: target(latest_model, xx), x.reshape((-1, 8) + x.shape[1:])
        )
        source_target = source_target.reshape(logq.shape)
        latest_target = latest_target.reshape(logq.shape)
        model_flux = model_flux.reshape(x.shape[:-1] + (flux.shape[1],))
        source_weight, source_valid, source_ess = normalized_weights(
            source_target - logq
        )
        latest_weight, latest_valid, latest_ess = normalized_weights(
            latest_target - logq
        )
        metrics = jnp.stack(
            (
                source_ess,
                jnp.max(source_weight, axis=0),
                source_valid,
                latest_ess,
                jnp.max(latest_weight, axis=0),
                latest_valid,
            ),
            axis=-1,
        )
        return (
            x,
            x_to_theta(x, runtime.latent_spec),
            logq,
            source_target,
            latest_target,
            source_weight,
            latest_weight,
            model_flux,
            metrics,
        )

    outputs: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "x",
            "theta",
            "logq",
            "source_logtarget",
            "latest_logtarget",
            "source_weight",
            "latest_weight",
            "model_flux",
        )
    }
    metric_rows = []
    for replica in range(replicas):
        per_replica = {key: [] for key in outputs}
        for offset in range(0, len(flux), 4):
            indices = np.arange(offset, offset + 4)
            payload = jax.tree_util.tree_map(
                lambda value, idx=indices: jnp.asarray(value[idx]).reshape(
                    (4, 1) + value.shape[1:]
                ),
                batch,
            )
            keys = jnp.stack(
                [
                    jax.random.fold_in(
                        jax.random.PRNGKey(int(manifest["seed"]) + replica), int(i)
                    )
                    for i in indices
                ]
            )
            result = jax.device_get(bank(candidate, payload, keys))
            *values, metrics = result
            for key, value in zip(per_replica, values, strict=True):
                per_replica[key].append(np.asarray(value)[:, :, 0])
            metrics = np.asarray(metrics)[:, 0]
            if not metrics[:, [2, 5]].all() or not np.isfinite(metrics).all():
                raise ValueError("invalid ordinary-IS bank; no fallback permitted")
            for local, case_index in enumerate(indices):
                metric_rows.append(
                    {
                        "case": manifest["cases"][int(case_index)],
                        "replica": replica,
                        "source_ess": float(metrics[local, 0]),
                        "source_ess_fraction": float(metrics[local, 0] / particles),
                        "source_max_weight": float(metrics[local, 1]),
                        "latest_ess": float(metrics[local, 3]),
                        "latest_ess_fraction": float(metrics[local, 3] / particles),
                        "latest_max_weight": float(metrics[local, 4]),
                    }
                )
            write(
                destination / "PROGRESS.json",
                {
                    "stage": "inference",
                    "replica": replica,
                    "objects_complete": int(offset + 4),
                    "objects": len(flux),
                    "percent": 100.0
                    * (replica * len(flux) + offset + 4)
                    / (replicas * len(flux)),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        for key, values in per_replica.items():
            outputs[key].append(np.concatenate(values, axis=0))
    archive = destination / "q_latest_nuts_bank.npz"
    np.savez_compressed(archive, **{k: np.stack(v) for k, v in outputs.items()})
    pd.DataFrame(metric_rows).to_csv(destination / "metrics.csv", index=False)
    write(
        final,
        {
            "status": "PREFLIGHT_PASS" if preflight else "INFERENCE_COMPLETE",
            "manifest_sha256": sha(root / "MANIFEST.json"),
            "objects": len(flux),
            "particles": particles,
            "replicas": replicas,
            "bank_sha256": sha(archive),
            "metrics_sha256": sha(destination / "metrics.csv"),
            "truth_used_for_sampling_or_weighting": False,
            "elapsed_seconds": time.monotonic() - started,
            "scientific_promotion": False,
        },
    )
    print(read(final), flush=True)


def _load_nuts_x(nuts_root: Path, case: str) -> np.ndarray:
    target = nuts_root / "nuts" / case / "B_dense_depth6"
    pieces = []
    for chain in range(8):
        paths = [
            path
            for path in sorted(
                (target / f"chain_{chain}/chunks").glob("part_*.parquet")
            )
            if not path.stem.endswith("_info")
        ]
        chain_parts = []
        for path in paths:
            frame = pd.read_parquet(path)
            columns = sorted(
                (c for c in frame if c.startswith("x_")), key=lambda c: int(c[2:])
            )
            chain_parts.append(frame[columns].to_numpy(np.float64))
        pieces.append(np.concatenate(chain_parts))
    return np.concatenate(pieces)


def audit(root: Path, *, platform: str = "gpu") -> None:
    """Compute local decoder spectra and one-sigma nonlinear interventions."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        prepare_adaptive_training_runtime,
    )
    from euclid_dsps.amortized.decoder import model_flux_from_x
    from euclid_dsps.amortized.jacobian_lens import decoder_jacobian_lens
    from euclid_dsps.amortized.train import load_checkpoint
    from euclid_dsps.calibration import (
        apply_global_sed_scale_to_flux,
        apply_per_band_flux_calibration_to_flux,
        global_sed_scale_config,
        per_band_flux_calibration_config,
    )
    from euclid_dsps.config import load_config
    from scripts.feniks_avi_experiments import (
        initialize_transport,
        load_optional_component,
    )

    manifest = _checked_manifest(root)
    inference_final = read(root / "inference/FINAL.json")
    if inference_final.get("status") != "INFERENCE_COMPLETE":
        raise ValueError("completed inference required")
    if not any(device.platform == platform for device in jax.local_devices()):
        raise ValueError(f"{platform} device required")
    destination = root / "audit"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "FINAL.json").exists():
        return
    write(destination / "PROGRESS.json", {"stage": "loading", "percent": 0.0})
    config = load_config(manifest["config"])
    model = load_checkpoint(manifest["source_checkpoint"], config)
    config = copy.deepcopy(config)
    config["amortized"]["encoder"]["transport_float64"] = True
    model = eqx.tree_at(lambda m: m.encoder, model, initialize_transport(model.encoder))
    model = eqx.tree_at(
        lambda m: m.prior,
        model,
        load_optional_component(manifest["latest_prior"], model.prior),
    )
    runtime = prepare_adaptive_training_runtime(
        config,
        destination / "runtime",
        train_indices_file=manifest["train_indices"],
        validation_indices_file=manifest["validation_indices"],
        validation_catalog_path=manifest["validation_catalog"],
        fixed_feature_stats_path=manifest["feature_stats"],
        train_population_prior=False,
    )
    with np.load(root / "inference/q_latest_nuts_bank.npz", allow_pickle=False) as data:
        x = np.asarray(data["x"], np.float64)
        weight = np.asarray(data["latest_weight"], np.float64)
    flux, flux_err, mask = _load_observations(manifest)
    scale_cfg = global_sed_scale_config(runtime.calibration_config)
    band_cfg = per_band_flux_calibration_config(runtime.calibration_config)
    log_sed = model.sed_scale.log_alpha_sed if scale_cfg.enabled else 0.0
    log_band = (
        model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((flux.shape[1],), dtype=jnp.float32)
    )
    spectra, loadings, interventions, summaries = [], [], [], []
    for index, case in enumerate(manifest["cases"]):
        draws = x[:, index].reshape((-1, x.shape[-1]))
        weights = weight[:, index].reshape(-1) / weight.shape[0]
        weights /= weights.sum()
        mean = np.sum(weights[:, None] * draws, axis=0)
        centered = draws - mean
        covariance = np.einsum("n,ni,nj->ij", weights, centered, centered)
        variance = np.maximum(np.diag(covariance), 1e-12)
        representative = draws[
            np.argmin(np.sum((draws - mean) ** 2 / variance, axis=1))
        ]

        def decode(xx):
            raw = model_flux_from_x(
                xx,
                runtime.latent_spec,
                runtime.context,
                runtime.model_args,
                runtime.parameter_names,
            )
            scaled = (
                apply_global_sed_scale_to_flux(raw, log_sed)
                if scale_cfg.enabled
                else raw
            )
            return (
                apply_per_band_flux_calibration_to_flux(scaled, log_band)
                if band_cfg.enabled
                else scaled
            )

        result = decoder_jacobian_lens(
            decode,
            jnp.asarray(representative),
            runtime.latent_spec,
            jnp.asarray(flux[index]),
            jnp.asarray(flux_err[index]),
            jnp.asarray(mask[index]),
            likelihood_type=runtime.likelihood_config.get("type", "gaussian"),
            student_t_dof=float(runtime.likelihood_config.get("student_t_dof", 2.0)),
            error_floor_frac=float(
                runtime.likelihood_config.get("error_floor_frac", 0.02)
            ),
            error_jitter=float(runtime.likelihood_config.get("error_jitter", 0.0)),
            posterior_covariance=jnp.asarray(covariance),
            prior=model.prior,
            prior_active=True,
        )
        arrays = {
            key: np.asarray(jax.device_get(value)) for key, value in result.items()
        }
        singular = arrays["singular_values"]
        relative = singular / max(float(singular[0]), 1e-300)
        vt = arrays["vt_full"]
        posterior_var = arrays["posterior_var"]
        base_flux = arrays["model_flux"]
        sigma_eff = arrays["sigma_eff"]
        for direction in range(vt.shape[0]):
            sfh_loading = float(np.sum(vt[direction, 5:] ** 2))
            spectra.append(
                {
                    "case": case,
                    "direction": direction,
                    "singular_value": float(singular[direction]),
                    "relative_singular_value": float(relative[direction]),
                    "posterior_sd": float(np.sqrt(max(posterior_var[direction], 0))),
                    "physical_loading": 1.0 - sfh_loading,
                    "sfh_loading": sfh_loading,
                }
            )
            for coordinate, value in enumerate(vt[direction]):
                loadings.append(
                    {
                        "case": case,
                        "direction": direction,
                        "parameter": runtime.latent_spec.names[coordinate],
                        "loading": float(value),
                        "squared_loading": float(value * value),
                    }
                )
            step = float(np.sqrt(max(posterior_var[direction], 0)))
            plus = np.asarray(
                jax.device_get(
                    decode(jnp.asarray(representative + step * vt[direction]))
                )
            )
            minus = np.asarray(
                jax.device_get(
                    decode(jnp.asarray(representative - step * vt[direction]))
                )
            )
            valid = mask[index] & np.isfinite(sigma_eff) & (sigma_eff > 0)
            delta = 0.5 * (plus - minus) / sigma_eff
            interventions.append(
                {
                    "case": case,
                    "direction": direction,
                    "posterior_sd_step": step,
                    "flux_change_rms_sigma": float(np.sqrt(np.mean(delta[valid] ** 2))),
                    "max_flux_change_sigma": float(np.max(np.abs(delta[valid]))),
                    "base_predictive_rms_sigma": float(
                        np.sqrt(
                            np.mean(
                                (
                                    (base_flux[valid] - flux[index, valid])
                                    / sigma_eff[valid]
                                )
                                ** 2
                            )
                        )
                    ),
                    "sfh_loading": sfh_loading,
                }
            )
        summaries.append(
            {
                "case": case,
                "condition_number": float(singular[0] / max(singular[-1], 1e-300)),
                "effective_rank_1e_2": int(np.sum(relative >= 1e-2)),
                "weak_direction_sfh_loading": (
                    float(np.mean(np.sum(vt[relative < 1e-2, 5:] ** 2, axis=1)))
                    if np.any(relative < 1e-2)
                    else np.nan
                ),
            }
        )
        write(
            destination / "PROGRESS.json",
            {
                "stage": "jacobian",
                "case": case,
                "percent": 100 * (index + 1) / len(manifest["cases"]),
            },
        )
    for name, records in (
        ("object_summary", summaries),
        ("singular_values", spectra),
        ("latent_direction_loadings", loadings),
        ("posterior_interventions", interventions),
    ):
        pd.DataFrame(records).to_parquet(destination / f"{name}.parquet", index=False)
        pd.DataFrame(records).to_csv(destination / f"{name}.csv", index=False)
    write(
        destination / "FINAL.json",
        {
            "status": "SFH_IDENTIFIABILITY_AUDIT_COMPLETE",
            "manifest_sha256": sha(root / "MANIFEST.json"),
            "objects": len(manifest["cases"]),
            "representative_point": "nearest joint draw to latest-IS weighted latent mean",
            "intervention": "plus/minus one posterior standard deviation along each local singular direction",
            "truth_used": False,
            "scientific_promotion": False,
        },
    )


def _weighted_resample(
    values: np.ndarray, weights: np.ndarray, n: int, seed: int
) -> np.ndarray:
    weights = np.asarray(weights, np.float64)
    weights = weights / weights.sum()
    return values[
        np.random.default_rng(seed).choice(len(values), n, replace=True, p=weights)
    ]


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    """Return deterministic weighted empirical quantiles."""
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    quantiles_array = np.asarray(quantiles, np.float64)
    if (
        values.ndim != 1
        or weights.shape != values.shape
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
        or np.any((quantiles_array < 0) | (quantiles_array > 1))
    ):
        raise ValueError("invalid weighted quantile inputs")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    centers = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / weights.sum()
    return np.interp(
        quantiles_array,
        centers,
        ordered_values,
        left=ordered_values[0],
        right=ordered_values[-1],
    )


def _corner_compare(
    path: Path,
    first: np.ndarray,
    second: np.ndarray,
    truth: np.ndarray,
    *,
    first_label: str,
    second_label: str,
    title: str,
) -> None:
    """Draw one readable five-dimensional two-distribution corner."""
    import matplotlib.pyplot as plt

    dimensions = len(PHYSICAL)
    figure, axes = plt.subplots(dimensions, dimensions, figsize=(10, 10))
    rng = np.random.default_rng(260913)
    first_take = first[rng.choice(len(first), min(1400, len(first)), replace=False)]
    second_take = second[rng.choice(len(second), min(1400, len(second)), replace=False)]
    limits = []
    for parameter in range(dimensions):
        pooled = np.concatenate((first[:, parameter], second[:, parameter]))
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        lo = min(float(lo), float(truth[parameter]))
        hi = max(float(hi), float(truth[parameter]))
        padding = max((hi - lo) * 0.04, 1e-8)
        lo -= padding
        hi += padding
        limits.append((float(lo), float(hi)))
    for row in range(dimensions):
        for column in range(dimensions):
            axis = axes[row, column]
            if row < column:
                axis.axis("off")
                continue
            if row == column:
                edges = np.linspace(*limits[column], 45)
                axis.hist(
                    first[:, column],
                    edges,
                    density=True,
                    histtype="step",
                    color="#D95F02",
                    linewidth=1.4,
                )
                axis.hist(
                    second[:, column],
                    edges,
                    density=True,
                    histtype="step",
                    color="#7251B5",
                    linewidth=1.4,
                    linestyle="--",
                )
                axis.axvline(truth[column], color="black", linewidth=1.5)
                axis.set_yticks([])
            else:
                axis.scatter(
                    first_take[:, column],
                    first_take[:, row],
                    s=4,
                    alpha=0.08,
                    color="#D95F02",
                    rasterized=True,
                )
                histogram, x_edges, y_edges = np.histogram2d(
                    second_take[:, column], second_take[:, row], bins=30
                )
                positive = histogram[histogram > 0]
                if positive.size:
                    levels = np.unique(np.quantile(positive, [0.55, 0.78, 0.92]))
                    if levels.size:
                        axis.contour(
                            0.5 * (x_edges[:-1] + x_edges[1:]),
                            0.5 * (y_edges[:-1] + y_edges[1:]),
                            histogram.T,
                            levels=levels,
                            colors="#7251B5",
                            linewidths=1.0,
                        )
                axis.scatter(
                    truth[column], truth[row], marker="x", s=34, color="black", zorder=4
                )
            axis.set_xlim(*limits[column])
            if row != column:
                axis.set_ylim(*limits[row])
            if row == dimensions - 1:
                axis.set_xlabel(PHYSICAL[column].replace("log10_", ""), fontsize=8)
            else:
                axis.set_xticklabels([])
            if column == 0 and row:
                axis.set_ylabel(PHYSICAL[row].replace("log10_", ""), fontsize=8)
            elif column:
                axis.set_yticklabels([])
            axis.tick_params(labelsize=7)
    figure.legend(
        handles=[
            plt.Line2D([], [], color="#D95F02", label=first_label),
            plt.Line2D([], [], color="#7251B5", linestyle="--", label=second_label),
            plt.Line2D([], [], color="black", marker="x", linestyle="", label="Truth"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(title, fontsize=13, y=0.975)
    figure.subplots_adjust(
        left=0.09, right=0.98, bottom=0.08, top=0.92, wspace=0.08, hspace=0.08
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def report(root: Path) -> None:
    """Build compact selection, NUTS, and SFH figures plus a decision report."""
    import matplotlib

    matplotlib.use("Agg")
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import yaml

    from euclid_dsps.amortized.latent import latent_spec_from_config, x_to_theta
    from euclid_dsps.amortized.mira import evaluate_feniks_mira
    from scripts.build_feniks_avi_wrapup import _match_nuts_truth
    from scripts.feniks_avi_inference import sample_frame

    manifest = _checked_manifest(root)
    for path, status in (
        (root / "inference/FINAL.json", "INFERENCE_COMPLETE"),
        (root / "audit/FINAL.json", "SFH_IDENTIFIABILITY_AUDIT_COMPLETE"),
        (root / "selection/FINAL.json", "EXACT_OBSERVED_SELECTION_COMPLETE"),
    ):
        if read(path).get("status") != status:
            raise ValueError(f"expected {status}: {path}")
    destination = root / "report"
    if (destination / "FINAL.json").exists():
        return
    (destination / "nuts").mkdir(parents=True, exist_ok=True)
    (destination / "selection").mkdir(exist_ok=True)
    (destination / "sfh").mkdir(exist_ok=True)
    config = yaml.safe_load(Path(manifest["config"]).read_text())
    spec = latent_spec_from_config(config)
    truth_dir = destination / "truth_match"
    truth_dir.mkdir(exist_ok=True)
    (truth_dir / "tables").mkdir(exist_ok=True)
    truth = _match_nuts_truth(
        Path(manifest["nuts_root"]), Path(manifest["config"]), truth_dir
    )
    with np.load(root / "inference/q_latest_nuts_bank.npz", allow_pickle=False) as data:
        theta = np.asarray(data["theta"], np.float64)
        source_weight = np.asarray(data["source_weight"], np.float64)
        latest_weight = np.asarray(data["latest_weight"], np.float64)
    cohort = pd.read_csv(Path(manifest["nuts_root"]) / "OBSERVED_COHORT.csv").fillna("")
    comparison_rows = []
    for index, item in cohort.iterrows():
        case = item["case"]
        raw = theta[:, index].reshape((-1, theta.shape[-1]))
        source_is = _weighted_resample(
            raw, source_weight[:, index].reshape(-1), 4096, 40 + index
        )
        latest_is = _weighted_resample(
            raw, latest_weight[:, index].reshape(-1), 4096, 80 + index
        )
        nuts_x = _load_nuts_x(Path(manifest["nuts_root"]), case)
        nuts = np.asarray(x_to_theta(jnp.asarray(nuts_x), spec), np.float64)
        nuts = nuts[
            np.random.default_rng(120 + index).choice(len(nuts), 4096, replace=False)
        ]
        fig, axes = plt.subplots(1, 5, figsize=(14, 2.7))
        for p, ax in enumerate(axes):
            pooled = np.concatenate(
                (raw[:, p], source_is[:, p], latest_is[:, p], nuts[:, p])
            )
            lo, hi = np.quantile(pooled, [0.005, 0.995])
            edges = np.linspace(lo, hi, 45)
            ax.hist(
                raw[:, p],
                edges,
                density=True,
                histtype="step",
                color="#2C7FB8",
                label="Encoder",
            )
            ax.hist(
                source_is[:, p],
                edges,
                density=True,
                histtype="step",
                color="#D95F02",
                label="IS source",
            )
            ax.hist(
                latest_is[:, p],
                edges,
                density=True,
                histtype="step",
                color="#1B9E77",
                label="IS latest",
            )
            ax.hist(
                nuts[:, p],
                edges,
                density=True,
                histtype="step",
                color="#7251B5",
                linestyle="--",
                label="NUTS",
            )
            ax.axvline(truth[case][p], color="black", linewidth=1.5, label="Truth")
            ax.set_title(PHYSICAL[p].replace("log10_", ""), fontsize=9)
            ax.set_yticks([])
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(
            f"{case} | {str(item.descriptive_tags).replace(';', ', ')}", fontsize=12
        )
        fig.tight_layout()
        fig.savefig(destination / "nuts" / f"{case}.png", dpi=180)
        plt.close(fig)
        _corner_compare(
            destination / "nuts" / f"{case}_source_target_corner.png",
            source_is,
            nuts,
            truth[case],
            first_label="Q_latest IS, source target",
            second_label="Historical NUTS",
            title=f"{case}: target-compatible posterior comparison",
        )
        _corner_compare(
            destination / "nuts" / f"{case}_current_target_corner.png",
            raw,
            latest_is,
            truth[case],
            first_label="Q_latest encoder",
            second_label="Q_latest ordinary IS",
            title=f"{case}: current latest-prior posterior",
        )
        nuts_iqr = np.maximum(
            np.quantile(nuts, 0.75, axis=0) - np.quantile(nuts, 0.25, axis=0), 1e-8
        )
        for p, name in enumerate(FENIKS_SPLINE15D_PARAMETERS):
            comparison_rows.append(
                {
                    "case": case,
                    "parameter": name,
                    "source_is_median_distance_over_nuts_iqr": float(
                        abs(np.median(source_is[:, p]) - np.median(nuts[:, p]))
                        / nuts_iqr[p]
                    ),
                    "latest_is_median_distance_over_nuts_iqr": float(
                        abs(np.median(latest_is[:, p]) - np.median(nuts[:, p]))
                        / nuts_iqr[p]
                    ),
                    "truth": float(truth[case][p]),
                }
            )
    pd.DataFrame(comparison_rows).to_csv(
        destination / "nuts_comparison.csv", index=False
    )

    parent = pd.read_parquet(root / "selection/true_parent.parquet")
    selected = pd.read_parquet(root / "selection/true_selected.parquet")
    with np.load(
        Path(manifest["overnight"]) / "arms/B_latest/prior_population.npz",
        allow_pickle=False,
    ) as data:
        learned_selected = np.asarray(data["selected_theta"], np.float64)
        learned_parent = np.asarray(data["theta"], np.float64)
    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8))
    for p, ax in enumerate(axes):
        values = [parent[PHYSICAL[p]], selected[PHYSICAL[p]], learned_selected[:, p]]
        lo, hi = np.quantile(np.concatenate(values), [0.005, 0.995])
        edges = np.linspace(lo, hi, 45)
        ax.hist(
            values[0],
            edges,
            weights=parent.population_weight,
            density=True,
            histtype="step",
            color="#6B7280",
            label="True parent",
        )
        ax.hist(
            values[1],
            edges,
            weights=selected.population_weight,
            density=True,
            histtype="step",
            color="black",
            linewidth=1.5,
            label="True selected",
        )
        ax.hist(
            values[2],
            edges,
            density=True,
            histtype="step",
            color="#D95F02",
            label="Learned selected",
        )
        ax.set_title(PHYSICAL[p].replace("log10_", ""), fontsize=9)
        ax.set_yticks([])
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Exact observed selection closure", fontsize=12)
    fig.tight_layout()
    fig.savefig(destination / "selection/physical_5d.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.1))
    for p, ax in enumerate(axes.flat, start=5):
        name = FENIKS_SPLINE15D_PARAMETERS[p]
        pooled = np.concatenate((parent[name], selected[name], learned_selected[:, p]))
        lo, hi = np.quantile(pooled, [0.005, 0.995])
        edges = np.linspace(lo, hi, 42)
        ax.hist(
            parent[name],
            edges,
            weights=parent.population_weight,
            density=True,
            histtype="step",
            color="#6B7280",
            label="True parent",
        )
        ax.hist(
            selected[name],
            edges,
            weights=selected.population_weight,
            density=True,
            histtype="step",
            color="black",
            linewidth=1.4,
            label="True selected",
        )
        ax.hist(
            learned_selected[:, p],
            edges,
            density=True,
            histtype="step",
            color="#D95F02",
            label="Learned selected",
        )
        ax.set_title(name.replace("sfh_dlog_sfr_", "SFH "), fontsize=9)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Selection closure in the ten SFH degrees of freedom", fontsize=12)
    fig.tight_layout()
    fig.savefig(destination / "selection/sfh_10d.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(14, 2.8))
    for p, ax in enumerate(axes):
        values = [parent[PHYSICAL[p]], learned_parent[:, p]]
        lo, hi = np.quantile(np.concatenate(values), [0.005, 0.995])
        edges = np.linspace(lo, hi, 45)
        ax.hist(
            values[0],
            edges,
            weights=parent.population_weight,
            density=True,
            histtype="step",
            color="black",
            linewidth=1.5,
            label="True parent",
        )
        ax.hist(
            values[1],
            edges,
            density=True,
            histtype="step",
            color="#2C7FB8",
            label="Learned parent",
        )
        ax.set_title(PHYSICAL[p].replace("log10_", ""), fontsize=9)
        ax.set_yticks([])
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Parent population closure", fontsize=12)
    fig.tight_layout()
    fig.savefig(destination / "selection/parent_physical_5d.png", dpi=180)
    plt.close(fig)
    selection_metrics = []
    from scipy.stats import wasserstein_distance

    for p, name in enumerate(FENIKS_SPLINE15D_PARAMETERS):
        selected_iqr = _weighted_quantile(
            selected[name].to_numpy(),
            selected.population_weight.to_numpy(),
            (0.25, 0.75),
        )
        parent_iqr = _weighted_quantile(
            parent[name].to_numpy(),
            parent.population_weight.to_numpy(),
            (0.25, 0.75),
        )
        scale = max(float(selected_iqr[1] - selected_iqr[0]), 1e-8)
        parent_scale = max(float(parent_iqr[1] - parent_iqr[0]), 1e-8)
        selection_metrics.append(
            {
                "parameter": name,
                "group": "physical" if p < 5 else "sfh",
                "selected_wasserstein_over_true_iqr": float(
                    wasserstein_distance(
                        selected[name],
                        learned_selected[:, p],
                        u_weights=selected.population_weight,
                    )
                    / scale
                ),
                "parent_wasserstein_over_true_iqr": float(
                    wasserstein_distance(
                        parent[name],
                        learned_parent[:, p],
                        u_weights=parent.population_weight,
                    )
                    / parent_scale
                ),
                "true_selection_mean_shift_over_parent_iqr": float(
                    (
                        np.average(selected[name], weights=selected.population_weight)
                        - np.average(parent[name], weights=parent.population_weight)
                    )
                    / parent_scale
                ),
                "learned_selection_mean_shift_over_parent_iqr": float(
                    (np.mean(learned_selected[:, p]) - np.mean(learned_parent[:, p]))
                    / parent_scale
                ),
            }
        )
    pd.DataFrame(selection_metrics).to_csv(
        destination / "selection_metrics.csv", index=False
    )

    prior_mira_rows = []
    for population_index, (population, truth_frame, learned) in enumerate(
        (
            ("parent", parent, learned_parent),
            ("selected", selected, learned_selected),
        )
    ):
        truth_values = _weighted_resample(
            truth_frame[list(FENIKS_SPLINE15D_PARAMETERS)].to_numpy(np.float64),
            truth_frame.population_weight.to_numpy(np.float64),
            512,
            26091400 + population_index,
        )
        mira_truth = pd.DataFrame(truth_values, columns=FENIKS_SPLINE15D_PARAMETERS)
        mira_truth.insert(0, "row_index", np.arange(len(mira_truth)))
        mira_truth.insert(0, "object_id", np.arange(len(mira_truth)))
        truth_path = destination / "selection" / f"{population}_mira_truth.parquet"
        mira_truth.to_parquet(truth_path, index=False)

        draw_indices = np.random.default_rng(26091500 + population_index).integers(
            0, len(learned), size=(len(mira_truth), 512)
        )
        posterior = learned[draw_indices]
        posterior_path = (
            destination / "selection" / f"learned_{population}_mira_draws.parquet"
        )
        sample_frame(
            posterior,
            np.arange(len(mira_truth)),
            FENIKS_SPLINE15D_PARAMETERS,
        ).to_parquet(posterior_path, index=False)
        mira_dir = destination / "selection" / f"{population}_mira"
        evaluate_feniks_mira(
            truth_path=truth_path,
            posterior_specs=[(f"learned_{population}", posterior_path)],
            out_dir=mira_dir,
            samples_per_object=512,
            seed=26091600 + population_index,
            num_regions=100,
            num_bootstrap=500,
        )
        population_scores = pd.read_csv(mira_dir / "mira_scores.csv")
        population_scores.insert(0, "population", population)
        prior_mira_rows.append(population_scores)
    prior_mira = pd.concat(prior_mira_rows, ignore_index=True)
    prior_mira.to_csv(destination / "selection_prior_mira.csv", index=False)

    spectra = pd.read_csv(root / "audit/singular_values.csv")
    interventions = pd.read_csv(root / "audit/posterior_interventions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for case, group in spectra.groupby("case"):
        axes[0].semilogy(
            group.direction, group.relative_singular_value, alpha=0.7, label=case
        )
    weak = spectra.sort_values(["case", "direction"]).pivot(
        index="case", columns="direction", values="sfh_loading"
    )
    image = axes[1].imshow(weak, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[0].set(
        xlabel="Local direction",
        ylabel="Relative singular value",
        title="Decoder information spectrum",
    )
    axes[1].set(
        xlabel="Local direction", ylabel="Object", title="SFH loading of each direction"
    )
    axes[1].set_yticks(range(len(weak.index)), weak.index)
    fig.colorbar(image, ax=axes[1], label="SFH squared loading")
    fig.tight_layout()
    fig.savefig(destination / "sfh/jacobian_identifiability.png", dpi=180)
    plt.close(fig)
    metrics = pd.read_csv(root / "inference/metrics.csv")
    selection_receipt = read(root / "selection/FINAL.json")
    physical_mira = prior_mira.loc[
        prior_mira.group == "physical_5d", ["population", "score", "bootstrap_std"]
    ].set_index("population")
    report_text = f"""# FENIKS next validation\n\n## Scope\n\nThis suite uses the frozen `Q_latest_refresh` encoder. Truth is used only after inference. The source-prior weights are the target-compatible comparison to the historical NUTS chains; latest-prior weights are the current posterior. Historical NUTS failed its saved convergence gate and remains a geometric diagnostic.\n\n## Selection closure\n\n- Parent objects: {selection_receipt["parent_objects"]}\n- Truly selected objects: {selection_receipt["selected_objects"]}\n- Empirical alpha: {selection_receipt["empirical_alpha"]:.6f} +/- {selection_receipt["empirical_alpha_standard_error"]:.6f}\n- Population-weighted alpha: {selection_receipt["weighted_alpha"]:.6f}\n- Rule: `{selection_receipt["selection_rule"]}` on saved noisy flux, never on truth.\n- Learned-parent physical 5D MIRA: {physical_mira.loc["parent", "score"]:.4f} +/- {physical_mira.loc["parent", "bootstrap_std"]:.4f}.\n- Learned-selected physical 5D MIRA: {physical_mira.loc["selected", "score"]:.4f} +/- {physical_mira.loc["selected", "bootstrap_std"]:.4f}.\n\n## Importance support on the eight NUTS observations\n\n- Median source-target ESS fraction: {metrics.source_ess_fraction.median():.4f}\n- Median latest-target ESS fraction: {metrics.latest_ess_fraction.median():.4f}\n- Source-target is the only valid direct NUTS comparison.\n\n## SFH identifiability\n\n- Median SFH loading among locally weak directions: {spectra.loc[spectra.relative_singular_value < 1e-2, "sfh_loading"].median():.3f}\n- Median nonlinear one-sigma flux change for SFH-dominated directions: {interventions.loc[interventions.sfh_loading > 0.5, "flux_change_rms_sigma"].median():.3f} sigma.\n\n## Decision gates\n\n1. Treat the selection model as empirically closed only if learned-selected distances improve over the parent baseline in `selection_metrics.csv`.\n2. Treat Q_latest as closer to NUTS only from source-target weights and only as geometry because NUTS did not converge.\n3. Proceed to the factorized 5D physical plus conditional-SFH flow only if weak local directions are SFH-dominated and their nonlinear flux interventions are small.\n"""
    (destination / "REPORT.md").write_text(report_text)
    artifacts = {
        str(path.relative_to(root)): sha(path)
        for path in destination.rglob("*")
        if path.is_file()
    }
    write(
        destination / "FINAL.json",
        {
            "status": "NEXT_VALIDATION_REPORT_COMPLETE",
            "manifest_sha256": sha(root / "MANIFEST.json"),
            "artifacts": artifacts,
            "truth_used_for_training_or_inference": False,
            "scientific_promotion": False,
        },
    )
    print(report_text, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--overnight", type=Path, required=True)
    prepare_parser.add_argument("--nuts-root", type=Path, required=True)
    prepare_parser.add_argument("--root", type=Path, required=True)
    for name in ("preflight", "infer", "audit", "report"):
        item = sub.add_parser(name)
        item.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.overnight, args.nuts_root, args.root)
    elif args.command == "preflight":
        infer(args.root, preflight=True)
    elif args.command == "infer":
        infer(args.root)
    elif args.command == "audit":
        audit(args.root)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
