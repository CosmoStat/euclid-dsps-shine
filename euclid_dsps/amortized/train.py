"""Training loop for FS2 amortized inference."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from euclid_dsps.calibration import (
    alpha_from_log_alpha,
    alpha_metadata,
    apply_global_sed_scale_to_flux,
    apply_per_band_flux_calibration_to_flux,
    global_sed_scale_config,
    global_sed_scale_prior_penalty,
    make_global_sed_scale_state,
    make_per_band_flux_calibration_state,
    per_band_flux_calibration_config,
    per_band_flux_calibration_metadata,
    per_band_flux_calibration_prior_penalty,
)
from euclid_dsps.filters import load_filters
from euclid_dsps.io import (
    ensure_dir,
    load_row_indices,
    truth_column_from_spec,
    write_json,
)
from euclid_dsps.model import dynamic_model_args, load_context

from .catalog_identity import write_catalog_fingerprint
from .collapse_gates import write_training_collapse_gate
from .config import amortized_config, require_amortized_dependencies
from .data import (
    PhotometryBatch,
    iter_photometry_batches_from_arrays,
    load_photometry_arrays_from_config,
)
from .decoder import model_flux_from_x
from .diagnostics import write_training_diagnostics
from .elbo import (
    AmortizedModel,
    is_deterministic_reconstruction,
    negative_elbo,
    objective_mode,
)
from .encoder import GaussianEncoder
from .features import (
    FeatureStats,
    compute_feature_stats,
    make_encoder_features,
    write_feature_stats,
)
from .flows import (
    RealNVPPrior,
    RQSplineCouplingPrior,
    StandardNormalPrior,
    assert_flow_integrity,
)
from .latent import (
    LatentSpec,
    initial_theta_from_config,
    latent_spec_from_config,
    latent_spec_to_jsonable,
    theta_to_x,
    write_latent_prior_geometry,
    x_to_theta,
)
from .likelihood import photometric_loglike, photometric_normalized_residual
from .posterior import (
    ConditionalFlowEncoder,
    PriorTransportGaussianEncoder,
    posterior_log_prob,
    sample_posterior,
)

eqx, optax = require_amortized_dependencies()

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a core dependency today
    tqdm = None


@dataclass(frozen=True)
class TrainingSplit:
    """Catalog-row split used by FS2 amortized training."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_redshift: np.ndarray
    validation_redshift: np.ndarray
    redshift_column: str | None
    redshift_bins: np.ndarray
    selection_mode: str
    stratified_strategy: str
    validation_fraction: float
    row_indices_file: str | None = None
    train_indices_file: str | None = None
    validation_indices_file: str | None = None


class LossBatch(NamedTuple):
    """JAX-friendly batch payload used by the compiled training step."""

    flux: jnp.ndarray
    flux_err: jnp.ndarray
    mask: jnp.ndarray
    features: jnp.ndarray
    truth_theta: jnp.ndarray


class JitLatentSpec(NamedTuple):
    """JAX-friendly latent transform spec used by compiled training."""

    names: tuple[str, ...]
    lower: jnp.ndarray
    upper: jnp.ndarray
    raw_center: jnp.ndarray | None = None
    raw_scale: jnp.ndarray | None = None
    normalization: str = "identity"
    transform_family: jnp.ndarray | None = None
    transform_location: jnp.ndarray | None = None
    transform_lambda: jnp.ndarray | None = None


class _StaticArg:
    """Hash static JIT payloads by identity instead of recursive contents."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f"_StaticArg({type(self.value).__name__})"


def build_amortized_model(
    config: dict[str, Any],
    key,
    *,
    latent_spec: LatentSpec | None = None,
) -> AmortizedModel:
    """Instantiate encoder and configured prior from config."""
    cfg = amortized_config(config)
    k_encoder, k_prior = jax.random.split(key)
    encoder_cfg = cfg["encoder"]
    input_dim = int(encoder_cfg.get("input_dim", 20))
    latent_dim = int(encoder_cfg.get("latent_dim", 16))
    if latent_spec is None:
        try:
            latent_spec = _latent_spec_for_amortized_config(config)
        except (KeyError, ValueError):
            latent_spec = None
    expected_latent_dim = latent_dim if latent_spec is None else len(latent_spec.names)
    if latent_dim != expected_latent_dim:
        raise ValueError(
            "amortized.encoder.latent_dim must match configured free parameters: "
            f"latent_dim={latent_dim}, expected={expected_latent_dim}"
        )
    encoder_kwargs = {
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "hidden_sizes": tuple(
            int(v) for v in encoder_cfg.get("hidden_sizes", [256, 256, 256])
        ),
        "activation": str(encoder_cfg.get("activation", "gelu")),
        "log_std_min": float(encoder_cfg.get("log_std_min", -6.0)),
        "log_std_max": float(encoder_cfg.get("log_std_max", 2.0)),
        "initial_log_std": float(encoder_cfg.get("initial_log_std", -1.0)),
    }
    encoder_type = str(encoder_cfg.get("type", "gaussian_mlp")).strip().lower()
    if encoder_type == "gaussian_mlp":
        encoder = GaussianEncoder(k_encoder, **encoder_kwargs)
    elif encoder_type in {"prior_transport_gaussian", "transport_gaussian"}:
        encoder = PriorTransportGaussianEncoder(k_encoder, **encoder_kwargs)
    elif encoder_type in {"conditional_flow", "conditional_normalizing_flow"}:
        encoder = ConditionalFlowEncoder(
            k_encoder,
            **encoder_kwargs,
            family=str(encoder_cfg.get("flow_family", "realnvp")),
            n_layers=int(encoder_cfg.get("flow_layers", 4)),
            hidden_size=int(encoder_cfg.get("flow_hidden_size", 128)),
            n_bins=int(encoder_cfg.get("flow_bins", 8)),
            scale_clamp=float(encoder_cfg.get("flow_scale_clamp", 0.5)),
            shift_clamp=float(encoder_cfg.get("flow_shift_clamp", 3.0)),
            tail_bound=float(encoder_cfg.get("flow_tail_bound", 8.0)),
            min_bin_width=float(encoder_cfg.get("flow_min_bin_width", 1.0e-3)),
            min_bin_height=float(encoder_cfg.get("flow_min_bin_height", 1.0e-3)),
            min_derivative=float(encoder_cfg.get("flow_min_derivative", 1.0e-3)),
            init_scale=float(encoder_cfg.get("flow_init_scale", 0.0)),
        )
    else:
        raise ValueError(f"Unsupported amortized encoder type: {encoder_type}")
    encoder = _initialize_encoder_mean_if_possible(
        config, encoder, latent_spec=latent_spec
    )
    prior = build_prior_from_config(
        config,
        k_prior,
        latent_dim=latent_dim,
        active_spec=latent_spec,
    )
    sed_scale = make_global_sed_scale_state(config)
    band_names = tuple(str(band["name"]) for band in config.get("bands", []))
    band_calibration = make_per_band_flux_calibration_state(config, band_names)
    return AmortizedModel(
        encoder=encoder,
        prior=prior,
        sed_scale=sed_scale,
        band_calibration=band_calibration,
    )


def build_prior_from_config(
    config: dict[str, Any],
    key,
    *,
    latent_dim: int,
    active_spec: LatentSpec | None = None,
):
    """Build or load the configured amortized prior source."""
    cfg = amortized_config(config)
    prior_cfg = cfg["prior"]
    source = str(prior_cfg.get("source", "joint_realnvp"))
    if source == "standard_normal":
        return StandardNormalPrior(latent_dim=int(latent_dim))
    if source in {"joint_realnvp", "realnvp"}:
        return RealNVPPrior(
            key,
            latent_dim=int(latent_dim),
            n_layers=int(prior_cfg.get("n_layers", 8)),
            hidden_size=int(prior_cfg.get("hidden_size", 128)),
            scale_clamp=float(prior_cfg.get("scale_clamp", 0.05)),
            shift_clamp=float(prior_cfg.get("shift_clamp", 5.0)),
            init=str(prior_cfg.get("init", "default")),
            init_scale=float(prior_cfg.get("init_scale", 1.0)),
        )
    if source in {
        "joint_rq_spline",
        "rq_spline",
        "rq_spline_coupling",
        "neural_spline",
        "neural_spline_flow",
    }:
        return RQSplineCouplingPrior(
            key,
            latent_dim=int(latent_dim),
            n_layers=int(prior_cfg.get("n_layers", 12)),
            hidden_size=int(prior_cfg.get("hidden_size", 256)),
            n_bins=int(prior_cfg.get("n_bins", 16)),
            tail_bound=float(prior_cfg.get("tail_bound", 12.0)),
            min_bin_width=float(prior_cfg.get("min_bin_width", 1.0e-3)),
            min_bin_height=float(prior_cfg.get("min_bin_height", 1.0e-3)),
            min_derivative=float(prior_cfg.get("min_derivative", 1.0e-3)),
            permutation=str(prior_cfg.get("permutation", "roll")),
            init=str(prior_cfg.get("init", "default")),
            init_scale=float(prior_cfg.get("init_scale", 1.0)),
        )
    if source == "supervised_checkpoint":
        checkpoint = prior_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "amortized.prior.source='supervised_checkpoint' requires "
                "amortized.prior.checkpoint"
            )
        from euclid_dsps.prior_learning.train import load_prior_checkpoint

        prior, _sidecar, prior_spec, _schema = load_prior_checkpoint(checkpoint)
        active_spec = active_spec or latent_spec_from_config(config)
        _validate_loaded_prior_spec(active_spec, prior_spec)
        return prior
    if source == "spline15d_checkpoint":
        checkpoint = prior_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError(
                "amortized.prior.source='spline15d_checkpoint' requires "
                "amortized.prior.checkpoint"
            )
        prior, prior_spec = _load_spline15d_prior(checkpoint, active_spec)
        _validate_loaded_prior_spec(active_spec or prior_spec, prior_spec)
        return prior
    raise ValueError(
        "amortized.prior.source must be one of "
        "'standard_normal', 'supervised_checkpoint', 'spline15d_checkpoint', "
        "'joint_realnvp', or 'rq_spline_coupling'"
    )


def _latent_spec_for_amortized_config(config: dict[str, Any]) -> LatentSpec:
    active_spec = latent_spec_from_config(config)
    cfg = amortized_config(config)
    prior_cfg = cfg["prior"]
    source = str(prior_cfg.get("source", "joint_realnvp"))
    if source not in {"supervised_checkpoint", "spline15d_checkpoint"}:
        return active_spec
    checkpoint = prior_cfg.get("checkpoint")
    if not checkpoint:
        return active_spec
    if source == "spline15d_checkpoint":
        _prior, prior_spec = _load_spline15d_prior(checkpoint, active_spec)
        _validate_loaded_prior_spec(
            active_spec, prior_spec, require_normalization=False
        )
        return prior_spec

    from euclid_dsps.prior_learning.train import load_prior_checkpoint

    _prior, _sidecar, prior_spec, _schema = load_prior_checkpoint(checkpoint)
    _validate_loaded_prior_spec(active_spec, prior_spec, require_normalization=False)
    return prior_spec


def _load_spline15d_prior(
    checkpoint: str | Path,
    active_spec: LatentSpec | None,
) -> tuple[RealNVPPrior | RQSplineCouplingPrior, LatentSpec]:
    """Load the spline-15D flow and expose its exact JAX marginal transform."""
    if active_spec is None:
        raise ValueError("spline15d_checkpoint requires an active latent spec")
    from euclid_dsps.prior_learning.spline15d import SPLINE15D_PARAMETER_NAMES
    from euclid_dsps.prior_learning.spline15d_checkpoint import (
        load_spline15d_flow_checkpoint,
    )

    prior, sidecar = load_spline15d_flow_checkpoint(checkpoint)
    if active_spec.names != SPLINE15D_PARAMETER_NAMES:
        raise ValueError(
            "Spline-15D active parameter order does not match the prior contract"
        )
    normalization = dict(sidecar.get("normalization", {}) or {})
    transforms = dict(normalization.get("transforms", {}) or {})
    if normalization.get("whitening") is not None:
        raise ValueError("Amortized spline-15D currently requires no whitening")
    if float(normalization.get("normalized_atom_half_width", 0.0)) != 0.0:
        raise ValueError(
            "Amortized spline-15D does not support runtime normalized atom jitter"
        )
    family = []
    center = []
    scale = []
    location = []
    lam = []
    for name in SPLINE15D_PARAMETER_NAMES:
        transform = dict(transforms.get(name, {}) or {})
        transform_family = str(transform.get("family", ""))
        if transform_family == "log":
            family.append(1)
            location.append(0.0)
            lam.append(1.0)
        elif transform_family == "shifted_asinh":
            family.append(0)
            location.append(float(transform["location"]))
            lam.append(float(transform["lambda"]))
        else:
            raise ValueError(
                f"Unsupported spline-15D transform for {name}: {transform_family}"
            )
        center.append(float(transform["center"]))
        scale.append(float(transform["scale"]))
    spec = LatentSpec(
        names=SPLINE15D_PARAMETER_NAMES,
        lower=active_spec.lower,
        upper=active_spec.upper,
        raw_center=jnp.asarray(center, dtype=jnp.float32),
        raw_scale=jnp.asarray(scale, dtype=jnp.float32),
        normalization="spline15d_mixed",
        transform_family=jnp.asarray(family, dtype=jnp.int32),
        transform_location=jnp.asarray(location, dtype=jnp.float32),
        transform_lambda=jnp.asarray(lam, dtype=jnp.float32),
    )
    return prior, spec


def _validate_loaded_prior_spec(
    active: LatentSpec,
    loaded: LatentSpec,
    *,
    require_normalization: bool = True,
) -> None:
    if active.names != loaded.names:
        raise ValueError(
            "Supervised prior latent names do not match amortized config: "
            f"checkpoint={loaded.names}, config={active.names}"
        )
    if not np.allclose(
        np.asarray(active.lower),
        np.asarray(loaded.lower),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("Supervised prior lower bounds do not match amortized config")
    if not np.allclose(
        np.asarray(active.upper),
        np.asarray(loaded.upper),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError("Supervised prior upper bounds do not match amortized config")
    if not require_normalization:
        return
    if active.normalization != loaded.normalization:
        raise ValueError(
            "Supervised prior latent normalization does not match amortized config: "
            f"checkpoint={loaded.normalization}, config={active.normalization}"
        )
    for field in ("transform_family", "transform_location", "transform_lambda"):
        active_value = getattr(active, field, None)
        loaded_value = getattr(loaded, field, None)
        if (active_value is None) != (loaded_value is None):
            raise ValueError(f"Supervised prior latent {field} does not match")
        if active_value is not None and not np.allclose(
            np.asarray(active_value), np.asarray(loaded_value), rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(f"Supervised prior latent {field} does not match")
    if not np.allclose(
        np.asarray(active.raw_center),
        np.asarray(loaded.raw_center),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Supervised prior latent raw centers do not match amortized config"
        )
    if not np.allclose(
        np.asarray(active.raw_scale),
        np.asarray(loaded.raw_scale),
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "Supervised prior latent raw scales do not match amortized config"
        )


def _initialize_encoder_mean_if_possible(
    config: dict[str, Any],
    encoder,
    *,
    latent_spec: LatentSpec | None = None,
) -> object:
    """Start the encoder from the configured physical initialization."""
    try:
        spec = latent_spec or latent_spec_from_config(config)
    except (KeyError, ValueError):
        return encoder
    theta0 = jnp.asarray(
        initial_theta_from_config(
            config,
            spec.names,
            np.asarray(spec.lower, dtype=float),
            np.asarray(spec.upper, dtype=float),
        ),
        dtype=jnp.float32,
    )
    base = encoder if isinstance(encoder, GaussianEncoder) else encoder.base
    target = (
        theta_to_x(theta0, spec)
        if isinstance(encoder, GaussianEncoder)
        else jnp.zeros_like(theta0)
    )
    zero_mean_weight = jnp.zeros_like(base.mean_head.weight)
    zero_log_std_weight = jnp.zeros_like(base.log_std_head.weight)
    zero_log_std_bias = jnp.zeros_like(base.log_std_head.bias)
    updated = eqx.tree_at(
        lambda enc: (
            enc.mean_head.weight,
            enc.mean_head.bias,
            enc.log_std_head.weight,
            enc.log_std_head.bias,
        ),
        base,
        (
            zero_mean_weight,
            target,
            zero_log_std_weight,
            zero_log_std_bias,
        ),
    )
    if isinstance(encoder, GaussianEncoder):
        return updated
    return eqx.tree_at(lambda enc: enc.base, encoder, updated)


def _initial_theta_diagnostics_payload(
    config: dict[str, Any],
    latent_spec: LatentSpec,
) -> dict[str, Any]:
    lower = np.asarray(latent_spec.lower, dtype=float)
    upper = np.asarray(latent_spec.upper, dtype=float)
    theta0 = initial_theta_from_config(config, latent_spec.names, lower, upper)
    span = np.maximum(upper - lower, 1.0e-12)
    unit_position = (theta0 - lower) / span
    distance_lower = theta0 - lower
    distance_upper = upper - theta0
    nearest_boundary_fraction = np.minimum(unit_position, 1.0 - unit_position)
    threshold = float(
        ((config.get("amortized", {}) or {}).get("encoder", {}) or {}).get(
            "initial_boundary_warning_fraction", 1.0e-3
        )
    )
    x0 = np.asarray(
        jax.device_get(theta_to_x(jnp.asarray(theta0, dtype=jnp.float32), latent_spec)),
        dtype=float,
    )
    free = (config.get("fit", {}) or {}).get("free_parameters", {}) or {}
    rows = []
    warnings = []
    for index, name in enumerate(latent_spec.names):
        raw_initial = (free.get(name, {}) or {}).get("initial")
        configured_initial = _finite_float_or_none(raw_initial)
        source = "config_initial" if configured_initial is not None else "midpoint"
        if configured_initial is not None and not np.isclose(
            configured_initial,
            theta0[index],
            rtol=0.0,
            atol=1.0e-12,
        ):
            source = "config_initial_clipped_to_bounds"
        near_boundary = bool(nearest_boundary_fraction[index] < threshold)
        if near_boundary:
            warnings.append(
                {
                    "parameter": str(name),
                    "theta_init": float(theta0[index]),
                    "nearest_boundary_fraction": float(
                        nearest_boundary_fraction[index]
                    ),
                }
            )
        rows.append(
            {
                "name": str(name),
                "source": source,
                "configured_initial": (
                    float(configured_initial)
                    if configured_initial is not None
                    else None
                ),
                "theta_init": float(theta0[index]),
                "network_x_init": float(x0[index]),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "unit_position": float(unit_position[index]),
                "distance_to_lower": float(distance_lower[index]),
                "distance_to_upper": float(distance_upper[index]),
                "nearest_boundary_fraction": float(nearest_boundary_fraction[index]),
                "near_boundary": near_boundary,
            }
        )
    return {
        "initialization_contract": (
            "fit.free_parameters.<name>.initial with physical midpoint fallback"
        ),
        "boundary_warning_fraction": threshold,
        "n_parameters": len(rows),
        "n_near_boundary": len(warnings),
        "parameters": rows,
        "warnings": warnings,
    }


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def make_optimizer(config: dict[str, Any]):
    """Build the joint encoder/prior optimizer."""
    training = amortized_config(config)["training"]
    transforms = []
    clip = float(training.get("gradient_clip_norm", 1.0))
    if clip > 0.0:
        transforms.append(optax.clip_by_global_norm(clip))
    transforms.append(
        optax.adamw(
            learning_rate=float(training.get("learning_rate", 1.0e-4)),
            weight_decay=float(training.get("weight_decay", 1.0e-5)),
        )
    )
    return optax.chain(*transforms)


def train_amortized_fs2(
    config: dict[str, Any],
    out_dir: Path,
    limit: int | None,
    batch_size: int,
    epochs: int,
    n_samples: int,
    seed: int,
    verbose: bool = True,
    progress: bool = True,
    dataset_label: str = "FS2",
    row_indices_file: str | Path | None = None,
    train_indices_file: str | Path | None = None,
    validation_indices_file: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
    start_epoch: int = 1,
) -> None:
    """Train encoder and RealNVP prior jointly on configured photometry."""
    out = ensure_dir(out_dir)
    cfg = amortized_config(config)
    objective_mode_name = objective_mode(cfg.get("objective", {}))
    redshift_bins_for_fingerprint = (
        (config.get("amortized", {}) or {}).get("data", {}) or {}
    ).get(
        "redshift_bins",
        None,
    )
    catalog_identity = write_catalog_fingerprint(
        out,
        config,
        redshift_bins=redshift_bins_for_fingerprint,
    )
    _log(verbose, f"[amortized] {dataset_label} joint encoder/RealNVP training")
    _log(verbose, f"[amortized] output directory: {out}")
    _log(
        verbose,
        "[amortized] run config: "
        f"limit={limit if limit is not None else 'all'} "
        f"batch_size={int(batch_size)} epochs={int(epochs)} "
        f"n_samples={int(n_samples)} seed={int(seed)} "
        f"objective={objective_mode_name}",
    )
    _log(
        verbose,
        f"[amortized] JAX backend: {jax.default_backend()} devices={jax.devices()}",
    )
    write_json(out / "normalized_config.json", config)
    split = build_training_split(
        config,
        limit=limit,
        seed=seed,
        row_indices_file=row_indices_file,
        train_indices_file=train_indices_file,
        validation_indices_file=validation_indices_file,
    )
    write_training_split_artifacts(out, split)
    _log(
        verbose,
        "[amortized] data split: "
        f"mode={split.selection_mode} strategy={split.stratified_strategy} "
        f"train={len(split.train_indices)} validation={len(split.validation_indices)} "
        f"z_column={split.redshift_column}",
    )
    catalog_batch_size = int(
        cfg["data"].get("catalog_batch_size", max(int(batch_size), 10_000))
    )
    jax_batch_size = _effective_jax_batch_size(cfg["training"], int(batch_size))
    if jax_batch_size != int(batch_size):
        _log(
            verbose,
            "[amortized] capping JAX/DSPS batch size: "
            f"requested_batch_size={int(batch_size)} jax_batch_size={jax_batch_size}",
        )
    data_parallel = _resolve_data_parallel_training(
        cfg["training"],
        jax_batch_size=int(jax_batch_size),
    )
    _log(
        verbose,
        "[amortized] data parallel: "
        f"requested={data_parallel['requested']} "
        f"effective={data_parallel['effective']} "
        f"local_devices={data_parallel['n_devices']} "
        f"global_jax_batch={data_parallel['global_batch_size']} "
        f"per_device_batch={data_parallel['per_device_batch_size']}",
    )
    _log(verbose, "[amortized] loading selected train photometry arrays...")
    train_arrays = load_photometry_arrays_from_config(
        config,
        batch_size=catalog_batch_size,
        row_indices=split.train_indices,
    )
    validation_arrays = None
    if len(split.validation_indices):
        _log(verbose, "[amortized] loading selected validation photometry arrays...")
        validation_arrays = load_photometry_arrays_from_config(
            config,
            batch_size=catalog_batch_size,
            row_indices=split.validation_indices,
        )
    _log(verbose, "[amortized] computing feature stats from train flux/errors...")
    feature_stats = compute_feature_stats(
        train_arrays.flux,
        train_arrays.flux_err,
        train_arrays.mask,
        band_names=train_arrays.band_names,
        flux_transform=str(cfg["features"].get("flux_transform", "asinh")),
    )
    write_feature_stats(out / "feature_stats.json", feature_stats)
    _log(
        verbose,
        "[amortized] feature stats ready: "
        f"{len(feature_stats.band_names)} bands, "
        f"feature_dim={2 * len(feature_stats.band_names)} "
        f"flux_transform={feature_stats.flux_transform}",
    )

    _log(verbose, "[amortized] loading configured filters...")
    filters = load_filters(config["bands"])
    _log(verbose, f"[amortized] loading DSPS context from {config['ssp_path']}...")
    context = load_context(
        config["ssp_path"],
        filters,
        n_sfh_bins=int(config["model"].get("n_sfh_bins", 96)),
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config.get("model"),
    )
    model_args = dynamic_model_args(context)
    jit_context = _StaticArg(context)
    _log(
        verbose,
        "[amortized] DSPS context ready: "
        f"{len(filters)} filters, n_sfh_bins={context.n_sfh_bins}, "
        f"model={context.model_config}",
    )
    latent_spec = _latent_spec_for_amortized_config(config)
    jit_latent_spec = JitLatentSpec(
        names=latent_spec.names,
        lower=latent_spec.lower,
        upper=latent_spec.upper,
        raw_center=latent_spec.raw_center,
        raw_scale=latent_spec.raw_scale,
        normalization=latent_spec.normalization,
        transform_family=latent_spec.transform_family,
        transform_location=latent_spec.transform_location,
        transform_lambda=latent_spec.transform_lambda,
    )
    _log(
        verbose,
        "[amortized] latent spec ready: "
        f"{len(latent_spec.names)} parameters, "
        f"first={latent_spec.names[0]}, last={latent_spec.names[-1]}",
    )
    initial_theta_diagnostics = _initial_theta_diagnostics_payload(config, latent_spec)
    write_json(out / "initial_theta_diagnostics.json", initial_theta_diagnostics)
    geometry_cfg = dict((config.get("amortized", {}) or {}).get("latent", {}) or {})
    latent_geometry = write_latent_prior_geometry(
        config,
        out,
        n_samples=int(geometry_cfg.get("geometry_samples", 200_000)),
        seed=int(geometry_cfg.get("geometry_seed", seed)),
    )
    if initial_theta_diagnostics["warnings"]:
        _log(
            verbose,
            "[amortized] initial theta boundary warnings: "
            f"{len(initial_theta_diagnostics['warnings'])} parameters within "
            f"{initial_theta_diagnostics['boundary_warning_fraction']:.3g} "
            "of a bound",
        )
    if latent_geometry.get("parameters_near_bounds_5pct"):
        _log(
            verbose,
            "[amortized] latent prior geometry warning: "
            "x~N(0,1) puts >5% mass near bounds for "
            + ", ".join(latent_geometry["parameters_near_bounds_5pct"]),
        )
    key = jax.random.PRNGKey(int(seed))
    key, model_key = jax.random.split(key)
    model = build_amortized_model(config, model_key, latent_spec=latent_spec)
    start_epoch = int(start_epoch)
    if start_epoch < 1 or start_epoch > int(epochs):
        raise ValueError(
            f"start_epoch must satisfy 1 <= start_epoch <= epochs; "
            f"got {start_epoch} and {epochs}"
        )
    if initial_checkpoint is not None:
        initial_checkpoint = Path(initial_checkpoint)
        if not initial_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing amortized initial checkpoint: {initial_checkpoint}"
            )
        model = eqx.tree_deserialise_leaves(initial_checkpoint, model)
        _log(
            verbose,
            "[amortized] warm restart: "
            f"checkpoint={initial_checkpoint} start_epoch={start_epoch} "
            "optimizer_state=reinitialized",
        )
    elif start_epoch != 1:
        raise ValueError("--start-epoch requires --initial-checkpoint")
    train_prior = _train_prior_jointly(cfg["prior"])
    prior_update_schedule = _prior_update_schedule(cfg["prior"])
    calibration_runtime_config = {"calibration": config.get("calibration", {}) or {}}
    sed_scale_cfg = global_sed_scale_config(calibration_runtime_config)
    train_alpha = bool(sed_scale_cfg.enabled and sed_scale_cfg.trainable)
    band_calibration_cfg = per_band_flux_calibration_config(calibration_runtime_config)
    train_band_calibration = bool(
        band_calibration_cfg.enabled
        and band_calibration_cfg.trainable
        and model.band_calibration is not None
    )
    _log(
        verbose,
        "[amortized] model built: "
        f"encoder_hidden={cfg['encoder'].get('hidden_sizes')} "
        f"prior_layers={cfg['prior'].get('n_layers')} "
        f"prior_hidden={cfg['prior'].get('hidden_size')} "
        f"prior_source={cfg['prior'].get('source')} "
        f"train_prior={train_prior} "
        f"prior_update_schedule={prior_update_schedule['mode']} "
        f"prior_freeze_epochs={prior_update_schedule['freeze_epochs']} "
        f"alpha_sed_enabled={sed_scale_cfg.enabled} "
        f"train_alpha={train_alpha} "
        f"per_band_calibration_enabled={band_calibration_cfg.enabled} "
        f"train_per_band_calibration={train_band_calibration}",
    )
    input_noise_cfg = _input_noise_config(cfg.get("input_noise", {}))
    if input_noise_cfg["enabled"]:
        _log(
            verbose,
            "[amortized] input noise enabled: "
            f"mode={input_noise_cfg['mode']} "
            f"sigma_scale={input_noise_cfg['sigma_scale']} "
            f"apply_to={input_noise_cfg['apply_to']}",
        )
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    pmap_train_step = None
    model_replicated = None
    opt_state_replicated = None
    if bool(data_parallel["enabled"]):
        pmap_train_step = _make_pmap_train_step(optimizer)
        model_replicated = _replicate_tree(model, data_parallel["devices"])
        opt_state_replicated = _replicate_tree(opt_state, data_parallel["devices"])

    ckpt_dir = ensure_dir(out / "checkpoints")
    initial_epoch = start_epoch - 1
    save_checkpoint(
        ckpt_dir / f"epoch_{initial_epoch:04d}.eqx",
        model,
        config=config,
        latent_spec=latent_spec,
        feature_stats=feature_stats,
        epoch=initial_epoch,
        metric=0.0,
        metric_name="initial",
    )
    snapshot_cfg = dict(cfg.get("training_snapshots", {}) or {})
    snapshot_arrays = (
        validation_arrays if validation_arrays is not None else train_arrays
    )
    if bool(snapshot_cfg.get("enabled", False)) and bool(
        snapshot_cfg.get("include_epoch_zero", True)
    ):
        key, snapshot_key = jax.random.split(key)
        _write_training_snapshot(
            out,
            epoch=initial_epoch,
            model=model,
            arrays=snapshot_arrays,
            feature_stats=feature_stats,
            latent_spec=latent_spec,
            key=snapshot_key,
            snapshot_config=snapshot_cfg,
            config=config,
            kl_weight=_kl_weight(
                initial_epoch,
                int(cfg["training"].get("kl_annealing_epochs", 5)),
                max_weight=float(cfg["training"].get("kl_weight_max", 1.0)),
            ),
            prior_source=str(cfg["prior"].get("source", "joint_realnvp")),
            prior_train_jointly=bool(train_prior),
            update_phase="initial",
            deterministic=is_deterministic_reconstruction(cfg.get("objective", {})),
        )
    rows = []
    configured_best_metric = str(
        cfg["training"].get("best_checkpoint_metric", "validation_negative_loglike")
    )
    best_checkpoint_metric = _effective_best_checkpoint_metric(
        configured_best_metric,
        has_validation=validation_arrays is not None,
    )
    best_checkpoint_min_epoch = _best_checkpoint_min_epoch(
        cfg["training"],
        has_validation=validation_arrays is not None,
    )
    best_metric_value = np.inf
    best_checkpoint_epoch: int | None = None
    start_time = time.time()
    checkpoint_every = int(cfg["output"].get("checkpoint_every", 1))
    diagnostics_every = int(cfg["output"].get("diagnostics_every", 1))
    save_training_curves = bool(cfg["output"].get("save_training_curves", True))
    validation_every = int(cfg["training"].get("validation_every", 1))
    epoch_shuffle = bool(cfg["data"].get("epoch_shuffle", True))
    kl_weight_max = float(cfg["training"].get("kl_weight_max", 1.0))
    _log_prior_safety_messages(
        verbose,
        cfg["prior"],
        train_prior=train_prior,
        kl_weight_max=kl_weight_max,
    )
    expected_train_rows = int(len(train_arrays.object_id))
    if bool(data_parallel["enabled"]) and expected_train_rows > 0:
        remainder = expected_train_rows % int(jax_batch_size)
        if remainder:
            expected_train_rows += int(jax_batch_size) - int(remainder)
    expected_batches = _expected_batch_count(expected_train_rows, int(jax_batch_size))
    val_expected_batches = (
        None
        if validation_arrays is None
        else _expected_batch_count(
            len(validation_arrays.object_id), int(jax_batch_size)
        )
    )
    validation_bin_rows: list[dict[str, float | int | str]] = []
    train_rng = np.random.default_rng(int(seed) + 10_000)
    _log(
        verbose,
        "[amortized] training start: "
        "first batch includes JAX/DSPS compilation and can be noticeably slower.",
    )
    for epoch in range(start_epoch, int(epochs) + 1):
        kl_weight = _kl_weight(
            epoch,
            int(cfg["training"].get("kl_annealing_epochs", 5)),
            max_weight=kl_weight_max,
        )
        objective_config = _objective_config_for_epoch(cfg, epoch)
        update_phase = _training_update_phase(
            cfg["prior"],
            epoch=epoch,
            train_prior=train_prior,
        )
        epoch_rows = []
        _log(
            verbose,
            "[amortized] epoch "
            f"{epoch}/{int(epochs)} start kl={kl_weight:.3f} "
            f"phase={update_phase} "
            "temp="
            f"{float(objective_config.get('likelihood_temperature', 1.0)):.3g}",
        )
        train_order = np.arange(len(train_arrays.object_id))
        if epoch_shuffle:
            train_rng.shuffle(train_order)
        train_order, data_parallel_padded_rows = _pad_epoch_order_for_data_parallel(
            train_order,
            global_batch_size=int(jax_batch_size),
            enabled=bool(data_parallel["enabled"]),
            rng=train_rng,
        )
        if data_parallel_padded_rows and verbose:
            _log(
                verbose,
                "[amortized] pmap epoch padding: "
                f"epoch={epoch} duplicated_rows={data_parallel_padded_rows} "
                f"global_batch={int(jax_batch_size)}",
            )
        batch_iter = iter_photometry_batches_from_arrays(
            train_arrays,
            batch_size=int(jax_batch_size),
            feature_stats=feature_stats,
            order=train_order,
            truth_names=(
                latent_spec.names
                if objective_mode_name == "neural_posterior_estimation"
                else None
            ),
        )
        with _progress_bar(
            enabled=bool(progress),
            total=expected_batches,
            desc=f"epoch {epoch}/{int(epochs)}",
            unit="batch",
        ) as pbar:
            for batch_index, batch in enumerate(batch_iter):
                key, step_key, noise_key = jax.random.split(key, 3)
                loss_batch = _loss_batch_with_input_noise(
                    batch,
                    feature_stats,
                    noise_key,
                    input_noise_cfg,
                )
                if bool(data_parallel["enabled"]):
                    if (
                        pmap_train_step is None
                        or model_replicated is None
                        or opt_state_replicated is None
                    ):
                        raise RuntimeError("pmap training state was not initialized")
                    sharded_batch = _shard_loss_batch(
                        loss_batch,
                        int(data_parallel["n_devices"]),
                    )
                    step_keys = jax.random.split(
                        step_key,
                        int(data_parallel["n_devices"]),
                    )
                    (
                        model_replicated,
                        opt_state_replicated,
                        _loss,
                        metrics,
                        grad_norms,
                        loss_finite_value,
                        grads_finite_value,
                        update_applied_value,
                    ) = pmap_train_step(
                        model_replicated,
                        opt_state_replicated,
                        sharded_batch,
                        jit_latent_spec,
                        jit_context,
                        model_args,
                        latent_spec.names,
                        step_keys,
                        int(n_samples),
                        float(kl_weight),
                        cfg["likelihood"],
                        calibration_runtime_config,
                        objective_config,
                        update_phase,
                        bool(train_alpha),
                        bool(train_band_calibration),
                    )
                    record = _metrics_record(_unreplicate_metrics(metrics))
                    record.update(
                        {
                            name: _unreplicate_scalar(value)
                            for name, value in grad_norms.items()
                        }
                    )
                    loss_finite = _unreplicate_bool(loss_finite_value)
                    grads_finite = _unreplicate_bool(grads_finite_value)
                    update_applied = _unreplicate_bool(update_applied_value)
                else:
                    (loss, metrics), grads = _loss_and_grads_jit(
                        model,
                        loss_batch,
                        jit_latent_spec,
                        jit_context,
                        model_args,
                        latent_spec.names,
                        step_key,
                        int(n_samples),
                        float(kl_weight),
                        cfg["likelihood"],
                        calibration_runtime_config,
                        objective_config,
                    )
                    grads = _apply_training_grad_masks(
                        grads,
                        update_phase=update_phase,
                        train_alpha=bool(train_alpha),
                        train_band_calibration=bool(train_band_calibration),
                    )
                    record = _metrics_record(metrics)
                    record.update(component_grad_norms(grads))
                    loss_finite = bool(
                        np.isfinite(float(np.asarray(jax.device_get(loss))))
                    )
                    grads_finite = tree_all_finite(grads)
                    update_applied = bool(loss_finite and grads_finite)
                    if update_applied:
                        updates, opt_state = optimizer.update(
                            grads,
                            opt_state,
                            eqx.filter(model, eqx.is_inexact_array),
                        )
                        model = eqx.apply_updates(model, updates)
                if not update_applied and verbose:
                    _log(
                        verbose,
                        "[amortized] skipped non-finite update: "
                        f"epoch={epoch} batch={batch_index} "
                        f"loss_finite={loss_finite} grads_finite={grads_finite} "
                        f"finite_fraction={record.get('finite_fraction', float('nan')):.3g}",
                    )
                record.update(
                    {
                        "split": "train",
                        "epoch": int(epoch),
                        "batch": int(batch_index),
                        "kl_definition": "E_q[logq - logp_beta]",
                        "kl_direction": "q_to_p",
                        "kl_weight": float(kl_weight),
                        "prior_source": str(
                            cfg["prior"].get("source", "joint_realnvp")
                        ),
                        "prior_train_jointly": float(train_prior),
                        "prior_update_schedule": str(
                            prior_update_schedule.get("mode", "joint")
                        ),
                        "prior_freeze_epochs": int(
                            prior_update_schedule.get("freeze_epochs", 0)
                        ),
                        "likelihood_temperature": float(
                            objective_config.get("likelihood_temperature", 1.0)
                        ),
                        "update_phase": update_phase,
                        "n_objects": int(batch.flux.shape[0]),
                        "data_parallel_mode": str(data_parallel["effective"]),
                        "data_parallel_devices": int(data_parallel["n_devices"]),
                        "data_parallel_per_device_batch_size": int(
                            data_parallel["per_device_batch_size"]
                        ),
                        "data_parallel_epoch_padded_rows": int(
                            data_parallel_padded_rows
                        ),
                        "loss_finite": float(loss_finite),
                        "grads_finite": float(grads_finite),
                        "update_applied": float(update_applied),
                    }
                )
                epoch_rows.append(record)
                rows.append(record)
                if validation_arrays is None and loss_finite:
                    checkpoint_metric_value = _checkpoint_metric_from_rows(
                        [record],
                        best_checkpoint_metric,
                    )
                    if _should_update_best_checkpoint(
                        epoch=epoch,
                        value=checkpoint_metric_value,
                        best_value=best_metric_value,
                        min_epoch=best_checkpoint_min_epoch,
                    ):
                        best_metric_value = float(checkpoint_metric_value)
                        best_checkpoint_epoch = int(epoch)
                        if bool(data_parallel["enabled"]):
                            model = _unreplicate_tree(model_replicated)
                        save_checkpoint(
                            ckpt_dir / "best.eqx",
                            model,
                            config=config,
                            latent_spec=latent_spec,
                            feature_stats=feature_stats,
                            epoch=epoch,
                            metric=best_metric_value,
                            metric_name=best_checkpoint_metric,
                        )
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{record['loss']:.3g}",
                        "nll": f"{record['negative_loglike']:.3g}",
                        "kl": f"{record['kl_mc_mean']:.3g}",
                        "enc_g": f"{record['encoder_grad_norm']:.2g}",
                        "prior_g": f"{record['prior_grad_norm']:.2g}",
                        "upd": int(update_applied),
                    },
                    refresh=False,
                )
                if verbose and not progress:
                    _log(verbose, _batch_progress_line(record))
        if bool(data_parallel["enabled"]):
            model = _unreplicate_tree(model_replicated)
        if not epoch_rows:
            raise ValueError("No FS2 batches were produced for amortized training")
        validation_rows = []
        epoch_bin_rows = []
        if (
            validation_arrays is not None
            and validation_every > 0
            and epoch % validation_every == 0
        ):
            _log(
                verbose,
                f"[amortized] epoch {epoch}/{int(epochs)} validation start",
            )
            key, val_key = jax.random.split(key)
            validation_rows, epoch_bin_rows = evaluate_validation_epoch(
                model,
                validation_arrays,
                split,
                feature_stats,
                latent_spec,
                context,
                model_args,
                val_key,
                batch_size=int(jax_batch_size),
                n_samples=int(n_samples),
                kl_weight=float(kl_weight),
                likelihood_config=cfg["likelihood"],
                calibration_config=calibration_runtime_config,
                objective_config=objective_config,
                progress=bool(progress),
                total=val_expected_batches,
                desc=f"val {epoch}/{int(epochs)}",
                epoch=epoch,
            )
            rows.extend(validation_rows)
            validation_bin_rows.extend(epoch_bin_rows)
            val_loss = _finite_mean([row["loss"] for row in validation_rows])
            val_nll = _finite_mean([row["negative_loglike"] for row in validation_rows])
            checkpoint_metric_value = _checkpoint_metric_from_rows(
                validation_rows,
                best_checkpoint_metric,
            )
            if _should_update_best_checkpoint(
                epoch=epoch,
                value=checkpoint_metric_value,
                best_value=best_metric_value,
                min_epoch=best_checkpoint_min_epoch,
            ):
                best_metric_value = float(checkpoint_metric_value)
                best_checkpoint_epoch = int(epoch)
                save_checkpoint(
                    ckpt_dir / "best.eqx",
                    model,
                    config=config,
                    latent_spec=latent_spec,
                    feature_stats=feature_stats,
                    epoch=epoch,
                    metric=best_metric_value,
                    metric_name=best_checkpoint_metric,
                )
            _log(
                verbose,
                "[amortized] epoch "
                f"{epoch}/{int(epochs)} validation done: "
                f"mean_loss={val_loss:.6g} mean_nll={val_nll:.6g} "
                f"{best_checkpoint_metric}={checkpoint_metric_value:.6g} "
                f"binned_rows={len(epoch_bin_rows)}",
            )
        save_checkpoint(
            ckpt_dir / "last.eqx",
            model,
            config=config,
            latent_spec=latent_spec,
            feature_stats=feature_stats,
            epoch=epoch,
            metric=_finite_mean([row["loss"] for row in epoch_rows]),
            metric_name="train_loss",
        )
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.eqx",
                model,
                config=config,
                latent_spec=latent_spec,
                feature_stats=feature_stats,
                epoch=epoch,
                metric=_finite_mean([row["loss"] for row in epoch_rows]),
                metric_name="train_loss",
            )
        if _should_write_training_snapshot(snapshot_cfg, epoch):
            key, snapshot_key = jax.random.split(key)
            _write_training_snapshot(
                out,
                epoch=epoch,
                model=model,
                arrays=snapshot_arrays,
                feature_stats=feature_stats,
                latent_spec=latent_spec,
                key=snapshot_key,
                snapshot_config=snapshot_cfg,
                config=config,
                kl_weight=float(kl_weight),
                prior_source=str(cfg["prior"].get("source", "joint_realnvp")),
                prior_train_jointly=bool(train_prior),
                update_phase=update_phase,
                deterministic=is_deterministic_reconstruction(cfg.get("objective", {})),
            )
        pd.DataFrame(rows).to_csv(out / "training_log.csv", index=False)
        if validation_bin_rows:
            pd.DataFrame(validation_bin_rows).to_csv(
                out / "validation_redshift_bin_metrics.csv",
                index=False,
            )
        write_training_progress(
            out,
            rows=rows,
            epoch=epoch,
            best_loss=best_metric_value,
            start_time=start_time,
            checkpoint_last="checkpoints/last.eqx",
            best_checkpoint_metric=best_checkpoint_metric,
            best_checkpoint_epoch=best_checkpoint_epoch,
        )
        if (
            save_training_curves
            and diagnostics_every > 0
            and epoch % diagnostics_every == 0
        ):
            write_training_diagnostics(out / "training_log.csv", out)
        prior_diag_cfg = dict(cfg.get("prior_predictive_diagnostics", {}) or {})
        if bool(prior_diag_cfg.get("enabled", False)) and (
            diagnostics_every <= 0 or epoch % diagnostics_every == 0
        ):
            key, prior_diag_key = jax.random.split(key)
            _write_prior_predictive_training_diagnostics(
                out,
                model,
                validation_arrays if validation_arrays is not None else train_arrays,
                latent_spec,
                context,
                model_args,
                latent_spec.names,
                prior_diag_key,
                config=config,
                epoch=epoch,
                band_names=train_arrays.band_names,
                calibration_config=calibration_runtime_config,
                n_prior_samples=int(prior_diag_cfg.get("n_prior_samples", 512)),
                n_observed=int(prior_diag_cfg.get("n_observed", 5000)),
                batch_size=int(prior_diag_cfg.get("batch_size", 256)),
            )
        epoch_loss = _finite_mean([row["loss"] for row in epoch_rows])
        epoch_nll = _finite_mean([row["negative_loglike"] for row in epoch_rows])
        epoch_kl = _finite_mean([row["kl_mc_mean"] for row in epoch_rows])
        epoch_updates = int(sum(row.get("update_applied", 0.0) for row in epoch_rows))
        _log(
            verbose,
            "[amortized] epoch "
            f"{epoch}/{int(epochs)} done: mean_loss={epoch_loss:.6g} "
            f"mean_nll={epoch_nll:.6g} mean_kl={epoch_kl:.6g} "
            f"updates={epoch_updates}/{len(epoch_rows)} "
            f"best_{best_checkpoint_metric}={best_metric_value:.6g} "
            f"last_checkpoint={ckpt_dir / 'last.eqx'}",
        )

    if not (ckpt_dir / "best.eqx").exists():
        fallback_metric = _finite_mean(
            [row["loss"] for row in rows if row.get("split") == "train"]
        )
        best_metric_value = fallback_metric
        best_checkpoint_epoch = int(epochs)
        save_checkpoint(
            ckpt_dir / "best.eqx",
            model,
            config=config,
            latent_spec=latent_spec,
            feature_stats=feature_stats,
            epoch=int(epochs),
            metric=fallback_metric,
            metric_name=f"{best_checkpoint_metric}_fallback_train_loss",
        )

    summary = {
        "epochs": int(epochs),
        "start_epoch": int(start_epoch),
        "initial_checkpoint": (
            str(initial_checkpoint) if initial_checkpoint is not None else None
        ),
        "optimizer_state_resumed": False,
        "batch_size": int(batch_size),
        "jax_batch_size": int(jax_batch_size),
        "data_parallel": {
            "requested": str(data_parallel["requested"]),
            "effective": str(data_parallel["effective"]),
            "enabled": bool(data_parallel["enabled"]),
            "local_device_count": int(data_parallel["n_devices"]),
            "global_batch_size": int(data_parallel["global_batch_size"]),
            "per_device_batch_size": int(data_parallel["per_device_batch_size"]),
        },
        "n_samples": int(n_samples),
        "limit": limit,
        "train_rows": int(len(split.train_indices)),
        "validation_rows": int(len(split.validation_indices)),
        "validation_fraction": float(split.validation_fraction),
        "selection_mode": split.selection_mode,
        "stratified_strategy": split.stratified_strategy,
        "redshift_column": split.redshift_column,
        "catalog_fingerprint": catalog_identity,
        "kl_definition": "E_q[logq - logp_beta]",
        "kl_direction": "q_to_p",
        "kl_annealing_epochs": int(cfg["training"].get("kl_annealing_epochs", 5)),
        "kl_weight_max": float(kl_weight_max),
        "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
        "prior_train_jointly": bool(train_prior),
        "prior_update_schedule": str(prior_update_schedule.get("mode", "joint")),
        "prior_freeze_epochs": int(prior_update_schedule.get("freeze_epochs", 0)),
        "objective_mode": objective_mode_name,
        "best_loss": float(best_metric_value),
        "best_checkpoint_metric": best_checkpoint_metric,
        "best_checkpoint_metric_requested": configured_best_metric,
        "best_checkpoint_min_epoch": int(best_checkpoint_min_epoch),
        "best_checkpoint_epoch": (
            int(best_checkpoint_epoch) if best_checkpoint_epoch is not None else None
        ),
        "elapsed_time_s": float(time.time() - start_time),
        "checkpoint_best": "checkpoints/best.eqx",
        "checkpoint_last": "checkpoints/last.eqx",
        "checkpoint_initial": f"checkpoints/epoch_{initial_epoch:04d}.eqx",
        "checkpoint_every": checkpoint_every,
        "diagnostics_every": diagnostics_every,
        "input_noise": input_noise_cfg,
        "initial_theta_diagnostics": {
            "path": "initial_theta_diagnostics.json",
            "n_near_boundary": int(initial_theta_diagnostics["n_near_boundary"]),
            "boundary_warning_fraction": float(
                initial_theta_diagnostics["boundary_warning_fraction"]
            ),
        },
        "latent_prior_geometry": {
            "path": "latent_prior_geometry.json",
            "csv": "latent_prior_geometry.csv",
            "plot": latent_geometry.get("plot"),
            "max_frac_within_either_5pct": float(
                latent_geometry.get("max_frac_within_either_5pct", np.nan)
            ),
            "parameters_near_bounds_5pct": list(
                latent_geometry.get("parameters_near_bounds_5pct", [])
            ),
        },
        "updates_applied": int(
            sum(row.get("update_applied", 0.0) for row in _training_rows(rows))
        ),
        "updates_skipped": int(
            sum(1.0 - row.get("update_applied", 0.0) for row in _training_rows(rows))
        ),
        "joint_training": {
            "encoder": True,
            "flow_prior": bool(train_prior),
            "realnvp_prior": bool(train_prior),
            "prior_source": str(cfg["prior"].get("source", "joint_realnvp")),
            "prior_update_schedule": _prior_update_schedule(cfg["prior"]),
            "global_sed_scale": bool(train_alpha),
            "per_band_flux_calibration": bool(train_band_calibration),
            "decoder": False,
            "encoder_objective": objective_mode_name,
            "kl_estimator": (
                "disabled_deterministic_reconstruction"
                if objective_mode_name == "deterministic_reconstruction"
                else "monte_carlo_logq_minus_logp"
            ),
        },
        "likelihood_temperature": {
            "initial": float(
                cfg["training"].get("likelihood_temperature_initial", 1.0)
            ),
            "final": float(cfg["training"].get("likelihood_temperature_final", 1.0)),
            "annealing_epochs": int(
                cfg["training"].get("likelihood_temperature_annealing_epochs", 0)
            ),
        },
        "posterior_regularization": dict(cfg.get("posterior_regularization", {}) or {}),
        "global_sed_scale": alpha_metadata(
            float(np.asarray(jax.device_get(model.sed_scale.log_alpha_sed))),
            sed_scale_cfg.prior_sigma_log_alpha,
        )
        | {
            "enabled": bool(sed_scale_cfg.enabled),
            "trainable": bool(train_alpha),
            "mode": sed_scale_cfg.mode,
        },
        "per_band_flux_calibration": _per_band_flux_calibration_summary(
            model,
            config,
            band_names=train_arrays.band_names,
            config_block=band_calibration_cfg,
            trainable=train_band_calibration,
        ),
    }
    write_json(out / "training_summary.json", summary)
    write_per_band_flux_calibration_artifacts(
        out,
        model,
        config,
        band_names=train_arrays.band_names,
        config_block=band_calibration_cfg,
        trainable=train_band_calibration,
    )
    if save_training_curves:
        write_training_diagnostics(out / "training_log.csv", out)
    try:
        gate = write_training_collapse_gate(out)
        summary["training_collapse_gate"] = {
            "path": str(out / "training_collapse_gate.json"),
            "status": gate.get("status"),
        }
        write_json(out / "training_summary.json", summary)
    except Exception as exc:
        summary["training_collapse_gate_warning"] = str(exc)
        write_json(out / "training_summary.json", summary)
    _log(verbose, "[amortized] training complete")
    _log(verbose, f"[amortized] summary: {out / 'training_summary.json'}")
    _log(verbose, f"[amortized] progress: {out / 'training_progress.json'}")
    _log(verbose, f"[amortized] best checkpoint: {out / 'checkpoints' / 'best.eqx'}")
    return


def save_checkpoint(
    path: str | Path,
    model: AmortizedModel,
    *,
    config: dict[str, Any],
    latent_spec: LatentSpec,
    feature_stats: FeatureStats,
    epoch: int,
    metric: float,
    metric_name: str | None = None,
) -> None:
    """Write an Equinox checkpoint plus a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    sidecar = {
        "epoch": int(epoch),
        "metric": float(metric),
        "metric_name": metric_name,
        "latent_spec": latent_spec_to_jsonable(latent_spec),
        "feature_stats_path": "../feature_stats.json",
        "amortized": amortized_config(config),
        "architecture": architecture_summary(config),
    }
    write_json(path.with_suffix(path.suffix + ".json"), sidecar)


def load_checkpoint(
    path: str | Path,
    config: dict[str, Any],
) -> AmortizedModel:
    """Load an amortized model checkpoint using the active config architecture."""
    path = Path(path)
    key = jax.random.PRNGKey(int(amortized_config(config)["training"].get("seed", 42)))
    template = build_amortized_model(config, key)
    try:
        model = eqx.tree_deserialise_leaves(path, template)
    except RuntimeError as exc:
        _raise_realnvp_mask_checkpoint_error(path, exc)
    if isinstance(model.prior, (RealNVPPrior, RQSplineCouplingPrior)):
        roundtrip_fail_atol = _checkpoint_prior_roundtrip_fail_atol(config)
        assert_flow_integrity(
            model.prior,
            context=f"amortized checkpoint load {path}",
            sample_count=64,
            roundtrip_fail_atol=roundtrip_fail_atol,
        )
    return model


def _checkpoint_prior_roundtrip_fail_atol(config: dict[str, Any]) -> float:
    """Use the serialized spline-flow tolerance when rebuilding its template."""
    prior_cfg = amortized_config(config)["prior"]
    if str(prior_cfg.get("source", "")) != "spline15d_checkpoint":
        return 1.0e-2
    checkpoint = prior_cfg.get("checkpoint")
    if not checkpoint:
        return 1.0e-2
    sidecar = Path(checkpoint).with_suffix(Path(checkpoint).suffix + ".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return float(payload.get("integrity_roundtrip_fail_atol", 1.0e-2))


def _raise_realnvp_mask_checkpoint_error(path: Path, exc: RuntimeError):
    message = str(exc)
    if "mask" in message and "changed dtype from bool" in message:
        raise RuntimeError(
            "This amortized checkpoint contains a RealNVP prior with float "
            "coupling masks from an older buggy training path where masks were "
            "trainable. It is not scientifically usable and must be retrained "
            f"with the fixed code: {path}"
        ) from exc
    raise exc


def _per_band_flux_calibration_summary(
    model: AmortizedModel,
    config: dict[str, Any],
    *,
    band_names: tuple[str, ...],
    config_block=None,
    trainable: bool | None = None,
) -> dict[str, Any]:
    """Return JSON-friendly per-band calibration metadata."""
    cfg = (
        config_block
        if config_block is not None
        else per_band_flux_calibration_config(config)
    )
    enabled = bool(cfg.enabled and model.band_calibration is not None)
    payload: dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "active": enabled,
        "trainable": bool(cfg.trainable if trainable is None else trainable),
        "mode": cfg.mode,
        "prior_sigma_log_alpha": float(cfg.prior_sigma_log_alpha),
        "prior_sigma_mag": float(cfg.prior_sigma_mag),
    }
    if not enabled:
        return payload | {
            "bands": {},
            "max_abs_delta_mag": 0.0,
            "mean_abs_delta_mag": 0.0,
            "total_prior_penalty": 0.0,
            "large_scale_warning": False,
        }
    logs = np.asarray(
        jax.device_get(model.band_calibration.log_alpha_band),
        dtype=float,
    )
    return payload | per_band_flux_calibration_metadata(
        logs,
        tuple(band_names),
        cfg.prior_sigma_log_alpha,
    )


def write_per_band_flux_calibration_artifacts(
    out: Path,
    model: AmortizedModel,
    config: dict[str, Any],
    *,
    band_names: tuple[str, ...],
    config_block=None,
    trainable: bool | None = None,
) -> None:
    """Write per-band calibration JSON, CSV, and a compact bar plot."""
    payload = _per_band_flux_calibration_summary(
        model,
        config,
        band_names=band_names,
        config_block=config_block,
        trainable=trainable,
    )
    write_json(out / "per_band_flux_calibration.json", payload)
    bands = payload.get("bands", {})
    if not bands:
        return
    rows = [
        {"band": band, **values}
        for band, values in bands.items()
        if isinstance(values, dict)
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "per_band_flux_calibration.csv", index=False)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.bar(
        frame["band"].astype(str),
        frame["delta_mag_band"].to_numpy(dtype=float),
        color="#3b82f6",
    )
    ax.set_ylabel("model offset (mag)")
    ax.set_xlabel("band")
    ax.set_title("Per-band flux calibration")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out / "per_band_flux_calibration.png", dpi=170)
    plt.close(fig)


def build_training_split(
    config: dict[str, Any],
    *,
    limit: int | None,
    seed: int,
    row_indices_file: str | Path | None = None,
    train_indices_file: str | Path | None = None,
    validation_indices_file: str | Path | None = None,
) -> TrainingSplit:
    """Build a reproducible train/validation row split for FS2."""
    cfg = amortized_config(config)
    data_cfg = cfg["data"]
    validation_fraction = float(data_cfg.get("validation_fraction", 0.1))
    validation_fraction = min(max(validation_fraction, 0.0), 0.9)
    selection_mode = str(data_cfg.get("selection_mode", "stratified_redshift"))
    stratified_strategy = str(data_cfg.get("stratified_strategy", "balanced"))
    rng = np.random.default_rng(int(data_cfg.get("selection_seed", seed)))
    n_rows = _catalog_num_rows(config["catalog_path"])
    redshift_column = _configured_redshift_column(config, data_cfg)
    redshift = _read_redshift_column(config["catalog_path"], redshift_column)
    if redshift is not None:
        n_rows = len(redshift)
    redshift_bins = np.asarray(
        data_cfg.get(
            "redshift_bins",
            [0.0, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.5, 6.0],
        ),
        dtype=float,
    )
    if redshift_bins.ndim != 1 or redshift_bins.size < 2:
        redshift_bins = np.asarray([0.0, 6.0], dtype=float)
    redshift_bins = np.unique(redshift_bins)

    if train_indices_file or validation_indices_file:
        if row_indices_file:
            raise ValueError(
                "--row-indices-file cannot be combined with explicit train/validation "
                "index files"
            )
        if limit is not None:
            raise ValueError(
                "--limit cannot be combined with explicit train/validation index "
                "files; create a smaller row-index file for smoke tests."
            )
        if not train_indices_file:
            raise ValueError("--train-indices-file is required with validation indices")
        train_indices = np.asarray(load_row_indices(train_indices_file), dtype=np.int64)
        validation_indices = (
            np.asarray(load_row_indices(validation_indices_file), dtype=np.int64)
            if validation_indices_file
            else np.asarray([], dtype=np.int64)
        )
        _validate_selected_indices(train_indices, n_rows, "train_indices_file")
        _validate_selected_indices(
            validation_indices,
            n_rows,
            "validation_indices_file",
        )
        overlap = np.intersect1d(train_indices, validation_indices)
        if overlap.size:
            raise ValueError(
                "train and validation index files overlap; first overlapping "
                f"row_index={int(overlap[0])}"
            )
        return TrainingSplit(
            train_indices=train_indices,
            validation_indices=validation_indices,
            train_redshift=_redshift_for_indices(redshift, train_indices),
            validation_redshift=_redshift_for_indices(redshift, validation_indices),
            redshift_column=redshift_column if redshift is not None else None,
            redshift_bins=redshift_bins,
            selection_mode="explicit_train_validation_files",
            stratified_strategy=stratified_strategy,
            validation_fraction=(
                float(validation_indices.size)
                / float(max(train_indices.size + validation_indices.size, 1))
            ),
            train_indices_file=str(train_indices_file),
            validation_indices_file=(
                str(validation_indices_file) if validation_indices_file else None
            ),
        )

    if row_indices_file:
        selected = np.asarray(load_row_indices(row_indices_file), dtype=np.int64)
        if limit is not None:
            selected = selected[: min(max(int(limit), 0), selected.size)]
        _validate_selected_indices(selected, n_rows, "row_indices_file")
        train_indices, validation_indices = _split_train_validation(
            selected,
            redshift,
            redshift_bins,
            validation_fraction,
            rng,
        )
        return TrainingSplit(
            train_indices=np.asarray(train_indices, dtype=np.int64),
            validation_indices=np.asarray(validation_indices, dtype=np.int64),
            train_redshift=_redshift_for_indices(redshift, train_indices),
            validation_redshift=_redshift_for_indices(redshift, validation_indices),
            redshift_column=redshift_column if redshift is not None else None,
            redshift_bins=redshift_bins,
            selection_mode="row_indices_file",
            stratified_strategy=stratified_strategy,
            validation_fraction=validation_fraction,
            row_indices_file=str(row_indices_file),
        )

    total = n_rows if limit is None else min(int(limit), n_rows)
    if selection_mode == "sequential":
        selected = np.arange(total, dtype=np.int64)
    elif selection_mode == "random" or redshift is None:
        selected = rng.choice(n_rows, size=total, replace=False).astype(np.int64)
    elif selection_mode == "stratified_redshift":
        selected = _select_stratified_indices(
            redshift,
            total=total,
            bins=redshift_bins,
            strategy=stratified_strategy,
            rng=rng,
        )
    else:
        raise ValueError(
            "amortized.data.selection_mode must be one of "
            "'sequential', 'random', or 'stratified_redshift'"
        )

    train_indices, validation_indices = _split_train_validation(
        selected,
        redshift,
        redshift_bins,
        validation_fraction,
        rng,
    )
    return TrainingSplit(
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        train_redshift=_redshift_for_indices(redshift, train_indices),
        validation_redshift=_redshift_for_indices(redshift, validation_indices),
        redshift_column=redshift_column if redshift is not None else None,
        redshift_bins=redshift_bins,
        selection_mode=selection_mode if redshift is not None else "random",
        stratified_strategy=stratified_strategy,
        validation_fraction=validation_fraction,
    )


def write_training_split_artifacts(out: Path, split: TrainingSplit) -> None:
    """Write reproducibility artifacts for a training split."""
    np.save(out / "train_indices.npy", split.train_indices)
    np.save(out / "validation_indices.npy", split.validation_indices)
    if split.train_redshift.size:
        np.save(out / "train_redshift_proxy.npy", split.train_redshift)
    if split.validation_redshift.size:
        np.save(out / "validation_redshift_proxy.npy", split.validation_redshift)
    write_json(
        out / "training_split.json",
        {
            "selection_mode": split.selection_mode,
            "stratified_strategy": split.stratified_strategy,
            "validation_fraction": split.validation_fraction,
            "redshift_column": split.redshift_column,
            "redshift_bins": split.redshift_bins.tolist(),
            "train_rows": int(len(split.train_indices)),
            "validation_rows": int(len(split.validation_indices)),
            "row_indices_file": split.row_indices_file,
            "train_indices_file": split.train_indices_file,
            "validation_indices_file": split.validation_indices_file,
            "train_redshift_finite": int(np.isfinite(split.train_redshift).sum()),
            "validation_redshift_finite": int(
                np.isfinite(split.validation_redshift).sum()
            ),
        },
    )


def evaluate_validation_epoch(
    model: AmortizedModel,
    arrays,
    split: TrainingSplit,
    feature_stats: FeatureStats,
    latent_spec: LatentSpec,
    context,
    model_args,
    key,
    *,
    batch_size: int,
    n_samples: int,
    kl_weight: float,
    likelihood_config: dict[str, Any],
    progress: bool,
    total: int | None,
    desc: str,
    epoch: int,
    calibration_config: dict[str, Any] | None = None,
    objective_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Evaluate validation ELBO and redshift-bin metrics without updates."""
    rows: list[dict[str, float | int | str]] = []
    object_rows: list[dict[str, float | int | str]] = []
    redshift_lookup = {
        int(object_id): float(z)
        for object_id, z in zip(
            np.asarray(arrays.object_id),
            split.validation_redshift,
            strict=False,
        )
    }
    jit_context = context if isinstance(context, _StaticArg) else _StaticArg(context)
    jit_latent_spec = (
        latent_spec
        if isinstance(latent_spec, JitLatentSpec)
        else JitLatentSpec(
            names=latent_spec.names,
            lower=latent_spec.lower,
            upper=latent_spec.upper,
            raw_center=latent_spec.raw_center,
            raw_scale=latent_spec.raw_scale,
            normalization=latent_spec.normalization,
            transform_family=latent_spec.transform_family,
            transform_location=latent_spec.transform_location,
            transform_lambda=latent_spec.transform_lambda,
        )
    )
    with _progress_bar(
        enabled=bool(progress),
        total=total,
        desc=desc,
        unit="batch",
    ) as pbar:
        for batch_index, batch in enumerate(
            iter_photometry_batches_from_arrays(
                arrays,
                batch_size=batch_size,
                feature_stats=feature_stats,
                truth_names=(
                    latent_spec.names
                    if objective_mode(objective_config) == "neural_posterior_estimation"
                    else None
                ),
            )
        ):
            key, batch_key = jax.random.split(key)
            metrics, object_metrics = _evaluation_metrics_jit(
                model,
                _loss_batch(batch),
                jit_latent_spec,
                jit_context,
                model_args,
                jit_latent_spec.names,
                batch_key,
                int(n_samples),
                float(kl_weight),
                likelihood_config,
                calibration_config,
                objective_config or {},
            )
            record = _metrics_record(metrics)
            record.update(
                {
                    "split": "validation",
                    "epoch": int(epoch),
                    "batch": int(batch_index),
                    "kl_weight": float(kl_weight),
                    "n_objects": int(batch.flux.shape[0]),
                    "loss_finite": float(np.isfinite(record["loss"])),
                    "grads_finite": 1.0,
                    "update_applied": 0.0,
                    "encoder_grad_norm": 0.0,
                    "prior_grad_norm": 0.0,
                    "alpha_grad_norm": 0.0,
                    "band_alpha_grad_norm": 0.0,
                    "joint_grad_norm": 0.0,
                    "encoder_grad_nonzero": 0.0,
                    "prior_grad_nonzero": 0.0,
                    "alpha_grad_nonzero": 0.0,
                    "band_alpha_grad_nonzero": 0.0,
                }
            )
            rows.append(record)
            object_rows.extend(
                _validation_object_rows(
                    batch,
                    object_metrics,
                    redshift_lookup,
                    split.redshift_bins,
                )
            )
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{record['loss']:.3g}",
                    "nll": f"{record['negative_loglike']:.3g}",
                    "kl": f"{record['kl_mc_mean']:.3g}",
                },
                refresh=False,
            )
    bin_rows = _redshift_bin_rows(
        object_rows,
        bins=split.redshift_bins,
        epoch=int(epoch),
    )
    return rows, bin_rows


def _evaluation_metrics(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
    objective_config,
):
    objective_config = objective_config or {}
    if objective_mode(objective_config) == "neural_posterior_estimation":
        _loss, metrics = _loss_with_metrics(
            model,
            batch,
            latent_spec,
            context,
            model_args,
            parameter_names,
            key,
            n_samples,
            kl_weight,
            likelihood_config,
            calibration_config,
            objective_config,
        )
        truth_x = theta_to_x(batch.truth_theta, latent_spec)
        object_nll = -posterior_log_prob(model, batch.features, truth_x)
        zeros = jnp.zeros_like(object_nll)
        return metrics, {
            "loss": object_nll,
            "negative_loglike": object_nll,
            "kl_mc_mean": zeros,
            "posterior_predictive_chi2": zeros,
        }
    deterministic = is_deterministic_reconstruction(objective_config)
    mean, log_std = model.encoder(batch.features)
    if deterministic:
        x_samples = mean[None, ...]
        logq = jnp.zeros(mean.shape[:-1], dtype=mean.dtype)[None, ...]
    else:
        posterior = sample_posterior(model, key, batch.features, int(n_samples))
        x_samples, logq = posterior.x, posterior.logq
    model_flux_raw = model_flux_from_x(
        x_samples,
        latent_spec,
        context,
        model_args,
        parameter_names,
    )
    scale_cfg = global_sed_scale_config(calibration_config)
    band_cfg = per_band_flux_calibration_config(calibration_config)
    log_alpha_sed = model.sed_scale.log_alpha_sed
    model_flux = (
        apply_global_sed_scale_to_flux(model_flux_raw, log_alpha_sed)
        if scale_cfg.enabled
        else model_flux_raw
    )
    log_alpha_band = (
        model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((model_flux_raw.shape[-1],), dtype=model_flux_raw.dtype)
    )
    model_flux = (
        apply_per_band_flux_calibration_to_flux(model_flux, log_alpha_band)
        if band_cfg.enabled
        else model_flux
    )
    alpha_prior_penalty = (
        global_sed_scale_prior_penalty(
            log_alpha_sed,
            scale_cfg.prior_sigma_log_alpha,
        )
        if scale_cfg.enabled
        else jnp.asarray(0.0, dtype=model_flux.dtype)
    )
    band_prior_penalty = (
        per_band_flux_calibration_prior_penalty(
            log_alpha_band,
            band_cfg.prior_sigma_log_alpha,
        )
        if band_cfg.enabled
        else jnp.asarray(0.0, dtype=model_flux.dtype)
    )
    loglike = photometric_loglike(
        obs_flux=batch.flux,
        model_flux=model_flux,
        obs_err=batch.flux_err,
        mask=batch.mask,
        likelihood_type=str(likelihood_config.get("type", "student_t")),
        student_t_dof=float(likelihood_config.get("student_t_dof", 2.0)),
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
    )
    logp = jnp.zeros_like(logq) if deterministic else posterior.logprior
    kl = logq - logp
    likelihood_temperature = jnp.asarray(
        max(float(objective_config.get("likelihood_temperature", 1.0)), 1.0e-6),
        dtype=loglike.dtype,
    )
    entropy_penalty = (
        jnp.asarray(0.0, dtype=loglike.dtype)
        if deterministic
        else _posterior_entropy_floor_penalty_train(log_std, objective_config)
    )
    objective = -loglike / likelihood_temperature + float(kl_weight) * kl
    chi = photometric_normalized_residual(
        obs_flux=batch.flux,
        model_flux=model_flux,
        obs_err=batch.flux_err,
        mask=batch.mask,
        error_floor_frac=float(likelihood_config.get("error_floor_frac", 0.02)),
        error_jitter=float(likelihood_config.get("error_jitter", 0.0)),
    )
    raw_flux_residual = jnp.where(
        batch.mask[None, :, :],
        model_flux - batch.flux[None, :, :],
        0.0,
    )
    n_valid_residual = jnp.maximum(jnp.sum(batch.mask) * x_samples.shape[0], 1)
    n_valid_flux = jnp.maximum(jnp.sum(batch.mask), 1)
    object_loss = jnp.mean(objective, axis=0)
    object_nll = jnp.mean(-loglike, axis=0)
    object_kl = jnp.mean(kl, axis=0)
    object_chi2 = jnp.mean(jnp.sum(chi**2, axis=-1), axis=0)
    metrics = {
        "loss": (
            jnp.mean(objective)
            + alpha_prior_penalty
            + band_prior_penalty
            + entropy_penalty
        ),
        "negative_loglike": jnp.mean(-loglike),
        "loglike_mean": jnp.mean(loglike),
        "logprior_mean": jnp.mean(logp),
        "logq_mean": jnp.mean(logq),
        "kl_mc_mean": jnp.mean(kl),
        "likelihood_temperature": likelihood_temperature,
        "entropy_floor_penalty": entropy_penalty,
        "posterior_entropy_mean": (
            jnp.asarray(0.0, dtype=loglike.dtype)
            if deterministic
            else _diag_gaussian_entropy_train(log_std).mean()
        ),
        "posterior_min_log_std": (
            jnp.asarray(0.0, dtype=loglike.dtype) if deterministic else jnp.min(log_std)
        ),
        "posterior_median_log_std": (
            jnp.asarray(0.0, dtype=loglike.dtype)
            if deterministic
            else jnp.median(log_std)
        ),
        "posterior_max_log_std": (
            jnp.asarray(0.0, dtype=loglike.dtype) if deterministic else jnp.max(log_std)
        ),
        "deterministic_reconstruction": jnp.asarray(
            1.0 if deterministic else 0.0,
            dtype=loglike.dtype,
        ),
        "effective_n_samples": jnp.asarray(x_samples.shape[0], dtype=loglike.dtype),
        "model_flux_mean": jnp.mean(model_flux),
        "mean_model_flux_raw": jnp.mean(model_flux_raw),
        "mean_model_flux_scaled": jnp.mean(model_flux),
        "log_alpha_sed": log_alpha_sed,
        "alpha_sed": alpha_from_log_alpha(log_alpha_sed),
        "alpha_prior_penalty": alpha_prior_penalty,
        "band_alpha_prior_penalty": band_prior_penalty,
        "max_abs_band_delta_mag": jnp.max(
            jnp.abs(-2.5 * log_alpha_band / jnp.log(jnp.asarray(10.0)))
        ),
        "residual_rms": jnp.sqrt(jnp.sum(chi**2) / n_valid_residual),
        "flux_residual_rms": jnp.sqrt(jnp.sum(raw_flux_residual**2) / n_valid_flux),
        "finite_fraction": jnp.mean(jnp.isfinite(objective)),
    }
    object_metrics = {
        "loss": object_loss,
        "negative_loglike": object_nll,
        "kl_mc_mean": object_kl,
        "posterior_predictive_chi2": object_chi2,
    }
    return metrics, object_metrics


def _validation_object_rows(
    batch: PhotometryBatch,
    object_metrics: dict[str, jnp.ndarray],
    redshift_lookup: dict[int, float],
    bins: np.ndarray,
) -> list[dict[str, float | int | str]]:
    object_ids = np.asarray(jax.device_get(batch.object_id), dtype=int)
    losses = np.asarray(jax.device_get(object_metrics["loss"]), dtype=float)
    nll = np.asarray(jax.device_get(object_metrics["negative_loglike"]), dtype=float)
    kl = np.asarray(jax.device_get(object_metrics["kl_mc_mean"]), dtype=float)
    chi2 = np.asarray(
        jax.device_get(object_metrics["posterior_predictive_chi2"]), dtype=float
    )
    rows = []
    for index, object_id in enumerate(object_ids):
        z = redshift_lookup.get(int(object_id), float("nan"))
        bin_index = _redshift_bin_index(z, bins)
        rows.append(
            {
                "object_id": int(object_id),
                "z_reference": float(z),
                "z_bin_index": int(bin_index),
                "loss": float(losses[index]),
                "negative_loglike": float(nll[index]),
                "kl_mc_mean": float(kl[index]),
                "posterior_predictive_chi2": float(chi2[index]),
            }
        )
    return rows


def _redshift_bin_rows(
    object_rows: list[dict[str, float | int | str]],
    *,
    bins: np.ndarray,
    epoch: int,
) -> list[dict[str, float | int | str]]:
    if not object_rows:
        return []
    frame = pd.DataFrame(object_rows)
    rows = []
    for bin_index, group in frame.groupby("z_bin_index", sort=True):
        bin_index = int(bin_index)
        if bin_index < 0:
            z_min = float("nan")
            z_max = float("nan")
            label = "missing"
        else:
            z_min = float(bins[bin_index])
            z_max = float(bins[bin_index + 1])
            label = f"{z_min:.3g}-{z_max:.3g}"
        z_reference = group["z_reference"].to_numpy(dtype=float)
        finite_z_reference = z_reference[np.isfinite(z_reference)]
        rows.append(
            {
                "epoch": int(epoch),
                "split": "validation",
                "z_bin_index": bin_index,
                "z_bin": label,
                "z_min": z_min,
                "z_max": z_max,
                "n_objects": int(len(group)),
                "loss": float(np.nanmean(group["loss"])),
                "negative_loglike": float(np.nanmean(group["negative_loglike"])),
                "kl_mc_mean": float(np.nanmean(group["kl_mc_mean"])),
                "posterior_predictive_chi2": float(
                    np.nanmedian(group["posterior_predictive_chi2"])
                ),
                "z_reference_median": (
                    float(np.median(finite_z_reference))
                    if finite_z_reference.size
                    else float("nan")
                ),
            }
        )
    return rows


def _catalog_num_rows(path: str | Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _validate_selected_indices(
    values: np.ndarray,
    n_rows: int,
    label: str,
) -> None:
    values = np.asarray(values, dtype=np.int64)
    if not values.size:
        return
    if int(values.min()) < 0 or int(values.max()) >= int(n_rows):
        raise ValueError(
            f"{label} contains row_index outside catalog bounds: "
            f"min={int(values.min())} max={int(values.max())} "
            f"catalog_rows={int(n_rows)}"
        )


def _configured_redshift_column(
    config: dict[str, Any],
    data_cfg: dict[str, Any],
) -> str | None:
    explicit = data_cfg.get("stratify_column")
    if explicit:
        return str(explicit)
    truth = config.get("truth", {}) or {}
    column = truth_column_from_spec(truth.get("redshift_column"))
    if column:
        return column
    redshift = config.get("redshift", {}) or {}
    return truth_column_from_spec(redshift.get("truth_column"))


def _read_redshift_column(
    catalog_path: str | Path,
    column: str | None,
) -> np.ndarray | None:
    if column is None:
        return None
    try:
        frame = pd.read_parquet(catalog_path, columns=[column])
    except Exception:
        return None
    if column not in frame:
        return None
    values = frame[column].to_numpy(dtype=float)
    return values


def _select_stratified_indices(
    redshift: np.ndarray,
    *,
    total: int,
    bins: np.ndarray,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    all_indices = np.arange(len(redshift), dtype=np.int64)
    if total >= len(all_indices):
        selected = all_indices.copy()
        rng.shuffle(selected)
        return selected
    finite = np.isfinite(redshift)
    if not finite.any():
        return rng.choice(all_indices, size=total, replace=False).astype(np.int64)
    if strategy == "proportional":
        finite_indices = all_indices[finite]
        if total <= len(finite_indices):
            return rng.choice(finite_indices, size=total, replace=False).astype(
                np.int64
            )
        selected = list(finite_indices)
        remaining = np.setdiff1d(all_indices, finite_indices, assume_unique=False)
        needed = total - len(selected)
        selected.extend(rng.choice(remaining, size=needed, replace=False).tolist())
        selected = np.asarray(selected, dtype=np.int64)
        rng.shuffle(selected)
        return selected
    if strategy != "balanced":
        raise ValueError(
            "amortized.data.stratified_strategy must be balanced or proportional"
        )
    groups = _indices_by_redshift_bin(redshift, bins)
    nonempty = [group for group in groups if len(group)]
    if not nonempty:
        return rng.choice(all_indices, size=total, replace=False).astype(np.int64)
    per_bin = int(np.ceil(total / float(len(nonempty))))
    selected: list[int] = []
    for group in nonempty:
        take = min(per_bin, len(group), total - len(selected))
        if take <= 0:
            break
        selected.extend(rng.choice(group, size=take, replace=False).tolist())
    if len(selected) < total:
        remaining = np.setdiff1d(all_indices, np.asarray(selected), assume_unique=False)
        selected.extend(
            rng.choice(remaining, size=total - len(selected), replace=False).tolist()
        )
    selected = np.asarray(selected[:total], dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _split_train_validation(
    selected: np.ndarray,
    redshift: np.ndarray | None,
    bins: np.ndarray,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(selected, dtype=np.int64)
    if validation_fraction <= 0.0 or len(selected) < 2:
        return selected, np.asarray([], dtype=np.int64)
    if redshift is None:
        shuffled = selected.copy()
        rng.shuffle(shuffled)
        n_val = int(round(validation_fraction * len(shuffled)))
        n_val = min(max(n_val, 1), len(shuffled) - 1)
        return shuffled[n_val:], shuffled[:n_val]
    val: list[int] = []
    train: list[int] = []
    for group in _indices_by_redshift_bin(
        redshift[selected], bins, base_indices=selected
    ):
        group = np.asarray(group, dtype=np.int64)
        if len(group) == 0:
            continue
        rng.shuffle(group)
        n_val = int(round(validation_fraction * len(group)))
        if len(group) > 1 and validation_fraction > 0.0:
            n_val = min(max(n_val, 1), len(group) - 1)
        val.extend(group[:n_val].tolist())
        train.extend(group[n_val:].tolist())
    if not train:
        shuffled = selected.copy()
        rng.shuffle(shuffled)
        return shuffled[1:], shuffled[:1]
    train = np.asarray(train, dtype=np.int64)
    val = np.asarray(val, dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _indices_by_redshift_bin(
    redshift: np.ndarray,
    bins: np.ndarray,
    *,
    base_indices: np.ndarray | None = None,
) -> list[np.ndarray]:
    if base_indices is None:
        base_indices = np.arange(len(redshift), dtype=np.int64)
    redshift = np.asarray(redshift, dtype=float)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    groups = []
    for index in range(len(bins) - 1):
        lo = bins[index]
        hi = bins[index + 1]
        if index == len(bins) - 2:
            mask = (redshift >= lo) & (redshift <= hi)
        else:
            mask = (redshift >= lo) & (redshift < hi)
        mask &= np.isfinite(redshift)
        groups.append(base_indices[mask])
    missing = base_indices[~np.isfinite(redshift)]
    if len(missing):
        groups.append(missing)
    return groups


def _redshift_for_indices(
    redshift: np.ndarray | None,
    indices: np.ndarray,
) -> np.ndarray:
    if redshift is None or len(indices) == 0:
        return np.asarray([], dtype=float)
    return np.asarray(redshift[np.asarray(indices, dtype=int)], dtype=float)


def _redshift_bin_index(z: float, bins: np.ndarray) -> int:
    if not np.isfinite(z):
        return -1
    index = int(np.searchsorted(bins, z, side="right") - 1)
    if index < 0 or index >= len(bins) - 1:
        return -1
    return index


def _loss_with_metrics(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
    objective_config,
):
    if objective_mode(objective_config) == "neural_posterior_estimation":
        if batch.truth_theta.shape[-1] != len(latent_spec.names):
            raise ValueError("NPE requires complete truth theta columns in every batch")
        truth_x = theta_to_x(batch.truth_theta, latent_spec)
        logq_truth = posterior_log_prob(model, batch.features, truth_x)
        loss = -jnp.mean(logq_truth)
        zero = jnp.asarray(0.0, dtype=loss.dtype)
        mean, log_std = model.encoder(batch.features)
        metrics = {
            "loss": loss,
            "negative_loglike": loss,
            "loglike_mean": jnp.mean(logq_truth),
            "logprior_mean": zero,
            "logq_mean": jnp.mean(logq_truth),
            "kl_mc_mean": zero,
            "likelihood_temperature": jnp.asarray(1.0, dtype=loss.dtype),
            "entropy_floor_penalty": zero,
            "posterior_entropy_mean": _diag_gaussian_entropy_train(log_std).mean(),
            "posterior_min_log_std": jnp.min(log_std),
            "posterior_median_log_std": jnp.median(log_std),
            "posterior_max_log_std": jnp.max(log_std),
            "deterministic_reconstruction": zero,
            "effective_n_samples": jnp.asarray(1.0, dtype=loss.dtype),
            "model_flux_mean": zero,
            "mean_model_flux_raw": zero,
            "mean_model_flux_scaled": zero,
            "log_alpha_sed": model.sed_scale.log_alpha_sed,
            "alpha_sed": alpha_from_log_alpha(model.sed_scale.log_alpha_sed),
            "alpha_prior_penalty": zero,
            "band_alpha_prior_penalty": zero,
            "max_abs_band_delta_mag": zero,
            "residual_rms": zero,
            "flux_residual_rms": zero,
            "finite_fraction": jnp.mean(jnp.isfinite(logq_truth)),
        }
        return loss, metrics
    return negative_elbo(
        model,
        batch,
        latent_spec,
        context,
        model_args,
        parameter_names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        calibration_config,
        objective_config,
    )


def _loss_batch(batch: PhotometryBatch) -> LossBatch:
    truth_theta = batch.truth_theta
    if truth_theta is None:
        truth_theta = jnp.zeros((batch.flux.shape[0], 0), dtype=batch.flux.dtype)
    return LossBatch(
        flux=batch.flux,
        flux_err=batch.flux_err,
        mask=batch.mask,
        features=batch.features,
        truth_theta=truth_theta,
    )


def _loss_batch_with_input_noise(
    batch: PhotometryBatch,
    feature_stats: FeatureStats,
    key,
    input_noise_config: dict[str, Any],
) -> LossBatch:
    if not bool(input_noise_config.get("enabled", False)):
        return _loss_batch(batch)
    sigma_scale = float(input_noise_config.get("sigma_scale", 1.0))
    if sigma_scale <= 0.0:
        return _loss_batch(batch)
    flux_err = jnp.where(jnp.isfinite(batch.flux_err), batch.flux_err, 0.0)
    noise = (
        sigma_scale
        * flux_err
        * jax.random.normal(
            key,
            shape=batch.flux.shape,
            dtype=batch.flux.dtype,
        )
    )
    noisy_flux = jnp.where(batch.mask, batch.flux + noise, batch.flux)
    return LossBatch(
        flux=batch.flux,
        flux_err=batch.flux_err,
        mask=batch.mask,
        features=make_encoder_features(noisy_flux, batch.flux_err, feature_stats),
        truth_theta=(
            batch.truth_theta
            if batch.truth_theta is not None
            else jnp.zeros((batch.flux.shape[0], 0), dtype=batch.flux.dtype)
        ),
    )


def _input_noise_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw or {})
    apply_to = str(cfg.get("apply_to", "train")).strip().lower()
    mode = str(cfg.get("mode", "encoder_flux")).strip().lower()
    sigma_scale = float(cfg.get("sigma_scale", 1.0))
    enabled = (
        bool(cfg.get("enabled", False))
        and sigma_scale > 0.0
        and apply_to
        in {
            "train",
            "training",
            "all",
        }
    )
    if enabled and mode != "encoder_flux":
        raise ValueError("amortized.input_noise.mode currently supports encoder_flux")
    if sigma_scale < 0.0:
        raise ValueError("amortized.input_noise.sigma_scale must be non-negative")
    return {
        "enabled": bool(enabled),
        "mode": mode,
        "sigma_scale": float(sigma_scale),
        "apply_to": apply_to,
        "target": "encoder_features_only",
    }


@eqx.filter_jit
def _loss_and_grads_jit(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
    objective_config,
):
    actual_context = context.value if isinstance(context, _StaticArg) else context
    return eqx.filter_value_and_grad(_loss_with_metrics, has_aux=True)(
        model,
        batch,
        latent_spec,
        actual_context,
        model_args,
        parameter_names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        calibration_config,
        objective_config,
    )


def _make_pmap_train_step(optimizer):
    pmap_array_axis = eqx.if_array(0)

    @eqx.filter_pmap(
        axis_name="devices",
        in_axes=(
            pmap_array_axis,
            pmap_array_axis,
            pmap_array_axis,
            None,
            None,
            None,
            None,
            pmap_array_axis,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        out_axes=(
            pmap_array_axis,
            pmap_array_axis,
            0,
            pmap_array_axis,
            pmap_array_axis,
            0,
            0,
            0,
        ),
    )
    def step(
        model,
        opt_state,
        batch,
        latent_spec,
        context,
        model_args,
        parameter_names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        calibration_config,
        objective_config,
        update_phase,
        train_alpha,
        train_band_calibration,
    ):
        actual_context = context.value if isinstance(context, _StaticArg) else context
        (loss, metrics), grads = eqx.filter_value_and_grad(
            _loss_with_metrics,
            has_aux=True,
        )(
            model,
            batch,
            latent_spec,
            actual_context,
            model_args,
            parameter_names,
            key,
            n_samples,
            kl_weight,
            likelihood_config,
            calibration_config,
            objective_config,
        )
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        metrics = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(value, axis_name="devices"),
            metrics,
        )
        grads = _apply_training_grad_masks(
            grads,
            update_phase=update_phase,
            train_alpha=bool(train_alpha),
            train_band_calibration=bool(train_band_calibration),
        )
        grad_norms = _component_grad_norms_jax(grads)
        loss_finite = jnp.isfinite(loss)
        grads_finite = _tree_all_finite_jax(grads)
        update_applied = jax.lax.pmin(
            (loss_finite & grads_finite).astype(jnp.int32),
            axis_name="devices",
        ).astype(jnp.bool_)
        safe_grads = _zero_tree_when_false(grads, update_applied)
        updates, new_opt_state = optimizer.update(
            safe_grads,
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        new_model = eqx.apply_updates(model, updates)
        model = _select_tree_when_false(new_model, model, update_applied)
        opt_state = _select_tree_when_false(new_opt_state, opt_state, update_applied)
        return (
            model,
            opt_state,
            loss,
            metrics,
            grad_norms,
            loss_finite,
            grads_finite,
            update_applied,
        )

    return step


def _apply_training_grad_masks(
    grads: AmortizedModel,
    *,
    update_phase: str,
    train_alpha: bool,
    train_band_calibration: bool,
) -> AmortizedModel:
    if update_phase in {"encoder", "joint_no_prior", "frozen_prior"}:
        grads = zero_prior_grads(grads)
    if update_phase == "prior":
        grads = zero_encoder_grads(grads)
        grads = zero_sed_scale_grads(grads)
        grads = zero_band_calibration_grads(grads)
    if not train_alpha:
        grads = zero_sed_scale_grads(grads)
    if not train_band_calibration:
        grads = zero_band_calibration_grads(grads)
    return grads


def _component_grad_norms_jax(grads: AmortizedModel) -> dict[str, jnp.ndarray]:
    encoder_norm = _tree_l2_norm_jax(getattr(grads, "encoder", None))
    prior_norm = _tree_l2_norm_jax(getattr(grads, "prior", None))
    alpha_norm = _tree_l2_norm_jax(getattr(grads, "sed_scale", None))
    band_alpha_norm = _tree_l2_norm_jax(getattr(grads, "band_calibration", None))
    return {
        "encoder_grad_norm": encoder_norm,
        "prior_grad_norm": prior_norm,
        "alpha_grad_norm": alpha_norm,
        "band_alpha_grad_norm": band_alpha_norm,
        "joint_grad_norm": _tree_l2_norm_jax(grads),
        "encoder_grad_nonzero": (encoder_norm > 0.0).astype(jnp.float32),
        "prior_grad_nonzero": (prior_norm > 0.0).astype(jnp.float32),
        "alpha_grad_nonzero": (alpha_norm > 0.0).astype(jnp.float32),
        "band_alpha_grad_nonzero": (band_alpha_norm > 0.0).astype(jnp.float32),
    }


def _tree_l2_norm_jax(tree) -> jnp.ndarray:
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = sum(jnp.sum(jnp.asarray(leaf) ** 2) for leaf in leaves)
    return jnp.sqrt(total)


def _tree_all_finite_jax(tree) -> jnp.ndarray:
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return jnp.asarray(True)
    flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return jnp.all(jnp.asarray(flags))


def _zero_tree_when_false(tree, predicate):
    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.where(predicate, leaf, jnp.zeros_like(leaf))
        return leaf

    return jax.tree_util.tree_map(zero_leaf, tree)


def _select_tree_when_false(true_tree, false_tree, predicate):
    def select_leaf(true_leaf, false_leaf):
        if eqx.is_array(true_leaf):
            return jnp.where(predicate, true_leaf, false_leaf)
        return true_leaf

    return jax.tree_util.tree_map(select_leaf, true_tree, false_tree)


@eqx.filter_jit
def _evaluation_metrics_jit(
    model,
    batch,
    latent_spec,
    context,
    model_args,
    parameter_names,
    key,
    n_samples,
    kl_weight,
    likelihood_config,
    calibration_config,
    objective_config,
):
    actual_context = context.value if isinstance(context, _StaticArg) else context
    return _evaluation_metrics(
        model,
        batch,
        latent_spec,
        actual_context,
        model_args,
        parameter_names,
        key,
        n_samples,
        kl_weight,
        likelihood_config,
        calibration_config,
        objective_config,
    )


def _kl_weight(epoch: int, annealing_epochs: int, *, max_weight: float = 1.0) -> float:
    max_weight = min(max(float(max_weight), 0.0), 1.0)
    if annealing_epochs <= 0:
        return max_weight
    return max_weight * min(1.0, max(0.0, float(epoch) / float(annealing_epochs)))


def _metrics_record(metrics: dict[str, jnp.ndarray]) -> dict[str, float]:
    return {
        name: float(np.asarray(jax.device_get(value)))
        for name, value in metrics.items()
    }


def component_grad_norms(grads: AmortizedModel) -> dict[str, float]:
    """Return L2 gradient norms for the jointly trained neural components."""
    encoder_norm = _tree_l2_norm(getattr(grads, "encoder", None))
    prior_norm = _tree_l2_norm(getattr(grads, "prior", None))
    alpha_norm = _tree_l2_norm(getattr(grads, "sed_scale", None))
    band_alpha_norm = _tree_l2_norm(getattr(grads, "band_calibration", None))
    return {
        "encoder_grad_norm": encoder_norm,
        "prior_grad_norm": prior_norm,
        "alpha_grad_norm": alpha_norm,
        "band_alpha_grad_norm": band_alpha_norm,
        "joint_grad_norm": _tree_l2_norm(grads),
        "encoder_grad_nonzero": float(encoder_norm > 0.0),
        "prior_grad_nonzero": float(prior_norm > 0.0),
        "alpha_grad_nonzero": float(alpha_norm > 0.0),
        "band_alpha_grad_nonzero": float(band_alpha_norm > 0.0),
    }


def zero_prior_grads(grads: AmortizedModel) -> AmortizedModel:
    """Return gradients with all trainable prior leaves set to zero."""
    prior_grads = getattr(grads, "prior", None)
    if prior_grads is None:
        return grads

    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.zeros_like(leaf)
        return leaf

    return eqx.tree_at(
        lambda tree: tree.prior,
        grads,
        jax.tree_util.tree_map(zero_leaf, prior_grads),
    )


def zero_encoder_grads(grads: AmortizedModel) -> AmortizedModel:
    """Return gradients with all trainable encoder leaves set to zero."""
    encoder_grads = getattr(grads, "encoder", None)
    if encoder_grads is None:
        return grads

    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.zeros_like(leaf)
        return leaf

    return eqx.tree_at(
        lambda tree: tree.encoder,
        grads,
        jax.tree_util.tree_map(zero_leaf, encoder_grads),
    )


def zero_sed_scale_grads(grads: AmortizedModel) -> AmortizedModel:
    """Return gradients with global SED-scale leaves set to zero."""
    sed_scale_grads = getattr(grads, "sed_scale", None)
    if sed_scale_grads is None:
        return grads

    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.zeros_like(leaf)
        return leaf

    return eqx.tree_at(
        lambda tree: tree.sed_scale,
        grads,
        jax.tree_util.tree_map(zero_leaf, sed_scale_grads),
    )


def zero_band_calibration_grads(grads: AmortizedModel) -> AmortizedModel:
    """Return gradients with per-band calibration leaves set to zero."""
    band_grads = getattr(grads, "band_calibration", None)
    if band_grads is None:
        return grads

    def zero_leaf(leaf):
        if eqx.is_inexact_array(leaf):
            return jnp.zeros_like(leaf)
        return leaf

    return eqx.tree_at(
        lambda tree: tree.band_calibration,
        grads,
        jax.tree_util.tree_map(zero_leaf, band_grads),
    )


def _train_prior_jointly(prior_cfg: dict[str, Any]) -> bool:
    source = str(prior_cfg.get("source", "joint_realnvp"))
    if source == "standard_normal":
        return False
    joint_sources = {
        "joint_realnvp",
        "realnvp",
        "joint_rq_spline",
        "rq_spline",
        "rq_spline_coupling",
        "neural_spline",
        "neural_spline_flow",
    }
    return bool(prior_cfg.get("train_jointly", source in joint_sources))


def _log_prior_safety_messages(
    enabled: bool,
    prior_cfg: dict[str, Any],
    *,
    train_prior: bool,
    kl_weight_max: float,
) -> None:
    if not enabled:
        return
    source = str(prior_cfg.get("source", "joint_realnvp"))
    if (
        source
        in {
            "joint_realnvp",
            "realnvp",
            "joint_rq_spline",
            "rq_spline",
            "rq_spline_coupling",
            "neural_spline",
            "neural_spline_flow",
        }
        and not train_prior
        and float(kl_weight_max) > 0.0
    ):
        _log(
            True,
            "[amortized] WARNING: joint flow prior is fixed while KL is "
            "enabled. This is a random fixed flow prior unless a checkpoint was "
            "loaded explicitly.",
        )
    if source == "standard_normal" and float(kl_weight_max) > 0.0:
        _log(
            True,
            "[amortized] KL prior is the fixed standard normal reference prior.",
        )
    if source in {"supervised_checkpoint", "spline15d_checkpoint"} and not train_prior:
        _log(
            True,
            "[amortized] supervised flow prior is frozen: KL updates the encoder "
            "but does not update flow weights.",
        )
    if source == "supervised_checkpoint" and train_prior:
        _log(
            True,
            "[amortized] supervised flow prior is being fine-tuned jointly; "
            "compare against truth-overlap diagnostics.",
        )


def _prior_update_schedule(prior_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(prior_cfg.get("update_schedule", "joint")).lower(),
        "freeze_epochs": max(0, int(prior_cfg.get("freeze_epochs", 0))),
        "update_every_epochs": max(1, int(prior_cfg.get("update_every_epochs", 1))),
    }


def _training_update_phase(
    prior_cfg: dict[str, Any],
    *,
    epoch: int,
    train_prior: bool,
) -> str:
    if not train_prior:
        return "joint_no_prior"
    schedule = _prior_update_schedule(prior_cfg)
    if int(epoch) <= int(schedule["freeze_epochs"]):
        return "frozen_prior"
    mode = str(schedule["mode"])
    if mode in {"joint", "delayed_joint"}:
        return "joint"
    if mode in {"alternating", "alternating_epochs"}:
        offset = int(epoch) - int(schedule["freeze_epochs"]) - 1
        period = 2 * int(schedule["update_every_epochs"])
        return (
            "prior"
            if offset % period >= int(schedule["update_every_epochs"])
            else "encoder"
        )
    if mode in {"encoder_then_prior", "prior_only_after_freeze"}:
        return "prior"
    raise ValueError(
        "amortized.prior.update_schedule must be one of "
        "joint, delayed_joint, alternating, encoder_then_prior"
    )


def _objective_config_for_epoch(
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    training = cfg["training"]
    temperature = _annealed_value(
        epoch=epoch,
        initial=float(training.get("likelihood_temperature_initial", 1.0)),
        final=float(training.get("likelihood_temperature_final", 1.0)),
        epochs=int(training.get("likelihood_temperature_annealing_epochs", 0)),
    )
    regularization = dict(cfg.get("posterior_regularization", {}) or {})
    if regularization:
        initial_weight = float(regularization.get("weight", 0.0))
        final_weight = float(regularization.get("final_weight", initial_weight))
        regularization["weight"] = _annealed_value(
            epoch=epoch,
            initial=initial_weight,
            final=final_weight,
            epochs=int(regularization.get("anneal_epochs", 0)),
        )
    objective = dict(cfg.get("objective", {}) or {})
    return {
        "mode": str(objective.get("mode", "stochastic_elbo")),
        "likelihood_temperature": float(temperature),
        "posterior_regularization": regularization,
    }


def _annealed_value(*, epoch: int, initial: float, final: float, epochs: int) -> float:
    if epochs <= 1:
        return float(final)
    t = min(1.0, max(0.0, (float(epoch) - 1.0) / (float(epochs) - 1.0)))
    return float(initial + t * (final - initial))


def _diag_gaussian_entropy_train(log_std: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum(
        1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * log_std,
        axis=-1,
    )


def _posterior_entropy_floor_penalty_train(
    log_std: jnp.ndarray,
    objective_config: dict[str, Any],
) -> jnp.ndarray:
    regularization = dict(objective_config.get("posterior_regularization", {}) or {})
    enabled = bool(regularization.get("entropy_floor_enabled", False))
    weight = float(regularization.get("weight", 0.0))
    if not enabled or weight <= 0.0:
        return jnp.asarray(0.0, dtype=log_std.dtype)
    min_log_std = regularization.get("min_log_std")
    if min_log_std is None:
        min_scale = float(regularization.get("min_scale", 0.0))
        if min_scale <= 0.0:
            return jnp.asarray(0.0, dtype=log_std.dtype)
        min_log_std = float(np.log(min_scale))
    deficit = jnp.maximum(
        jnp.asarray(float(min_log_std), dtype=log_std.dtype) - log_std,
        jnp.asarray(0.0, dtype=log_std.dtype),
    )
    return jnp.asarray(weight, dtype=log_std.dtype) * jnp.mean(deficit**2)


def tree_all_finite(tree) -> bool:
    """Return whether every inexact array leaf in ``tree`` is finite."""
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return True
    finite_flags = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return bool(np.asarray(jax.device_get(jnp.all(jnp.asarray(finite_flags)))))


def architecture_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Return a compact JSON architecture summary for checkpoint sidecars."""
    cfg = amortized_config(config)
    train_prior = _train_prior_jointly(cfg["prior"])
    scale_cfg = global_sed_scale_config(config)
    band_cfg = per_band_flux_calibration_config(config)
    optimized = ["encoder"]
    if train_prior:
        source = str(cfg["prior"].get("source", "joint_realnvp"))
        optimized.append("realnvp_prior" if "realnvp" in source else "flow_prior")
    if scale_cfg.enabled and scale_cfg.trainable:
        optimized.append("global_log_alpha_sed")
    if band_cfg.enabled and band_cfg.trainable:
        optimized.append("per_band_log_alpha")
    return {
        "encoder": {
            "type": cfg["encoder"].get("type", "gaussian_mlp"),
            "input_dim": int(cfg["encoder"].get("input_dim", 20)),
            "latent_dim": int(cfg["encoder"].get("latent_dim", 16)),
            "input_contract": (
                f"flux_{int(cfg['features'].get('n_flux_bands', 10))}"
                f"_then_err_{int(cfg['features'].get('n_error_bands', 10))}"
            ),
            "flux_transform": cfg["features"].get("flux_transform", "asinh"),
            "error_transform": cfg["features"].get("error_transform", "log"),
            "hidden_sizes": list(cfg["encoder"].get("hidden_sizes", [])),
            "activation": cfg["encoder"].get("activation", "gelu"),
            "posterior_family": str(cfg["encoder"].get("type", "gaussian_mlp")),
            "flow_family": cfg["encoder"].get("flow_family"),
            "flow_layers": int(cfg["encoder"].get("flow_layers", 0)),
            "flow_hidden_size": int(cfg["encoder"].get("flow_hidden_size", 0)),
        },
        "prior": {
            "type": cfg["prior"].get("type", "realnvp"),
            "source": cfg["prior"].get("source", "joint_realnvp"),
            "train_jointly": _train_prior_jointly(cfg["prior"]),
            "checkpoint": cfg["prior"].get("checkpoint"),
            "latent_dim": int(cfg["encoder"].get("latent_dim", 16)),
            "n_layers": int(cfg["prior"].get("n_layers", 8)),
            "hidden_size": int(cfg["prior"].get("hidden_size", 128)),
            "scale_clamp": float(cfg["prior"].get("scale_clamp", 0.05)),
            "shift_clamp": float(cfg["prior"].get("shift_clamp", 5.0)),
            "n_bins": int(cfg["prior"].get("n_bins", 16)),
            "tail_bound": float(cfg["prior"].get("tail_bound", 12.0)),
            "min_bin_width": float(cfg["prior"].get("min_bin_width", 1.0e-3)),
            "min_bin_height": float(cfg["prior"].get("min_bin_height", 1.0e-3)),
            "min_derivative": float(cfg["prior"].get("min_derivative", 1.0e-3)),
            "permutation": str(cfg["prior"].get("permutation", "roll")),
            "init": str(cfg["prior"].get("init", "default")),
            "init_scale": float(cfg["prior"].get("init_scale", 1.0)),
            "density": "exact_change_of_variables",
        },
        "global_sed_scale": {
            "enabled": scale_cfg.enabled,
            "mode": scale_cfg.mode,
            "trainable": scale_cfg.trainable,
            "parameterization": "log_alpha",
            "prior_sigma_log_alpha": scale_cfg.prior_sigma_log_alpha,
        },
        "per_band_flux_calibration": {
            "enabled": band_cfg.enabled,
            "mode": band_cfg.mode,
            "trainable": band_cfg.trainable,
            "parameterization": "log_alpha_per_band",
            "prior_sigma_log_alpha": band_cfg.prior_sigma_log_alpha,
            "prior_sigma_mag": band_cfg.prior_sigma_mag,
        },
        "data_parallel": {
            "requested": str(cfg["training"].get("data_parallel", "single")),
            "semantics": (
                "single uses the one-device step; auto uses pmap only when "
                "more than one local JAX device is visible; pmap requires "
                "multiple local devices and averages gradients with lax.pmean"
            ),
        },
        "decoder": {
            "type": "fixed_dsps",
            "trainable": False,
            "input": "theta_popcosmos_16",
            "output": f"flux_{int(cfg['features'].get('n_flux_bands', 10))}",
        },
        "objective": {
            "loss": "negative_elbo",
            "kl_estimator": "monte_carlo_logq_minus_logp",
            "likelihood": cfg["likelihood"].get("type", "student_t"),
            "jointly_optimized_components": optimized,
        },
    }


def _tree_l2_norm(tree) -> float:
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(tree) if eqx.is_inexact_array(leaf)
    ]
    if not leaves:
        return 0.0
    total = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return float(np.asarray(jax.device_get(jnp.sqrt(total))))


def _finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _effective_best_checkpoint_metric(
    configured: str,
    *,
    has_validation: bool,
) -> str:
    metric = str(configured or "").strip() or "validation_negative_loglike"
    if has_validation:
        return metric
    if metric.startswith("validation_"):
        return "train_loss"
    return metric


def _best_checkpoint_min_epoch(
    training_cfg: dict[str, Any],
    *,
    has_validation: bool,
) -> int:
    configured = training_cfg.get("best_checkpoint_min_epoch")
    if configured is None:
        configured = training_cfg.get("kl_annealing_epochs", 1) if has_validation else 1
    return max(1, int(configured))


def _checkpoint_metric_from_rows(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> float:
    column = _checkpoint_metric_column(metric_name)
    return _finite_mean([float(row.get(column, float("nan"))) for row in rows])


def _checkpoint_metric_column(metric_name: str) -> str:
    metric = str(metric_name)
    for prefix in ("validation_", "train_"):
        if metric.startswith(prefix):
            metric = metric[len(prefix) :]
            break
    aliases = {
        "nll": "negative_loglike",
        "negative_log_likelihood": "negative_loglike",
        "posterior_predictive_chi2": "posterior_predictive_chi2",
    }
    return aliases.get(metric, metric)


def _should_update_best_checkpoint(
    *,
    epoch: int,
    value: float,
    best_value: float,
    min_epoch: int,
) -> bool:
    return int(epoch) >= int(min_epoch) and np.isfinite(value) and value < best_value


def write_training_progress(
    out: Path,
    *,
    rows: list[dict[str, float]],
    epoch: int,
    best_loss: float,
    start_time: float,
    checkpoint_last: str,
    best_checkpoint_metric: str | None = None,
    best_checkpoint_epoch: int | None = None,
) -> None:
    if not rows:
        return
    first_loss = float(rows[0]["loss"])
    last_loss = float(rows[-1]["loss"])
    write_json(
        out / "training_progress.json",
        {
            "epoch": int(epoch),
            "steps": int(len(rows)),
            "first_loss": first_loss,
            "last_loss": last_loss,
            "best_loss": float(best_loss),
            "best_checkpoint_metric": best_checkpoint_metric,
            "best_checkpoint_epoch": (
                int(best_checkpoint_epoch)
                if best_checkpoint_epoch is not None
                else None
            ),
            "loss_improved_from_first": bool(
                np.isfinite(first_loss)
                and np.isfinite(best_loss)
                and best_loss < first_loss
            ),
            "elapsed_time_s": float(time.time() - start_time),
            "checkpoint_last": checkpoint_last,
            "latest_encoder_grad_norm": float(rows[-1].get("encoder_grad_norm", 0.0)),
            "latest_prior_grad_norm": float(rows[-1].get("prior_grad_norm", 0.0)),
            "updates_applied": int(
                sum(row.get("update_applied", 0.0) for row in _training_rows(rows))
            ),
            "updates_skipped": int(
                sum(
                    1.0 - row.get("update_applied", 0.0) for row in _training_rows(rows)
                )
            ),
        },
    )


def _expected_batch_count(limit: int | None, batch_size: int) -> int | None:
    if limit is None:
        return None
    if limit <= 0:
        return 0
    return int(np.ceil(float(limit) / float(max(batch_size, 1))))


def _effective_jax_batch_size(training_config: dict[str, Any], batch_size: int) -> int:
    requested = max(int(batch_size), 1)
    configured = training_config.get("jax_batch_size")
    if configured is None:
        return requested
    value = int(configured)
    if value <= 0:
        raise ValueError("amortized.training.jax_batch_size must be positive")
    return min(requested, value)


def _resolve_data_parallel_training(
    training_config: dict[str, Any],
    *,
    jax_batch_size: int,
) -> dict[str, Any]:
    requested = str(training_config.get("data_parallel", "single")).strip().lower()
    aliases = {"none": "single", "false": "single", "true": "auto"}
    requested = aliases.get(requested, requested)
    if requested not in {"single", "auto", "pmap"}:
        raise ValueError(
            "amortized.training.data_parallel must be one of: single, auto, pmap"
        )
    devices = tuple(jax.local_devices())
    n_devices = len(devices)
    if requested == "single":
        enabled = False
        effective = "single"
    elif requested == "auto":
        enabled = n_devices > 1
        effective = "pmap" if enabled else "single"
    else:
        if n_devices < 2:
            raise ValueError(
                "amortized.training.data_parallel='pmap' requires at least two "
                f"local JAX devices; visible devices={devices}"
            )
        enabled = True
        effective = "pmap"
    if enabled and int(jax_batch_size) % n_devices != 0:
        raise ValueError(
            "In pmap mode, amortized.training.jax_batch_size must be divisible "
            f"by the local device count: jax_batch_size={int(jax_batch_size)} "
            f"devices={n_devices}"
        )
    return {
        "requested": requested,
        "effective": effective,
        "enabled": bool(enabled),
        "devices": devices,
        "n_devices": int(n_devices),
        "global_batch_size": int(jax_batch_size),
        "per_device_batch_size": (
            int(jax_batch_size) // int(n_devices) if enabled else int(jax_batch_size)
        ),
    }


def _pad_epoch_order_for_data_parallel(
    order: np.ndarray,
    *,
    global_batch_size: int,
    enabled: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    order = np.asarray(order, dtype=np.int64)
    if not enabled:
        return order, 0
    if len(order) == 0:
        return order, 0
    remainder = len(order) % int(global_batch_size)
    if remainder == 0:
        return order, 0
    pad_count = int(global_batch_size) - int(remainder)
    padding = rng.choice(order, size=pad_count, replace=True)
    return np.concatenate([order, np.asarray(padding, dtype=np.int64)]), pad_count


def _shard_loss_batch(batch: LossBatch, n_devices: int) -> LossBatch:
    if int(n_devices) <= 1:
        return batch

    def shard_array(value):
        value = jnp.asarray(value)
        if value.shape[0] % int(n_devices) != 0:
            raise ValueError(
                "pmap training batch leading dimension must be divisible by "
                f"device count: shape={value.shape} devices={n_devices}"
            )
        return value.reshape(
            (int(n_devices), value.shape[0] // int(n_devices), *value.shape[1:])
        )

    return LossBatch(
        flux=shard_array(batch.flux),
        flux_err=shard_array(batch.flux_err),
        mask=shard_array(batch.mask),
        features=shard_array(batch.features),
        truth_theta=shard_array(batch.truth_theta),
    )


def _replicate_tree(tree, devices: tuple[Any, ...]):
    n_devices = len(tuple(devices))

    def replicate_leaf(leaf):
        if eqx.is_array(leaf):
            value = jnp.asarray(leaf)
            return jnp.broadcast_to(value, (n_devices, *value.shape))
        return leaf

    return jax.tree_util.tree_map(replicate_leaf, tree)


def _unreplicate_tree(tree):
    def first_leaf(leaf):
        if eqx.is_array(leaf):
            return leaf[0]
        return leaf

    return jax.tree_util.tree_map(first_leaf, tree)


def _unreplicate_scalar(value) -> float:
    arr = np.asarray(jax.device_get(value))
    if arr.shape:
        arr = arr.reshape(-1)[0]
    return float(arr)


def _unreplicate_bool(value) -> bool:
    arr = np.asarray(jax.device_get(value))
    if arr.shape:
        arr = arr.reshape(-1)[0]
    return bool(arr)


def _unreplicate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        name: jnp.asarray(np.asarray(jax.device_get(value)).reshape(-1)[0])
        for name, value in metrics.items()
    }


def _write_prior_predictive_training_diagnostics(
    out: Path,
    model: AmortizedModel,
    arrays,
    latent_spec: LatentSpec,
    context,
    model_args,
    parameter_names: tuple[str, ...],
    key,
    *,
    config: dict[str, Any],
    epoch: int,
    band_names: tuple[str, ...],
    calibration_config: dict[str, Any],
    n_prior_samples: int,
    n_observed: int,
    batch_size: int,
) -> None:
    n_prior_samples = max(1, int(n_prior_samples))
    batch_size = max(1, int(batch_size))
    prior_x = model.prior.sample(key, n_prior_samples)
    prior_flux_raw = []
    for start in range(0, n_prior_samples, batch_size):
        prior_flux_raw.append(
            model_flux_from_x(
                prior_x[start : start + batch_size],
                latent_spec,
                context,
                model_args,
                parameter_names,
            )
        )
    model_flux_raw = jnp.concatenate(prior_flux_raw, axis=0)
    scale_cfg = global_sed_scale_config(calibration_config)
    band_cfg = per_band_flux_calibration_config(calibration_config)
    model_flux = (
        apply_global_sed_scale_to_flux(model_flux_raw, model.sed_scale.log_alpha_sed)
        if scale_cfg.enabled
        else model_flux_raw
    )
    log_alpha_band = (
        model.band_calibration.log_alpha_band
        if band_cfg.enabled and model.band_calibration is not None
        else jnp.zeros((model_flux_raw.shape[-1],), dtype=model_flux_raw.dtype)
    )
    model_flux = (
        apply_per_band_flux_calibration_to_flux(model_flux, log_alpha_band)
        if band_cfg.enabled
        else model_flux
    )
    observed_flux = np.asarray(arrays.flux, dtype=float)
    if len(observed_flux) > int(n_observed):
        rng = np.random.default_rng(int(epoch))
        observed_flux = observed_flux[
            rng.choice(len(observed_flux), size=int(n_observed), replace=False)
        ]
    prior_flux = np.asarray(jax.device_get(model_flux), dtype=float)
    rows = _prior_predictive_color_distance_rows(
        observed_flux,
        prior_flux,
        band_names=band_names,
        epoch=epoch,
    )
    if not rows:
        return
    path = out / "training_prior_predictive_color_metrics.csv"
    frame = pd.DataFrame(rows)
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat(
            [previous.loc[previous["epoch"] != int(epoch)], frame],
            ignore_index=True,
        )
    frame = frame.sort_values(["epoch", "color"])
    frame.to_csv(path, index=False)
    _write_prior_predictive_color_plot(frame, out)


def _prior_predictive_color_distance_rows(
    observed_flux: np.ndarray,
    prior_flux: np.ndarray,
    *,
    band_names: tuple[str, ...],
    epoch: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    obs_log = np.log10(np.clip(np.asarray(observed_flux, dtype=float), 1.0e-300, None))
    prior_log = np.log10(np.clip(np.asarray(prior_flux, dtype=float), 1.0e-300, None))
    for index in range(len(band_names) - 1):
        obs_color = obs_log[:, index] - obs_log[:, index + 1]
        prior_color = prior_log[:, index] - prior_log[:, index + 1]
        obs_color = obs_color[np.isfinite(obs_color)]
        prior_color = prior_color[np.isfinite(prior_color)]
        if obs_color.size == 0 or prior_color.size == 0:
            continue
        q = np.linspace(0.05, 0.95, 19)
        obs_q = np.quantile(obs_color, q)
        prior_q = np.quantile(prior_color, q)
        rows.append(
            {
                "epoch": int(epoch),
                "color": f"{band_names[index]}-{band_names[index + 1]}",
                "quantile_l1": float(np.mean(np.abs(obs_q - prior_q))),
                "observed_median": float(np.median(obs_color)),
                "prior_median": float(np.median(prior_color)),
                "observed_std": float(np.std(obs_color)),
                "prior_std": float(np.std(prior_color)),
            }
        )
    return rows


def _write_prior_predictive_color_plot(frame: pd.DataFrame, out: Path) -> None:
    if frame.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    summary = (
        frame.groupby("epoch", sort=True)["quantile_l1"]
        .mean()
        .reset_index(name="mean_quantile_l1")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(
        summary["epoch"].to_numpy(dtype=float),
        summary["mean_quantile_l1"].to_numpy(dtype=float),
        marker="o",
        lw=1.8,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean adjacent-color quantile L1")
    ax.set_title("Prior predictive color distance")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "prior_predictive_color_distance.png", dpi=160)
    plt.close(fig)


def _should_write_training_snapshot(
    snapshot_config: dict[str, Any],
    epoch: int,
) -> bool:
    if not bool(snapshot_config.get("enabled", False)):
        return False
    every_epochs = int(snapshot_config.get("every_epochs", 5) or 0)
    if every_epochs <= 0:
        return False
    return int(epoch) > 0 and int(epoch) % every_epochs == 0


def _write_training_snapshot(
    out: Path,
    *,
    epoch: int,
    model: AmortizedModel,
    arrays,
    feature_stats: FeatureStats,
    latent_spec: LatentSpec,
    key,
    snapshot_config: dict[str, Any],
    config: dict[str, Any],
    kl_weight: float,
    prior_source: str,
    prior_train_jointly: bool,
    update_phase: str,
    deterministic: bool,
) -> None:
    snap = ensure_dir(out / "training_snapshots" / f"epoch_{int(epoch):04d}")
    n_available = int(arrays.flux.shape[0])
    n_objects = min(
        n_available,
        max(int(snapshot_config.get("n_validation_objects", 256)), 1),
    )
    seed = int(snapshot_config.get("seed", 260617)) + int(epoch)
    rng = np.random.default_rng(seed)
    if n_available > n_objects:
        order = np.sort(rng.choice(n_available, size=n_objects, replace=False))
    else:
        order = np.arange(n_available, dtype=int)

    flux = jnp.asarray(arrays.flux[order], dtype=jnp.float32)
    flux_err = jnp.asarray(arrays.flux_err[order], dtype=jnp.float32)
    features = make_encoder_features(flux, flux_err, feature_stats)
    mean, log_std = model.encoder(features)
    mean_np = np.asarray(jax.device_get(mean), dtype=np.float32)
    log_std_np = np.asarray(jax.device_get(log_std), dtype=np.float32)

    object_id = np.asarray(arrays.object_id[order])
    row_index = _snapshot_row_index(arrays, order)
    posterior_summary = pd.DataFrame(
        {
            "row_index": row_index,
            "object_id": object_id,
            "epoch": int(epoch),
        }
    )
    for index, name in enumerate(latent_spec.names):
        posterior_summary[f"x_mean_{name}"] = mean_np[:, index]
        posterior_summary[f"x_log_std_{name}"] = log_std_np[:, index]
    n_posterior_samples = (
        1
        if deterministic
        else max(int(snapshot_config.get("posterior_samples", 64)), 1)
    )
    key, posterior_key, prior_key = jax.random.split(key, 3)
    if deterministic:
        posterior_x = mean_np[None, ...]
    else:
        posterior_x = np.asarray(
            jax.device_get(
                sample_posterior(model, posterior_key, features, n_posterior_samples).x
            ),
            dtype=np.float32,
        )
    posterior_median = np.median(posterior_x, axis=0)
    theta_median = _theta_frame_from_x(posterior_median, latent_spec, prefix="")
    posterior_summary = pd.concat([posterior_summary, theta_median], axis=1)
    posterior_summary.to_parquet(snap / "posterior_summary.parquet", index=False)
    posterior_samples = _snapshot_sample_frame(
        posterior_x,
        latent_spec,
        row_index=row_index,
        object_id=object_id,
        epoch=epoch,
        kind="posterior_mean" if deterministic else "posterior_encoder_sample",
    )
    posterior_samples.to_parquet(snap / "posterior_samples.parquet", index=False)

    prior_sample_count = max(int(snapshot_config.get("prior_samples", 4096)), 1)
    prior_x = np.asarray(
        jax.device_get(model.prior.sample(prior_key, prior_sample_count)),
        dtype=np.float32,
    )
    prior_samples = _theta_frame_from_x(prior_x, latent_spec, prefix="")
    prior_samples.insert(0, "sample_id", np.arange(len(prior_samples), dtype=np.int64))
    prior_samples.insert(0, "epoch", int(epoch))
    prior_samples.to_parquet(snap / "prior_samples.parquet", index=False)

    truth_path, truth_reason = _write_snapshot_truth_frame(
        snap,
        arrays=arrays,
        order=order,
        latent_spec=latent_spec,
    )
    plots = []
    if bool(snapshot_config.get("write_corner", True)):
        max_corner_rows = int(snapshot_config.get("max_corner_rows", 4000))
        plots.extend(
            _write_training_snapshot_corner_plots(
                snap,
                posterior_samples=posterior_samples,
                prior_samples=prior_samples,
                truth_path=truth_path,
                max_rows=max_corner_rows,
            )
        )

    if bool(snapshot_config.get("write_jacobian_lens", False)):
        lens_dir = ensure_dir(snap / "jacobian_lens")
        write_json(
            lens_dir / "SKIPPED.json",
            {
                "reason": (
                    "training snapshots keep Jacobian diagnostics on the dedicated "
                    "sharded amortized-jacobian-lens-diffsky CLI path"
                ),
                "configured_objects": int(
                    snapshot_config.get("jacobian_lens_objects", 32)
                ),
            },
        )

    summary = {
        "epoch": int(epoch),
        "n_validation_objects": int(n_objects),
        "posterior_samples_per_object": int(n_posterior_samples),
        "prior_samples": int(prior_sample_count),
        "posterior_interpretation": (
            "deterministic_encoder_mean"
            if deterministic
            else (
                "bayesian_posterior_under_encoder_and_prior"
                if float(kl_weight) > 0.0
                else "encoder_distribution_not_bayesian"
            )
        ),
        "kl_definition": "E_q[logq - logp_beta]",
        "kl_direction": "q_to_p",
        "kl_weight": float(kl_weight),
        "prior_source": str(prior_source),
        "prior_train_jointly": bool(prior_train_jointly),
        "update_phase": str(update_phase),
        "truth_theta": {
            "path": truth_path.name if truth_path is not None else None,
            "reason": truth_reason,
        },
        "plots": plots,
        "decode_prior_flux": bool(snapshot_config.get("decode_prior_flux", False)),
        "decode_prior_flux_note": "not run in lightweight snapshot hook",
        "config_objective": dict(
            (config.get("amortized", {}) or {}).get("objective", {})
        ),
    }
    write_json(snap / "snapshot_metrics.json", summary)
    write_json(
        snap / "artifact_manifest.json",
        {
            "epoch": int(epoch),
            "files": sorted(path.name for path in snap.iterdir() if path.is_file()),
            "subdirectories": sorted(
                path.name for path in snap.iterdir() if path.is_dir()
            ),
        },
    )


def _snapshot_posterior_x(
    key,
    mean: np.ndarray,
    log_std: np.ndarray,
    *,
    n_samples: int,
    deterministic: bool,
) -> np.ndarray:
    if deterministic:
        return mean[None, ...]
    eps = np.asarray(
        jax.device_get(
            jax.random.normal(
                key,
                (int(n_samples),) + tuple(mean.shape),
                dtype=jnp.float32,
            )
        ),
        dtype=np.float32,
    )
    return mean[None, ...] + np.exp(log_std[None, ...]) * eps


def _snapshot_sample_frame(
    x_samples: np.ndarray,
    latent_spec: LatentSpec,
    *,
    row_index: np.ndarray,
    object_id: np.ndarray,
    epoch: int,
    kind: str,
) -> pd.DataFrame:
    n_samples, n_objects, latent_dim = x_samples.shape
    flat = x_samples.reshape(n_samples * n_objects, latent_dim)
    frame = _theta_frame_from_x(flat, latent_spec, prefix="")
    frame.insert(0, "sample_kind", str(kind))
    frame.insert(
        0,
        "sample_id",
        np.repeat(np.arange(n_samples, dtype=np.int64), n_objects),
    )
    frame.insert(0, "object_id", np.tile(object_id, n_samples))
    frame.insert(0, "row_index", np.tile(row_index, n_samples))
    frame.insert(0, "epoch", int(epoch))
    return frame


def _theta_frame_from_x(
    x: np.ndarray,
    latent_spec: LatentSpec,
    *,
    prefix: str,
) -> pd.DataFrame:
    theta = np.asarray(
        jax.device_get(x_to_theta(jnp.asarray(x, dtype=jnp.float32), latent_spec)),
        dtype=np.float32,
    )
    return pd.DataFrame(
        {
            f"{prefix}{name}": theta[:, index]
            for index, name in enumerate(latent_spec.names)
        }
    )


def _snapshot_row_index(arrays, order: np.ndarray) -> np.ndarray:
    if getattr(arrays, "row_index", None) is None:
        return np.asarray(order, dtype=np.int64)
    return np.asarray(arrays.row_index, dtype=np.int64)[order]


def _write_snapshot_truth_frame(
    snap: Path,
    *,
    arrays,
    order: np.ndarray,
    latent_spec: LatentSpec,
) -> tuple[Path | None, str]:
    truth = getattr(arrays, "truth", None)
    if not truth:
        return None, "photometry arrays do not carry truth theta columns"
    missing = [name for name in latent_spec.names if name not in truth]
    if missing:
        return None, "missing truth columns: " + ", ".join(missing[:8])
    frame = pd.DataFrame(
        {
            "row_index": _snapshot_row_index(arrays, order),
            "object_id": np.asarray(arrays.object_id[order]),
        }
    )
    for name in latent_spec.names:
        frame[name] = np.asarray(truth[name])[order]
    path = snap / "truth_theta.parquet"
    frame.to_parquet(path, index=False)
    return path, "available"


def _write_training_snapshot_corner_plots(
    snap: Path,
    *,
    posterior_samples: pd.DataFrame,
    prior_samples: pd.DataFrame,
    truth_path: Path | None,
    max_rows: int,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_json(
            snap / "corner_plot_metadata.json",
            {"plots": [], "reason": "matplotlib_unavailable"},
        )
        return []
    truth = pd.read_parquet(truth_path) if truth_path is not None else pd.DataFrame()
    groups = {
        "useful": _training_snapshot_useful_columns(
            posterior_samples,
            prior_samples,
            truth,
        ),
        "full18": _training_snapshot_full_columns(
            posterior_samples, prior_samples, truth
        ),
    }
    plots: list[str] = []
    metadata: list[dict[str, Any]] = []
    for suffix, columns in groups.items():
        if len(columns) < 2:
            metadata.append(
                {
                    "plot": f"corner_prior_posterior_truth_{suffix}.png",
                    "written": False,
                    "reason": "fewer_than_two_finite_columns",
                    "columns": columns,
                }
            )
            continue
        path = snap / f"corner_prior_posterior_truth_{suffix}.png"
        _write_overlay_corner_like(
            path,
            plt,
            columns=columns,
            posterior=posterior_samples,
            prior=prior_samples,
            truth=truth,
            max_rows=max_rows,
        )
        plots.append(path.name)
        metadata.append(
            {
                "plot": path.name,
                "written": True,
                "columns": columns,
                "max_rows": int(max_rows),
            }
        )
        if not truth.empty:
            truth_path_out = snap / f"corner_truth_only_{suffix}.png"
            _write_overlay_corner_like(
                truth_path_out,
                plt,
                columns=columns,
                posterior=pd.DataFrame(),
                prior=pd.DataFrame(),
                truth=truth,
                max_rows=max_rows,
            )
            plots.append(truth_path_out.name)
    write_json(snap / "corner_plot_metadata.json", {"plots": metadata})
    return plots


def _training_snapshot_full_columns(*frames: pd.DataFrame) -> list[str]:
    order = [
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
        "diffstar_lgmcrit",
        "diffstar_lgy_at_mcrit",
        "diffstar_indx_lo",
        "diffstar_indx_hi",
        "diffstar_lg_qt",
        "diffstar_qlglgdt",
        "diffstar_lg_drop",
        "diffstar_lg_rejuv",
        "diffmah_logm0",
        "diffmah_logtc",
        "diffmah_early_index",
        "diffmah_late_index",
        "diffmah_t_peak",
    ]
    return _finite_columns(order, frames)


def _training_snapshot_useful_columns(*frames: pd.DataFrame) -> list[str]:
    order = [
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
        "log10_sfr_at_obs",
        "log10_ssfr_at_obs",
        "diffstar_lgmcrit",
        "diffstar_lgy_at_mcrit",
        "diffmah_logm0",
        "diffmah_t_peak",
    ]
    return _finite_columns(order, frames)


def _finite_columns(order: list[str], frames: tuple[pd.DataFrame, ...]) -> list[str]:
    columns = []
    for column in order:
        for frame in frames:
            if frame.empty or column not in frame:
                continue
            values = frame[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size > 1 and np.nanmax(finite) > np.nanmin(finite):
                columns.append(column)
                break
    return columns


def _write_overlay_corner_like(
    path: Path,
    plt,
    *,
    columns: list[str],
    posterior: pd.DataFrame,
    prior: pd.DataFrame,
    truth: pd.DataFrame,
    max_rows: int,
) -> None:
    columns = columns[: min(len(columns), 8)]
    n_cols = len(columns)
    fig, axes = plt.subplots(
        n_cols,
        n_cols,
        figsize=(max(4.0, 1.45 * n_cols), max(4.0, 1.45 * n_cols)),
        squeeze=False,
    )
    datasets = [
        ("truth", truth, "#202020", 0.7),
        ("prior", prior, "#4c78a8", 0.45),
        ("posterior", posterior, "#f58518", 0.45),
    ]
    for i, y_name in enumerate(columns):
        for j, x_name in enumerate(columns):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue
            for label, frame, color, alpha in datasets:
                if frame.empty or x_name not in frame or y_name not in frame:
                    continue
                if i == j:
                    values = _snapshot_plot_values(frame, x_name, max_rows=max_rows)
                    if values.size == 0:
                        continue
                    ax.hist(
                        values,
                        bins=32,
                        density=True,
                        histtype="step",
                        color=color,
                        alpha=min(alpha + 0.25, 1.0),
                        label=label,
                    )
                else:
                    sub = (
                        frame[[x_name, y_name]]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
                    if sub.empty:
                        continue
                    if len(sub) > max_rows:
                        sub = sub.sample(int(max_rows), random_state=17)
                    ax.scatter(
                        sub[x_name].to_numpy(dtype=float),
                        sub[y_name].to_numpy(dtype=float),
                        s=2,
                        alpha=alpha,
                        color=color,
                        linewidths=0.0,
                    )
            if i == n_cols - 1:
                ax.set_xlabel(x_name, fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != j:
                ax.set_ylabel(y_name, fontsize=7)
            elif j > 0:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", labelsize=6, length=2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _snapshot_plot_values(
    frame: pd.DataFrame,
    column: str,
    *,
    max_rows: int,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > int(max_rows):
        rng = np.random.default_rng(17)
        indices = np.sort(rng.choice(len(values), size=int(max_rows), replace=False))
        values = values[indices]
    return values


def _training_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("split", "train") == "train"]


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _batch_progress_line(record: dict[str, float]) -> str:
    return (
        "[amortized] "
        f"epoch={int(record['epoch'])} batch={int(record['batch'])} "
        f"n={int(record['n_objects'])} loss={record['loss']:.6g} "
        f"nll={record['negative_loglike']:.6g} kl={record['kl_mc_mean']:.6g} "
        f"logq={record['logq_mean']:.6g} logprior={record['logprior_mean']:.6g} "
        f"enc_grad={record['encoder_grad_norm']:.3g} "
        f"prior_grad={record['prior_grad_norm']:.3g} "
        f"update={int(record.get('update_applied', 0.0))}"
    )


def _progress_bar(*, enabled: bool, total: int | None, desc: str, unit: str):
    if not enabled or tqdm is None:
        return _NullProgress()
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.05,
    )


class _NullProgress:
    def update(self, _: int = 1) -> None:
        return None

    def set_postfix(self, *_args, **_kwargs) -> None:
        return None

    def __enter__(self) -> _NullProgress:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None
