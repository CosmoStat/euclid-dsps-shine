"""Fail-closed configuration contract for the final FENIKS SC-ASMC-EM run."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .adaptive_bridge_smc import AdaptiveBridgeSMCConfig
from .adaptive_smc_training import AdaptiveSMCProposalConfig
from .config import amortized_config
from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    OBSERVED_SELECTION_CONTRACT,
    TARGET_POPULATION_CONTRACT,
)


@dataclass(frozen=True)
class SCASMCEMSchedule:
    outer_iterations: int
    sleep_epochs: int
    posterior_distillation_epochs: int
    active_bootstrap_count: int
    prior_mstep_max_steps: int
    posterior_updates_per_sleep: int
    ema_decay: float
    preflight_objects: int
    posterior_bank_shard_objects: int
    job_budget_seconds: float


@dataclass(frozen=True)
class SCASMCEMHierarchy:
    primary: AdaptiveBridgeSMCConfig
    fallback: AdaptiveBridgeSMCConfig
    extended: AdaptiveBridgeSMCConfig
    proposal: AdaptiveSMCProposalConfig
    minimum_is_ess_fraction: float
    maximum_is_weight: float


def sc_asmc_em_config_hash(config: dict[str, Any]) -> str:
    """Return the canonical hash binding all workflow configuration values."""
    normalized = copy.deepcopy(config)
    amortized_config(normalized)
    # Runtime loading removes every truth-related metadata field so neither the
    # redshift nor physical truth columns can enter an observed-only read. These
    # unused fields must therefore not distinguish checkpoint provenance.
    normalized["truth"] = {"parameter_columns": {}}
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def sc_asmc_em_schedule(config: dict[str, Any]) -> SCASMCEMSchedule:
    raw = _sc_block(config)
    sleep = dict(raw.get("sleep", {}) or {})
    distillation = dict(raw.get("q_distillation", {}) or {})
    bootstrap = dict(raw.get("active_bootstrap", {}) or {})
    mstep = dict(raw.get("prior_mstep", {}) or {})
    preflight = dict(raw.get("preflight", {}) or {})
    bank = dict(raw.get("posterior_bank", {}) or {})
    return SCASMCEMSchedule(
        outer_iterations=int(raw.get("outer_iterations", 2)),
        sleep_epochs=int(sleep.get("epochs", 12)),
        posterior_distillation_epochs=int(distillation.get("epochs", 3)),
        active_bootstrap_count=int(bootstrap.get("hard_objects", 192)),
        prior_mstep_max_steps=int(mstep.get("max_steps", 100)),
        posterior_updates_per_sleep=int(
            distillation.get("posterior_updates_per_sleep", 3)
        ),
        ema_decay=float(raw.get("ema_decay", 0.999)),
        preflight_objects=int(preflight.get("objects", 512)),
        posterior_bank_shard_objects=int(bank.get("objects_per_shard", 128)),
        job_budget_seconds=float(preflight.get("job_budget_seconds", 72_000.0)),
    )


def sc_asmc_em_hierarchy(config: dict[str, Any]) -> SCASMCEMHierarchy:
    raw = dict(_sc_block(config).get("e_step", {}) or {})
    proposal_raw = dict(raw.get("proposal", {}) or {})
    proposal = AdaptiveSMCProposalConfig(
        posterior_unit_fraction=float(proposal_raw.get("q_t1_fraction", 0.70)),
        posterior_tempered_fraction=float(proposal_raw.get("q_t1p5_fraction", 0.20)),
        posterior_temperature=float(proposal_raw.get("temperature", 1.50)),
        prior_fraction=float(proposal_raw.get("prior_fraction", 0.10)),
    )
    return SCASMCEMHierarchy(
        primary=_smc_config(raw.get("primary", {})),
        fallback=_smc_config(raw.get("fallback", {})),
        extended=_smc_config(raw.get("extended", {})),
        proposal=proposal,
        minimum_is_ess_fraction=float(raw.get("minimum_is_ess_fraction", 0.10)),
        maximum_is_weight=float(raw.get("maximum_is_weight", 0.80)),
    )


def validate_sc_asmc_em_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate every non-negotiable scientific and runtime choice."""
    normalized = copy.deepcopy(config)
    cfg = amortized_config(normalized)
    raw = _sc_block(config)
    schedule = sc_asmc_em_schedule(config)
    hierarchy = sc_asmc_em_hierarchy(config)
    errors: list[str] = []

    truth_columns = (config.get("truth", {}) or {}).get("parameter_columns", {}) or {}
    _require(not truth_columns, "truth.parameter_columns must be empty", errors)
    _require(
        raw.get("c0_scope_statement") == C0_SCOPE_STATEMENT,
        "the canonical C0 scope statement is required",
        errors,
    )

    performance = dict(raw.get("performance", {}) or {})
    _require(
        bool(performance.get("autotune_object_micro_batch")),
        "object micro-batch autotuning must be enabled",
        errors,
    )
    _require(
        0.85 <= float(performance.get("target_device_memory_fraction", 0.0)) <= 0.90,
        "device-memory target must be between 0.85 and 0.90",
        errors,
    )
    _require(
        float(performance.get("maximum_compiled_memory_fraction", 0.0)) == 0.90,
        "compiled-memory ceiling must equal 0.90",
        errors,
    )
    _require(
        int(performance.get("prefetch_host_batches", 0)) >= 1,
        "host batch prefetch must be enabled",
        errors,
    )
    _require(
        tuple(performance.get("fixed_particle_shapes", ())) == (64, 128),
        "fixed particle shapes must be K64 and K128",
        errors,
    )
    _require(
        performance.get("mixed_precision") is False,
        "mixed precision requires a separate equivalence-tested ablation",
        errors,
    )
    _require(
        raw.get("target_population") == TARGET_POPULATION_CONTRACT,
        "target_population must be p_eta(theta | C0)",
        errors,
    )
    _require(
        raw.get("observed_selection") == OBSERVED_SELECTION_CONTRACT,
        "only the observed r<25 selection may be corrected",
        errors,
    )
    _require(schedule.outer_iterations == 2, "outer_iterations must equal 2", errors)
    _require(10 <= schedule.sleep_epochs <= 15, "sleep epochs must be 10--15", errors)
    _require(
        2 <= schedule.posterior_distillation_epochs <= 5,
        "q distillation epochs must be 2--5",
        errors,
    )
    _require(
        128 <= schedule.active_bootstrap_count <= 256,
        "active bootstrap count must be 128--256",
        errors,
    )
    _require(
        50 <= schedule.prior_mstep_max_steps <= 200,
        "prior M-step max_steps must be 50--200",
        errors,
    )
    _require(
        schedule.posterior_updates_per_sleep == 3,
        "q distillation must use three bank updates per sleep update",
        errors,
    )
    _require(schedule.ema_decay == 0.999, "EMA decay must equal 0.999", errors)
    _require(
        schedule.preflight_objects == 512, "preflight must use 512 objects", errors
    )

    latent = cfg["latent"]
    _require(
        str(latent.get("normalization")) == "bounded_mixed_warp",
        "latent normalization must be bounded_mixed_warp",
        errors,
    )
    _require(
        not latent.get("normalization_checkpoint"),
        "truth-derived latent normalization checkpoints are forbidden",
        errors,
    )
    _require(
        len(latent.get("raw_scales", {})) == 15, "15 raw scales are required", errors
    )
    expected_latents = (
        "z_obs",
        "log10_stellar_mass",
        "log10_stellar_metallicity",
        "dust_av",
        "dust_delta",
        *(f"sfh_dlog_sfr_{index:02d}" for index in range(1, 11)),
    )
    _require(
        set(latent.get("raw_scales", {})) == set(expected_latents),
        "raw scales must cover exactly the 15 spline latents",
        errors,
    )
    _require(
        all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (latent.get("raw_scales", {}) or {}).values()
        ),
        "all raw scales must be finite and positive",
        errors,
    )
    warps = dict(latent.get("warps", {}) or {})
    _require(
        set(warps) == set(expected_latents),
        "bounded mixed warps must cover exactly the 15 spline latents",
        errors,
    )
    for name in expected_latents:
        warp = dict(warps.get(name, {}) or {})
        expected_family = "log1p" if name == "dust_av" else "asinh"
        expected_lambda = (
            0.2
            if name == "dust_av"
            else (
                0.5
                if name in {"z_obs", "log10_stellar_metallicity", "dust_delta"}
                else 1.0
            )
        )
        _require(
            str(warp.get("family")) == expected_family,
            f"bounded warp family mismatch for {name}",
            errors,
        )
        _require(
            float(warp.get("lambda", float("nan"))) == expected_lambda,
            f"bounded warp lambda mismatch for {name}",
            errors,
        )
        if expected_family == "asinh":
            _require(
                str(warp.get("center")) == "fit_initial",
                f"bounded warp center for {name} must be fit_initial",
                errors,
            )

    features = cfg["features"]
    _require(
        int(features.get("n_flux_bands", 0)) == 18, "18 flux bands required", errors
    )
    _require(
        int(features.get("n_error_bands", 0)) == 18, "18 error bands required", errors
    )
    append_mask = bool(features.get("append_mask", False))
    _require(
        str(features.get("flux_transform")) == "asinh",
        "flux features must use asinh",
        errors,
    )
    _require(
        str(features.get("error_transform")) == "log",
        "error features must use log",
        errors,
    )
    _require(
        float(features.get("error_epsilon", 0.0)) > 0.0,
        "error feature epsilon must be positive",
        errors,
    )
    expected_input_dim = 54 if append_mask else 36
    encoder = cfg["encoder"]
    _require(
        int(encoder.get("input_dim", 0)) == expected_input_dim,
        "encoder input dimension mismatch",
        errors,
    )
    _require(
        str(encoder.get("type")) == "conditional_flow",
        "q must be a conditional flow",
        errors,
    )
    _require(
        int(encoder.get("latent_dim", 0)) == 15,
        "q latent dimension must be 15",
        errors,
    )
    _require(
        str(encoder.get("flow_family")) == "realnvp",
        "q flow family must be RealNVP",
        errors,
    )
    _require(
        str(encoder.get("flow_output_space")) == "latent_x",
        "q flow must operate directly in latent x space",
        errors,
    )
    _require(
        str(encoder.get("context_encoder")) == "residual_photometry",
        "q requires direct residual photometry context",
        errors,
    )
    _require(
        int(encoder.get("residual_trunk_width", 0)) == 512,
        "q trunk width must be 512",
        errors,
    )
    _require(
        int(encoder.get("residual_blocks", 0)) == 3,
        "q must have three residual blocks",
        errors,
    )
    _require(
        int(encoder.get("residual_representation_width", 0)) == 256,
        "q representation must be 256",
        errors,
    )
    _require(
        int(encoder.get("residual_context_dim", 0)) == 128,
        "q context must be 128",
        errors,
    )
    _require(
        int(encoder.get("flow_layers", 0)) == 6, "q flow must have six layers", errors
    )
    _require(
        int(encoder.get("flow_hidden_size", 0)) == 256,
        "q coupling width must be 256",
        errors,
    )
    _require(
        float(encoder.get("flow_scale_clamp", 0.0)) == 0.45,
        "q scale clamp must be 0.45",
        errors,
    )
    _require(
        float(encoder.get("flow_shift_clamp", 0.0)) == 3.0,
        "q shift clamp must be 3.0",
        errors,
    )
    _require(
        float(encoder.get("flow_init_scale", 1.0)) == 0.0,
        "q flow must start at identity",
        errors,
    )
    _require(
        str(encoder.get("flow_permutation")) == "alternating_roll",
        "q requires identity-preserving roll permutations",
        errors,
    )
    _require(
        int(encoder.get("base_components", 0)) == 1,
        "q base_components must be one",
        errors,
    )
    _require(
        float(encoder.get("log_std_min")) == -4.0, "q log_std_min must be -4", errors
    )
    _require(
        float(encoder.get("log_std_max")) == 2.5, "q log_std_max must be 2.5", errors
    )
    _require(
        float(encoder.get("initial_log_std")) == 0.0,
        "q initial log_std must be zero",
        errors,
    )

    prior = cfg["prior"]
    _require(
        str(prior.get("source")) in {"joint_realnvp", "realnvp"},
        "prior must be a fresh RealNVP",
        errors,
    )
    _require(
        str(prior.get("type")) == "realnvp",
        "prior type must be RealNVP",
        errors,
    )
    _require(
        not prior.get("checkpoint"),
        "prior checkpoint warm starts are forbidden",
        errors,
    )
    _require(int(prior.get("n_layers", 0)) == 8, "prior must have eight layers", errors)
    _require(
        int(prior.get("hidden_size", 0)) == 256, "prior hidden size must be 256", errors
    )
    _require(
        float(prior.get("scale_clamp", 0.0)) == 0.25,
        "prior scale clamp must be 0.25",
        errors,
    )
    _require(
        float(prior.get("shift_clamp", 0.0)) == 3.0,
        "prior shift clamp must be 3.0",
        errors,
    )
    _require(
        str(prior.get("init")) == "identity", "prior must start at identity", errors
    )
    _require(
        float(prior.get("init_scale", 1.0)) == 0.0,
        "prior final layers must be exactly zero",
        errors,
    )
    _require(
        str(prior.get("permutation")) == "alternating_roll",
        "prior requires identity-preserving roll permutations",
        errors,
    )

    likelihood = cfg["likelihood"]
    _require(
        str(likelihood.get("type")) == "gaussian",
        "main likelihood must be Gaussian",
        errors,
    )
    sleep = dict(cfg["objective"].get("sleep", {}) or {})
    _require(bool(sleep.get("enabled")), "Gaussian sleep must be enabled", errors)
    _require(
        str(sleep.get("noise_family")) == "match_likelihood",
        "sleep noise must match Gaussian likelihood",
        errors,
    )
    sleep_selection = dict(sleep.get("selection", {}) or {})
    _require(
        bool(sleep_selection.get("enabled")),
        "sleep must apply observed selection after noise",
        errors,
    )
    _require(
        str(sleep_selection.get("band")) == "lsst_r"
        and float(sleep_selection.get("max_mag_ab", float("nan"))) == 25.0,
        "sleep selection must be observed r<25",
        errors,
    )
    selection = dict(cfg["objective"].get("selection_correction", {}) or {})
    _require(
        bool(selection.get("enabled")), "selection correction must be enabled", errors
    )
    _require(
        str(selection.get("gradient_estimator")) == "score_function",
        "selection requires score-function gradient",
        errors,
    )
    _require(
        str(selection.get("score_control_variate")) == "centered",
        "selection score control variate must be centered",
        errors,
    )
    _require(
        str(selection.get("survey_noise")) == "gaussian_m5",
        "selection completeness must use Gaussian PhotoErr",
        errors,
    )
    _require(
        str(selection.get("band")) == "lsst_r", "selection band must be lsst_r", errors
    )
    _require(
        float(selection.get("max_mag_ab")) == 25.0,
        "selection limit must be r<25",
        errors,
    )

    fractions = tuple(
        float(value) for value in hierarchy.proposal.normalized_fractions()
    )
    _require(
        all(
            math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-6)
            for value, expected in zip(fractions, (0.7, 0.2, 0.1), strict=True)
        ),
        "proposal fractions must be 0.70/0.20/0.10",
        errors,
    )
    _require(
        hierarchy.proposal.posterior_temperature == 1.5,
        "proposal temperature must be 1.5",
        errors,
    )
    _require(
        hierarchy.minimum_is_ess_fraction == 0.10, "IS ESS gate must be 0.10", errors
    )
    _require(
        hierarchy.maximum_is_weight == 0.80, "IS max-weight gate must be 0.80", errors
    )
    _validate_smc_level("primary", hierarchy.primary, 64, 16, 0.30, errors)
    _validate_smc_level("fallback", hierarchy.fallback, 128, 32, 0.15, errors)
    _validate_smc_level("extended", hierarchy.extended, 128, 48, 0.15, errors)

    preflight = dict(raw.get("preflight", {}) or {})
    _require(
        float(preflight.get("minimum_resolved_fraction", 0.0)) == 0.95,
        "preflight resolved gate must be 0.95",
        errors,
    )
    _require(
        float(preflight.get("maximum_unresolved_fraction", 1.0)) == 0.05,
        "preflight unresolved gate must be 0.05",
        errors,
    )
    _require(
        float(preflight.get("maximum_extended_fraction", 1.0)) == 0.15,
        "preflight extended gate must be 0.15",
        errors,
    )
    _require(
        int(preflight.get("maximum_attempts", 0)) == 2,
        "preflight may run exactly twice",
        errors,
    )
    _require(
        float(preflight.get("job_budget_seconds", 0.0)) > 0.0,
        "preflight job budget must be positive",
        errors,
    )
    _require(
        0.0
        <= float(preflight.get("projected_non_estep_overhead_fraction", -1.0))
        <= 0.5,
        "preflight non-E-step overhead must be in [0, 0.5]",
        errors,
    )

    mstep = dict(raw.get("prior_mstep", {}) or {})
    _require(
        float(mstep.get("learning_rate", 0.0)) == 1.0e-5,
        "prior learning rate must be 1e-5",
        errors,
    )
    _require(
        float(mstep.get("gradient_clip_norm", 0.0)) == 5.0,
        "prior gradient clip must be 5",
        errors,
    )
    sleep_schedule = dict(raw.get("sleep", {}) or {})
    _require(
        str(sleep_schedule.get("optimizer")) == "adamw",
        "sleep optimizer must be AdamW",
        errors,
    )
    _require(
        float(sleep_schedule.get("peak_learning_rate", 0.0)) == 2.0e-4,
        "sleep peak learning rate must be 2e-4",
        errors,
    )
    _require(
        float(sleep_schedule.get("final_learning_rate", 0.0)) == 2.0e-5,
        "sleep final learning rate must be 2e-5",
        errors,
    )
    _require(
        float(sleep_schedule.get("warmup_fraction", 0.0)) == 0.05,
        "sleep warmup fraction must be 0.05",
        errors,
    )
    _require(
        float(sleep_schedule.get("weight_decay", 0.0)) == 1.0e-6,
        "sleep weight decay must be 1e-6",
        errors,
    )
    _require(
        float(sleep_schedule.get("gradient_clip_norm", 0.0)) == 10.0,
        "sleep gradient clip must be 10",
        errors,
    )
    distillation = dict(raw.get("q_distillation", {}) or {})
    _require(
        float(distillation.get("learning_rate", 0.0)) == 2.0e-5,
        "q distillation learning rate must be 2e-5",
        errors,
    )
    _require(
        float(distillation.get("gradient_clip_norm", 0.0)) in {10.0, 20.0},
        "q distillation gradient clip must be 10 or 20",
        errors,
    )
    _require(
        float(distillation.get("warmup_fraction", 0.0)) == 0.05,
        "q distillation warmup fraction must be 0.05",
        errors,
    )
    _require(
        float(distillation.get("ema_decay", 0.0)) == 0.999,
        "q distillation EMA decay must be 0.999",
        errors,
    )

    calibrations = config.get("calibration", {}) or {}
    for name, value in calibrations.items():
        _require(
            not bool((value or {}).get("trainable", False)),
            f"calibration {name} must be frozen",
            errors,
        )
    forbidden = str(raw).lower()
    _require(
        "nuts" not in forbidden, "NUTS is forbidden in the training workflow", errors
    )
    _require("warm_start" not in forbidden, "warm starts are forbidden", errors)

    if errors:
        raise ValueError("invalid SC-ASMC-EM configuration:\n- " + "\n- ".join(errors))
    return {
        "status": "valid",
        "truth_used": False,
        "input_dim": expected_input_dim,
        "schedule": schedule,
        "hierarchy": hierarchy,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": TARGET_POPULATION_CONTRACT,
        "observed_selection": OBSERVED_SELECTION_CONTRACT,
    }


def _sc_block(config: dict[str, Any]) -> dict[str, Any]:
    raw = (config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("amortized.sc_asmc_em must be a mapping")
    return dict(raw)


def _smc_config(raw: Any) -> AdaptiveBridgeSMCConfig:
    values = dict(raw or {})
    return AdaptiveBridgeSMCConfig(
        n_particles=int(values.get("n_particles", 64)),
        target_conditional_ess_fraction=float(
            values.get("target_conditional_ess", 0.75)
        ),
        resample_ess_fraction=float(values.get("resample_ess_fraction", 0.50)),
        max_stages=int(values.get("max_stages", 16)),
        steps_after_resample=int(values.get("steps_after_resample", 3)),
        final_steps_at_beta1=int(values.get("final_steps_at_beta1", 1)),
        rw_scale=float(values.get("rw_scale", 0.30)),
        rw_adapt_target_acceptance=float(
            values.get("rw_adapt_target_acceptance", 0.30)
        ),
        rw_adapt_rate=float(values.get("rw_adapt_rate", 1.0)),
        rw_scale_min=float(values.get("rw_scale_min", 0.05)),
        rw_scale_max=float(values.get("rw_scale_max", 1.0)),
        hard_final_ess_fraction=float(values.get("hard_final_ess_fraction", 0.30)),
        hard_min_mutation_acceptance=float(
            values.get("hard_min_mutation_acceptance", 0.10)
        ),
        hard_min_ancestor_ess_fraction=float(
            values.get("hard_min_ancestor_ess_fraction", 0.05)
        ),
        hard_min_epsilon_squared_jump=float(
            values.get("hard_min_epsilon_squared_jump", 1.0e-4)
        ),
        bisection_steps=int(values.get("bisection_steps", 32)),
    )


def _validate_smc_level(
    name: str,
    value: AdaptiveBridgeSMCConfig,
    particles: int,
    stages: int,
    rw_scale: float,
    errors: list[str],
) -> None:
    _require(
        value.n_particles == particles, f"{name} SMC requires K={particles}", errors
    )
    _require(value.max_stages == stages, f"{name} SMC requires {stages} stages", errors)
    _require(
        value.rw_scale == rw_scale, f"{name} SMC requires rw_scale={rw_scale}", errors
    )
    _require(
        value.target_conditional_ess_fraction == 0.75,
        f"{name} SMC target CESS must be 0.75",
        errors,
    )
    _require(
        value.resample_ess_fraction == 0.50,
        f"{name} SMC resample ESS must be 0.50",
        errors,
    )


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)
