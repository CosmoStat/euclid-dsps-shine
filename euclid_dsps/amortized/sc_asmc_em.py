"""Contracts and orchestration helpers for selection-corrected amortized SMC-EM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    OBSERVED_SELECTION_CONTRACT,
    POSTERIOR_METHOD_CODES,
    TARGET_POPULATION_CONTRACT,
)

GENERALIZED_EM_PHASES = (
    "sleep_bootstrap",
    "budget_preflight",
    "e_step_1",
    "prior_m_step_1",
    "bank_reweight_1",
    "q_distillation_1",
    "e_step_2",
    "prior_m_step_2",
    "optional_q_distillation_2",
    "final_report",
)


@dataclass(frozen=True)
class HierarchyDispatch:
    method: np.ndarray
    resolved: np.ndarray
    primary_attempted: np.ndarray
    fallback_attempted: np.ndarray
    extended_attempted: np.ndarray

    def method_names(self) -> np.ndarray:
        names = np.empty(len(self.method), dtype=object)
        reverse = {value: key for key, value in POSTERIOR_METHOD_CODES.items()}
        for code, name in reverse.items():
            names[np.asarray(self.method) == code] = name
        return names.astype(str)


@dataclass(frozen=True)
class PreflightGate:
    status: str
    continue_full_catalogue: bool
    active_bootstrap_required: bool
    checks: dict[str, bool]
    metrics: dict[str, float]
    method_counts: dict[str, int]
    projected_full_run_wall_seconds: float
    attempt: int


@dataclass(frozen=True)
class PhaseIsolationContract:
    phase: str
    q_frozen: bool
    prior_frozen: bool
    posterior_bank_frozen: bool
    trainable_components: tuple[str, ...]


PHASE_ISOLATION = {
    "sleep_bootstrap": PhaseIsolationContract(
        "sleep_bootstrap", False, True, True, ("q",)
    ),
    "e_step": PhaseIsolationContract("e_step", True, True, True, ()),
    "q_distillation": PhaseIsolationContract(
        "q_distillation", False, True, True, ("q",)
    ),
    "prior_m_step": PhaseIsolationContract(
        "prior_m_step", True, False, True, ("prior",)
    ),
}


def dispatch_posterior_hierarchy(
    direct_is_accepted: np.ndarray,
    primary_succeeded: np.ndarray,
    fallback_succeeded: np.ndarray,
    extended_succeeded: np.ndarray,
) -> HierarchyDispatch:
    """Assign exactly one method while applying expensive kernels only to failures."""
    direct = np.asarray(direct_is_accepted, dtype=bool)
    primary_success = np.asarray(primary_succeeded, dtype=bool)
    fallback_success = np.asarray(fallback_succeeded, dtype=bool)
    extended_success = np.asarray(extended_succeeded, dtype=bool)
    if not (
        direct.shape
        == primary_success.shape
        == fallback_success.shape
        == extended_success.shape
    ):
        raise ValueError("posterior hierarchy flags must have identical shapes")
    primary_attempted = ~direct
    primary_resolved = primary_attempted & primary_success
    fallback_attempted = primary_attempted & ~primary_success
    fallback_resolved = fallback_attempted & fallback_success
    extended_attempted = fallback_attempted & ~fallback_success
    extended_resolved = extended_attempted & extended_success
    resolved = direct | primary_resolved | fallback_resolved | extended_resolved
    method = np.full(direct.shape, POSTERIOR_METHOD_CODES["unresolved"], dtype=np.int8)
    method[direct] = POSTERIOR_METHOD_CODES["IS"]
    method[primary_resolved] = POSTERIOR_METHOD_CODES["primary SMC"]
    method[fallback_resolved] = POSTERIOR_METHOD_CODES["fallback SMC"]
    method[extended_resolved] = POSTERIOR_METHOD_CODES["extended SMC"]
    return HierarchyDispatch(
        method=method,
        resolved=resolved,
        primary_attempted=primary_attempted,
        fallback_attempted=fallback_attempted,
        extended_attempted=extended_attempted,
    )


def evaluate_budget_preflight(
    dispatch: HierarchyDispatch,
    *,
    elapsed_seconds: float,
    dsps_evaluations: np.ndarray,
    stage_count: np.ndarray,
    mutation_acceptance: np.ndarray,
    ancestry_ess: np.ndarray,
    movement_squared: np.ndarray,
    beta_final: np.ndarray,
    full_catalogue_objects: int,
    e_step_iterations: int = 2,
    parallel_shards: int = 1,
    job_budget_seconds: float,
    non_estep_overhead_fraction: float = 0.20,
    attempt: int = 1,
) -> PreflightGate:
    """Apply the fail-closed 512-object cost and resolution gate."""
    n_objects = int(len(dispatch.method))
    if n_objects <= 0:
        raise ValueError("preflight requires at least one object")
    if int(e_step_iterations) != 2:
        raise ValueError("SC-ASMC-EM requires exactly two E-step iterations")
    if int(parallel_shards) <= 0 or int(full_catalogue_objects) <= 0:
        raise ValueError("preflight catalogue and shard counts must be positive")
    if not 0.0 <= float(non_estep_overhead_fraction) <= 1.0:
        raise ValueError("preflight non-E-step overhead fraction must be in [0, 1]")
    if not np.isfinite(elapsed_seconds) or float(elapsed_seconds) <= 0.0:
        raise ValueError("preflight elapsed_seconds must be finite and positive")
    method_counts = {
        name: int(np.sum(dispatch.method == code))
        for name, code in POSTERIOR_METHOD_CODES.items()
    }
    resolved_fraction = float(np.mean(dispatch.resolved))
    unresolved_fraction = 1.0 - resolved_fraction
    extended_fraction = float(np.mean(dispatch.extended_attempted))
    projected_estep = (
        float(elapsed_seconds)
        / n_objects
        * int(full_catalogue_objects)
        * int(e_step_iterations)
        / int(parallel_shards)
    )
    projected = projected_estep * (1.0 + float(non_estep_overhead_fraction))
    evaluations = np.asarray(dsps_evaluations, dtype=float)
    stages = np.asarray(stage_count, dtype=float)
    acceptance = np.asarray(mutation_acceptance, dtype=float)
    ancestry = np.asarray(ancestry_ess, dtype=float)
    movement = np.asarray(movement_squared, dtype=float)
    beta = np.asarray(beta_final, dtype=float)
    for name, values in {
        "dsps_evaluations": evaluations,
        "stage_count": stages,
        "mutation_acceptance": acceptance,
        "ancestry_ess": ancestry,
        "movement_squared": movement,
        "beta_final": beta,
    }.items():
        if values.shape != (n_objects,):
            raise ValueError(f"preflight {name} must have shape {(n_objects,)}")
    checks = {
        "resolved_fraction_at_least_0p95": resolved_fraction >= 0.95,
        "unresolved_fraction_at_most_0p05": unresolved_fraction <= 0.05,
        "extended_fraction_at_most_0p15": extended_fraction <= 0.15,
        "projected_cost_within_job_budget": projected <= float(job_budget_seconds),
        "diagnostics_finite": bool(
            np.all(np.isfinite(evaluations))
            and np.all(np.isfinite(stages))
            and np.all(np.isfinite(beta))
        ),
    }
    passed = all(checks.values())
    can_bootstrap = not passed and int(attempt) == 1
    status = "PASS" if passed else ("ACTIVE_BOOTSTRAP" if can_bootstrap else "ABORT")
    finite_acceptance = acceptance[np.isfinite(acceptance)]
    finite_ancestry = ancestry[np.isfinite(ancestry)]
    finite_movement = movement[np.isfinite(movement)]
    metrics = {
        "objects": float(n_objects),
        "resolved_fraction": resolved_fraction,
        "unresolved_fraction": unresolved_fraction,
        "direct_is_fraction": method_counts["IS"] / n_objects,
        "primary_smc_fraction": method_counts["primary SMC"] / n_objects,
        "fallback_smc_fraction": method_counts["fallback SMC"] / n_objects,
        "extended_smc_fraction": method_counts["extended SMC"] / n_objects,
        "beta_one_fraction": float(np.mean(beta >= 1.0 - 1.0e-6)),
        "median_stage_count": float(np.median(stages)),
        "median_mutation_acceptance": (
            float(np.median(finite_acceptance))
            if finite_acceptance.size
            else float("nan")
        ),
        "median_ancestry_ess": (
            float(np.median(finite_ancestry)) if finite_ancestry.size else float("nan")
        ),
        "median_movement_squared": (
            float(np.median(finite_movement)) if finite_movement.size else float("nan")
        ),
        "mean_dsps_evaluations_per_object": float(np.mean(evaluations)),
        "projected_two_estep_wall_seconds": projected_estep,
        "projected_non_estep_overhead_fraction": float(non_estep_overhead_fraction),
        "projected_full_run_wall_seconds": projected,
    }
    return PreflightGate(
        status=status,
        continue_full_catalogue=passed,
        active_bootstrap_required=can_bootstrap,
        checks=checks,
        metrics=metrics,
        method_counts=method_counts,
        projected_full_run_wall_seconds=projected,
        attempt=int(attempt),
    )


def select_active_bootstrap_rows(
    row_index: np.ndarray,
    dispatch: HierarchyDispatch,
    *,
    ess_fraction: np.ndarray,
    max_weight: np.ndarray,
    stage_count: np.ndarray,
    count: int,
) -> np.ndarray:
    """Select 128--256 hardest preflight objects without truth information."""
    if not 128 <= int(count) <= 256:
        raise ValueError("active bootstrap count must be between 128 and 256")
    rows = np.asarray(row_index, dtype=np.int64)
    if len(rows) < int(count):
        raise ValueError("active bootstrap count exceeds preflight cohort")
    ess = np.asarray(ess_fraction, dtype=float)
    maximum = np.asarray(max_weight, dtype=float)
    stages = np.asarray(stage_count, dtype=float)
    if not (rows.shape == ess.shape == maximum.shape == stages.shape):
        raise ValueError("active bootstrap diagnostics must share object shape")
    unresolved = ~np.asarray(dispatch.resolved, dtype=bool)
    extended = np.asarray(dispatch.extended_attempted, dtype=bool)
    normalized_stages = stages / max(float(np.nanmax(stages)), 1.0)
    hardness = (
        100.0 * unresolved.astype(float)
        + 10.0 * extended.astype(float)
        + np.nan_to_num(1.0 - ess, nan=1.0)
        + np.nan_to_num(maximum, nan=1.0)
        + normalized_stages
    )
    order = np.lexsort((rows, -hardness))
    return np.sort(rows[order[: int(count)]])


def stratified_preflight_indices(
    row_index: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    r_band_index: int,
    flux_limit: float,
    count: int = 512,
    seed: int = 260824,
) -> np.ndarray:
    """Build the observed-only preflight cohort from four requested strata."""
    rows = np.asarray(row_index, dtype=np.int64)
    flux_values = np.asarray(flux, dtype=float)
    errors = np.asarray(flux_err, dtype=float)
    if flux_values.shape != errors.shape or flux_values.ndim != 2:
        raise ValueError("preflight flux and errors must share shape [objects, bands]")
    if flux_values.shape[0] != len(rows):
        raise ValueError("preflight row indices do not match photometry")
    if not 0 <= int(r_band_index) < flux_values.shape[1]:
        raise ValueError("preflight r_band_index is out of bounds")
    requested = min(int(count), len(rows))
    if requested <= 0:
        raise ValueError("preflight count must be positive")
    safe_errors = np.maximum(errors, np.finfo(float).tiny)
    r_flux = flux_values[:, int(r_band_index)]
    r_error = safe_errors[:, int(r_band_index)]
    margin = np.abs((r_flux - float(flux_limit)) / r_error)
    snr = r_flux / r_error
    error_level = np.nanmedian(np.log(safe_errors), axis=1)
    reference = max(0, int(r_band_index) - 1)
    following = min(flux_values.shape[1] - 1, int(r_band_index) + 1)
    color = np.arcsinh(flux_values[:, reference] / safe_errors[:, reference])
    color -= np.arcsinh(flux_values[:, following] / safe_errors[:, following])
    codes = np.stack(
        tuple(
            _quantile_codes(value, bins=4)
            for value in (margin, snr, error_level, color)
        ),
        axis=1,
    )
    strata: dict[tuple[int, ...], list[int]] = {}
    rng = np.random.default_rng(int(seed))
    for index, code in enumerate(codes):
        strata.setdefault(tuple(int(item) for item in code), []).append(index)
    queues = []
    for key in sorted(strata):
        values = np.asarray(strata[key], dtype=np.int64)
        queues.append(list(rng.permutation(values)))
    chosen: list[int] = []
    while len(chosen) < requested and any(queues):
        next_queues = []
        for queue in queues:
            if queue and len(chosen) < requested:
                chosen.append(int(queue.pop()))
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return np.sort(rows[np.asarray(chosen, dtype=np.int64)])


def sc_asmc_parameter_count(*, input_dim: int, latent_dim: int = 15) -> dict[str, int]:
    """Return the exact trainable count for the configured residual-q/prior family."""
    if int(latent_dim) != 15:
        raise ValueError("the final FENIKS SC-ASMC-EM contract has 15 latents")
    first = int(input_dim) * 512 + 512
    residual = 3 * (2 * 512 + 2 * (512 * 512 + 512))
    representation = 512 * 256 + 256
    heads = 2 * (256 * 15 + 15) + (256 * 128 + 128)
    q_trunk = first + residual + representation + heads
    q_flow_layer = (15 + 128) * 256 + 256
    q_flow_layer += 256 * 256 + 256
    q_flow_layer += 256 * (2 * 15) + 2 * 15
    q_flow = 6 * q_flow_layer
    prior_network = (15 * 256 + 256) + (256 * 256 + 256) + (256 * 15 + 15)
    prior = 8 * 2 * prior_network
    return {
        "q_trunk_and_heads": q_trunk,
        "q_conditional_realnvp": q_flow,
        "q_total": q_trunk + q_flow,
        "prior_total": prior,
        "total": q_trunk + q_flow + prior,
    }


def validate_phase_isolation(
    phase: str,
    *,
    q_frozen: bool,
    prior_frozen: bool,
    posterior_bank_frozen: bool,
    trainable_components: tuple[str, ...],
) -> None:
    try:
        contract = PHASE_ISOLATION[str(phase)]
    except KeyError as error:
        raise ValueError(
            f"unknown SC-ASMC-EM phase isolation contract: {phase}"
        ) from error
    actual = PhaseIsolationContract(
        str(phase),
        bool(q_frozen),
        bool(prior_frozen),
        bool(posterior_bank_frozen),
        tuple(trainable_components),
    )
    if actual != contract:
        raise ValueError(
            f"phase isolation violation: expected {contract}, got {actual}"
        )


def write_phase_receipt(
    path: str | Path,
    *,
    phase: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if phase not in GENERALIZED_EM_PHASES:
        raise ValueError(f"unknown generalized-EM phase: {phase}")
    receipt = {
        "status": "complete",
        "phase": phase,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": TARGET_POPULATION_CONTRACT,
        "observed_selection": OBSERVED_SELECTION_CONTRACT,
        "truth_used_for_training_or_selection": False,
        **payload,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return receipt


def generalized_em_pseudocode() -> tuple[str, ...]:
    """Machine-readable exactly-two-iteration generalized-EM contract."""
    return (
        "train q0 from scratch on selected Gaussian sleep with p0 frozen",
        "run the 512-object preflight; allow one bounded active bootstrap",
        "freeze q0,p0 and persist full-catalogue bank B0",
        "freeze B0,q0 and update p0->p1 with data+log(alpha)+trust",
        "reweight B0 by p1/p0 and refresh only low-ESS objects",
        "freeze B1,p1 and distill q0->q1 with 3 bank updates per sleep replay",
        "freeze q1,p1; rerun IS on all and SMC-refresh only weak objects",
        "freeze B2,q1 and update p1->p2 with data+log(alpha)+trust",
        "optionally distill q2 briefly, then stop after EM iteration 2",
    )


def _quantile_codes(values: np.ndarray, *, bins: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    safe = np.where(
        finite, array, np.nanmedian(array[finite]) if np.any(finite) else 0.0
    )
    edges = np.unique(np.quantile(safe, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) <= 2:
        return np.zeros(len(array), dtype=np.int8)
    return np.searchsorted(edges[1:-1], safe, side="right").astype(np.int8)
