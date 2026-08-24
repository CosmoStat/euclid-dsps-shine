"""Fail-closed validation and final receipt for SC-ASMC-EM."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .posterior_bank import (
    C0_SCOPE_STATEMENT,
    OBSERVED_SELECTION_CONTRACT,
    TARGET_POPULATION_CONTRACT,
    is_posterior_bank_shard_complete,
    sha256_file,
    validate_posterior_bank_manifest_provenance,
)
from .sc_asmc_config import sc_asmc_em_config_hash, validate_sc_asmc_em_config
from .sc_asmc_manifest import validate_sc_asmc_manifest
from .sc_asmc_report import summarize_posterior_bank


def validate_production_smoke_gate(
    smoke_root: str | Path,
    config: dict[str, Any],
    catalogue_path: str | Path,
) -> dict[str, Any]:
    """Require an immutable four-device smoke for the exact production inputs."""
    validate_sc_asmc_em_config(config)
    root = Path(smoke_root).resolve()
    catalogue = Path(catalogue_path).resolve()
    receipt_path = root / "smoke" / "SMOKE_PASS.json"
    marker_path = root / "smoke" / "SMOKE_PASS"
    receipt = _required_receipt(receipt_path)
    if marker_path.read_text(encoding="utf-8").strip() != sha256_file(receipt_path):
        raise ValueError("4-H100 smoke marker does not bind its receipt")
    _validate_phase_receipt("four_device_smoke", receipt)
    selection = receipt.get("selection_gradient") or {}
    if not selection.get("finite") or not selection.get("nonzero"):
        raise ValueError("4-H100 smoke selection gradient did not pass")
    if len(receipt.get("devices") or []) != 4:
        raise ValueError("smoke receipt does not prove four local devices")
    if receipt.get("dataset_sha256") != sha256_file(catalogue):
        raise ValueError("smoke dataset differs from production dataset")
    if receipt.get("workflow_config_hash") != sc_asmc_em_config_hash(config):
        raise ValueError("smoke configuration differs from production configuration")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if receipt.get("code_commit") != commit:
        raise ValueError("smoke code commit differs from production commit")
    bank = Path(receipt.get("posterior_bank", ""))
    if not bank.is_file() or receipt.get("posterior_bank_sha256") != sha256_file(bank):
        raise ValueError("smoke posterior bank hash mismatch")
    manifest = root / "manifest" / "run_manifest.json"
    if not manifest.is_file() or receipt.get("run_manifest_sha256") != sha256_file(
        manifest
    ):
        raise ValueError("smoke run manifest hash mismatch")
    return receipt


def validate_and_write_final_receipt(
    config: dict[str, Any],
    *,
    run_root: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate every frozen phase and write the publication-facing receipt."""
    validate_sc_asmc_em_config(config)
    root = Path(run_root)
    manifest_path = root / "manifest" / "run_manifest.json"
    manifest = validate_sc_asmc_manifest(manifest_path)
    if manifest.get("config_sha256") != sc_asmc_em_config_hash(config):
        raise ValueError("run manifest workflow configuration hash mismatch")
    _validate_no_truth_config(config)
    if manifest["c0_scope_statement"] != C0_SCOPE_STATEMENT:
        raise ValueError("run manifest has a noncanonical C0 scope statement")
    if manifest["target_population"] != TARGET_POPULATION_CONTRACT:
        raise ValueError("run manifest target population is not p_eta(theta | C0)")
    if manifest["observed_selection"] != OBSERVED_SELECTION_CONTRACT:
        raise ValueError("run manifest selection event is not observed r<25")
    if manifest.get("truth_columns_requested") != []:
        raise ValueError("run manifest requested truth columns")

    training = _required_receipt(root / "TRAINING_COMPLETE.json")
    if int(training.get("outer_iterations", -1)) != 2:
        raise ValueError("final training marker must contain exactly two EM iterations")
    if training.get("truth_used") is not False:
        raise ValueError("training completion marker does not prove no-truth execution")
    if list(root.glob("mstep3*")) or list(root.glob("distill3*")):
        raise ValueError("run root contains a forbidden third EM iteration")
    preflight = _required_receipt(root / "preflight" / "PREFLIGHT_PASS.json")
    if not preflight.get("continue_full_catalogue"):
        raise ValueError("integrated preflight did not authorize the full catalogue")

    phase_receipts = {
        "initialization": _required_receipt(
            root / "initialization" / "initialization_receipt.json"
        ),
        "sleep": _required_receipt(root / "sleep" / "sleep_receipt.json"),
        "preflight": preflight,
        "prior_mstep_1": _required_receipt(
            root / "mstep1" / "prior_mstep_1_receipt.json"
        ),
        "q_distillation_1": _required_receipt(
            root / "distill1" / "q_distillation_em1_receipt.json"
        ),
        "prior_mstep_2": _required_receipt(
            root / "mstep2" / "prior_mstep_2_receipt.json"
        ),
    }
    active = root / "preflight" / "active_bootstrap" / "active_bootstrap_receipt.json"
    if active.is_file():
        phase_receipts["bounded_active_bootstrap"] = _required_receipt(active)
    for name, receipt in phase_receipts.items():
        _validate_phase_receipt(name, receipt)

    expected_rows = np.load(
        manifest["artifacts"]["selected_rows"]["path"], allow_pickle=False
    )
    banks = {}
    for label in ("em1", "em1_p1", "em2", "em2_p2"):
        bank_path = root / "banks" / label / "posterior_bank_manifest.json"
        banks[label] = _validate_bank(bank_path, expected_rows, manifest)
    final_summary = summarize_posterior_bank(
        root / "banks" / "em2_p2" / "posterior_bank_manifest.json"
    )
    maximum_unresolved = float(
        (
            ((config.get("amortized", {}) or {}).get("sc_asmc_em", {}) or {}).get(
                "preflight", {}
            )
            or {}
        ).get("maximum_unresolved_fraction", 0.05)
    )
    if final_summary["unresolved_fraction"] > maximum_unresolved:
        raise RuntimeError(
            "final posterior bank exceeds the configured unresolved-fraction gate"
        )

    feature_stats = root / "runtime" / "feature_stats.json"
    latent = root / "runtime" / "latent_transform_provenance.json"
    for path in (feature_stats, latent):
        if not path.is_file():
            raise FileNotFoundError(path)
    feature_payload = _read_json(feature_stats)
    latent_payload = _read_json(latent)
    if feature_payload.get("truth_used") is not False:
        raise ValueError("feature statistics do not carry a no-truth receipt")
    if latent_payload.get("truth_used") is not False:
        raise ValueError("latent transform does not carry a no-truth receipt")
    if latent_payload.get("coordinate_information_source") != (
        "fit_bounds_fit_initials_and_config_only"
    ):
        raise ValueError(
            "latent transform provenance uses an unsupported information source"
        )
    checkpoints = _validate_checkpoints(
        root,
        phase_receipts,
        workflow_config_hash=manifest["config_sha256"],
        latent_transform_hash=latent_payload["transform_hash"],
        feature_stats_hash=feature_payload["feature_stats_hash"],
    )
    performance = _validate_performance_receipt(
        root / "runtime" / "estep_micro_batch_autotune.json",
        workflow_config_hash=manifest["config_sha256"],
        latent_transform_hash=latent_payload["transform_hash"],
        feature_stats_hash=feature_payload["feature_stats_hash"],
    )

    report_path = root / "report" / "report_receipt.json"
    report = _required_receipt(report_path)
    _validate_phase_receipt("report", report)
    for name, record in report.get("artifacts", {}).items():
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"final report artifact hash mismatch: {name}")
    frozen_model = _validate_frozen_model(
        report,
        q_checkpoint=checkpoints["q1_ema"],
        prior_checkpoint=checkpoints["p2"],
        final_bank=banks["em2_p2"],
        feature_stats_path=feature_stats,
        workflow_config_hash=manifest["config_sha256"],
    )

    code = manifest.get("code", {})
    payload = {
        "status": "PASS",
        "workflow": "Selection-Corrected Amortized SMC-EM",
        "acronym": "SC-ASMC-EM",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": TARGET_POPULATION_CONTRACT,
        "observed_selection": OBSERVED_SELECTION_CONTRACT,
        "scope_limit": "No inference claim is made outside C0.",
        "config_sha256": manifest["config_sha256"],
        "probabilistic_model": {
            "population": "x_i ~ p_eta(x), target p_eta(theta | C0)",
            "physical_transform": "theta_i = T(x_i)",
            "forward_model": "fhat_i = DSPS(theta_i)",
            "observation": ("flux_i | x_i,fluxerr_i ~ Normal(fhat_i, fluxerr_i^2)"),
            "object_target": (
                "log pi_i(x) = log p(flux_i|x,fluxerr_i) + log p_eta(x) + constant"
            ),
            "selection": "beta(x)=P(A=1|x), alpha_eta=E_p_eta[beta(x)]",
            "prior_loss": (
                "-mean_i sum_k stop(w_ik) log p_eta(x_ik) + log(alpha_eta) "
                "+ lambda_trust KL(p_eta_old || p_eta)"
            ),
            "selection_in_object_weights": False,
        },
        "em_contract": {
            "outer_iterations": 2,
            "phase_snapshots_frozen": True,
            "simultaneous_q_prior_training": False,
            "nuts_in_training": False,
            "static_rws_in_training": False,
        },
        "no_truth_training": {
            "truth_used": False,
            "truth_columns_requested": [],
            "manifest": True,
            "preflight": True,
            "e_steps": True,
            "m_steps": True,
            "checkpoint_selection": True,
            "report": True,
            "truth_allowed_only_after_frozen_receipt": True,
        },
        "dataset": manifest["dataset"],
        "code": code,
        "selected_objects": int(manifest["objects"]["selected"]),
        "final_bank": final_summary,
        "posterior_banks": banks,
        "checkpoints": checkpoints,
        "frozen_model": frozen_model,
        "latent_transform": {
            "path": str(latent.resolve()),
            "sha256": sha256_file(latent),
            "semantic_hash": latent_payload["transform_hash"],
        },
        "feature_stats": {
            "path": str(feature_stats.resolve()),
            "sha256": sha256_file(feature_stats),
            "semantic_hash": feature_payload["feature_stats_hash"],
            "source": "observed train split only",
        },
        "performance": performance,
        "report": {
            "path": str(report_path.resolve()),
            "sha256": sha256_file(report_path),
            "artifacts": report["artifacts"],
        },
        "post_freeze_only_commands": [
            "truth closure (coverage, TARP/MIRA, bias, population recovery)",
            "NUTS reference on 4-8 galaxies",
        ],
    }
    destination = (
        Path(out_path) if out_path is not None else root / "FINAL_RECEIPT.json"
    )
    _atomic_json(destination, payload)
    pass_marker = root / "FINAL_PASS"
    temporary = pass_marker.with_name(f".{pass_marker.name}.{os.getpid()}.tmp")
    temporary.write_text(sha256_file(destination) + "\n", encoding="utf-8")
    os.replace(temporary, pass_marker)
    return payload


def _validate_performance_receipt(
    path: str | Path,
    *,
    workflow_config_hash: str | None = None,
    latent_transform_hash: str | None = None,
    feature_stats_hash: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    payload = _required_receipt(source)
    if payload.get("status") != "complete":
        raise ValueError("E-step micro-batch autotune is incomplete")
    if payload.get("truth_used") is not False:
        raise ValueError("E-step micro-batch autotune used truth")
    if payload.get("c0_scope_statement") != C0_SCOPE_STATEMENT:
        raise ValueError("E-step micro-batch autotune lacks canonical C0 scope")
    expected = {
        "workflow_config_hash": workflow_config_hash,
        "latent_transform_hash": latent_transform_hash,
        "feature_stats_hash": feature_stats_hash,
    }
    for name, value in expected.items():
        if value is not None and payload.get(name) != value:
            raise ValueError(f"E-step micro-batch autotune mismatch: {name}")
    selected = int(payload.get("selected_batch_size", 0))
    devices = int(payload.get("device_count", 0))
    if selected <= 0 or devices <= 0 or selected % devices:
        raise ValueError("E-step micro-batch is not device-divisible")
    if payload.get("mode") == "xla_compiled_memory":
        if not payload.get("attempts"):
            raise ValueError("XLA micro-batch autotune has no compiler analysis")
        if float(payload.get("maximum_compiled_memory_fraction", 0.0)) > 0.90:
            raise ValueError("XLA micro-batch autotune exceeded the 0.90 ceiling")
    return {
        "autotune": _artifact_record(source),
        "mode": payload.get("mode"),
        "selected_batch_size": selected,
        "device_count": devices,
        "device_kinds": payload.get("device_kinds", []),
        "target_device_memory_fraction": payload.get("target_device_memory_fraction"),
        "selected_compiler_peak_memory_fraction": payload.get(
            "selected_compiler_peak_memory_fraction"
        ),
        "host_prefetch_enabled": True,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_frozen_model(
    report: dict[str, Any],
    *,
    q_checkpoint: dict[str, Any],
    prior_checkpoint: dict[str, Any],
    final_bank: dict[str, Any],
    feature_stats_path: Path,
    workflow_config_hash: str,
) -> dict[str, Any]:
    record = report.get("frozen_model") or {}
    if record.get("model_components") != "q1_ema + p2":
        raise ValueError("final report does not freeze the q1 EMA + p2 model")
    checkpoint = Path((record.get("checkpoint") or {}).get("path", ""))
    sidecar_path = Path((record.get("sidecar") or {}).get("path", ""))
    if not checkpoint.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("frozen final model checkpoint or sidecar")
    if sha256_file(checkpoint) != record["checkpoint"]["sha256"]:
        raise ValueError("frozen final model checkpoint hash mismatch")
    if sha256_file(sidecar_path) != record["sidecar"]["sha256"]:
        raise ValueError("frozen final model sidecar hash mismatch")
    sidecar = _read_json(sidecar_path)
    if (
        sidecar.get("truth_used") is not False
        or sidecar.get("truth_columns_requested") != []
    ):
        raise ValueError("frozen final model lacks a no-truth contract")
    if sidecar.get("c0_scope_statement") != C0_SCOPE_STATEMENT:
        raise ValueError("frozen final model lacks the canonical C0 scope")
    if sidecar.get("target_population") != TARGET_POPULATION_CONTRACT:
        raise ValueError("frozen final model has an invalid target population")
    if sidecar.get("model_components") != "q1_ema + p2":
        raise ValueError("frozen final model has ambiguous components")
    expected_links = {
        "q_checkpoint": q_checkpoint["path"],
        "q_checkpoint_sha256": q_checkpoint["sha256"],
        "prior_checkpoint": prior_checkpoint["path"],
        "prior_checkpoint_sha256": prior_checkpoint["sha256"],
        "final_bank_manifest": final_bank["path"],
        "final_bank_manifest_sha256": final_bank["sha256"],
        "feature_stats_path": str(feature_stats_path.resolve()),
        "feature_stats_sha256": sha256_file(feature_stats_path),
        "workflow_config_hash": workflow_config_hash,
    }
    if any(sidecar.get(name) != value for name, value in expected_links.items()):
        raise ValueError("frozen final model input provenance mismatch")
    return {
        "checkpoint": record["checkpoint"],
        "sidecar": record["sidecar"],
        "model_components": "q1_ema + p2",
        "validated": True,
    }


def _validate_no_truth_config(config: dict[str, Any]) -> None:
    truth = (config.get("truth", {}) or {}).get("parameter_columns", {}) or {}
    if truth:
        raise ValueError("SC-ASMC-EM config contains truth parameter mappings")
    objective = (config.get("amortized", {}) or {}).get("objective", {}) or {}
    if float(objective.get("prior_truth_weight", 0.0)) != 0.0:
        raise ValueError("SC-ASMC-EM prior truth loss must be disabled")
    if float(objective.get("npe_weight", 0.0)) != 0.0:
        raise ValueError("SC-ASMC-EM supervised NPE truth loss must be disabled")


def _validate_phase_receipt(name: str, receipt: dict[str, Any]) -> None:
    if receipt.get("status") not in {"PASS", "complete"}:
        raise ValueError(f"SC-ASMC-EM phase is incomplete: {name}")
    if receipt.get("truth_used") is not False:
        raise ValueError(f"SC-ASMC-EM phase lacks no-truth evidence: {name}")
    if receipt.get("c0_scope_statement") != C0_SCOPE_STATEMENT:
        raise ValueError(f"SC-ASMC-EM phase lacks canonical C0 scope: {name}")


def _validate_bank(
    manifest_path: Path,
    expected_rows: np.ndarray,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = _required_receipt(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"posterior bank is incomplete: {manifest_path}")
    provenance = manifest["provenance"]
    validate_posterior_bank_manifest_provenance(
        manifest,
        expected_fields={
            "dataset_hash": run_manifest["dataset"]["sha256"],
            "workflow_config_hash": run_manifest["config_sha256"],
        },
    )
    if provenance["dataset_hash"] != run_manifest["dataset"]["sha256"]:
        raise ValueError("posterior-bank dataset hash mismatch")
    if provenance.get("workflow_config_hash") != run_manifest["config_sha256"]:
        raise ValueError("posterior-bank workflow configuration hash mismatch")
    if provenance["c0_scope_statement"] != C0_SCOPE_STATEMENT:
        raise ValueError("posterior-bank C0 scope mismatch")
    if provenance["target_population"] != TARGET_POPULATION_CONTRACT:
        raise ValueError("posterior-bank target population mismatch")
    if provenance["selection_contract"].get("enters_object_weights") is not False:
        raise ValueError("posterior bank allows selection in object weights")
    if provenance["likelihood_contract"].get("family") != "gaussian":
        raise ValueError("posterior bank does not use the Gaussian main likelihood")
    rows = []
    for record in manifest["shards"]:
        path = Path(record["path"])
        if not is_posterior_bank_shard_complete(path, validate_arrays=True):
            raise ValueError(f"posterior-bank shard failed validation: {path}")
        with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
            rows.append(np.asarray(arrays["row_index"], dtype=np.int64))
    actual = np.concatenate(rows)
    if not np.array_equal(np.sort(actual), np.sort(expected_rows)):
        raise ValueError("posterior bank does not cover the selected catalogue")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "objects": int(len(actual)),
        "shards": int(len(manifest["shards"])),
        "covers_selected_catalogue_exactly": True,
        "provenance": provenance,
    }


def _validate_checkpoints(
    root: Path,
    receipts: dict[str, dict[str, Any]],
    *,
    workflow_config_hash: str,
    latent_transform_hash: str,
    feature_stats_hash: str,
) -> dict[str, dict[str, Any]]:
    initialization = receipts["initialization"]
    candidates = {
        "p0": initialization["prior_p0"],
        "q_sleep_raw": {
            "path": receipts["sleep"]["q_raw_checkpoint"],
            "sha256": receipts["sleep"]["q_raw_sha256"],
        },
        "q_sleep_ema": {
            "path": receipts["sleep"]["q_ema_checkpoint"],
            "sha256": receipts["sleep"]["q_ema_sha256"],
        },
        "p1": {
            "path": receipts["prior_mstep_1"]["prior_checkpoint"],
            "sha256": receipts["prior_mstep_1"]["prior_sha256"],
        },
        "q1_raw": {
            "path": receipts["q_distillation_1"]["q_raw_checkpoint"],
            "sha256": receipts["q_distillation_1"]["q_raw_sha256"],
        },
        "q1_ema": {
            "path": receipts["q_distillation_1"]["q_ema_checkpoint"],
            "sha256": receipts["q_distillation_1"]["q_ema_sha256"],
        },
        "p2": {
            "path": receipts["prior_mstep_2"]["prior_checkpoint"],
            "sha256": receipts["prior_mstep_2"]["prior_sha256"],
        },
    }
    if "bounded_active_bootstrap" in receipts:
        candidates["q_active_ema"] = {
            "path": receipts["bounded_active_bootstrap"]["q_ema_checkpoint"],
            "sha256": receipts["bounded_active_bootstrap"]["q_ema_sha256"],
        }
    result = {}
    for name, record in candidates.items():
        path = Path(record["path"])
        expected = record["sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen checkpoint hash mismatch: {name}")
        sidecar_path = path.with_suffix(path.suffix + ".json")
        if not sidecar_path.is_file():
            raise FileNotFoundError(sidecar_path)
        sidecar = _read_json(sidecar_path)
        if sidecar.get("sha256") != expected:
            raise ValueError(f"frozen checkpoint sidecar hash mismatch: {name}")
        if sidecar.get("workflow_config_hash") != workflow_config_hash:
            raise ValueError(f"frozen checkpoint config mismatch: {name}")
        if sidecar.get("latent_transform_hash") != latent_transform_hash:
            raise ValueError(f"frozen checkpoint latent mismatch: {name}")
        if sidecar.get("feature_stats_hash") != feature_stats_hash:
            raise ValueError(f"frozen checkpoint feature mismatch: {name}")
        if sidecar.get("truth_used") is not False:
            raise ValueError(f"frozen checkpoint lacks no-truth evidence: {name}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": expected,
            "sidecar": _artifact_record(sidecar_path),
        }
    return result


def _required_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return _read_json(source)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
