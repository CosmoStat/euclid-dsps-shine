from __future__ import annotations

import copy
import json
import subprocess

import pytest

from euclid_dsps.amortized.posterior_bank import C0_SCOPE_STATEMENT, sha256_file
from euclid_dsps.amortized.sc_asmc_config import sc_asmc_em_config_hash
from euclid_dsps.amortized.sc_asmc_validate import (
    _validate_frozen_model,
    _validate_no_truth_config,
    _validate_performance_receipt,
    _validate_phase_receipt,
    validate_production_smoke_gate,
)
from euclid_dsps.config import load_config


def test_final_validation_rejects_any_truth_mapping() -> None:
    config = load_config("configs/experiments/feniks_sc_asmc_em_r25.yaml")
    _validate_no_truth_config(config)
    invalid = copy.deepcopy(config)
    invalid["truth"] = {"parameter_columns": {"z_obs": "z_obs"}}

    with pytest.raises(ValueError, match="truth parameter mappings"):
        _validate_no_truth_config(invalid)


def test_phase_receipt_requires_no_truth_and_canonical_c0_scope() -> None:
    receipt = {
        "status": "PASS",
        "truth_used": False,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
    }
    _validate_phase_receipt("toy", receipt)

    invalid = dict(receipt, truth_used=True)
    with pytest.raises(ValueError, match="no-truth evidence"):
        _validate_phase_receipt("toy", invalid)


def test_production_gate_requires_hash_bound_matching_four_device_smoke(
    tmp_path,
) -> None:
    config = load_config("configs/experiments/feniks_sc_asmc_em_r25.yaml")
    catalogue = tmp_path / "catalog.parquet"
    catalogue.write_bytes(b"observed catalogue")
    config["catalog_path"] = str(catalogue.resolve())
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    bank = tmp_path / "bank.json"
    manifest = tmp_path / "manifest" / "run_manifest.json"
    manifest.parent.mkdir()
    bank.write_bytes(b"bank")
    manifest.write_bytes(b"manifest")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = {
        "status": "PASS",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "truth_used": False,
        "devices": ["H100"] * 4,
        "selection_gradient": {"finite": True, "nonzero": True},
        "dataset_sha256": sha256_file(catalogue),
        "workflow_config_hash": sc_asmc_em_config_hash(config),
        "code_commit": commit,
        "posterior_bank": str(bank.resolve()),
        "posterior_bank_sha256": sha256_file(bank),
        "run_manifest_sha256": sha256_file(manifest),
    }
    receipt_path = smoke / "SMOKE_PASS.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (smoke / "SMOKE_PASS").write_text(
        sha256_file(receipt_path) + "\n",
        encoding="utf-8",
    )

    assert validate_production_smoke_gate(tmp_path, config, catalogue) == receipt
    catalogue.write_bytes(b"changed")
    with pytest.raises(ValueError, match="dataset differs"):
        validate_production_smoke_gate(tmp_path, config, catalogue)


def test_frozen_model_requires_q1_ema_p2_and_no_truth(tmp_path) -> None:
    checkpoint = tmp_path / "final.eqx"
    sidecar = tmp_path / "final.eqx.json"
    q_checkpoint = tmp_path / "q1_ema.eqx"
    prior_checkpoint = tmp_path / "p2.eqx"
    final_bank = tmp_path / "posterior_bank_manifest.json"
    feature_stats = tmp_path / "feature_stats.json"
    for path, value in (
        (q_checkpoint, b"q"),
        (prior_checkpoint, b"p"),
        (final_bank, b"bank"),
        (feature_stats, b"features"),
    ):
        path.write_bytes(value)
    checkpoint.write_bytes(b"model")
    q_record = {
        "path": str(q_checkpoint.resolve()),
        "sha256": sha256_file(q_checkpoint),
    }
    prior_record = {
        "path": str(prior_checkpoint.resolve()),
        "sha256": sha256_file(prior_checkpoint),
    }
    bank_record = {"path": str(final_bank.resolve()), "sha256": sha256_file(final_bank)}
    sidecar.write_text(
        json.dumps(
            {
                "truth_used": False,
                "truth_columns_requested": [],
                "c0_scope_statement": C0_SCOPE_STATEMENT,
                "target_population": "p_eta(theta | C0)",
                "model_components": "q1_ema + p2",
                "q_checkpoint": q_record["path"],
                "q_checkpoint_sha256": q_record["sha256"],
                "prior_checkpoint": prior_record["path"],
                "prior_checkpoint_sha256": prior_record["sha256"],
                "final_bank_manifest": bank_record["path"],
                "final_bank_manifest_sha256": bank_record["sha256"],
                "feature_stats_path": str(feature_stats.resolve()),
                "feature_stats_sha256": sha256_file(feature_stats),
                "workflow_config_hash": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    report = {
        "frozen_model": {
            "model_components": "q1_ema + p2",
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "sidecar": {"path": str(sidecar), "sha256": sha256_file(sidecar)},
        }
    }

    arguments = {
        "q_checkpoint": q_record,
        "prior_checkpoint": prior_record,
        "final_bank": bank_record,
        "feature_stats_path": feature_stats,
        "workflow_config_hash": "c" * 64,
    }
    assert _validate_frozen_model(report, **arguments)["validated"] is True
    invalid = copy.deepcopy(report)
    invalid["frozen_model"]["model_components"] = "q0 + p2"
    with pytest.raises(ValueError, match="q1 EMA"):
        _validate_frozen_model(invalid, **arguments)


def test_final_performance_receipt_requires_device_divisible_autotune(tmp_path) -> None:
    path = tmp_path / "autotune.json"
    payload = {
        "status": "complete",
        "mode": "xla_compiled_memory",
        "truth_used": False,
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "selected_batch_size": 128,
        "device_count": 4,
        "device_kinds": ["H100"] * 4,
        "workflow_config_hash": "1" * 64,
        "latent_transform_hash": "2" * 64,
        "feature_stats_hash": "3" * 64,
        "maximum_compiled_memory_fraction": 0.90,
        "attempts": [{"batch_size": 128, "status": "compiled"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        _validate_performance_receipt(
            path,
            workflow_config_hash="1" * 64,
            latent_transform_hash="2" * 64,
            feature_stats_hash="3" * 64,
        )["selected_batch_size"]
        == 128
    )
    payload["selected_batch_size"] = 127
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="device-divisible"):
        _validate_performance_receipt(path)
