from __future__ import annotations

import json

import numpy as np
import pytest

from euclid_dsps.amortized.posterior_bank import C0_SCOPE_STATEMENT, sha256_file
from euclid_dsps.amortized.sc_asmc_postfreeze import (
    choose_postfreeze_nuts_records,
    dense_weighted_particle_draws,
    validate_postfreeze_gate,
)


def test_dense_closure_draws_preserve_joint_particles_and_are_reproducible() -> None:
    particles = np.asarray(
        [
            [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]],
            [[3.0, 13.0], [4.0, 14.0], [99.0, 99.0]],
        ],
        dtype=np.float32,
    )
    weights = np.asarray([[0.0, 0.25, 0.75], [0.5, 0.5, 0.0]])
    kwargs = {
        "samples": 64,
        "seed": 7,
        "row_indices": np.asarray([5, 9]),
    }

    first = dense_weighted_particle_draws(
        particles, weights, np.asarray([3, 2]), **kwargs
    )
    second = dense_weighted_particle_draws(
        particles, weights, np.asarray([3, 2]), **kwargs
    )

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 64, 2)
    assert np.all(first[..., 1] - first[..., 0] == 10.0)
    assert not np.any(first[1, :, 0] == 99.0)


def test_nuts_cohort_spans_available_methods_and_requires_four_to_eight() -> None:
    records = [
        {
            "row_index": index,
            "object_id": str(index),
            "method": index % 4,
            "resolved": True,
            "difficulty": float(index),
            "ess_fraction": 0.5,
            "max_weight": 0.2,
        }
        for index in range(20)
    ]
    chosen = choose_postfreeze_nuts_records(records, count=8)

    assert len(chosen) == 8
    assert len({record["row_index"] for record in chosen}) == 8
    assert {record["method"] for record in chosen} == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="between 4 and 8"):
        choose_postfreeze_nuts_records(records, count=3)


def test_postfreeze_gate_is_bound_to_no_truth_final_receipt(tmp_path) -> None:
    checkpoint = tmp_path / "final.eqx"
    sidecar = tmp_path / "final.eqx.json"
    checkpoint.write_bytes(b"model")
    sidecar.write_text("{}\n", encoding="utf-8")
    receipt = {
        "status": "PASS",
        "c0_scope_statement": C0_SCOPE_STATEMENT,
        "target_population": "p_eta(theta | C0)",
        "no_truth_training": {"truth_used": False},
        "frozen_model": {
            "model_components": "q1_ema + p2",
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "sidecar": {"path": str(sidecar), "sha256": sha256_file(sidecar)},
        },
    }
    receipt_path = tmp_path / "FINAL_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "FINAL_PASS").write_text(
        sha256_file(receipt_path) + "\n", encoding="utf-8"
    )

    assert validate_postfreeze_gate(tmp_path)["status"] == "PASS"
    (tmp_path / "FINAL_PASS").write_text("bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        validate_postfreeze_gate(tmp_path)
