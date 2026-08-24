from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from euclid_dsps.amortized.posterior_bank import (
    C0_SCOPE_STATEMENT,
    OBSERVED_SELECTION_CONTRACT,
    PosteriorBankProvenance,
    PosteriorBankShard,
    is_posterior_bank_shard_complete,
    iter_posterior_bank_shards,
    low_reweight_ess_rows,
    merge_posterior_bank_shards,
    read_posterior_bank_shard,
    replace_posterior_bank_rows,
    reweight_posterior_particles,
    validate_posterior_bank_manifest_provenance,
    write_posterior_bank_shard,
)


def _provenance() -> PosteriorBankProvenance:
    digest = "1" * 64
    return PosteriorBankProvenance(
        dataset_hash=digest,
        workflow_config_hash="0" * 64,
        q_checkpoint_hash="2" * 64,
        q_ema_hash="3" * 64,
        prior_checkpoint_hash="4" * 64,
        latent_transform_hash="5" * 64,
        feature_stats_hash="6" * 64,
        likelihood_contract={"family": "gaussian"},
        selection_contract={
            "event": OBSERVED_SELECTION_CONTRACT,
            "enters_object_weights": False,
        },
        code_commit="179053c",
        upstream_selection_provenance={"domain": "C0", "audited": True},
    )


def _shard(start: int = 0, objects: int = 3) -> PosteriorBankShard:
    capacity = 4
    weights = np.full((objects, capacity), 1.0 / capacity)
    particles = np.arange(objects * capacity * 2, dtype=np.float32).reshape(
        objects, capacity, 2
    )
    return PosteriorBankShard(
        row_index=np.arange(start, start + objects, dtype=np.int64),
        object_id=np.asarray([f"object-{i}" for i in range(start, start + objects)]),
        method=np.asarray([0, 1, 2][:objects], dtype=np.int8),
        particles=particles,
        normalized_weights=weights,
        source_logprior=np.zeros((objects, capacity)),
        particle_count=np.full(objects, capacity, dtype=np.int16),
        ess=np.full(objects, capacity, dtype=float),
        max_weight=np.full(objects, 1.0 / capacity),
        beta_final=np.ones(objects),
        logz=np.zeros(objects),
        stage_count=np.arange(objects),
        acceptance=np.full(objects, 0.5),
        ancestor_ess=np.full(objects, 3.0),
        unique_ancestor_fraction=np.full(objects, 0.75),
        movement_squared=np.full(objects, 0.2),
        moved_particle_fraction=np.full(objects, 0.75),
        dsps_evaluations=np.full(objects, 100),
        resolved=np.ones(objects, dtype=bool),
        features=np.ones((objects, 36), dtype=np.float32),
    )


def test_posterior_bank_roundtrip_resume_and_tamper_detection(tmp_path) -> None:
    original = _shard()
    metadata = write_posterior_bank_shard(tmp_path, 0, original, _provenance())
    path = tmp_path / "shards" / "shard_00000"
    restored = read_posterior_bank_shard(path)

    assert metadata["object_count"] == 3
    assert is_posterior_bank_shard_complete(path, validate_arrays=True)
    np.testing.assert_array_equal(restored.row_index, original.row_index)
    np.testing.assert_allclose(restored.particles, original.particles)
    resumed = write_posterior_bank_shard(tmp_path, 0, original, _provenance())
    assert resumed == metadata

    changed = replace(_provenance(), prior_checkpoint_hash="a" * 64)
    with pytest.raises(ValueError, match="resume provenance mismatch"):
        write_posterior_bank_shard(tmp_path, 0, original, changed)

    (path / "COMPLETE.json").write_text("{}", encoding="utf-8")
    assert not is_posterior_bank_shard_complete(path)


def test_posterior_bank_sharding_streaming_merge_and_exact_rows(tmp_path) -> None:
    write_posterior_bank_shard(tmp_path, 0, _shard(0, 3), _provenance())
    write_posterior_bank_shard(tmp_path, 1, _shard(3, 3), _provenance())
    paths = sorted((tmp_path / "shards").iterdir())
    manifest = merge_posterior_bank_shards(
        tmp_path,
        paths,
        expected_row_indices=np.arange(6),
    )

    assert manifest["object_count"] == 6
    assert manifest["shard_count"] == 2
    assert manifest["provenance"]["c0_scope_statement"] == C0_SCOPE_STATEMENT
    validated_provenance = validate_posterior_bank_manifest_provenance(
        manifest,
        expected_fields={"workflow_config_hash": "0" * 64},
    )
    assert validated_provenance.dataset_hash == "1" * 64
    with pytest.raises(ValueError, match="workflow_config_hash"):
        validate_posterior_bank_manifest_provenance(
            manifest,
            expected_fields={"workflow_config_hash": "f" * 64},
        )
    streamed = list(
        iter_posterior_bank_shards(tmp_path / "posterior_bank_manifest.json")
    )
    assert [item.object_count for item in streamed] == [3, 3]
    with pytest.raises(ValueError, match="expected catalogue"):
        merge_posterior_bank_shards(
            tmp_path / "bad",
            paths,
            expected_row_indices=np.arange(7),
        )


def test_bank_reweight_identity_prior_ratio_and_selective_refresh() -> None:
    shard = _shard()
    identity, identity_ess, identity_max = reweight_posterior_particles(
        shard.normalized_weights,
        shard.source_logprior,
        shard.source_logprior,
        shard.particle_count,
    )
    np.testing.assert_allclose(identity, shard.normalized_weights)
    np.testing.assert_allclose(identity_ess, 4.0)
    np.testing.assert_allclose(identity_max, 0.25)

    new_logprior = np.zeros_like(shard.source_logprior)
    new_logprior[1] = np.log([1000.0, 1.0, 1.0, 1.0])
    reweighted, ess, maximum = reweight_posterior_particles(
        shard.normalized_weights,
        shard.source_logprior,
        new_logprior,
        shard.particle_count,
    )
    assert reweighted[1, 0] > 0.99
    assert ess[1] < 2.0
    assert maximum[1] > 0.99
    rows = low_reweight_ess_rows(
        shard.row_index,
        ess,
        shard.particle_count,
        minimum_ess_fraction=0.5,
    )
    np.testing.assert_array_equal(rows, [1])


def test_bank_rejects_selection_inside_object_weights() -> None:
    provenance = replace(
        _provenance(),
        selection_contract={
            "event": OBSERVED_SELECTION_CONTRACT,
            "enters_object_weights": True,
        },
    )
    with pytest.raises(ValueError, match="must not enter object posterior weights"):
        provenance.validate()


def test_selective_bank_row_replacement_preserves_unselected_rows() -> None:
    base = _shard(0, 3)
    replacement = _shard(1, 1)
    replacement = replace(
        replacement,
        particles=np.full_like(replacement.particles, 99.0),
        object_id=np.asarray(["object-1"]),
    )

    merged = replace_posterior_bank_rows(base, replacement)

    np.testing.assert_allclose(merged.particles[0], base.particles[0])
    np.testing.assert_allclose(merged.particles[1], 99.0)
    np.testing.assert_allclose(merged.particles[2], base.particles[2])
