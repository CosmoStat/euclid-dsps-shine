import pytest

from euclid_dsps.amortized.population_vem import sha256_file
from scripts.feniks_transport_precision import pin_reference
from scripts.run_feniks_sc_drws_balanced_npe import write


def test_reference_receipts_and_inputs_are_pinned(tmp_path):
    manifest = dict(
        method="qualified_objective_pilot_v1",
        objective_pilot={},
        qualified_night={},
        config_sha256="config",
        rows_sha256="rows",
        seed=1,
        objects_per_group=1,
        objective_recipe={},
    )
    write(tmp_path / "RUN_MANIFEST.json", manifest)
    audits = []
    for group in ("observed", "simulated"):
        for start in (0, 1):
            name = f"cases/{group}_000/audit_start_{start}/AUDIT.json"
            write(tmp_path / name, dict(status="PASS"))
            audits.append(
                dict(
                    case=f"{group}_000",
                    start=start,
                    artifacts={name: sha256_file(tmp_path / name)},
                )
            )
    write(tmp_path / "OBJECTIVE_AUDIT.json", dict(audits=audits))
    write(
        tmp_path / "FINAL.json",
        dict(
            status="OBJECTIVE_AUDIT_NOT_PASSED",
            optimization_started=False,
            scientific_promotion=False,
            artifacts={
                "OBJECTIVE_AUDIT.json": dict(
                    sha256=sha256_file(tmp_path / "OBJECTIVE_AUDIT.json")
                )
            },
        ),
    )
    reference = pin_reference(tmp_path, manifest)
    assert len(reference["hashes"]) == 7
    with pytest.raises(ValueError, match="reference mismatch: seed"):
        pin_reference(tmp_path, {**manifest, "seed": 2})
    write(tmp_path / name, dict(status="tampered"))
    with pytest.raises(ValueError):
        pin_reference(tmp_path, manifest)


def test_precision_requires_objective_source(tmp_path):
    from scripts.feniks_qualified_local_vi import prepare

    with pytest.raises(ValueError, match="objective"):
        prepare(
            tmp_path / "new",
            tmp_path / "source",
            tmp_path / "night",
            objective_precision_root=tmp_path / "audit",
        )
    assert not (tmp_path / "new").exists()
