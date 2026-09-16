import json

import pandas as pd
import pytest

from scripts.feniks_objective_night import digest, gate, prepare, read, write


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, value)


@pytest.fixture
def night(tmp_path):
    pilot, root = tmp_path / "pilot", tmp_path / "night"
    pilot.mkdir()
    (pilot / "config.yaml").write_text("test: true\n")
    (pilot / "observed_rows.npy").write_bytes(b"test")
    put(pilot, "SUBMISSION.json", dict(jobs=dict(DIAGNOSTIC_JOB="123")))
    put(
        pilot,
        "RUN_MANIFEST.json",
        dict(
            method="qualified_objective_transport64_pilot_v1",
            objects_per_group=8,
            transport_contract="conditional_transport_float64_v1",
            config_sha256=digest(pilot / "config.yaml"),
            rows_sha256=digest(pilot / "observed_rows.npy"),
            objective_recipe=dict(decoder_draw_budgets=[1024, 4096]),
        ),
    )
    prepare(pilot, root, "123")
    audits = []
    for group in ("observed", "simulated"):
        for index in range(8):
            case = f"cases/{group}_{index:03d}"
            outcomes = []
            for start in (0, 1):
                audits.append(
                    dict(
                        case=f"{group}_{index:03d}",
                        start=start,
                        status="PASS",
                        artifacts={},
                    )
                )
                local = f"{case}/wake_{start}/draws_04096"
                summary = dict(pooled_draws=4096, raw_ess=dict(fraction_median=0.04))
                put(pilot, f"{local}/SUMMARY.json", summary)
                path = pilot / local / "parameters.eqx"
                path.write_bytes(b"checkpoint")
                put(
                    pilot,
                    f"{local}/TRANSPORT_CONTRACT.json",
                    dict(
                        version="conditional_transport_float64_v1",
                        checkpoint_sha256=digest(path),
                        manifest_sha256=digest(pilot / "RUN_MANIFEST.json"),
                    ),
                )
                put(
                    pilot,
                    f"{case}/source_{start}/SUMMARY.json",
                    dict(pooled_draws=4096, raw_ess=dict(fraction_median=0.02)),
                )
                pd.DataFrame(
                    dict(
                        decoder_draws=range(256, 4097, 256), update_applied=[True] * 16
                    )
                ).to_csv(pilot / case / f"wake_{start}/optimization.csv", index=False)
                outcomes.append(
                    dict(
                        arm="wake",
                        start=start,
                        decoder_draws=4096,
                        checkpoint_sha256=digest(path),
                        applied_updates=16,
                        summary=summary,
                    )
                )
            put(pilot, f"{case}/COMPLETE.json", dict(outcomes=outcomes))
    put(pilot, "OBJECTIVE_AUDIT.json", dict(status="PASS", audits=audits))
    put(
        pilot,
        "FINAL.json",
        dict(
            status="OBJECTIVE_PILOT_COMPLETE",
            cases_complete=16,
            scientific_promotion=False,
            transport_contract="conditional_transport_float64_v1",
            artifacts={
                "OBJECTIVE_AUDIT.json": dict(
                    sha256=digest(pilot / "OBJECTIVE_AUDIT.json")
                )
            },
        ),
    )
    return pilot, root


def test_qualified_extension_and_fixed_protocol(night):
    pilot, root = night
    reference = gate(root)
    assert reference["hashes"]["FINAL.json"] == digest(pilot / "FINAL.json")
    assert read(root / "NIGHT_GATE.json")["status"] == "PASS"
    manifest = read(root / "RUN_MANIFEST.json")
    assert manifest["objective_recipe"]["decoder_draw_budgets"] == [1024, 4096]
    assert manifest["objective_execution_recipe"]["decoder_draw_budgets"] == [
        4096,
        16384,
        32768,
    ]
    assert manifest["seconds"] == 32400
    with pytest.raises(ValueError, match="dependency"):
        prepare(pilot, root.parent / "bad", "124")


def test_no_gain_blocks_extension(night):
    pilot, root = night
    for path in pilot.glob("cases/*/source_*/SUMMARY.json"):
        write(path, dict(pooled_draws=4096, raw_ess=dict(fraction_median=0.05)))
    with pytest.raises(ValueError, match="insufficient"):
        gate(root)
    assert read(root / "NIGHT_GATE.json")["status"] == "NOT_PASSED"


def test_rejected_wake_blocks_even_with_ess_gain(night):
    pilot, root = night
    for path in pilot.glob("cases/*/wake_*/optimization.csv"):
        history = pd.read_csv(path)
        history["update_applied"] = False
        history.to_csv(path, index=False)
    for path in pilot.glob("cases/*/COMPLETE.json"):
        complete = read(path)
        for outcome in complete["outcomes"]:
            outcome["applied_updates"] = 0
        write(path, complete)
    with pytest.raises(ValueError, match="insufficient"):
        gate(root)


@pytest.mark.parametrize(
    "damage", ["missing_final", "checkpoint", "manifest", "receipt"]
)
def test_bad_evidence_blocks_extension(night, damage):
    pilot, root = night
    if damage == "missing_final":
        (pilot / "FINAL.json").unlink()
    elif damage == "checkpoint":
        next(pilot.glob("cases/*/wake_*/draws_*/parameters.eqx")).write_bytes(
            b"changed"
        )
    elif damage == "manifest":
        (pilot / "RUN_MANIFEST.json").write_text("{}")
    else:
        path = next(pilot.glob("cases/*/COMPLETE.json"))
        value = json.loads(path.read_text())
        value["outcomes"][0]["applied_updates"] = 0
        write(path, value)
    with pytest.raises((ValueError, OSError)):
        gate(root)
