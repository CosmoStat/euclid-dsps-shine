"""Pin the unsuccessful objective audit used for an audit-only precision replay."""

from pathlib import Path

from euclid_dsps.amortized.population_vem import sha256_file
from scripts.feniks_support_probe import verify_source
from scripts.run_feniks_sc_drws_balanced_npe import read


def pin_reference(root, manifest):
    root = Path(root).resolve()
    old, final = read(root / "RUN_MANIFEST.json"), read(root / "FINAL.json")
    if (
        old["method"] != "qualified_objective_pilot_v1"
        or final["status"] != "OBJECTIVE_AUDIT_NOT_PASSED"
        or final["optimization_started"] is not False
        or final["scientific_promotion"] is not False
    ):
        raise ValueError("expected completed objective audit without optimization")
    for key in (
        "objective_pilot",
        "qualified_night",
        "config_sha256",
        "rows_sha256",
        "seed",
        "objects_per_group",
        "objective_recipe",
    ):
        if old[key] != manifest[key]:
            raise ValueError(f"precision replay reference mismatch: {key}")
    aggregate = read(root / "OBJECTIVE_AUDIT.json")
    if (
        sha256_file(root / "OBJECTIVE_AUDIT.json")
        != final["artifacts"]["OBJECTIVE_AUDIT.json"]["sha256"]
    ):
        raise ValueError("reference aggregate changed")
    expected = {
        (f"{g}_{i:03d}", s)
        for g in ("observed", "simulated")
        for i in range(manifest["objects_per_group"])
        for s in (0, 1)
    }
    if (
        len(aggregate["audits"]) != len(expected)
        or {(a["case"], a["start"]) for a in aggregate["audits"]} != expected
    ):
        raise ValueError("reference must contain all prescribed starts")
    hashes = {
        name: sha256_file(root / name)
        for name in ("RUN_MANIFEST.json", "FINAL.json", "OBJECTIVE_AUDIT.json")
    }
    for audit in aggregate["audits"]:
        hashes.update(audit["artifacts"])
    reference = dict(path=str(root), hashes=hashes)
    verify_source(reference)
    return reference
