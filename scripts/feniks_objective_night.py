"""CPU preparation/readback for a fail-closed fixed-parent overnight extension."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(pilot, root, dependency=None):
    pilot, root = pilot.resolve(), root.resolve()
    manifest = read(pilot / "RUN_MANIFEST.json")
    if dependency is not None and str(
        read(pilot / "SUBMISSION.json")["jobs"]["DIAGNOSTIC_JOB"]
    ) != str(dependency):
        raise ValueError("dependency does not match pilot submission")
    if (
        manifest["method"] != "qualified_objective_transport64_pilot_v1"
        or "night_extension" in manifest
        or "adaptation_contract" in manifest
    ):
        raise ValueError("expected original transport64 pilot")
    if manifest["objects_per_group"] != 8 or manifest["objective_recipe"][
        "decoder_draw_budgets"
    ] != [1024, 4096]:
        raise ValueError("unexpected pilot protocol")
    root.mkdir(parents=True, exist_ok=False)
    for name, key in (
        ("config.yaml", "config_sha256"),
        ("observed_rows.npy", "rows_sha256"),
    ):
        if digest(pilot / name) != manifest[key]:
            raise ValueError(f"pilot input changed: {name}")
        shutil.copy2(pilot / name, root / name)
    manifest.update(
        code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        seconds=32400,
        allocation_gpu_hours=10,
        maximum_decoder_evaluations=5000000,
        experiment_seed_offset=100000000,
        objective_execution_recipe={
            **manifest["objective_recipe"],
            "decoder_draw_budgets": [4096, 16384, 32768],
        },
        night_extension=dict(
            path=str(pilot),
            manifest_sha256=digest(pilot / "RUN_MANIFEST.json"),
            criteria=dict(
                minimum_median_ess_ratio=1.1,
                minimum_accepted_updates=4,
                minimum_eligible_trajectory_fraction=0.5,
                maximum_ess_regression_fraction=0.5,
            ),
        ),
        interpretation="Longer fixed-parent local reverse/wake comparison, gated by pilot evidence. Same development objects; fresh evaluation/optimization seeds. Not amortized population training or scientific promotion.",
    )
    write(root / "RUN_MANIFEST.json", manifest)


def gate(root):
    manifest = read(root / "RUN_MANIFEST.json")
    reference = manifest["night_extension"]
    pilot = Path(reference["path"])
    hashes = {}

    def checked(name):
        hashes[name] = digest(pilot / name)
        return read(pilot / name)

    if digest(pilot / "RUN_MANIFEST.json") != reference["manifest_sha256"]:
        raise ValueError("pilot manifest changed")
    checked("RUN_MANIFEST.json")
    final = checked("FINAL.json")
    if (
        final["status"] != "OBJECTIVE_PILOT_COMPLETE"
        or final["cases_complete"] != 16
        or final["scientific_promotion"] is not False
        or final["transport_contract"] != manifest["transport_contract"]
    ):
        raise ValueError("pilot not successfully completed under the required contract")
    audit = checked("OBJECTIVE_AUDIT.json")
    if (
        hashes["OBJECTIVE_AUDIT.json"]
        != final["artifacts"]["OBJECTIVE_AUDIT.json"]["sha256"]
    ):
        raise ValueError("pilot audit changed")
    if audit["status"] != "PASS" or len(audit["audits"]) != 32:
        raise ValueError("pilot audits not all passed")
    expected = {
        (f"{g}_{i:03d}", s)
        for g in ("observed", "simulated")
        for i in range(8)
        for s in (0, 1)
    }
    if {(a["case"], a["start"]) for a in audit["audits"]} != expected:
        raise ValueError("pilot audit coverage mismatch")
    for item in audit["audits"]:
        if item["status"] != "PASS":
            raise ValueError("non-PASS pilot audit")
        for name, expected in item["artifacts"].items():
            hashes[name] = digest(pilot / name)
            if hashes[name] != expected:
                raise ValueError("pilot audit artifact changed")
    rows = []
    criteria = reference["criteria"]
    for group in ("observed", "simulated"):
        for index in range(8):
            case = f"cases/{group}_{index:03d}"
            complete = checked(f"{case}/COMPLETE.json")
            for start in (0, 1):
                (outcome,) = [
                    o
                    for o in complete["outcomes"]
                    if o["arm"] == "wake"
                    and o["start"] == start
                    and o["decoder_draws"] == 4096
                ]
                local = f"{case}/wake_{start}/draws_04096"
                summary = checked(f"{local}/SUMMARY.json")
                contract = checked(f"{local}/TRANSPORT_CONTRACT.json")
                name = f"{local}/parameters.eqx"
                hashes[name] = digest(pilot / name)
                if (
                    hashes[name] != outcome["checkpoint_sha256"]
                    or contract["checkpoint_sha256"] != hashes[name]
                    or contract["version"] != manifest["transport_contract"]
                    or contract["manifest_sha256"] != reference["manifest_sha256"]
                    or summary != outcome["summary"]
                ):
                    raise ValueError("wake checkpoint or summary contract mismatch")
                source = checked(f"{case}/source_{start}/SUMMARY.json")
                if summary["pooled_draws"] != 4096 or source["pooled_draws"] != 4096:
                    raise ValueError("expected matched K4096 pilot evaluations")
                history_name = f"{case}/wake_{start}/optimization.csv"
                hashes[history_name] = digest(pilot / history_name)
                history = pd.read_csv(pilot / history_name)
                if (
                    len(history) != 16
                    or history.decoder_draws.tolist() != list(range(256, 4097, 256))
                    or not history.update_applied.isin([True, False, 0, 1]).all()
                ):
                    raise ValueError("unexpected wake history")
                accepted = int(history.update_applied.sum())
                if accepted != outcome["applied_updates"]:
                    raise ValueError("wake acceptance receipt mismatch")
                ess, initial = (
                    summary["raw_ess"]["fraction_median"],
                    source["raw_ess"]["fraction_median"],
                )
                if not np.isfinite([ess, initial]).all() or ess <= 0 or initial <= 0:
                    raise ValueError("invalid support metrics")
                rows.append(
                    dict(
                        group=group,
                        case=index,
                        start=start,
                        ess_ratio=ess / initial,
                        accepted=accepted,
                    )
                )
    table = pd.DataFrame(rows)
    groups = {}
    for group, part in table.groupby("group"):
        groups[group] = dict(
            median_ess_ratio=float(part.ess_ratio.median()),
            eligible_trajectory_fraction=float(
                (part.accepted >= criteria["minimum_accepted_updates"]).mean()
            ),
            ess_regression_fraction=float((part.ess_ratio < 1).mean()),
        )
    passed = all(
        g["median_ess_ratio"] >= criteria["minimum_median_ess_ratio"]
        and g["eligible_trajectory_fraction"]
        >= criteria["minimum_eligible_trajectory_fraction"]
        and g["ess_regression_fraction"] <= criteria["maximum_ess_regression_fraction"]
        for g in groups.values()
    )
    receipt = dict(
        status="PASS" if passed else "NOT_PASSED",
        groups=groups,
        criteria=criteria,
        reference=dict(path=str(pilot), hashes=hashes),
        scientific_promotion=False,
        interpretation="Resource-allocation gate for a development extension, not posterior qualification.",
    )
    write(root / "NIGHT_GATE.json", receipt)
    table.to_csv(root / "night_gate_pairs.csv", index=False)
    if not passed:
        raise ValueError(
            "wake acceptance or paired support gain insufficient for extension"
        )
    return receipt["reference"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dependency", required=True)
    args = parser.parse_args()
    prepare(args.pilot, args.root, args.dependency)


if __name__ == "__main__":
    main()
