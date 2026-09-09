"""Read full VI audits and prescribed pilot checkpoints, never select winners."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def summarize(root):
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if manifest["method"] not in (
        "qualified_objective_pilot_v1",
        "qualified_transport_precision_audit_v1",
    ):
        raise ValueError("expected objective pilot root")
    path = root / "OBJECTIVE_AUDIT.json"
    if not path.exists():
        print("Full objective audit not yet written")
        return
    audit = json.loads(path.read_text())
    for item in audit["audits"]:
        for name, expected in item["artifacts"].items():
            with (root / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"objective audit artifact changed: {name}")
    print("Native objective audit:", audit["status"])
    columns = ["case", "start", "status"]
    if "transport_precision_reference" in manifest:
        columns += ["native_reference_status", "transport64_status"]
    print(pd.DataFrame(audit["audits"])[columns].to_string(index=False))
    for item in audit["audits"]:
        if item["status"] == "PASS":
            continue
        report = json.loads(
            (
                root
                / "cases"
                / item["case"]
                / f"audit_start_{item['start']}"
                / "AUDIT.json"
            ).read_text()
        )
        native = report.get("variants", {}).get("native", {})
        unresolved = Counter(
            f"{check['block']}:{name}:{value['status']}"
            for check in native.get("checks", [])
            for name, value in check["components"].items()
            if value["status"] != "PASS"
        )
        identities = [
            f"{check['block']}:{name}"
            for check in native.get("checks", [])
            for name, passed in check["identities"].items()
            if not passed
        ]
        print(
            item["case"],
            item["start"],
            "unresolved:",
            dict(unresolved),
            "identities:",
            identities,
        )
    if "transport_precision_reference" in manifest:
        for item in audit["audits"]:
            p = (
                root
                / "cases"
                / item["case"]
                / f"audit_start_{item['start']}"
                / "TRANSPORT_PRECISION.json"
            )
            detail = json.loads(p.read_text())
            unresolved64 = [
                f"{check['block']}:{check['direction']}:{name}:{value['status']}"
                for check in detail["transport64_audit"]
                .get("variants", {})
                .get("native", {})
                .get("checks", [])
                for name, value in check["components"].items()
                if value["status"] != "PASS"
            ]
            print(
                item["case"],
                item["start"],
                "dtypes:",
                detail["dtypes"],
                "center deltas:",
                detail["center_max_abs_delta"],
                "transport64 unresolved:",
                unresolved64,
            )
        print(
            "Transport precision diagnostic only. No optimizer, no override of the original audit, no selection or promotion."
        )
        return
    rows = []
    for case in sorted((root / "cases").glob("*")):
        for p in sorted(case.rglob("SUMMARY.json")):
            rel = p.relative_to(case)
            summary = json.loads(p.read_text())
            used = int(rel.parts[1][6:]) if len(rel.parts) == 3 else 0
            history = p.parent.parent / "optimization.csv"
            attempts = (
                pd.read_csv(history) if used and history.exists() else pd.DataFrame()
            )
            attempts = (
                attempts[attempts.decoder_draws <= used]
                if not attempts.empty
                else attempts
            )
            rows.append(
                dict(
                    case=case.name,
                    variant=rel.parts[0],
                    decoder_draws=used,
                    K=summary["pooled_draws"],
                    ess_fraction=summary["raw_ess"]["fraction_median"],
                    bad_k=summary["pareto_k"]["gt_0p7_or_nonfinite_fraction"],
                    max_weight=summary["maximum_raw_weight"]["p90"],
                    evidence_delta=summary["replicate_abs_log_evidence_delta"],
                    negative_elbo=summary["negative_elbo"],
                    residual_rms=summary["residual_rms"],
                    attempts=len(attempts),
                    applied_updates=int(attempts.update_applied.sum())
                    if "update_applied" in attempts
                    else len(attempts),
                )
            )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("No pilot evaluations. Non-PASS native audit blocks every optimizer.")
    print(
        "Development comparison, not posterior qualification. No selection or promotion."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)


if __name__ == "__main__":
    main()
