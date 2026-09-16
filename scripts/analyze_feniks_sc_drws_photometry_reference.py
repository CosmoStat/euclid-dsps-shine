#!/usr/bin/env python3
"""Summarize photometry qualification, or replay exported assets on CPU."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(source, output):
    import jax
    import numpy as np

    from euclid_dsps.amortized.local_vi_diagnostic import Budget
    from euclid_dsps.amortized.photometry_reference import (
        analyze_snapshot,
        finite_report,
        write_progress,
    )

    receipt = json.loads((source / "SNAPSHOT.json").read_text())
    archive = source / "FIXED_SPECTRA.npz"
    if digest(archive) != receipt["sha256"]:
        raise ValueError("changed spectrum export")
    if jax.default_backend() != "cpu":
        raise ValueError("CPU replay: set JAX_PLATFORMS=cpu before launching")
    output.mkdir(parents=True, exist_ok=False)
    with np.load(archive, allow_pickle=False) as bank:
        arrays = dict(bank)
    budget = Budget(2400, 1000)
    result, _ = analyze_snapshot(
        arrays,
        budget,
        progress=lambda i, s, r, p: write_progress(output, i, s, r, p, budget),
    )
    result.update(
        source_snapshot_sha256=receipt["sha256"],
        source_code_commit=receipt["code_commit"],
        replay_script_sha256=digest(__file__),
        replay_source_sha256={
            name: digest(Path(__file__).resolve().parents[1] / name)
            for name in (
                "euclid_dsps/photometry_quadrature.py",
                "euclid_dsps/amortized/photometry_reference.py",
                "euclid_dsps/model.py",
            )
        },
        jax_version=jax.__version__,
        budget=budget.snapshot(),
    )
    (output / "PHOTOMETRY_REFERENCE.json").write_text(
        json.dumps(finite_report(result), indent=2, allow_nan=False) + "\n"
    )
    final = dict(
        status=result["status"],
        scientific_promotion=False,
        artifacts={
            n: dict(sha256=digest(output / n))
            for n in ("PHOTOMETRY_REFERENCE.json", "photometry_reference.csv")
        },
    )
    (output / "FINAL.json").write_text(json.dumps(final, indent=2) + "\n")


def summarize(root, point):
    import pandas as pd

    final = json.loads((root / "FINAL.json").read_text())
    if final["status"] != "PHOTOMETRY_REFERENCE_COMPLETE":
        raise ValueError("photometry reference not complete")
    for name, entry in final["artifacts"].items():
        if digest(root / name) != entry["sha256"]:
            raise ValueError(f"changed artifact: {name}")
    report = json.loads((root / "PHOTOMETRY_REFERENCE.json").read_text())
    print(report["status"], "numerical_checks=", report["numerical_reference_checks"])
    print(
        "next_stage=",
        report["next_stage"],
        "scientific_promotion=",
        report["scientific_promotion"],
    )
    for entry in report["points"]:
        print("\nPOINT", entry["point_index"], "z=", entry["physical_z"])
        for key in (
            "numerical_reference_checks",
            "max_abs_legacy_reference_delta_sigma",
            "max_abs_normalization_delta_sigma",
            "max_abs_order4_order8_delta_sigma",
            "max_abs_eager_jit_projection_delta_sigma",
            "max_abs_fixed_projection_full_model_delta_sigma",
        ):
            print(key, entry[key])
        print("integrations:", entry["integration_comparisons"])
        print(
            "candidate gradient:",
            {
                s: sum(x["status"] == s for x in entry["candidate_gradient_checks"])
                for s in ("PASS", "FAIL", "INCONCLUSIVE")
            },
        )
        print(
            "nearest knot delta_z:",
            {k: v["nearest_knot_distance_z"] for k, v in entry["knots"].items()},
        )
    df = pd.read_csv(root / "photometry_reference.csv")
    df = df[df.point_index == point].copy()
    df["candidate_ad_vs_reference_fd"] = abs(
        df.ad_sigma_per_z - df.analytic_fd_sigma_per_z
    )
    df["self_ad_vs_fd"] = abs(df.ad_sigma_per_z - df.fd_sigma_per_z)
    print("\nPoint", point, "maximum per-band derivative disagreement in sigma/z:")
    print(
        df.groupby(["method", "step_index", "physical_z_step"])[
            ["self_ad_vs_fd", "candidate_ad_vs_reference_fd"]
        ]
        .max()
        .to_string()
    )
    print("No automatic decoder replacement, bank reuse, NPE or population training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--point", type=int, default=0)
    parser.add_argument("--replay-out", type=Path)
    args = parser.parse_args()
    if args.replay_out:
        replay(args.root, args.replay_out)
    summarize(args.replay_out or args.root, args.point)
