"""Inspect synchronized small artifacts without requiring remote banks or models."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, sha, write


def run(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contract = sha(root / "MANIFEST.json")
    # This review's prose is specific to the synchronized experiment, not a generic verdict.
    if contract != "7ffdc8e285389a1d87e74d6b6d3898dddde404d0803d327421ebca1a540b4d2b":
        raise ValueError("This evidence review is specific to the 20260927_163901 run")
    verified, missing = set(), set()
    for receipt in root.rglob("FINAL.json"):
        if "analysis" in receipt.relative_to(root).parts:
            continue
        result = read(receipt)
        if result["contract"] != contract or result["status"] != "COMPLETE":
            raise ValueError(f"Invalid receipt: {receipt}")
        for name, expected in result["artifacts"].items():
            path = (receipt.parent / name).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f"Unexpected artifact path: {path}")
            if path.exists():
                if sha(path) != expected:
                    raise ValueError(f"Artifact changed: {path}")
                verified.add(str(path.relative_to(root.resolve())))
            else:
                missing.add(str(path.relative_to(root.resolve())))
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    names = ["legacy", "affine_mass", "local_128", "local_256"]
    rows = [read(root / n / "capacity/FINAL.json") for n in names]
    marginals = pd.concat(
        [
            pd.read_csv(root / n / "capacity/marginals.csv").assign(candidate=n)
            for n in names
        ]
    )
    photo = pd.read_csv(root / "replay/paired_photometry.csv")
    band = read(root / "prepare/replay_selection.json")["band"]
    r = photo[photo.band.eq(band)].pivot(
        index=["pair", "cohort"], columns="case", values="noiseless_flux"
    )
    amps = []
    for cohort, group in r.groupby(level="cohort"):
        for case in ("legacy", "affine_mass", "local_128"):
            logratio = np.log10(group[case] / group.anchor)
            amps.append(
                dict(
                    cohort=cohort,
                    case=case,
                    median_flux_ratio=float(np.median(group[case] / group.anchor)),
                    median_abs_log10_ratio=float(np.median(abs(logratio))),
                    q90_abs_log10_ratio=float(np.quantile(abs(logratio), 0.9)),
                )
            )
    pd.DataFrame(amps).to_csv(out / "paired_amplification.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, group in zip(axes, ("physical", "sfh"), strict=True):
        part = marginals[marginals.group.eq(group)].pivot(
            index="parameter", columns="candidate", values="w1_over_truth_iqr"
        )
        for name in names:
            ax.plot(range(len(part)), part[name], "o-", label=name)
        ax.set_xticks(range(len(part)), part.index, rotation=45, ha="right", fontsize=8)
        ax.set_title(group)
        ax.set_ylabel("Wasserstein / truth IQR")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "physical_sfh_errors.png", dpi=150)
    plt.close(fig)
    result = dict(
        verified_artifacts=sorted(verified),
        unavailable_artifacts=sorted(missing),
        capacity=[
            {
                k: r[k]
                for k in (
                    "candidate",
                    "gates",
                    "limits",
                    "baseline",
                    "physical_sw",
                    "validation_cdf_max",
                    "physical_marginal_max",
                )
            }
            for r in rows
        ],
        replay=read(root / "replay/FINAL.json"),
        population_recovery_validated=False,
    )
    write(out / "AUDIT.json", result)
    text = """# Reference redesign review, 2026-09-27

## Findings

- local_256 passes all five physical marginal tolerances (maximum W1/IQR 0.0651).
- It fails validation CDF 0.0320 > 0.0300, driven by stellar mass; the train optimum is 0.02301.
- Physical SW 0.05059 exceeds 0.035 and the empirical comparator 0.01340.
- The LP optimizes a maximum CDF error in transformed coordinates, not physical SW. This is not a certificate of minimum attainable physical SW.
- The local family improves substantially; do not infer that 512 components are necessary from this test.
- All 32 deliberately selected extreme rows already have native-anchor r flux above the validation q99.9. They are not all created by kernel perturbations. This targeted sample does not estimate their population abundance.
- Local physical kernels reduce paired flux perturbations; affine log-mass alone has little effect on most pairs. The analytic finite-mass-moment correction remains valid.
- SFH population errors remain large (last contrast W1/IQR 0.889). Broad individual SFH posteriors are acceptable; a wrong population conditional can still bias photometric population learning. No such conditional has been validated here.

## Next

Keep the local_256 basis fixed. Fit a TRAIN-only integrated CDF objective in physical IQR units, while bounding the original TRAIN CDF residual; retain the same validation thresholds. Use independent draws for evaluation. This separates objective mismatch from geometry without a new basis sweep or DSPS.

If qualified, automatically generate one new compatible reference bank, retrain the classifier, fit the parent from observed fluxes only, and measure parent/selected/photometric closure. Truth-fitted diagnostic weights must never be used as the production prior. Do not retrain the individual posterior on an unqualified parent.

## Figures

1. ../local_256/capacity/physical.png: direct physical truth comparison.
2. physical_sfh_errors.png: physical improvements versus unresolved SFH population mismatch.
3. ../replay/paired_bright_replay.png: anchor brightness versus kernel amplification, not population abundance.
4. ../report/reference_capacity_comparison.png: all candidates, remembering CDF geometry differs for legacy.
"""
    text += f"\nVerified {len(verified)} available receipt artifacts; {len(missing)} excluded artifacts unavailable.\n"
    (out / "ANALYSIS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    run(p.parse_args().root.resolve())
