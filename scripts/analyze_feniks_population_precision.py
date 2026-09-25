"""Analyze the lightweight rsync mirror without remote banks or any refitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from euclid_dsps.amortized.forward_population import PHYSICAL
from scripts.feniks_avi_experiments import read, sha, write
from scripts.feniks_population_precision import (
    ARMS,
    RATIOS,
    fixed_metrics,
    load_geometry,
    metric_reference,
)

LABELS = {
    "z_obs": "Redshift",
    "log10_stellar_mass": "log stellar\nmass",
    "log10_stellar_metallicity": "log stellar\nmetallicity",
    "dust_av": "Dust Av",
    "dust_delta": "Dust slope",
}


def verify_mirror(root: Path) -> dict:
    """Verify present files; excluded large arrays are explicitly not verified."""
    checked, absent = [], []
    repeats = read(root / "MANIFEST.json")["settings"]["bootstraps"]
    receipts = [root / "cache/FINAL.json", root / "decoder/FINAL.json"]
    receipts += [root / "metric" / arm / "FINAL.json" for arm in ARMS]
    receipts += [
        root / "bootstrap" / f"repeat_{repeat:03d}" / f"{arm}_{ratio}" / "FINAL.json"
        for repeat in range(repeats)
        for arm in ARMS
        for ratio in RATIOS
    ]
    for path in receipts:
        receipt = read(path)
        if receipt.get("status") != "COMPLETE":
            raise ValueError(f"incomplete receipt: {path}")
        for name, digest in receipt["artifacts"].items():
            artifact = path.parent / name
            if not artifact.is_file():
                absent.append(str(artifact.relative_to(root)))
            elif sha(artifact) != digest:
                raise ValueError(f"corrupted mirror artifact: {artifact}")
            else:
                checked.append(str(artifact.relative_to(root)))
    return dict(verified=len(checked), missing_not_verified=absent)


def decoder_comparison(root: Path) -> dict:
    model = read(root / "decoder/decoder_model.json")["model"]
    baseline = model.get("photometry_integrator", "legacy_trapezoid_v1")
    frame = pd.read_csv(root / "decoder/residuals.csv")
    joined = frame[frame.variant == "baseline"].merge(
        frame[frame.variant == "merged"],
        on=["row", "band"],
        suffixes=("_baseline", "_merged"),
        validate="one_to_one",
    )
    if joined.empty or len(joined) * 2 != len(frame):
        raise ValueError("decoder variants do not cover identical row-band pairs")
    receipt = read(root / "decoder/FINAL.json")
    return dict(
        baseline_integrator=baseline,
        comparison_integrator="merged_gauss4_v1",
        distinct_integrators=baseline != "merged_gauss4_v1",
        maximum_flux_difference=float(
            abs(joined.decoded_flux_baseline - joined.decoded_flux_merged).max()
        ),
        catalogue_compatibility_pass=receipt["merged_p95_max"]
        <= read(root / "MANIFEST.json")["settings"]["contracts"][
            "decoder_p95_abs_sigma"
        ],
        numerical_improvement_demonstrated=baseline != "merged_gauss4_v1"
        and receipt["merged_p95_max"] < receipt["baseline_p95_max"],
    )


def analyze(root: Path, draws: int = 65536) -> None:
    import jax

    if not jax.config.x64_enabled:
        raise ValueError("set JAX_ENABLE_X64=true for matching metric arithmetic")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    settings = read(root / "MANIFEST.json")["settings"]
    integrity = verify_mirror(root)
    write(out / "mirror_integrity.json", integrity)
    precision = pd.read_csv(root / "report/metric_precision.csv")
    boot = pd.read_csv(root / "report/bootstrap.csv")
    expected = settings["bootstraps"] * len(ARMS) * len(RATIOS)
    if len(boot) != expected or boot.duplicated(["repeat", "arm", "ratio"]).any():
        raise ValueError("missing or duplicated bootstrap cells")
    geometry, theta = load_geometry(root)
    scale, directions = metric_reference(root)
    full, fits = {}, []
    for arm in ARMS:
        for ratio in RATIOS:
            key = (arm, ratio)
            path = root / f"cache/full_{arm}_{ratio}.npz"
            with np.load(path) as arrays:
                full[key] = arrays["parent"]
            fits.append(dict(arm=arm, ratio=ratio, **read(path.with_suffix(".json"))))
    fit_frame = pd.DataFrame(fits)
    fit_frame.to_csv(out / "full_fit_certificates.csv", index=False)

    seeds = settings["metric_seeds"]
    replay = []
    for (arm, ratio), weights in full.items():
        values = fixed_metrics(
            theta(weights, 4096, seeds[0]),
            theta(geometry["true_parent"], 4096, seeds[0]),
            scale,
            directions,
        )
        original = precision[
            (precision.arm == arm)
            & (precision.left == ratio)
            & (precision.right == "truth")
            & (precision.draws == 4096)
            & (precision.seed == seeds[0])
        ].iloc[0]
        replay.append(abs(values["parent_sw"] - original.parent_sw))
    if max(replay) > 1e-8:
        raise ValueError(f"local metric replay differs from cluster: {max(replay)}")

    rows = []
    for seed in seeds:
        references = {key: theta(w, draws, seed) for key, w in full.items()}
        for arm in ARMS:
            for ratio in RATIOS:
                for repeat in range(settings["bootstraps"]):
                    cell = (
                        root / "bootstrap" / f"repeat_{repeat:03d}" / f"{arm}_{ratio}"
                    )
                    with np.load(cell / "weights.npz") as arrays:
                        values = fixed_metrics(
                            theta(arrays["parent"], draws, seed),
                            references[(arm, ratio)],
                            scale,
                            directions,
                        )
                    rows.append(
                        dict(
                            arm=arm,
                            ratio=ratio,
                            repeat=repeat,
                            seed=seed,
                            draws=draws,
                            **values,
                        )
                    )
        print(f"Re-evaluated saved weights: seed={seed}, draws={draws}", flush=True)
        pd.DataFrame(rows).to_csv(out / "bootstrap_precision.csv", index=False)
    fine = pd.DataFrame(rows)
    # Multiple evaluation seeds are NOT new independent catalogue bootstraps.
    per_repeat = (
        fine.groupby(["arm", "ratio", "repeat"])
        .parent_sw.agg(mean="mean", seed_std="std", minimum="min", maximum="max")
        .reset_index()
    )
    per_repeat.to_csv(out / "bootstrap_by_repeat.csv", index=False)
    summary = per_repeat.groupby(["arm", "ratio"]).agg(
        repeats=("repeat", "count"),
        median_sw=("mean", "median"),
        maximum_sw=("mean", "max"),
        maximum_seed_std=("seed_std", "max"),
    )
    summary.to_csv(out / "bootstrap_precision_summary.csv")

    closure_draws = int(precision.draws.max())
    closure = precision[
        (precision.draws == closure_draws)
        & (precision.right == "truth")
        & (precision.left != "truth")
    ]
    marginal = closure.groupby(["arm", "left"])[
        [f"w1_{name}" for name in PHYSICAL]
    ].mean()
    marginal.to_csv(out / "physical_marginal_errors.csv")
    comparison = decoder_comparison(root)
    write(out / "decoder_comparison.json", comparison)
    checks = dict(
        metric_replay_max_difference=max(replay),
        full_fit_maximum_kkt=float(fit_frame.kkt_gap.max()),
        bootstrap_maximum_kkt=float(boot.kkt_gap.max()),
        full_fit_maximum_sum_error=float(
            abs(fit_frame[["parent_sum", "selected_sum"]] - 1).max().max()
        ),
        bootstrap_maximum_sum_error=float(
            abs(boot[["parent_sum", "selected_sum"]] - 1).max().max()
        ),
        full_fit_seconds=[
            float(fit_frame.elapsed_seconds.min()),
            float(fit_frame.elapsed_seconds.max()),
        ],
        bootstrap_fit_seconds=[
            float(boot.elapsed_seconds.min()),
            float(boot.elapsed_seconds.max()),
        ],
        bootstrap_draws=draws,
        bootstrap_repeats=settings["bootstraps"],
        classifier_retrained=False,
        population_refitted=False,
        dsps_called=False,
        alpha_resampled=False,
        full_posterior_validated=False,
        production_ready=False,
        decoder=comparison,
    )
    write(out / "CHECKS.json", checks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, arm in zip(axes, ARMS, strict=True):
        for ratio, color in (("exact", "#555555"), ("photometric", "#167a9f")):
            table = (
                precision[
                    (precision.arm == arm)
                    & (precision.left == ratio)
                    & (precision.right == "truth")
                ]
                .groupby("draws")
                .parent_sw.agg(["mean", "std"])
            )
            axis.errorbar(
                table.index,
                table["mean"],
                yerr=table["std"],
                color=color,
                marker="o",
                label=ratio,
            )
        axis.set(
            xscale="log",
            xlabel="Evaluation draws",
            ylabel="Parent SW / fixed IQR",
            title=arm.replace("_", " "),
            ylim=(0, None),
        )
        axis.legend()
        axis.grid(alpha=0.2)
        axis.set_xticks(
            settings["metric_draws"], [f"{n:,}" for n in settings["metric_draws"]]
        )
    fig.suptitle("Same fitted densities; only measurement precision changes")
    fig.tight_layout()
    fig.savefig(out / "01_metric_precision.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=True)
    for axis, (arm, ratio) in zip(axes, full, strict=True):
        coarse = boot[(boot.arm == arm) & (boot.ratio == ratio)].set_index("repeat")
        group = per_repeat[(per_repeat.arm == arm) & (per_repeat.ratio == ratio)]
        for item in group.itertuples():
            axis.plot(
                [0, 1],
                [coarse.loc[item.repeat, "parent_sw"], item.mean],
                color="#999999",
                alpha=0.55,
            )
        axis.scatter(
            np.zeros(len(coarse)), coarse.parent_sw, color="#d38b28", label="1 seed"
        )
        axis.errorbar(
            np.ones(len(group)),
            group["mean"],
            yerr=group.seed_std,
            fmt="o",
            color="#167a9f",
            label=f"{len(seeds)} seed mean +/- SD",
        )
        axis.axhline(
            settings["contracts"]["historical_bootstrap_sw"],
            color="#333333",
            ls="--",
            lw=0.8,
        )
        axis.set(
            xticks=[0, 1],
            xticklabels=[str(settings["bootstrap_metric_draws"]), str(draws)],
            xlabel="Evaluation draws",
            title=f"{arm.split('_')[0]} / {ratio}",
        )
    axes[0].set_ylabel("SW to full-catalogue fit / fixed IQR")
    axes[-1].legend(fontsize=7)
    fig.suptitle(
        f"{settings['bootstraps']} catalogue resamples per arm, unchanged weights (dashed: historical reference)"
    )
    fig.tight_layout()
    fig.savefig(out / "02_bootstrap_precision.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    im = axis.imshow(marginal.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0)
    axis.set(
        xticks=range(5),
        xticklabels=[LABELS[n] for n in PHYSICAL],
        yticks=range(len(marginal)),
        yticklabels=[f"{a.split('_')[0]} / {r}" for a, r in marginal.index],
        title=f"Parent marginal errors against in-family truth ({closure_draws} draws, mean of {len(seeds)} seeds)",
    )
    for (i, j), value in np.ndenumerate(marginal.to_numpy()):
        axis.text(
            j,
            i,
            f"{value:.4f}",
            ha="center",
            va="center",
            color="white" if value > 0.04 else "black",
        )
    fig.colorbar(im, ax=axis, label="1D Wasserstein / fixed IQR")
    fig.tight_layout()
    fig.savefig(out / "03_physical_errors.png", dpi=160)
    plt.close(fig)

    residuals = pd.read_csv(root / "decoder/residuals.csv")
    residuals = residuals[residuals.variant == "baseline"].copy()
    bands = pd.read_csv(root / "decoder/bands.csv")
    bands = bands[bands.variant == "baseline"].sort_values("p95_abs", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(bands.band, bands.p95_abs, color="#167a9f")
    axes[0].axhline(
        settings["contracts"]["decoder_p95_abs_sigma"], color="black", ls="--"
    )
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].set(
        title="Catalogue mismatch persists",
        ylabel="p95 absolute residual / reported sigma",
    )
    for band in bands.band.iloc[:3]:
        part = residuals[residuals.band == band]
        axes[1].scatter(part.snr, abs(part.residual_sigma), s=16, alpha=0.7, label=band)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Reference flux / reported sigma",
        ylabel="Absolute residual / reported sigma",
        title="Bright objects expose small relative errors",
    )
    axes[1].legend()
    comparison_label = (
        "distinct integrators"
        if comparison["distinct_integrators"]
        else "SAME integrator"
    )
    if comparison["distinct_integrators"]:
        fig.suptitle(
            f"{comparison['baseline_integrator']} versus {comparison['comparison_integrator']}"
        )
    else:
        fig.suptitle(
            f"Baseline and comparison: {comparison['baseline_integrator']} ({comparison_label})"
        )
    fig.tight_layout()
    fig.savefig(out / "04_decoder_contract.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(), flush=True)
    print(f"Analysis: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--draws", type=int, default=65536)
    args = parser.parse_args()
    if args.draws < 16:
        parser.error("--draws must be >=16")
    analyze(args.root.resolve(), args.draws)
