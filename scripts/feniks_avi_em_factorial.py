"""Factorial Q0/Q4 x P0/P4 diagnosis after selection-corrected AVI EM."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kstest, wasserstein_distance

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
from scripts.feniks_avi_em import _components, _files_hash, _manifest, _require_final
from scripts.feniks_avi_experiments import read, sha, write

PHYSICAL = tuple(FENIKS_SPLINE15D_PARAMETERS[:5])
SFH = tuple(FENIKS_SPLINE15D_PARAMETERS[5:])
VARIANTS = ("Q0_P0", "Q4_P0", "Q0_P4", "Q4_P4")
POPULATION_SAMPLES = 65536


def _weighted_truth(path: Path, *, count: int, seed: int) -> pd.DataFrame:
    names = list(FENIKS_SPLINE15D_PARAMETERS)
    frame = pd.read_parquet(path)
    values = frame[names].to_numpy(np.float64)
    weight = (
        frame["population_weight"].to_numpy(np.float64)
        if "population_weight" in frame
        else np.ones(len(frame), dtype=np.float64)
    )
    if (
        not len(frame)
        or not np.isfinite(values).all()
        or not np.isfinite(weight).all()
        or np.any(weight < 0)
        or weight.sum() <= 0
    ):
        raise ValueError(f"invalid weighted truth population: {path}")
    rng = np.random.default_rng(seed)
    index = rng.choice(len(frame), size=count, replace=True, p=weight / weight.sum())
    result = frame.iloc[index][names].reset_index(drop=True)
    result.insert(0, "row_index", np.arange(count, dtype=np.int64))
    result.insert(0, "object_id", result.row_index)
    return result


def prepare(em_root: Path, output: Path, *, particles: int = 4096) -> None:
    """Freeze the completed EM endpoints into a four-cell inference contract."""
    manifest = _manifest(em_root)
    if output.exists():
        raise FileExistsError(output)
    if particles < 256 or particles % 8:
        raise ValueError("particles must be >=256 and divisible by eight")
    final_cycle = int(manifest["cycles"])
    if final_cycle != 4:
        raise ValueError(
            f"Q0/Q4 factorial requires exactly four completed EM cycles, got {final_cycle}"
        )
    q0, p0 = _components(em_root, 0)
    q4, p4 = _components(em_root, final_cycle)
    _require_final(
        em_root / "cycles" / f"cycle_{final_cycle:02d}" / "FINAL.json",
        "EM_CYCLE_COMPLETE",
    )
    rows = np.load(em_root / "validation.npy", allow_pickle=False)
    names = list(FENIKS_SPLINE15D_PARAMETERS)
    truth = (
        pd.read_parquet(manifest["validation_catalog"], columns=names).iloc[rows].copy()
    )
    if len(rows) < 2 or not np.isfinite(truth[names].to_numpy(np.float64)).all():
        raise ValueError("finite fixed-cohort truth is required for the report")
    truth.insert(0, "row_index", rows)
    truth.insert(0, "object_id", rows)

    output.mkdir(parents=True)
    (output / "logs").mkdir()
    truth.to_parquet(output / "inference_truth.parquet", index=False)
    population_count = POPULATION_SAMPLES
    _weighted_truth(
        Path(manifest["true_selected"]), count=population_count, seed=26091401
    ).to_parquet(output / "selected_population_truth.parquet", index=False)
    _weighted_truth(
        Path(manifest["true_parent"]), count=population_count, seed=26091402
    ).to_parquet(output / "parent_population_truth.parquet", index=False)

    variants = [
        {
            "name": "Q0_P0",
            "encoder": str(q0.resolve()),
            "prior": str(p0.resolve()),
            "prior_label": "P0",
            "emit_prior_population": True,
        },
        {
            "name": "Q4_P0",
            "encoder": str(q4.resolve()),
            "prior": str(p0.resolve()),
            "prior_label": "P0",
            "emit_prior_population": False,
        },
        {
            "name": "Q0_P4",
            "encoder": str(q0.resolve()),
            "prior": str(p4.resolve()),
            "prior_label": "P4",
            "emit_prior_population": True,
        },
        {
            "name": "Q4_P4",
            "encoder": str(q4.resolve()),
            "prior": str(p4.resolve()),
            "prior_label": "P4",
            "emit_prior_population": False,
        },
    ]
    dependencies = [
        em_root / "MANIFEST.json",
        em_root / "source_config.yaml",
        em_root / "train.npy",
        em_root / "validation.npy",
        em_root / "cycles" / f"cycle_{final_cycle:02d}" / "FINAL.json",
        q0,
        p0,
        q4,
        p4,
        Path(manifest["true_selected"]),
        Path(manifest["true_parent"]),
        output / "inference_truth.parquet",
        output / "selected_population_truth.parquet",
        output / "parent_population_truth.parquet",
    ]
    write(
        output / "MANIFEST.json",
        {
            "version": 1,
            "suite": "avi_em_q0_q4_p0_p4_factorial_v1",
            "training": str(em_root.resolve()),
            "source_training": manifest["source_training"],
            "source": manifest["source"],
            "config": str((em_root / "source_config.yaml").resolve()),
            "train_indices": str((em_root / "train.npy").resolve()),
            "validation_indices": str((em_root / "validation.npy").resolve()),
            "validation_catalog": manifest["validation_catalog"],
            "variants": variants,
            "particles": int(particles),
            "saved_draws": 512,
            "replicas": 2,
            "objects": len(rows),
            "seed": 26091400,
            "prior_samples": population_count,
            "prior_selected_resamples": population_count,
            "prior_mira_objects": min(512, len(rows)),
            "factorial_contract": {
                "Q0_P0": "baseline",
                "Q4_P0": "encoder effect at fixed P0",
                "Q0_P4": "prior effect at fixed Q0",
                "Q4_P4": "complete EM endpoint",
            },
            "posterior_weight_contract": "ordinary full_15d logtarget-logproposal",
            "selection_in_object_weights": False,
            "truth_used_for_sampling_or_weighting": False,
            "truth_role": "dependent report only",
            "scientific_promotion": False,
            "hashes": _files_hash(dependencies),
        },
    )
    print(f"Prepared {len(variants)} factorial inference cells at {output}", flush=True)


def infer(output: Path, task: int) -> None:
    from scripts.feniks_avi_overnight import infer as run_inference

    run_inference(output, task)


def _cube(path: Path, truth_rows: np.ndarray) -> np.ndarray:
    names = list(FENIKS_SPLINE15D_PARAMETERS)
    frame = pd.read_parquet(path)
    counts = frame.groupby("row_index", sort=False).size()
    if not len(counts) or counts.nunique() != 1:
        raise ValueError(f"ragged posterior table: {path}")
    ordered = frame.sort_values(["row_index", "sample_id"])
    rows = ordered.row_index.drop_duplicates().to_numpy(np.int64)
    if not np.array_equal(rows, truth_rows):
        raise ValueError(f"posterior/truth identity mismatch: {path}")
    return ordered[names].to_numpy(np.float64).reshape(len(rows), counts.iloc[0], -1)


def _posterior_calibration(output: Path, manifest: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(FENIKS_SPLINE15D_PARAMETERS)
    truth_frame = pd.read_parquet(output / "inference_truth.parquet").sort_values(
        "row_index"
    )
    truth_rows = truth_frame.row_index.to_numpy(np.int64)
    truth = truth_frame[names].to_numpy(np.float64)
    levels = np.asarray((0.50, 0.68, 0.90, 0.95))
    pit_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    cubes: dict[tuple[str, int], np.ndarray] = {}
    for variant in manifest["variants"]:
        label = variant["name"]
        for replica in range(int(manifest["replicas"])):
            cube = _cube(output / "arms" / label / f"is_{replica}.parquet", truth_rows)
            cubes[(label, replica)] = cube
            pit = np.mean(cube <= truth[:, None, :], axis=1)
            for index, name in enumerate(names):
                ks = kstest(pit[:, index], "uniform")
                pit_rows.append(
                    {
                        "variant": label,
                        "replica": replica,
                        "parameter": name,
                        "group": "physical" if index < 5 else "sfh",
                        "pit_mean": float(pit[:, index].mean()),
                        "pit_ks": float(ks.statistic),
                        "pit_ks_pvalue": float(ks.pvalue),
                    }
                )
            for group, indices in (("physical", range(5)), ("sfh", range(5, 15))):
                index = list(indices)
                for level in levels:
                    tail = (1.0 - level) / 2.0
                    low = np.quantile(cube[:, :, index], tail, axis=1)
                    high = np.quantile(cube[:, :, index], 1.0 - tail, axis=1)
                    coverage_rows.append(
                        {
                            "variant": label,
                            "replica": replica,
                            "group": group,
                            "nominal": level,
                            "coverage": float(
                                np.mean(
                                    (truth[:, index] >= low) & (truth[:, index] <= high)
                                )
                            ),
                        }
                    )
    pit_frame = pd.DataFrame(pit_rows)
    coverage_frame = pd.DataFrame(coverage_rows)
    pit_frame.to_csv(output / "report/factorial_pit.csv", index=False)
    coverage_frame.to_csv(output / "report/factorial_coverage.csv", index=False)

    fig, axes = plt.subplots(len(VARIANTS), len(PHYSICAL), figsize=(16, 9), sharex=True)
    bins = np.linspace(0, 1, 11)
    for row, variant in enumerate(VARIANTS):
        cube = cubes[(variant, 0)]
        pit = np.mean(cube <= truth[:, None, :], axis=1)
        for column, name in enumerate(PHYSICAL):
            axis = axes[row, column]
            axis.hist(pit[:, column], bins=bins, density=True, color="#3572A5")
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
            axis.set_title(name.replace("log10_", "").replace("_", " "))
            if column == 0:
                axis.set_ylabel(variant)
            if row == len(VARIANTS) - 1:
                axis.set_xlabel("PIT")
    fig.suptitle("Physical posterior PIT after ordinary importance sampling")
    fig.tight_layout()
    fig.savefig(output / "report/factorial_physical_pit.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis, group in zip(axes, ("physical", "sfh"), strict=True):
        data = coverage_frame[coverage_frame.group.eq(group)]
        for variant in VARIANTS:
            values = data[data.variant.eq(variant)].groupby("nominal").coverage.mean()
            axis.plot(values.index, values.values, marker="o", label=variant)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(group)
        axis.set_xlabel("Nominal central coverage")
        axis.set_ylabel("Empirical coverage")
        axis.grid(alpha=0.2)
    axes[-1].legend(frameon=False)
    fig.suptitle("Factorial posterior coverage")
    fig.tight_layout()
    fig.savefig(output / "report/factorial_coverage.png", dpi=190)
    plt.close(fig)


def _population_closure(output: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    from scripts.feniks_avi_overnight import population_weights

    names = list(FENIKS_SPLINE15D_PARAMETERS)
    selected_truth = pd.read_parquet(output / "selected_population_truth.parquet")
    parent_truth = pd.read_parquet(output / "parent_population_truth.parquet")
    rows: list[dict[str, Any]] = []
    for variant in manifest["variants"]:
        label = variant["name"]
        for replica in range(int(manifest["replicas"])):
            parts = sorted(
                (output / "arms" / label / f"bank_{replica}").glob("*.parquet")
            )
            if not parts:
                raise FileNotFoundError(f"missing bank for {label} replica {replica}")
            bank = pd.concat(
                [pd.read_parquet(path) for path in parts], ignore_index=True
            )
            selected_weight, parent_weight, support = population_weights(bank)
            values = bank[names].to_numpy(np.float64)
            for population, weight, reference in (
                ("selected", selected_weight, selected_truth),
                ("parent", parent_weight, parent_truth),
            ):
                for index, name in enumerate(names):
                    target = reference[name].to_numpy(np.float64)
                    scale = max(
                        float(np.quantile(target, 0.75) - np.quantile(target, 0.25)),
                        1.0e-6,
                    )
                    rows.append(
                        {
                            "variant": label,
                            "replica": replica,
                            "population": population,
                            "parameter": name,
                            "group": "physical" if index < 5 else "sfh",
                            "wasserstein_over_truth_iqr": float(
                                wasserstein_distance(
                                    values[:, index], target, u_weights=weight
                                )
                                / scale
                            ),
                            **support,
                        }
                    )
    result = pd.DataFrame(rows)
    result.to_csv(output / "report/factorial_population_closure.csv", index=False)
    return result


def _factorial_effects(cells: pd.DataFrame) -> pd.DataFrame:
    values = cells.set_index("variant")
    effects = (
        ("Q effect at P0", "Q4_P0", "Q0_P0"),
        ("Q effect at P4", "Q4_P4", "Q0_P4"),
        ("P effect at Q0", "Q0_P4", "Q0_P0"),
        ("P effect at Q4", "Q4_P4", "Q4_P0"),
    )
    metrics = [name for name in cells if name != "variant"]
    rows = []
    for metric in metrics:
        for effect, left, right in effects:
            rows.append(
                {
                    "effect": effect,
                    "metric": metric,
                    "delta": float(
                        values.loc[left, metric] - values.loc[right, metric]
                    ),
                }
            )
        rows.append(
            {
                "effect": "QxP interaction",
                "metric": metric,
                "delta": float(
                    values.loc["Q4_P4", metric]
                    - values.loc["Q4_P0", metric]
                    - values.loc["Q0_P4", metric]
                    + values.loc["Q0_P0", metric]
                ),
            }
        )
    return pd.DataFrame(rows)


def _summary(output: Path, manifest: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_rows = []
    for variant in manifest["variants"]:
        frame = pd.read_csv(output / "arms" / variant["name"] / "metrics.csv")
        for replica, values in frame.groupby("replica"):
            metric_rows.append(
                {
                    "variant": variant["name"],
                    "replica": int(replica),
                    "median_ess_fraction": float(values.ess_fraction.median()),
                    "q10_ess_fraction": float(values.ess_fraction.quantile(0.1)),
                    "median_max_weight": float(values.max_weight.median()),
                    "median_raw_predictive_rms": float(
                        values.raw_predictive_rms.median()
                    ),
                    "median_is_predictive_rms": float(
                        values.is_predictive_rms.median()
                    ),
                    "median_log_evidence": float(values.log_evidence.median()),
                    "median_resampled_unique": float(values.resampled_unique.median()),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "report/factorial_support.csv", index=False)
    population = _population_closure(output, manifest)
    _posterior_calibration(output, manifest)

    mira = pd.read_csv(output / "report/posterior_mira_scores.csv")
    mira["variant"] = mira.model.str.replace(r"_(raw|is)$", "", regex=True)
    mira["kind"] = mira.model.str.extract(r"_(raw|is)$")[0]
    mira_summary = mira.groupby(["variant", "kind", "group"], as_index=False).agg(
        score=("score", "mean"), bootstrap_std=("bootstrap_std", "mean")
    )
    mira_summary.to_csv(output / "report/factorial_mira.csv", index=False)
    closure_summary = (
        population.groupby(
            ["variant", "replica", "population", "group"], as_index=False
        )
        .wasserstein_over_truth_iqr.median()
        .rename(columns={"wasserstein_over_truth_iqr": "median_wasserstein_over_iqr"})
    )
    closure_summary.to_csv(
        output / "report/factorial_population_summary.csv", index=False
    )

    averaged = metrics.groupby("variant", sort=False).mean(numeric_only=True)
    mira_cells = mira_summary.pivot(
        index="variant", columns=["group", "kind"], values="score"
    )
    mira_cells.columns = [f"mira_{group}_{kind}" for group, kind in mira_cells.columns]
    closure_cells = (
        closure_summary.groupby(["variant", "population"], as_index=False)
        .median_wasserstein_over_iqr.mean()
        .pivot(
            index="variant",
            columns="population",
            values="median_wasserstein_over_iqr",
        )
        .rename(
            columns={
                "selected": "selected_physical_w1_over_iqr",
                "parent": "parent_physical_w1_over_iqr",
            }
        )
    )
    cells = averaged.join(mira_cells).join(closure_cells).reindex(VARIANTS)
    cells.index.name = "variant"
    cells.reset_index().to_csv(output / "report/factorial_cells.csv", index=False)
    effects = _factorial_effects(cells.reset_index())
    effects.to_csv(output / "report/factorial_effects.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    averaged = averaged.reindex(VARIANTS)
    axes[0].bar(VARIANTS, averaged.median_ess_fraction, color="#0072B2")
    axes[0].set_ylabel("Median ESS fraction")
    axes[1].bar(VARIANTS, averaged.q10_ess_fraction, color="#009E73")
    axes[1].set_ylabel("Q10 ESS fraction")
    axes[2].bar(VARIANTS, averaged.median_is_predictive_rms, color="#D55E00")
    axes[2].set_ylabel("Median IS predictive RMS")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Factorial encoder/prior support")
    fig.tight_layout()
    fig.savefig(output / "report/factorial_support.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, group in zip(axes, ("physical_5d", "sfh_contrasts_10d"), strict=True):
        part = mira_summary[mira_summary.group.eq(group)]
        for offset, kind in enumerate(("raw", "is")):
            values = part[part.kind.eq(kind)].set_index("variant").reindex(VARIANTS)
            x = np.arange(len(VARIANTS)) + (offset - 0.5) * 0.34
            axis.bar(x, values.score, width=0.34, label=kind)
        axis.axhline(2 / 3, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(len(VARIANTS)), VARIANTS, rotation=25)
        axis.set_title(group)
        axis.set_ylabel("MIRA score")
    axes[-1].legend(frameon=False)
    fig.suptitle("Factorial posterior calibration")
    fig.tight_layout()
    fig.savefig(output / "report/factorial_mira.png", dpi=190)
    plt.close(fig)

    physical = closure_summary[closure_summary.group.eq("physical")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, population_name in zip(axes, ("selected", "parent"), strict=True):
        part = physical[physical.population.eq(population_name)]
        for replica in range(int(manifest["replicas"])):
            values = (
                part[part.replica.eq(replica)].set_index("variant").reindex(VARIANTS)
            )
            axis.plot(
                VARIANTS,
                values.median_wasserstein_over_iqr,
                marker="o",
                label=f"replica {replica}",
            )
        axis.set_title(population_name)
        axis.set_ylabel("Median W1 / truth IQR, physical 5D")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(alpha=0.2)
    axes[-1].legend(frameon=False)
    fig.suptitle("Factorial population closure")
    fig.tight_layout()
    fig.savefig(output / "report/factorial_population.png", dpi=190)
    plt.close(fig)

    physical_closure = (
        closure_summary[
            closure_summary.group.eq("physical")
            & closure_summary.population.eq("selected")
        ]
        .groupby("variant")
        .median_wasserstein_over_iqr.mean()
    )
    lines = [
        "# AVI EM factorial diagnosis",
        "",
        "The four cells use the same objects, likelihood, K=4096 and two replicas.",
        "Truth is read only by this dependent report.",
        "",
        "## Mean over replicas",
        "",
        "| Variant | Median ESS | Q10 ESS | Raw MIRA 5D | IS MIRA 5D | Selected 5D W1/IQR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = averaged.loc[variant]
        lines.append(
            f"| {variant} | {row.median_ess_fraction:.5f} | "
            f"{row.q10_ess_fraction:.5f} | "
            f"{cells.loc[variant, 'mira_physical_5d_raw']:.4f} | "
            f"{cells.loc[variant, 'mira_physical_5d_is']:.4f} | "
            f"{physical_closure.loc[variant]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation contract",
            "",
            "- Q4_P0 minus Q0_P0 isolates the encoder effect.",
            "- Q0_P4 minus Q0_P0 isolates the prior effect.",
            "- Q4_P4 is the complete EM endpoint.",
            "- `factorial_effects.csv` contains both conditional effects and the QxP interaction.",
            "- A fixed point is not sufficient: inspect MIRA, PIT, coverage, support and truth closure together.",
        ]
    )
    report_path = output / "report/FACTORIAL_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifacts = {
        str(path.relative_to(output)): sha(path)
        for path in sorted((output / "report").rglob("*"))
        if path.is_file() and path.name not in {"FINAL.json", "FACTORIAL_FINAL.json"}
    }
    write(
        output / "report/FACTORIAL_FINAL.json",
        {
            "status": "AVI_EM_FACTORIAL_REPORT_COMPLETE",
            "manifest_sha256": sha(output / "MANIFEST.json"),
            "variants": list(VARIANTS),
            "truth_used_for_sampling_or_weighting": False,
            "scientific_promotion": False,
            "artifacts": artifacts,
        },
    )
    print(report_path.read_text(), flush=True)


def report(output: Path) -> None:
    from scripts.feniks_avi_overnight import report as base_report

    manifest = read(output / "MANIFEST.json")
    for path, digest in manifest["hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"changed factorial input: {path}")
    base_report(output)
    _summary(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "infer", "report"))
    parser.add_argument("--em-root", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if args.em_root is None:
            parser.error("prepare requires --em-root")
        prepare(args.em_root.resolve(), root, particles=args.particles)
    elif args.mode == "infer":
        if args.task is None:
            parser.error("infer requires --task")
        infer(root, args.task)
    else:
        report(root)


if __name__ == "__main__":
    main()
