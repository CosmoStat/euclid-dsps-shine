"""Create readable scientific diagnostics for a completed AVI overnight run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, wasserstein_distance

from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS

PARAMETERS = tuple(FENIKS_SPLINE15D_PARAMETERS)
PHYSICAL = PARAMETERS[:5]
SFH = PARAMETERS[5:]
VARIANTS = (
    "B_source",
    "Q_source_refresh",
    "B_latest",
    "Q_latest_refresh",
    "B_scratch_prior",
)
PRIOR_VARIANTS = {
    "source": "B_source",
    "latest": "B_latest",
    "scratch": "B_scratch_prior",
}
DISPLAY = {
    "B_source": "B4 + source prior",
    "Q_source_refresh": "refreshed q + source prior",
    "B_latest": "B4 + learned prior",
    "Q_latest_refresh": "refreshed q + learned prior",
    "B_scratch_prior": "B4 + scratch prior",
}
SHORT = {
    "z_obs": r"$z$",
    "log10_stellar_mass": r"$\log_{10} M_\star$",
    "log10_stellar_metallicity": r"$\log_{10} Z_\star$",
    "dust_av": r"$A_V$",
    "dust_delta": r"$\delta_{\rm dust}$",
    **{
        f"sfh_dlog_sfr_{index:02d}": rf"$\Delta \log \mathrm{{SFR}}_{{{index}}}$"
        for index in range(1, 11)
    },
}
COLORS = {
    "parent": "#3F4650",
    "selected": "#2878B5",
    "learned": "#E07A28",
    "baseline": "#6B7280",
    "refresh": "#159A80",
    "negative": "#B34E59",
    "truth": "#D13F3F",
}


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "axes.edgecolor": "#C9CED6",
            "axes.labelcolor": "#20242A",
            "axes.titlecolor": "#20242A",
            "axes.grid": True,
            "grid.color": "#E8EBEF",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.color": "#515862",
            "ytick.color": "#515862",
            "savefig.bbox": "tight",
        }
    )


def _save(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor="white")
    plt.close(fig)


def _density(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    hist, edges = np.histogram(finite, bins=edges, density=True)
    hist = gaussian_filter1d(hist, sigma=1.1, mode="nearest")
    return (edges[:-1] + edges[1:]) / 2.0, hist


def _limits(*arrays: np.ndarray) -> tuple[float, float]:
    pooled = np.concatenate([np.asarray(item)[np.isfinite(item)] for item in arrays])
    low, high = np.quantile(pooled, [0.002, 0.998])
    if not high > low:
        low, high = float(np.min(pooled)), float(np.max(pooled) + 1.0)
    pad = 0.04 * (high - low)
    return float(low - pad), float(high + pad)


def _density_panel(axis, values, edges, *, color, label, fill=False, linestyle="-"):
    x, y = _density(values, edges)
    if fill:
        axis.fill_between(x, 0, y, color=color, alpha=0.20, linewidth=0)
    axis.plot(x, y, color=color, linewidth=2.0, linestyle=linestyle, label=label)


def plot_population_triplet(
    path: Path,
    parent: pd.DataFrame,
    selected: pd.DataFrame,
    learned: np.ndarray,
    names: tuple[str, ...],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    columns = 3 if len(names) <= 5 else 5
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for index, name in enumerate(names):
        axis = axes[index]
        p = parent[name].to_numpy(np.float64)
        s = selected[name].to_numpy(np.float64)
        q = learned[:, PARAMETERS.index(name)]
        low, high = _limits(p, s, q)
        edges = np.linspace(low, high, 75)
        _density_panel(
            axis,
            p,
            edges,
            color=COLORS["parent"],
            label="true parent C0",
        )
        _density_panel(
            axis,
            s,
            edges,
            color=COLORS["selected"],
            label="true selected catalogue",
            linestyle="--",
        )
        _density_panel(
            axis,
            q,
            edges,
            color=COLORS["learned"],
            label=r"learned selected prior $\beta p_\eta/\alpha$",
            fill=True,
        )
        axis.set_title(SHORT[name])
        axis.set_xlim(low, high)
        axis.set_yticks([])
        axis.set_ylabel("density" if index % columns == 0 else "")
    for axis in axes[len(names) :]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, fontsize=15, fontweight="semibold", y=1.035)
    fig.subplots_adjust(top=0.84, hspace=0.34, wspace=0.22)
    _save(fig, path)


def plot_posterior_population(
    path: Path,
    parent: pd.DataFrame,
    selected: pd.DataFrame,
    posterior: pd.DataFrame,
    names: tuple[str, ...],
    title: str,
) -> None:
    learned = posterior.loc[:, list(PARAMETERS)].to_numpy(np.float64)
    plot_population_triplet(path, parent, selected, learned, names, title)


def _mira_value(mira: pd.DataFrame, variant: str, kind: str, group: str) -> float:
    rows = mira.loc[mira.model.eq(f"{variant}_{kind}") & mira.group.eq(group), "score"]
    if len(rows) != 2:
        raise ValueError(f"expected two MIRA replicas for {variant}/{kind}/{group}")
    return float(rows.mean())


def build_run_summary(root: Path) -> pd.DataFrame:
    mira = pd.read_csv(root / "report/posterior_mira_scores.csv")
    joint = pd.read_csv(root / "report/joint_5d_distribution_metrics.csv")
    population = pd.read_csv(root / "report/population_distribution_metrics.csv")
    rows = []
    for variant in VARIANTS:
        metrics = pd.read_csv(root / "arms" / variant / "metrics.csv")
        selected_joint = joint.loc[
            joint.distribution.eq(f"{variant}_selected_5d")
        ].iloc[0]
        sfh_error = population.loc[
            population.source.eq(variant)
            & population.population.eq("selected_posterior_aggregate")
            & population.parameter.str.startswith("sfh_")
        ]
        rows.append(
            {
                "variant": variant,
                "display": DISPLAY[variant],
                "median_ess": float(metrics.ess.median()),
                "median_ess_fraction": float(metrics.ess_fraction.median()),
                "fraction_ess_below_5": float((metrics.ess < 5).mean()),
                "median_max_weight": float(metrics.max_weight.median()),
                "raw_predictive_rms": float(metrics.raw_predictive_rms.median()),
                "is_predictive_rms": float(metrics.is_predictive_rms.median()),
                "physical_5d_mira_raw": _mira_value(
                    mira, variant, "raw", "physical_5d"
                ),
                "physical_5d_mira_is": _mira_value(mira, variant, "is", "physical_5d"),
                "sfh_10d_mira_raw": _mira_value(
                    mira, variant, "raw", "sfh_contrasts_10d"
                ),
                "sfh_10d_mira_is": _mira_value(
                    mira, variant, "is", "sfh_contrasts_10d"
                ),
                "selected_5d_energy": float(
                    selected_joint.standardized_energy_distance
                ),
                "selected_5d_mmd2": float(selected_joint.rbf_mmd_squared),
                "selected_5d_correlation_rmse": float(selected_joint.correlation_rmse),
                "median_sfh_wasserstein_over_iqr": float(
                    sfh_error.wasserstein_over_target_iqr.median()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_scorecard(path: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    y = np.arange(len(summary))
    labels = summary.display.tolist()
    specs = (
        ("median_ess", "median ESS / 4096", "higher is better", True),
        (
            "selected_5d_energy",
            "selected population: 5D energy",
            "lower is better",
            False,
        ),
        ("physical_5d_mira_is", "posterior MIRA: physical 5D", "ideal = 2/3", True),
        (
            "median_sfh_wasserstein_over_iqr",
            "selected population: median SFH W1 / truth IQR",
            "lower is better",
            False,
        ),
    )
    colors = [
        COLORS["baseline"],
        COLORS["refresh"],
        "#D19A37",
        "#087F8C",
        COLORS["negative"],
    ]
    for axis, (column, title, subtitle, _) in zip(axes.ravel(), specs, strict=True):
        values = summary[column].to_numpy()
        axis.scatter(values, y, s=85, c=colors, zorder=3, edgecolor="white")
        for value, yi in zip(values, y, strict=True):
            axis.text(
                value, yi - 0.18, f"{value:.3g}", ha="center", va="top", fontsize=8
            )
        axis.set_yticks(y, labels if axis in axes[:, 0] else [])
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontweight="semibold", y=1.085)
        axis.text(
            0, 1.01, subtitle, transform=axis.transAxes, color="#68707B", fontsize=9
        )
        if column == "physical_5d_mira_is":
            axis.axvline(2 / 3, color=COLORS["truth"], linestyle="--", linewidth=1.4)
    fig.suptitle(
        "Run comparison: support, calibration and population closure", fontsize=16
    )
    fig.subplots_adjust(top=0.88, hspace=0.34, wspace=0.28)
    _save(fig, path)


def plot_genealogy(path: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    row = summary.set_index("variant")
    fig, axis = plt.subplots(figsize=(14, 7.2))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 7)
    axis.axis("off")
    positions = {
        "B_source": (2.2, 4.15),
        "Q_source_refresh": (6.8, 5.25),
        "B_latest": (6.8, 3.0),
        "Q_latest_refresh": (11.4, 3.0),
        "B_scratch_prior": (6.8, 0.8),
    }
    node_colors = {
        "B_source": "#E8ECF1",
        "Q_source_refresh": "#DDF3ED",
        "B_latest": "#FFF0D6",
        "Q_latest_refresh": "#D7F0F2",
        "B_scratch_prior": "#F7DEE1",
    }

    def node(name: str) -> None:
        x, y = positions[name]
        item = row.loc[name]
        patch = FancyBboxPatch(
            (x - 1.55, y - 0.72),
            3.1,
            1.44,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#AAB1BA",
            facecolor=node_colors[name],
        )
        axis.add_patch(patch)
        axis.text(x, y + 0.37, DISPLAY[name], ha="center", fontweight="semibold")
        axis.text(
            x,
            y - 0.02,
            f"ESS {item.median_ess:.0f} | MIRA5 IS {item.physical_5d_mira_is:.3f}",
            ha="center",
            fontsize=9,
        )
        axis.text(
            x,
            y - 0.37,
            f"energy {item.selected_5d_energy:.3f} | SFH W1/IQR {item.median_sfh_wasserstein_over_iqr:.1f}",
            ha="center",
            fontsize=8.5,
            color="#555D66",
        )

    def arrow(
        source: str,
        target: str,
        label: str,
        *,
        label_xy: tuple[float, float],
        bend: float = 0.0,
    ) -> None:
        start = positions[source]
        end = positions[target]
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.6,
            color="#6D747D",
            connectionstyle=f"arc3,rad={bend}",
            shrinkA=58,
            shrinkB=58,
        )
        axis.add_patch(patch)
        axis.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            fontsize=9,
            color="#4D545D",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )

    for name in positions:
        node(name)
    arrow(
        "B_source",
        "Q_source_refresh",
        "q refresh\nsource prior fixed",
        label_xy=(4.8, 5.10),
        bend=-0.08,
    )
    arrow(
        "B_source",
        "B_latest",
        "prior swap\nq fixed",
        label_xy=(4.75, 3.48),
        bend=0.05,
    )
    arrow(
        "B_latest",
        "Q_latest_refresh",
        "q refresh\nlearned prior fixed",
        label_xy=(9.1, 3.48),
    )
    arrow(
        "B_source",
        "B_scratch_prior",
        "scratch prior\nq fixed",
        label_xy=(4.55, 1.72),
        bend=0.10,
    )
    axis.text(
        0.65,
        6.55,
        "Experiment genealogy",
        fontsize=18,
        fontweight="semibold",
    )
    axis.text(
        0.65,
        6.18,
        "Every node uses the same 512 objects, likelihood and K=4096 ordinary-IS contract.",
        color="#626A74",
    )
    _save(fig, path)


def plot_paired_ess(path: Path, root: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    comparisons = (
        ("B_source", "Q_source_refresh", "source prior"),
        ("B_latest", "Q_latest_refresh", "learned prior"),
    )
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for axis, (before, after, label) in zip(axes, comparisons, strict=True):
        left = pd.read_csv(root / "arms" / before / "metrics.csv")
        right = pd.read_csv(root / "arms" / after / "metrics.csv")
        paired = left.merge(
            right, on=["row_index", "replica"], suffixes=("_before", "_after")
        )
        ratio = paired.ess_after.to_numpy() / paired.ess_before.to_numpy()
        x = np.sort(np.log2(np.maximum(ratio, 1e-12)))
        y = (np.arange(len(x)) + 0.5) / len(x)
        axis.plot(x, y, color=COLORS["refresh"], linewidth=2.3)
        axis.axvline(0, color="#737A83", linestyle="--")
        axis.axhline(0.5, color="#D4D8DE", linewidth=1)
        axis.set_xlabel(r"paired ESS gain $\log_2(ESS_{refresh}/ESS_{B4})$")
        axis.set_ylabel("fraction of object-replicas")
        axis.set_title(label, loc="left", fontweight="semibold")
        median = float(np.median(ratio))
        improved = float(np.mean(ratio > 1))
        axis.text(
            0.04,
            0.93,
            f"median x{median:.2f}\nimproved: {100 * improved:.1f}%",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#D8DCE2", "pad": 5},
        )
        for _, item in paired.iterrows():
            records.append(
                {
                    "comparison": label,
                    "row_index": int(item.row_index),
                    "replica": int(item.replica),
                    "ess_before": float(item.ess_before),
                    "ess_after": float(item.ess_after),
                    "ess_ratio": float(item.ess_after / item.ess_before),
                }
            )
    fig.suptitle("What each q refresh changes object by object", fontsize=15)
    fig.subplots_adjust(top=0.84, wspace=0.24)
    _save(fig, path)
    return pd.DataFrame(records)


def plot_mira(path: Path, root: Path) -> None:
    import matplotlib.pyplot as plt

    mira = pd.read_csv(root / "report/posterior_mira_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    groups = (("physical_5d", "physical 5D"), ("sfh_contrasts_10d", "SFH 10D"))
    colors = ["#747B84", "#159A80", "#D19A37", "#087F8C", "#B34E59"]
    for axis, (group, title) in zip(axes, groups, strict=True):
        for index, (variant, color) in enumerate(zip(VARIANTS, colors, strict=True)):
            raw = _mira_value(mira, variant, "raw", group)
            corrected = _mira_value(mira, variant, "is", group)
            axis.plot([raw, corrected], [index, index], color=color, linewidth=2)
            axis.scatter(raw, index, color="white", edgecolor=color, s=70, zorder=3)
            axis.scatter(corrected, index, color=color, s=70, zorder=3)
        axis.axvline(
            2 / 3,
            color=COLORS["truth"],
            linestyle="--",
            linewidth=1.5,
            label="ideal 2/3",
        )
        axis.set_title(title, fontweight="semibold")
        axis.set_xlabel("MIRA score")
        axis.set_yticks(np.arange(len(VARIANTS)), [DISPLAY[item] for item in VARIANTS])
        axis.invert_yaxis()
        axis.legend(
            handles=[
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    markerfacecolor="white",
                    markeredgecolor="#555",
                    linestyle="",
                    label="raw q",
                ),
                plt.Line2D(
                    [], [], marker="o", color="#555", linestyle="", label="ordinary IS"
                ),
                plt.Line2D(
                    [], [], color=COLORS["truth"], linestyle="--", label="ideal 2/3"
                ),
            ],
            loc="lower right",
        )
    fig.suptitle(
        "Calibration before and after exact ordinary importance weighting", fontsize=15
    )
    fig.subplots_adjust(top=0.84, wspace=0.14)
    _save(fig, path)


def plot_prior_mira(path: Path, root: Path) -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_csv(root / "report/prior_mira_scores.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), sharey=True)
    colors = {
        "source": COLORS["baseline"],
        "latest": COLORS["learned"],
        "scratch": COLORS["negative"],
    }
    groups = (("physical_5d", "physical 5D"), ("sfh_contrasts_10d", "SFH 10D"))
    y = np.arange(2)
    for axis, (population, title) in zip(
        axes, (("selected", "selected prior"), ("parent", "parent prior")), strict=True
    ):
        for offset, label in zip(
            (-0.16, 0.0, 0.16), ("source", "latest", "scratch"), strict=True
        ):
            scores, low, high = [], [], []
            for group, _ in groups:
                model = f"prior_{label}_{population}"
                row = frame.loc[
                    frame.population.eq(population)
                    & frame.model.eq(model)
                    & frame.group.eq(group)
                ].iloc[0]
                scores.append(float(row.score))
                low.append(float(row.score - row.bootstrap_q025))
                high.append(float(row.bootstrap_q975 - row.score))
            axis.errorbar(
                scores,
                y + offset,
                xerr=np.asarray([low, high]),
                fmt="o",
                markersize=7,
                capsize=3,
                linewidth=1.5,
                color=colors[label],
                label=f"{label} prior",
            )
        axis.axvline(2 / 3, color=COLORS["truth"], linestyle="--", linewidth=1.5)
        axis.set_yticks(y, [item[1] for item in groups])
        axis.invert_yaxis()
        axis.set_xlabel("MIRA score with 95% bootstrap interval")
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.legend(loc="center")
    fig.suptitle(
        "Prior calibration: physical 5D passes while joint SFH fails", fontsize=15
    )
    fig.subplots_adjust(top=0.82, wspace=0.16)
    _save(fig, path)


def _bounds_from_config(root: Path) -> dict[str, tuple[float, float]]:
    import yaml

    manifest = json.loads((root / "MANIFEST.json").read_text())
    candidates = [
        Path(manifest.get("config", "")),
        root.parent / "avi_encoder_experiments_v6/source_config.yaml",
        Path("outputs/avi_encoder_experiments_v6/source_config.yaml"),
    ]
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError("source_config.yaml required for boundary diagnostics")
    config = yaml.safe_load(config_path.read_text())
    free = config["fit"]["free_parameters"]
    return {name: tuple(map(float, free[name]["bounds"])) for name in PARAMETERS}


def build_sfh_diagnostics(root: Path, out: Path) -> pd.DataFrame:
    truth = pd.read_parquet(
        root / "selected_population_truth.parquet", columns=list(SFH)
    )
    bounds = _bounds_from_config(root)
    datasets: dict[str, np.ndarray] = {
        "true_selected": truth.to_numpy(np.float64),
    }
    object_iqr: dict[str, dict[str, float]] = {}
    for variant, kind in (
        ("B_source", "raw"),
        ("B_source", "is"),
        ("Q_latest_refresh", "raw"),
        ("Q_latest_refresh", "is"),
    ):
        frame = pd.concat(
            [
                pd.read_parquet(
                    root / "arms" / variant / f"{kind}_{replica}.parquet",
                    columns=["row_index", *SFH],
                )
                for replica in (0, 1)
            ],
            ignore_index=True,
        )
        key = f"{variant}_{kind}"
        datasets[key] = frame.loc[:, list(SFH)].to_numpy(np.float64)
        object_iqr[key] = {}
        for name in SFH:
            grouped = frame.groupby("row_index")[name]
            widths = grouped.quantile(0.75) - grouped.quantile(0.25)
            object_iqr[key][name] = float(widths.median())
    for prior_label, variant in PRIOR_VARIANTS.items():
        with np.load(root / "arms" / variant / "prior_population.npz") as archive:
            datasets[f"prior_{prior_label}_selected"] = np.asarray(
                archive["selected_theta"][:, 5:]
            )

    truth_values = datasets["true_selected"]
    rows = []
    for dataset, values in datasets.items():
        for index, name in enumerate(SFH):
            sample = values[:, index]
            target = truth_values[:, index]
            target_iqr = float(np.subtract(*np.quantile(target, [0.75, 0.25])))
            low, high = bounds[name]
            tolerance = 0.01 * (high - low)
            rows.append(
                {
                    "dataset": dataset,
                    "parameter": name,
                    "q01": float(np.quantile(sample, 0.01)),
                    "q05": float(np.quantile(sample, 0.05)),
                    "median": float(np.median(sample)),
                    "q95": float(np.quantile(sample, 0.95)),
                    "q99": float(np.quantile(sample, 0.99)),
                    "iqr": float(np.subtract(*np.quantile(sample, [0.75, 0.25]))),
                    "truth_iqr": target_iqr,
                    "iqr_over_truth_iqr": float(
                        np.subtract(*np.quantile(sample, [0.75, 0.25]))
                        / max(target_iqr, 1e-12)
                    ),
                    "median_object_iqr_over_truth_iqr": (
                        float(object_iqr[dataset][name] / max(target_iqr, 1e-12))
                        if dataset in object_iqr
                        else np.nan
                    ),
                    "wasserstein_over_truth_iqr": float(
                        wasserstein_distance(sample, target) / max(target_iqr, 1e-12)
                    ),
                    "lower_boundary_fraction": float(
                        np.mean(sample <= low + tolerance)
                    ),
                    "upper_boundary_fraction": float(
                        np.mean(sample >= high - tolerance)
                    ),
                    "fit_lower": low,
                    "fit_upper": high,
                    "fit_width_over_truth_iqr": float(
                        (high - low) / max(target_iqr, 1e-12)
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(out / "sfh_diagnostics.csv", index=False)
    return result


def plot_sfh_diagnostics(path: Path, diagnostics: pd.DataFrame, root: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"SFH {index}" for index in range(1, 11)]
    x = np.arange(10)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    series = (
        ("B_source_raw", "B4 raw", COLORS["baseline"], "--"),
        ("Q_latest_refresh_raw", "refresh raw", COLORS["refresh"], "-"),
        ("Q_latest_refresh_is", "refresh + IS", COLORS["learned"], "-"),
    )
    for dataset, label, color, linestyle in series:
        rows = (
            diagnostics.loc[diagnostics.dataset.eq(dataset)]
            .set_index("parameter")
            .loc[list(SFH)]
        )
        axes[0].plot(
            x,
            rows.median_object_iqr_over_truth_iqr,
            marker="o",
            label=label,
            color=color,
            linestyle=linestyle,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median per-object IQR / truth-population IQR")
    axes[0].set_title(
        "Individual posteriors remain broader than the SFH population",
        loc="left",
        fontweight="semibold",
    )
    axes[0].legend(ncol=3)

    for label, color in (
        ("source", COLORS["baseline"]),
        ("latest", COLORS["learned"]),
        ("scratch", COLORS["negative"]),
    ):
        rows = (
            diagnostics.loc[diagnostics.dataset.eq(f"prior_{label}_selected")]
            .set_index("parameter")
            .loc[list(SFH)]
        )
        boundary = rows.lower_boundary_fraction + rows.upper_boundary_fraction
        axes[1].plot(x, boundary, marker="o", label=f"{label} prior", color=color)
    axes[1].set_ylabel("mass in outer 1% of fit bounds")
    axes[1].set_title(
        "Learned priors retain boundary-seeking SFH mass",
        loc="left",
        fontweight="semibold",
    )
    axes[1].legend(ncol=3)

    mira = pd.read_csv(root / "report/posterior_mira_scores.csv")
    for model, label, color in (
        ("B_source_raw", "B4 raw", COLORS["baseline"]),
        ("Q_latest_refresh_raw", "refresh raw", COLORS["refresh"]),
        ("Q_latest_refresh_is", "refresh + IS", COLORS["learned"]),
    ):
        values = []
        for name in SFH:
            values.append(
                float(
                    mira.loc[
                        mira.model.eq(model) & mira.group.eq(f"marginal_{name}"),
                        "score",
                    ].mean()
                )
            )
        axes[2].plot(x, values, marker="o", label=label, color=color)
    axes[2].axhline(2 / 3, color=COLORS["truth"], linestyle="--", label="ideal 2/3")
    axes[2].set_ylabel("marginal MIRA")
    axes[2].set_title(
        "The failure is structured: late SFH contrasts are strongly miscalibrated",
        loc="left",
        fontweight="semibold",
    )
    axes[2].legend(ncol=4)
    axes[2].set_xticks(x, labels)
    fig.suptitle("Why the SFH posteriors look wrong", fontsize=16)
    fig.subplots_adjust(top=0.91, hspace=0.30)
    _save(fig, path)


def _representative_rows(root: Path) -> pd.DataFrame:
    truth = pd.read_parquet(root / "inference_truth.parquet")
    b = pd.read_csv(root / "arms/B_latest/metrics.csv").query("replica == 0")
    q = pd.read_csv(root / "arms/Q_latest_refresh/metrics.csv").query("replica == 0")
    metrics = b.merge(q, on="row_index", suffixes=("_b", "_q"))
    physical = truth.loc[:, list(PHYSICAL)].to_numpy(np.float64)
    center = np.median(physical, axis=0)
    scale = np.maximum(np.subtract(*np.quantile(physical, [0.84, 0.16], axis=0)), 1e-8)
    distance = np.sqrt(np.mean(((physical - center) / scale) ** 2, axis=1))
    truth = truth.copy()
    truth["central_distance"] = distance
    metrics = metrics.merge(
        truth[["row_index", "z_obs", "central_distance"]], on="row_index"
    )
    candidates = [
        ("typical", int(metrics.loc[metrics.central_distance.idxmin(), "row_index"])),
        ("high redshift", int(metrics.loc[metrics.z_obs.idxmax(), "row_index"])),
        ("lowest refreshed ESS", int(metrics.loc[metrics.ess_q.idxmin(), "row_index"])),
        (
            "largest ESS gain",
            int(metrics.loc[(metrics.ess_q / metrics.ess_b).idxmax(), "row_index"]),
        ),
    ]
    output = []
    used = set()
    for reason, row_index in candidates:
        if row_index in used:
            continue
        used.add(row_index)
        item = metrics.loc[metrics.row_index.eq(row_index)].iloc[0]
        output.append(
            {
                "reason": reason,
                "row_index": row_index,
                "ess_b_latest": float(item.ess_b),
                "ess_q_latest": float(item.ess_q),
                "ess_gain": float(item.ess_q / item.ess_b),
            }
        )
    return pd.DataFrame(output)


def plot_individual_corner(
    path: Path, before: np.ndarray, after: np.ndarray, truth: np.ndarray, title: str
) -> None:
    import corner
    import matplotlib.pyplot as plt

    ranges = []
    for index in range(5):
        low, high = _limits(
            before[:, index], after[:, index], np.asarray([truth[index]])
        )
        ranges.append((low, high))
    figure = corner.corner(
        after[:, :5],
        labels=[SHORT[name] for name in PHYSICAL],
        color=COLORS["refresh"],
        range=ranges,
        bins=28,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.80, 0.95),
        hist_kwargs={"linewidth": 2.0},
        contour_kwargs={"linewidths": 1.8},
        truths=truth,
        truth_color=COLORS["truth"],
        truth_kwargs={"linewidth": 1.7},
        label_kwargs={"fontsize": 11},
        max_n_ticks=4,
    )
    corner.corner(
        before[:, :5],
        fig=figure,
        color=COLORS["baseline"],
        range=ranges,
        bins=28,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.50, 0.80, 0.95),
        hist_kwargs={"linewidth": 1.5, "linestyle": "--"},
        contour_kwargs={"linewidths": 1.1, "linestyles": "--"},
        max_n_ticks=4,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            color=COLORS["baseline"],
            linestyle="--",
            linewidth=2,
            label="B4 + learned prior, IS",
        ),
        plt.Line2D(
            [],
            [],
            color=COLORS["refresh"],
            linewidth=2.5,
            label="refreshed q + learned prior, IS",
        ),
        plt.Line2D([], [], color=COLORS["truth"], linewidth=2, label="truth"),
    ]
    figure.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.96, 0.96))
    figure.suptitle(title, fontsize=16, fontweight="semibold", y=1.01)
    _save(figure, path)


def plot_individual_marginals(
    path: Path, before: pd.DataFrame, after: pd.DataFrame, truth: pd.Series, title: str
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(17, 9))
    for index, name in enumerate(PARAMETERS):
        axis = axes.ravel()[index]
        b = before[name].to_numpy(np.float64)
        q = after[name].to_numpy(np.float64)
        low, high = _limits(b, q, np.asarray([truth[name]]))
        edges = np.linspace(low, high, 55)
        _density_panel(
            axis, q, edges, color=COLORS["refresh"], label="refreshed + IS", fill=True
        )
        _density_panel(
            axis, b, edges, color=COLORS["baseline"], label="B4 + IS", linestyle="--"
        )
        axis.axvline(truth[name], color=COLORS["truth"], linewidth=1.7)
        axis.set_title(SHORT[name])
        axis.set_xlim(low, high)
        axis.set_yticks([])
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=COLORS["truth"], label="truth"))
    labels.append("truth")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(title, fontsize=16, fontweight="semibold", y=1.03)
    fig.subplots_adjust(top=0.89, hspace=0.34, wspace=0.22)
    _save(fig, path)


def plot_individuals(root: Path, out: Path) -> pd.DataFrame:
    chosen = _representative_rows(root)
    truth = pd.read_parquet(root / "inference_truth.parquet").set_index("row_index")
    b = pd.read_parquet(root / "arms/B_latest/is_0.parquet")
    q = pd.read_parquet(root / "arms/Q_latest_refresh/is_0.parquet")
    target = out / "individual"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for order, item in chosen.iterrows():
        row_index = int(item.row_index)
        before = b.loc[b.row_index.eq(row_index)].copy()
        after = q.loc[q.row_index.eq(row_index)].copy()
        truth_row = truth.loc[row_index]
        stem = f"{order + 1:02d}_row_{row_index}"
        subtitle = (
            f"{item.reason}; ESS {item.ess_b_latest:.1f} -> {item.ess_q_latest:.1f} "
            f"(x{item.ess_gain:.2f})"
        )
        plot_individual_corner(
            target / f"{stem}_corner_physical_5d.png",
            before.loc[:, list(PHYSICAL)].to_numpy(np.float64),
            after.loc[:, list(PHYSICAL)].to_numpy(np.float64),
            truth_row.loc[list(PHYSICAL)].to_numpy(np.float64),
            subtitle,
        )
        plot_individual_marginals(
            target / f"{stem}_marginals_15d.png",
            before,
            after,
            truth_row,
            subtitle,
        )
        records.append(
            {
                **item.to_dict(),
                "corner_physical_5d": str(
                    (target / f"{stem}_corner_physical_5d.png").resolve()
                ),
                "marginals_15d": str((target / f"{stem}_marginals_15d.png").resolve()),
            }
        )
    result = pd.DataFrame(records)
    result.to_csv(out / "individual_objects.csv", index=False)
    return result


def selection_truth_audit(parent: pd.DataFrame, selected: pd.DataFrame) -> dict:
    common = list(PARAMETERS)
    left = parent.loc[:, common].to_numpy(np.float64)
    right = selected.loc[:, common].to_numpy(np.float64)
    same_rows = bool(
        len(parent) == len(selected)
        and np.array_equal(parent.row_index.to_numpy(), selected.row_index.to_numpy())
    )
    return {
        "same_row_identities": same_rows,
        "same_physical_values_exactly": bool(np.array_equal(left[:, :5], right[:, :5])),
        "maximum_absolute_15d_difference": float(np.max(np.abs(left - right))),
        "maximum_physical_5d_difference": float(
            np.max(np.abs(left[:, :5] - right[:, :5]))
        ),
        "distinct_selected_truth_cohort": bool(not same_rows),
        "interpretation": (
            "The saved parent and selected truth tables do not form distinct latent "
            "cohorts; selection-population closure cannot be claimed from their overlay."
        ),
    }


def sfh_ess_association(root: Path, out: Path) -> pd.DataFrame:
    population = pd.read_parquet(root / "selected_population_truth.parquet").set_index(
        "row_index"
    )
    truth = pd.read_parquet(root / "inference_truth.parquet").set_index("row_index")
    metrics = (
        pd.read_csv(root / "arms/Q_latest_refresh/metrics.csv")
        .groupby("row_index")
        .ess.mean()
    )
    center = population.loc[:, list(SFH)].median()
    scale = population.loc[:, list(SFH)].quantile(0.75) - population.loc[
        :, list(SFH)
    ].quantile(0.25)
    standardized = (truth.loc[metrics.index, list(SFH)] - center) / scale
    rows = []
    overall = np.sqrt(np.mean(np.square(standardized.to_numpy()), axis=1))
    rho, pvalue = spearmanr(np.log(metrics.to_numpy()), overall)
    rows.append(
        {
            "feature": "joint_sfh_robust_distance",
            "spearman_rho": float(rho),
            "pvalue": float(pvalue),
        }
    )
    for name in SFH:
        rho, pvalue = spearmanr(
            np.log(metrics.to_numpy()), np.abs(standardized[name].to_numpy())
        )
        rows.append(
            {"feature": name, "spearman_rho": float(rho), "pvalue": float(pvalue)}
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "sfh_ess_association.csv", index=False)
    return result


def _write_readme(
    root: Path,
    out: Path,
    summary: pd.DataFrame,
    sfh: pd.DataFrame,
    association: pd.DataFrame,
    audit: dict,
) -> None:
    b_latest = summary.set_index("variant").loc["B_latest"]
    q_latest = summary.set_index("variant").loc["Q_latest_refresh"]
    truth_rows = sfh.loc[sfh.dataset.eq("true_selected")]
    latest_rows = sfh.loc[sfh.dataset.eq("prior_latest_selected")]
    max_boundary = latest_rows.loc[
        (
            latest_rows.lower_boundary_fraction + latest_rows.upper_boundary_fraction
        ).idxmax()
    ]
    joint_association = association.loc[
        association.feature.eq("joint_sfh_robust_distance")
    ].iloc[0]
    text = f"""# AVI overnight analysis

## Main result

`Q_latest_refresh` is the strongest proposal in this suite. Its median ordinary-IS ESS is
{q_latest.median_ess:.1f}/4096 versus {b_latest.median_ess:.1f}/4096 for `B_latest`, while
the median raw predictive RMS changes from {b_latest.raw_predictive_rms:.3f} to
{q_latest.raw_predictive_rms:.3f}. The exact-IS predictive RMS is stable
({b_latest.is_predictive_rms:.3f} versus {q_latest.is_predictive_rms:.3f}), as expected
for two proposals targeting the same learned-prior posterior.

Physical 5D calibration remains strong before IS (`Q_latest_refresh` MIRA
{q_latest.physical_5d_mira_raw:.3f}, ideal 0.667), and is the best exact-IS result here
({q_latest.physical_5d_mira_is:.3f}). The remaining support is not production-grade:
the median ESS fraction is {q_latest.median_ess_fraction:.3%} and
{q_latest.fraction_ess_below_5:.1%} of object-replicas have ESS below 5.

## Population closure

The learned prior improves the selected physical 5D aggregate: standardized energy
distance changes from {summary.set_index("variant").loc["B_source"].selected_5d_energy:.3f}
for the source lineage to {b_latest.selected_5d_energy:.3f} for `B_latest`.
Refreshing q on that prior changes correlation recovery more than marginal distance;
inspect `run_scorecard.png` and `run_genealogy.png`.

The requested parent/selected truth comparison is not a valid selection-closure test in
this receipt. `same_row_identities={audit["same_row_identities"]}` and the physical 5D
values are exactly equal. The small 15D differences are at most
{audit["maximum_absolute_15d_difference"]:.2e}. The plots intentionally display both
curves, but their overlap is a data-contract warning, not evidence of perfect selection
correction.

## SFH diagnosis

The SFH failure precedes importance weighting. Raw 10D MIRA is about
{q_latest.sfh_10d_mira_raw:.3f} and exact-IS MIRA is {q_latest.sfh_10d_mira_is:.3f}, far
from 0.667. The refreshed exact posteriors have a median per-object IQR inflation
of {sfh.loc[sfh.dataset.eq("Q_latest_refresh_is"), "median_object_iqr_over_truth_iqr"].median():.1f}x
relative to the truth-population IQR. Fit boxes are themselves a median
{truth_rows.fit_width_over_truth_iqr.median():.1f}x wider than the truth IQR.

The latest selected prior has its largest outer-bound mass for
`{max_boundary.parameter}`:
{(max_boundary.lower_boundary_fraction + max_boundary.upper_boundary_fraction):.1%}.
This, together with the alternating 0/15-like modes in SFH 02/04 and the narrow true
late-time contrasts, points to a prior/coordinate support mismatch plus weak
photometric identifiability. Ordinary IS cannot repair missing proposal support; it only
concentrates the already broad 15D draws and reduces Monte Carlo diversity.
The refreshed ESS is not associated with being in a rare SFH-truth tail
(Spearman rho {joint_association.spearman_rho:.3f}, p={joint_association.pvalue:.3f});
the mismatch is global rather than confined to a few unusual galaxies.

## What next

1. Regenerate a genuinely selected truth cohort from the parent catalogue using the
   exact noisy `lsst_r < 29` event, and fail closed when parent and selected row IDs are
   identical. Re-run only the population closure report first.
2. Replace fit-bound-centered SFH normalization with empirical, truth-free sleep-bank
   geometry (robust center/scale or a monotone quantile warp), version its hash, and
   retrain both prior and q from fresh checkpoints.
3. Factor the 15D density as a physical 5D expert flow followed by a shared conditional
   SFH flow. Keep the exact joint density, but train the conditional SFH block with
   stronger sleep supervision and SFH-block tempering/weight clipping diagnostics.
4. Continue the learned-prior lineage, not scratch. Alternate short prior M-steps and q
   refreshes, accepting a cycle only when physical 5D MIRA is retained, median ESS rises,
   and SFH boundary mass/IQR inflation fall on an independent cohort.
5. Diagnose identifiability with decoder Jacobian singular values and posterior
   predictive interventions along each SFH contrast. Merge or regularize directions
   that are photometrically null instead of buying more mixture experts.

## Figure map

- `population/`: one prior lineage per figure, exactly three population curves.
- `individual/`: four representative exact-IS posteriors with truth overlays.
- `run_genealogy.png`: controlled lineage and headline metrics.
- `paired_ess_refresh.png`: paired per-object effect of each q refresh.
- `mira_raw_vs_is.png`: raw proposal versus ordinary-IS calibration.
- `prior_mira.png`: parent and selected prior calibration by parameter block.
- `sfh_failure_diagnostics.png`: width, boundary mass and marginal calibration.
- `sfh_ess_association.csv`: support versus SFH-truth extremeness.
"""
    (out / "README.md").write_text(text)


def analyze(root: Path, output: Path | None) -> Path:
    root = root.resolve()
    output = (output or root / "analysis_v2").resolve()
    final = json.loads((root / "report/FINAL.json").read_text())
    if final.get("status") != "OVERNIGHT_REPORT_COMPLETE":
        raise ValueError("completed overnight report required")
    output.mkdir(parents=True, exist_ok=True)
    _style()

    parent = pd.read_parquet(root / "parent_population_truth.parquet")
    selected = pd.read_parquet(root / "selected_population_truth.parquet")
    audit = selection_truth_audit(parent, selected)
    (output / "selection_truth_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    population_dir = output / "population"
    for prior_label, variant in PRIOR_VARIANTS.items():
        with np.load(root / "arms" / variant / "prior_population.npz") as archive:
            learned = np.asarray(archive["selected_theta"])
        plot_population_triplet(
            population_dir / f"{prior_label}_selected_prior_physical_5d.png",
            parent,
            selected,
            learned,
            PHYSICAL,
            f"{prior_label.capitalize()} lineage: selected physical population",
        )
        plot_population_triplet(
            population_dir / f"{prior_label}_selected_prior_sfh_10d.png",
            parent,
            selected,
            learned,
            SFH,
            f"{prior_label.capitalize()} lineage: selected SFH population",
        )

    q_latest_is = pd.concat(
        [
            pd.read_parquet(root / "arms/Q_latest_refresh" / f"is_{replica}.parquet")
            for replica in (0, 1)
        ],
        ignore_index=True,
    )
    plot_posterior_population(
        population_dir / "q_latest_selected_posterior_physical_5d.png",
        parent,
        selected,
        q_latest_is,
        PHYSICAL,
        "Learned-prior lineage: selected posterior aggregate",
    )
    plot_posterior_population(
        population_dir / "q_latest_selected_posterior_sfh_10d.png",
        parent,
        selected,
        q_latest_is,
        SFH,
        "Learned-prior lineage: selected posterior SFH aggregate",
    )
    del q_latest_is

    summary = build_run_summary(root)
    summary.to_csv(output / "run_summary.csv", index=False)
    plot_scorecard(output / "run_scorecard.png", summary)
    plot_genealogy(output / "run_genealogy.png", summary)
    paired = plot_paired_ess(output / "paired_ess_refresh.png", root)
    paired.to_csv(output / "paired_ess_refresh.csv", index=False)
    plot_mira(output / "mira_raw_vs_is.png", root)
    plot_prior_mira(output / "prior_mira.png", root)

    sfh = build_sfh_diagnostics(root, output)
    plot_sfh_diagnostics(output / "sfh_failure_diagnostics.png", sfh, root)
    association = sfh_ess_association(root, output)
    plot_individuals(root, output)
    _write_readme(root, output, summary, sfh, association, audit)

    artifacts = sorted(
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    )
    receipt = {
        "status": "AVI_OVERNIGHT_ANALYSIS_COMPLETE",
        "source_root": str(root),
        "joint_draws_preserved": True,
        "point_estimates_used_as_distributions": False,
        "selection_truth_cohort_distinct": audit["distinct_selected_truth_cohort"],
        "artifacts": artifacts,
    }
    (output / "FINAL.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Analysis complete: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.root, args.output)


if __name__ == "__main__":
    main()
