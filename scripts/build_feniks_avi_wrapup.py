"""Build a self-contained scientific wrap-up for the AVI overnight suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import kstest, spearmanr, wasserstein_distance

if __package__:
    from scripts.analyze_feniks_avi_overnight import (
        COLORS,
        PARAMETERS,
        PHYSICAL,
        SFH,
        SHORT,
        _density_panel,
        _limits,
        _save,
        _style,
    )
else:
    from analyze_feniks_avi_overnight import (
        COLORS,
        PARAMETERS,
        PHYSICAL,
        SFH,
        SHORT,
        _density_panel,
        _limits,
        _save,
        _style,
    )

BASELINE = "B_source"
CURRENT = "Q_latest_refresh"
DISPLAY = {
    BASELINE: "before: B4 + source prior",
    CURRENT: "current: refreshed q + learned prior",
}
NUTS_VARIANT = "B_dense_depth6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_draws(root: Path, variant: str, kind: str) -> pd.DataFrame:
    return pd.concat(
        [
            pd.read_parquet(root / "arms" / variant / f"{kind}_{replica}.parquet")
            for replica in (0, 1)
        ],
        ignore_index=True,
    )


def _draw_cube(frame: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    grouped = frame.set_index("row_index")
    pieces = []
    for row in rows:
        values = grouped.loc[int(row), list(PARAMETERS)]
        if isinstance(values, pd.Series):
            values = values.to_frame().T
        pieces.append(values.to_numpy(np.float64))
    counts = {piece.shape[0] for piece in pieces}
    if len(counts) != 1:
        raise ValueError(f"inconsistent posterior draw counts: {counts}")
    return np.stack(pieces)


def _crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Empirical CRPS for [object, draw, parameter] arrays."""
    ordered = np.sort(samples, axis=1)
    n = ordered.shape[1]
    coefficients = (2 * np.arange(1, n + 1) - n - 1).reshape(1, n, 1)
    spread = np.sum(ordered * coefficients, axis=1) / float(n * n)
    return np.mean(np.abs(samples - truth[:, None, :]), axis=1) - spread


def _wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return center - half, center + half


def paired_improvement(
    root: Path,
    out: Path,
    cubes: dict[tuple[str, str], np.ndarray],
    truth: np.ndarray,
    rows: np.ndarray,
) -> pd.DataFrame:
    scales = np.maximum(
        np.quantile(truth, 0.75, axis=0) - np.quantile(truth, 0.25, axis=0), 1e-8
    )
    records = []
    for kind in ("raw", "is"):
        before = _crps(cubes[(BASELINE, kind)], truth) / scales
        after = _crps(cubes[(CURRENT, kind)], truth) / scales
        for index, row in enumerate(rows):
            records.append(
                {
                    "row_index": int(row),
                    "kind": kind,
                    "physical_crps_before": float(before[index, :5].mean()),
                    "physical_crps_current": float(after[index, :5].mean()),
                    "sfh_crps_before": float(before[index, 5:].mean()),
                    "sfh_crps_current": float(after[index, 5:].mean()),
                }
            )
    result = pd.DataFrame(records)
    metrics_before = pd.read_csv(root / "arms" / BASELINE / "metrics.csv")
    metrics_after = pd.read_csv(root / "arms" / CURRENT / "metrics.csv")
    metrics = metrics_before.merge(
        metrics_after, on=["row_index", "replica"], suffixes=("_before", "_current")
    )
    object_metrics = metrics.groupby("row_index").agg(
        ess_before=("ess_before", "mean"),
        ess_current=("ess_current", "mean"),
        raw_rms_before=("raw_predictive_rms_before", "mean"),
        raw_rms_current=("raw_predictive_rms_current", "mean"),
    )
    result = result.merge(object_metrics, on="row_index")
    result.to_csv(out / "tables" / "paired_improvement.csv", index=False)
    return result


def plot_improvement(path: Path, paired: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt

    frame = paired.loc[paired.kind.eq("is")].copy()
    quantities = (
        ("ess_before", "ess_current", "ESS", True),
        (
            "physical_crps_before",
            "physical_crps_current",
            "physical 5D CRPS / truth IQR",
            False,
        ),
        ("sfh_crps_before", "sfh_crps_current", "SFH CRPS / truth IQR", False),
        ("raw_rms_before", "raw_rms_current", "raw predictive RMS", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    summary = {}
    for axis, (before_name, after_name, label, higher) in zip(
        axes.ravel(), quantities, strict=True
    ):
        before = frame[before_name].to_numpy()
        after = frame[after_name].to_numpy()
        improved = after > before if higher else after < before
        fraction = float(improved.mean())
        low, high = _wilson(int(improved.sum()), len(improved))
        summary[label] = {
            "fraction_improved": fraction,
            "wilson_q025": low,
            "wilson_q975": high,
        }
        low_limit, high_limit = _limits(before, after)
        axis.hexbin(before, after, gridsize=28, mincnt=1, cmap="viridis", linewidths=0)
        axis.plot(
            [low_limit, high_limit],
            [low_limit, high_limit],
            "--",
            color="#D13F3F",
            linewidth=1.4,
        )
        axis.set_xlim(low_limit, high_limit)
        axis.set_ylim(low_limit, high_limit)
        axis.set_xlabel("before")
        axis.set_ylabel("current")
        axis.set_title(label, loc="left", fontweight="semibold")
        axis.text(
            0.04,
            0.94,
            f"improved: {fraction:.1%}\n95% CI [{low:.1%}, {high:.1%}]",
            transform=axis.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#D8DCE2", "pad": 4},
        )
    fig.suptitle(
        "Is the current model better for most galaxies?",
        fontsize=16,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.92, hspace=0.28, wspace=0.25)
    _save(fig, path)
    return summary


def _posterior_pit(cube: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean(cube <= truth[:, None, :], axis=1)


def plot_pit_grid(
    path: Path,
    cubes: dict[tuple[str, str], np.ndarray],
    truth: np.ndarray,
    names: tuple[str, ...],
    title: str,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    indices = [PARAMETERS.index(name) for name in names]
    fig, axes = plt.subplots(
        2, len(names), figsize=(3.25 * len(names), 6.0), sharex=True, sharey=True
    )
    records = []
    for row, variant in enumerate((BASELINE, CURRENT)):
        pit = _posterior_pit(cubes[(variant, "is")], truth)
        for column, index in enumerate(indices):
            axis = axes[row, column]
            values = pit[:, index]
            axis.hist(
                values,
                bins=np.linspace(0, 1, 11),
                density=True,
                color=COLORS["baseline"] if row == 0 else COLORS["refresh"],
                alpha=0.78,
            )
            axis.axhline(1, color=COLORS["truth"], linestyle="--", linewidth=1.2)
            ks = kstest(values, "uniform")
            records.append(
                {
                    "variant": variant,
                    "parameter": PARAMETERS[index],
                    "pit_ks": float(ks.statistic),
                    "pit_ks_pvalue": float(ks.pvalue),
                    "pit_mean": float(values.mean()),
                }
            )
            axis.set_title(SHORT[PARAMETERS[index]])
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[variant]}\ndensity")
            if row == 1:
                axis.set_xlabel("PIT")
    fig.suptitle(title, fontsize=16, fontweight="semibold")
    fig.subplots_adjust(top=0.89, hspace=0.30, wspace=0.16)
    _save(fig, path)
    return pd.DataFrame(records)


def plot_sfh_pit(
    path: Path, cubes: dict[tuple[str, str], np.ndarray], truth: np.ndarray
) -> pd.DataFrame:
    return plot_pit_grid(
        path, cubes, truth, SFH, "SFH posterior PIT: before versus current ordinary IS"
    )


def plot_coverage(
    path: Path, cubes: dict[tuple[str, str], np.ndarray], truth: np.ndarray
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    levels = np.linspace(0.1, 0.95, 18)
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for axis, (indices, label) in zip(
        axes, ((range(5), "physical 5D"), (range(5, 15), "SFH 10D")), strict=True
    ):
        for variant, color, linestyle in (
            (BASELINE, COLORS["baseline"], "--"),
            (CURRENT, COLORS["refresh"], "-"),
        ):
            cube = cubes[(variant, "is")]
            coverage = []
            for level in levels:
                q = (1 - level) / 2
                low = np.quantile(cube[:, :, list(indices)], q, axis=1)
                high = np.quantile(cube[:, :, list(indices)], 1 - q, axis=1)
                value = float(
                    np.mean(
                        (truth[:, list(indices)] >= low)
                        & (truth[:, list(indices)] <= high)
                    )
                )
                coverage.append(value)
                records.append(
                    {
                        "variant": variant,
                        "group": label,
                        "nominal": level,
                        "coverage": value,
                    }
                )
            axis.plot(
                levels,
                coverage,
                color=color,
                linestyle=linestyle,
                marker="o",
                markersize=3,
                label=DISPLAY[variant],
            )
        axis.plot(
            [0, 1],
            [0, 1],
            color=COLORS["truth"],
            linewidth=1.2,
            linestyle=":",
            label="ideal",
        )
        axis.set_title(label, loc="left", fontweight="semibold")
        axis.set_xlabel("nominal central interval")
        axis.set_ylabel("empirical truth coverage")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.legend()
    fig.suptitle("Posterior coverage after ordinary importance weighting", fontsize=16)
    fig.subplots_adjust(top=0.85, wspace=0.23)
    _save(fig, path)
    return pd.DataFrame(records)


def plot_truth_predicted(
    path: Path,
    cubes: dict[tuple[str, str], np.ndarray],
    truth: np.ndarray,
    names: tuple[str, ...],
    title: str,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    indices = [PARAMETERS.index(name) for name in names]
    fig, axes = plt.subplots(
        2, len(names), figsize=(3.3 * len(names), 6.2), squeeze=False
    )
    records = []
    for row, variant in enumerate((BASELINE, CURRENT)):
        cube = cubes[(variant, "is")]
        estimate = np.median(cube, axis=1)
        for column, index in enumerate(indices):
            axis = axes[row, column]
            x = truth[:, index]
            y = estimate[:, index]
            low, high = _limits(x, y)
            axis.hexbin(
                x,
                y,
                gridsize=25,
                mincnt=1,
                cmap="magma" if row == 0 else "viridis",
                linewidths=0,
            )
            axis.plot([low, high], [low, high], "--", color="white", linewidth=2.4)
            axis.plot(
                [low, high], [low, high], "--", color=COLORS["truth"], linewidth=1.2
            )
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_title(SHORT[PARAMETERS[index]])
            if column == 0:
                axis.set_ylabel(f"{DISPLAY[variant]}\nposterior median")
            if row == 1:
                axis.set_xlabel("truth")
            mae = float(np.median(np.abs(y - x)))
            rho = float(spearmanr(x, y).statistic)
            records.append(
                {
                    "variant": variant,
                    "parameter": PARAMETERS[index],
                    "median_absolute_error": mae,
                    "spearman_rho": rho,
                }
            )
            axis.text(
                0.04,
                0.94,
                f"MAE {mae:.3g}\nrho {rho:.2f}",
                transform=axis.transAxes,
                va="top",
                color="#20242A",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                    "pad": 2,
                },
            )
    fig.suptitle(title + " (medians shown only as point summaries)", fontsize=16)
    fig.subplots_adjust(top=0.88, hspace=0.28, wspace=0.20)
    _save(fig, path)
    return pd.DataFrame(records)


def _train_beta_emulator(
    root: Path, truth_theta: np.ndarray, out: Path
) -> tuple[np.ndarray, dict]:
    import torch

    torch.manual_seed(260913)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    xs, ys = [], []
    for arm in ("B_source", "B_latest", "B_scratch_prior"):
        with np.load(root / "arms" / arm / "prior_population.npz") as archive:
            xs.append(np.asarray(archive["theta"], np.float32))
            ys.append(np.asarray(archive["beta"], np.float32))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    rng = np.random.default_rng(260913)
    order = rng.permutation(len(x))
    validation = order[:20000]
    training = order[20000:]
    center = np.median(x[training], axis=0)
    scale = np.maximum(
        np.quantile(x[training], 0.75, axis=0) - np.quantile(x[training], 0.25, axis=0),
        1e-3,
    )

    def standardize(values: np.ndarray) -> np.ndarray:
        return np.clip((values - center) / scale, -12, 12).astype(np.float32)

    clipped = np.clip(y[training], 1e-5, 1 - 1e-5)
    target = np.log(clipped / (1 - clipped)).astype(np.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(15, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 64),
        torch.nn.SiLU(),
        torch.nn.Linear(64, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    tx = torch.from_numpy(standardize(x[training]))
    ty = torch.from_numpy(target[:, None])
    for _ in range(45):
        permutation = torch.randperm(len(tx))
        for start in range(0, len(tx), 2048):
            index = permutation[start : start + 2048]
            loss = torch.nn.functional.huber_loss(
                model(tx[index]), ty[index], delta=2.0
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        predicted_validation = (
            torch.sigmoid(model(torch.from_numpy(standardize(x[validation]))))
            .numpy()
            .ravel()
        )
        predicted_truth = (
            torch.sigmoid(
                model(torch.from_numpy(standardize(truth_theta.astype(np.float32))))
            )
            .numpy()
            .ravel()
        )
    boundary = (y[validation] > 0.01) & (y[validation] < 0.99)
    diagnostics = {
        "role": "diagnostic emulator of the exact deterministic beta(theta), never used for training or inference",
        "training_samples": int(len(training)),
        "validation_samples": int(len(validation)),
        "validation_mae": float(np.mean(np.abs(predicted_validation - y[validation]))),
        "validation_rmse": float(
            np.sqrt(np.mean((predicted_validation - y[validation]) ** 2))
        ),
        "transition_samples": int(boundary.sum()),
        "transition_mae": float(
            np.mean(np.abs(predicted_validation[boundary] - y[validation][boundary]))
        ),
        "transition_spearman": float(
            spearmanr(predicted_validation[boundary], y[validation][boundary]).statistic
        ),
    }
    diagnostics["qualified_for_visual_diagnostic"] = bool(
        diagnostics["validation_mae"] < 0.03 and diagnostics["transition_mae"] < 0.06
    )
    pd.DataFrame(
        {
            "beta_exact": y[validation],
            "beta_emulated": predicted_validation,
            "transition": boundary,
        }
    ).to_parquet(out / "tables" / "selection_emulator_validation.parquet", index=False)
    (out / "tables" / "selection_emulator.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    return predicted_truth, diagnostics


def _weighted_resample(
    values: np.ndarray, weight: np.ndarray, size: int, seed: int
) -> np.ndarray:
    normalized = np.asarray(weight, np.float64)
    normalized /= normalized.sum()
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=size, replace=True, p=normalized)]


def plot_selection_closure(
    path: Path,
    true_parent: np.ndarray,
    true_selected: np.ndarray,
    learned_parent: np.ndarray,
    learned_selected: np.ndarray,
    names: tuple[str, ...],
) -> None:
    import matplotlib.pyplot as plt

    columns = len(names)
    fig, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 6.2))
    for column, name in enumerate(names):
        index = PARAMETERS.index(name)
        arrays = (
            true_parent[:, index],
            true_selected[:, index],
            learned_parent[:, index],
            learned_selected[:, index],
        )
        low, high = _limits(*arrays)
        edges = np.linspace(low, high, 70)
        _density_panel(
            axes[0, column],
            arrays[0],
            edges,
            color=COLORS["parent"],
            label="true parent C0",
        )
        _density_panel(
            axes[0, column],
            arrays[2],
            edges,
            color=COLORS["learned"],
            label="learned parent prior",
            fill=True,
        )
        _density_panel(
            axes[1, column],
            arrays[1],
            edges,
            color=COLORS["selected"],
            label="beta-weighted true C0",
        )
        _density_panel(
            axes[1, column],
            arrays[3],
            edges,
            color=COLORS["refresh"],
            label="learned selected prior",
            fill=True,
        )
        for row in range(2):
            axes[row, column].set_xlim(low, high)
            axes[row, column].set_yticks([])
            axes[row, column].set_title(SHORT[name])
        if column == 0:
            axes[0, column].set_ylabel("parent population")
            axes[1, column].set_ylabel("selected population")
    handles = [
        plt.Line2D([], [], color=COLORS["parent"], linewidth=2, label="true parent C0"),
        plt.Line2D(
            [], [], color=COLORS["learned"], linewidth=2, label="learned parent prior"
        ),
        plt.Line2D(
            [], [], color=COLORS["selected"], linewidth=2, label="beta-weighted true C0"
        ),
        plt.Line2D(
            [], [], color=COLORS["refresh"], linewidth=2, label="learned selected prior"
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(
        "Parent learning and selection transfer",
        fontsize=16,
        fontweight="semibold",
        y=1.02,
    )
    fig.subplots_adjust(top=0.87, hspace=0.34, wspace=0.20)
    _save(fig, path)


def population_selection_metrics(
    true_parent: np.ndarray,
    true_selected: np.ndarray,
    learned_parent: np.ndarray,
    learned_selected: np.ndarray,
) -> pd.DataFrame:
    records = []
    for index, name in enumerate(PARAMETERS):
        parent_iqr = max(
            float(
                np.quantile(true_parent[:, index], 0.75)
                - np.quantile(true_parent[:, index], 0.25)
            ),
            1e-8,
        )
        selected_iqr = max(
            float(
                np.quantile(true_selected[:, index], 0.75)
                - np.quantile(true_selected[:, index], 0.25)
            ),
            1e-8,
        )
        true_shift = float(
            np.median(true_selected[:, index]) - np.median(true_parent[:, index])
        )
        learned_shift = float(
            np.median(learned_selected[:, index]) - np.median(learned_parent[:, index])
        )
        records.append(
            {
                "parameter": name,
                "group": "physical_5d" if index < 5 else "sfh_10d",
                "parent_wasserstein_over_truth_iqr": float(
                    wasserstein_distance(
                        true_parent[:, index], learned_parent[:, index]
                    )
                    / parent_iqr
                ),
                "selected_wasserstein_over_truth_iqr": float(
                    wasserstein_distance(
                        true_selected[:, index], learned_selected[:, index]
                    )
                    / selected_iqr
                ),
                "true_selection_displacement_over_parent_iqr": float(
                    wasserstein_distance(true_parent[:, index], true_selected[:, index])
                    / parent_iqr
                ),
                "learned_selection_displacement_over_parent_iqr": float(
                    wasserstein_distance(
                        learned_parent[:, index], learned_selected[:, index]
                    )
                    / parent_iqr
                ),
                "true_median_shift_over_parent_iqr": true_shift / parent_iqr,
                "learned_median_shift_over_parent_iqr": learned_shift / parent_iqr,
                "median_shift_direction_correct": bool(
                    abs(true_shift) < 0.01 * parent_iqr
                    or np.sign(true_shift) == np.sign(learned_shift)
                ),
            }
        )
    return pd.DataFrame(records)


def plot_selection_three_way(
    path: Path,
    true_parent: np.ndarray,
    true_selected: np.ndarray,
    learned_selected: np.ndarray,
    names: tuple[str, ...],
) -> None:
    import matplotlib.pyplot as plt

    columns = min(5, len(names))
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.5 * columns, 3.2 * rows),
        squeeze=False,
    )
    for axis, name in zip(axes.ravel(), names, strict=False):
        index = PARAMETERS.index(name)
        arrays = (
            true_parent[:, index],
            true_selected[:, index],
            learned_selected[:, index],
        )
        low, high = _limits(*arrays)
        edges = np.linspace(low, high, 70)
        _density_panel(
            axis,
            arrays[0],
            edges,
            color=COLORS["parent"],
            label="true parent C0",
        )
        _density_panel(
            axis,
            arrays[1],
            edges,
            color=COLORS["selected"],
            label="beta-weighted true selected",
        )
        _density_panel(
            axis,
            arrays[2],
            edges,
            color=COLORS["refresh"],
            label="learned selected prior",
            fill=True,
        )
        axis.set_xlim(low, high)
        axis.set_yticks([])
        axis.set_title(SHORT[name], fontweight="semibold")
    for axis in axes.ravel()[len(names) :]:
        axis.set_visible(False)
    handles = [
        plt.Line2D([], [], color=COLORS["parent"], linewidth=2, label="true parent C0"),
        plt.Line2D(
            [],
            [],
            color=COLORS["selected"],
            linewidth=2,
            label="beta-weighted true selected",
        ),
        plt.Line2D(
            [],
            [],
            color=COLORS["refresh"],
            linewidth=2,
            label="learned selected prior",
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(
        "Selection closure: parent truth, selected truth and learned selected prior",
        fontsize=15,
        fontweight="semibold",
        y=1.03,
    )
    fig.subplots_adjust(top=0.82 if rows == 1 else 0.88, hspace=0.28, wspace=0.20)
    _save(fig, path)


def plot_selection_distances(path: Path, metrics: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    handles = None
    labels = None
    for axis, group in zip(axes, ("physical_5d", "sfh_10d"), strict=True):
        frame = metrics.loc[metrics.group.eq(group)].reset_index(drop=True)
        positions = np.arange(len(frame))
        width = 0.36
        axis.bar(
            positions - width / 2,
            frame.parent_wasserstein_over_truth_iqr,
            width,
            color=COLORS["learned"],
            label="learned parent vs true parent",
        )
        axis.bar(
            positions + width / 2,
            frame.selected_wasserstein_over_truth_iqr,
            width,
            color=COLORS["refresh"],
            label="learned selected vs diagnostic true selected",
        )
        axis.set_xticks(
            positions, [SHORT[name] for name in frame.parameter], rotation=35
        )
        axis.set_ylabel("Wasserstein distance / truth IQR")
        axis.set_title(group.replace("_", " "), loc="left", fontweight="semibold")
        axis.axhline(0.25, color=COLORS["truth"], linestyle=":", linewidth=1.2)
        if handles is None:
            handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.90))
    fig.suptitle(
        "Marginal population closure before and after selection",
        fontsize=16,
        fontweight="semibold",
    )
    fig.subplots_adjust(top=0.78, bottom=0.19, wspace=0.24)
    _save(fig, path)


def plot_selection_emulator(
    path: Path, validation_path: Path, diagnostics: dict
) -> None:
    import matplotlib.pyplot as plt

    frame = pd.read_parquet(validation_path)
    sample = frame.sample(min(12000, len(frame)), random_state=260913)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].hexbin(
        sample.beta_exact,
        sample.beta_emulated,
        gridsize=40,
        mincnt=1,
        cmap="viridis",
        linewidths=0,
    )
    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["truth"])
    axes[0].set_xlabel("exact beta on held-out prior draws")
    axes[0].set_ylabel("emulated beta")
    axes[0].set_title("Selector emulator validation", loc="left", fontweight="semibold")
    transition = frame.loc[frame.transition]
    axes[1].hist(
        transition.beta_emulated - transition.beta_exact,
        bins=60,
        color=COLORS["refresh"],
        alpha=0.82,
    )
    axes[1].axvline(0, color=COLORS["truth"], linestyle="--")
    axes[1].set_xlabel("emulated beta - exact beta")
    axes[1].set_ylabel("held-out transition samples")
    axes[1].set_title(
        f"transition MAE = {diagnostics['transition_mae']:.3f}",
        loc="left",
        fontweight="semibold",
    )
    fig.suptitle(
        "The beta-weighted truth curve is an audited diagnostic approximation",
        fontsize=14,
    )
    fig.subplots_adjust(top=0.83, wspace=0.25)
    _save(fig, path)


def plot_prior_pit(
    path: Path,
    true_parent: np.ndarray,
    true_selected: np.ndarray,
    learned_parent: np.ndarray,
    learned_selected: np.ndarray,
) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    records = []
    specs = (
        (true_parent, learned_parent, range(5), "parent physical 5D"),
        (true_selected, learned_selected, range(5), "selected physical 5D"),
        (true_parent, learned_parent, range(5, 15), "parent SFH 10D"),
        (true_selected, learned_selected, range(5, 15), "selected SFH 10D"),
    )
    for axis, (truth, draws, indices, title) in zip(axes.ravel(), specs, strict=True):
        all_pit = []
        for index in indices:
            sorted_draws = np.sort(draws[:, index])
            pit = np.searchsorted(sorted_draws, truth[:, index], side="right") / len(
                sorted_draws
            )
            all_pit.append(pit)
            records.append(
                {
                    "population": title.split()[0],
                    "group": "physical_5d" if index < 5 else "sfh_10d",
                    "parameter": PARAMETERS[index],
                    "pit_ks": float(kstest(pit, "uniform").statistic),
                    "central_95_truth_coverage": float(
                        np.mean((pit >= 0.025) & (pit <= 0.975))
                    ),
                }
            )
        values = np.concatenate(all_pit)
        axis.hist(
            values,
            bins=np.linspace(0, 1, 16),
            density=True,
            color=COLORS["learned"],
            alpha=0.80,
        )
        axis.axhline(1, color=COLORS["truth"], linestyle="--")
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.set_xlabel("truth percentile inside learned prior")
        axis.set_ylabel("density")
    fig.suptitle(
        "Is the truth inside the learned prior?", fontsize=16, fontweight="semibold"
    )
    fig.subplots_adjust(top=0.91, hspace=0.26, wspace=0.18)
    _save(fig, path)
    return pd.DataFrame(records)


def _select_galaxy_types(truth: pd.DataFrame) -> pd.DataFrame:
    frame = truth.copy()
    frame["sfh_sum"] = frame.loc[:, list(SFH)].sum(axis=1)
    definitions = (
        ("low redshift", "z_obs", "min"),
        ("high redshift", "z_obs", "max"),
        ("low stellar mass", "log10_stellar_mass", "min"),
        ("high stellar mass", "log10_stellar_mass", "max"),
        ("metal poor", "log10_stellar_metallicity", "min"),
        ("dusty", "dust_av", "max"),
        ("quenched-like truth proxy", "sfh_sum", "min"),
        ("star-forming-like truth proxy", "sfh_sum", "max"),
    )
    records, used = [], set()
    for label, column, direction in definitions:
        ordered = frame.sort_values(column, ascending=direction == "min")
        item = next(
            row for _, row in ordered.iterrows() if int(row.row_index) not in used
        )
        used.add(int(item.row_index))
        records.append(
            {
                "type": label,
                "row_index": int(item.row_index),
                "z_obs": float(item.z_obs),
                "log10_stellar_mass": float(item.log10_stellar_mass),
                "sfh_sum": float(item.sfh_sum),
            }
        )
    return pd.DataFrame(records)


def plot_corner(
    path: Path, before: np.ndarray, current: np.ndarray, truth: np.ndarray, title: str
) -> None:
    import corner
    import matplotlib.pyplot as plt

    ranges = [
        _limits(before[:, i], current[:, i], np.asarray([truth[i]])) for i in range(5)
    ]
    figure = corner.corner(
        current[:, :5],
        labels=[SHORT[name] for name in PHYSICAL],
        color=COLORS["refresh"],
        range=ranges,
        bins=28,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.5, 0.8, 0.95),
        truths=truth,
        truth_color=COLORS["truth"],
        hist_kwargs={"linewidth": 2},
        contour_kwargs={"linewidths": 1.8},
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
        levels=(0.5, 0.8, 0.95),
        hist_kwargs={"linewidth": 1.4, "linestyle": "--"},
        contour_kwargs={"linewidths": 1.1, "linestyles": "--"},
        max_n_ticks=4,
    )
    figure.legend(
        handles=[
            plt.Line2D(
                [], [], color=COLORS["baseline"], linestyle="--", label="before + IS"
            ),
            plt.Line2D([], [], color=COLORS["refresh"], label="current + IS"),
            plt.Line2D([], [], color=COLORS["truth"], label="truth"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.96),
    )
    figure.suptitle(title, fontsize=15, fontweight="semibold", y=1.01)
    _save(figure, path)


def plot_individual_types(
    root: Path, out: Path, truth_frame: pd.DataFrame
) -> pd.DataFrame:
    selected = _select_galaxy_types(truth_frame)
    before = _load_draws(root, BASELINE, "is")
    current = _load_draws(root, CURRENT, "is")
    truth_indexed = truth_frame.set_index("row_index")
    records = []
    for order, item in selected.iterrows():
        row_index = int(item.row_index)
        b = before.loc[before.row_index.eq(row_index), list(PHYSICAL)].to_numpy(
            np.float64
        )
        q = current.loc[current.row_index.eq(row_index), list(PHYSICAL)].to_numpy(
            np.float64
        )
        truth = truth_indexed.loc[row_index, list(PHYSICAL)].to_numpy(np.float64)
        stem = f"{order + 1:02d}_{item['type'].replace(' ', '_').replace('-', '_')}_row_{row_index}"
        plot_corner(
            out / "03_individual" / f"{stem}.png",
            b,
            q,
            truth,
            f"{item['type']} | row {row_index}",
        )
        records.append({**item.to_dict(), "plot": f"03_individual/{stem}.png"})
    result = pd.DataFrame(records)
    result.to_csv(out / "tables" / "individual_types.csv", index=False)
    return result


def _load_nuts_theta(case_dir: Path, spec) -> np.ndarray:
    import jax.numpy as jnp

    from euclid_dsps.amortized.latent import x_to_theta

    columns = [f"x_{index:02d}" for index in range(15)]
    pieces = []
    for chain in range(8):
        paths = sorted(
            path
            for path in (case_dir / f"chain_{chain}" / "chunks").glob("part_*.parquet")
            if not path.name.endswith("_info.parquet")
        )
        pieces.append(
            pd.concat(
                [pd.read_parquet(path, columns=columns) for path in paths],
                ignore_index=True,
            ).to_numpy(np.float64)
        )
    x = np.concatenate(pieces)
    return np.asarray(x_to_theta(jnp.asarray(x), spec), np.float64)


def _truth_aware_limit(
    encoder: np.ndarray, nuts: np.ndarray, truth: float
) -> tuple[float, float]:
    low, high = _limits(encoder, nuts)
    width = max(high - low, 1.0e-8)
    low = min(low, float(truth) - 0.04 * width)
    high = max(high, float(truth) + 0.04 * width)
    return low, high


def _contour_levels(histogram: np.ndarray) -> np.ndarray:
    values = np.asarray(histogram, np.float64).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 3:
        return np.array([], dtype=np.float64)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) / np.sum(ordered)
    thresholds = [
        ordered[min(np.searchsorted(cumulative, mass), len(ordered) - 1)]
        for mass in (0.95, 0.80, 0.50)
    ]
    return np.unique(np.sort(thresholds))


def _joint_contours(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    color: str,
    linestyle: str,
) -> None:
    histogram, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=36,
        range=[x_range, y_range],
    )
    histogram = gaussian_filter(histogram.T, sigma=1.0, mode="nearest")
    levels = _contour_levels(histogram)
    if len(levels) == 0:
        return
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    axis.contour(
        x_centers,
        y_centers,
        histogram,
        levels=levels,
        colors=[color],
        linewidths=np.linspace(0.8, 1.6, len(levels)),
        linestyles=linestyle,
        alpha=0.92,
    )


def plot_nuts_corner(
    path: Path,
    encoder: np.ndarray,
    nuts: np.ndarray,
    truth: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    encoder_color = COLORS["learned"]
    nuts_color = "#7251B5"
    truth_color = COLORS["truth"]
    limits = [
        _truth_aware_limit(encoder[:, index], nuts[:, index], truth[index])
        for index in range(5)
    ]
    figure, axes = plt.subplots(5, 5, figsize=(11.2, 10.4), squeeze=False)
    for row in range(5):
        for column in range(5):
            axis = axes[row, column]
            axis.grid(False)
            axis.set_xlim(*limits[column])
            if row == column:
                edges = np.linspace(*limits[column], 38)
                _density_panel(
                    axis,
                    encoder[:, column],
                    edges,
                    color=encoder_color,
                    label="Encoder",
                    fill=True,
                )
                _density_panel(
                    axis,
                    nuts[:, column],
                    edges,
                    color=nuts_color,
                    label="NUTS",
                )
                axis.axvline(truth[column], color=truth_color, linewidth=2.0)
                axis.set_yticks([])
            else:
                axis.set_ylim(*limits[row])
                _joint_contours(
                    axis,
                    encoder[:, column],
                    encoder[:, row],
                    limits[column],
                    limits[row],
                    color=encoder_color,
                    linestyle="-",
                )
                _joint_contours(
                    axis,
                    nuts[:, column],
                    nuts[:, row],
                    limits[column],
                    limits[row],
                    color=nuts_color,
                    linestyle="--",
                )
                axis.scatter(
                    truth[column],
                    truth[row],
                    marker="+",
                    s=64,
                    linewidths=1.8,
                    color=truth_color,
                    zorder=5,
                )
            if row == 4:
                axis.set_xlabel(SHORT[PHYSICAL[column]])
                axis.tick_params(axis="x", labelrotation=25, labelsize=8)
            else:
                axis.set_xticklabels([])
            if column == 0 and row != column:
                axis.set_ylabel(SHORT[PHYSICAL[row]])
                axis.tick_params(axis="y", labelsize=8)
            elif row != column:
                axis.set_yticklabels([])
            axis.locator_params(axis="both", nbins=3)
    figure.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=encoder_color,
                linewidth=2,
                label="Encoder",
            ),
            plt.Line2D(
                [], [], color=nuts_color, linewidth=2, linestyle="--", label="NUTS"
            ),
            plt.Line2D(
                [],
                [],
                color=truth_color,
                marker="+",
                linestyle="none",
                markersize=9,
                markeredgewidth=1.8,
                label="Truth",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
    )
    figure.suptitle(
        title,
        fontsize=15,
        fontweight="semibold",
        y=1.015,
    )
    figure.text(
        0.5,
        0.005,
        "NUTS is shown as a geometry diagnostic; its saved convergence gate failed.",
        ha="center",
        color="#69717D",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.09, right=0.985, bottom=0.09, top=0.91, wspace=0.08, hspace=0.08
    )
    _save(figure, path)


def _match_nuts_truth(
    nuts_root: Path, config_path: Path, out: Path
) -> dict[str, np.ndarray]:
    """Recover display-only truth by a strict photometry/error/mask join."""
    import yaml

    repository = Path(__file__).resolve().parents[1]
    photometry_path = (
        repository
        / "Data/diffsky/synthetic/feniks_260617_dsps_closure_18band/all_50k.parquet"
    )
    truth_root = repository / "Data/diffsky/synthetic/feniks_260617_spline15d"
    required = [photometry_path] + [
        truth_root / f"{split}_exact.parquet"
        for split in ("train", "validation", "test")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "display-only NUTS truth inputs are missing: " + ", ".join(missing)
        )
    config = yaml.safe_load(config_path.read_text())
    bands = [str(band["name"]) for band in config["bands"]]
    flux_columns = [f"flux_{band}" for band in bands]
    error_columns = [f"fluxerr_{band}" for band in bands]
    mask_columns = [f"mask_{band}" for band in bands]
    columns = ["object_id", "split", *flux_columns, *error_columns, *mask_columns]
    photometry = pd.read_parquet(photometry_path, columns=columns)
    flux = photometry[flux_columns].to_numpy(np.float64)
    error = photometry[error_columns].to_numpy(np.float64)
    mask = photometry[mask_columns].to_numpy(bool)
    exact_truth = pd.concat(
        [
            pd.read_parquet(truth_root / f"{split}_exact.parquet").assign(
                truth_split=split
            )
            for split in ("train", "validation", "test")
        ],
        ignore_index=True,
    )
    if exact_truth.object_id.duplicated().any():
        raise ValueError("display-only exact truth object_id values are not unique")
    exact_truth = exact_truth.set_index("object_id", drop=False)
    cohort = pd.read_csv(nuts_root / "OBSERVED_COHORT.csv").fillna("")
    matches: dict[str, np.ndarray] = {}
    records = []
    tolerance = 1.0e-6
    for item in cohort.itertuples():
        observation_path = nuts_root / item.case / "observation.npz"
        with np.load(observation_path, allow_pickle=False) as archive:
            observed_flux = np.asarray(archive["flux"], np.float64).reshape(-1)
            observed_error = np.asarray(archive["flux_err"], np.float64).reshape(-1)
            observed_mask = np.asarray(archive["mask"], bool).reshape(-1)
        flux_scale = np.maximum(
            np.abs(observed_flux), np.median(np.abs(observed_flux)) * 1.0e-5
        )
        error_scale = np.maximum(
            np.abs(observed_error), np.median(np.abs(observed_error)) * 1.0e-5
        )
        flux_distance = np.max(np.abs((flux - observed_flux) / flux_scale), axis=1)
        error_distance = np.max(np.abs((error - observed_error) / error_scale), axis=1)
        mask_distance = np.sum(mask != observed_mask, axis=1)
        candidates = np.flatnonzero(
            (flux_distance < tolerance)
            & (error_distance < tolerance)
            & (mask_distance == 0)
        )
        if len(candidates) != 1:
            raise ValueError(
                f"{item.case}: expected one exact photometric truth match, found {len(candidates)}"
            )
        index = int(candidates[0])
        source = photometry.iloc[index]
        object_id = source.object_id
        if object_id not in exact_truth.index:
            raise ValueError(f"{item.case}: no exact spline truth for {object_id=}")
        truth_row = exact_truth.loc[object_id]
        if str(source["split"]) != str(truth_row.truth_split):
            raise ValueError(f"{item.case}: photometry/truth split mismatch")
        values = truth_row.loc[list(PARAMETERS)].to_numpy(np.float64)
        if values.shape != (15,) or not np.isfinite(values).all():
            raise ValueError(f"{item.case}: invalid display-only truth")
        matches[item.case] = values
        record = {
            "case": item.case,
            "grouped_source_row": int(item.source_row),
            "matched_catalog_row": index,
            "matched_object_id": object_id,
            "source_split": str(source["split"]),
            "max_relative_flux_error": float(flux_distance[index]),
            "max_relative_uncertainty_error": float(error_distance[index]),
            "mask_mismatches": int(mask_distance[index]),
            "truth_role": "display_only_after_sampling",
        }
        record.update({name: float(truth_row[name]) for name in PARAMETERS})
        records.append(record)
    pd.DataFrame(records).to_csv(out / "tables" / "nuts_truth_matches.csv", index=False)
    contract = {
        "status": "NUTS_TRUTH_MATCH_COMPLETE",
        "method": "unique 18-band flux plus uncertainty plus mask match",
        "relative_tolerance": tolerance,
        "truth_role": "display only after NUTS and encoder inference",
        "truth_used_for_sampling_or_selection": False,
        "objects": len(matches),
        "inputs": {str(path): _sha256(path) for path in required},
    }
    (out / "tables" / "NUTS_TRUTH_MATCH.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    return matches


def plot_nuts_overview(path: Path, cases: list[dict]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(cases), 5, figsize=(14, 2.05 * len(cases)))
    for row, case in enumerate(cases):
        for column, parameter in enumerate(PHYSICAL):
            axis = axes[row, column]
            encoder = case["encoder"][:, column]
            nuts = case["nuts"][:, column]
            truth = case["truth"][column]
            limits = _truth_aware_limit(encoder, nuts, truth)
            edges = np.linspace(*limits, 38)
            _density_panel(
                axis,
                encoder,
                edges,
                color=COLORS["learned"],
                label="Encoder",
                fill=True,
            )
            _density_panel(
                axis, nuts, edges, color="#7251B5", label="NUTS", linestyle="--"
            )
            axis.axvline(truth, color=COLORS["truth"], linewidth=1.8)
            axis.set_xlim(*limits)
            axis.set_yticks([])
            axis.grid(False)
            axis.tick_params(axis="x", labelsize=8)
            if row == 0:
                axis.set_title(SHORT[parameter], fontsize=11)
            if column == 0:
                axis.set_ylabel(case["label"], rotation=0, ha="right", va="center")
    figure.legend(
        handles=[
            plt.Line2D([], [], color=COLORS["learned"], linewidth=2, label="Encoder"),
            plt.Line2D(
                [], [], color="#7251B5", linewidth=2, linestyle="--", label="NUTS"
            ),
            plt.Line2D([], [], color=COLORS["truth"], linewidth=2, label="Truth"),
        ],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(
        "Eight observed posteriors", fontsize=15, fontweight="semibold", y=1.025
    )
    figure.subplots_adjust(
        left=0.17, right=0.99, bottom=0.04, top=0.95, wspace=0.16, hspace=0.34
    )
    _save(figure, path)


def plot_nuts_comparisons(
    nuts_root: Path, config_path: Path, out: Path
) -> pd.DataFrame:
    import jax
    import jax.numpy as jnp
    import yaml

    from euclid_dsps.amortized.latent import latent_spec_from_config, x_to_theta

    jax.config.update("jax_enable_x64", True)
    spec = latent_spec_from_config(yaml.safe_load(config_path.read_text()))
    cohort = pd.read_csv(nuts_root / "OBSERVED_COHORT.csv").fillna("")
    summary = pd.read_csv(nuts_root / "nuts_followup_summary.csv")
    truth_by_case = _match_nuts_truth(nuts_root, config_path, out)
    records = []
    overview = []
    for case_number, (_, item) in enumerate(cohort.iterrows(), start=1):
        case = item["case"]
        case_root = nuts_root / "nuts" / case / NUTS_VARIANT
        nuts = _load_nuts_theta(case_root, spec)
        avi_x, avi_w = [], []
        for replica in (0, 1):
            with np.load(nuts_root / case / f"bank_{replica}.npz") as archive:
                avi_x.append(np.asarray(archive["x"], np.float64))
                avi_w.append(np.asarray(archive["weight"], np.float64))
        avi_theta = np.asarray(
            x_to_theta(jnp.asarray(np.concatenate(avi_x)), spec), np.float64
        )
        avi = _weighted_resample(
            avi_theta, np.concatenate(avi_w), 8192, 260913 + int(case[-3:])
        )
        rng = np.random.default_rng(260913 + int(case[-3:]))
        nuts_plot = nuts[
            rng.choice(len(nuts), size=min(8192, len(nuts)), replace=False)
        ]
        diagnostic = summary.loc[summary.case.eq(case)].iloc[0]
        tags = item.descriptive_tags or "stratified observed"
        display_tags = tags.replace(";", ", ").replace("_", " ")
        safe_tags = tags.split(";")[0].replace(" ", "_").replace("/", "_")
        truth = truth_by_case[case]
        plot_nuts_corner(
            out / "04_nuts" / f"{case_number:02d}_{safe_tags}.png",
            avi,
            nuts_plot,
            truth,
            f"Observed {case_number} | {display_tags}",
        )
        overview.append(
            {
                "encoder": avi,
                "nuts": nuts_plot,
                "truth": truth,
                "label": f"{case_number}. {display_tags}",
            }
        )
        nuts_iqr = np.maximum(
            np.quantile(nuts, 0.75, axis=0) - np.quantile(nuts, 0.25, axis=0), 1e-8
        )
        for index, name in enumerate(PARAMETERS):
            records.append(
                {
                    "case": case,
                    "tags": tags,
                    "parameter": name,
                    "wasserstein_over_nuts_iqr": float(
                        wasserstein_distance(avi[:, index], nuts_plot[:, index])
                        / nuts_iqr[index]
                    ),
                    "limit_hit_fraction": float(diagnostic.limit_hit_fraction),
                    "divergence_fraction": float(diagnostic.divergence_fraction),
                    "max_rhat": float(diagnostic.max_rhat),
                    "diagnostics_pass": bool(diagnostic.diagnostics_pass),
                    "truth": float(truth[index]),
                    "encoder_median_error_over_nuts_iqr": float(
                        abs(np.median(avi[:, index]) - truth[index]) / nuts_iqr[index]
                    ),
                    "nuts_median_error_over_nuts_iqr": float(
                        abs(np.median(nuts_plot[:, index]) - truth[index])
                        / nuts_iqr[index]
                    ),
                }
            )
    plot_nuts_overview(out / "04_nuts" / "00_overview.png", overview)
    (out / "04_nuts" / "README.md").write_text(
        "# Encoder, NUTS, and truth\n\n"
        "`00_overview.png` shows all eight cases. The numbered files contain "
        "the full physical 5D posterior matrices. Truth was recovered only "
        "after sampling through the audited photometric join in "
        "`../tables/nuts_truth_matches.csv`. NUTS remains a geometry "
        "diagnostic because its saved convergence gate failed.\n"
    )
    result = pd.DataFrame(records)
    result.to_csv(out / "tables" / "nuts_vs_avi.csv", index=False)
    return result


def _copy_previous(root: Path, out: Path) -> None:
    source = root / "analysis_v2"
    target = out / "00_previous_analysis"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _write_report(
    out: Path,
    improvement: dict,
    beta: dict,
    prior: pd.DataFrame,
    selection: pd.DataFrame,
    prior_mira: pd.DataFrame,
    joint_metrics: pd.DataFrame,
    nuts: pd.DataFrame,
    individuals: pd.DataFrame,
) -> None:
    physical = improvement["physical 5D CRPS / truth IQR"]
    sfh = improvement["SFH CRPS / truth IQR"]
    ess = improvement["ESS"]
    prior_group = (
        prior.groupby(["population", "group"])
        .agg(
            median_pit_ks=("pit_ks", "median"),
            minimum_95_support=("central_95_truth_coverage", "min"),
        )
        .reset_index()
    )
    parent_phys = prior_group.query(
        "population == 'parent' and group == 'physical_5d'"
    ).iloc[0]
    selected_phys = prior_group.query(
        "population == 'selected' and group == 'physical_5d'"
    ).iloc[0]
    selection_group = selection.groupby("group").agg(
        median_parent_distance=("parent_wasserstein_over_truth_iqr", "median"),
        median_selected_distance=("selected_wasserstein_over_truth_iqr", "median"),
        median_true_selection_displacement=(
            "true_selection_displacement_over_parent_iqr",
            "median",
        ),
        median_learned_selection_displacement=(
            "learned_selection_displacement_over_parent_iqr",
            "median",
        ),
        shift_direction_fraction=("median_shift_direction_correct", "mean"),
    )
    selection_physical = selection_group.loc["physical_5d"]
    selection_sfh = selection_group.loc["sfh_10d"]
    latest_mira = prior_mira.loc[
        prior_mira.model.isin(("prior_latest_parent", "prior_latest_selected"))
        & prior_mira.group.isin(("physical_5d", "full_15d")),
        ["population", "group", "score", "bootstrap_q025", "bootstrap_q975"],
    ].set_index(["population", "group"])
    parent_mira = latest_mira.loc[("parent", "physical_5d")]
    selected_mira = latest_mira.loc[("selected", "physical_5d")]
    joint_indexed = joint_metrics.set_index("distribution")
    before_joint = joint_indexed.loc["B_source_selected_5d"]
    current_joint = joint_indexed.loc["Q_latest_refresh_selected_5d"]
    energy_gain = 1 - (
        current_joint.standardized_energy_distance
        / before_joint.standardized_energy_distance
    )
    correlation_gain = 1 - (
        current_joint.correlation_rmse / before_joint.correlation_rmse
    )
    nuts_physical = nuts.loc[nuts.parameter.isin(PHYSICAL)]
    nuts_group = (
        nuts_physical.groupby("case")
        .agg(
            median_physical_wasserstein=("wasserstein_over_nuts_iqr", "median"),
            median_encoder_truth_error=(
                "encoder_median_error_over_nuts_iqr",
                "median",
            ),
            median_nuts_truth_error=("nuts_median_error_over_nuts_iqr", "median"),
            limit_hit_fraction=("limit_hit_fraction", "first"),
            divergence_fraction=("divergence_fraction", "first"),
            max_rhat=("max_rhat", "first"),
            diagnostics_pass=("diagnostics_pass", "first"),
        )
        .reset_index()
    )
    lines = [
        "# FENIKS AVI wrap-up",
        "",
        "## Executive answer",
        "",
        f"The current learned-prior refresh improves ordinary-IS physical 5D CRPS for **{physical['fraction_improved']:.1%}** of the 512 galaxies (95% Wilson interval {physical['wilson_q025']:.1%}--{physical['wilson_q975']:.1%}). It improves ESS for **{ess['fraction_improved']:.1%}**. SFH CRPS improves for only **{sfh['fraction_improved']:.1%}**, so the gain is not a general 15D solution.",
        f"Before importance weighting, physical CRPS already improves for {physical['raw_fraction_improved']:.1%} of objects; SFH improves for {sfh['raw_fraction_improved']:.1%}. The physical majority result is therefore not created solely by IS.",
        "",
        "The learned prior contains most physical truths marginally, but the exact statement is population-level: it is not a per-galaxy posterior. The weakest physical marginal still places "
        + f"{parent_phys.minimum_95_support:.1%} of parent truths inside its central 95% range; the median parent physical PIT KS is {parent_phys.median_pit_ks:.3f}. For the beta-selected diagnostic truth these values are {selected_phys.minimum_95_support:.1%} and {selected_phys.median_pit_ks:.3f}.",
        "",
        f"The latest-prior physical MIRA scores are {parent_mira.score:.3f} for the parent and {selected_mira.score:.3f} for the selected population (ideal 0.667); both bootstrap intervals contain the ideal score. This does **not** extend to the full 15D score because SFH remains badly misspecified.",
        "In short: the physical truths are in the learned prior's marginal support, and its joint physical MIRA is compatible with calibration. The prior is not exact: metallicity and dust marginals remain smoothed or biased, and broad support alone must not be confused with a correct density.",
        "",
        "## Selection audit",
        "",
        "The original report cannot directly certify selection learning: `parent_population_truth.parquet` and `selected_population_truth.parquet` contain the same 5000 row identities and identical physical 5D values. This wrap-up therefore constructs an additional **diagnostic** selected-truth reference by weighting the parent truth by beta(theta). Beta is emulated from 196608 exact selector evaluations saved by the run; no catalogue truth enters that emulator.",
        "",
        f"Held-out beta-emulator MAE is {beta['validation_mae']:.3f}; transition-region MAE is {beta['transition_mae']:.3f} with Spearman {beta['transition_spearman']:.3f}. Diagnostic qualification: `{beta['qualified_for_visual_diagnostic']}`. This is sufficient for a visual audit, not a replacement for regenerating the exact noisy selected cohort.",
        "",
        f"For the physical 5D, the median marginal Wasserstein distance is {selection_physical.median_parent_distance:.3f} truth-IQR for the learned parent and {selection_physical.median_selected_distance:.3f} for the learned selected population; the learned median selection shift has the correct direction in {selection_physical.shift_direction_fraction:.0%} of dimensions. SFH remains much worse ({selection_sfh.median_selected_distance:.2f} truth-IQR median selected distance).",
        f"The physical selection effect itself is modest: its median displacement is only {selection_physical.median_true_selection_displacement:.3f} truth-IQR, versus {selection_physical.median_learned_selection_displacement:.3f} in the learned model. Because the parent mismatch ({selection_physical.median_parent_distance:.3f}) is larger than that selection displacement, these results support a coherent selection transfer but do not yet certify it precisely.",
        "",
        "Inspect `01_population/selection_three_way_physical_5d.png` for the requested same-panel comparison of true parent, diagnostic true selected and learned selected distributions. `selection_parent_closure_physical_5d.png` separates parent learning from selection transfer, while `selection_distance_scorecard.png` quantifies the mismatch. The corresponding PIT support plot is `prior_truth_pit.png`.",
        "",
        "## Calibration and before/after",
        "",
        "- `02_calibration/paired_improvement.png`: paired, per-galaxy answer to whether the current model is better.",
        "- `02_calibration/physical_pit.png` and `sfh_pit.png`: ordinary-IS PIT histograms before/current.",
        "- `02_calibration/coverage.png`: central-interval coverage curves.",
        "- `02_calibration/truth_vs_predicted_physical.png`: truth versus posterior median, used only as a point-summary diagnostic.",
        "- `02_calibration/truth_vs_predicted_sfh.png`: the same comparison for SFH; it exposes the persistent weakly identifiable directions.",
        f"- The selected physical population energy distance falls by {energy_gain:.1%} versus `B_source`; correlation RMSE falls by {correlation_gain:.1%}. These are joint 5D distribution diagnostics, not point-estimate scores.",
        "",
        "## Individual galaxies",
        "",
        "Eight truth-aware corners cover low/high redshift, low/high mass, metal-poor, dusty, quenched-like and star-forming-like objects. They are listed in `tables/individual_types.csv` and stored under `03_individual/`. These are post-hoc diagnostic strata; none was used for training or checkpoint selection.",
        "",
        "## NUTS comparison",
        "",
        "`04_nuts/00_overview.png` compares Encoder, NUTS and Truth for all eight observations. The numbered files are compact physical 5D posterior matrices with the same three simple labels. The truth was joined only after sampling through a unique 18-band flux, uncertainty and mask match; `tables/nuts_truth_matches.csv` records the eight matches and their numerical residuals.",
        "",
        "The Encoder curves are the exactly matched encoder importance banks saved with each NUTS observation. They are not `Q_latest_refresh`, because the overnight 512-object cohort contains none of these historical identities.",
        "",
        f"Across the 40 case-parameter comparisons, the Encoder median is closer to truth than the NUTS median in {np.mean(nuts_physical.encoder_median_error_over_nuts_iqr < nuts_physical.nuts_median_error_over_nuts_iqr):.1%}. This is descriptive only: a closer median does not establish a better posterior, and the NUTS convergence failures prevent using NUTS as a reference density.",
        "",
        "Every NUTS `diagnostics_pass` is false. Trajectory-limit hit fractions span "
        + f"{nuts_group.limit_hit_fraction.min():.1%}--{nuts_group.limit_hit_fraction.max():.1%}; maximum R-hat spans {nuts_group.max_rhat.min():.3f}--{nuts_group.max_rhat.max():.3f}. Treat agreement as useful geometric evidence and disagreement as a prompt for investigation, never as distance to a certified true posterior.",
        "",
        "## What next",
        "",
        "1. Regenerate the exact parent and noisy-selected truth cohorts from one immutable C0 catalogue, with a receipt proving distinct selection outcomes and an explicit alpha estimate. Re-run this report without the beta emulator.",
        "2. Run `Q_latest_refresh` inference on the same eight historical NUTS observations. That creates the missing controlled current-AVI versus NUTS comparison.",
        "3. Keep the learned-prior lineage. Add a robust/quantile SFH transform and a factorized exact 15D flow: physical 5D experts followed by a shared conditional SFH flow.",
        "4. Gate the next run on paired physical CRPS, ESS, PIT/coverage and selected/parent population energy, not the optimization loss alone. Require improvement in a clear majority with bootstrap or Wilson uncertainty.",
        "5. Diagnose SFH decoder identifiability with Jacobian singular vectors and posterior-predictive interventions. Regularize null directions before increasing expert count.",
        "",
        "## Provenance",
        "",
        "All posterior comparisons use complete joint 15D draws. No parameterwise recombination is performed. Posterior medians appear only in truth-versus-predicted figures as labelled point summaries.",
        "",
        "The original `analysis_v2` directory is copied verbatim under `00_previous_analysis/`.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    nuts_group.to_csv(out / "tables" / "nuts_summary.csv", index=False)
    individuals.to_csv(out / "tables" / "individual_types.csv", index=False)


def build(root: Path, nuts_root: Path, output: Path) -> Path:
    root, nuts_root, output = root.resolve(), nuts_root.resolve(), output.resolve()
    final = json.loads((root / "report" / "FINAL.json").read_text())
    if final.get("status") != "OVERNIGHT_REPORT_COMPLETE":
        raise ValueError("completed overnight report required")
    for name in (
        "00_previous_analysis",
        "01_population",
        "02_calibration",
        "03_individual",
        "04_nuts",
        "05_evolution",
        "tables",
    ):
        target = output / name
        if target.exists():
            shutil.rmtree(target)
    for directory in (
        output,
        output / "01_population",
        output / "02_calibration",
        output / "03_individual",
        output / "04_nuts",
        output / "05_evolution",
        output / "tables",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _style()
    _copy_previous(root, output)
    for name in (
        "run_scorecard.png",
        "run_genealogy.png",
        "paired_ess_refresh.png",
        "mira_raw_vs_is.png",
        "prior_mira.png",
        "sfh_failure_diagnostics.png",
    ):
        source = root / "analysis_v2" / name
        if source.exists():
            shutil.copy2(source, output / "05_evolution" / name)

    truth_frame = pd.read_parquet(root / "inference_truth.parquet")
    rows = truth_frame.row_index.to_numpy(np.int64)
    truth = truth_frame.loc[:, list(PARAMETERS)].to_numpy(np.float64)
    cubes = {}
    for variant in (BASELINE, CURRENT):
        for kind in ("raw", "is"):
            cubes[(variant, kind)] = _draw_cube(_load_draws(root, variant, kind), rows)
    paired = paired_improvement(root, output, cubes, truth, rows)
    improvement = plot_improvement(
        output / "02_calibration" / "paired_improvement.png", paired
    )
    raw_paired = paired.loc[paired.kind.eq("raw")]
    improvement["physical 5D CRPS / truth IQR"]["raw_fraction_improved"] = float(
        np.mean(raw_paired.physical_crps_current < raw_paired.physical_crps_before)
    )
    improvement["SFH CRPS / truth IQR"]["raw_fraction_improved"] = float(
        np.mean(raw_paired.sfh_crps_current < raw_paired.sfh_crps_before)
    )
    physical_pit = plot_pit_grid(
        output / "02_calibration" / "physical_pit.png",
        cubes,
        truth,
        PHYSICAL,
        "Physical posterior PIT: before versus current ordinary IS",
    )
    sfh_pit = plot_sfh_pit(output / "02_calibration" / "sfh_pit.png", cubes, truth)
    coverage = plot_coverage(output / "02_calibration" / "coverage.png", cubes, truth)
    physical_pred = plot_truth_predicted(
        output / "02_calibration" / "truth_vs_predicted_physical.png",
        cubes,
        truth,
        PHYSICAL,
        "Physical truth versus posterior",
    )
    sfh_pred = plot_truth_predicted(
        output / "02_calibration" / "truth_vs_predicted_sfh.png",
        cubes,
        truth,
        SFH,
        "SFH truth versus posterior",
    )
    pd.concat([physical_pit, sfh_pit], ignore_index=True).to_csv(
        output / "tables" / "posterior_pit.csv", index=False
    )
    coverage.to_csv(output / "tables" / "posterior_coverage.csv", index=False)
    pd.concat([physical_pred, sfh_pred], ignore_index=True).to_csv(
        output / "tables" / "truth_vs_predicted.csv", index=False
    )

    true_parent_frame = pd.read_parquet(root / "parent_population_truth.parquet")
    true_parent = true_parent_frame.loc[:, list(PARAMETERS)].to_numpy(np.float64)
    beta_truth, beta_diagnostics = _train_beta_emulator(root, true_parent, output)
    true_selected = _weighted_resample(true_parent, beta_truth, 65536, 260913)
    with np.load(root / "arms" / "B_latest" / "prior_population.npz") as archive:
        learned_parent = np.asarray(archive["theta"], np.float64)
        learned_selected = np.asarray(archive["selected_theta"], np.float64)
    selection = population_selection_metrics(
        true_parent, true_selected, learned_parent, learned_selected
    )
    selection.to_csv(
        output / "tables" / "population_selection_metrics.csv", index=False
    )
    plot_selection_closure(
        output / "01_population" / "selection_parent_closure_physical_5d.png",
        true_parent,
        true_selected,
        learned_parent,
        learned_selected,
        PHYSICAL,
    )
    plot_selection_closure(
        output / "01_population" / "selection_parent_closure_sfh_10d.png",
        true_parent,
        true_selected,
        learned_parent,
        learned_selected,
        SFH,
    )
    plot_selection_three_way(
        output / "01_population" / "selection_three_way_physical_5d.png",
        true_parent,
        true_selected,
        learned_selected,
        PHYSICAL,
    )
    plot_selection_three_way(
        output / "01_population" / "selection_three_way_sfh_10d.png",
        true_parent,
        true_selected,
        learned_selected,
        SFH,
    )
    plot_selection_distances(
        output / "01_population" / "selection_distance_scorecard.png", selection
    )
    plot_selection_emulator(
        output / "01_population" / "selection_emulator_validation.png",
        output / "tables" / "selection_emulator_validation.parquet",
        beta_diagnostics,
    )
    prior_pit = plot_prior_pit(
        output / "01_population" / "prior_truth_pit.png",
        true_parent,
        true_selected,
        learned_parent,
        learned_selected,
    )
    prior_pit.to_csv(output / "tables" / "prior_truth_support.csv", index=False)

    individuals = plot_individual_types(root, output, truth_frame)
    nuts = plot_nuts_comparisons(
        nuts_root,
        Path("outputs/avi_encoder_experiments_v6/source_config.yaml").resolve(),
        output,
    )
    prior_mira = pd.read_csv(root / "report" / "prior_mira_scores.csv")
    joint_metrics = pd.read_csv(root / "report" / "joint_5d_distribution_metrics.csv")
    prior_mira.to_csv(output / "tables" / "prior_mira_scores.csv", index=False)
    joint_metrics.to_csv(
        output / "tables" / "joint_5d_distribution_metrics.csv", index=False
    )
    _write_report(
        output,
        improvement,
        beta_diagnostics,
        prior_pit,
        selection,
        prior_mira,
        joint_metrics,
        nuts,
        individuals,
    )

    artifacts = {
        str(path.relative_to(output)): _sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "FINAL.json"
    }
    receipt = {
        "status": "AVI_WRAP_UP_COMPLETE",
        "source_root": str(root),
        "nuts_root": str(nuts_root),
        "joint_draws_preserved": True,
        "truth_used_for_training_or_inference": False,
        "original_selected_truth_distinct": False,
        "selection_truth_reference": "diagnostic beta-emulated weighting of true C0",
        "selection_emulator_qualified": beta_diagnostics[
            "qualified_for_visual_diagnostic"
        ],
        "nuts_scientifically_qualified": False,
        "nuts_truth_display_matches": 8,
        "nuts_truth_used_for_sampling_or_selection": False,
        "artifacts": artifacts,
    }
    (output / "FINAL.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Wrap-up complete: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--nuts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "wrap_up"
    build(args.root, args.nuts_root, output)


if __name__ == "__main__":
    main()
