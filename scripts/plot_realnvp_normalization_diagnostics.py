#!/usr/bin/env python3
"""Plot the exact input normalization of a supervised Diffsky RealNVP run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PARAMETERS: dict[str, tuple[str, str, str]] = {
    "z_obs": ("Observed redshift z", "Observation / cosmology", "Sets luminosity distance and the rest-frame wavelength mapping."),
    "log10_stellar_mass": ("Stellar mass log10(M*/Msun)", "Stellar population", "Sets the overall stellar luminosity scale."),
    "log10_stellar_metallicity": ("Stellar metallicity log10(Z*/Zsun)", "Stellar population", "Sets the stellar metal content and its SED."),
    "dust_av": ("Dust attenuation A_V [mag]", "Dust attenuation", "Sets the overall attenuation strength."),
    "dust_delta": ("Dust-law slope offset delta", "Dust attenuation", "Tilts the attenuation curve relative to the reference dust law."),
    "diffstar_lgmcrit": ("SF-efficiency pivot mass log10(Mcrit/Msun)", "SFH (Diffstar)", "Sets the halo-mass pivot of the star-formation efficiency relation."),
    "diffstar_lgy_at_mcrit": ("SF-efficiency amplitude at Mcrit (log10)", "SFH (Diffstar)", "Sets the star-formation efficiency at the pivot halo mass."),
    "diffstar_indx_lo": ("Low-mass slope of SF efficiency", "SFH (Diffstar)", "Controls how star-formation efficiency changes below the pivot mass."),
    "diffstar_indx_hi": ("High-mass slope of SF efficiency", "SFH (Diffstar)", "Controls how star-formation efficiency changes above the pivot mass."),
    "diffstar_lg_qt": ("Quenching time log10(tq / Gyr)", "SFH (Diffstar)", "Sets when quenching begins in cosmic time."),
    "diffstar_qlglgdt": ("Quenching transition log-width", "SFH (Diffstar)", "Sets how rapidly the star-formation history transitions into quenching."),
    "diffstar_lg_drop": ("Quenching depth log10(drop)", "SFH (Diffstar)", "Sets the suppression depth after quenching."),
    "diffstar_lg_rejuv": ("Rejuvenation amplitude log10", "SFH (Diffstar)", "Sets the strength of star formation after a quenching episode."),
    "diffmah_logm0": ("Halo mass at a=1 log10(Mhalo/Msun)", "Halo assembly (Diffmah)", "Sets the late-time halo mass scale."),
    "diffmah_logtc": ("Halo transition time log10(tc / Gyr)", "Halo assembly (Diffmah)", "Sets when halo mass accretion changes regime."),
    "diffmah_early_index": ("Early halo-growth index", "Halo assembly (Diffmah)", "Controls the early-time halo mass-accretion slope."),
    "diffmah_late_index": ("Late halo-growth index", "Halo assembly (Diffmah)", "Controls the late-time halo mass-accretion slope."),
    "diffmah_t_peak": ("Halo assembly peak time [Gyr]", "Halo assembly (Diffmah)", "Sets the time of peak halo mass accretion."),
}

USEFUL_PARAMETER_ORDER = (
    "z_obs",
    "log10_stellar_mass",
    "log10_stellar_metallicity",
    "dust_av",
    "dust_delta",
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffmah_logm0",
    "diffmah_t_peak",
)

DIRAC_PARAMETER_ORDER = (
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
)

GAUSSIAN_PROBABILITIES = np.linspace(0.005, 0.995, 199)
GAUSSIAN_QUANTILES = np.asarray(
    [NormalDist().inv_cdf(float(probability)) for probability in GAUSSIAN_PROBABILITIES]
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Supervised prior run directory.")
    parser.add_argument("--dataset", type=Path, required=True, help="FENIKS train parquet used for the plot.")
    parser.add_argument("--out", type=Path, help="Output directory; defaults to --run.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hist(ax: plt.Axes, values: np.ndarray, *, color: str, xlabel: str) -> None:
    finite = values[np.isfinite(values)]
    ax.hist(finite, bins=60, density=True, histtype="stepfilled", color=color, alpha=0.72)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)


def _standardize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    center = float(np.mean(values))
    scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return (values - center) / scale, center, scale


def _gaussian_quantile_rmse(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 3:
        return float("nan")
    quantiles = np.quantile(finite, GAUSSIAN_PROBABILITIES)
    return float(np.sqrt(np.mean((quantiles - GAUSSIAN_QUANTILES) ** 2)))


def _fit_asinh_transform(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | bool], list[dict[str, float]]]:
    values = np.asarray(values, dtype=float)
    base_scale = max(float(np.std(values)), 1.0e-6)
    lambdas = base_scale * np.power(10.0, np.linspace(-4.0, 4.0, 161))
    physical_quantiles = np.quantile(values, GAUSSIAN_PROBABILITIES)
    scan_rows: list[dict[str, float]] = []
    best: tuple[float, float, float, float] | None = None
    for value in lambdas:
        transformed = value * np.arcsinh(values / value)
        _, center, scale = _standardize(transformed)
        quantiles = value * np.arcsinh(physical_quantiles / value)
        normalized_quantiles = (quantiles - center) / scale
        score = float(np.sqrt(np.mean((normalized_quantiles - GAUSSIAN_QUANTILES) ** 2)))
        scan_rows.append({"lambda": float(value), "gaussian_quantile_rmse": score})
        if best is None or score < best[0]:
            best = (score, float(value), center, scale)
    assert best is not None
    score, best_lambda, center, scale = best
    transformed = best_lambda * np.arcsinh(values / best_lambda)
    normalized = (transformed - center) / scale
    metadata: dict[str, float | bool] = {
        "lambda": best_lambda,
        "center": center,
        "scale": scale,
        "gaussian_quantile_rmse": score,
        "lambda_at_scan_min": bool(np.isclose(best_lambda, lambdas[0])),
        "lambda_at_scan_max": bool(np.isclose(best_lambda, lambdas[-1])),
    }
    return normalized, metadata, scan_rows


def _parameter_label(name: str) -> tuple[str, str, str]:
    return PARAMETERS.get(name, (name, "Unlabelled parameter", "No description available."))


def _write_glossary(
    path: Path,
    rows: list[dict[str, object]],
    *,
    dataset: Path,
    checkpoint_dataset: str,
    scale_floor: float,
) -> None:
    lines = [
        "# RealNVP normalization diagnostic",
        "",
        "The RealNVP receives `x = (logit((theta - lower)/(upper - lower)) - raw_center) / raw_scale`.",
        "`theta` is the physical parameter; `lower` and `upper` are its configured hard bounds.",
        "The physical value is first mapped to the unit interval. `clip` prevents an infinite logit at either bound.",
        "`raw_center` and `raw_scale` are the mean and standard deviation of this raw logit on the checkpoint training split.",
        "`raw_scale` is clipped to `[0.1, 10]` after it is measured on the training split.",
        "",
        f"- Dataset plotted: `{dataset}`",
        f"- Dataset used by this checkpoint: `{checkpoint_dataset}`",
        "- The plots apply the stored checkpoint center and scale to the dataset plotted above.",
        "",
        "| Role | Readable parameter | What it controls | Stored name | Physical bounds | raw center | raw scale (unclipped -> used) | clipped at lower / upper |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {group} | {label} | {description} | `{name}` | [{lower:.3g}, {upper:.3g}] | "
            "{raw_center:.3f} | {raw_scale_unclipped:.3f} -> {raw_scale:.3f} | "
            "{clipped_low_fraction:.2%} / {clipped_high_fraction:.2%} |".format(**row)
        )
    largest_tail = max(rows, key=lambda row: float(row["x_abs_max"]))
    floored = [
        row for row in rows if np.isclose(float(row["raw_scale"]), scale_floor, rtol=0.0, atol=1.0e-8)
    ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Centering/scaling only makes amplitudes comparable. It does not Gaussianize discrete spikes or remove logit-induced tails.",
            "- Largest amplitude in the plotted dataset: `{name}` with `max(|x|) = {x_abs_max:.2f}`.".format(
                **largest_tail
            ),
        ]
    )
    if floored:
        names = ", ".join(f"`{row['name']}`" for row in floored)
        lines.append(
            "- The checkpoint applied the scale floor "
            f"{scale_floor:g} to {names}; their flow-input standard deviation can therefore remain below 1."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_before_after_plot(
    *,
    names: list[str],
    theta: np.ndarray,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    selected_names: tuple[str, ...],
    title: str,
    path: Path,
) -> None:
    indices = [names.index(name) for name in selected_names if name in names]
    fig, axes = plt.subplots(
        len(indices),
        2,
        figsize=(13, 2.35 * len(indices)),
        constrained_layout=True,
    )
    for row, index in enumerate(indices):
        name = names[index]
        label, role, _ = _parameter_label(name)
        _hist(axes[row, 0], theta[:, index], color="#3979a6", xlabel=label)
        _hist(
            axes[row, 1],
            x[:, index],
            color="#b85c36",
            xlabel="RealNVP input x after logit + standardization",
        )
        axes[row, 0].set_title(f"{role}: {label}", loc="left", fontsize=9, pad=9)
        axes[row, 0].axvline(lower[index], color="0.25", linestyle="--", linewidth=0.8)
        axes[row, 0].axvline(upper[index], color="0.25", linestyle="--", linewidth=0.8)
        axes[row, 1].axvline(0.0, color="0.25", linestyle="--", linewidth=0.8)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_dirac_zoom(
    *,
    names: list[str],
    theta: np.ndarray,
    path: Path,
) -> pd.DataFrame:
    selected_names = [name for name in DIRAC_PARAMETER_ORDER if name in names]
    fig, axes = plt.subplots(
        len(selected_names),
        2,
        figsize=(13, 3.0 * len(selected_names)),
        constrained_layout=True,
    )
    rows = []
    bins = np.linspace(-0.5, 0.5, 101)
    for row, name in enumerate(selected_names):
        index = names.index(name)
        values = theta[:, index]
        unique, counts = np.unique(values, return_counts=True)
        dominant_index = int(np.argmax(counts))
        dominant_value = float(unique[dominant_index])
        dominant_count = int(counts[dominant_index])
        residual = values - dominant_value
        non_atom = residual[values != dominant_value]
        label, role, _ = _parameter_label(name)
        atom_note = (
            f"Exact value: theta = {dominant_value:.12g}\n"
            f"Rows at this value: {dominant_count:,} ({dominant_count / len(values):.3%})"
        )

        axes[row, 0].hist(values, bins=60, color="#3979a6", alpha=0.75)
        axes[row, 0].set_title(f"{role}: {label}", loc="left", fontsize=9)
        axes[row, 0].set_xlabel("Physical theta")
        axes[row, 0].set_ylabel("Count")
        axes[row, 0].text(
            0.98,
            0.92,
            atom_note,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

        axes[row, 1].hist(residual, bins=bins, color="#b85c36", alpha=0.75)
        axes[row, 1].set_yscale("log")
        axes[row, 1].axvline(0.0, color="0.2", linestyle="--", linewidth=0.9)
        axes[row, 1].set_title("Atom zoom: all values retained (log count scale)", fontsize=9)
        axes[row, 1].set_xlabel("theta - dominant value")
        axes[row, 1].set_ylabel("Count (log scale)")
        in_window = int(np.sum(np.abs(non_atom) <= 0.5))
        axes[row, 1].text(
            0.02,
            0.92,
            f"{atom_note}\nother values in zoom = {in_window}/{len(non_atom)}",
            transform=axes[row, 1].transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
        rows.append(
            {
                "parameter": name,
                "readable_parameter": label,
                "dominant_value": dominant_value,
                "dominant_count": dominant_count,
                "dominant_fraction": dominant_count / len(values),
                "non_atom_count": int(len(non_atom)),
                "unique_value_count": int(len(unique)),
                "non_atom_residual_min": float(np.min(non_atom)),
                "non_atom_residual_max": float(np.max(non_atom)),
            }
        )
    fig.suptitle(
        "Dirac-like quenching parameters: global distribution and +/-0.5 atom zoom",
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def _normal_density_grid() -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(-5.0, 5.0, 500)
    density = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
    return grid, density


def _write_asinh_comparison_plot(
    *,
    names: list[str],
    selected_names: tuple[str, ...],
    theta: np.ndarray,
    current_x: np.ndarray,
    asinh_x: np.ndarray,
    summary: pd.DataFrame,
    title: str,
    path: Path,
) -> None:
    indices = [names.index(name) for name in selected_names if name in names]
    summary_by_name = summary.set_index("parameter")
    normal_x, normal_density = _normal_density_grid()
    fig, axes = plt.subplots(
        len(indices),
        3,
        figsize=(17, 2.6 * len(indices)),
        constrained_layout=True,
    )
    for row, index in enumerate(indices):
        name = names[index]
        label, role, _ = _parameter_label(name)
        result = summary_by_name.loc[name]
        _hist(axes[row, 0], theta[:, index], color="#3979a6", xlabel=label)
        axes[row, 0].set_title(f"{role}: {label}", loc="left", fontsize=9)

        _hist(axes[row, 1], current_x[:, index], color="#b85c36", xlabel="Current standardized logit x")
        axes[row, 1].plot(normal_x, normal_density, color="0.15", linewidth=1.0)
        axes[row, 1].set_xlim(-5.0, 5.0)
        axes[row, 1].set_title(f"Current Gaussian RMSE = {result['current_gaussian_rmse']:.3f}", fontsize=9)

        _hist(axes[row, 2], asinh_x[:, index], color="#5d8f54", xlabel="Per-parameter asinh + standardization")
        axes[row, 2].plot(normal_x, normal_density, color="0.15", linewidth=1.0)
        axes[row, 2].set_xlim(-5.0, 5.0)
        axes[row, 2].set_title(
            f"lambda = {result['lambda']:.3g} ({result['asinh_regime']}); Gaussian RMSE = {result['asinh_gaussian_rmse']:.3f}",
            fontsize=9,
        )
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_asinh_report(path: Path, summary: pd.DataFrame, dataset: Path) -> None:
    lines = [
        "# Per-parameter asinh normalization study",
        "",
        f"Dataset: `{dataset}`",
        "",
        "Candidate transform for each physical parameter:",
        "",
        "`y = lambda * asinh(theta / lambda)`",
        "",
        "`x_asinh = (y - mean_train(y)) / std_train(y)`",
        "",
        "Lambda is selected independently for each parameter by minimizing the RMSE between transformed empirical quantiles and standard-normal quantiles.",
        "The current RMSE uses the exact checkpoint logit, center, and scale applied to the plotted 18-band dataset; the linear and asinh transforms are fitted on this dataset.",
        "",
        "| Parameter | Role | lambda | Regime | Current RMSE | Linear RMSE | Asinh RMSE | Dominant exact fraction | Assessment |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.to_dict(orient="records"):
        if row["dominant_fraction"] >= 0.5:
            assessment = "Discrete-continuous mixture: a monotone transform cannot remove the atom."
        elif row["asinh_regime"] == "linear limit":
            assessment = "No compression benefit: ordinary linear standardization is preferred."
        elif row["asinh_gaussian_rmse"] < 0.9 * row["linear_gaussian_rmse"]:
            assessment = "Asinh improves the marginal relative to linear standardization."
        elif row["current_gaussian_rmse"] < row["asinh_gaussian_rmse"]:
            assessment = "The current bounded-logit marginal is closer to Gaussian."
        else:
            assessment = "No material asinh improvement."
        lines.append(
            "| `{parameter}` | {role} | {lambda:.5g} | {asinh_regime} | "
            "{current_gaussian_rmse:.3f} | {linear_gaussian_rmse:.3f} | "
            "{asinh_gaussian_rmse:.3f} | {dominant_fraction:.3%} | {assessment} |".format(
                assessment=assessment,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Limitation",
            "",
            "An invertible monotone transform preserves repeated values. The four quenching coordinates with a 96.493% exact atom require a mixed discrete/continuous model, parameter removal, or explicit conditioning on quenched state; lambda tuning alone cannot make them Gaussian.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)
    sidecar = _read_json(args.run / "checkpoints" / "best.eqx.json")
    spec = sidecar["latent_spec"]
    names = list(spec["names"])
    lower = np.asarray(spec["lower"], dtype=float)
    upper = np.asarray(spec["upper"], dtype=float)
    center = np.asarray(spec["raw_center"], dtype=float)
    scale = np.asarray(spec["raw_scale"], dtype=float)
    if spec.get("normalization") != "truth_standardized_logit":
        raise ValueError("This script expects a truth_standardized_logit checkpoint.")

    schema = sidecar["schema"]["parameters"]
    source_by_name = {entry["name"]: entry["column"] for entry in schema}
    source_columns = [source_by_name[name] for name in names]
    frame = pd.read_parquet(args.dataset, columns=source_columns)
    theta = frame[source_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(theta).all(axis=1)
    theta = theta[finite]
    if theta.size == 0:
        raise ValueError(f"No finite rows in {args.dataset}")

    unit = (theta - lower) / (upper - lower)
    raw = np.log(np.clip(unit, 1.0e-6, 1.0 - 1.0e-6)) - np.log1p(-np.clip(unit, 1.0e-6, 1.0 - 1.0e-6))
    x = (raw - center) / scale
    raw_scale_unclipped = np.std(raw, axis=0)
    asinh_x = np.empty_like(theta)
    asinh_rows: list[dict[str, object]] = []
    asinh_scan_rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        values = theta[:, index]
        normalized, metadata, scan_rows = _fit_asinh_transform(values)
        asinh_x[:, index] = normalized
        linear_x, _, _ = _standardize(values)
        unique, counts = np.unique(values, return_counts=True)
        dominant_index = int(np.argmax(counts))
        label, role, _ = _parameter_label(name)
        asinh_rows.append(
            {
                "parameter": name,
                "readable_parameter": label,
                "role": role,
                "lambda": metadata["lambda"],
                "asinh_center": metadata["center"],
                "asinh_scale": metadata["scale"],
                "current_gaussian_rmse": _gaussian_quantile_rmse(x[:, index]),
                "linear_gaussian_rmse": _gaussian_quantile_rmse(linear_x),
                "asinh_gaussian_rmse": metadata["gaussian_quantile_rmse"],
                "dominant_value": float(unique[dominant_index]),
                "dominant_fraction": float(counts[dominant_index] / len(values)),
                "unique_value_count": int(len(unique)),
                "lambda_at_scan_min": metadata["lambda_at_scan_min"],
                "lambda_at_scan_max": metadata["lambda_at_scan_max"],
                "asinh_regime": (
                    "log-like limit"
                    if metadata["lambda_at_scan_min"]
                    else "linear limit"
                    if metadata["lambda_at_scan_max"]
                    else "finite compression"
                ),
            }
        )
        for scan_row in scan_rows:
            asinh_scan_rows.append({"parameter": name, **scan_row})
    asinh_summary = pd.DataFrame(asinh_rows)
    asinh_summary.to_csv(out / "realnvp_asinh_transform_summary.csv", index=False)
    pd.DataFrame(asinh_scan_rows).to_csv(out / "realnvp_asinh_lambda_scan.csv", index=False)

    rows: list[dict[str, object]] = []
    for index, name in enumerate(names):
        label, group, description = _parameter_label(name)
        rows.append(
            {
                "group": group,
                "label": label,
                "description": description,
                "name": name,
                "source_column": source_by_name[name],
                "lower": lower[index],
                "upper": upper[index],
                "raw_center": center[index],
                "raw_scale": scale[index],
                "raw_scale_unclipped": raw_scale_unclipped[index],
                "theta_mean": np.mean(theta[:, index]),
                "theta_std": np.std(theta[:, index]),
                "x_mean": np.mean(x[:, index]),
                "x_std": np.std(x[:, index]),
                "x_abs_q99": np.quantile(np.abs(x[:, index]), 0.99),
                "x_abs_max": np.max(np.abs(x[:, index])),
                "clipped_low_fraction": np.mean(unit[:, index] <= 1.0e-6),
                "clipped_high_fraction": np.mean(unit[:, index] >= 1.0 - 1.0e-6),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "realnvp_normalization_statistics.csv", index=False)
    checkpoint_dataset = str(sidecar.get("source_dataset", sidecar["prior_learning"]["dataset"]))
    scale_floor = float(sidecar["prior_learning"]["latent"]["min_raw_scale"])
    _write_glossary(
        out / "realnvp_parameter_glossary.md",
        rows,
        dataset=args.dataset,
        checkpoint_dataset=checkpoint_dataset,
        scale_floor=scale_floor,
    )
    (out / "realnvp_normalization_metadata.json").write_text(
        json.dumps(
            {
                "dataset_plotted": str(args.dataset),
                "checkpoint_training_dataset": checkpoint_dataset,
                "normalization": spec["normalization"],
                "formula": "x = (logit(clip((theta - lower)/(upper - lower), 1e-6, 1 - 1e-6)) - raw_center) / raw_scale",
                "n_finite_rows": int(len(theta)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_before_after_plot(
        names=names,
        theta=theta,
        x=x,
        lower=lower,
        upper=upper,
        selected_names=tuple(names),
        title="FENIKS 18D supervised prior: before and after current normalization",
        path=out / "realnvp_1d_before_after_normalization.png",
    )
    _write_before_after_plot(
        names=names,
        theta=theta,
        x=x,
        lower=lower,
        upper=upper,
        selected_names=USEFUL_PARAMETER_ORDER,
        title="FENIKS useful parameters: before and after current normalization",
        path=out / "realnvp_useful_1d_before_after_normalization.png",
    )
    dirac_stats = _write_dirac_zoom(
        names=names,
        theta=theta,
        path=out / "realnvp_dirac_zoom.png",
    )
    dirac_stats.to_csv(out / "realnvp_dirac_atom_statistics.csv", index=False)
    _write_asinh_comparison_plot(
        names=names,
        selected_names=tuple(names),
        theta=theta,
        current_x=x,
        asinh_x=asinh_x,
        summary=asinh_summary,
        title="FENIKS 18D: current normalization vs per-parameter asinh compression",
        path=out / "realnvp_asinh_comparison_all18.png",
    )
    _write_asinh_comparison_plot(
        names=names,
        selected_names=USEFUL_PARAMETER_ORDER,
        theta=theta,
        current_x=x,
        asinh_x=asinh_x,
        summary=asinh_summary,
        title="FENIKS useful parameters: current normalization vs per-parameter asinh",
        path=out / "realnvp_asinh_comparison_useful.png",
    )
    _write_asinh_report(
        out / "realnvp_asinh_transform_report.md",
        asinh_summary,
        args.dataset,
    )

    fig, axes = plt.subplots(6, 3, figsize=(16, 19), constrained_layout=True)
    for axis, index in zip(axes.ravel(), range(len(names)), strict=True):
        label, group, _ = _parameter_label(names[index])
        _hist(axis, raw[:, index], color="#6a9b4f", xlabel="Raw bounded logit")
        axis.set_title(f"{group}\n{label}", fontsize=9)
        axis.axvline(center[index], color="0.25", linestyle="--", linewidth=0.8)
    fig.suptitle("Intermediate step: raw logit before centering/scaling (dashed = training mean)", fontsize=14)
    fig.savefig(out / "realnvp_1d_raw_logit.png", dpi=180)
    plt.close(fig)

    print(f"Wrote {out / 'realnvp_1d_before_after_normalization.png'}")
    print(f"Wrote {out / 'realnvp_useful_1d_before_after_normalization.png'}")
    print(f"Wrote {out / 'realnvp_1d_raw_logit.png'}")
    print(f"Wrote {out / 'realnvp_dirac_zoom.png'}")
    print(f"Wrote {out / 'realnvp_dirac_atom_statistics.csv'}")
    print(f"Wrote {out / 'realnvp_asinh_comparison_all18.png'}")
    print(f"Wrote {out / 'realnvp_asinh_comparison_useful.png'}")
    print(f"Wrote {out / 'realnvp_asinh_transform_summary.csv'}")
    print(f"Wrote {out / 'realnvp_asinh_transform_report.md'}")
    print(f"Wrote {out / 'realnvp_normalization_statistics.csv'}")
    print(f"Wrote {out / 'realnvp_parameter_glossary.md'}")
    print(f"Wrote {out / 'realnvp_normalization_metadata.json'}")


if __name__ == "__main__":
    main()
