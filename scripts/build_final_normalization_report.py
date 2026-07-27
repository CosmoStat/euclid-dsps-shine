#!/usr/bin/env python3
"""Build final hybrid and Dirac-preserving normalization diagnostics."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_invertible_prior_normalizations import (
    _forward,
    _gaussian_score,
    _inverse,
    _normal_curve,
)
from plot_realnvp_normalization_diagnostics import PARAMETERS
from scipy.optimize import minimize_scalar

ATOM_NAMES = (
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_atom_centered_asinh(values: np.ndarray, atom_value: float) -> dict[str, Any]:
    continuous = values[values != atom_value]
    residual = continuous - atom_value
    data_scale = max(float(np.std(residual)), 1.0e-12)

    def objective(log10_relative_lambda: float) -> float:
        value = data_scale * 10.0**log10_relative_lambda
        transformed = value * np.arcsinh(residual / value)
        normalized = (transformed - np.mean(transformed)) / np.std(transformed)
        return _gaussian_score(normalized)

    result = minimize_scalar(
        objective,
        bounds=(-4.0, 4.0),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    value = data_scale * 10.0 ** float(result.x)
    transformed = value * np.arcsinh(residual / value)
    output_scale = float(np.sqrt(np.mean(transformed**2)))
    conditional_center = float(np.mean(transformed) / output_scale)
    conditional_std = float(np.std(transformed) / output_scale)
    conditional_score = _gaussian_score(
        (transformed / output_scale - conditional_center) / conditional_std
    )
    return {
        "family": "atom_centered_asinh",
        "atom_value": float(atom_value),
        "lambda": value,
        "output_scale": output_scale,
        "conditional_normalized_center": conditional_center,
        "conditional_normalized_std": conditional_std,
        "conditional_gaussian_score": conditional_score,
        "computation": (
            "lambda minimizes conditional non-atom quantile RMSE to N(0,1); "
            "output_scale is the train RMS around the atom after asinh"
        ),
    }


def _forward_v2(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    if spec["family"] != "atom_centered_asinh":
        return _forward(values, spec)
    transformed = spec["lambda"] * np.arcsinh(
        (values - spec["atom_value"]) / spec["lambda"]
    )
    return transformed / spec["output_scale"]


def _inverse_v2(values: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    if spec["family"] != "atom_centered_asinh":
        return _inverse(values, spec)
    return spec["atom_value"] + spec["lambda"] * np.sinh(
        spec["output_scale"] * values / spec["lambda"]
    )


def _split_values(frame: pd.DataFrame, schema: dict[str, str]) -> dict[str, np.ndarray]:
    return {
        name: frame[column].to_numpy(dtype=float) for name, column in schema.items()
    }


def _plot_before_after(
    *,
    names: list[str],
    train: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    specs: dict[str, dict[str, Any]],
    version: str,
    title: str,
    path: Path,
) -> None:
    normal_x, normal_density = _normal_curve()
    fig, axes = plt.subplots(
        len(names),
        2,
        figsize=(14, 2.7 * len(names)),
        constrained_layout=True,
    )
    for row, name in enumerate(names):
        values = train[name]
        test_values = evaluation[name]
        label, role, _ = PARAMETERS[name]
        spec = specs[name]
        axes[row, 0].hist(values, bins=60, density=True, color="#3979a6", alpha=0.75)
        axes[row, 0].set_title(f"{role}: {label}", loc="left", fontsize=9)
        axes[row, 0].set_xlabel("Before: physical theta")
        axes[row, 0].set_ylabel("Density")

        if version == "hybrid" and spec["family"] == "mixed_atom_continuous":
            atom_value = spec["atom_value"]
            continuous_spec = spec["continuous_transform"]
            train_continuous = values[values != atom_value]
            test_continuous = test_values[test_values != atom_value]
            normalized = _forward(train_continuous, continuous_spec)
            test_normalized = _forward(test_continuous, continuous_spec)
            axes[row, 1].hist(
                normalized,
                bins=60,
                density=True,
                color="#5d8f54",
                alpha=0.75,
            )
            axes[row, 1].plot(normal_x, normal_density, color="0.15", linewidth=1.0)
            axes[row, 1].set_xlim(-5.0, 5.0)
            axes[row, 1].set_title(
                f"After: continuous branch; test RMSE={_gaussian_score(test_normalized):.3f}",
                fontsize=9,
            )
            axes[row, 1].text(
                0.02,
                0.92,
                f"Shared discrete atom retained exactly\ntheta={atom_value:.12g}",
                transform=axes[row, 1].transAxes,
                va="top",
                fontsize=8,
            )
        else:
            normalized = (
                _forward_v2(values, spec)
                if version == "dirac_preserved"
                else _forward(values, spec)
            )
            test_normalized = (
                _forward_v2(test_values, spec)
                if version == "dirac_preserved"
                else _forward(test_values, spec)
            )
            if spec["family"] == "atom_centered_asinh":
                axes[row, 1].hist(
                    normalized,
                    bins=80,
                    color="#b85c36",
                    alpha=0.75,
                )
                axes[row, 1].set_yscale("log")
                axes[row, 1].set_title(
                    f"After: atom retained exactly at x=0; test rows={len(test_values)}",
                    fontsize=9,
                )
                axes[row, 1].set_ylabel("Count (log scale)")
                axes[row, 1].text(
                    0.02,
                    0.92,
                    f"theta atom={spec['atom_value']:.12g}\nx atom=0 exactly",
                    transform=axes[row, 1].transAxes,
                    va="top",
                    fontsize=8,
                )
            else:
                axes[row, 1].hist(
                    normalized,
                    bins=60,
                    density=True,
                    color="#5d8f54",
                    alpha=0.75,
                )
                axes[row, 1].plot(normal_x, normal_density, color="0.15", linewidth=1.0)
                axes[row, 1].set_xlim(-5.0, 5.0)
                axes[row, 1].set_title(
                    f"After: {spec['family']}; test RMSE={_gaussian_score(test_normalized):.3f}",
                    fontsize=9,
                )
                axes[row, 1].set_ylabel("Density")
        axes[row, 1].set_xlabel("After: normalized x")
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _formula(spec: dict[str, Any], version: str) -> tuple[str, str, str]:
    family = spec["family"]
    if family == "mixed_atom_continuous":
        child_formula, child_inverse, child_compute = _formula(
            spec["continuous_transform"],
            version,
        )
        return (
            f"b=1[theta==a]; if b=1 store atom a, else {child_formula}",
            f"if b=1 theta=a, else {child_inverse}",
            "The shared atom value and frequency are counted exactly on train. "
            + child_compute,
        )
    if family == "atom_centered_asinh":
        return (
            "x = lambda*asinh((theta-a)/lambda) / output_scale",
            "theta = a + lambda*sinh(output_scale*x/lambda)",
            "a is the exact train atom; lambda minimizes Gaussian quantile RMSE on non-atom rows; output_scale is their transformed train RMS around the atom, keeping the atom at x=0 and the continuous branch at order-unity amplitude.",
        )
    if family == "affine":
        return (
            "x = (theta-center)/scale",
            "theta = center + scale*x",
            "center and scale are the train mean and standard deviation.",
        )
    if family == "wide_bound_logit":
        return (
            "u=(theta-lower)/(upper-lower); x=(logit(u)-center)/scale",
            "theta=lower+(upper-lower)*sigmoid(center+scale*x)",
            "lower/upper are widened physical bounds; center/scale are train raw-logit moments. No clipping is applied.",
        )
    if family == "asinh":
        return (
            "y=lambda*asinh(theta/lambda); x=(y-center)/scale",
            "theta=lambda*sinh((center+scale*x)/lambda)",
            "lambda minimizes train Gaussian quantile RMSE; center/scale are transformed train moments.",
        )
    if family == "shifted_asinh":
        return (
            "y=lambda*asinh((theta-shift)/lambda); x=(y-center)/scale",
            "theta=shift+lambda*sinh((center+scale*x)/lambda)",
            "shift and lambda minimize train Gaussian quantile RMSE; center/scale are transformed train moments.",
        )
    if family == "quantile_spline":
        return (
            "x = monotone_spline(theta_knots, normal_quantile_knots; theta)",
            "theta = monotone_spline(normal_quantile_knots, theta_knots; x)",
            "Train quantiles define 257 monotone knots; endpoints are train min/max and tails use positive linear extrapolation.",
        )
    raise ValueError(f"Unsupported family: {family}")


def _parameter_values(spec: dict[str, Any]) -> dict[str, Any]:
    if spec["family"] == "mixed_atom_continuous":
        return {
            "atom_value": spec["atom_value"],
            "atom_fraction_train": spec["atom_fraction_train"],
            "continuous_transform": _parameter_values(spec["continuous_transform"]),
        }
    return {
        key: value
        for key, value in spec.items()
        if key not in {"family", "theta_knots", "normal_knots", "computation"}
    }


def _details_html(spec: dict[str, Any]) -> str:
    values = _parameter_values(spec)
    content = html.escape(json.dumps(values, indent=2))
    knots = ""
    target = spec.get("continuous_transform", spec)
    if target["family"] == "quantile_spline":
        knot_payload = {
            "theta_knots": target["theta_knots"],
            "normal_knots": target["normal_knots"],
        }
        knots = (
            "<details><summary>Full spline knots</summary><pre>"
            + html.escape(json.dumps(knot_payload, indent=2))
            + "</pre></details>"
        )
    return f"<pre>{content}</pre>{knots}"


def _build_html(
    *,
    out: Path,
    names: list[str],
    hybrid_specs: dict[str, dict[str, Any]],
    v2_specs: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    datasets: dict[str, Path],
) -> None:
    metric_index = metrics.set_index(["parameter", "version"])
    parameter_sections = []
    for name in names:
        label, role, description = PARAMETERS[name]
        versions = []
        for version, title, specs in (
            ("hybrid", "Version A: hybrid atom + continuous", hybrid_specs),
            ("dirac_preserved", "Version B: one 18D vector, Dirac retained", v2_specs),
        ):
            spec = specs[name]
            formula, inverse, computation = _formula(spec, version)
            row = metric_index.loc[(name, version)]
            versions.append(f"""
                <div class="version">
                  <h4>{html.escape(title)}</h4>
                  <dl>
                    <dt>Family</dt><dd><code>{html.escape(spec['family'])}</code></dd>
                    <dt>Forward</dt><dd><code>{html.escape(formula)}</code></dd>
                    <dt>Inverse</dt><dd><code>{html.escape(inverse)}</code></dd>
                    <dt>Computed from train</dt><dd>{html.escape(computation)}</dd>
                    <dt>Test diagnostic</dt><dd>Gaussian RMSE {row['test_gaussian_rmse']:.4f}; max |x| {row['test_x_abs_max']:.4f}; round-trip {row['test_roundtrip_max_abs']:.3g}</dd>
                  </dl>
                  {_details_html(spec)}
                </div>
                """)
        parameter_sections.append(f"""
            <section class="parameter" id="{html.escape(name)}">
              <h3>{html.escape(label)} <code>{html.escape(name)}</code></h3>
              <p class="role">{html.escape(role)}. {html.escape(description)}</p>
              {''.join(versions)}
            </section>
            """)

    rows = []
    for name in names:
        hybrid = metric_index.loc[(name, "hybrid")]
        simple = metric_index.loc[(name, "dirac_preserved")]
        rows.append(
            f"<tr><td><code>{html.escape(name)}</code></td>"
            f"<td>{html.escape(hybrid['family'])}</td><td>{hybrid['test_gaussian_rmse']:.3f}</td>"
            f"<td>{html.escape(simple['family'])}</td><td>{simple['test_gaussian_rmse']:.3f}</td></tr>"
        )
    dataset_rows = "".join(
        f"<li><strong>{html.escape(split)}</strong>: <code>{html.escape(str(path))}</code></li>"
        for split, path in datasets.items()
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FENIKS 18D normalization report</title>
<style>
body {{ margin: 0; font: 15px/1.5 system-ui, sans-serif; color: #17212b; background: #fff; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 80px; }}
h1, h2, h3, h4 {{ line-height: 1.2; letter-spacing: 0; }}
h1 {{ font-size: 30px; margin: 0 0 10px; }}
h2 {{ margin-top: 42px; padding-bottom: 8px; border-bottom: 2px solid #d5dde5; }}
h3 {{ margin-bottom: 5px; }} h4 {{ margin: 22px 0 8px; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
code {{ background: #eef2f5; padding: 1px 4px; }}
pre {{ overflow-x: auto; background: #f5f7f9; border-left: 3px solid #5d8f54; padding: 12px; font-size: 12px; }}
.lead {{ font-size: 17px; max-width: 900px; }}
.warning {{ border-left: 4px solid #b85c36; padding: 10px 14px; background: #fff7f2; }}
.recommend {{ border-left: 4px solid #3979a6; padding: 10px 14px; background: #f2f7fb; }}
img {{ width: 100%; height: auto; display: block; margin: 18px 0; border: 1px solid #d5dde5; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
th, td {{ text-align: left; border-bottom: 1px solid #d5dde5; padding: 7px 8px; vertical-align: top; }}
th {{ background: #f5f7f9; }}
.parameter {{ padding: 18px 0 28px; border-bottom: 1px solid #d5dde5; }}
.role {{ color: #4b5965; margin-top: 0; }}
dl {{ display: grid; grid-template-columns: 170px 1fr; gap: 5px 12px; }}
dt {{ font-weight: 650; }} dd {{ margin: 0; }}
details {{ margin: 10px 0; }} summary {{ cursor: pointer; font-weight: 650; }}
@media (max-width: 720px) {{ main {{ padding: 22px 14px 60px; }} dl {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body><main>
<h1>FENIKS 18D invertible normalization</h1>
<p class="lead">Final train/validation/test analysis of two normalization designs. All numerical transform values come from train only. The test split exposed one fragile stellar-mass family and therefore became a robustness audit rather than a fully untouched holdout; a new external holdout is required for an unbiased final score.</p>
<ul>{dataset_rows}</ul>

<h2>Recommendation</h2>
<p class="recommend"><strong>Version A, the hybrid model, is the statistically correct design.</strong> One shared binary state represents the four exact Diffstar atoms. The continuous branch is normalized and modeled conditionally. Samples from the atom branch reproduce the four values exactly.</p>
<p class="warning"><strong>Version B is operationally simpler but approximate.</strong> Atom-centered asinh maps each exact atom to <code>x=0</code> and remains invertible. A single continuous flow still cannot assign positive probability to that exact point; it learns a narrow peak and will not reproduce exact atoms without a non-invertible snapping rule.</p>
<p>Hard clipping is excluded because it destroys invertibility. After the selected smooth transforms, no test coordinate requires clipping.</p>

<h2>Version A: hybrid</h2>
<p><code>b ~ Bernoulli(pi)</code>. If <code>b=1</code>, the four quenching parameters equal their exact atom values. If <code>b=0</code>, their conditional continuous values use invertible shifted-asinh transforms. The same indicator is shared because all four atom masks are identical.</p>
<img src="normalization_v1_hybrid_before_after.png" alt="Hybrid normalization before and after">

<h2>Version B: Dirac retained in one vector</h2>
<p>For each atomic coordinate, <code>x=lambda*asinh((theta-a)/lambda)/output_scale</code>. This sends <code>theta=a</code> exactly to <code>x=0</code>, retains the atom in the normalized training data, and has an analytic inverse. <code>output_scale</code> is the RMS of the transformed non-atom residuals, so the minority branch remains at order-unity amplitude.</p>
<img src="normalization_v2_dirac_preserved_before_after.png" alt="Dirac-preserving normalization before and after">

<h2>Final test comparison</h2>
<p>For atomic coordinates, the hybrid RMSE measures only the conditional continuous branch. The Dirac-preserved RMSE measures the complete atom-plus-continuous marginal and is therefore not expected to approach zero.</p>
<table><thead><tr><th>Parameter</th><th>Hybrid family</th><th>Hybrid test RMSE</th><th>Dirac-preserved family</th><th>Dirac-preserved test RMSE</th></tr></thead><tbody>{''.join(rows)}</tbody></table>

<h2>Per-parameter formulas and fitted values</h2>
{''.join(parameter_sections)}
</main></body></html>
"""
    (out / "normalization_final_report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)
    sidecar = _read_json(args.run / "checkpoints" / "best.eqx.json")
    names = list(sidecar["latent_spec"]["names"])
    schema = {
        entry["name"]: entry["column"] for entry in sidecar["schema"]["parameters"]
    }
    columns = [schema[name] for name in names]
    train_frame = pd.read_parquet(args.train, columns=columns)
    validation_frame = pd.read_parquet(args.validation, columns=columns)
    test_frame = pd.read_parquet(args.test, columns=columns)
    train = _split_values(train_frame, schema)
    validation = _split_values(validation_frame, schema)
    test = _split_values(test_frame, schema)
    hybrid_specs = _read_json(args.run / "invertible_normalization_specs.json")[
        "transforms"
    ]
    v2_specs = json.loads(json.dumps(hybrid_specs))
    for name in ATOM_NAMES:
        hybrid = hybrid_specs[name]
        v2_specs[name] = _fit_atom_centered_asinh(train[name], hybrid["atom_value"])

    metric_rows = []
    for name in names:
        for version, specs in (("hybrid", hybrid_specs), ("dirac_preserved", v2_specs)):
            spec = specs[name]
            train_values = train[name]
            validation_values = validation[name]
            test_values = test[name]
            if version == "hybrid" and spec["family"] == "mixed_atom_continuous":
                atom_value = spec["atom_value"]
                transform = spec["continuous_transform"]
                train_values = train_values[train_values != atom_value]
                validation_values = validation_values[validation_values != atom_value]
                test_values = test_values[test_values != atom_value]
                train_x = _forward(train_values, transform)
                validation_x = _forward(validation_values, transform)
                test_x = _forward(test_values, transform)
                reconstructed = _inverse(test_x, transform)
                family = "atom + " + transform["family"]
            else:
                train_x = _forward_v2(train_values, spec)
                validation_x = _forward_v2(validation_values, spec)
                test_x = _forward_v2(test_values, spec)
                reconstructed = _inverse_v2(test_x, spec)
                family = spec["family"]
            metric_rows.append(
                {
                    "parameter": name,
                    "version": version,
                    "family": family,
                    "train_gaussian_rmse": _gaussian_score(train_x),
                    "validation_gaussian_rmse": _gaussian_score(validation_x),
                    "test_gaussian_rmse": _gaussian_score(test_x),
                    "test_x_abs_q999": float(np.quantile(np.abs(test_x), 0.999)),
                    "test_x_abs_max": float(np.max(np.abs(test_x))),
                    "test_roundtrip_max_abs": float(
                        np.max(np.abs(reconstructed - test_values))
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "normalization_final_test_metrics.csv", index=False)
    (out / "normalization_v2_dirac_preserved_specs.json").write_text(
        json.dumps({"transforms": v2_specs}, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_before_after(
        names=names,
        train=train,
        evaluation=test,
        specs=hybrid_specs,
        version="hybrid",
        title="Version A: hybrid atom + continuous normalization (before / after)",
        path=out / "normalization_v1_hybrid_before_after.png",
    )
    _plot_before_after(
        names=names,
        train=train,
        evaluation=test,
        specs=v2_specs,
        version="dirac_preserved",
        title="Version B: one 18D vector with exact Dirac retained (before / after)",
        path=out / "normalization_v2_dirac_preserved_before_after.png",
    )
    _build_html(
        out=out,
        names=names,
        hybrid_specs=hybrid_specs,
        v2_specs=v2_specs,
        metrics=metrics,
        datasets={
            "train": args.train,
            "validation": args.validation,
            "test": args.test,
        },
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
