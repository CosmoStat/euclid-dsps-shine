"""Read-only replay of native and spline fluxes and catalogue weighting.

This diagnoses a historical synthetic catalogue. It does not regenerate data,
fit a prior or authorize production. Input truth is used only for this audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.config import load_config


def replay_configs(generator: dict, inference: dict) -> dict[str, dict]:
    """Hold assets fixed; isolate the spline projection and numerical changes."""
    for key in ("ssp_path", "bands", "cosmos_sed", "nebular_emission"):
        if generator.get(key) != inference.get(key):
            raise ValueError(f"Unmatched {key}: do not conflate asset and SFH changes")
    native = copy.deepcopy(generator)
    legacy = copy.deepcopy(inference)
    for config in (native, legacy):
        config["model"].update(
            photometry_integrator="legacy_trapezoid_v1",
            mdf_weight_precision="float32_legacy",
            spline_precision="float32_legacy",
        )
    if native["model"]["sfh_model"] != "diffsky_basic":
        raise ValueError("Expected native diffsky_basic generation")
    if legacy["model"]["sfh_model"] != "spline15d":
        raise ValueError("Expected spline15d inference")
    return {
        "native_legacy": native,
        "spline_legacy": legacy,
        "spline_current": copy.deepcopy(inference),
    }


def weighting_summary(catalogue: pd.DataFrame, truth: pd.DataFrame) -> dict:
    for frame in (catalogue, truth):
        if frame.object_id.duplicated().any():
            raise ValueError("Nonunique object_id")
    paired = truth.merge(
        catalogue[["object_id", "galaxy_weight"]],
        on="object_id",
        validate="one_to_one",
        how="left",
    )
    if paired.galaxy_weight.isna().any():
        raise ValueError("Truth identities missing from source catalogue")
    result = {}
    for label, frame, z_column in (
        ("catalogue", catalogue, "redshift_true"),
        ("truth_cohort", paired, "z_obs"),
    ):
        weights = frame.galaxy_weight.to_numpy(float)
        z = frame[z_column].to_numpy(float)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Invalid proposal weights")
        if not np.isfinite(z).all():
            raise ValueError("Invalid truth redshifts")
        result[label] = {
            "rows": len(frame),
            "mean_z_unweighted": float(z.mean()),
            "mean_z_proposal_weighted_again": float(np.average(z, weights=weights)),
        }
    result["truth_weights_equal_retained_proposal_weights"] = bool(
        np.array_equal(paired.population_weight, paired.galaxy_weight)
    )
    return result


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
    from euclid_dsps.filters import load_filters
    from euclid_dsps.model import dynamic_model_args, load_context
    from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
    from euclid_dsps.parameters import DIFFSKY_BASIC_PARAMETER_NAMES
    from euclid_dsps.photometry import abmag_to_fnu_cgs
    from euclid_dsps.synthetic_diffsky.photometry import theta_from_truth_frame

    if args.limit < 1 or args.batch_size < 1:
        raise ValueError("Positive limit and batch size required")
    # A new output root keeps cluster receipts and historical catalogues intact.
    args.out.mkdir(parents=True, exist_ok=False)
    catalogue = pd.read_parquet(args.catalogue)
    truth = pd.read_parquet(args.truth)
    summary = weighting_summary(catalogue, truth)
    manifest = yaml.safe_load(args.generator_manifest.read_text())
    summary["generator_commit"] = manifest["repo_git_sha"]
    summary["generator_selection"] = manifest["synthetic_diffsky"]["selection"]
    summary["generator_output_layers"] = manifest["synthetic_diffsky"]["output_layers"]
    summary["input_sha256"] = {
        str(path): _sha(path)
        for path in (
            args.catalogue,
            args.truth,
            args.rows_csv,
            args.generator_config,
            args.inference_config,
            args.generator_manifest,
        )
    }
    summary["audit_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    summary["audit_script_sha256"] = _sha(Path(__file__))
    summary["package_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("jax", "dsps", "diffstar", "diffmah")
    }
    _write(args.out / "provenance.json", summary)
    rows = pd.read_csv(args.rows_csv)
    indices = rows.loc[rows.variant == "baseline", "row"].unique()[: args.limit]
    frame = catalogue.iloc[indices]
    aligned = truth.set_index("object_id").loc[frame.object_id]
    bands = [b["name"] for b in load_config(args.inference_config)["bands"]]
    target = frame[[f"flux_true_{b}" for b in bands]].to_numpy(float)
    sigma = frame[[f"fluxerr_{b}" for b in bands]].to_numpy(float)
    if not len(frame) or not np.isfinite(target).all() or not np.all(sigma > 0):
        raise ValueError("Invalid replay rows or photometry")
    if not np.isfinite(sigma).all():
        raise ValueError("Nonfinite reported errors")
    configs = replay_configs(
        load_config(args.generator_config), load_config(args.inference_config)
    )
    _write(args.out / "effective_configs.json", configs)
    all_metrics = []
    for variant, config in configs.items():
        names = (
            DIFFSKY_BASIC_PARAMETER_NAMES
            if variant == "native_legacy"
            else FENIKS_SPLINE15D_PARAMETERS
        )
        theta = (
            theta_from_truth_frame(frame)
            if variant == "native_legacy"
            else aligned[list(names)].to_numpy(np.float32)
        )
        context = load_context(
            config["ssp_path"],
            load_filters(config["bands"]),
            n_sfh_bins=config["model"]["n_sfh_bins"],
            cosmos_config=config.get("cosmos_sed"),
            nebular_emission=config.get("nebular_emission", "ssp_flux"),
            model_config=config["model"],
        )
        model_args = dynamic_model_args(context)
        fluxes = []
        for start in range(0, len(theta), args.batch_size):
            chunk = jnp.asarray(theta[start : start + args.batch_size])
            mags = model_mags_from_theta_matrix_jax(context, model_args, chunk, names)
            # Match the generator's host float64 AB conversion, not float32.
            fluxes.append(np.asarray(abmag_to_fnu_cgs(np.asarray(mags, dtype=float))))
            progress = {
                "variant": variant,
                "rows": min(start + len(chunk), len(theta)),
                "total": len(theta),
            }
            _write(args.out / "PROGRESS.json", progress)
            print(json.dumps(progress), flush=True)
        predicted = np.concatenate(fluxes)
        if not np.isfinite(predicted).all():
            raise ValueError(f"Nonfinite replay: {variant}")
        residual = (predicted - target) / sigma
        pd.DataFrame(
            {
                "row": np.repeat(indices, len(bands)),
                "object_id": np.repeat(frame.object_id.to_numpy(), len(bands)),
                "band": np.tile(bands, len(frame)),
                "predicted_flux": predicted.ravel(),
                "saved_flux": target.ravel(),
                "sigma": sigma.ravel(),
                "residual_sigma": residual.ravel(),
            }
        ).to_csv(args.out / f"{variant}_residuals.csv", index=False)
        for b, band in enumerate(bands):
            all_metrics.append(
                {
                    "variant": variant,
                    "band": band,
                    "objects": len(frame),
                    "p95_abs_sigma": float(np.quantile(np.abs(residual[:, b]), 0.95)),
                    "max_abs_sigma": float(np.max(np.abs(residual[:, b]))),
                }
            )
        jax.clear_caches()
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(args.out / "replay_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    for variant, group in metrics.groupby("variant", sort=False):
        ax.plot(bands, group.p95_abs_sigma, marker="o", label=variant)
    ax.axhline(0.25, linestyle="--", color="black", label="Historical 0.25 screen")
    ax.set(
        yscale="log",
        ylabel="p95 absolute noiseless flux residual / reported sigma",
        title=f"Same {len(frame)} objects, same assets; no training",
    )
    ax.tick_params(axis="x", rotation=70)
    ax.legend()
    fig.savefig(args.out / "native_vs_spline_replay.png", dpi=160)
    plt.close(fig)
    _write(
        args.out / "FINAL.json",
        {
            "status": "CATALOGUE_PROVENANCE_AUDIT_COMPLETE",
            "objects": len(frame),
            "production_ready": False,
            "catalogue_modified": False,
            "truth_used_for_training": False,
            "historical_environment_recreated": False,
            "comparison": "current source with generator-compatible numerical settings",
        },
    )
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "catalogue",
        "truth",
        "rows-csv",
        "generator-config",
        "inference-config",
        "generator-manifest",
        "out",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
