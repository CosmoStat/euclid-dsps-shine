"""Prepare a versioned 15D parent catalogue from saved weighted Diffsky proposals."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from euclid_dsps.config import load_config
from euclid_dsps.synthetic_diffsky.coherent_parent import (
    catalogue_checks,
    observe,
    partition_source_shards,
    sample_parent,
)
from scripts.feniks_avi_experiments import read, runtime_asset_paths, sha, write

SPLITS = ("train", "validation", "test")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def finish(out: Path, files: list[Path], contract: str, **values) -> None:
    write(
        out / "FINAL.json",
        dict(
            status="COMPLETE",
            contract=contract,
            artifacts={os.path.relpath(p, out): sha(p) for p in files},
            **values,
        ),
    )


def complete(out: Path, contract: str) -> bool:
    if not (out / "FINAL.json").is_file():
        return False
    record = read(out / "FINAL.json")
    if record.get("status") != "COMPLETE" or record.get("contract") != contract:
        raise ValueError(f"Incompatible receipt: {out}")
    for name, digest in record["artifacts"].items():
        if not (out / name).is_file() or sha(out / name) != digest:
            raise ValueError(f"Corrupt completed artifact: {out / name}")
    return True


def settings(root: Path) -> tuple[dict, dict, str]:
    m = read(root / "MANIFEST.json")
    for name, digest in m["frozen_files"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Prepared configuration changed: {name}")
    return m, m["settings"], sha(root / "MANIFEST.json")


def prepare(source: Path, decoder_config: Path, root: Path, config: Path) -> None:
    if root.exists():
        raise FileExistsError(f"Use --resume for an existing root: {root}")
    cfg = yaml.safe_load(config.read_text())
    if cfg["selection"] != dict(band="lsst_r", max_mag_ab=29.0):
        raise ValueError("This dataset contract is observed r<29 only")
    if cfg["support"] != dict(require_metallicity_unclipped=True):
        raise ValueError("Declare changes to physical support in a new workflow")
    for key in (
        "rows_per_task",
        "checkpoint_rows",
        "decoder_batch_size",
        "projection_batch_size",
    ):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    original = yaml.safe_load((source / "manifest.yaml").read_text())
    incoming = load_config(decoder_config)
    decoder = {
        k: copy.deepcopy(incoming[k])
        for k in (
            "model",
            "bands",
            "ssp_path",
            "cosmos_sed",
            "nebular_emission",
            "calibration",
        )
        if k in incoming
    }
    for key, value in dict(
        sfh_model="spline15d",
        photometry_integrator="merged_gauss4_v1",
        mdf_weight_precision="float64_v1",
        spline_precision="float64_v1",
    ).items():
        if decoder["model"].get(key) != value:
            raise ValueError(f"Expected current frozen decoder {key}={value}")
    if any(
        v.get("enabled", False)
        for v in decoder.get("calibration", {}).values()
        if isinstance(v, dict)
    ):
        raise ValueError("Cannot discard an enabled calibration when rephotometering")
    if decoder["model"].get("agn_model", "none") != "none":
        raise ValueError("Expected the existing no-AGN 15D model")
    # Resolve physical assets, never inherit old catalogues/checkpoints/objectives.
    decoder["ssp_path"] = str(Path(decoder["ssp_path"]).resolve())
    for band in decoder["bands"]:
        if band.get("filter", {}).get("path"):
            band["filter"]["path"] = str(Path(band["filter"]["path"]).resolve())
    for key, value in list(decoder["model"].items()):
        if key.endswith("_path") and value:
            decoder["model"][key] = str(Path(value).resolve())
    assets = {str(p): sha(p) for p in runtime_asset_paths(decoder)}
    noise = original["synthetic_diffsky"]["flux_error_model"]
    if noise["type"] != "m5_depth":
        raise ValueError("Expected saved Gaussian m5 generation contract")
    sources, tasks = {}, []
    for split in SPLITS:
        n, expected = cfg["parent_rows"][split], cfg["expected_proposal_shards"][split]
        if not isinstance(n, int) or n < 1:
            raise ValueError("Positive parent split sizes required")
        paths = sorted((source / "proposals" / split).glob("shard_*.parquet"))
        expected_names = [f"shard_{i:05d}.parquet" for i in range(expected)]
        if [p.name for p in paths] != expected_names or original["splits"][split][
            "n_shards"
        ] != expected:
            raise ValueError(f"Raw shard inventory mismatch: {split}")
        sources[split] = dict(
            paths=[str(p) for p in paths],
            source_seed=original["splits"][split]["source_seed"],
        )
        for start in range(0, n, cfg["rows_per_task"]):
            tasks.append(
                dict(
                    split=split, start=start, stop=min(n, start + cfg["rows_per_task"])
                )
            )
    original_sources = sources
    sources, split_audit = partition_source_shards(
        sources, cfg["parent_rows"], cfg["seed"]
    )
    root.mkdir(parents=True)
    for name in (
        "sampling",
        "photometry",
        "dataset/parent",
        "dataset/selected_r29",
        "report",
        "logs",
    ):
        (root / name).mkdir(parents=True)
    write(root / "decoder.json", decoder)
    write(root / "noise.json", noise)
    (root / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write(
        root / "MANIFEST.json",
        dict(
            version=1,
            source=str(source),
            source_manifest_sha256=sha(source / "manifest.yaml"),
            source_decoder_config=str(decoder_config),
            source_decoder_sha256=sha(decoder_config),
            sources=sources,
            original_sources=original_sources,
            split_audit=split_audit,
            tasks=tasks,
            settings=cfg,
            assets=assets,
            frozen_files={
                name: sha(root / name)
                for name in ("decoder.json", "noise.json", "experiment.yaml")
            },
            purpose="coherent_projected_Feniks_parent_benchmark_not_training",
            parent_scope="photometrically unselected within saved unclipped SSP metallicity support",
            regenerate_diffsky=False,
            prior_fitted=False,
            posterior_fitted=False,
        ),
    )
    print(
        json.dumps(
            dict(
                parent_rows=cfg["parent_rows"],
                simulations=sum(cfg["parent_rows"].values()),
                tasks=len(tasks),
                resources=cfg["resources"],
                selection=cfg["selection"],
                no_photometric_preselection=True,
                proposal_weights_applied_once=True,
                split_audit=split_audit,
            ),
            indent=2,
        )
    )


def sample(root: Path, task: int) -> None:
    m, cfg, contract = settings(root)
    split = SPLITS[task]
    out = root / "sampling" / split
    out.mkdir(exist_ok=True)
    if complete(out, contract):
        return

    def progress(value):
        write(out / "PROGRESS.json", value)
        print(json.dumps(dict(split=split, **value)), flush=True)

    source = m["sources"][split]
    frame, summary = sample_parent(
        [Path(p) for p in source["paths"]],
        source["source_split"],
        source["source_seed"],
        cfg["parent_rows"][split],
        cfg["seed"] + task,
        sha=sha,
        progress=progress,
    )
    frame.insert(
        0, "object_id", np.arange(len(frame), dtype=np.int64) + task * 1_000_000_000
    )
    frame.insert(1, "split", split)
    atomic_parquet(frame, out / "native_parent.parquet")
    write(out / "sampling.json", summary)
    finish(
        out,
        [out / "native_parent.parquet", out / "sampling.json"],
        contract,
        rows=len(frame),
        unique_proposals=summary["unique_proposals"],
    )


def photometer(config: dict, batch_size: int):
    import jax
    import jax.numpy as jnp

    from euclid_dsps.filters import load_filters
    from euclid_dsps.model import dynamic_model_args, load_context
    from euclid_dsps.parameter_vectors import model_mags_from_theta_matrix_jax
    from euclid_dsps.photometry import abmag_to_fnu_cgs_jax
    from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES

    if not jax.config.jax_enable_x64:
        raise ValueError("JAX_ENABLE_X64=true required")
    context = load_context(
        config["ssp_path"],
        load_filters(config["bands"]),
        n_sfh_bins=config["model"]["n_sfh_bins"],
        cosmos_config=config.get("cosmos_sed"),
        nebular_emission=config.get("nebular_emission", "ssp_flux"),
        model_config=config["model"],
    )
    dynamic = dynamic_model_args(context)

    @jax.jit
    def forward(theta):
        return abmag_to_fnu_cgs_jax(
            model_mags_from_theta_matrix_jax(
                context, dynamic, theta, SPLINE15D_PARAMETER_NAMES
            )
        )

    def predict(theta):
        parts = []
        for start in range(0, len(theta), batch_size):
            chunk = np.asarray(theta[start : start + batch_size], np.float64)
            n = len(chunk)
            if n < batch_size:
                chunk = np.pad(chunk, ((0, batch_size - n), (0, 0)), mode="edge")
            parts.append(np.asarray(forward(jnp.asarray(chunk)))[:n])
        return np.concatenate(parts)

    return predict


def photometry(root: Path, task_id: int) -> None:
    from euclid_dsps.prior_learning.spline15d import project_diffsky_frame_to_spline15d
    from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES

    m, cfg, contract = settings(root)
    task = m["tasks"][task_id]
    out = root / "photometry" / f"task_{task_id:03d}"
    out.mkdir(exist_ok=True)
    if complete(out, contract):
        return
    sample_out = root / "sampling" / task["split"]
    if not complete(sample_out, contract):
        raise ValueError("Sampling incomplete")
    for path, digest in m["assets"].items():
        if sha(path) != digest:
            raise ValueError(f"Decoder asset changed: {path}")
    native = pd.read_parquet(sample_out / "native_parent.parquet").iloc[
        task["start"] : task["stop"]
    ]
    config, noise = read(root / "decoder.json"), read(root / "noise.json")
    predict = photometer(config, cfg["decoder_batch_size"])
    paths = []
    for start in range(0, len(native), cfg["checkpoint_rows"]):
        block = out / f"block_{start:06d}"
        block.mkdir(exist_ok=True)
        if not complete(block, contract):
            source = native.iloc[start : start + cfg["checkpoint_rows"]].reset_index(
                drop=True
            )
            truth, _ = project_diffsky_frame_to_spline15d(
                source,
                n_sfh_bins=config["model"]["n_sfh_bins"],
                batch_size=cfg["projection_batch_size"],
            )
            if not np.array_equal(source.object_id, truth.object_id):
                raise ValueError("Projection changed identities")
            frame = source.merge(
                truth, on="object_id", validate="one_to_one", sort=False
            )
            values = frame[list(SPLINE15D_PARAMETER_NAMES)].to_numpy(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite projected truth")
            frame = observe(
                frame,
                predict(values),
                config["bands"],
                noise,
                seed=cfg["seed"] + 100_000 * (task_id + 1) + start,
                selection=cfg["selection"],
            )
            atomic_parquet(frame, block / "parent.parquet")
            finish(block, [block / "parent.parquet"], contract, rows=len(frame))
        paths.append(block / "parent.parquet")
        progress = dict(
            stage="photometry",
            done=min(start + cfg["checkpoint_rows"], len(native)),
            total=len(native),
        )
        write(out / "PROGRESS.json", progress)
        print(json.dumps(dict(task=task_id, **progress)), flush=True)
    frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    n = min(cfg["contracts"]["replay_objects_per_task"], len(frame))
    # Different batch shape replays the canonical saved theta, not a q draw.
    replay = photometer(config, 1)(
        frame[list(SPLINE15D_PARAMETER_NAMES)].to_numpy()[:n]
    )
    bands = [b["name"] for b in config["bands"]]
    saved = frame[[f"flux_true_{b}" for b in bands]].to_numpy()[:n]
    errors = frame[[f"fluxerr_{b}" for b in bands]].to_numpy()[:n]
    residual = float(np.max(np.abs(replay - saved) / errors))
    write(out / "checks.json", dict(replay_objects=n, max_replay_abs_sigma=residual))
    if (
        not np.isfinite(residual)
        or residual > cfg["contracts"]["maximum_replay_abs_sigma"]
    ):
        raise ValueError(f"Saved-theta replay failed: {residual}")
    finish(
        out,
        [*paths, out / "checks.json"],
        contract,
        rows=len(frame),
        selected=int(frame.selected_r29.sum()),
    )


def report(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from euclid_dsps.prior_learning.spline15d_schema import SPLINE15D_PARAMETER_NAMES

    m, cfg, contract = settings(root)
    out = root / "report"
    completed = [
        complete(root / "photometry" / f"task_{i:03d}", contract)
        for i in range(len(m["tasks"]))
    ]
    if not all(completed):
        write(
            out / "SUMMARY.json",
            dict(
                status="PARTIAL",
                completed=sum(completed),
                tasks=len(completed),
                ready_for_population_benchmark=False,
            ),
        )
        return
    samples, checks, source_ids, files = {}, {}, {}, []
    for split in SPLITS:
        paths = []
        for i, task in enumerate(m["tasks"]):
            if task["split"] == split:
                paths.extend(
                    sorted(
                        (root / "photometry" / f"task_{i:03d}").glob(
                            "block_*/parent.parquet"
                        )
                    )
                )
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        if len(frame) != cfg["parent_rows"][split]:
            raise ValueError(f"Unexpected parent size: {split}")
        source_ids[split] = set(frame.effective_proposal_key)
        samples[split] = frame
        checks[split] = catalogue_checks(
            frame, read(root / "decoder.json")["bands"], cfg["selection"]
        )
        checks[split]["sfh_exact_zero_counts"] = {
            n: int(frame[n].eq(0).sum()) for n in SPLINE15D_PARAMETER_NAMES[5:]
        }
        for kind, subset in (
            ("parent", frame),
            ("selected_r29", frame.loc[frame.selected_r29]),
        ):
            path = root / "dataset" / kind / f"{split}.parquet"
            atomic_parquet(subset, path)
            files.append(path)
    overlaps = {
        f"{a}_{b}": len(source_ids[a] & source_ids[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    failures = []
    if any(overlaps.values()):
        failures.append("effective_proposal_split_overlap")
    for split, check in checks.items():
        for row in check["noise"]:
            if (
                abs(row["mean"]) > cfg["contracts"]["maximum_noise_mean_abs"]
                or abs(row["std"] - 1) > cfg["contracts"]["maximum_noise_std_error"]
            ):
                failures.append(f"noise:{split}:{row['band']}")
        tolerance = (
            cfg["contracts"]["selection_sigma_tolerance"] * check["selection_count_std"]
            + 1.0
        )
        if abs(check["selection_count_difference"]) > tolerance:
            failures.append(f"selection_efficiency:{split}")
    write(
        out / "checks.json",
        dict(
            splits=checks,
            effective_proposal_overlap=overlaps,
            source_partition=m["split_audit"],
            failures=failures,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    frame = pd.concat(samples.values(), ignore_index=True)
    for ax, name in zip(axes, ("z_obs", "log10_stellar_mass", "dust_av"), strict=True):
        bins = np.linspace(
            float(frame[name].min()), float(frame[name].max()) + 1e-10, 60
        )
        ax.hist(
            frame[name],
            bins=bins,
            density=True,
            histtype="step",
            label="Unselected parent",
        )
        ax.hist(
            frame.loc[frame.selected_r29, name],
            bins=bins,
            density=True,
            histtype="step",
            label="Observed r<29",
        )
        ax.set(xlabel=name, ylabel="Density")
    axes[0].legend()
    fig.savefig(out / "parent_vs_selected.png", dpi=160)
    plt.close(fig)
    dataset = dict(
        status="CONTRACT_FAILED" if failures else "COHERENT_PARENT_DATASET_COMPLETE",
        ready_for_population_benchmark=not failures,
        ready_for_production=False,
        independent_physics_validation=False,
        native_Feniks_photometry_preserved=False,
        target="projected 15D Feniks parent within unclipped SSP-metallicity support",
        population_weight_role="unit weights after weighted proposal resampling",
        photometric_selection="observed lsst_r<29 only; no preceding S/N or magnitude cuts",
        noise_family="independent Gaussian m5",
        mask_model="all_observed",
        all_15_latent_dimensions_preserved=True,
        truth_used_for_training=False,
        decoder_sha256=sha(root / "decoder.json"),
        noise_sha256=sha(root / "noise.json"),
        artifacts={str(p.relative_to(root)): sha(p) for p in files},
        failures=failures,
        splits={
            split: {
                k: checks[split][k]
                for k in ("rows", "selected", "empirical_alpha", "analytic_alpha")
            }
            for split in SPLITS
        },
    )
    write(root / "dataset/CONTRACT.json", dataset)
    write(out / "SUMMARY.json", dataset)
    (root / "ROADMAP_STATUS.md").write_text(
        "# Coherent parent dataset\n\n"
        f"Dataset contracts: {'FAIL' if failures else 'PASS'}.\n\n"
        "This run prepares data only. No parent prior or individual posterior was trained.\n"
        "Still required: independent numerical physics checks, reference-prior/SFH/support "
        "compatibility, observed-only population fit and held-out 15D posterior calibration.\n"
    )
    if failures:
        raise RuntimeError(f"Dataset contracts failed: {failures}")
    finish(
        out,
        [
            out / "checks.json",
            out / "SUMMARY.json",
            out / "parent_vs_selected.png",
            root / "dataset/CONTRACT.json",
            *files,
        ],
        contract,
        ready_for_population_benchmark=True,
        ready_for_production=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "sample", "photometry", "report"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--decoder-config", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--task", type=int, default=0)
    args = p.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        prepare(
            args.source.resolve(),
            args.decoder_config.resolve(),
            root,
            args.config.resolve(),
        )
    elif args.mode == "sample":
        sample(root, args.task)
    elif args.mode == "photometry":
        photometry(root, args.task)
    else:
        report(root)


if __name__ == "__main__":
    main()
