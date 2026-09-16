"""Frozen AVI comparison: raw/IS joint samples and existing MIRA diagnostics."""

from __future__ import annotations

import argparse
import copy
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feniks_avi_experiments import read, sha, write


def joint_resample(theta, weights, count, rng):
    """Multinomial resampling preserves whole vectors, including duplicates."""
    theta, weights = np.asarray(theta), np.asarray(weights)
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid IS bank; never replace it with uniform weights")
    indices = rng.choice(len(weights), count, replace=True, p=weights / weights.sum())
    return theta[indices], indices


def sample_frame(theta, rows, names):
    """theta has [objects, draws, physical parameters] axes."""
    n, k, dim = theta.shape
    if len(rows) != n or len(names) != dim or not np.isfinite(theta).all():
        raise ValueError("invalid physical posterior table")
    frame = pd.DataFrame(theta.reshape(n * k, dim), columns=names)
    frame.insert(0, "sample_id", np.tile(np.arange(k), n))
    frame.insert(0, "row_index", np.repeat(rows, k))
    # Row identity is scoped to the single frozen validation catalogue.
    frame.insert(0, "object_id", np.repeat(rows, k))
    return frame


def prepare(training: Path, out: Path, particles: int = 4096) -> None:
    from euclid_dsps.amortized.avi_experiments import ARMS
    from euclid_dsps.amortized.mira import FENIKS_SPLINE15D_PARAMETERS
    from scripts.feniks_avi_experiments import check_inputs

    if out.exists():
        raise FileExistsError(out)
    if particles < 256 or particles % 8:
        raise ValueError("K must be >=256 and divisible by decoder block 8")
    m = check_inputs(training)
    rows = np.load(training / "validation.npy", allow_pickle=False)
    # Canonical columns are an explicit contract, not guessed truth aliases.
    truth = (
        pd.read_parquet(
            m["validation_catalog"], columns=list(FENIKS_SPLINE15D_PARAMETERS)
        )
        .iloc[rows]
        .copy()
    )
    if len(rows) < 2 or not np.isfinite(truth.to_numpy()).all():
        raise ValueError(
            "all fixed validation objects must have finite canonical truth"
        )
    truth.insert(0, "row_index", rows)
    truth.insert(0, "object_id", rows)
    hashes = {
        str((training / "MANIFEST.json").resolve()): sha(training / "MANIFEST.json")
    }
    for arm in ARMS:
        d = training / "arms" / arm.name
        final = read(d / "FINAL.json")
        if final["status"] != "TRAINING_COMPLETE" or final["manifest_sha256"] != sha(
            training / "MANIFEST.json"
        ):
            raise ValueError(f"incomplete or incompatible arm: {arm.name}")
        for file in ("encoder.eqx", "FINAL.json"):
            hashes[str((d / file).resolve())] = sha(d / file)
    out.mkdir(parents=True)
    truth.to_parquet(out / "inference_truth.parquet", index=False)
    write(
        out / "MANIFEST.json",
        dict(
            training=str(training.resolve()),
            arms=["source", *[a.name for a in ARMS]],
            particles=particles,
            saved_draws=256,
            replicas=2,
            seed=26091271,
            objects=len(rows),
            hashes=hashes,
            truth_sha256=sha(out / "inference_truth.parquet"),
            cohort="development validation, not untouched test",
            truth_role="MIRA only; never sampling, weighting or selection",
            proposal="learned q mixture including its gate, not training defensive r",
            selection_in_object_weights=False,
            population_prior="learned and frozen",
            scientific_promotion=False,
        ),
    )


def infer(out: Path, task: int, *, platform: str = "gpu") -> None:
    import fcntl

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from euclid_dsps.amortized.adaptive_smc_trainer import (
        prepare_adaptive_training_runtime,
    )
    from euclid_dsps.amortized.avi_experiments import (
        ARMS,
        log_prob,
        normalized_weights,
        with_encoder,
    )
    from euclid_dsps.amortized.features import make_encoder_features
    from euclid_dsps.amortized.latent import x_to_theta
    from euclid_dsps.amortized.posterior import sample_posterior
    from euclid_dsps.amortized.posterior_target import posterior_log_target
    from euclid_dsps.amortized.proposal_expressivity import (
        IndependentFlowMixture,
        sample_independent_mixture,
    )
    from euclid_dsps.amortized.train import LossBatch, load_checkpoint
    from euclid_dsps.config import load_config
    from scripts.feniks_avi_experiments import (
        check_inputs,
        initialize_candidate,
        initialize_transport,
        tree_digest,
    )

    contract = read(out / "MANIFEST.json")
    for path, digest in contract["hashes"].items():
        if sha(path) != digest:
            raise ValueError(f"changed inference input: {path}")
    training = Path(contract["training"])
    m = check_inputs(training)
    name = contract["arms"][task]
    dest = out / "arms" / name
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (dest / "FINAL.json").exists():
        return
    devices = tuple(jax.local_devices())
    if (
        len(devices) != 4
        or any(d.platform != platform for d in devices)
        or not jax.config.x64_enabled
    ):
        raise ValueError("four devices and float64 required")
    start = time.monotonic()
    write(dest / "PROGRESS.json", {"stage": "loading", "arm": name})
    config = load_config(training / "source_config.yaml")
    model = load_checkpoint(m["source"]["checkpoint"], config)
    config = copy.deepcopy(config)
    config["amortized"]["encoder"]["transport_float64"] = True
    model = eqx.tree_at(lambda a: a.encoder, model, initialize_transport(model.encoder))
    runtime_dir = dest / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    rt = prepare_adaptive_training_runtime(
        config,
        runtime_dir,
        train_indices_file=training / "train.npy",
        validation_indices_file=training / "validation.npy",
        validation_catalog_path=m["validation_catalog"],
        fixed_feature_stats_path=m["source"]["feature_stats"],
        train_population_prior=False,
    )
    if rt.likelihood_config.get("type", "gaussian") != "gaussian":
        raise ValueError("qualified Gaussian target required")
    if not rt.selection_objective_config["selection_correction"]["enabled"]:
        raise ValueError("selection correction must remain enabled")
    candidate = model.encoder
    if task:
        final = read(training / "arms" / name / "FINAL.json")
        if (
            tree_digest((model.prior, model.sed_scale, model.band_calibration))
            != final["prior_frozen_sha256"]
        ):
            raise ValueError("frozen prior/calibration mismatch")
        template = initialize_candidate(
            model, config, rt.latent_spec, ARMS[task - 1], m["seed"]
        )
        candidate = eqx.tree_deserialise_leaves(
            training / "arms" / name / "encoder.eqx", template
        )
    arrays = rt.validation_arrays
    rows = np.load(training / "validation.npy", allow_pickle=False)
    if not np.array_equal(arrays.row_index, rows):
        raise ValueError("runtime changed evaluation identity order")
    features = make_encoder_features(
        arrays.flux, arrays.flux_err, rt.feature_stats, arrays.mask
    )
    batch = LossBatch(
        jnp.asarray(arrays.flux),
        jnp.asarray(arrays.flux_err),
        jnp.asarray(arrays.mask),
        features,
        jnp.zeros((len(rows), 0), jnp.float32),
    )
    batch = jax.device_get(batch)
    k = contract["particles"]

    @partial(eqx.filter_pmap, in_axes=(None, 0, 0), devices=devices)
    def bank(c, b, key):
        x = (
            sample_independent_mixture(model, c, key, b.features, k).x
            if isinstance(c, IndependentFlowMixture)
            else sample_posterior(with_encoder(model, c), key, b.features, k).x
        )
        q = log_prob(model, c, b.features, x)

        def decode(xx):
            result = posterior_log_target(
                model,
                xx,
                b,
                rt.latent_spec,
                rt.context,
                rt.model_args,
                rt.parameter_names,
                rt.likelihood_config,
                rt.calibration_config,
            )
            residual = jnp.where(
                b.mask[None],
                (result.model_flux - b.flux[None])
                / jnp.maximum(b.flux_err[None], 1e-30),
                0.0,
            )
            squared = jnp.sum(residual**2, axis=-1) / jnp.maximum(
                jnp.sum(b.mask, axis=-1), 1
            )
            return result.logtarget, squared

        target, squared = jax.lax.map(decode, x.reshape((-1, 8) + x.shape[1:]))
        target, squared = target.reshape(q.shape), squared.reshape(q.shape)
        w, valid, ess = normalized_weights(target - q)
        metrics = jnp.stack(
            (
                ess,
                jnp.max(w, axis=0),
                jnp.sqrt(jnp.mean(squared, axis=0)),
                jnp.sqrt(jnp.sum(w * squared, axis=0)),
                jax.scipy.special.logsumexp(target - q, axis=0) - jnp.log(k),
                valid,
            ),
            axis=-1,
        )
        return x_to_theta(x, rt.latent_spec), w, metrics

    records = []
    for replica in range(contract["replicas"]):
        raw_frames, is_frames = [], []
        for offset in range(0, len(rows), 4):
            indices = np.minimum(np.arange(offset, offset + 4), len(rows) - 1)
            payload = jax.tree_util.tree_map(
                lambda a, idx=indices: jnp.asarray(a[idx]).reshape(
                    (4, 1) + a.shape[1:]
                ),
                batch,
            )
            keys = jnp.stack(
                [
                    jax.random.fold_in(
                        jax.random.PRNGKey(contract["seed"] + replica), int(i)
                    )
                    for i in indices
                ]
            )
            print(
                f"[avi-infer] {name} replica={replica} objects={offset}/{len(rows)} start",
                flush=True,
            )
            theta, weights, metrics = jax.device_get(bank(candidate, payload, keys))
            n = min(4, len(rows) - offset)
            theta, weights, metrics = theta[:n, :, 0], weights[:n, :, 0], metrics[:n, 0]
            if not metrics[:, -1].all() or not np.isfinite(metrics).all():
                raise ValueError(
                    "invalid evaluation bank; no objects silently discarded"
                )
            selected = []
            bank_frames = []
            for j in range(n):
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [contract["seed"], replica, int(indices[j]), 99]
                    )
                )
                draws, draw_ids = joint_resample(
                    theta[j], weights[j], contract["saved_draws"], rng
                )
                selected.append(draws)
                records.append(
                    dict(
                        row_index=int(rows[offset + j]),
                        replica=replica,
                        ess=float(metrics[j, 0]),
                        ess_fraction=float(metrics[j, 0] / k),
                        max_weight=float(metrics[j, 1]),
                        raw_predictive_rms=float(metrics[j, 2]),
                        is_predictive_rms=float(metrics[j, 3]),
                        log_evidence=float(metrics[j, 4]),
                        resampled_unique=int(len(np.unique(draw_ids))),
                    )
                )
                frame = sample_frame(
                    theta[j : j + 1],
                    rows[offset + j : offset + j + 1],
                    rt.latent_spec.names,
                )
                frame["weight"] = weights[j]
                bank_frames.append(frame)
            bank_dir = dest / f"bank_{replica}"
            bank_dir.mkdir(exist_ok=True)
            pd.concat(bank_frames).to_parquet(
                bank_dir / f"part_{offset:05}.parquet", index=False
            )
            raw_frames.append(
                sample_frame(
                    theta[:, : contract["saved_draws"]],
                    rows[offset : offset + n],
                    rt.latent_spec.names,
                )
            )
            is_frames.append(
                sample_frame(
                    np.stack(selected), rows[offset : offset + n], rt.latent_spec.names
                )
            )
            write(
                dest / "PROGRESS.json",
                dict(
                    stage="inference",
                    arm=name,
                    replica=replica,
                    objects_complete=offset + n,
                    objects=len(rows),
                    elapsed_seconds=time.monotonic() - start,
                ),
            )
        for label, frames in (("raw", raw_frames), ("is", is_frames)):
            pd.concat(frames).to_parquet(
                dest / f"{label}_{replica}.parquet", index=False
            )
    pd.DataFrame(records).to_csv(dest / "metrics.csv", index=False)
    artifacts = {p.name: sha(p) for p in dest.glob("*.parquet")}
    artifacts["metrics.csv"] = sha(dest / "metrics.csv")
    write(
        dest / "FINAL.json",
        dict(
            status="INFERENCE_COMPLETE",
            manifest_sha256=sha(out / "MANIFEST.json"),
            arm=name,
            objects=len(rows),
            replicas=contract["replicas"],
            particles=k,
            elapsed_seconds=time.monotonic() - start,
            scientific_promotion=False,
            artifacts=artifacts,
        ),
    )


def mira(out: Path) -> None:
    from euclid_dsps.amortized.mira import evaluate_feniks_mira

    m = read(out / "MANIFEST.json")
    if sha(out / "inference_truth.parquet") != m["truth_sha256"]:
        raise ValueError("truth diagnostic table changed")
    for name in m["arms"]:
        final = read(out / "arms" / name / "FINAL.json")
        if final["status"] != "INFERENCE_COMPLETE" or final["manifest_sha256"] != sha(
            out / "MANIFEST.json"
        ):
            raise ValueError("all frozen inference arms must complete first")
        for file, digest in final["artifacts"].items():
            if sha(out / "arms" / name / file) != digest:
                raise ValueError(f"changed inference output: {name}/{file}")
    summaries = []
    for replica in range(m["replicas"]):
        specs = [
            (f"{name}_{kind}", out / "arms" / name / f"{kind}_{replica}.parquet")
            for name in m["arms"]
            for kind in ("raw", "is")
        ]
        evaluate_feniks_mira(
            truth_path=out / "inference_truth.parquet",
            posterior_specs=specs,
            out_dir=out / f"mira_{replica}",
            samples_per_object=m["saved_draws"],
            seed=m["seed"],
            num_regions=100,
            num_bootstrap=1000,
        )
    for name in m["arms"]:
        frame = pd.read_csv(out / "arms" / name / "metrics.csv")
        values = frame.groupby("row_index").median(numeric_only=True)
        summaries.append(
            dict(
                arm=name,
                **values[
                    [
                        "ess",
                        "ess_fraction",
                        "max_weight",
                        "raw_predictive_rms",
                        "is_predictive_rms",
                    ]
                ]
                .median()
                .to_dict(),
                fraction_ess_below_5=float((values.ess < 5).mean()),
            )
        )
    pd.DataFrame(summaries).to_csv(out / "comparison.csv", index=False)
    plot_comparison(out, m)
    write(
        out / "FINAL.json",
        dict(
            status="INFERENCE_MIRA_COMPLETE",
            scientific_promotion=False,
            interpretation="development comparison; finite-K IS and resampling are not exact posterior draws",
        ),
    )


def plot_comparison(out: Path, manifest: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("ess_fraction", "max_weight", "raw_predictive_rms", "is_predictive_rms")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for name in manifest["arms"]:
        frame = (
            pd.read_csv(out / "arms" / name / "metrics.csv")
            .groupby("row_index")
            .median(numeric_only=True)
        )
        for metric, ax in zip(metrics, axes, strict=True):
            values = np.sort(frame[metric])
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=name)
            ax.set_xlabel(metric)
            ax.set_ylabel("Fraction of galaxies")
            if metric != "max_weight":
                ax.set_xscale("log")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)

    # Fixed row positions, not a selection based on truth or apparent success.
    truth = pd.read_parquet(out / "inference_truth.parquet")
    names = [c for c in truth.columns if c not in ("object_id", "row_index")]
    selected = truth.iloc[
        np.unique(np.linspace(0, len(truth) - 1, min(8, len(truth)), dtype=int))
    ]
    tables = {
        (name, kind): pd.read_parquet(out / "arms" / name / f"{kind}_0.parquet")
        for name in ("source", "B_experts", "E_experts_elbo")
        for kind in ("raw", "is")
    }
    plots = out / "marginals"
    plots.mkdir(exist_ok=True)
    for _, row in selected.iterrows():
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        for parameter, ax in zip(names, axes.flat, strict=False):
            for (name, kind), table in tables.items():
                values = table.loc[table.object_id.eq(row.object_id), parameter]
                ax.hist(
                    values,
                    bins=35,
                    density=True,
                    histtype="step",
                    label=f"{name} {kind}",
                    color={
                        "source": "black",
                        "B_experts": "tab:blue",
                        "E_experts_elbo": "tab:green",
                    }[name],
                    linestyle="--" if kind == "raw" else "-",
                )
            ax.axvline(row[parameter], color="black", linewidth=1, label="truth")
            ax.set_title(parameter, fontsize=9)
        for ax in list(axes.flat)[len(names) :]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(
            f"Catalogue row {int(row.row_index)}: replica 0, joint raw/IS samples"
        )
        fig.tight_layout()
        fig.savefig(plots / f"row_{int(row.row_index)}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("prepare", "infer", "mira"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--particles", type=int, default=4096)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.training is None:
            parser.error("prepare requires --training")
        prepare(args.training, args.root, args.particles)
    elif args.mode == "infer":
        infer(args.root, args.task)
    else:
        mira(args.root)


if __name__ == "__main__":
    main()
